#!/usr/bin/env python3
"""
Leaf Completion v3 - Handles BOTH occlusion AND edge cropping.

The key insight:
- v2 trained on "occlusion" (leaf covered by something)
- But real incomplete leaves are often "cropped" (cut at image boundary)

This version adds edge-crop augmentation to handle both cases.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def _print(msg: str):
    print(msg, flush=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =============================================================================
# Model (same as v2)
# =============================================================================

def _gn_groups(ch: int, max_groups: int = 8) -> int:
    g = min(max_groups, ch)
    while g > 1 and (ch % g) != 0:
        g -= 1
    return max(1, g)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetCompletion(nn.Module):
    def __init__(self, in_ch: int = 4, base_ch: int = 32):
        super().__init__()
        c1, c2, c3, c4, c5 = base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16

        self.enc1 = DoubleConv(in_ch, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.bottleneck = DoubleConv(c4, c5)

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.dec4 = DoubleConv(c4 + c4, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = DoubleConv(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1)

        self.out = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self._match_size(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self._match_size(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self._match_size(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self._match_size(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)

    def _match_size(self, x, target):
        if x.shape[-2:] != target.shape[-2:]:
            x = F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        return x


# =============================================================================
# Preprocessing
# =============================================================================

@dataclass
class LetterboxInfo:
    scale: float
    x0: int
    y0: int
    new_w: int
    new_h: int
    orig_w: int
    orig_h: int


def letterbox(img: np.ndarray, size: int, pad_value: int = 255) -> Tuple[np.ndarray, LetterboxInfo]:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    if img.ndim == 3:
        canvas = np.full((size, size, img.shape[2]), pad_value, dtype=np.uint8)
    else:
        canvas = np.full((size, size), pad_value, dtype=np.uint8)

    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized

    return canvas, LetterboxInfo(scale, x0, y0, new_w, new_h, w, h)


def unletterbox(img: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    cropped = img[info.y0:info.y0+info.new_h, info.x0:info.x0+info.new_w]
    return cv2.resize(cropped, (info.orig_w, info.orig_h), interpolation=cv2.INTER_NEAREST)


# =============================================================================
# EDGE CROP AUGMENTATION (NEW!)
# =============================================================================

def crop_from_edge(
    rgb: np.ndarray,
    mask: np.ndarray,
    crop_percent: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate edge cropping - like the leaf extends beyond frame.

    Returns:
        cropped_rgb: Image with part removed (filled with white)
        visible_mask: Mask of what's still visible
        full_mask: Original full mask (ground truth)
    """
    h, w = mask.shape[:2]

    # Find mask bounding box
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return rgb, mask, mask

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    mask_w = x_max - x_min
    mask_h = y_max - y_min

    # Choose which edge(s) to crop from
    # 0=left, 1=right, 2=top, 3=bottom, 4=corner
    edge = random.randint(0, 4)
    crop_amount = random.uniform(0.15, crop_percent)

    cropped_rgb = rgb.copy()
    visible_mask = mask.copy()

    if edge == 0:  # Crop from left
        cut_x = x_min + int(mask_w * crop_amount)
        cropped_rgb[:, :cut_x] = 255  # White background
        visible_mask[:, :cut_x] = 0
    elif edge == 1:  # Crop from right
        cut_x = x_max - int(mask_w * crop_amount)
        cropped_rgb[:, cut_x:] = 255
        visible_mask[:, cut_x:] = 0
    elif edge == 2:  # Crop from top
        cut_y = y_min + int(mask_h * crop_amount)
        cropped_rgb[:cut_y, :] = 255
        visible_mask[:cut_y, :] = 0
    elif edge == 3:  # Crop from bottom
        cut_y = y_max - int(mask_h * crop_amount)
        cropped_rgb[cut_y:, :] = 255
        visible_mask[cut_y:, :] = 0
    else:  # Corner crop
        corner = random.randint(0, 3)
        cut_x = int(mask_w * crop_amount * 0.7)
        cut_y = int(mask_h * crop_amount * 0.7)
        if corner == 0:  # top-left
            cropped_rgb[:y_min+cut_y, :x_min+cut_x] = 255
            visible_mask[:y_min+cut_y, :x_min+cut_x] = 0
        elif corner == 1:  # top-right
            cropped_rgb[:y_min+cut_y, x_max-cut_x:] = 255
            visible_mask[:y_min+cut_y, x_max-cut_x:] = 0
        elif corner == 2:  # bottom-left
            cropped_rgb[y_max-cut_y:, :x_min+cut_x] = 255
            visible_mask[y_max-cut_y:, :x_min+cut_x] = 0
        else:  # bottom-right
            cropped_rgb[y_max-cut_y:, x_max-cut_x:] = 255
            visible_mask[y_max-cut_y:, x_max-cut_x:] = 0

    return cropped_rgb, visible_mask, mask


def occlude_with_shape(
    rgb: np.ndarray,
    mask: np.ndarray,
    occlusion_percent: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Add random shape occlusion on top of leaf.
    """
    h, w = mask.shape[:2]

    # Find mask center
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return rgb, mask, mask

    cx = int(xs.mean())
    cy = int(ys.mean())

    # Create occluder shape
    occ_mask = np.zeros((h, w), dtype=np.uint8)

    # Random shape near leaf
    offset_x = random.randint(-w//4, w//4)
    offset_y = random.randint(-h//4, h//4)
    shape_cx = cx + offset_x
    shape_cy = cy + offset_y

    shape_type = random.choice(['ellipse', 'rect', 'poly'])
    min_size = max(15, min(h, w) // 6)
    max_size = max(min_size + 10, min(h, w) // 2)

    if shape_type == 'ellipse':
        ax = random.randint(min_size, max_size)
        ay = random.randint(min_size, max_size)
        angle = random.randint(0, 180)
        cv2.ellipse(occ_mask, (shape_cx, shape_cy), (ax, ay), angle, 0, 360, 255, -1)
    elif shape_type == 'rect':
        rw = random.randint(min_size, max_size)
        rh = random.randint(min_size, max_size)
        x1, y1 = max(0, shape_cx - rw//2), max(0, shape_cy - rh//2)
        x2, y2 = min(w, shape_cx + rw//2), min(h, shape_cy + rh//2)
        cv2.rectangle(occ_mask, (x1, y1), (x2, y2), 255, -1)
    else:
        n_pts = random.randint(4, 8)
        angles = np.sort(np.random.uniform(0, 2*np.pi, n_pts))
        radii = np.random.uniform(min_size, max_size, n_pts)
        pts = np.stack([
            (shape_cx + radii * np.cos(angles)).clip(0, w-1),
            (shape_cy + radii * np.sin(angles)).clip(0, h-1)
        ], axis=1).astype(np.int32)
        cv2.fillPoly(occ_mask, [pts], 255)

    # Draw occluder on image
    occluded_rgb = rgb.copy()
    occ_color = [random.randint(100, 200) for _ in range(3)]
    occluded_rgb[occ_mask > 0] = occ_color

    # Visible mask = original mask minus occluder
    visible_mask = mask.copy()
    visible_mask[occ_mask > 0] = 0

    return occluded_rgb, visible_mask, mask


# =============================================================================
# Dataset with BOTH occlusion and edge crop
# =============================================================================

def auto_generate_mask(rgb: np.ndarray) -> np.ndarray:
    """Auto-generate mask for leaf on white background using color."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lower = np.array([25, 30, 30])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    lower_y = np.array([15, 30, 30])
    upper_y = np.array([35, 255, 255])
    mask_y = cv2.inRange(hsv, lower_y, upper_y)
    mask = cv2.bitwise_or(mask, mask_y)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest], -1, 255, -1)

    return mask


def augment_pair(rgb: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply augmentations."""
    if random.random() > 0.5:
        rgb = cv2.flip(rgb, 1)
        mask = cv2.flip(mask, 1)
    if random.random() > 0.5:
        rgb = cv2.flip(rgb, 0)
        mask = cv2.flip(mask, 0)
    k = random.randint(0, 3)
    if k > 0:
        rgb = np.rot90(rgb, k)
        mask = np.rot90(mask, k)
    if random.random() > 0.5:
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-20, 20)
        rgb = np.clip(alpha * rgb.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(rgb), np.ascontiguousarray(mask)


class LeafCompletionDatasetV3(Dataset):
    """
    Dataset that applies BOTH edge cropping AND occlusion.
    """

    def __init__(
        self,
        images_dir: Path,
        size: int = 256,
        crop_prob: float = 0.5,      # Probability of edge crop
        occlude_prob: float = 0.5,   # Probability of occlusion
        crop_percent: float = 0.35,
        occlude_percent: float = 0.35,
        augment: bool = True,
    ):
        self.size = size
        self.crop_prob = crop_prob
        self.occlude_prob = occlude_prob
        self.crop_percent = crop_percent
        self.occlude_percent = occlude_percent
        self.augment = augment

        exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        self.paths = [p for p in Path(images_dir).iterdir() if p.suffix.lower() in exts]

        if not self.paths:
            raise RuntimeError(f"No images found in {images_dir}")

        _print(f"Found {len(self.paths)} images")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Load image
        rgb = cv2.imread(str(self.paths[idx]), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Auto-generate mask
        full_mask = auto_generate_mask(rgb)

        # Augment
        if self.augment:
            rgb, full_mask = augment_pair(rgb, full_mask)

        # Letterbox
        rgb_sq, _ = letterbox(rgb, self.size, pad_value=255)
        mask_sq, _ = letterbox(full_mask, self.size, pad_value=0)

        # Apply degradation (crop and/or occlude)
        visible_rgb = rgb_sq.copy()
        visible_mask = mask_sq.copy()

        # Edge crop with some probability
        if random.random() < self.crop_prob:
            visible_rgb, visible_mask, _ = crop_from_edge(
                visible_rgb, visible_mask, self.crop_percent
            )

        # Occlusion with some probability
        if random.random() < self.occlude_prob:
            visible_rgb, visible_mask, _ = occlude_with_shape(
                visible_rgb, visible_mask, self.occlude_percent
            )

        # Ensure we have SOME degradation
        if np.array_equal(visible_mask, mask_sq):
            # Force at least a small crop
            visible_rgb, visible_mask, _ = crop_from_edge(
                visible_rgb, visible_mask, 0.15
            )

        # Prepare tensors
        rgb_t = torch.from_numpy(visible_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        vis_t = torch.from_numpy(visible_mask[None, ...].astype(np.float32) / 255.0)
        x = torch.cat([rgb_t, vis_t], dim=0)  # (4, H, W)

        y = torch.from_numpy(mask_sq[None, ...].astype(np.float32) / 255.0)  # (1, H, W)

        return x, y


# =============================================================================
# Loss & Training
# =============================================================================

def dice_loss(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    return (1 - (2 * intersection + smooth) / (union + smooth)).mean()


def combined_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    return bce + dice_loss(logits, targets)


def compute_iou(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) >= threshold).float()
    intersection = (preds * targets).sum()
    union = ((preds + targets) > 0).float().sum()
    return (intersection / max(union, 1)).item()


def train(args):
    set_seed(args.seed)

    device = args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    _print(f"Using device: {device}")

    dataset = LeafCompletionDatasetV3(
        images_dir=Path(args.images),
        size=args.size,
        crop_prob=args.crop_prob,
        occlude_prob=args.occlude_prob,
        crop_percent=args.crop_percent,
        occlude_percent=args.occlude_percent,
        augment=True,
    )

    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)

    model = UNetCompletion(in_ch=4, base_ch=args.base_channels).to(device)

    if args.resume and Path(args.resume).exists():
        _print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['state_dict'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    _print(f"Training config:")
    _print(f"  Images: {len(dataset)}")
    _print(f"  Steps: {args.steps}")
    _print(f"  Crop prob: {args.crop_prob}, Occlude prob: {args.occlude_prob}")

    data_iter = iter(dataloader)
    ema_loss = None

    for step in range(1, args.steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = combined_loss(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        ema_loss = loss_val if ema_loss is None else 0.9 * ema_loss + 0.1 * loss_val

        if step % 100 == 0 or step == 1:
            iou = compute_iou(logits.detach(), y.detach())
            _print(f"[{step:5d}/{args.steps}] loss={ema_loss:.4f} iou={iou:.4f}")

        if step % 1000 == 0:
            torch.save({
                'state_dict': model.state_dict(),
                'step': step,
                'size': args.size,
            }, str(Path(args.output).with_suffix(f'.step{step}.pth')))

    torch.save({
        'state_dict': model.state_dict(),
        'step': args.steps,
        'size': args.size,
    }, args.output)
    _print(f"Done! Saved to {args.output}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Leaf Completion v3 - Edge crop + Occlusion")
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_p = subparsers.add_parser('train')
    train_p.add_argument('--images', required=True, help='Complete leaf images folder')
    train_p.add_argument('--output', required=True)
    train_p.add_argument('--steps', type=int, default=5000)
    train_p.add_argument('--batch', type=int, default=4)
    train_p.add_argument('--lr', type=float, default=1e-4)
    train_p.add_argument('--size', type=int, default=256)
    train_p.add_argument('--base-channels', type=int, default=32)
    train_p.add_argument('--device', default='mps')
    train_p.add_argument('--seed', type=int, default=42)
    train_p.add_argument('--resume', default=None)
    train_p.add_argument('--crop-prob', type=float, default=0.6)
    train_p.add_argument('--occlude-prob', type=float, default=0.4)
    train_p.add_argument('--crop-percent', type=float, default=0.35)
    train_p.add_argument('--occlude-percent', type=float, default=0.35)

    args = parser.parse_args()

    if args.command == 'train':
        train(args)


if __name__ == '__main__':
    main()
