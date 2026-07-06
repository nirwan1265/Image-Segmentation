#!/usr/bin/env python3
"""
Mask Completion - Simple U-Net for partial mask → full mask.

No RGB. Just mask to mask. Clean and simple.

Usage:
    # Train
    python mask_completion.py train --masks /path/to/complete_mask --output model.pth

    # Test (synthetic occlusion on complete masks)
    python mask_completion.py test --model model.pth --masks /path/to/complete_mask --output results/

    # Inference (on actual partial masks)
    python mask_completion.py infer --model model.pth --mask partial.png --output completed.png
"""
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
# U-Net Model (mask→mask, 1 channel in, 1 channel out)
# =============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetMask(nn.Module):
    """Simple U-Net: 1 channel in → 1 channel out."""

    def __init__(self, base_ch: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch*2, base_ch*4, base_ch*8

        self.enc1 = DoubleConv(1, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)

        self.pool = nn.MaxPool2d(2)

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

        d3 = self.up3(e4)
        d3 = self._match(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self._match(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self._match(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)

    def _match(self, x, target):
        if x.shape[-2:] != target.shape[-2:]:
            x = F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        return x


# =============================================================================
# Synthetic Occlusion
# =============================================================================

def occlude_mask_from_edge(mask: np.ndarray, occlusion_range: Tuple[float, float] = (0.15, 0.45)) -> np.ndarray:
    """
    Remove part of the mask from edges to simulate occlusion/cropping.
    Returns the partial/visible mask.
    """
    h, w = mask.shape
    target_occ = random.uniform(*occlusion_range)

    # Find mask bounding box
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return mask

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    mask_w = x_max - x_min
    mask_h = y_max - y_min

    partial = mask.copy()
    current_occ = 0.0
    original_area = (mask > 127).sum()

    # Randomly remove from edges until target occlusion reached
    attempts = 0
    while current_occ < target_occ and attempts < 10:
        edge = random.randint(0, 3)  # 0=left, 1=right, 2=top, 3=bottom
        cut_amount = random.uniform(0.1, 0.3)

        if edge == 0:  # left
            cut_x = x_min + int(mask_w * cut_amount)
            partial[:, :cut_x] = 0
        elif edge == 1:  # right
            cut_x = x_max - int(mask_w * cut_amount)
            partial[:, cut_x:] = 0
        elif edge == 2:  # top
            cut_y = y_min + int(mask_h * cut_amount)
            partial[:cut_y, :] = 0
        else:  # bottom
            cut_y = y_max - int(mask_h * cut_amount)
            partial[cut_y:, :] = 0

        current_area = (partial > 127).sum()
        current_occ = 1.0 - (current_area / max(original_area, 1))
        attempts += 1

    return partial


def occlude_mask_with_shape(mask: np.ndarray, occlusion_range: Tuple[float, float] = (0.15, 0.40)) -> np.ndarray:
    """
    Occlude mask with random shapes (simulating another leaf on top).
    """
    h, w = mask.shape
    target_occ = random.uniform(*occlusion_range)

    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return mask

    cx, cy = int(xs.mean()), int(ys.mean())
    mask_area = (mask > 127).sum()

    partial = mask.copy()
    current_occ = 0.0
    attempts = 0

    while current_occ < target_occ and attempts < 5:
        # Random shape near mask center
        shape_type = random.choice(['ellipse', 'rect'])
        offset_x = random.randint(-w//4, w//4)
        offset_y = random.randint(-h//4, h//4)
        scx, scy = cx + offset_x, cy + offset_y

        occ_shape = np.zeros((h, w), dtype=np.uint8)

        if shape_type == 'ellipse':
            ax = random.randint(max(5, w//8), w//3)
            ay = random.randint(max(5, h//8), h//3)
            angle = random.randint(0, 180)
            cv2.ellipse(occ_shape, (scx, scy), (ax, ay), angle, 0, 360, 255, -1)
        else:
            rw = random.randint(w//6, w//2)
            rh = random.randint(h//6, h//2)
            x1, y1 = max(0, scx - rw//2), max(0, scy - rh//2)
            x2, y2 = min(w, scx + rw//2), min(h, scy + rh//2)
            cv2.rectangle(occ_shape, (x1, y1), (x2, y2), 255, -1)

        # Remove occluded part from mask
        partial[occ_shape > 0] = 0

        current_area = (partial > 127).sum()
        current_occ = 1.0 - (current_area / max(mask_area, 1))
        attempts += 1

    return partial


def occlude_mask_with_leaf(mask: np.ndarray, other_mask: np.ndarray,
                           occlusion_range: Tuple[float, float] = (0.15, 0.30),
                           num_overlays: int = 1) -> np.ndarray:
    """
    Occlude mask by overlaying another leaf mask on top (realistic leaf-on-leaf occlusion).

    Args:
        mask: The target mask to occlude
        other_mask: Another leaf mask to use as the occluding shape
        occlusion_range: Target occlusion percentage range
        num_overlays: Number of overlay copies to place

    Returns:
        Partial mask with leaf-shaped occlusion
    """
    h, w = mask.shape
    target_occ = random.uniform(*occlusion_range)

    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return mask

    # Get mask bounding box and center
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cx, cy = int(xs.mean()), int(ys.mean())
    original_area = (mask > 127).sum()

    partial = mask.copy()
    current_occ = 0.0

    for _ in range(num_overlays):
        if current_occ >= target_occ:
            break

        # Prepare the occluding leaf - resize to reasonable size relative to target
        occ_h, occ_w = other_mask.shape

        # Scale occluding mask to be 30-70% of target mask size
        scale = random.uniform(0.3, 0.7)
        mask_w = x_max - x_min
        mask_h = y_max - y_min
        new_w = max(10, int(mask_w * scale))
        new_h = max(10, int(mask_h * scale))

        occluder = cv2.resize(other_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        occluder = (occluder > 127).astype(np.uint8) * 255

        # Random rotation
        angle = random.randint(0, 360)
        M = cv2.getRotationMatrix2D((new_w//2, new_h//2), angle, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        rot_w = int(new_h * sin + new_w * cos)
        rot_h = int(new_h * cos + new_w * sin)
        M[0, 2] += (rot_w - new_w) / 2
        M[1, 2] += (rot_h - new_h) / 2
        occluder = cv2.warpAffine(occluder, M, (rot_w, rot_h))

        # Random position - place so it overlaps with the target mask at EDGES ONLY
        # Real occlusion happens when one leaf covers the edge of another, not the middle
        place_mode = random.choice(['edge_left', 'edge_right', 'edge_top', 'edge_bottom'])

        if place_mode == 'edge_left':
            px = x_min - rot_w // 2 + random.randint(0, rot_w // 3)
            py = random.randint(y_min, y_max) - rot_h // 2
        elif place_mode == 'edge_right':
            px = x_max - rot_w // 3 + random.randint(0, rot_w // 3)
            py = random.randint(y_min, y_max) - rot_h // 2
        elif place_mode == 'edge_top':
            px = random.randint(x_min, x_max) - rot_w // 2
            py = y_min - rot_h // 2 + random.randint(0, rot_h // 3)
        else:  # edge_bottom
            px = random.randint(x_min, x_max) - rot_w // 2
            py = y_max - rot_h // 3 + random.randint(0, rot_h // 3)

        # Create full-size occlusion mask
        occ_full = np.zeros((h, w), dtype=np.uint8)

        # Calculate valid paste region
        src_x1 = max(0, -px)
        src_y1 = max(0, -py)
        src_x2 = min(rot_w, w - px)
        src_y2 = min(rot_h, h - py)

        dst_x1 = max(0, px)
        dst_y1 = max(0, py)
        dst_x2 = min(w, px + rot_w)
        dst_y2 = min(h, py + rot_h)

        if src_x2 > src_x1 and src_y2 > src_y1:
            occ_full[dst_y1:dst_y2, dst_x1:dst_x2] = occluder[src_y1:src_y2, src_x1:src_x2]

        # Remove occluded area from partial mask
        partial[occ_full > 127] = 0

        current_area = (partial > 127).sum()
        current_occ = 1.0 - (current_area / max(original_area, 1))

    return partial


# =============================================================================
# Dataset
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


def letterbox(img: np.ndarray, size: int, pad_value: int = 0) -> Tuple[np.ndarray, LetterboxInfo]:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    canvas = np.full((size, size), pad_value, dtype=np.uint8)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized

    return canvas, LetterboxInfo(scale, x0, y0, new_w, new_h, w, h)


def unletterbox(img: np.ndarray, info: LetterboxInfo, is_mask: bool = True) -> np.ndarray:
    """Reverse letterboxing - crop padding and resize back to original size.

    For masks, use INTER_LINEAR to preserve more detail when downscaling,
    then threshold if needed.
    """
    cropped = img[info.y0:info.y0+info.new_h, info.x0:info.x0+info.new_w]
    # Use LINEAR for better quality when downscaling masks
    interp = cv2.INTER_LINEAR if is_mask else cv2.INTER_NEAREST
    return cv2.resize(cropped, (info.orig_w, info.orig_h), interpolation=interp)


class MaskCompletionDataset(Dataset):
    def __init__(self, masks_dir: Path, size: int = 128, augment: bool = True,
                 occlusion_range: Tuple[float, float] = (0.15, 0.30),
                 num_overlays: int = 1, use_leaf_occlusion: bool = True):
        self.size = size
        self.augment = augment
        self.occlusion_range = occlusion_range
        self.num_overlays = num_overlays
        self.use_leaf_occlusion = use_leaf_occlusion

        exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        self.paths = [p for p in Path(masks_dir).iterdir() if p.suffix.lower() in exts]

        if not self.paths:
            raise RuntimeError(f"No masks found in {masks_dir}")

        _print(f"Found {len(self.paths)} masks")
        _print(f"Occlusion: range={occlusion_range}, overlays={num_overlays}, leaf_occlusion={use_leaf_occlusion}")

    def __len__(self):
        return len(self.paths)

    def _load_random_other_mask(self, exclude_idx: int) -> np.ndarray:
        """Load a random mask from the dataset (for leaf-on-leaf occlusion)."""
        other_idx = exclude_idx
        while other_idx == exclude_idx:
            other_idx = random.randint(0, len(self.paths) - 1)

        other = cv2.imread(str(self.paths[other_idx]), cv2.IMREAD_GRAYSCALE)
        if other is None:
            # Fallback to same mask
            other = cv2.imread(str(self.paths[exclude_idx]), cv2.IMREAD_GRAYSCALE)
        return (other > 127).astype(np.uint8) * 255

    def __getitem__(self, idx):
        # Load mask
        mask = cv2.imread(str(self.paths[idx]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not load {self.paths[idx]}")

        mask = (mask > 127).astype(np.uint8) * 255

        # Augment
        if self.augment:
            # Flip
            if random.random() > 0.5:
                mask = cv2.flip(mask, 1)
            if random.random() > 0.5:
                mask = cv2.flip(mask, 0)
            # Rotate
            k = random.randint(0, 3)
            if k > 0:
                mask = np.rot90(mask, k)
            mask = np.ascontiguousarray(mask)

            # ASPECT RATIO augmentation - stretch/squeeze to create varied shapes
            # This teaches model to handle wide/flat or tall/narrow masks
            if random.random() > 0.3:  # 70% chance
                aspect_change = random.uniform(0.5, 2.0)  # 0.5x to 2x aspect ratio
                h, w = mask.shape
                if aspect_change > 1:
                    # Make wider
                    new_w = int(w * aspect_change)
                    mask = cv2.resize(mask, (new_w, h), interpolation=cv2.INTER_NEAREST)
                else:
                    # Make taller
                    new_h = int(h / aspect_change)
                    mask = cv2.resize(mask, (w, new_h), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 127).astype(np.uint8) * 255

        # Letterbox
        mask_sq, _ = letterbox(mask, self.size, pad_value=0)

        # SCALE augmentation - randomly add padding to make mask smaller in canvas
        # This teaches model to handle masks that fill 30-80% of canvas
        if self.augment and random.random() > 0.4:  # 60% chance
            # Find mask bounds
            ys, xs = np.where(mask_sq > 127)
            if len(ys) > 0:
                y1, y2 = ys.min(), ys.max() + 1
                x1, x2 = xs.min(), xs.max() + 1
                # Crop to bounds
                cropped = mask_sq[y1:y2, x1:x2]
                # Random scale down (30-80% of canvas)
                target_fill = random.uniform(0.3, 0.8)
                current_fill = cropped.size / (self.size * self.size)
                if current_fill > target_fill:
                    scale = np.sqrt(target_fill / current_fill)
                    new_h = max(8, int(cropped.shape[0] * scale))
                    new_w = max(8, int(cropped.shape[1] * scale))
                    small = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    small = (small > 127).astype(np.uint8) * 255
                    # Place randomly in canvas
                    mask_sq = np.zeros((self.size, self.size), dtype=np.uint8)
                    max_y = self.size - new_h
                    max_x = self.size - new_w
                    py = random.randint(0, max(0, max_y))
                    px = random.randint(0, max(0, max_x))
                    mask_sq[py:py+new_h, px:px+new_w] = small

        # Create synthetic occlusion - MIX of methods for robustness
        # 50% leaf occlusion, 30% edge cuts, 20% shape occlusion
        occlusion_type = random.random()

        if self.use_leaf_occlusion and occlusion_type < 0.5:
            # Leaf-on-leaf occlusion (realistic, curved edges)
            other_mask = self._load_random_other_mask(idx)
            other_sq, _ = letterbox(other_mask, self.size, pad_value=0)
            partial = occlude_mask_with_leaf(
                mask_sq, other_sq,
                occlusion_range=self.occlusion_range,
                num_overlays=self.num_overlays
            )
        elif occlusion_type < 0.8:
            # Edge cuts (straight lines cutting from edges)
            partial = occlude_mask_from_edge(mask_sq, self.occlusion_range)
        else:
            # Shape occlusion (ellipse/rectangle)
            partial = occlude_mask_with_shape(mask_sq, self.occlusion_range)

        # To tensors
        x = torch.from_numpy(partial[None, ...].astype(np.float32) / 255.0)
        y = torch.from_numpy(mask_sq[None, ...].astype(np.float32) / 255.0)

        return x, y


# =============================================================================
# Loss
# =============================================================================

def dice_loss(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    return (1 - (2 * intersection + smooth) / (union + smooth)).mean()


def combined_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    return bce + dice_loss(logits, targets)


def compute_iou(logits, targets):
    preds = (torch.sigmoid(logits) >= 0.5).float()
    intersection = (preds * targets).sum()
    union = ((preds + targets) > 0).float().sum()
    return (intersection / max(union, 1)).item()


# =============================================================================
# Train
# =============================================================================

def train(args):
    set_seed(args.seed)

    device = args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    _print(f"Device: {device}")

    occlusion_range = (args.occlusion_min, args.occlusion_max)
    use_leaf_occlusion = not args.no_leaf_occlusion

    dataset = MaskCompletionDataset(
        Path(args.masks),
        size=args.size,
        augment=True,
        occlusion_range=occlusion_range,
        num_overlays=args.num_overlays,
        use_leaf_occlusion=use_leaf_occlusion
    )
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)

    model = UNetMask(base_ch=args.base_ch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    _print(f"Training: {len(dataset)} masks, {args.steps} steps, size={args.size}")

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

        ema_loss = loss.item() if ema_loss is None else 0.9 * ema_loss + 0.1 * loss.item()

        if step % 100 == 0 or step == 1:
            iou = compute_iou(logits.detach(), y.detach())
            _print(f"[{step:5d}/{args.steps}] loss={ema_loss:.4f} iou={iou:.4f}")

    torch.save({'state_dict': model.state_dict(), 'size': args.size}, args.output)
    _print(f"Saved: {args.output}")


# =============================================================================
# Test (with synthetic occlusion + visualization)
# =============================================================================

def test(args):
    device = args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'

    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    size = ckpt.get('size', 128)

    model = UNetMask(base_ch=32).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    masks_dir = Path(args.masks)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {'.png', '.jpg', '.jpeg'}
    mask_paths = list(masks_dir.glob('*'))
    mask_paths = [p for p in mask_paths if p.suffix.lower() in exts][:args.num]

    _print(f"Testing on {len(mask_paths)} masks...")

    all_viz = []
    ious = []

    for mask_path in mask_paths:
        _print(f"  {mask_path.name}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8) * 255
        h, w = mask.shape

        # Letterbox
        mask_sq, info = letterbox(mask, size, pad_value=0)

        # Synthetic occlusion
        if random.random() > 0.5:
            partial_sq = occlude_mask_from_edge(mask_sq, (0.2, 0.4))
        else:
            partial_sq = occlude_mask_with_shape(mask_sq, (0.2, 0.4))

        # Predict
        with torch.no_grad():
            x = torch.from_numpy(partial_sq[None, None, ...].astype(np.float32) / 255.0).to(device)
            logits = model(x)
            pred_sq = (torch.sigmoid(logits[0, 0]) >= 0.5).cpu().numpy().astype(np.uint8) * 255

        # IoU
        gt_bin = (mask_sq > 127).astype(np.float32)
        pred_bin = (pred_sq > 127).astype(np.float32)
        intersection = (gt_bin * pred_bin).sum()
        union = ((gt_bin + pred_bin) > 0).sum()
        iou = intersection / max(union, 1)
        ious.append(iou)

        # Visualization - scale up for visibility
        vis_scale = max(3, 200 // size)
        sz = size * vis_scale

        gt_big = cv2.resize(mask_sq, (sz, sz), interpolation=cv2.INTER_NEAREST)
        partial_big = cv2.resize(partial_sq, (sz, sz), interpolation=cv2.INTER_NEAREST)
        pred_big = cv2.resize(pred_sq, (sz, sz), interpolation=cv2.INTER_NEAREST)

        # Create panels
        def mask_to_viz(m, color):
            viz = np.full((sz, sz, 3), 255, dtype=np.uint8)
            viz[m > 127] = color
            cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(viz, cnt, -1, (0, 0, 0), 2)
            return viz

        panel_gt = mask_to_viz(gt_big, (150, 200, 150))  # Light green
        panel_partial = mask_to_viz(partial_big, (200, 200, 150))  # Light yellow
        panel_pred = mask_to_viz(pred_big, (200, 150, 200))  # Light magenta

        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel_gt, "Ground Truth", (5, 20), font, 0.5, (0, 100, 0), 1)
        cv2.putText(panel_partial, "Partial (input)", (5, 20), font, 0.5, (100, 100, 0), 1)
        cv2.putText(panel_pred, f"Predicted (IoU:{iou:.2f})", (5, 20), font, 0.5, (100, 0, 100), 1)

        # Combine
        row = np.concatenate([panel_gt, panel_partial, panel_pred], axis=1)
        all_viz.append(row)

    # Grid
    if all_viz:
        grid = np.concatenate(all_viz, axis=0)
        grid_path = out_dir / "test_grid.png"
        cv2.imwrite(str(grid_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        _print(f"\nMean IoU: {np.mean(ious):.4f}")
        _print(f"Saved: {grid_path}")


# =============================================================================
# Load Model (for external use)
# =============================================================================

def load_model(path: str, device: str = 'mps') -> Tuple[nn.Module, dict]:
    """Load a trained shape completion model.

    Args:
        path: Path to .pth checkpoint
        device: 'mps', 'cuda', or 'cpu'

    Returns:
        (model, meta) where meta contains training info like 'size'
    """
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'

    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    size = ckpt.get('size', 128)

    model = UNetMask(base_ch=32).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    meta = {
        'size': size,
        'step': ckpt.get('step', 0),
    }
    return model, meta


def predict_mask(model: nn.Module, partial_mask: np.ndarray, size: int = 128, threshold: float = 0.5) -> np.ndarray:
    """Predict completed mask from partial mask.

    Args:
        model: Loaded UNetMask model
        partial_mask: Binary mask (H, W) uint8 or bool
        size: Model input size (should match training)
        threshold: Probability threshold for binary output

    Returns:
        Completed binary mask (H, W) uint8 (0 or 255)
    """
    device = next(model.parameters()).device

    # Convert to proper format
    if partial_mask.dtype == bool:
        partial_mask = partial_mask.astype(np.uint8) * 255
    elif partial_mask.max() <= 1:
        partial_mask = (partial_mask * 255).astype(np.uint8)

    mask_sq, info = letterbox(partial_mask, size, pad_value=0)

    with torch.no_grad():
        x = torch.from_numpy(mask_sq[None, None, ...].astype(np.float32) / 255.0).to(device)
        logits = model(x)
        # Get soft probabilities (0-255 range for unletterboxing)
        probs_sq = (torch.sigmoid(logits[0, 0]).cpu().numpy() * 255).astype(np.uint8)

    # Unletterbox the SOFT probabilities (better interpolation than binary)
    probs = unletterbox(probs_sq, info, is_mask=True)

    # THEN threshold to create binary mask
    pred = (probs >= (threshold * 255)).astype(np.uint8) * 255
    return pred


# =============================================================================
# Inference (CLI)
# =============================================================================

def infer(args):
    device = args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'

    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    size = ckpt.get('size', 128)

    model = UNetMask(base_ch=32).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(np.uint8) * 255

    mask_sq, info = letterbox(mask, size, pad_value=0)

    with torch.no_grad():
        x = torch.from_numpy(mask_sq[None, None, ...].astype(np.float32) / 255.0).to(device)
        logits = model(x)
        pred_sq = (torch.sigmoid(logits[0, 0]) >= 0.5).cpu().numpy().astype(np.uint8) * 255

    pred = unletterbox(pred_sq, info)
    cv2.imwrite(args.output, pred)
    _print(f"Saved: {args.output}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Mask Completion U-Net")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Train
    train_p = subparsers.add_parser('train')
    train_p.add_argument('--masks', required=True, help='Complete masks folder')
    train_p.add_argument('--output', required=True, help='Output model path')
    train_p.add_argument('--steps', type=int, default=3000)
    train_p.add_argument('--batch', type=int, default=8)
    train_p.add_argument('--lr', type=float, default=1e-3)
    train_p.add_argument('--size', type=int, default=128)
    train_p.add_argument('--base-ch', type=int, default=32)
    train_p.add_argument('--device', default='mps')
    train_p.add_argument('--seed', type=int, default=42)
    # Occlusion settings
    train_p.add_argument('--occlusion-min', type=float, default=0.15, help='Min occlusion ratio (default: 0.15)')
    train_p.add_argument('--occlusion-max', type=float, default=0.30, help='Max occlusion ratio (default: 0.30)')
    train_p.add_argument('--num-overlays', type=int, default=1, help='Number of leaf overlays for occlusion (default: 1)')
    train_p.add_argument('--no-leaf-occlusion', action='store_true', help='Use old edge/shape occlusion instead of leaf-on-leaf')

    # Test
    test_p = subparsers.add_parser('test')
    test_p.add_argument('--model', required=True)
    test_p.add_argument('--masks', required=True)
    test_p.add_argument('--output', required=True)
    test_p.add_argument('--num', type=int, default=6)
    test_p.add_argument('--device', default='mps')

    # Infer
    infer_p = subparsers.add_parser('infer')
    infer_p.add_argument('--model', required=True)
    infer_p.add_argument('--mask', required=True, help='Partial mask to complete')
    infer_p.add_argument('--output', required=True)
    infer_p.add_argument('--device', default='mps')

    args = parser.parse_args()

    if args.command == 'train':
        train(args)
    elif args.command == 'test':
        test(args)
    elif args.command == 'infer':
        infer(args)


if __name__ == '__main__':
    main()
