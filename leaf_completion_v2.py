#!/usr/bin/env python3
"""
Leaf Completion v2 - Fixed and improved amodal leaf segmentation.

Fixes from v1:
1. Fixed --use-geom bug (geometric shapes now actually drawn on image)
2. Added proper data augmentation (flips, rotations, color jitter)
3. More aggressive occlusion settings
4. Better train/inference alignment
5. Shape-only mode option (mask→mask, no RGB domain issues)

Usage:
    # Train
    python leaf_completion_v2.py train --images ./images --masks ./masks --output model.pth

    # Inference
    python leaf_completion_v2.py infer --model model.pth --image leaf.png --visible-mask partial.png --output complete.png

    # Test with synthetic occlusion (to validate model)
    python leaf_completion_v2.py test --model model.pth --images ./images --masks ./masks --output ./test_results
"""
from __future__ import annotations

import argparse
import random
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Utilities
# =============================================================================

def _print(msg: str):
    print(msg, flush=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =============================================================================
# Model Architecture (same U-Net but cleaner)
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
    """
    U-Net for mask completion.

    Modes:
    - in_ch=4: RGB (3) + visible mask (1) → full mask (1)
    - in_ch=1: visible mask only → full mask (shape-only mode)
    """

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
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
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
    """Letterbox image to square."""
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

    info = LetterboxInfo(scale, x0, y0, new_w, new_h, w, h)
    return canvas, info


def unletterbox(img: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Reverse letterbox to original size."""
    cropped = img[info.y0:info.y0+info.new_h, info.x0:info.x0+info.new_w]
    return cv2.resize(cropped, (info.orig_w, info.orig_h), interpolation=cv2.INTER_NEAREST)


# =============================================================================
# Occlusion Generation (FIXED)
# =============================================================================

def create_random_shape(h: int, w: int, center: Tuple[int, int] = None) -> np.ndarray:
    """Create a random geometric shape mask."""
    mask = np.zeros((h, w), dtype=np.uint8)

    if center is None:
        cx, cy = random.randint(0, w-1), random.randint(0, h-1)
    else:
        cx, cy = center

    shape_type = random.choice(['ellipse', 'rect', 'poly'])
    min_size = max(10, min(h, w) // 8)
    max_size = max(min_size + 10, min(h, w) // 3)

    if shape_type == 'ellipse':
        ax = random.randint(min_size, max_size)
        ay = random.randint(min_size, max_size)
        angle = random.randint(0, 180)
        cv2.ellipse(mask, (cx, cy), (ax, ay), angle, 0, 360, 255, -1)

    elif shape_type == 'rect':
        rw = random.randint(min_size, max_size)
        rh = random.randint(min_size, max_size)
        x1, y1 = max(0, cx - rw//2), max(0, cy - rh//2)
        x2, y2 = min(w, cx + rw//2), min(h, cy + rh//2)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    else:  # poly
        n_pts = random.randint(4, 8)
        angles = np.sort(np.random.uniform(0, 2*np.pi, n_pts))
        radii = np.random.uniform(min_size, max_size, n_pts)
        pts = np.stack([
            (cx + radii * np.cos(angles)).clip(0, w-1),
            (cy + radii * np.sin(angles)).clip(0, h-1)
        ], axis=1).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)

    return mask


def paste_occluder(
    base_rgb: np.ndarray,
    occluder_rgba: np.ndarray,
    cx: int, cy: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Paste RGBA occluder onto RGB image with alpha blending.
    Returns (modified_image, occlusion_mask).
    """
    h_base, w_base = base_rgb.shape[:2]
    h_occ, w_occ = occluder_rgba.shape[:2]

    # Calculate paste region
    x1 = cx - w_occ // 2
    y1 = cy - h_occ // 2
    x2 = x1 + w_occ
    y2 = y1 + h_occ

    # Clip to image bounds
    src_x1 = max(0, -x1)
    src_y1 = max(0, -y1)
    src_x2 = w_occ - max(0, x2 - w_base)
    src_y2 = h_occ - max(0, y2 - h_base)

    dst_x1 = max(0, x1)
    dst_y1 = max(0, y1)
    dst_x2 = min(w_base, x2)
    dst_y2 = min(h_base, y2)

    # Check for valid overlap
    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return base_rgb, np.zeros((h_base, w_base), dtype=np.uint8)

    # Extract regions
    src_region = occluder_rgba[src_y1:src_y2, src_x1:src_x2]
    dst_region = base_rgb[dst_y1:dst_y2, dst_x1:dst_x2]

    # Alpha blend
    alpha = src_region[:, :, 3:4].astype(np.float32) / 255.0
    rgb = src_region[:, :, :3].astype(np.float32)
    blended = (alpha * rgb + (1 - alpha) * dst_region.astype(np.float32)).astype(np.uint8)

    # Create output
    result = base_rgb.copy()
    result[dst_y1:dst_y2, dst_x1:dst_x2] = blended

    # Create occlusion mask
    occ_mask = np.zeros((h_base, w_base), dtype=np.uint8)
    occ_mask[dst_y1:dst_y2, dst_x1:dst_x2] = (src_region[:, :, 3] > 0).astype(np.uint8) * 255

    return result, occ_mask


def generate_occlusion(
    rgb: np.ndarray,
    full_mask: np.ndarray,
    occluder_bank: List[np.ndarray],
    occ_fraction_range: Tuple[float, float] = (0.1, 0.5),
    n_occluders: Tuple[int, int] = (1, 3),
    use_geom: bool = True,
    edge_bias: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic occlusion on an image.

    Returns:
        (occluded_rgb, visible_mask)

    FIXED: Geometric shapes now actually modify the image!
    """
    h, w = full_mask.shape[:2]
    target_frac = random.uniform(*occ_fraction_range)

    # Find edge points for biased placement
    edge_points = []
    if edge_bias > 0:
        dist = cv2.distanceTransform((full_mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
        edge_region = (dist > 0) & (dist <= 10)
        ys, xs = np.where(edge_region)
        if len(xs) > 0:
            edge_points = list(zip(xs.tolist(), ys.tolist()))

    def pick_center():
        if edge_points and random.random() < edge_bias:
            return random.choice(edge_points)
        return (random.randint(0, w-1), random.randint(0, h-1))

    occluded_rgb = rgb.copy()
    total_occ_mask = np.zeros((h, w), dtype=np.uint8)

    n = random.randint(*n_occluders)

    for _ in range(n):
        cx, cy = pick_center()

        # Choose occluder type
        if occluder_bank and (not use_geom or random.random() > 0.3):
            # Use leaf occluder from bank
            occluder = random.choice(occluder_bank)
            scale = random.uniform(0.5, 1.5)
            new_size = (int(occluder.shape[1] * scale), int(occluder.shape[0] * scale))
            if new_size[0] > 5 and new_size[1] > 5:
                occluder_scaled = cv2.resize(occluder, new_size, interpolation=cv2.INTER_LINEAR)
                occluded_rgb, occ_mask = paste_occluder(occluded_rgb, occluder_scaled, cx, cy)
                total_occ_mask = np.maximum(total_occ_mask, occ_mask)
        else:
            # Use geometric shape - FIXED: now actually draws on image!
            shape_mask = create_random_shape(h, w, center=(cx, cy))

            # Draw the shape on the image with a random color/texture
            if random.random() < 0.5:
                # Solid color
                color = [random.randint(50, 200) for _ in range(3)]
                occluded_rgb[shape_mask > 0] = color
            else:
                # Blur/smudge effect
                blurred = cv2.GaussianBlur(occluded_rgb, (31, 31), 0)
                occluded_rgb[shape_mask > 0] = blurred[shape_mask > 0]

            total_occ_mask = np.maximum(total_occ_mask, shape_mask)

    # Calculate current occlusion fraction
    leaf_pixels = (full_mask > 0).sum()
    if leaf_pixels > 0:
        current_frac = ((total_occ_mask > 0) & (full_mask > 0)).sum() / leaf_pixels

        # Add more occlusion if needed
        attempts = 0
        while current_frac < target_frac and attempts < 5:
            cx, cy = pick_center()
            shape_mask = create_random_shape(h, w, center=(cx, cy))

            # FIXED: Draw shape on image
            color = [random.randint(50, 200) for _ in range(3)]
            occluded_rgb[shape_mask > 0] = color
            total_occ_mask = np.maximum(total_occ_mask, shape_mask)

            current_frac = ((total_occ_mask > 0) & (full_mask > 0)).sum() / leaf_pixels
            attempts += 1

    # Compute visible mask
    visible_mask = ((full_mask > 0) & (total_occ_mask == 0)).astype(np.uint8) * 255

    return occluded_rgb, visible_mask


# =============================================================================
# Data Augmentation (NEW)
# =============================================================================

def augment_pair(rgb: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply augmentations to RGB image and mask together."""
    # Random horizontal flip
    if random.random() > 0.5:
        rgb = cv2.flip(rgb, 1)
        mask = cv2.flip(mask, 1)

    # Random vertical flip
    if random.random() > 0.5:
        rgb = cv2.flip(rgb, 0)
        mask = cv2.flip(mask, 0)

    # Random 90-degree rotation
    k = random.randint(0, 3)
    if k > 0:
        rgb = np.rot90(rgb, k)
        mask = np.rot90(mask, k)

    # Random brightness/contrast for RGB only
    if random.random() > 0.5:
        alpha = random.uniform(0.8, 1.2)  # contrast
        beta = random.uniform(-20, 20)    # brightness
        rgb = np.clip(alpha * rgb.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # Ensure contiguous
    rgb = np.ascontiguousarray(rgb)
    mask = np.ascontiguousarray(mask)

    return rgb, mask


# =============================================================================
# Dataset
# =============================================================================

def find_image_mask_pairs(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    """Find matching image-mask pairs."""
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

    # Build mask index
    mask_index = {}
    for p in masks_dir.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            # Try various stem variations
            stem = p.stem
            for suffix in ['_mask', '_m', '_seg']:
                if stem.endswith(suffix):
                    stem = stem[:-len(suffix)]
                    break
            mask_index.setdefault(stem, []).append(p)

    # Find pairs
    pairs = []
    for p in images_dir.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        stem = p.stem
        # Try to find matching mask
        if stem in mask_index:
            pairs.append((p, mask_index[stem]))
        else:
            # Try without _crop suffix
            for suffix in ['_crop', '_cropped']:
                if stem.endswith(suffix):
                    alt_stem = stem[:-len(suffix)]
                    if alt_stem in mask_index:
                        pairs.append((p, mask_index[alt_stem]))
                        break

    return pairs


def load_mask_union(mask_paths: List[Path]) -> Optional[np.ndarray]:
    """Load and union multiple masks."""
    union = None
    for p in mask_paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = (m > 127).astype(np.uint8)
        if union is None:
            union = m
        else:
            if union.shape != m.shape:
                m = cv2.resize(m, (union.shape[1], union.shape[0]), interpolation=cv2.INTER_NEAREST)
            union = np.maximum(union, m)
    return union


class LeafCompletionDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        size: int = 256,
        occ_fraction: Tuple[float, float] = (0.15, 0.5),
        n_occluders: Tuple[int, int] = (1, 4),
        use_geom: bool = True,
        edge_bias: float = 0.5,
        shape_only: bool = False,  # If True, input is just visible mask (no RGB)
        augment: bool = True,
    ):
        self.size = size
        self.occ_fraction = occ_fraction
        self.n_occluders = n_occluders
        self.use_geom = use_geom
        self.edge_bias = edge_bias
        self.shape_only = shape_only
        self.augment = augment

        # Find pairs
        self.pairs = find_image_mask_pairs(Path(images_dir), Path(masks_dir))
        if not self.pairs:
            raise RuntimeError(f"No image-mask pairs found in {images_dir} / {masks_dir}")

        _print(f"Found {len(self.pairs)} image-mask pairs")

        # Build occluder bank from first N images
        self.occluder_bank = []
        for img_path, mask_paths in self.pairs[:min(100, len(self.pairs))]:
            rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if rgb is None:
                continue
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            mask = load_mask_union(mask_paths)
            if mask is None or mask.sum() < 100:
                continue
            # Create RGBA
            alpha = (mask * 255).astype(np.uint8)
            rgba = np.dstack([rgb, alpha])
            self.occluder_bank.append(rgba)

        _print(f"Built occluder bank with {len(self.occluder_bank)} items")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_paths = self.pairs[idx]

        # Load image and mask
        rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        full_mask = load_mask_union(mask_paths)

        if rgb is None or full_mask is None:
            raise RuntimeError(f"Failed to load {img_path}")

        # Resize mask to match image if needed
        if full_mask.shape[:2] != rgb.shape[:2]:
            full_mask = cv2.resize(full_mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Augment
        if self.augment:
            rgb, full_mask = augment_pair(rgb, full_mask)

        # Letterbox to square
        rgb_sq, info = letterbox(rgb, self.size, pad_value=255)
        mask_sq, _ = letterbox((full_mask * 255).astype(np.uint8), self.size, pad_value=0)
        mask_sq = (mask_sq > 127).astype(np.uint8)

        # Generate occlusion
        occluded_rgb, visible_mask = generate_occlusion(
            rgb_sq, mask_sq,
            self.occluder_bank,
            occ_fraction_range=self.occ_fraction,
            n_occluders=self.n_occluders,
            use_geom=self.use_geom,
            edge_bias=self.edge_bias,
        )

        # Prepare tensors
        if self.shape_only:
            # Input: just visible mask
            x = torch.from_numpy(visible_mask[None, ...].astype(np.float32) / 255.0)
        else:
            # Input: RGB + visible mask
            rgb_t = torch.from_numpy(occluded_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
            vis_t = torch.from_numpy(visible_mask[None, ...].astype(np.float32) / 255.0)
            x = torch.cat([rgb_t, vis_t], dim=0)

        # Target: full mask
        y = torch.from_numpy(mask_sq[None, ...].astype(np.float32))

        return x, y


# =============================================================================
# Loss Functions
# =============================================================================

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return (1 - dice).mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    return bce + dice


def compute_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) >= threshold).float()
    intersection = (preds * targets).sum()
    union = ((preds + targets) > 0).float().sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (intersection / union).item()


# =============================================================================
# Training
# =============================================================================

def train(args):
    set_seed(args.seed)

    # Device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        _print("CUDA not available, using CPU")
        device = 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        _print("MPS not available, using CPU")
        device = 'cpu'

    _print(f"Using device: {device}")

    # Dataset
    dataset = LeafCompletionDataset(
        images_dir=Path(args.images),
        masks_dir=Path(args.masks),
        size=args.size,
        occ_fraction=(args.occ_min, args.occ_max),
        n_occluders=(args.occ_count_min, args.occ_count_max),
        use_geom=args.use_geom,
        edge_bias=args.edge_bias,
        shape_only=args.shape_only,
        augment=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # Model
    in_channels = 1 if args.shape_only else 4
    model = UNetCompletion(in_ch=in_channels, base_ch=args.base_channels).to(device)

    # Resume if specified
    if args.resume and Path(args.resume).exists():
        _print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    _print(f"Training config:")
    _print(f"  Samples: {len(dataset)}")
    _print(f"  Steps: {args.steps}")
    _print(f"  Batch: {args.batch}")
    _print(f"  LR: {args.lr}")
    _print(f"  Size: {args.size}")
    _print(f"  Shape-only mode: {args.shape_only}")
    _print(f"  Occlusion: {args.occ_min}-{args.occ_max}")

    # Training loop
    data_iter = iter(dataloader)
    ema_loss = None

    for step in range(1, args.steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x = x.to(device)
        y = y.to(device)

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
            lr = scheduler.get_last_lr()[0]
            _print(f"[{step:5d}/{args.steps}] loss={ema_loss:.4f} iou={iou:.4f} lr={lr:.2e}")

        # Save checkpoint periodically
        if step % 1000 == 0:
            ckpt_path = Path(args.output).with_suffix(f'.step{step}.pth')
            torch.save({
                'state_dict': model.state_dict(),
                'step': step,
                'shape_only': args.shape_only,
                'size': args.size,
            }, ckpt_path)
            _print(f"Saved checkpoint: {ckpt_path}")

    # Save final model
    torch.save({
        'state_dict': model.state_dict(),
        'step': args.steps,
        'shape_only': args.shape_only,
        'size': args.size,
    }, args.output)
    _print(f"Done! Saved to {args.output}")


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def predict(
    model: nn.Module,
    rgb: np.ndarray,
    visible_mask: np.ndarray,
    size: int,
    device: str,
    shape_only: bool = False,
    threshold: float = 0.5,
) -> np.ndarray:
    """Run inference to complete a partial mask."""
    model.eval()

    # Letterbox
    rgb_sq, info = letterbox(rgb, size, pad_value=255)
    vis_sq, _ = letterbox(visible_mask, size, pad_value=0)

    # Prepare input
    if shape_only:
        x = torch.from_numpy(vis_sq[None, None, ...].astype(np.float32) / 255.0)
    else:
        rgb_t = torch.from_numpy(rgb_sq.astype(np.float32) / 255.0).permute(2, 0, 1)
        vis_t = torch.from_numpy(vis_sq[None, ...].astype(np.float32) / 255.0)
        x = torch.cat([rgb_t, vis_t], dim=0).unsqueeze(0)

    x = x.to(device)

    # Predict
    logits = model(x)
    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask_sq = (probs >= threshold).astype(np.uint8) * 255

    # Unletterbox
    mask = unletterbox(mask_sq, info)
    return mask


def infer(args):
    # Load model
    _print(f"Loading model from {args.model}")
    ckpt = torch.load(args.model, map_location='cpu')
    shape_only = ckpt.get('shape_only', False)
    size = ckpt.get('size', 256)

    in_ch = 1 if shape_only else 4
    model = UNetCompletion(in_ch=in_ch, base_ch=32)
    model.load_state_dict(ckpt['state_dict'])

    device = args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    model.to(device)
    model.eval()

    # Load inputs
    _print(f"Loading image: {args.image}")
    rgb = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Could not load {args.image}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    _print(f"Loading visible mask: {args.visible_mask}")
    vis_mask = cv2.imread(args.visible_mask, cv2.IMREAD_GRAYSCALE)
    if vis_mask is None:
        raise FileNotFoundError(f"Could not load {args.visible_mask}")

    # Run prediction
    _print("Running completion...")
    completed = predict(model, rgb, vis_mask, size, device, shape_only, args.threshold)

    # Save
    _print(f"Saving to {args.output}")
    cv2.imwrite(args.output, completed)

    # Optional visualization
    if args.visualize:
        vis_path = Path(args.output).with_suffix('.viz.png')
        # Create side-by-side: original | visible mask | completed
        h, w = rgb.shape[:2]
        vis_resized = cv2.resize(vis_mask, (w, h))
        comp_resized = cv2.resize(completed, (w, h))

        viz = np.concatenate([
            rgb,
            cv2.cvtColor(vis_resized, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(comp_resized, cv2.COLOR_GRAY2RGB),
        ], axis=1)
        cv2.imwrite(str(vis_path), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
        _print(f"Visualization saved to {vis_path}")

    _print("Done!")


# =============================================================================
# Test (validate with synthetic occlusion)
# =============================================================================

def test(args):
    """Test model on images with synthetic occlusion."""
    # Load model
    ckpt = torch.load(args.model, map_location='cpu')
    shape_only = ckpt.get('shape_only', False)
    size = ckpt.get('size', 256)

    in_ch = 1 if shape_only else 4
    model = UNetCompletion(in_ch=in_ch, base_ch=32)
    model.load_state_dict(ckpt['state_dict'])

    device = args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    model.to(device)
    model.eval()

    # Find test images
    pairs = find_image_mask_pairs(Path(args.images), Path(args.masks))
    if not pairs:
        _print("No test images found")
        return

    # Output dir
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    _print(f"Testing on {len(pairs)} images...")

    ious = []
    for i, (img_path, mask_paths) in enumerate(pairs[:args.max_images]):
        rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        full_mask = load_mask_union(mask_paths)

        if rgb is None or full_mask is None:
            continue

        if full_mask.shape[:2] != rgb.shape[:2]:
            full_mask = cv2.resize(full_mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Letterbox
        rgb_sq, info = letterbox(rgb, size, pad_value=255)
        mask_sq, _ = letterbox((full_mask * 255).astype(np.uint8), size, pad_value=0)
        mask_sq = (mask_sq > 127).astype(np.uint8)

        # Generate synthetic occlusion
        occluded_rgb, visible_mask = generate_occlusion(
            rgb_sq, mask_sq, [],
            occ_fraction_range=(0.2, 0.4),
            n_occluders=(2, 4),
            use_geom=True,
            edge_bias=0.5,
        )

        # Predict
        completed = predict(model, occluded_rgb, visible_mask, size, device, shape_only)

        # Compute IoU
        pred_binary = (completed > 127).astype(np.float32)
        gt_binary = mask_sq.astype(np.float32)
        intersection = (pred_binary * gt_binary).sum()
        union = ((pred_binary + gt_binary) > 0).sum()
        iou = intersection / max(union, 1)
        ious.append(iou)

        # Save visualization
        viz = np.concatenate([
            occluded_rgb,
            cv2.cvtColor(visible_mask, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(completed, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor((mask_sq * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB),
        ], axis=1)
        viz_path = out_dir / f"{img_path.stem}_test.png"
        cv2.imwrite(str(viz_path), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))

        if (i + 1) % 10 == 0:
            _print(f"  Processed {i+1}/{min(len(pairs), args.max_images)}")

    mean_iou = np.mean(ious) if ious else 0
    _print(f"Mean IoU: {mean_iou:.4f}")
    _print(f"Results saved to {out_dir}")


# =============================================================================
# Auto Mask Generation (for leaves on white background)
# =============================================================================

def auto_generate_mask(rgb: np.ndarray, method: str = 'color') -> np.ndarray:
    """
    Auto-generate mask for leaf on white/light background.

    Methods:
    - 'color': Use HSV color thresholding (good for green leaves)
    - 'threshold': Simple brightness threshold
    - 'edge': Edge detection + flood fill
    """
    if method == 'color':
        # Convert to HSV and threshold for green
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # Green hue range (roughly 35-85 in OpenCV's 0-180 scale)
        lower_green = np.array([25, 30, 30])
        upper_green = np.array([95, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Also catch yellowish-green
        lower_yellow = np.array([15, 30, 30])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.bitwise_or(mask, mask_yellow)

    elif method == 'threshold':
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # Leaves are darker than white background
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    else:  # edge
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # Dilate edges and flood fill
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)
        # Flood fill from corners (assumed background)
        h, w = edges.shape
        flood = edges.copy()
        cv2.floodFill(flood, None, (0, 0), 255)
        cv2.floodFill(flood, None, (w-1, 0), 255)
        cv2.floodFill(flood, None, (0, h-1), 255)
        cv2.floodFill(flood, None, (w-1, h-1), 255)
        mask = cv2.bitwise_not(flood)

    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep only largest component
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest], -1, 255, -1)

    return mask


def generate_masks(args):
    """Auto-generate masks for leaf images."""
    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    images = [p for p in in_dir.iterdir() if p.suffix.lower() in exts]

    _print(f"Generating masks for {len(images)} images...")

    for i, img_path in enumerate(images):
        rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        mask = auto_generate_mask(rgb, method=args.method)

        # Save mask
        mask_path = out_dir / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), mask)

        # Optionally save visualization
        if args.visualize:
            viz = rgb.copy()
            viz[mask > 0] = (viz[mask > 0] * 0.5 + np.array([255, 0, 255]) * 0.5).astype(np.uint8)
            viz_path = out_dir / f"{img_path.stem}_viz.png"
            cv2.imwrite(str(viz_path), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))

        if (i + 1) % 10 == 0:
            _print(f"  Processed {i+1}/{len(images)}")

    _print(f"Done! Masks saved to {out_dir}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Leaf Completion v2")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Train command
    train_p = subparsers.add_parser('train', help='Train the completion model')
    train_p.add_argument('--images', required=True, help='Images directory')
    train_p.add_argument('--masks', required=True, help='Masks directory')
    train_p.add_argument('--output', required=True, help='Output model path (.pth)')
    train_p.add_argument('--steps', type=int, default=5000)
    train_p.add_argument('--batch', type=int, default=4)
    train_p.add_argument('--lr', type=float, default=1e-4)
    train_p.add_argument('--size', type=int, default=256)
    train_p.add_argument('--base-channels', type=int, default=32)
    train_p.add_argument('--device', default='mps', choices=['cpu', 'cuda', 'mps'])
    train_p.add_argument('--seed', type=int, default=42)
    train_p.add_argument('--resume', default=None, help='Resume from checkpoint')
    # Occlusion settings
    train_p.add_argument('--occ-min', type=float, default=0.15, help='Min occlusion fraction')
    train_p.add_argument('--occ-max', type=float, default=0.50, help='Max occlusion fraction')
    train_p.add_argument('--occ-count-min', type=int, default=1)
    train_p.add_argument('--occ-count-max', type=int, default=4)
    train_p.add_argument('--use-geom', action='store_true', default=True)
    train_p.add_argument('--no-geom', dest='use_geom', action='store_false')
    train_p.add_argument('--edge-bias', type=float, default=0.5)
    # Mode
    train_p.add_argument('--shape-only', action='store_true',
                         help='Use only visible mask as input (no RGB)')

    # Infer command
    infer_p = subparsers.add_parser('infer', help='Run inference')
    infer_p.add_argument('--model', required=True, help='Model checkpoint')
    infer_p.add_argument('--image', required=True, help='Input image')
    infer_p.add_argument('--visible-mask', required=True, help='Visible/partial mask')
    infer_p.add_argument('--output', required=True, help='Output completed mask')
    infer_p.add_argument('--device', default='mps', choices=['cpu', 'cuda', 'mps'])
    infer_p.add_argument('--threshold', type=float, default=0.5)
    infer_p.add_argument('--visualize', action='store_true')

    # Test command
    test_p = subparsers.add_parser('test', help='Test with synthetic occlusion')
    test_p.add_argument('--model', required=True, help='Model checkpoint')
    test_p.add_argument('--images', required=True, help='Test images directory')
    test_p.add_argument('--masks', required=True, help='Test masks directory')
    test_p.add_argument('--output', required=True, help='Output directory for results')
    test_p.add_argument('--device', default='mps', choices=['cpu', 'cuda', 'mps'])
    test_p.add_argument('--max-images', type=int, default=50)

    # Generate masks command
    genmask_p = subparsers.add_parser('generate-masks', help='Auto-generate masks for leaves on white background')
    genmask_p.add_argument('--input', required=True, help='Input images directory')
    genmask_p.add_argument('--output', required=True, help='Output masks directory')
    genmask_p.add_argument('--method', default='color', choices=['color', 'threshold', 'edge'],
                           help='Mask generation method')
    genmask_p.add_argument('--visualize', action='store_true', help='Save visualizations')

    args = parser.parse_args()

    if args.command == 'train':
        train(args)
    elif args.command == 'infer':
        infer(args)
    elif args.command == 'test':
        test(args)
    elif args.command == 'generate-masks':
        generate_masks(args)


if __name__ == '__main__':
    main()
