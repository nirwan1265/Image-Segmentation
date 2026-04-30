"""
Tip-only semantic segmentation model (no SAM).

This module provides:
- A lightweight U-Net model for binary tip segmentation
- Consistent letterbox preprocessing + unletterbox postprocessing
- Checkpoint load/save helpers

The checkpoint format is a dict saved via torch.save with keys:
  - state_dict: model weights
  - arch: string (currently "unet_small")
  - input_size: int (square size the model was trained on)
  - threshold: float (default mask threshold)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Model
# ----------------------------


def _gn_groups(ch: int, max_groups: int = 8) -> int:
    """Pick a GroupNorm group count that divides ch."""
    g = min(max_groups, ch)
    while g > 1 and (ch % g) != 0:
        g -= 1
    return max(1, g)


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        g1 = _gn_groups(out_ch)
        g2 = _gn_groups(out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g1, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g2, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):  # noqa: D401
        return self.net(x)


class UNetSmall(nn.Module):
    """A small U-Net for 1-class segmentation."""

    def __init__(self, in_ch: int = 3, base_ch: int = 32, out_ch: int = 1):
        super().__init__()
        c1, c2, c3, c4, c5 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16

        self.enc1 = _DoubleConv(in_ch, c1)
        self.enc2 = _DoubleConv(c1, c2)
        self.enc3 = _DoubleConv(c2, c3)
        self.enc4 = _DoubleConv(c3, c4)
        self.bot = _DoubleConv(c4, c5)

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.Conv2d(c5, c4, 1)
        self.dec4 = _DoubleConv(c4 + c4, c4)
        self.up3 = nn.Conv2d(c4, c3, 1)
        self.dec3 = _DoubleConv(c3 + c3, c3)
        self.up2 = nn.Conv2d(c3, c2, 1)
        self.dec2 = _DoubleConv(c2 + c2, c2)
        self.up1 = nn.Conv2d(c2, c1, 1)
        self.dec1 = _DoubleConv(c1 + c1, c1)

        self.out = nn.Conv2d(c1, out_ch, 1)

    def forward(self, x):  # noqa: D401
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bot(self.pool(e4))

        d4 = F.interpolate(b, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.up4(d4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = F.interpolate(d4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.up3(d3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.up2(d2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.up1(d1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


class UNetResNet18(nn.Module):
    """U-Net decoder with ResNet18 encoder (ImageNet-pretrained optional)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision import models
        except Exception as e:
            raise RuntimeError("torchvision is required for UNetResNet18.") from e

        # load encoder
        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None
        try:
            self.encoder = models.resnet18(weights=weights)
        except Exception:
            # If weights download fails, fall back to random init
            self.encoder = models.resnet18(weights=None)

        # Encoder layers
        self.enc0 = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
        )  # /2
        self.pool = self.encoder.maxpool            # /4
        self.enc1 = self.encoder.layer1             # /4
        self.enc2 = self.encoder.layer2             # /8
        self.enc3 = self.encoder.layer3             # /16
        self.enc4 = self.encoder.layer4             # /32

        # Decoder
        self.up4 = nn.Conv2d(512, 256, 1)
        self.dec4 = _DoubleConv(256 + 256, 256)
        self.up3 = nn.Conv2d(256, 128, 1)
        self.dec3 = _DoubleConv(128 + 128, 128)
        self.up2 = nn.Conv2d(128, 64, 1)
        self.dec2 = _DoubleConv(64 + 64, 64)
        self.up1 = nn.Conv2d(64, 64, 1)
        self.dec1 = _DoubleConv(64 + 64, 64)
        self.up0 = nn.Conv2d(64, 32, 1)
        self.dec0 = _DoubleConv(32, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):  # noqa: D401
        x0 = self.enc0(x)                 # /2
        x1 = self.enc1(self.pool(x0))     # /4
        x2 = self.enc2(x1)                # /8
        x3 = self.enc3(x2)                # /16
        x4 = self.enc4(x3)                # /32

        d4 = F.interpolate(x4, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.up4(d4)
        d4 = self.dec4(torch.cat([d4, x3], dim=1))

        d3 = F.interpolate(d4, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.up3(d3)
        d3 = self.dec3(torch.cat([d3, x2], dim=1))

        d2 = F.interpolate(d3, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.up2(d2)
        d2 = self.dec2(torch.cat([d2, x1], dim=1))

        d1 = F.interpolate(d2, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.up1(d1)
        d1 = self.dec1(torch.cat([d1, x0], dim=1))

        d0 = F.interpolate(d1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        d0 = self.up0(d0)
        d0 = self.dec0(d0)

        return self.out(d0)


def build_tipseg_model(arch: str = "unet_small", pretrained: bool = True) -> nn.Module:
    arch = (arch or "unet_small").strip().lower()
    if arch == "unet_small":
        return UNetSmall(in_ch=3, base_ch=32, out_ch=1)
    if arch == "unet_resnet18":
        return UNetResNet18(pretrained=bool(pretrained))
    raise ValueError(f"Unknown tipseg arch: {arch}")


# ----------------------------
# Pre/Post processing
# ----------------------------


@dataclass(frozen=True)
class LetterboxMeta:
    scale: float
    x0: int
    y0: int
    new_w: int
    new_h: int
    orig_w: int
    orig_h: int


def letterbox_rgb(img_rgb_u8: np.ndarray, size: int, pad_value: int = 255) -> Tuple[np.ndarray, LetterboxMeta]:
    """Resize while keeping aspect ratio, pad to square."""
    if img_rgb_u8 is None:
        raise ValueError("img_rgb_u8 is None")
    if img_rgb_u8.dtype != np.uint8:
        img_rgb_u8 = np.clip(img_rgb_u8, 0, 255).astype(np.uint8)

    H, W = img_rgb_u8.shape[:2]
    size = int(size)
    if size <= 0:
        raise ValueError("size must be > 0")

    scale = float(size) / float(max(H, W))
    new_w = max(1, int(round(W * scale)))
    new_h = max(1, int(round(H * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img_rgb_u8, (new_w, new_h), interpolation=interp)

    canvas = np.full((size, size, 3), int(pad_value), dtype=np.uint8)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    meta = LetterboxMeta(scale=scale, x0=x0, y0=y0, new_w=new_w, new_h=new_h, orig_w=W, orig_h=H)
    return canvas, meta


def letterbox_mask(mask_u8: np.ndarray, size: int, meta: LetterboxMeta) -> np.ndarray:
    """Letterbox a binary mask using precomputed meta."""
    if mask_u8.dtype != np.uint8:
        mask_u8 = (mask_u8 > 0).astype(np.uint8)

    resized = cv2.resize(mask_u8, (meta.new_w, meta.new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[meta.y0:meta.y0 + meta.new_h, meta.x0:meta.x0 + meta.new_w] = resized
    return canvas


def unletterbox_mask(mask_sq: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Undo letterboxing: crop padding then resize back to original image size."""
    cropped = mask_sq[meta.y0:meta.y0 + meta.new_h, meta.x0:meta.x0 + meta.new_w]
    out = cv2.resize(cropped, (meta.orig_w, meta.orig_h), interpolation=cv2.INTER_NEAREST)
    return out


def _keep_components(mask_u8: np.ndarray, min_area: int = 0, keep_largest: bool = True) -> np.ndarray:
    mask_u8 = (mask_u8 > 0).astype(np.uint8)
    if mask_u8.sum() == 0:
        return mask_u8

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return mask_u8

    # stats[0] is background
    comps = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        comps.append((area, i))
    if not comps:
        return np.zeros_like(mask_u8)

    if keep_largest:
        _, i = max(comps, key=lambda t: t[0])
        return (labels == i).astype(np.uint8)

    out = np.zeros_like(mask_u8)
    for _, i in comps:
        out[labels == i] = 1
    return out


@torch.no_grad()
def predict_tip_mask(
    model: nn.Module,
    img_rgb_u8: np.ndarray,
    input_size: int,
    device: str = "cpu",
    threshold: float = 0.5,
    min_area: int = 0,
    keep_largest: bool = True,
) -> np.ndarray:
    """Return a boolean mask in original image size."""
    model.eval()
    input_size = int(input_size)
    thr = float(threshold)

    sq, meta = letterbox_rgb(img_rgb_u8, input_size, pad_value=255)
    x = torch.from_numpy(sq).to(torch.float32) / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    x = x.to(device)

    logits = model(x)
    probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    mask_sq = (probs >= thr).astype(np.uint8)
    mask_sq = _keep_components(mask_sq, min_area=min_area, keep_largest=keep_largest)
    mask = unletterbox_mask(mask_sq, meta)
    return mask.astype(bool)


# ----------------------------
# Checkpoints
# ----------------------------


def save_tipseg_checkpoint(path: str | Path, model: nn.Module, input_size: int, threshold: float = 0.5, arch: str = "unet_small") -> None:
    save_dict: Dict[str, Any] = {
        "state_dict": model.state_dict(),
        "arch": arch,
        "input_size": int(input_size),
        "threshold": float(threshold),
    }
    torch.save(save_dict, str(path))


def load_tipseg_checkpoint(path: str | Path, device: str = "cpu") -> Tuple[nn.Module, Dict[str, Any]]:
    p = str(path)
    try:
        ckpt = torch.load(p, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(p, map_location=device)

    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError("Invalid tipseg checkpoint (expected dict with state_dict).")

    arch = ckpt.get("arch", "unet_small")
    # pretrained=False because we will load full weights from checkpoint
    model = build_tipseg_model(arch=arch, pretrained=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()

    meta = {
        "arch": arch,
        "input_size": int(ckpt.get("input_size", 512)),
        "threshold": float(ckpt.get("threshold", 0.5)),
        "device": device,
    }
    return model, meta
