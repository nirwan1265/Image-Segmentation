"""
Leaf completion model (no SAM).

Predict a full leaf mask from an occluded RGB image + a partial (visible) mask.
Input channels = 4 (RGB + partial mask).
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
# Model blocks
# ----------------------------


def _gn_groups(ch: int, max_groups: int = 8) -> int:
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

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch: int = 4, base_ch: int = 32, out_ch: int = 1):
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

    def forward(self, x):
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


class UNetResNet(nn.Module):
    """U-Net decoder with ResNet encoder (18/34/50)."""

    def __init__(self, resnet_name: str = "resnet18", in_ch: int = 4, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision import models
        except Exception as e:
            raise RuntimeError("torchvision is required for UNetResNet.") from e

        resnet_name = (resnet_name or "resnet18").lower()
        if resnet_name not in ("resnet18", "resnet34", "resnet50"):
            raise ValueError(f"Unsupported ResNet: {resnet_name}")

        weights = None
        if pretrained:
            try:
                weights = {
                    "resnet18": models.ResNet18_Weights.DEFAULT,
                    "resnet34": models.ResNet34_Weights.DEFAULT,
                    "resnet50": models.ResNet50_Weights.DEFAULT,
                }[resnet_name]
            except Exception:
                weights = None

        try:
            self.encoder = getattr(models, resnet_name)(weights=weights)
        except Exception:
            self.encoder = getattr(models, resnet_name)(weights=None)

        # Replace conv1 if in_ch != 3
        if in_ch != 3:
            old = self.encoder.conv1
            new = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=False)
            with torch.no_grad():
                if old.weight.shape[1] == 3:
                    new.weight[:, :3] = old.weight
                    if in_ch > 3:
                        mean = old.weight.mean(dim=1, keepdim=True)
                        new.weight[:, 3:in_ch] = mean.repeat(1, in_ch - 3, 1, 1)
                else:
                    new.weight.copy_(old.weight)
            self.encoder.conv1 = new

        # Encoder stages
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

        if resnet_name in ("resnet18", "resnet34"):
            c0, c1, c2, c3, c4 = 64, 64, 128, 256, 512
        else:
            # resnet50 bottleneck channels
            c0, c1, c2, c3, c4 = 64, 256, 512, 1024, 2048

        self.up4 = nn.Conv2d(c4, c3, 1)
        self.dec4 = _DoubleConv(c3 + c3, c3)
        self.up3 = nn.Conv2d(c3, c2, 1)
        self.dec3 = _DoubleConv(c2 + c2, c2)
        self.up2 = nn.Conv2d(c2, c1, 1)
        self.dec2 = _DoubleConv(c1 + c1, c1)
        self.up1 = nn.Conv2d(c1, c0, 1)
        self.dec1 = _DoubleConv(c0 + c0, c0)
        self.up0 = nn.Conv2d(c0, 32, 1)
        self.dec0 = _DoubleConv(32, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x0 = self.enc0(x)
        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

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


class UNetEfficientNet(nn.Module):
    """U-Net decoder with EfficientNet encoder (b3/b4)."""

    def __init__(self, variant: str = "b3", in_ch: int = 4, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision import models
        except Exception as e:
            raise RuntimeError("torchvision is required for UNetEfficientNet.") from e

        variant = (variant or "b3").lower()
        if variant not in ("b3", "b4"):
            raise ValueError(f"Unsupported EfficientNet variant: {variant}")

        weights = None
        if pretrained:
            try:
                weights = {
                    "b3": models.EfficientNet_B3_Weights.DEFAULT,
                    "b4": models.EfficientNet_B4_Weights.DEFAULT,
                }[variant]
            except Exception:
                weights = None

        try:
            self.encoder = getattr(models, f"efficientnet_{variant}")(weights=weights)
        except Exception:
            self.encoder = getattr(models, f"efficientnet_{variant}")(weights=None)

        # Replace first conv if in_ch != 3
        if in_ch != 3:
            first = self.encoder.features[0][0]
            new = nn.Conv2d(in_ch, first.out_channels, kernel_size=first.kernel_size,
                            stride=first.stride, padding=first.padding, bias=False)
            with torch.no_grad():
                if first.weight.shape[1] == 3:
                    new.weight[:, :3] = first.weight
                    if in_ch > 3:
                        mean = first.weight.mean(dim=1, keepdim=True)
                        new.weight[:, 3:in_ch] = mean.repeat(1, in_ch - 3, 1, 1)
                else:
                    new.weight.copy_(first.weight)
            self.encoder.features[0][0] = new

        # Infer feature indices & channels once (CPU)
        self._feat_idxs, self._feat_chs = self._infer_feature_info(in_ch=in_ch)

        c0, c1, c2, c3, c4 = self._feat_chs
        self.up4 = nn.Conv2d(c4, c3, 1)
        self.dec4 = _DoubleConv(c3 + c3, c3)
        self.up3 = nn.Conv2d(c3, c2, 1)
        self.dec3 = _DoubleConv(c2 + c2, c2)
        self.up2 = nn.Conv2d(c2, c1, 1)
        self.dec2 = _DoubleConv(c1 + c1, c1)
        self.up1 = nn.Conv2d(c1, c0, 1)
        self.dec1 = _DoubleConv(c0 + c0, c0)
        self.up0 = nn.Conv2d(c0, 32, 1)
        self.dec0 = _DoubleConv(32, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def _infer_feature_info(self, in_ch: int = 4):
        feats = []
        x = torch.zeros(1, in_ch, 256, 256)
        prev_hw = None
        with torch.no_grad():
            for i, layer in enumerate(self.encoder.features):
                x = layer(x)
                hw = x.shape[-2:]
                if prev_hw is None or hw != prev_hw:
                    feats.append((i, x.shape[1], hw))
                    prev_hw = hw
        if len(feats) < 5:
            feats = [(i, x.shape[1], x.shape[-2:]) for i in range(len(self.encoder.features))]
        if len(feats) > 5:
            feats = [feats[0]] + feats[-4:]
        idxs = [f[0] for f in feats]
        chs = [f[1] for f in feats]
        return idxs, chs

    def forward(self, x):
        orig_size = x.shape[-2:]  # save original input size before encoder overwrites x
        feats = []
        for i, layer in enumerate(self.encoder.features):
            x = layer(x)
            if i in self._feat_idxs:
                feats.append(x)
        if len(feats) < 5:
            # fallback: pad with last feature
            while len(feats) < 5:
                feats.append(feats[-1])
        f0, f1, f2, f3, f4 = feats[-5:]

        d4 = F.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.up4(d4)
        d4 = self.dec4(torch.cat([d4, f3], dim=1))

        d3 = F.interpolate(d4, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.up3(d3)
        d3 = self.dec3(torch.cat([d3, f2], dim=1))

        d2 = F.interpolate(d3, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.up2(d2)
        d2 = self.dec2(torch.cat([d2, f1], dim=1))

        d1 = F.interpolate(d2, size=f0.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.up1(d1)
        d1 = self.dec1(torch.cat([d1, f0], dim=1))

        d0 = F.interpolate(d1, size=orig_size, mode="bilinear", align_corners=False)
        d0 = self.up0(d0)
        d0 = self.dec0(d0)
        return self.out(d0)


class UNetPlusPlusWrapper(nn.Module):
    """Optional U-Net++ via segmentation_models_pytorch (if available)."""

    def __init__(self, encoder_name: str = "resnet34", in_ch: int = 4, pretrained: bool = True):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except Exception as e:
            raise RuntimeError("segmentation_models_pytorch is required for U-Net++.") from e
        encoder_weights = "imagenet" if pretrained else None
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_ch,
            classes=1,
            activation=None,
        )

    def forward(self, x):
        return self.model(x)


def build_completion_model(arch: str = "unet_resnet18", pretrained: bool = True) -> nn.Module:
    arch = (arch or "unet_resnet18").strip().lower()
    if arch == "unet_small":
        return UNetSmall(in_ch=4, base_ch=32, out_ch=1)
    if arch in ("unet_resnet18", "unet_resnet34", "unet_resnet50"):
        resnet_name = arch.replace("unet_", "")
        return UNetResNet(resnet_name=resnet_name, in_ch=4, pretrained=bool(pretrained))
    if arch in ("unet_efficientnet_b3", "unet_efficientnet_b4"):
        variant = arch.split("_")[-1]
        return UNetEfficientNet(variant=variant, in_ch=4, pretrained=bool(pretrained))
    if arch == "unetpp_resnet34":
        return UNetPlusPlusWrapper(encoder_name="resnet34", in_ch=4, pretrained=bool(pretrained))
    if arch == "unetpp_resnet50":
        return UNetPlusPlusWrapper(encoder_name="resnet50", in_ch=4, pretrained=bool(pretrained))
    raise ValueError(f"Unknown completion arch: {arch}")


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
    if img_rgb_u8 is None:
        raise ValueError("img_rgb_u8 is None")
    if img_rgb_u8.dtype != np.uint8:
        img_rgb_u8 = np.clip(img_rgb_u8, 0, 255).astype(np.uint8)
    H, W = img_rgb_u8.shape[:2]
    scale = float(size) / float(max(H, W))
    new_w = int(round(W * scale))
    new_h = int(round(H * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img_rgb_u8, (new_w, new_h), interpolation=interp)
    canvas = np.full((size, size, 3), int(pad_value), dtype=np.uint8)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    meta = LetterboxMeta(scale=scale, x0=x0, y0=y0, new_w=new_w, new_h=new_h, orig_w=W, orig_h=H)
    return canvas, meta


def letterbox_mask(mask_u8: np.ndarray, size: int, meta: LetterboxMeta) -> np.ndarray:
    if mask_u8.dtype != np.uint8:
        mask_u8 = (mask_u8 > 0).astype(np.uint8)
    resized = cv2.resize(mask_u8, (meta.new_w, meta.new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[meta.y0:meta.y0 + meta.new_h, meta.x0:meta.x0 + meta.new_w] = resized
    return canvas


def unletterbox_mask(mask_sq: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    cropped = mask_sq[meta.y0:meta.y0 + meta.new_h, meta.x0:meta.x0 + meta.new_w]
    out = cv2.resize(cropped, (meta.orig_w, meta.orig_h), interpolation=cv2.INTER_NEAREST)
    return out


@torch.no_grad()
def predict_completion_mask(
    model: nn.Module,
    img_rgb_u8: np.ndarray,
    partial_mask: np.ndarray,
    input_size: int,
    device: str = "cpu",
    threshold: float = 0.5,
) -> np.ndarray:
    """Return boolean completed mask in original image size."""
    model.eval()
    input_size = int(input_size)
    thr = float(threshold)

    sq, meta = letterbox_rgb(img_rgb_u8, input_size, pad_value=255)
    pm = (partial_mask > 0).astype(np.uint8)
    pm_sq = letterbox_mask(pm, input_size, meta)

    x = torch.from_numpy(sq).to(torch.float32) / 255.0
    x = x.permute(2, 0, 1)  # (3,H,W)
    pm_t = torch.from_numpy(pm_sq[None, ...]).to(torch.float32)
    x = torch.cat([x, pm_t], dim=0).unsqueeze(0).to(device)  # (1,4,H,W)

    logits = model(x)
    probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    mask_sq = (probs >= thr).astype(np.uint8)
    mask = unletterbox_mask(mask_sq, meta)
    return mask.astype(bool)


# ----------------------------
# Checkpoints
# ----------------------------


def save_completion_checkpoint(path: str | Path, model: nn.Module, input_size: int, threshold: float = 0.5,
                              arch: str = "unet_resnet18", pretrained: bool = True) -> None:
    save_dict: Dict[str, Any] = {
        "state_dict": model.state_dict(),
        "arch": arch,
        "input_size": int(input_size),
        "threshold": float(threshold),
        "pretrained": bool(pretrained),
    }
    torch.save(save_dict, str(path))


def load_completion_checkpoint(path: str | Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(str(path), map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError("Invalid completion checkpoint (missing state_dict)")
    arch = ckpt.get("arch", "unet_resnet18")
    pretrained = bool(ckpt.get("pretrained", True))
    model = build_completion_model(arch=arch, pretrained=pretrained)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, ckpt
