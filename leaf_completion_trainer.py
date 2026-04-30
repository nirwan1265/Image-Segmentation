#!/usr/bin/env python3
"""
Train a leaf completion model (no SAM).

Input: occluded RGB image + partial (visible) mask.
Target: full leaf mask.
"""
from __future__ import annotations

import argparse
import random
import os
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from leaf_completion_model import (
    build_completion_model,
    save_completion_checkpoint,
)


def _p(msg: str):
    print(msg, flush=True)


def _overlay_mask(rgb_u8: np.ndarray, mask: np.ndarray, color=(255, 0, 255), alpha: float = 0.5) -> np.ndarray:
    out = rgb_u8.copy()
    m = mask.astype(bool)
    if m.any():
        overlay = np.zeros_like(out, dtype=np.uint8)
        overlay[:] = np.array(color, dtype=np.uint8)
        out[m] = (alpha * overlay[m] + (1 - alpha) * out[m]).astype(np.uint8)
    return out


def _prepare_square(rgb: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    H, W = rgb.shape[:2]
    scale = size / max(H, W)
    newW, newH = int(round(W * scale)), int(round(H * scale))
    rgb = cv2.resize(rgb, (newW, newH), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (newW, newH), interpolation=cv2.INTER_NEAREST)
    padL = (size - newW) // 2
    padT = (size - newH) // 2
    rgb_sq = np.full((size, size, 3), 255, np.uint8)
    m_sq = np.zeros((size, size), np.uint8)
    rgb_sq[padT:padT + newH, padL:padL + newW] = rgb
    m_sq[padT:padT + newH, padL:padL + newW] = (mask > 0).astype(np.uint8)
    return rgb_sq, m_sq


def _make_preview_image(
    model: nn.Module,
    samples: list[tuple[np.ndarray, np.ndarray]],
    bank_rgba: list,
    size: int,
    device: str,
    occ_min: float,
    occ_max: float,
    occ_count_min: int,
    occ_count_max: int,
    edge_bias: float,
    edge_band_px: int,
    edge_only: bool,
    use_geom: bool,
    thr: float = 0.5,
) -> np.ndarray | None:
    if not samples:
        return None
    rows = []
    model.eval()
    with torch.no_grad():
        for rgb, full_mask in samples:
            rgb_sq, m_sq = _prepare_square(rgb, full_mask, size)
            img_occ, vis_mask = occlude_with_bank(
                rgb_sq, m_sq, bank_rgba,
                min_occ=occ_min, max_occ=occ_max,
                occ_count_min=occ_count_min, occ_count_max=occ_count_max,
                use_geom=use_geom,
                edge_bias=edge_bias,
                edge_band_px=edge_band_px,
                edge_only=edge_only,
            )
            x = img_occ.astype(np.float32) / 255.0
            x = np.transpose(x, (2, 0, 1))
            vm = vis_mask[None, ...].astype(np.float32)
            x = np.concatenate([x, vm], axis=0)
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)
            logits = model(x_t)
            probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            pred = (probs >= thr).astype(np.uint8)

            vis_overlay = _overlay_mask(img_occ, vis_mask, color=(0, 255, 0), alpha=0.35)
            pred_overlay = _overlay_mask(rgb_sq, pred, color=(255, 0, 255), alpha=0.5)
            gt_overlay = _overlay_mask(rgb_sq, m_sq, color=(0, 255, 255), alpha=0.5)
            row = np.concatenate([vis_overlay, pred_overlay, gt_overlay], axis=1)
            rows.append(row)
    grid = np.concatenate(rows, axis=0)
    return grid


def _random_shape_mask(h, w, center=None):
    m = np.zeros((h, w), np.uint8)
    if h < 2 or w < 2:
        return m
    k = random.choice([1, 2, 3])
    for _ in range(k):
        kind = random.choice(["rect", "ellipse", "poly"])
        if center is None:
            cx, cy = random.randrange(0, w), random.randrange(0, h)
        else:
            cx, cy = int(center[0]), int(center[1])

        if kind == "rect":
            if center is None:
                x1 = random.randrange(0, w - 1)
                y1 = random.randrange(0, h - 1)
                x2 = random.randrange(x1 + 1, w)
                y2 = random.randrange(y1 + 1, h)
            else:
                min_dim = max(6, min(h, w) // 12)
                max_dim = max(min_dim + 2, min(h, w) // 3)
                rw = random.randrange(min_dim, max_dim)
                rh = random.randrange(min_dim, max_dim)
                x1 = max(0, cx - rw // 2)
                y1 = max(0, cy - rh // 2)
                x2 = min(w, x1 + rw)
                y2 = min(h, y1 + rh)
            if x2 > x1 and y2 > y1:
                cv2.rectangle(m, (x1, y1), (x2, y2), 255, -1)
        elif kind == "ellipse":
            ax, ay = max(6, w//12), max(6, h//12)
            ang = random.randrange(0, 180)
            cv2.ellipse(m, (cx, cy), (ax, ay), ang, 0, 360, 255, -1)
        else:
            if center is None:
                pts = np.stack([np.random.randint(0, w, 6), np.random.randint(0, h, 6)], 1).astype(np.int32)
            else:
                min_dim = max(6, min(h, w) // 12)
                max_dim = max(min_dim + 2, min(h, w) // 3)
                angles = np.random.uniform(0, 2*np.pi, 6)
                radii = np.random.uniform(min_dim, max_dim, 6)
                xs = (cx + radii * np.cos(angles)).clip(0, w-1)
                ys = (cy + radii * np.sin(angles)).clip(0, h-1)
                pts = np.stack([xs, ys], 1).astype(np.int32)
            cv2.fillConvexPoly(m, pts, 255)
    return m


def _edge_points_from_mask(mask_u8: np.ndarray, band_px: int = 8):
    if band_px <= 0:
        return None
    m = (mask_u8 > 0).astype(np.uint8)
    if m.sum() == 0:
        return None
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    edge = (dist > 0) & (dist <= float(band_px))
    ys, xs = np.where(edge)
    if len(xs) == 0:
        return None
    return list(zip(xs.tolist(), ys.tolist()))


def _paste_rgba(src_rgba: np.ndarray, dst_rgb: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """Paste RGBA onto RGB with alpha blending. Returns alpha mask of pasted region."""
    Hs, Ws = src_rgba.shape[:2]
    Hd, Wd = dst_rgb.shape[:2]
    x1 = int(round(cx - Ws/2)); y1 = int(round(cy - Hs/2))
    x2, y2 = x1 + Ws, y1 + Hs

    sx1 = max(0, -x1); sy1 = max(0, -y1)
    dx1 = max(0, x1);  dy1 = max(0, y1)
    sx2 = Ws - max(0, x2 - Wd); sy2 = Hs - max(0, y2 - Hd)
    dx2 = min(Wd, x2); dy2 = min(Hd, y2)

    # Return empty mask if no overlap
    alpha_mask = np.zeros((Hd, Wd), dtype=np.uint8)
    if sx2 <= sx1 or sy2 <= sy1:
        return alpha_mask

    roi_src = src_rgba[sy1:sy2, sx1:sx2]
    roi_dst = dst_rgb[dy1:dy2, dx1:dx2]
    alpha = roi_src[..., 3:4].astype(np.float32) / 255.0
    color = roi_src[..., :3].astype(np.float32)
    dst_rgb[dy1:dy2, dx1:dx2] = (alpha * color + (1 - alpha) * roi_dst.astype(np.float32)).astype(np.uint8)

    # Return the alpha mask (where alpha > 0)
    alpha_mask[dy1:dy2, dx1:dx2] = (roi_src[..., 3] > 0).astype(np.uint8) * 255
    return alpha_mask


def occlude_with_bank(
    base_rgb: np.ndarray,
    target_mask: np.ndarray,
    bank_rgba: list,
    min_occ: float = 0.15,
    max_occ: float = 0.50,
    occ_count_min: int = 1,
    occ_count_max: int = 3,
    use_geom: bool = True,
    edge_bias: float = 0.0,
    edge_band_px: int = 8,
    edge_only: bool = False,
):
    H, W = target_mask.shape
    occ = np.zeros((H, W), np.uint8)
    img_occ = base_rgb.copy()
    target_frac = random.uniform(min_occ, max_occ)

    edge_pts = None
    if edge_bias > 0 or edge_only:
        edge_pts = _edge_points_from_mask(target_mask, band_px=edge_band_px)

    def _pick_center():
        if edge_only and edge_pts:
            return random.choice(edge_pts)
        if edge_pts and random.random() < edge_bias:
            return random.choice(edge_pts)
        return (random.randrange(0, W), random.randrange(0, H))

    n = random.randint(occ_count_min, occ_count_max)
    for _ in range(n):
        if not bank_rgba:
            break
        rgba = random.choice(bank_rgba)
        scale = 0.5 + 1.2 * random.random()
        rgba_res = cv2.resize(rgba, (int(rgba.shape[1]*scale), int(rgba.shape[0]*scale)), interpolation=cv2.INTER_LINEAR)
        cx, cy = _pick_center()
        # _paste_rgba now returns the alpha mask of pasted region
        alpha_mask = _paste_rgba(rgba_res, img_occ, cx, cy)
        occ = cv2.max(occ, alpha_mask)

    def _frac():
        inter = cv2.bitwise_and(occ, (target_mask > 0).astype(np.uint8) * 255)
        denom = max(1, int((target_mask > 0).sum()))
        return float((inter > 0).sum()) / float(denom)

    tries, f = 0, _frac()
    while f < target_frac and tries < 6:
        if use_geom:
            cx, cy = _pick_center()
            occ = cv2.bitwise_or(occ, _random_shape_mask(H, W, center=(cx, cy)))
        elif bank_rgba:
            rgba = random.choice(bank_rgba)
            scale = 0.6 + 1.4 * random.random()
            rgba_res = cv2.resize(rgba, (int(rgba.shape[1]*scale), int(rgba.shape[0]*scale)), interpolation=cv2.INTER_LINEAR)
            cx, cy = _pick_center()
            alpha_mask = _paste_rgba(rgba_res, img_occ, cx, cy)
            occ = cv2.max(occ, alpha_mask)
        f, tries = _frac(), tries + 1

    # visible mask
    visible = np.logical_and(target_mask > 0, occ == 0).astype(np.uint8)
    return img_occ, visible


def _build_mask_index(masks_dir: Path):
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    index = {}
    for p in Path(masks_dir).rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        stem = p.stem
        bases = {stem}
        for suf in ("_mask", "_m"):
            if stem.endswith(suf):
                bases.add(stem[: -len(suf)])
        if "_inst" in stem:
            bases.add(stem.split("_inst")[0])
        for base in bases:
            index.setdefault(base, []).append(p)
    return index


def _candidate_stems_from_image(stem: str):
    stems = {stem}
    if "_crop_" in stem:
        stems.add(stem.replace("_crop_", "_"))
    if stem.endswith("_crop"):
        stems.add(stem[: -len("_crop")])
    if "_crop" in stem:
        stems.add(stem.replace("_crop", ""))
    return stems


def _load_mask_union(mask_paths):
    if not mask_paths:
        return None
    union = None
    for mp in mask_paths:
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = (m > 0).astype(np.uint8)
        if union is None:
            union = m
        else:
            if union.shape != m.shape:
                m = cv2.resize(m, (union.shape[1], union.shape[0]), interpolation=cv2.INTER_NEAREST)
            union = np.maximum(union, m)
    return union


def _load_rgb_and_mask_for_image(img_path: Path, mask_index: dict):
    stems = _candidate_stems_from_image(img_path.stem)
    mask_paths = []
    for s in stems:
        if s in mask_index:
            mask_paths = mask_index.get(s, [])
            if mask_paths:
                break
    if not mask_paths:
        return None, None
    rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if rgb is None:
        return None, None
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    full_mask = _load_mask_union(mask_paths)
    if full_mask is None:
        return None, None
    if full_mask.shape[:2] != rgb.shape[:2]:
        full_mask = cv2.resize(full_mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    return rgb, full_mask


class CompletionDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        size: int = 512,
        occ_min: float = 0.05,
        occ_max: float = 0.20,
        occ_count_min: int = 1,
        occ_count_max: int = 1,
        edge_bias: float = 0.0,
        edge_band_px: int = 8,
        edge_only: bool = False,
        use_geom: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.size = int(size)
        self.mask_index = _build_mask_index(self.masks_dir)

        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        self.paths = [p for p in sorted(self.images_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]
        # keep only those with masks
        paired = []
        for p in self.paths:
            stems = _candidate_stems_from_image(p.stem)
            if any(s in self.mask_index for s in stems):
                paired.append(p)
        self.paths = paired
        if not self.paths:
            raise RuntimeError(f"No image/mask pairs found in {self.images_dir} + {self.masks_dir}")

        # build occluder bank
        self.bank = []
        for p in self.paths[: min(200, len(self.paths))]:
            rgb, m = _load_rgb_and_mask_for_image(p, self.mask_index)
            if rgb is None or m is None or m.sum() < 100:
                continue
            a = (m * 255).astype(np.uint8)
            self.bank.append(np.dstack([rgb, a]))

        self.occ_min = occ_min
        self.occ_max = occ_max
        self.occ_count_min = occ_count_min
        self.occ_count_max = occ_count_max
        self.edge_bias = edge_bias
        self.edge_band_px = edge_band_px
        self.edge_only = edge_only
        self.use_geom = use_geom

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rgb, full_mask = _load_rgb_and_mask_for_image(self.paths[idx], self.mask_index)
        if rgb is None or full_mask is None:
            raise RuntimeError(f"Missing mask for {self.paths[idx]}")

        # resize and pad to square (same as occlusion_augmentation)
        H, W = rgb.shape[:2]
        scale = self.size / max(H, W)
        newW, newH = int(round(W * scale)), int(round(H * scale))
        rgb = cv2.resize(rgb, (newW, newH), interpolation=cv2.INTER_AREA)
        full_mask = cv2.resize(full_mask, (newW, newH), interpolation=cv2.INTER_NEAREST)

        padL = (self.size - newW) // 2
        padT = (self.size - newH) // 2
        # Match inference letterbox padding (white background); using gray here makes the model
        # learn pad artifacts and can cause boundary speckle on real white backgrounds.
        rgb_sq = np.full((self.size, self.size, 3), 255, np.uint8)
        m_sq = np.zeros((self.size, self.size), np.uint8)
        rgb_sq[padT:padT + newH, padL:padL + newW] = rgb
        m_sq[padT:padT + newH, padL:padL + newW] = (full_mask > 0).astype(np.uint8)

        img_occ, vis_mask = occlude_with_bank(
            rgb_sq, m_sq, self.bank,
            min_occ=self.occ_min, max_occ=self.occ_max,
            occ_count_min=self.occ_count_min, occ_count_max=self.occ_count_max,
            use_geom=self.use_geom,
            edge_bias=self.edge_bias,
            edge_band_px=self.edge_band_px,
            edge_only=self.edge_only,
        )

        # input: occluded RGB + visible mask channel
        x = img_occ.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # (3,H,W)
        vm = vis_mask[None, ...].astype(np.float32)
        x = np.concatenate([x, vm], axis=0)  # (4,H,W)
        y = m_sq[None, ...].astype(np.float32)  # (1,H,W)
        return torch.from_numpy(x), torch.from_numpy(y)


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(2, 3)) + eps
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps
    return (1.0 - (num / den)).mean()


def edge_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    # Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=logits.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=logits.device).view(1, 1, 3, 3)
    gx = F.conv2d(probs, sobel_x, padding=1)
    gy = F.conv2d(probs, sobel_y, padding=1)
    ex = torch.sqrt(gx * gx + gy * gy + 1e-6)
    gx_t = F.conv2d(targets, sobel_x, padding=1)
    gy_t = F.conv2d(targets, sobel_y, padding=1)
    et = torch.sqrt(gx_t * gx_t + gy_t * gy_t + 1e-6)
    return F.l1_loss(ex, et)


def iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) >= thr).to(torch.uint8)
    t = (targets >= 0.5).to(torch.uint8)
    inter = (preds & t).sum(dim=(2, 3)).to(torch.float32)
    union = (preds | t).sum(dim=(2, 3)).to(torch.float32)
    iou = torch.where(union > 0, inter / union, torch.where(inter == 0, torch.ones_like(union), torch.zeros_like(union)))
    return float(iou.mean().item())


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dev = (args.device or "cpu").strip().lower()
    if dev == "mps" and not torch.backends.mps.is_available():
        _p("WARN: MPS requested but not available. Falling back to CPU.")
        dev = "cpu"
    if dev == "cuda" and not torch.cuda.is_available():
        _p("WARN: CUDA requested but not available. Falling back to CPU.")
        dev = "cpu"

    ds = CompletionDataset(
        Path(args.images), Path(args.masks),
        size=args.size,
        occ_min=args.occ_min,
        occ_max=args.occ_max,
        occ_count_min=args.occ_count_min,
        occ_count_max=args.occ_count_max,
        edge_bias=args.edge_bias,
        edge_band_px=args.edge_band,
        edge_only=args.edge_only,
        use_geom=args.use_geom,
    )
    dl = DataLoader(ds, batch_size=max(1, int(args.batch)), shuffle=True, num_workers=0, drop_last=True)

    model = build_completion_model(arch=args.arch, pretrained=args.pretrained).to(dev)
    if args.resume and Path(args.resume).exists():
        try:
            ckpt = torch.load(str(args.resume), map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"], strict=True)
                _p(f"Resumed weights from: {args.resume}")
        except Exception as e:
            _p(f"WARN: Failed to resume: {e}")

    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)

    _p(f"Samples: {len(ds)}")
    _p(f"Steps: {args.steps}, LR: {args.lr}, Size: {args.size}, Device: {dev}, Batch: {args.batch}")
    _p(f"Arch: {args.arch} (pretrained={args.pretrained}), Edge loss: {args.edge_w}")

    # Validation preview setup (optional)
    val_samples = []
    if args.val_images and args.val_masks:
        try:
            val_images = Path(args.val_images)
            val_masks = Path(args.val_masks)
            if val_images.is_dir() and val_masks.is_dir():
                mask_index = _build_mask_index(val_masks)
                exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
                cand = [p for p in sorted(val_images.iterdir()) if p.is_file() and p.suffix.lower() in exts]
                for p in cand:
                    rgb, m = _load_rgb_and_mask_for_image(p, mask_index)
                    if rgb is None or m is None:
                        continue
                    val_samples.append((rgb, m))
                    if len(val_samples) >= int(args.preview_count):
                        break
        except Exception as e:
            _p(f"WARN: Failed to load validation samples: {e}")

    it = iter(dl)
    ema = None
    for step in range(1, args.steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl)
            x, y = next(it)

        x = x.to(dev)
        y = y.to(dev)
        logits = model(x)
        loss = bce(logits, y) + dice_loss_with_logits(logits, y)
        if args.edge_w > 0:
            loss = loss + float(args.edge_w) * edge_loss_with_logits(logits, y)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if ema is None:
            ema = float(loss.detach().cpu())
        else:
            ema = 0.9 * ema + 0.1 * float(loss.detach().cpu())

        if step % 100 == 0 or step == 1:
            iou = iou_from_logits(logits.detach(), y.detach(), thr=0.5)
            _p(f"[{step}/{args.steps}] loss={ema:.4f} iou={iou:.4f}")

        if args.preview_every and val_samples and (step % int(args.preview_every) == 0 or step == 1):
            try:
                preview = _make_preview_image(
                    model,
                    val_samples,
                    ds.bank,
                    size=int(args.size),
                    device=dev,
                    occ_min=args.occ_min,
                    occ_max=args.occ_max,
                    occ_count_min=args.occ_count_min,
                    occ_count_max=args.occ_count_max,
                    edge_bias=args.edge_bias,
                    edge_band_px=args.edge_band,
                    edge_only=args.edge_only,
                    use_geom=args.use_geom,
                    thr=0.5,
                )
                if preview is not None:
                    out_dir = Path(args.preview_out) if args.preview_out else Path(os.getenv("TMPDIR", "/tmp")) / "leaf_completion_previews"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"preview_step_{step:06d}.png"
                    cv2.imwrite(str(out_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
                    _p(f"PREVIEW: {out_path}")
            except Exception as e:
                _p(f"WARN: Preview failed: {e}")

    save_completion_checkpoint(args.out, model, input_size=args.size, threshold=0.5,
                               arch=args.arch, pretrained=args.pretrained)
    _p(f"Done. Saved to {args.out}")


def main():
    ap = argparse.ArgumentParser("Train leaf completion model (no SAM)")
    ap.add_argument("--images", required=True, help="Folder with leaf images")
    ap.add_argument("--masks", required=True, help="Folder with leaf masks (same stem)")
    ap.add_argument("--out", required=True, help="Output .pth path")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--resume", default=None)
    ap.add_argument(
        "--arch",
        default="unet_resnet18",
        choices=[
            "unet_small",
            "unet_resnet18",
            "unet_resnet34",
            "unet_resnet50",
            "unet_efficientnet_b3",
            "unet_efficientnet_b4",
            "unetpp_resnet34",
            "unetpp_resnet50",
        ],
    )
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--edge-w", type=float, default=0.2)

    # occlusion settings
    ap.add_argument("--occ-min", type=float, default=0.05)
    ap.add_argument("--occ-max", type=float, default=0.20)
    ap.add_argument("--occ-count-min", type=int, default=1)
    ap.add_argument("--occ-count-max", type=int, default=1)
    ap.add_argument("--edge-bias", type=float, default=0.0)
    ap.add_argument("--edge-band", type=int, default=8)
    ap.add_argument("--edge-only", action="store_true")
    ap.add_argument("--use-geom", action="store_true", help="Use geometric blobs in addition to leaf occluders")
    ap.add_argument("--seed", type=int, default=0)
    # optional validation preview
    ap.add_argument("--val-images", default=None)
    ap.add_argument("--val-masks", default=None)
    ap.add_argument("--preview-every", type=int, default=0)
    ap.add_argument("--preview-count", type=int, default=3)
    ap.add_argument("--preview-out", default=None)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
