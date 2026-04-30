#!/usr/bin/env python3
"""
Train a tip-only semantic segmentation model (no SAM at inference).

Dataset format (root picked in the app "Target Segment" tab):
  images/ : RGB images
  masks/  : *_instXX.png are positive tip masks (binary)
           *_nomask.txt marks images as "no target tip" (negative)

For each image, the target mask is the union of all *_inst*.png masks.
If no inst masks exist:
  - include the image as a negative sample only if *_nomask.txt exists (and --allow-empty)

Example:
  python tip_segmenter_trainer.py \
    --images /path/to/images \
    --masks /path/to/masks \
    --out /path/to/tip_segmenter.pth \
    --steps 8000 --lr 1e-4 --size 512 --device mps --batch 2 --allow-empty
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tip_segmenter_model import build_tipseg_model, letterbox_rgb, letterbox_mask, save_tipseg_checkpoint


def _p(msg: str):
    # Print with flush so GUI subprocess log updates in real-time.
    print(msg, flush=True)


def _list_images(images_dir: Path):
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    return [p for p in sorted(images_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]


def _find_inst_masks(masks_dir: Path, stem: str):
    return sorted(masks_dir.glob(f"{stem}_inst*.png"))


class TipSegDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, size: int = 512, allow_empty: bool = True, augment: bool = True):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.size = int(size)
        self.allow_empty = bool(allow_empty)
        self.augment = bool(augment)

        self.images = _list_images(self.images_dir)
        if not self.images:
            raise RuntimeError(f"No images found in {self.images_dir}")

        self.items: list[tuple[Path, list[Path]]] = []
        self._pos_pix = 0
        self._tot_pix = 0

        for img_path in self.images:
            stem = img_path.stem
            inst = _find_inst_masks(self.masks_dir, stem)
            if inst:
                self.items.append((img_path, inst))
            else:
                marker = self.masks_dir / f"{stem}_nomask.txt"
                if self.allow_empty and marker.exists():
                    self.items.append((img_path, []))

        if not self.items:
            raise RuntimeError("No labeled samples found. Add *_inst*.png masks and/or *_nomask.txt markers.")

        # Estimate pos_weight from raw masks (quick and good enough)
        for img_path, inst in self.items:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            H, W = img.shape[:2]
            self._tot_pix += int(H * W)
            if not inst:
                continue
            m_union = np.zeros((H, W), dtype=np.uint8)
            for mp in inst:
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                if m.shape[:2] != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                m_union = np.maximum(m_union, (m > 127).astype(np.uint8))
            self._pos_pix += int(m_union.sum())

        pos_frac = float(self._pos_pix) / float(max(1, self._tot_pix))
        # If masks are extremely small, cap the weight so training stays stable.
        self.pos_weight = float(min(max(1.0, (1.0 - pos_frac) / max(1e-6, pos_frac)), 50.0))

    def __len__(self):
        return len(self.items)

    def _read_rgb(self, p: Path):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _read_union_mask(self, inst_paths: list[Path], shape_hw: tuple[int, int]):
        H, W = shape_hw
        if not inst_paths:
            return np.zeros((H, W), dtype=np.uint8)
        m_union = np.zeros((H, W), dtype=np.uint8)
        for mp in inst_paths:
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if m.shape[:2] != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            m_union = np.maximum(m_union, (m > 127).astype(np.uint8))
        return m_union

    def _aug(self, img: np.ndarray, mask: np.ndarray):
        # flips
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if random.random() < 0.2:
            img = np.ascontiguousarray(img[::-1, :])
            mask = np.ascontiguousarray(mask[::-1, :])

        # mild brightness/contrast jitter
        if random.random() < 0.3:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-10, 10)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return img, mask

    def __getitem__(self, idx):
        img_path, inst = self.items[idx]
        img = self._read_rgb(img_path)
        H, W = img.shape[:2]
        mask = self._read_union_mask(inst, (H, W))

        if self.augment:
            img, mask = self._aug(img, mask)

        sq, meta = letterbox_rgb(img, self.size, pad_value=255)
        mask_sq = letterbox_mask(mask, self.size, meta)

        x = torch.from_numpy(sq).to(torch.float32) / 255.0
        x = x.permute(2, 0, 1)  # (3,H,W)
        y = torch.from_numpy(mask_sq[None, ...]).to(torch.float32)  # (1,H,W) in {0,1}
        return x, y


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    # (B,1,H,W)
    num = 2.0 * (probs * targets).sum(dim=(2, 3)) + eps
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps
    return (1.0 - (num / den)).mean()


def iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) >= thr).to(torch.uint8)
    t = (targets >= 0.5).to(torch.uint8)
    inter = (preds & t).sum(dim=(2, 3)).to(torch.float32)
    union = (preds | t).sum(dim=(2, 3)).to(torch.float32)
    # If both empty => IoU=1, else IoU=0 when union is 0
    iou = torch.where(union > 0, inter / union, torch.where(inter == 0, torch.ones_like(union), torch.zeros_like(union)))
    return float(iou.mean().item())


def train(images_dir: str, masks_dir: str, out_path: str, steps: int = 8000, lr: float = 1e-4,
          size: int = 512, device: str = "cpu", batch_size: int = 2, allow_empty: bool = True, seed: int = 0,
          resume: str | None = None, arch: str = "unet_resnet18", pretrained: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dev = (device or "cpu").strip().lower()
    if dev == "mps" and not torch.backends.mps.is_available():
        _p("WARN: MPS requested but not available. Falling back to CPU.")
        dev = "cpu"
    if dev == "cuda" and not torch.cuda.is_available():
        _p("WARN: CUDA requested but not available. Falling back to CPU.")
        dev = "cpu"

    ds = TipSegDataset(Path(images_dir), Path(masks_dir), size=size, allow_empty=allow_empty, augment=True)
    dl = DataLoader(ds, batch_size=max(1, int(batch_size)), shuffle=True, num_workers=0, drop_last=True)

    model = build_tipseg_model(arch=arch, pretrained=pretrained).to(dev)
    if resume:
        rp = Path(resume)
        if rp.exists():
            try:
                try:
                    ckpt = torch.load(str(rp), map_location="cpu", weights_only=True)
                except TypeError:
                    ckpt = torch.load(str(rp), map_location="cpu")
                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    model.load_state_dict(ckpt["state_dict"], strict=True)
                    _p(f"Resumed weights from: {rp}")
            except Exception as e:
                _p(f"WARN: Failed to resume from {rp}: {e}")

    pos_w = torch.tensor([ds.pos_weight], dtype=torch.float32, device=dev)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)

    _p(f"Samples: {len(ds)}  (pos_weight≈{ds.pos_weight:.2f})")
    _p(f"Steps: {steps}, LR: {lr}, Size: {size}, Device: {dev}, Batch: {batch_size}, Allow empty: {allow_empty}")
    _p(f"Arch: {arch} (pretrained={pretrained})")

    it = iter(dl)
    ema = None
    for step in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl)
            x, y = next(it)

        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)

        logits = model(x)
        loss = bce(logits, y) + dice_loss_with_logits(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        v = float(loss.item())
        ema = v if ema is None else (0.98 * ema + 0.02 * v)

        if step % 100 == 0 or step == steps - 1:
            with torch.no_grad():
                iou = iou_from_logits(logits, y, thr=0.5)
            _p(f"[{step}/{steps}] loss={ema:.4f} iou={iou:.4f}")

    save_tipseg_checkpoint(out_path, model, input_size=size, threshold=0.5, arch=arch)
    _p(f"Done. Saved to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Train tip-only segmenter")
    ap.add_argument("--images", required=True, help="images/ dir")
    ap.add_argument("--masks", required=True, help="masks/ dir")
    ap.add_argument("--out", required=True, help="output .pth")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--allow-empty", action="store_true", help="Include *_nomask.txt as negative samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="", help="Optional existing .pth to resume weights from")
    ap.add_argument("--arch", default="unet_resnet18", choices=["unet_small", "unet_resnet18"])
    ap.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained encoder (if available)")
    args = ap.parse_args()

    train(
        images_dir=args.images,
        masks_dir=args.masks,
        out_path=args.out,
        steps=args.steps,
        lr=args.lr,
        size=args.size,
        device=args.device,
        batch_size=args.batch,
        allow_empty=bool(args.allow_empty),
        seed=args.seed,
        resume=(args.resume.strip() or None),
        arch=args.arch,
        pretrained=bool(args.pretrained),
    )
