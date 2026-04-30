#!/usr/bin/env python3
"""
Tip classifier training for target-only segmentation.

Trains a small CNN to classify SAM2 candidate masks as TIP vs NOT TIP.
Uses dataset folders:
  images/ : RGB images
  masks/  : *_instXX.png for positive masks, *_nomask.txt for no-target images

Example:
  python tip_classifier_trainer.py \
    --images /path/to/images \
    --masks /path/to/masks \
    --out /path/to/tip_classifier.pth \
    --steps 5000 --lr 1e-4 --device cpu
"""

import os
import argparse
import random
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from torchvision import models, transforms
except Exception as e:
    raise RuntimeError("torchvision is required for tip classifier training.") from e


def _list_images(images_dir: Path):
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    return [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in exts and p.is_file()]


def _find_mask_files(masks_dir: Path, stem: str):
    masks = sorted(masks_dir.glob(f"{stem}_inst*.png"))
    if not masks:
        masks = sorted(masks_dir.glob(f"{stem}*.png"))
        masks = [m for m in masks if m.stem != stem]
    return masks


def _mask_bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2


def _crop_with_pad(img: np.ndarray, mask: np.ndarray, pad_frac: float = 0.2):
    H, W = img.shape[:2]
    bbox = _mask_bbox(mask)
    if bbox is None:
        return None, None
    x1, y1, x2, y2 = bbox
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    pad = int(max(w, h) * pad_frac)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W - 1, x2 + pad)
    y2 = min(H - 1, y2 + pad)
    crop = img[y1:y2 + 1, x1:x2 + 1].copy()
    mask_crop = mask[y1:y2 + 1, x1:x2 + 1].copy()
    return crop, mask_crop


def _iou_box(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
    inter = iw * ih
    area_a = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    area_b = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


class TipDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, input_size: int = 224,
                 neg_per_pos: int = 1, neg_per_empty: int = 3):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.input_size = input_size
        self.neg_per_pos = max(0, int(neg_per_pos))
        self.neg_per_empty = max(0, int(neg_per_empty))

        self.images = _list_images(images_dir)
        if not self.images:
            raise RuntimeError(f"No images found in {images_dir}")

        self.pos_items = []
        self.img_to_pos_boxes = {}
        for img_path in self.images:
            stem = img_path.stem
            mask_files = _find_mask_files(masks_dir, stem)
            boxes = []
            for mp in mask_files:
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                m = (m > 127).astype(np.uint8)
                bb = _mask_bbox(m)
                if bb is None:
                    continue
                boxes.append(bb)
                self.pos_items.append((img_path, mp, 1))
            if boxes:
                self.img_to_pos_boxes[img_path] = boxes

        # estimate target size from positives
        hs, ws = [], []
        for img_path, mp, _ in self.pos_items:
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            m = (m > 127).astype(np.uint8)
            bb = _mask_bbox(m)
            if bb is None:
                continue
            x1, y1, x2, y2 = bb
            ws.append(x2 - x1 + 1)
            hs.append(y2 - y1 + 1)
        self.med_w = int(np.median(ws)) if ws else 128
        self.med_h = int(np.median(hs)) if hs else 128

        # build negative items (image-only; crop sampled in __getitem__)
        self.neg_items = []
        for img_path in self.images:
            stem = img_path.stem
            no_marker = masks_dir / f"{stem}_nomask.txt"
            if no_marker.exists():
                for _ in range(self.neg_per_empty):
                    self.neg_items.append((img_path, None, 0))
            else:
                # add some negatives from positive images too
                if img_path in self.img_to_pos_boxes and self.neg_per_pos > 0:
                    for _ in range(self.neg_per_pos):
                        self.neg_items.append((img_path, None, 0))

        # balance
        self.items = self.pos_items + self.neg_items
        if not self.items:
            raise RuntimeError("No training samples found (check masks/ and _nomask.txt files).")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.items)

    def _sample_neg_crop(self, img: np.ndarray, img_path: Path):
        H, W = img.shape[:2]
        w = int(self.med_w * random.uniform(0.7, 1.3))
        h = int(self.med_h * random.uniform(0.7, 1.3))
        w = max(16, min(w, W))
        h = max(16, min(h, H))
        boxes = self.img_to_pos_boxes.get(img_path, [])

        for _ in range(10):
            x1 = random.randint(0, max(0, W - w))
            y1 = random.randint(0, max(0, H - h))
            x2, y2 = x1 + w - 1, y1 + h - 1
            if boxes:
                if max((_iou_box((x1, y1, x2, y2), b) for b in boxes), default=0.0) > 0.05:
                    continue
            return img[y1:y2 + 1, x1:x2 + 1].copy()
        # fallback
        return img[0:h, 0:w].copy()

    def __getitem__(self, idx):
        img_path, mask_path, label = self.items[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if label == 1 and mask_path is not None:
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise RuntimeError(f"Failed to read mask: {mask_path}")
            m = (m > 127).astype(np.uint8)
            crop, mask_crop = _crop_with_pad(img, m, pad_frac=0.2)
            if crop is None:
                raise RuntimeError("Empty mask crop.")
            # mask the crop (black background)
            mask3 = (mask_crop[..., None] > 0)
            crop = np.where(mask3, crop, 0)
        else:
            crop = self._sample_neg_crop(img, img_path)

        x = self.transform(crop)
        y = torch.tensor([float(label)], dtype=torch.float32)
        return x, y


def build_model(arch: str = "resnet18"):
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model


def train(images_dir, masks_dir, out_path, steps=5000, lr=1e-4, device="cpu",
          batch_size=16, input_size=224, neg_per_pos=1, neg_per_empty=3, arch="resnet18"):
    device = (device or "cpu").strip().lower()
    ds = TipDataset(Path(images_dir), Path(masks_dir), input_size=input_size,
                    neg_per_pos=neg_per_pos, neg_per_empty=neg_per_empty)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = build_model(arch=arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()

    print(f"Dataset samples: {len(ds)}  (pos={len(ds.pos_items)}, neg={len(ds.neg_items)})")
    print(f"Training steps: {steps}, batch={batch_size}, device={device}, arch={arch}")

    it = iter(dl)
    running = 0.0
    for step in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl)
            x, y = next(it)
        x = x.to(device)
        y = y.to(device)

        logits = model(x).squeeze(1)
        loss = crit(logits, y.squeeze(1))

        opt.zero_grad()
        loss.backward()
        opt.step()

        running = running * 0.98 + 0.02 * float(loss.item())
        if step % 200 == 0:
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                acc = (preds == y.squeeze(1)).float().mean().item()
            print(f"[{step}/{steps}] loss={running:.4f} acc={acc:.3f}")

    save_dict = {
        "state_dict": model.state_dict(),
        "arch": arch,
        "input_size": input_size,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    torch.save(save_dict, out_path)
    print(f"Saved classifier to: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Train tip classifier")
    ap.add_argument("--images", required=True, help="Images dir")
    ap.add_argument("--masks", required=True, help="Masks dir")
    ap.add_argument("--out", required=True, help="Output .pth")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--neg-per-pos", type=int, default=1)
    ap.add_argument("--neg-per-empty", type=int, default=3)
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "mobilenet_v3_small"])
    args = ap.parse_args()

    train(
        images_dir=args.images,
        masks_dir=args.masks,
        out_path=args.out,
        steps=args.steps,
        lr=args.lr,
        device=args.device,
        batch_size=args.batch,
        input_size=args.input_size,
        neg_per_pos=args.neg_per_pos,
        neg_per_empty=args.neg_per_empty,
        arch=args.arch,
    )
