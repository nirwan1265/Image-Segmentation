#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Leaf Segmenter – interactive GUI for enhancement + SAM2 auto-masking + selective saving.

Run:  python plant_segmenter_fixed.py

Requires: numpy, Pillow, opencv-python, torch, hydra, omegaconf, and your local
SAM2 repo installed (or PYTHONPATH to it) so that build_sam2 / generators import.
"""

import os
import csv
import threading
from pathlib import Path
from dataclasses import dataclass
import torch

import numpy as np
import cv2
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import json, time, shutil
import subprocess, shlex, sys


# Hydra/OmegaConf
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

# --- SAM2 imports (repo must be importable) ---
try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
except Exception as e:  # keep friendly error for GUI
    SAM2AutomaticMaskGenerator = None
    build_sam2 = None
    _sam2_import_error = e
else:
    _sam2_import_error = None

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception:
    SAM2ImagePredictor = None

from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# =========================
# Image helpers (RGB uint8)
# =========================
def _hydra_reinit_to_dir(cfg_dir: str):
    try:
        GlobalHydra.instance().clear()
    except Exception:
        pass
    initialize_config_dir(config_dir=cfg_dir, job_name="sam2_gui")

import traceback
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

def _show_info(msg, title="Info"):
    try:
        messagebox.showinfo(title, str(msg) if msg else "(no details)")
    except Exception:
        print(f"[INFO:{title}] {msg}")

def _show_err(where, exc):
    tb = traceback.format_exc()
    print(f"\n[ERROR:{where}] {exc}\n{tb}\n")
    try:
        messagebox.showerror(f"Error: {where}", f"{exc}\n\nSee terminal for details.")
    except Exception:
        pass


# ---------- phenotyping helpers ----------
import math, re

def _color_stats(rgb, mask_bool):
    R = rgb[..., 0][mask_bool].astype(np.float32)
    G = rgb[..., 1][mask_bool].astype(np.float32)
    B = rgb[..., 2][mask_bool].astype(np.float32)

    def stats(ch):
        if ch.size == 0:
            return dict(mean=0.0, median=0.0, sum=0.0, std=0.0)
        return dict(mean=float(ch.mean()), median=float(np.median(ch)),
                    sum=float(ch.sum()), std=float(ch.std()))
    return stats(R), stats(G), stats(B)

def _color_stats_hsv(rgb, mask_bool):
    """
    Compute H/S/V channel stats over the masked region.
    OpenCV HSV: H∈[0,179], S,V∈[0,255] for uint8.
    """
    if rgb.dtype != np.uint8:
        rgb8 = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        rgb8 = rgb
    hsv = cv2.cvtColor(rgb8, cv2.COLOR_RGB2HSV)

    H = hsv[..., 0][mask_bool].astype(np.float32)
    S = hsv[..., 1][mask_bool].astype(np.float32)
    V = hsv[..., 2][mask_bool].astype(np.float32)

    def stats(ch):
        if ch.size == 0:
            return dict(mean=0.0, median=0.0, sum=0.0, std=0.0)
        return dict(mean=float(ch.mean()),
                    median=float(np.median(ch)),
                    sum=float(ch.sum()),
                    std=float(ch.std()))
    return stats(H), stats(S), stats(V)


def _pca_angle_deg(mask_bool):
    ys, xs = np.nonzero(mask_bool)
    if xs.size < 2:
        return 0.0
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    pts -= pts.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(pts, full_matrices=False)
    vx, vy = Vt[0, 0], Vt[0, 1]
    return math.degrees(math.atan2(vy, vx))

def _rotate_mask(mask_bool, angle_deg):
    h, w = mask_bool.shape
    M = cv2.getRotationMatrix2D((w/2.0, h/2.0), angle_deg, 1.0)
    m_u8 = (mask_bool.astype(np.uint8) * 255)
    m_rot = cv2.warpAffine(m_u8, M, (w, h), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (m_rot > 0)

def _length_width_after_deskew(mask_bool):
    """Rotate so major axis is vertical; length from vertical span; width from row spans."""
    ang = _pca_angle_deg(mask_bool)
    m_rot = _rotate_mask(mask_bool, -ang)

    h, w = m_rot.shape
    rows_present = np.any(m_rot, axis=1)
    y_idx = np.where(rows_present)[0]
    if y_idx.size == 0:
        return dict(angle_deg=-ang, length_px=0.0, width_px_max=0.0, width_px_p95=0.0)

    y_top, y_bot = int(y_idx.min()), int(y_idx.max())
    length_px = float(y_bot - y_top + 1)

    widths = []
    for y in range(y_top, y_bot + 1):
        xs = np.where(m_rot[y])[0]
        if xs.size:
            widths.append(xs.max() - xs.min() + 1)
    widths = np.array(widths, dtype=np.float32) if len(widths) else np.array([0.0], dtype=np.float32)
    return dict(
        angle_deg=-ang,
        length_px=float(length_px),
        width_px_max=float(widths.max()),
        width_px_p95=float(np.percentile(widths, 95))
    )

def _pca_major_minor(mask_bool):
    ys, xs = np.nonzero(mask_bool)
    if xs.size < 2:
        axis_w  = int(xs.max()-xs.min()+1 if xs.size else 0)
        axis_h  = int(ys.max()-ys.min()+1 if ys.size else 0)
        return 0.0, 0.0, axis_w, axis_h
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mu = pts.mean(axis=0, keepdims=True)
    X = pts - mu
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    proj = X @ V
    length_major = proj[:, 0].max() - proj[:, 0].min()
    width_minor  = proj[:, 1].max()  - proj[:, 1].min()
    axis_w  = xs.max() - xs.min() + 1
    axis_h  = ys.max() - ys.min() + 1
    return float(length_major), float(width_minor), int(axis_w), int(axis_h)


# ====== Predict/Extend helpers (non-ML shape completion) ======
def _mask_to_contour_pts(mask_bool: np.ndarray):
    m8 = (mask_bool.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    pts = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.int32)
    return np.ascontiguousarray(pts)

def _rosette_circle_extend(mask_bool: np.ndarray, strength=1.0):
    """
    Arabidopsis-style completion: fit a circle and grow it a bit.
    strength ~ 0.7..1.5  (1.0 is a mild, safe grow)
    """
    pts = _mask_to_contour_pts(mask_bool)
    if pts is None or pts.shape[0] < 3:
        return mask_bool
    (cx, cy), r = cv2.minEnclosingCircle(pts.astype(np.float32))
    # increase radius gently
    r2 = float(r) * (1.10 + 0.20 * (float(strength) - 1.0))  # ~+10% at strength=1
    H, W = mask_bool.shape
    out = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(r2)), 1, thickness=-1)
    return np.logical_or(mask_bool, out.astype(bool))

def _convex_hull_fill(mask_bool: np.ndarray):
    pts = _mask_to_contour_pts(mask_bool)
    if pts is None or pts.shape[0] < 3:
        # nothing to hull or too few points
        return mask_bool

    # Hull as Nx2 int32, contiguous
    hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.int32)
    hull = np.ascontiguousarray(hull)

    # Destination image must be C-contiguous uint8
    H, W = mask_bool.shape[:2]
    hull_mask = np.zeros((H, W), dtype=np.uint8)
    hull_mask = np.ascontiguousarray(hull_mask)

    # Fill
    cv2.fillConvexPoly(hull_mask, hull, 1)

    return hull_mask.astype(bool)



def _largest_component_bool(mask_bool: np.ndarray):
    """Return largest connected component (4-conn) as bool, or None."""
    m = (mask_bool.astype(np.uint8) > 0).astype(np.uint8)
    num, lab = cv2.connectedComponents(m, connectivity=4)
    if num <= 1:
        return None
    # bincount over labels (skip background 0)
    counts = np.bincount(lab.ravel())
    lbl = int(np.argmax(counts[1:]) + 1)
    comp = (lab == lbl)
    return comp

def _rosette_hull_wedge_extend(mask_bool: np.ndarray, strength=1.0):
    """
    Arabidopsis-style: fill only the largest convexity 'wedge' (hull - mask).
    strength: 0.7..1.5 -> controls slight dilation of the wedge.
    """
    # 1) convex hull of visible mask
    hull_bool = _convex_hull_fill(mask_bool)
    # 2) candidate wedges = hull minus mask
    added = np.logical_and(hull_bool, ~mask_bool)
    if not added.any():
        return mask_bool  # nothing to add
    # 3) keep largest wedge only
    wedge = _largest_component_bool(added)
    if wedge is None or not wedge.any():
        return mask_bool
    # 4) optional gentle dilation of the wedge
    if strength > 1.01:
        k = max(1, int(round(2 * (strength - 1.0))))  # small kernel
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
        wedge = cv2.morphologyEx(wedge.astype(np.uint8), cv2.MORPH_DILATE, se).astype(bool)
    # 5) final
    return np.logical_or(mask_bool, wedge)

def _rosette_ellipse_scale_extend(mask_bool: np.ndarray, scale=1.12):
    pts = _mask_to_contour_pts(mask_bool)
    if pts is None or pts.shape[0] < 5:
        return mask_bool
    try:
        (cx, cy), (MA, ma), ang = cv2.fitEllipse(pts.astype(np.float32))
    except cv2.error:
        return mask_bool
    MA2, ma2 = max(3, MA*scale), max(3, ma*scale)
    H, W = mask_bool.shape
    out = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(out, (int(round(cx)), int(round(cy))), (int(round(MA2/2)), int(round(ma2/2))),
                ang, 0, 360, 1, thickness=-1)
    return np.logical_or(mask_bool, out.astype(bool))




def _pca_orientation_full(mask_bool: np.ndarray):
    ys, xs = np.nonzero(mask_bool)
    if xs.size < 10:
        return None
    X = np.column_stack([xs, ys]).astype(np.float32)
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(1, len(Xc)-1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    vmaj = eigvecs[:, order[0]]
    vmaj = vmaj / (np.linalg.norm(vmaj) + 1e-8)
    vperp = np.array([-vmaj[1], vmaj[0]], dtype=np.float32)
    proj = (Xc @ vmaj)
    length = float(proj.max() - proj.min())
    width  = float((Xc @ vperp).max() - (Xc @ vperp).min())
    return mu, vmaj, vperp, length, width, X, proj

def _tapered_extension(mask_bool: np.ndarray, k_extend=0.6):
    """Extend along major axis by k_extend * current length with a triangular taper."""
    H, W = mask_bool.shape
    info = _pca_orientation_full(mask_bool)
    if info is None:
        return mask_bool
    mu, vmaj, vperp, length, width, X, proj = info

    # tip direction = side with larger mean projection
    tip_is_plus = proj.mean() > 0
    direction = vmaj if tip_is_plus else -vmaj

    extend_len = max(8.0, k_extend * max(20.0, length))
    front_thr  = np.percentile(proj, 85)
    proj_vals  = proj
    front_max  = proj_vals.max() if tip_is_plus else -proj_vals.min()
    base_center = mu + direction * (front_max)

    base_width = max(4.0, 0.5 * width)
    half_base  = 0.5 * base_width

    tip = mu + direction * (front_max + extend_len)
    p1  = base_center + vperp * half_base
    p2  = base_center - vperp * half_base

    poly = np.stack([p1, p2, tip]).astype(np.int32)
    ext = np.zeros_like(mask_bool, dtype=np.uint8)
    cv2.fillConvexPoly(ext, poly, 1)
    ext = cv2.morphologyEx(ext, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
    return (mask_bool | (ext > 0))



def predict_extend_mask(mask_bool: np.ndarray, method: str = "auto", strength: float = 1.0,
                        forbid_mask: np.ndarray | None = None):
    """
    Non-ML mask completion. Returns a new bool mask (same shape).
    method: 'auto' | 'rosette' | 'blade'
    """
    if mask_bool is None:
        return None
    base = mask_bool.astype(bool)
    mode = (method or "auto").lower().strip()

    # choose mode based on shape if auto
    if mode == "auto":
        maj, minw, *_ = _pca_major_minor(base)
        ratio = (maj / (minw + 1e-6)) if minw >= 0 else 0.0
        mode = "blade" if ratio >= 2.2 else "rosette"

    if mode == "blade":
        pred = _tapered_extension(base, k_extend=0.6 * float(strength))
    else:
        # rosette-style: hull wedge first; fallback to circle/ellipse growth
        pred = _rosette_hull_wedge_extend(base, strength=float(strength))
        if pred is None or np.array_equal(pred, base):
            pred = _rosette_circle_extend(base, strength=float(strength))

    pred = pred.astype(bool)
    if forbid_mask is not None:
        pred = np.logical_and(pred, ~forbid_mask.astype(bool))
        pred = np.logical_or(base, pred)
    return pred




from pathlib import Path
from omegaconf import OmegaConf
try:
    from omegaconf import DictConfig
except Exception:
    class DictConfig:  # fallback
        pass

from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

def _compose_from_yaml(yaml_path: str):
    conf_dir = str(Path(yaml_path).parent)
    conf_name = Path(yaml_path).stem
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=conf_dir, job_name="sam2_gui", version_base=None):
        return compose(config_name=conf_name)

def _resolve_sam2_cfg(cfg_field, ckpt_path: str | None = None, fallback_short="sam2.1_hiera_l"):
    if cfg_field is None:
        return fallback_short
    if isinstance(cfg_field, DictConfig):
        return cfg_field
    if isinstance(cfg_field, dict):
        return OmegaConf.create(cfg_field)

    s = str(cfg_field).strip()
    if not s:
        return fallback_short

    p = Path(s)
    if p.is_file() and s.lower().endswith((".yaml", ".yml")):
        return _compose_from_yaml(s)

    if p.is_dir():
        guess = p / "sam2.1_hiera_l.yaml"
        if guess.exists():
            return _compose_from_yaml(str(guess))
        for y in sorted(p.glob("*.y*ml")):
            if "hiera" in y.stem:
                return _compose_from_yaml(str(y))
        cands = sorted(p.glob("*.y*ml"))
        if cands:
            return _compose_from_yaml(str(cands[0]))

    guesses = []
    env_dir = os.environ.get("SAM2_CONFIG_DIR")
    if env_dir:
        guesses += [str(Path(env_dir) / "sam2.1"), env_dir]
    if ckpt_path:
        ck = Path(ckpt_path)
        repo_root = ck.parent.parent if ck.suffix == ".pt" else ck.parent
        guesses += [str(repo_root / "configs" / "sam2.1"), str(repo_root / "configs")]
    for d in guesses:
        y = Path(d) / f"{s}.yaml"
        if y.exists():
            return _compose_from_yaml(str(y))

    return s  # let build_sam2 resolve a package-shortname





def ensure_uint8_rgb(arr):
    arr = np.array(arr)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)

def rotate_left_90(img_rgb_uint8: np.ndarray) -> np.ndarray:
    return cv2.rotate(img_rgb_uint8, cv2.ROTATE_90_COUNTERCLOCKWISE)

def save_binary_mask(mask_bool, out_path):
    out = (mask_bool.astype(np.uint8) * 255)
    cv2.imwrite(str(out_path), out)

def save_masked_crop_rgba(image_rgb_uint8, mask_bool, bbox, out_path, erode_px=0, feather_px=0):
    """Save transparent crop (halo-less if you use erode/feather)."""
    x, y, w, h = map(int, bbox)
    x2, y2 = x + w, y + h
    x, y = max(0, x), max(0, y)

    crop_img = image_rgb_uint8[y:y2, x:x2, :]
    crop_msk = mask_bool[y:y2, x:x2].astype(np.uint8)

    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, 2*erode_px+1),)*2)
        crop_msk = cv2.erode(crop_msk, k, iterations=1)

    if feather_px > 0:
        dist = cv2.distanceTransform((crop_msk > 0).astype(np.uint8), cv2.DIST_L2, 3)
        alpha = np.clip(dist / float(feather_px), 0, 1) * 255.0
        alpha = alpha.astype(np.uint8)
    else:
        alpha = crop_msk * 255

    rgba = np.dstack([crop_img, alpha])
    cv2.imwrite(str(out_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

def mask_iou(a_bool, b_bool):
    inter = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum() + 1e-6
    return inter / union

def dedupe_by_mask_iou(masks, iou_thresh=0.80):
    kept = []
    for m in sorted(masks, key=lambda z: z["area"], reverse=True):
        seg = m["segmentation"].astype(bool)
        if any(mask_iou(seg, k["segmentation"].astype(bool)) > iou_thresh for k in kept):
            continue
        kept.append(m)
    return kept

def split_masks_by_cc(masks, min_area=50, max_components=None):
    """
    Split masks that contain multiple disconnected components into separate masks.
    Returns a new list (original multi-blob mask is replaced by its components).
    """
    out = []
    for m in masks:
        seg = m.get("segmentation")
        if not isinstance(seg, np.ndarray):
            out.append(m)
            continue
        seg_u8 = (seg > 0).astype(np.uint8)
        if seg_u8.ndim != 2:
            # unexpected shape; keep original
            out.append(m)
            continue

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seg_u8, connectivity=8)
        if num_labels <= 2:
            out.append(m)
            continue

        comps = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < int(min_area):
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            comp_seg = (labels == label).astype(np.uint8)
            comp = dict(m)
            comp["segmentation"] = comp_seg
            comp["bbox"] = [x, y, w, h]
            comp["area"] = float(area)
            meta = dict(m.get("meta", {}))
            meta["split"] = True
            meta["split_components"] = int(num_labels - 1)
            comp["meta"] = meta
            comps.append(comp)

        if len(comps) <= 1:
            out.append(m)
        else:
            if max_components is not None and len(comps) > int(max_components):
                comps = sorted(comps, key=lambda z: z.get("area", 0), reverse=True)[: int(max_components)]
            out.extend(comps)
    return out



# =========================
#  Canvas/Image coordinate helpers 
# =========================
def _canvas_geometry(self):
    """Return (cx, cy, new_w, new_h, W, H) for current drawn image box."""
    if self._img_for_preview is None:
        return None
    H, W = self._img_for_preview.shape[:2]
    cw = int(self.canvas.winfo_width())
    ch = int(self.canvas.winfo_height())
    fit = min(cw / max(1, W), ch / max(1, H))
    scale = fit if getattr(self, "_fit_mode", True) else fit * getattr(self, "_zoom", 1.0)
    new_w = max(1, int(W * scale))
    new_h = max(1, int(H * scale))
    cx, cy = cw // 2 + self._pan[0], ch // 2 + self._pan[1]
    return cx, cy, new_w, new_h, W, H

def _canvas_to_image_xy(self, x_canvas, y_canvas):
    g = self._canvas_geometry()
    if g is None:
        return None
    cx, cy, new_w, new_h, W, H = g
    left  = cx - new_w // 2
    top   = cy - new_h // 2
    # map into the scaled image rect, then back to original pixels
    x_rel = (x_canvas - left)
    y_rel = (y_canvas - top)
    if x_rel < 0 or y_rel < 0 or x_rel > new_w or y_rel > new_h:
        return None
    x_img = int(round(x_rel * (W / new_w)))
    y_img = int(round(y_rel * (H / new_h)))
    x_img = max(0, min(W - 1, x_img))
    y_img = max(0, min(H - 1, y_img))
    return x_img, y_img

def _image_to_canvas_xy(self, x_img, y_img):
    g = self._canvas_geometry()
    if g is None:
        return None
    cx, cy, new_w, new_h, W, H = g
    left  = cx - new_w // 2
    top   = cy - new_h // 2
    x_rel = x_img * (new_w / W)
    y_rel = y_img * (new_h / H)
    return int(left + x_rel), int(top + y_rel)


def _clear_crop_overlay(self):
    self._crop_rect_img = None
    if self._crop_canvas_id:
        try: self.canvas.delete(self._crop_canvas_id)
        except Exception: pass
    self._crop_canvas_id = None

def _update_crop_buttons(self):
    enabled = (self._crop_mode.get() and self._crop_rect_img is not None)
    self._btn_crop_apply.configure(state="normal" if enabled else "disabled")
    self._btn_crop_cancel.configure(state="normal" if self._crop_mode.get() else "disabled")


# =========================
# Crop interactions 
# =========================
def _crop_start(self, event):
    if self._img_for_preview is None:
        return
    self._crop_start_canvas = (event.x, event.y)
    # initialize selection at a single point
    p = self._canvas_to_image_xy(event.x, event.y)
    if p:
        self._crop_rect_img = (*p, *p)
        self._draw_crop_overlay()
        self._update_crop_buttons()

def _crop_drag(self, event):
    if self._img_for_preview is None or self._crop_start_canvas is None:
        return
    p0 = self._canvas_to_image_xy(*self._crop_start_canvas)
    p1 = self._canvas_to_image_xy(event.x, event.y)
    if not p0 or not p1:
        return
    x0, y0 = p0; x1, y1 = p1
    x1, x2 = sorted((x0, x1)); y1, y2 = sorted((y0, y1))
    # avoid zero-size rects
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    self._crop_rect_img = (x1, y1, x2, y2)
    self._draw_crop_overlay()
    self._update_crop_buttons()

def _crop_end(self, event):
    # nothing extra; selection already stored during drag
    pass

def _apply_crop(self):
    """Commit crop to the working image. This *commits your current rotation*
    by replacing img_orig with the cropped, rotated view and resetting angle to 0°."""
    if not self._crop_rect_img or self._img_for_preview is None:
        return
    x1, y1, x2, y2 = self._crop_rect_img
    # crop the *rotated* base so shapes stay consistent with what user saw
    base = self._base_image()
    if base is None:
        return
    x1 = max(0, min(base.shape[1]-1, x1))
    x2 = max(0, min(base.shape[1],   x2))
    y1 = max(0, min(base.shape[0]-1, y1))
    y2 = max(0, min(base.shape[0],   y2))
    if x2 <= x1+1 or y2 <= y1+1:
        return

    cropped = base[y1:y2, x1:x2, :].copy()

    # commit: new "original"; reset rotation; clear masks/preview
    self.img_orig = cropped
    self.rot_angle.set(0.0)
    self._draw_knob()
    self.sr = None
    self.lb.delete(0, tk.END)

    # reset view and leave crop mode
    self._clear_crop_overlay()
    self._zoom_fit()
    self._crop_mode.set(False)
    self._bind_canvas_events()
    self._update_crop_buttons()

    # show it
    self.img = self._base_image()
    self.img_preview = None
    self.show_image(self.img)

def _cancel_crop(self):
    self._clear_crop_overlay()
    self._update_crop_buttons()
    # keep crop mode toggled on/off as chosen; just clear selection


# =========================
# Enhancers
# =========================

def preprocess_for_edges(
    img_rgb_uint8,
    brightness=0, contrast=1.0,
    use_unsharp=True, unsharp_kernel_size=9, unsharp_sigma=10.0, unsharp_amount=1.5,
    use_laplacian=False,
    gamma=None
):
    x = img_rgb_uint8.astype(np.uint8)

    if brightness != 0 or contrast != 1.0:
        x = cv2.addWeighted(x, contrast, np.zeros_like(x), 0, brightness)

    if use_unsharp:
        k = (int(unsharp_kernel_size), int(unsharp_kernel_size))
        blur = cv2.GaussianBlur(x, k, unsharp_sigma)
        x = cv2.addWeighted(x, unsharp_amount, blur, -(unsharp_amount - 1.0), 0)

    if use_laplacian:
        lap = cv2.Laplacian(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.CV_64F)
        lap = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        x = np.dstack([lap, lap, lap])

    if gamma is not None and gamma > 0:
        tab = np.clip((np.arange(256) / 255.0) ** (1.0 / gamma) * 255.0, 0, 255).astype(np.uint8)
        x = cv2.LUT(x, tab)

    return np.ascontiguousarray(x)

def enhance_leaf_edges_rgb(
    img_rgb_uint8,
    hsv_h_low=25, hsv_h_high=95, hsv_s_min=40, hsv_v_min=40,
    clahe_clip=2.0, clahe_tiles=8,
    bilateral_d=7, bilateral_sigma=50,
    unsharp_amount=1.5, unsharp_sigma=10, unsharp_ksize=9,
    sobel_blend=0.12
):
    x = img_rgb_uint8.astype(np.uint8)

    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    green = (h >= hsv_h_low) & (h <= hsv_h_high) & (s >= hsv_s_min) & (v >= hsv_v_min)
    green = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1).astype(bool)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tiles, clahe_tiles))
    v_clahe = clahe.apply(v)
    v_eq = v.copy()
    if green.any():
        v_eq[green] = v_clahe[green]
    else:
        v_eq = v_clahe

    hsv2 = hsv.copy(); hsv2[..., 2] = v_eq
    rgb_eq = cv2.cvtColor(hsv2, cv2.COLOR_HSV2RGB)

    rgb_bi = cv2.bilateralFilter(rgb_eq, d=int(bilateral_d), sigmaColor=bilateral_sigma, sigmaSpace=bilateral_sigma)

    gray = cv2.cvtColor(rgb_bi, cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel = cv2.normalize(cv2.magnitude(sx, sy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    hsv3 = cv2.cvtColor(rgb_bi, cv2.COLOR_RGB2HSV)
    hsv3[..., 2] = np.clip(hsv3[..., 2].astype(np.float32) + sobel_blend * sobel, 0, 255).astype(np.uint8)
    rgb_edge = cv2.cvtColor(hsv3, cv2.COLOR_HSV2RGB)

    blur = cv2.GaussianBlur(rgb_edge, (int(unsharp_ksize), int(unsharp_ksize)), unsharp_sigma)
    sharp = cv2.addWeighted(rgb_edge, unsharp_amount, blur, -(unsharp_amount - 1.0), 0)

    return np.ascontiguousarray(sharp)

def flatten_background_whiten(img_rgb_uint8, val_min=200, sat_max=35, morph_open=3, morph_close=5):
    hsv = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bg = (v >= val_min) & (s <= sat_max)
    if morph_open > 0:
        k = np.ones((morph_open, morph_open), np.uint8)
        bg = cv2.morphologyEx(bg.astype(np.uint8), cv2.MORPH_OPEN, k, iterations=1).astype(bool)
    if morph_close > 0:
        k = np.ones((morph_close, morph_close), np.uint8)
        bg = cv2.morphologyEx(bg.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=1).astype(bool)

    out = img_rgb_uint8.copy()
    out[bg] = 255
    return out


# =========================
# Advanced Enhancement Functions
# =========================

def compute_vegetation_indices(rgb):
    """
    Compute plant-specific vegetation indices from RGB.
    Returns dict with ExG, GRVI, VARI, TGI, GLI as normalized uint8 images.
    """
    R = rgb[..., 0].astype(np.float32)
    G = rgb[..., 1].astype(np.float32)
    B = rgb[..., 2].astype(np.float32)

    # Normalize to 0-1 range
    r = R / 255.0
    g = G / 255.0
    b = B / 255.0

    # Excess Green Index (ExG) - highlights green vegetation
    ExG = 2*g - r - b

    # Green-Red Vegetation Index (GRVI)
    GRVI = (g - r) / (g + r + 1e-6)

    # Visible Atmospherically Resistant Index (VARI)
    VARI = (g - r) / (g + r - b + 1e-6)

    # Triangular Greenness Index (TGI)
    TGI = g - 0.39*r - 0.61*b

    # Green Leaf Index (GLI)
    GLI = (2*g - r - b) / (2*g + r + b + 1e-6)

    # Normalize each to 0-255 uint8
    def normalize_to_uint8(arr):
        arr = np.clip(arr, np.percentile(arr, 1), np.percentile(arr, 99))
        return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return {
        'ExG': normalize_to_uint8(ExG),
        'GRVI': normalize_to_uint8(GRVI),
        'VARI': normalize_to_uint8(VARI),
        'TGI': normalize_to_uint8(TGI),
        'GLI': normalize_to_uint8(GLI),
    }


def enhance_with_vegetation_index(rgb, index_type='ExG', blend=0.3):
    """
    Enhance image by blending with a vegetation index.
    index_type: 'ExG', 'GRVI', 'VARI', 'TGI', 'GLI'
    blend: 0.0-1.0, how much of the index to blend in
    """
    indices = compute_vegetation_indices(rgb)
    idx_img = indices.get(index_type, indices['ExG'])

    # Convert index to 3-channel for blending
    idx_rgb = cv2.cvtColor(idx_img, cv2.COLOR_GRAY2RGB)

    # Blend with original
    enhanced = cv2.addWeighted(rgb, 1.0 - blend, idx_rgb, blend, 0)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def denoise_nlm(rgb, h=10, template_size=7, search_size=21):
    """
    Non-local means denoising - preserves edges better than median/mean.
    h: filter strength (higher = more denoising, 10 is good default)
    template_size: should be odd, 7 is good
    search_size: should be odd, 21 is good
    """
    # Ensure odd sizes
    template_size = template_size if template_size % 2 == 1 else template_size + 1
    search_size = search_size if search_size % 2 == 1 else search_size + 1
    return cv2.fastNlMeansDenoisingColored(rgb, None, h, h, template_size, search_size)


def single_scale_retinex(rgb, sigma=80):
    """
    Single-Scale Retinex - removes illumination effects, enhances details.
    Good for uneven lighting conditions.
    sigma: Gaussian blur sigma (higher = more illumination removal)
    """
    rgb_f = rgb.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(rgb_f, (0, 0), sigma)
    retinex = np.log10(rgb_f) - np.log10(blur + 1.0)

    # Normalize each channel separately
    result = np.zeros_like(rgb, dtype=np.uint8)
    for c in range(3):
        result[..., c] = cv2.normalize(retinex[..., c], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return result


def multi_scale_retinex(rgb, sigmas=(15, 80, 250), weights=None):
    """
    Multi-Scale Retinex with Color Restoration (MSRCR).
    Combines multiple scales for better illumination correction.
    sigmas: tuple of Gaussian sigmas for different scales
    weights: optional weights for each scale (defaults to equal)
    """
    if weights is None:
        weights = [1.0 / len(sigmas)] * len(sigmas)

    rgb_f = rgb.astype(np.float32) + 1.0
    log_rgb = np.log10(rgb_f)

    msr = np.zeros_like(rgb_f)
    for sigma, weight in zip(sigmas, weights):
        blur = cv2.GaussianBlur(rgb_f, (0, 0), sigma)
        msr += weight * (log_rgb - np.log10(blur + 1.0))

    # Color restoration
    intensity = np.mean(rgb_f, axis=2, keepdims=True)
    color_restoration = np.log10(125.0 * rgb_f / (intensity + 1.0) + 1.0)
    msr = msr * color_restoration

    # Normalize
    result = np.zeros_like(rgb, dtype=np.uint8)
    for c in range(3):
        result[..., c] = cv2.normalize(msr[..., c], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return result


def morphological_tophat(rgb, kernel_size=50):
    """
    Top-hat and Black-hat transform for illumination normalization.
    Removes uneven background illumination.
    kernel_size: size of structuring element (larger = removes larger variations)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Process each channel
    result = np.zeros_like(rgb)
    for c in range(3):
        channel = rgb[..., c]
        tophat = cv2.morphologyEx(channel, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(channel, cv2.MORPH_BLACKHAT, kernel)
        enhanced = cv2.add(channel, tophat)
        enhanced = cv2.subtract(enhanced, blackhat)
        result[..., c] = enhanced

    return result


def guided_filter_enhance(rgb, radius=8, eps=0.04):
    """
    Edge-preserving smoothing using guided filter.
    Better than bilateral filter for preserving sharp edges.
    radius: filter radius
    eps: regularization (smaller = sharper edges)
    """
    try:
        # Check if ximgproc is available
        rgb_f = rgb.astype(np.float32) / 255.0
        # Use green channel as guide (best for plants)
        guide = rgb_f[..., 1]

        filtered = np.zeros_like(rgb_f)
        for c in range(3):
            filtered[..., c] = cv2.ximgproc.guidedFilter(guide, rgb_f[..., c], radius, eps)

        return (filtered * 255).clip(0, 255).astype(np.uint8)
    except AttributeError:
        # ximgproc not available, fall back to bilateral
        return cv2.bilateralFilter(rgb, radius, eps * 1000, eps * 1000)


def enhance_lab_green(rgb, l_factor=1.0, a_shift=-10, b_shift=0):
    """
    Enhance in LAB color space.
    The 'a' channel is the red-green axis, so negative shift enhances green.
    l_factor: luminance multiplier
    a_shift: shift in a channel (negative = more green)
    b_shift: shift in b channel (negative = more blue)
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    # L channel: 0-255
    lab[..., 0] = np.clip(lab[..., 0] * l_factor, 0, 255)

    # a channel: 0-255, 128 is neutral (negative shift = more green)
    lab[..., 1] = np.clip(lab[..., 1] + a_shift, 0, 255)

    # b channel: 0-255, 128 is neutral (negative shift = more blue)
    lab[..., 2] = np.clip(lab[..., 2] + b_shift, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def white_balance_grayworld(rgb):
    """
    Gray World white balance correction.
    Assumes average color should be gray - corrects color cast.
    Good for standardizing colors across different lighting conditions.
    """
    result = rgb.astype(np.float32)

    # Calculate average for each channel
    avg_r = np.mean(result[..., 0])
    avg_g = np.mean(result[..., 1])
    avg_b = np.mean(result[..., 2])

    # Calculate overall average
    avg_all = (avg_r + avg_g + avg_b) / 3.0

    # Scale each channel
    if avg_r > 0:
        result[..., 0] = result[..., 0] * (avg_all / avg_r)
    if avg_g > 0:
        result[..., 1] = result[..., 1] * (avg_all / avg_g)
    if avg_b > 0:
        result[..., 2] = result[..., 2] * (avg_all / avg_b)

    return np.clip(result, 0, 255).astype(np.uint8)


def white_balance_max_white(rgb, percentile=99):
    """
    Max-White white balance - assumes brightest pixels should be white.
    percentile: use this percentile as "white" (99 avoids outliers)
    """
    result = rgb.astype(np.float32)

    for c in range(3):
        max_val = np.percentile(result[..., c], percentile)
        if max_val > 0:
            result[..., c] = result[..., c] * (255.0 / max_val)

    return np.clip(result, 0, 255).astype(np.uint8)


def difference_of_gaussians(rgb, sigma1=1.0, sigma2=3.0, blend=0.3):
    """
    Difference of Gaussians - enhances edges, similar to biological vision.
    sigma1: smaller sigma (detail)
    sigma2: larger sigma (context)
    blend: how much DoG to add to original
    """
    g1 = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), sigma1)
    g2 = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), sigma2)

    dog = g1 - g2

    # Normalize DoG to visible range
    dog = cv2.normalize(dog, None, -128, 128, cv2.NORM_MINMAX)

    # Blend with original
    enhanced = rgb.astype(np.float32) + blend * dog

    return np.clip(enhanced, 0, 255).astype(np.uint8)


def local_contrast_normalization(rgb, kernel_size=31):
    """
    Local contrast normalization - enhances local details.
    Divides by local standard deviation.
    kernel_size: size of local region (must be odd)
    """
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    result = np.zeros_like(rgb, dtype=np.float32)

    for c in range(3):
        channel = rgb[..., c].astype(np.float32)

        # Local mean
        local_mean = cv2.GaussianBlur(channel, (kernel_size, kernel_size), 0)

        # Local variance
        local_sq_mean = cv2.GaussianBlur(channel**2, (kernel_size, kernel_size), 0)
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0) + 1e-6)

        # Normalize
        normalized = (channel - local_mean) / local_std

        # Scale back to 0-255
        result[..., c] = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)

    return result.astype(np.uint8)


def adaptive_gamma(rgb, clip_limit=2.0):
    """
    Adaptive gamma correction based on image histogram.
    Automatically determines optimal gamma for the image.
    """
    # Convert to LAB
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[..., 0].astype(np.float32) / 255.0

    # Calculate optimal gamma based on mean luminance
    mean_l = np.mean(l_channel)

    # If image is dark, use gamma < 1 to brighten; if bright, use gamma > 1
    if mean_l < 0.5:
        gamma = 0.5 + mean_l  # Range: 0.5-1.0 for dark images
    else:
        gamma = mean_l + 0.5  # Range: 1.0-1.5 for bright images

    # Apply gamma to L channel
    l_corrected = np.power(l_channel, 1.0 / gamma)
    lab[..., 0] = (l_corrected * 255).clip(0, 255).astype(np.uint8)

    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def shadow_highlight_correction(rgb, shadow_amount=0.3, highlight_amount=0.3):
    """
    Correct shadows and highlights separately.
    shadow_amount: how much to lift shadows (0-1)
    highlight_amount: how much to reduce highlights (0-1)
    """
    # Convert to LAB for luminance manipulation
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l = lab[..., 0].astype(np.float32) / 255.0

    # Shadow mask (dark areas)
    shadow_mask = 1.0 - l
    shadow_mask = np.power(shadow_mask, 2)  # Concentrate on darkest areas

    # Highlight mask (bright areas)
    highlight_mask = l
    highlight_mask = np.power(highlight_mask, 2)  # Concentrate on brightest areas

    # Apply corrections
    l_corrected = l + shadow_amount * shadow_mask * (1.0 - l)
    l_corrected = l_corrected - highlight_amount * highlight_mask * l_corrected

    lab[..., 0] = (l_corrected * 255).clip(0, 255).astype(np.uint8)

    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# =========================
# SAM2 generator config
# =========================

def make_mask_generator(sam2_model):
    return SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=32,
        points_per_batch=32,
        pred_iou_thresh=0.90,
        stability_score_thresh=0.80,
        crop_n_layers=1,
        crop_overlap_ratio=0.30,
        crop_n_points_downscale_factor=2,
        box_nms_thresh=0.6,
        min_mask_region_area=800,
        use_m2m=True,
        output_mode="binary_mask",
    )


# =========================
# Config resolver
# =========================

from pathlib import Path
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

def _compose_from_yaml(yaml_path: str):
    """Compose a Hydra/OmegaConf config from a specific YAML file."""
    conf_dir = str(Path(yaml_path).parent)
    conf_name = Path(yaml_path).stem
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=conf_dir, job_name="sam2_gui", version_base=None):
        return compose(config_name=conf_name)

def _resolve_sam2_cfg(cfg_field, ckpt_path: str | None = None, fallback_short="sam2.1_hiera_l"):
    """
    Accepts:
      - dict              -> converts to DictConfig
      - DictConfig        -> returns as-is
      - path to *.yaml    -> compose from that YAML
      - directory path    -> tries common filenames inside it
      - short name (str)  -> tries to find YAML near checkpoint or $SAM2_CONFIG_DIR; 
                             otherwise returns the short name (so build_sam2 can resolve pkg configs).
      - None              -> uses fallback_short
    """
    # 1) already structured
    if cfg_field is None:
        return fallback_short
    if isinstance(cfg_field, DictConfig):
        return cfg_field
    if isinstance(cfg_field, dict):
        return OmegaConf.create(cfg_field)

    # 2) string-like
    s = str(cfg_field).strip()
    if not s:
        return fallback_short

    # YAML file?
    if Path(s).is_file() and s.lower().endswith((".yaml", ".yml")):
        return _compose_from_yaml(s)

    # Config directory?
    if Path(s).is_dir():
        # common name in SAM2 repos
        guess = Path(s) / "sam2.1_hiera_l.yaml"
        if guess.exists():
            return _compose_from_yaml(str(guess))
        # otherwise pick any reasonable YAML in that folder
        for y in sorted(Path(s).glob("*.y*ml")):
            if "hiera" in y.stem:
                return _compose_from_yaml(str(y))
        # last resort: first yaml
        cands = sorted(Path(s).glob("*.y*ml"))
        if cands:
            return _compose_from_yaml(str(cands[0]))

    # Short name → search typical places (near ckpt or $SAM2_CONFIG_DIR)
    guesses = []
    env_dir = os.environ.get("SAM2_CONFIG_DIR")
    if env_dir:
        guesses += [str(Path(env_dir) / "sam2.1"), env_dir]
    if ckpt_path:
        ck = Path(ckpt_path)
        repo_root = ck.parent.parent if ck.suffix == ".pt" else ck.parent
        guesses += [str(repo_root / "configs" / "sam2.1"), str(repo_root / "configs")]
    for d in guesses:
        y = Path(d) / f"{s}.yaml"
        if y.exists():
            return _compose_from_yaml(str(y))

    # Give up → let build_sam2 resolve package configs by short name
    return s



# =========================
# GUI
# =========================

@dataclass
class SegResult:
    masks: list
    img_color: np.ndarray
    img_seg: np.ndarray
    rotate_applied: bool


class ToolTip:
    """Modern tooltip with fade-in animation and rounded corners."""
    def __init__(self, widget, text, delay=400):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tip_window = None
        self.id = None
        self.alpha = 0.0
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _schedule(self, event=None):
        self._hide()
        self.id = self.widget.after(self.delay, self._show)

    def _show(self):
        if self.tip_window:
            return
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4

        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_attributes("-topmost", True)
        try:
            tw.wm_attributes("-alpha", 0.0)
        except Exception:
            pass

        # Tooltip styling
        frame = tk.Frame(tw, bg="#1a1a2e", bd=0, relief="flat")
        frame.pack()

        label = tk.Label(
            frame,
            text=self.text,
            bg="#1a1a2e",
            fg="#eaf4f4",
            font=("Helvetica", 10),
            padx=10,
            pady=6,
        )
        label.pack()

        tw.update_idletasks()
        tw_width = tw.winfo_reqwidth()
        x = x - tw_width // 2
        tw.wm_geometry(f"+{x}+{y}")

        self._fade_in()

    def _fade_in(self):
        if not self.tip_window:
            return
        self.alpha = min(1.0, self.alpha + 0.15)
        try:
            self.tip_window.wm_attributes("-alpha", self.alpha)
        except Exception:
            pass
        if self.alpha < 1.0:
            self.widget.after(20, self._fade_in)

    def _hide(self, event=None):
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None
        self.alpha = 0.0


class AnimatedSpinner:
    """Animated loading spinner overlay."""
    def __init__(self, parent, colors):
        self.parent = parent
        self.colors = colors
        self.canvas = None
        self.angle = 0
        self.animating = False
        self.label_text = "Processing..."

    def show(self, text="Processing..."):
        self.label_text = text
        if self.canvas:
            return

        self.canvas = tk.Canvas(
            self.parent,
            bg=self.colors['canvas_bg'],
            highlightthickness=0,
        )
        self.canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Semi-transparent overlay
        self.canvas.create_rectangle(
            0, 0, 2000, 2000,
            fill=self.colors['bg_dark'],
            stipple="gray50",
            tags="overlay"
        )

        self.animating = True
        self._draw_spinner()

    def _draw_spinner(self):
        if not self.canvas or not self.animating:
            return

        self.canvas.delete("spinner")
        self.canvas.delete("text")

        cx = self.canvas.winfo_width() // 2
        cy = self.canvas.winfo_height() // 2

        if cx < 10:
            cx, cy = 200, 150

        r = 30
        # Draw arc segments
        for i in range(8):
            start_angle = self.angle + i * 45
            alpha = 1.0 - (i * 0.12)
            color = self._blend_color(self.colors['accent'], self.colors['bg_dark'], alpha)
            self.canvas.create_arc(
                cx - r, cy - r, cx + r, cy + r,
                start=start_angle, extent=30,
                style="arc", width=4, outline=color,
                tags="spinner"
            )

        # Text below spinner
        self.canvas.create_text(
            cx, cy + r + 25,
            text=self.label_text,
            fill=self.colors['text_light'],
            font=("Helvetica", 12, "bold"),
            tags="text"
        )

        self.angle = (self.angle + 15) % 360
        self.parent.after(50, self._draw_spinner)

    def _blend_color(self, c1, c2, alpha):
        """Blend two hex colors."""
        try:
            r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
            r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
            r = int(r1 * alpha + r2 * (1 - alpha))
            g = int(g1 * alpha + g2 * (1 - alpha))
            b = int(b1 * alpha + b2 * (1 - alpha))
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return c1

    def hide(self):
        self.animating = False
        if self.canvas:
            self.canvas.destroy()
            self.canvas = None


class LeafSegmenterGUI:
    def __init__(self, root):
        self.root = root
        root.title("🌿 Leaf Segmenter — SAM2 Plant Phenotyping Tool")

        # ═══════════════════════════════════════════════════════════════════════
        # COLOR THEME - Modern Plant-inspired palette
        # ═══════════════════════════════════════════════════════════════════════
        self.colors = {
            'bg_dark': '#1a2f1a',       # Deep forest - main background
            'bg_medium': '#2d4a2d',     # Forest green - panels
            'bg_light': '#4a7c4a',      # Sage green - sections
            'bg_pale': '#e8f5e8',       # Mint cream - inputs/entries
            'accent': '#4caf50',        # Vibrant green - buttons
            'accent_hover': '#66bb6a',  # Lighter green - hover
            'accent_active': '#81c784', # Active state
            'text_light': '#f1f8e9',    # Warm white text on dark
            'text_dark': '#1b5e20',     # Deep green text on light
            'canvas_bg': '#0d1f0d',     # Dark forest - canvas background
            'warning': '#ff9800',       # Orange for warnings
            'success': '#4caf50',       # Green for success
            'error': '#f44336',         # Red for errors
            'info': '#2196f3',          # Blue for info
            'border': '#3d5c3d',        # Subtle border color
            'highlight': '#a5d6a7',     # Highlight color
        }

        # Unicode icons for sections
        self.icons = {
            'model': '🧠',
            'image': '🖼️',
            'enhance': '✨',
            'sam': '🎯',
            'phenotype': '📊',
            'action': '⚡',
            'masks': '🎭',
            'preview': '👁️',
            'training': '🔬',
            'folder': '📁',
            'save': '💾',
            'load': '📂',
            'segment': '✂️',
            'success': '✓',
            'warning': '⚠️',
            'info': 'ℹ️',
        }
        c = self.colors

        # Configure root window
        root.configure(bg=c['bg_dark'])

        # Configure ttk styles
        style = ttk.Style()
        style.theme_use('clam')  # 'clam' theme allows more customization

        # Main frame style
        style.configure('TFrame', background=c['bg_dark'])
        style.configure('TLabel', background=c['bg_dark'], foreground=c['text_light'])
        style.configure('TCheckbutton', background=c['bg_dark'], foreground=c['text_light'])
        style.configure('TRadiobutton', background=c['bg_dark'], foreground=c['text_light'])

        # LabelFrame styles for different sections
        style.configure('TLabelframe', background=c['bg_medium'], bordercolor=c['bg_light'])
        style.configure('TLabelframe.Label', background=c['bg_medium'], foreground=c['text_light'],
                       font=('Helvetica', 11, 'bold'))

        # Special styles for different panel types
        style.configure('Model.TLabelframe', background=c['bg_medium'])
        style.configure('Model.TLabelframe.Label', background=c['bg_medium'], foreground=c['text_light'])

        style.configure('Options.TLabelframe', background=c['bg_medium'])
        style.configure('Options.TLabelframe.Label', background=c['bg_medium'], foreground=c['text_light'],
                       font=('Helvetica', 10, 'bold'))

        style.configure('Preview.TLabelframe', background=c['bg_medium'])
        style.configure('Preview.TLabelframe.Label', background=c['bg_medium'], foreground=c['text_light'])

        style.configure('Masks.TLabelframe', background=c['bg_light'])
        style.configure('Masks.TLabelframe.Label', background=c['bg_light'], foreground=c['text_dark'])

        style.configure('Training.TLabelframe', background=c['bg_pale'])
        style.configure('Training.TLabelframe.Label', background=c['bg_pale'], foreground=c['text_dark'],
                       font=('Helvetica', 11, 'bold'))

        # Button styles - Modern flat design
        style.configure('TButton',
                       background=c['accent'],
                       foreground=c['text_light'],
                       borderwidth=0,
                       focuscolor=c['accent'],
                       padding=(12, 6),
                       font=('Helvetica', 10))
        style.map('TButton',
                 background=[('active', c['accent_hover']), ('pressed', c['accent_active'])],
                 foreground=[('active', c['text_dark']), ('pressed', c['text_dark'])])

        # Accent button (for important actions) - Bolder styling
        style.configure('Accent.TButton',
                       background=c['accent'],
                       foreground=c['text_light'],
                       font=('Helvetica', 11, 'bold'),
                       padding=(14, 8))
        style.map('Accent.TButton',
                 background=[('active', c['accent_hover']), ('pressed', c['accent_active'])],
                 foreground=[('active', c['text_dark'])])

        # Secondary button style (less prominent)
        style.configure('Secondary.TButton',
                       background=c['bg_medium'],
                       foreground=c['text_light'],
                       font=('Helvetica', 10),
                       padding=(10, 5))
        style.map('Secondary.TButton',
                 background=[('active', c['bg_light'])],
                 foreground=[('active', c['text_light'])])

        # Icon button style (small square buttons)
        style.configure('Icon.TButton',
                       background=c['bg_medium'],
                       foreground=c['text_light'],
                       font=('Helvetica', 12),
                       padding=(6, 4),
                       width=3)
        style.map('Icon.TButton',
                 background=[('active', c['accent'])])

        # Entry style
        style.configure('TEntry',
                       fieldbackground=c['bg_pale'],
                       foreground=c['text_dark'],
                       insertcolor=c['text_dark'])

        # Combobox style
        style.configure('TCombobox',
                       fieldbackground=c['bg_pale'],
                       background=c['bg_pale'],
                       foreground=c['text_dark'])

        # Scale/slider style
        style.configure('TScale',
                       background=c['bg_medium'],
                       troughcolor=c['bg_dark'])

        # Scrollbar style
        style.configure('TScrollbar',
                       background=c['bg_light'],
                       troughcolor=c['bg_dark'],
                       bordercolor=c['bg_medium'],
                       arrowcolor=c['text_light'])

        # Spinbox style
        style.configure('TSpinbox',
                       fieldbackground=c['bg_pale'],
                       foreground=c['text_dark'])

        # PanedWindow style
        style.configure('TPanedwindow', background=c['bg_dark'])

        # Separator style
        style.configure('TSeparator', background=c['bg_light'])

        # Notebook style (if used)
        style.configure('TNotebook', background=c['bg_dark'])
        style.configure('TNotebook.Tab', background=c['bg_medium'], foreground=c['text_light'])

        self.img_path = None
        self.img = None
        self.img_preview = None
        self.sr: SegResult | None = None
        self.sam2_model = None
        # --- crop tool state ---
        self._crop_mode = tk.BooleanVar(value=False)  # toolbar toggle
        self._crop_canvas_id = None                   # rectangle overlay on canvas
        self._crop_start_canvas = None                # (x,y) canvas coords
        self._crop_rect_img = None                    # (x1,y1,x2,y2) in IMAGE pixels

        # preview state
        self._img_for_preview = None  # numpy array currently shown
        self._tk_img_id = None        # canvas image item id
        self._zoom = 1.0              # >1 = zoomed in (relative to "fit")
        self._pan = [0, 0]            # dx, dy in canvas pixels
        self._fit_mode = True         # Fit-to-window vs custom zoom
        self._drag_start = None       # for panning

        self.img_orig = None             # unmodified RGB image as loaded
        self.rot_angle = tk.DoubleVar(value=0.0)   # degrees, CCW positive

        # knob drawing state
        self._knob = None
        self._knob_center = (36, 36)     # pixels in the knob canvas
        self._knob_r = 28

        # click-to-pick editing state (must exist before make_preview_frame binds events)
        self._edit_mode = tk.StringVar(value="none")   # 'none' | 'deselect' | 'select'
        self._picks = set()                            # indices of masks clicked on canvas
        self._pick_mode = tk.BooleanVar(value=False)   # toolbar toggle for pick mode
        self._suppress_listbox_select = False          # guard: avoid preview swap on programmatic selection
        self._pick_blacklist = {0}                     # mask indices to never pick in preview
        self._last_pick_candidates = None              # for cycling candidates under cursor
        self._last_pick_xy = None
        self._last_pick_cycle_idx = 0
        self._busy_win = None                          # simple modal busy indicator

        # --- NEW LAYOUT: Resizable main + training, status bar at bottom ---
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)     # main paned window expands
        root.grid_rowconfigure(1, weight=0)     # status bar fixed

        self.main_paned = ttk.PanedWindow(root, orient="vertical")
        self.main_paned.grid(row=0, column=0, sticky="nsew")

        main_top = ttk.Frame(self.main_paned)
        self.main_paned.add(main_top, weight=3)

        # Configure main_top grid (left + right)
        main_top.grid_columnconfigure(0, weight=0)  # left panel - fixed width
        main_top.grid_columnconfigure(1, weight=1)  # right panel - expandable
        main_top.grid_rowconfigure(0, weight=1)

        # ═══════════════════════════════════════════════════════════════════════
        # LEFT PANEL: Scrollable container for all controls
        # ═══════════════════════════════════════════════════════════════════════
        left_container = ttk.Frame(main_top)
        left_container.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=6)

        # Canvas + Scrollbar for scrolling
        self._left_canvas = tk.Canvas(left_container, width=420, highlightthickness=0,
                                      bg=self.colors['bg_dark'])
        self._left_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=self._left_canvas.yview)
        self._left_canvas.configure(yscrollcommand=self._left_scrollbar.set)

        self._left_scrollbar.pack(side="right", fill="y")
        self._left_canvas.pack(side="left", fill="both", expand=True)

        # Frame inside canvas to hold all controls (scrollable)
        self.left_panel = ttk.Frame(self._left_canvas)
        self._left_canvas_window = self._left_canvas.create_window((0, 0), window=self.left_panel, anchor="nw")

        # Update scroll region when content changes
        def _on_left_configure(e):
            self._left_canvas.configure(scrollregion=self._left_canvas.bbox("all"))
        self.left_panel.bind("<Configure>", _on_left_configure)

        # Make canvas resize with container width
        def _on_canvas_configure(e):
            self._left_canvas.itemconfig(self._left_canvas_window, width=e.width)
        self._left_canvas.bind("<Configure>", _on_canvas_configure)

        # Enable mousewheel scrolling on left panel (bind directly to canvas and children)
        def _on_left_mousewheel(e):
            self._left_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
            return "break"  # Prevent propagation
        def _on_left_scroll_linux(e, direction):
            self._left_canvas.yview_scroll(direction, "units")
            return "break"

        # Bind to the canvas itself
        self._left_canvas.bind("<MouseWheel>", _on_left_mousewheel)
        self._left_canvas.bind("<Button-4>", lambda e: _on_left_scroll_linux(e, -1))
        self._left_canvas.bind("<Button-5>", lambda e: _on_left_scroll_linux(e, 1))

        # Also bind to the inner frame so scrolling works when hovering over widgets
        self.left_panel.bind("<MouseWheel>", _on_left_mousewheel)
        self.left_panel.bind("<Button-4>", lambda e: _on_left_scroll_linux(e, -1))
        self.left_panel.bind("<Button-5>", lambda e: _on_left_scroll_linux(e, 1))

        # ═══════════════════════════════════════════════════════════════════════
        # RIGHT PANEL: Preview (top) + Masks (bottom) in a PanedWindow
        # ═══════════════════════════════════════════════════════════════════════
        self.right_panel = ttk.PanedWindow(main_top, orient="vertical")
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=6)

        # Build the frames
        self.make_model_frame(self.left_panel)
        self.make_options_frame(self.left_panel)
        self.make_preview_frame(self.right_panel)
        self.make_masks_frame(self.right_panel)
        

        # click-to-pick editing state
        self._edit_mode = tk.StringVar(value="none")   # 'none' | 'deselect' | 'select'
        self._picks = set()                            # mask indices clicked on canvas

        
        # --- batch mode state ---
        self.batch_dir: str | None = None
        self.batch_images: list[str] = []
        self.batch_idx: int = -1
        self._batch_mask_cache = {}  # img_path -> SegResult
        self._batch_out_dir = None

        # UI vars for the Fine-tune tab
        self.train_auto_prompt = tk.BooleanVar(value=True)
        self.train_ckpt_var = tk.StringVar(value="")            # falls back to Model panel if left empty
        self.train_cfg_var  = tk.StringVar(value="(auto-detect)")
        self.train_out_var  = tk.StringVar(value=str(Path.home()/ "sam2_arabidopsis.pth"))
        self.train_steps_var= tk.IntVar(value=10000)
        self.train_lr_var   = tk.DoubleVar(value=1e-5)
        self.train_size_var = tk.IntVar(value=1024)
        self.train_device_var = tk.StringVar(value="mps")  # Mac default

        # UI vars for the Hidden Structure (Occlusion) tab
        self.occ_data_var = tk.StringVar(value="")
        self.occ_ckpt_var = tk.StringVar(value="")
        self.occ_cfg_var = tk.StringVar(value="(auto-detect)")  # Will auto-detect from checkpoint
        self.occ_out_var = tk.StringVar(value=str(Path.home()/ "sam2_hidden_structure.pth"))
        self.occ_size_var = tk.IntVar(value=512)
        self.occ_steps_var = tk.IntVar(value=20000)
        self.occ_lr_var = tk.DoubleVar(value=1e-4)
        self.occ_min_var = tk.DoubleVar(value=0.15)
        self.occ_max_var = tk.DoubleVar(value=0.50)
        self.occ_count_min_var = tk.IntVar(value=1)
        self.occ_count_max_var = tk.IntVar(value=3)
        self.occ_device_var = tk.StringVar(value="cuda")

        # UI vars for the Target Segment tab
        self.target_root = None
        self.target_root_var = tk.StringVar(value="")
        self.target_examples = []
        self.target_ckpt_var = tk.StringVar(value="")
        self.target_cfg_var = tk.StringVar(value="(auto-detect)")
        self.target_out_var = tk.StringVar(value=str(Path.home()/ "sam2_target_segment.pth"))
        self.target_steps_var = tk.IntVar(value=2000)
        self.target_lr_var = tk.DoubleVar(value=1e-5)
        self.target_size_var = tk.IntVar(value=512)
        self.target_device_var = tk.StringVar(value="cpu")
        self.target_batch_var = tk.IntVar(value=2)
        self.target_allow_empty_var = tk.BooleanVar(value=True)
        self.target_resume_var = tk.BooleanVar(value=False)
        self.target_arch_var = tk.StringVar(value="unet_resnet18")
        self.target_pretrained_var = tk.BooleanVar(value=True)
        # Tip-only segmentation model (no SAM at inference)
        self.target_use_tipseg = tk.BooleanVar(value=False)
        self.target_tipseg_thresh = tk.DoubleVar(value=0.50)
        self.target_tipseg_min_area = tk.IntVar(value=200)
        self.target_tipseg_keep_largest = tk.BooleanVar(value=True)
        self.tipseg_use_tiles = tk.BooleanVar(value=True)
        self.tipseg_tile_size = tk.IntVar(value=512)
        self.tipseg_stride = tk.IntVar(value=256)
        self.tipseg_color_guided = tk.BooleanVar(value=True)
        self.tipseg_color_min_area = tk.IntVar(value=600)
        self.tipseg_hue_low = tk.IntVar(value=10)
        self.tipseg_hue_high = tk.IntVar(value=40)
        self.tipseg_sat_min = tk.IntVar(value=35)
        self.tipseg_val_min = tk.IntVar(value=40)
        self.tipseg_val_brown_max = tk.IntVar(value=200)
        self.tipseg_min_leaf_pct = tk.DoubleVar(value=2.0)
        self.tipseg_min_stress_pct = tk.DoubleVar(value=0.0)
        self.tipseg_stop_after_first = tk.BooleanVar(value=False)
        self.tipseg_model = None
        self.tipseg_meta = {}

        # (Legacy) Target filter/classifier path (kept for now, but the UI now prefers tipseg)
        self.target_filter_enable = tk.BooleanVar(value=False)
        self.target_filter_k = tk.DoubleVar(value=2.5)
        self.target_filter_stats = None
        self.target_clf = None
        self.target_clf_meta = {}
        self.target_use_classifier = tk.BooleanVar(value=True)
        self.target_cls_thresh = tk.DoubleVar(value=0.50)
        self.target_cls_keep_best = tk.BooleanVar(value=True)

        # Track which SAM weights are active (base vs fine-tuned)
        self._sam_weights_tag = "(none)"

        # Build the training panel (resizable)
        train_frame = self.make_training_frame(self.main_paned)
        self.main_paned.add(train_frame, weight=1)

        # ═══════════════════════════════════════════════════════════════════════
        # STATUS BAR - Bottom of the window
        # ═══════════════════════════════════════════════════════════════════════
        self.make_status_bar(root)

        # Initialize the animated spinner
        self._spinner = AnimatedSpinner(root, self.colors)

        # click-to-pick editing state
        self._edit_mode = tk.StringVar(value="none")  # 'none' | 'deselect' | 'select'
        self._picks = set()                            # set of mask indices picked on canvas

        # Bind keyboard shortcuts
        self._bind_global_shortcuts()

        # --- training state ---
        self.train_root = None              # dataset root containing images/ and masks/
        self.train_images_dir = None
        self.train_masks_dir = None
        self.train_examples = []            # [{ "image": path, "masks": [paths...] }]

        # Show welcome message after a short delay (so widgets are rendered)
        self.root.after(500, lambda: self.set_status("Welcome! Load a SAM2 model and open an image to begin.", "info"))

    # ---- Frames ----
    def _add_left_pane(self, parent, widget, weight=1, fill="x", expand=False):
        """Add a widget to a PanedWindow if available, else pack it normally."""
        if isinstance(parent, ttk.PanedWindow):
            parent.add(widget, weight=weight)
        else:
            widget.pack(fill=fill, expand=expand, pady=(0, 8))

    def make_model_frame(self, parent):
        c = self.colors
        icon = self.icons.get('model', '')
        f = ttk.LabelFrame(parent, text=f"  {icon} Model  ", padding=(10, 8), style='TLabelframe')
        self._add_left_pane(parent, f, weight=1, fill="x", expand=False)

        # Row 0: Checkpoint
        row0 = ttk.Frame(f)
        row0.pack(fill="x", pady=3)
        ttk.Label(row0, text="Checkpoint:", width=12).pack(side="left")
        self.e_ckpt = ttk.Entry(row0, width=38)
        self.e_ckpt.pack(side="left", fill="x", expand=True, padx=(0, 6))
        btn_ckpt = ttk.Button(row0, text="…", width=3, command=self.pick_ckpt, style='Icon.TButton')
        btn_ckpt.pack(side="left")
        ToolTip(btn_ckpt, "Browse for SAM2 checkpoint file (.pt)")

        # Row 1: Config
        row1 = ttk.Frame(f)
        row1.pack(fill="x", pady=3)
        ttk.Label(row1, text="Config:", width=12).pack(side="left")
        self.e_cfg = ttk.Entry(row1, width=38)
        self.e_cfg.insert(0, "sam2.1_hiera_l")
        self.e_cfg.pack(side="left", fill="x", expand=True, padx=(0, 6))
        btn_cfg = ttk.Button(row1, text="…", width=3, command=self.pick_cfg, style='Icon.TButton')
        btn_cfg.pack(side="left")
        ToolTip(btn_cfg, "Browse for SAM2 config YAML file")

        # Row 2: Device + postprocessing
        row2 = ttk.Frame(f)
        row2.pack(fill="x", pady=3)
        lbl_dev = ttk.Label(row2, text="Device:", width=12)
        lbl_dev.pack(side="left")
        self.e_dev = ttk.Entry(row2, width=8)
        self.e_dev.insert(0, "cpu")
        self.e_dev.pack(side="left")
        ToolTip(self.e_dev, "Device to run model on (cpu, cuda, mps)")
        self.chk_post = tk.BooleanVar(value=False)
        chk = ttk.Checkbutton(row2, text="Postprocessing", variable=self.chk_post)
        chk.pack(side="left", padx=(12, 0))
        ToolTip(chk, "Apply SAM2 postprocessing to masks")

        # Row 3: Load buttons
        row3 = ttk.Frame(f)
        row3.pack(fill="x", pady=(10, 4))
        btn_load = ttk.Button(row3, text="⬆ Load Model", command=self.load_model, style='Accent.TButton')
        btn_load.pack(side="left", padx=(0, 8))
        ToolTip(btn_load, "Load SAM2 model from checkpoint and config")
        btn_bundle = ttk.Button(row3, text="📦 Load Bundle…", command=self.load_bundle, style='Accent.TButton')
        btn_bundle.pack(side="left")
        ToolTip(btn_bundle, "Load a pre-packaged SAM2 bundle (.pt file with embedded config)")


    def make_options_frame(self, parent):
        c = self.colors
        # Create a scrollable container for all options
        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True)

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 1: Image Input
        # ═══════════════════════════════════════════════════════════════════════
        sec1 = ttk.LabelFrame(container, text=f"  {self.icons['image']} Image Input  ", padding=(10, 8), style='Options.TLabelframe')
        self._add_left_pane(container, sec1, weight=1, fill="x", expand=False)

        # Open buttons row
        btn_row = ttk.Frame(sec1)
        btn_row.pack(fill="x", pady=(0, 8))
        btn_open = ttk.Button(btn_row, text="📂 Open Image…", command=self.open_image)
        btn_open.pack(side="left")
        ToolTip(btn_open, "Open a single image file (Ctrl+O)")
        btn_folder = ttk.Button(btn_row, text="📁 Open Folder…", command=self.open_folder)
        btn_folder.pack(side="left", padx=(8, 0))
        ToolTip(btn_folder, "Open a folder of images for batch processing")

        # Rotation control (compact horizontal layout)
        rot_row = ttk.Frame(sec1)
        rot_row.pack(fill="x")

        ttk.Label(rot_row, text="🔄 Rotate:").pack(side="left")
        self.chk_rotate = getattr(self, "chk_rotate", tk.BooleanVar(value=True))

        self._knob = tk.Canvas(rot_row, width=50, height=50, bg=c['bg_pale'],
                              highlightthickness=2, highlightbackground=c['accent'])
        self._knob.pack(side="left", padx=(8, 4))
        self._knob.bind("<Button-1>", self._knob_down)
        self._knob.bind("<B1-Motion>", self._knob_drag)
        self._knob_center = (25, 25)
        self._knob_r = 20
        ToolTip(self._knob, "Drag to rotate image\n(or use spinbox for precise angle)")

        self.spin_angle = ttk.Spinbox(rot_row, from_=-180, to=180, increment=1,
                                      textvariable=self.rot_angle, width=5,
                                      command=self._angle_from_spin)
        self.spin_angle.pack(side="left")
        ttk.Label(rot_row, text="°").pack(side="left")
        btn_reset = ttk.Button(rot_row, text="↺", width=3, command=lambda: self._set_angle(0), style='Icon.TButton')
        btn_reset.pack(side="left", padx=(8, 0))
        ToolTip(btn_reset, "Reset rotation to 0°")

        self.spin_angle.bind("<Return>", lambda e: self._angle_from_spin())
        self.spin_angle.bind("<FocusOut>", lambda e: self._angle_from_spin())
        self._draw_knob()

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 2: Image Enhancement (Redesigned)
        # ═══════════════════════════════════════════════════════════════════════
        sec2 = ttk.LabelFrame(container, text=f"  {self.icons['enhance']} Enhancement  ", padding=(10, 8), style='Options.TLabelframe')
        self._add_left_pane(container, sec2, weight=1, fill="x", expand=False)

        # --- Pipeline Selection ---
        self.use_green = tk.BooleanVar(value=True)
        self.use_classic = tk.BooleanVar(value=False)
        self.enhance_pipeline = tk.StringVar(value="plant")

        def _on_pipeline_change(*args):
            mode = self.enhance_pipeline.get()
            self.use_green.set(mode in ("plant", "both"))
            self.use_classic.set(mode in ("basic", "both"))

        pipeline_row = ttk.Frame(sec2)
        pipeline_row.pack(fill="x", pady=(0, 8))
        ttk.Label(pipeline_row, text="Pipeline:", width=9).pack(side="left")
        combo_pipeline = ttk.Combobox(pipeline_row, width=14, state="readonly",
                                       textvariable=self.enhance_pipeline,
                                       values=("none", "plant", "basic", "both"))
        combo_pipeline.pack(side="left")
        combo_pipeline.bind("<<ComboboxSelected>>", _on_pipeline_change)
        ToolTip(combo_pipeline, "none: No auto-enhancement\nplant: Green-aware (CLAHE, bilateral, edges)\nbasic: Brightness/contrast/gamma\nboth: Apply both pipelines")

        # ─── Adjustments Sub-section ───
        adj_label = ttk.Label(sec2, text="─── Adjustments ───", font=("Helvetica", 9, "italic"))
        adj_label.pack(anchor="w", pady=(4, 2))

        self.s_brightness = tk.IntVar(value=0)
        self.s_contrast = tk.DoubleVar(value=1.0)
        self.s_gamma = tk.DoubleVar(value=1.0)

        # Brightness with value display
        br_row = ttk.Frame(sec2)
        br_row.pack(fill="x", pady=1)
        ttk.Label(br_row, text="Brightness", width=9).pack(side="left")
        self._br_scale = ttk.Scale(br_row, from_=-100, to=100, variable=self.s_brightness, orient="horizontal")
        self._br_scale.pack(side="left", fill="x", expand=True)
        self._br_val = ttk.Label(br_row, text="0", width=4)
        self._br_val.pack(side="left")
        self.s_brightness.trace_add("write", lambda *_: self._br_val.configure(text=str(self.s_brightness.get())))

        # Contrast with value display
        ct_row = ttk.Frame(sec2)
        ct_row.pack(fill="x", pady=1)
        ttk.Label(ct_row, text="Contrast", width=9).pack(side="left")
        self._ct_scale = ttk.Scale(ct_row, from_=0.5, to=2.0, variable=self.s_contrast, orient="horizontal")
        self._ct_scale.pack(side="left", fill="x", expand=True)
        self._ct_val = ttk.Label(ct_row, text="1.0", width=4)
        self._ct_val.pack(side="left")
        self.s_contrast.trace_add("write", lambda *_: self._ct_val.configure(text=f"{self.s_contrast.get():.1f}"))

        # Gamma with value display
        gm_row = ttk.Frame(sec2)
        gm_row.pack(fill="x", pady=1)
        ttk.Label(gm_row, text="Gamma", width=9).pack(side="left")
        self._gm_scale = ttk.Scale(gm_row, from_=0.5, to=2.5, variable=self.s_gamma, orient="horizontal")
        self._gm_scale.pack(side="left", fill="x", expand=True)
        self._gm_val = ttk.Label(gm_row, text="1.0", width=4)
        self._gm_val.pack(side="left")
        self.s_gamma.trace_add("write", lambda *_: self._gm_val.configure(text=f"{self.s_gamma.get():.1f}"))

        # ─── Sharpening Sub-section ───
        sharp_label = ttk.Label(sec2, text="─── Sharpening ───", font=("Helvetica", 9, "italic"))
        sharp_label.pack(anchor="w", pady=(8, 2))

        self.chk_unsharp = tk.BooleanVar(value=False)
        self.unsharp_amount = tk.DoubleVar(value=1.5)
        self.unsharp_sigma = tk.DoubleVar(value=10.0)
        self.unsharp_ksize = tk.IntVar(value=9)

        us_row1 = ttk.Frame(sec2)
        us_row1.pack(fill="x", pady=1)
        chk_us = ttk.Checkbutton(us_row1, text="Unsharp Mask", variable=self.chk_unsharp)
        chk_us.pack(side="left")
        ToolTip(chk_us, "Sharpen edges using unsharp masking")
        ttk.Label(us_row1, text="Amount:").pack(side="left", padx=(12, 2))
        ttk.Scale(us_row1, from_=0.5, to=3.0, variable=self.unsharp_amount, orient="horizontal", length=60).pack(side="left")
        ttk.Label(us_row1, text="σ:").pack(side="left", padx=(8, 2))
        ttk.Entry(us_row1, width=4, textvariable=self.unsharp_sigma).pack(side="left")
        ttk.Label(us_row1, text="Size:").pack(side="left", padx=(8, 2))
        ttk.Entry(us_row1, width=3, textvariable=self.unsharp_ksize).pack(side="left")

        self.chk_laplacian = tk.BooleanVar(value=False)
        lap_row = ttk.Frame(sec2)
        lap_row.pack(fill="x", pady=1)
        chk_lap = ttk.Checkbutton(lap_row, text="Laplacian Edge", variable=self.chk_laplacian)
        chk_lap.pack(side="left")
        ToolTip(chk_lap, "Convert to edge-detected image (grayscale)")

        # ─── Background Sub-section ───
        bg_label = ttk.Label(sec2, text="─── Background ───", font=("Helvetica", 9, "italic"))
        bg_label.pack(anchor="w", pady=(8, 2))

        self.chk_whiten = tk.BooleanVar(value=False)
        self.chk_darken_bg = tk.BooleanVar(value=False)
        self.s_val_min = tk.IntVar(value=200)
        self.s_sat_max = tk.IntVar(value=35)

        bg_row1 = ttk.Frame(sec2)
        bg_row1.pack(fill="x", pady=1)
        chk_wh = ttk.Checkbutton(bg_row1, text="⬜ Whiten BG", variable=self.chk_whiten)
        chk_wh.pack(side="left")
        ToolTip(chk_wh, "Make bright, low-saturation areas white")
        chk_dk = ttk.Checkbutton(bg_row1, text="⬛ Darken BG", variable=self.chk_darken_bg)
        chk_dk.pack(side="left", padx=(12, 0))
        ToolTip(chk_dk, "Make bright, low-saturation areas dark (black)")
        ttk.Label(bg_row1, text="V≥").pack(side="left", padx=(12, 2))
        ttk.Entry(bg_row1, width=4, textvariable=self.s_val_min).pack(side="left")
        ttk.Label(bg_row1, text="S≤").pack(side="left", padx=(8, 2))
        ttk.Entry(bg_row1, width=4, textvariable=self.s_sat_max).pack(side="left")

        # ─── Denoising Sub-section ───
        dn_label = ttk.Label(sec2, text="─── Denoising ───", font=("Helvetica", 9, "italic"))
        dn_label.pack(anchor="w", pady=(8, 2))

        self.dn_median_on = tk.BooleanVar(value=False)
        self.dn_median_ksize = tk.IntVar(value=5)
        self.dn_mean_on = tk.BooleanVar(value=False)
        self.dn_mean_ksize = tk.IntVar(value=3)

        dn_row = ttk.Frame(sec2)
        dn_row.pack(fill="x", pady=1)
        chk_med = ttk.Checkbutton(dn_row, text="Median", variable=self.dn_median_on)
        chk_med.pack(side="left")
        ToolTip(chk_med, "Median filter - good for salt & pepper noise")
        ttk.Entry(dn_row, width=3, textvariable=self.dn_median_ksize).pack(side="left", padx=(2, 12))
        chk_mean = ttk.Checkbutton(dn_row, text="Mean", variable=self.dn_mean_on)
        chk_mean.pack(side="left")
        ToolTip(chk_mean, "Mean/box filter - general smoothing")
        ttk.Entry(dn_row, width=3, textvariable=self.dn_mean_ksize).pack(side="left", padx=(2, 0))

        # ─── Edge Enhancement Sub-section ───
        edge_label = ttk.Label(sec2, text="─── Edge Enhancement ───", font=("Helvetica", 9, "italic"))
        edge_label.pack(anchor="w", pady=(8, 2))

        self.ed_on = tk.BooleanVar(value=False)
        self.ed_width = tk.IntVar(value=3)
        self.ed_amount = tk.DoubleVar(value=0.35)

        ed_row = ttk.Frame(sec2)
        ed_row.pack(fill="x", pady=1)
        chk_ed = ttk.Checkbutton(ed_row, text="Edge Darken", variable=self.ed_on)
        chk_ed.pack(side="left")
        ToolTip(chk_ed, "Darken pixels near edges - helps SAM detect boundaries")
        ttk.Label(ed_row, text="Width:").pack(side="left", padx=(12, 2))
        ttk.Entry(ed_row, width=3, textvariable=self.ed_width).pack(side="left")
        ttk.Label(ed_row, text="Amount:").pack(side="left", padx=(8, 2))
        ttk.Scale(ed_row, from_=0.0, to=1.0, variable=self.ed_amount, orient="horizontal", length=60).pack(side="left")

        # ─── Output Options Sub-section ───
        out_label = ttk.Label(sec2, text="─── Output Options ───", font=("Helvetica", 9, "italic"))
        out_label.pack(anchor="w", pady=(8, 2))

        self.s_halo_erode = tk.IntVar(value=1)
        self.s_halo_feather = tk.IntVar(value=2)
        self.s_close_iters = tk.IntVar(value=1)

        out_row = ttk.Frame(sec2)
        out_row.pack(fill="x", pady=1)
        ttk.Label(out_row, text="Erode:").pack(side="left")
        ttk.Entry(out_row, width=3, textvariable=self.s_halo_erode).pack(side="left", padx=(2, 8))
        ToolTip(out_row, "Erode mask edges to remove halo")
        ttk.Label(out_row, text="Feather:").pack(side="left")
        ttk.Entry(out_row, width=3, textvariable=self.s_halo_feather).pack(side="left", padx=(2, 8))
        ttk.Label(out_row, text="Close:").pack(side="left")
        ttk.Entry(out_row, width=3, textvariable=self.s_close_iters).pack(side="left", padx=(2, 0))

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 2b: Advanced Enhancement (NEW)
        # ═══════════════════════════════════════════════════════════════════════
        sec2b = ttk.LabelFrame(container, text="  🔬 Advanced Enhancement  ", padding=(10, 8), style='Options.TLabelframe')
        sec2b.pack(fill="x", pady=(0, 8))

        # --- Vegetation Index ---
        self.use_veg_index = tk.BooleanVar(value=False)
        self.veg_index_type = tk.StringVar(value="ExG")
        self.veg_index_blend = tk.DoubleVar(value=0.3)

        veg_row = ttk.Frame(sec2b)
        veg_row.pack(fill="x", pady=(0, 4))
        chk_veg = ttk.Checkbutton(veg_row, text="🌿 Vegetation Index", variable=self.use_veg_index)
        chk_veg.pack(side="left")
        ToolTip(chk_veg, "Enhance using plant-specific color indices")
        ttk.Label(veg_row, text="Type:").pack(side="left", padx=(8, 2))
        combo_veg = ttk.Combobox(veg_row, width=6, state="readonly", textvariable=self.veg_index_type,
                                  values=("ExG", "GRVI", "VARI", "TGI", "GLI"))
        combo_veg.pack(side="left")
        ToolTip(combo_veg, "ExG: Excess Green\nGRVI: Green-Red Index\nVARI: Visible Atm. Resistant\nTGI: Triangular Green\nGLI: Green Leaf Index")
        ttk.Label(veg_row, text="Blend:").pack(side="left", padx=(8, 2))
        ttk.Scale(veg_row, from_=0.0, to=1.0, variable=self.veg_index_blend, orient="horizontal", length=80).pack(side="left")

        # --- White Balance ---
        self.use_white_balance = tk.BooleanVar(value=False)
        self.white_balance_type = tk.StringVar(value="grayworld")

        wb_row = ttk.Frame(sec2b)
        wb_row.pack(fill="x", pady=(0, 4))
        chk_wb = ttk.Checkbutton(wb_row, text="⚪ White Balance", variable=self.use_white_balance)
        chk_wb.pack(side="left")
        ToolTip(chk_wb, "Correct color cast for consistent colors")
        ttk.Label(wb_row, text="Method:").pack(side="left", padx=(8, 2))
        combo_wb = ttk.Combobox(wb_row, width=10, state="readonly", textvariable=self.white_balance_type,
                                 values=("grayworld", "max_white"))
        combo_wb.pack(side="left")
        ToolTip(combo_wb, "grayworld: Assumes avg should be gray\nmax_white: Assumes brightest is white")

        # --- Retinex (Illumination Correction) ---
        self.use_retinex = tk.BooleanVar(value=False)
        self.retinex_type = tk.StringVar(value="multi")
        self.retinex_sigma = tk.IntVar(value=80)

        ret_row = ttk.Frame(sec2b)
        ret_row.pack(fill="x", pady=(0, 4))
        chk_ret = ttk.Checkbutton(ret_row, text="☀️ Retinex", variable=self.use_retinex)
        chk_ret.pack(side="left")
        ToolTip(chk_ret, "Remove illumination effects - great for uneven lighting")
        ttk.Label(ret_row, text="Type:").pack(side="left", padx=(8, 2))
        combo_ret = ttk.Combobox(ret_row, width=8, state="readonly", textvariable=self.retinex_type,
                                  values=("single", "multi"))
        combo_ret.pack(side="left")
        ToolTip(combo_ret, "single: Single-scale (faster)\nmulti: Multi-scale (better)")
        ttk.Label(ret_row, text="σ:").pack(side="left", padx=(8, 2))
        ttk.Entry(ret_row, width=4, textvariable=self.retinex_sigma).pack(side="left")

        # --- LAB Color Enhancement ---
        self.use_lab = tk.BooleanVar(value=False)
        self.lab_l_factor = tk.DoubleVar(value=1.0)
        self.lab_a_shift = tk.IntVar(value=-10)

        lab_row = ttk.Frame(sec2b)
        lab_row.pack(fill="x", pady=(0, 4))
        chk_lab = ttk.Checkbutton(lab_row, text="🎨 LAB Enhance", variable=self.use_lab)
        chk_lab.pack(side="left")
        ToolTip(chk_lab, "Enhance in LAB color space (a-channel controls green)")
        ttk.Label(lab_row, text="L×:").pack(side="left", padx=(8, 2))
        ttk.Entry(lab_row, width=4, textvariable=self.lab_l_factor).pack(side="left")
        ttk.Label(lab_row, text="a+:").pack(side="left", padx=(8, 2))
        ttk.Entry(lab_row, width=4, textvariable=self.lab_a_shift).pack(side="left")
        ToolTip(ttk.Label(lab_row, text="(-=green)"), "Negative values enhance green")

        # --- Second row of advanced options ---
        # --- NLM Denoising ---
        self.use_nlm = tk.BooleanVar(value=False)
        self.nlm_h = tk.IntVar(value=10)

        nlm_row = ttk.Frame(sec2b)
        nlm_row.pack(fill="x", pady=(0, 4))
        chk_nlm = ttk.Checkbutton(nlm_row, text="🔇 NLM Denoise", variable=self.use_nlm)
        chk_nlm.pack(side="left")
        ToolTip(chk_nlm, "Non-local means denoising (better edge preservation)")
        ttk.Label(nlm_row, text="Strength:").pack(side="left", padx=(8, 2))
        ttk.Scale(nlm_row, from_=1, to=30, variable=self.nlm_h, orient="horizontal", length=80).pack(side="left")

        # --- Morphological Top-hat ---
        self.use_tophat = tk.BooleanVar(value=False)
        self.tophat_size = tk.IntVar(value=50)

        th_row = ttk.Frame(sec2b)
        th_row.pack(fill="x", pady=(0, 4))
        chk_th = ttk.Checkbutton(th_row, text="🎩 Top-hat", variable=self.use_tophat)
        chk_th.pack(side="left")
        ToolTip(chk_th, "Morphological illumination normalization")
        ttk.Label(th_row, text="Kernel:").pack(side="left", padx=(8, 2))
        ttk.Entry(th_row, width=4, textvariable=self.tophat_size).pack(side="left")

        # --- Guided Filter ---
        self.use_guided = tk.BooleanVar(value=False)
        self.guided_radius = tk.IntVar(value=8)
        self.guided_eps = tk.DoubleVar(value=0.04)

        gf_row = ttk.Frame(sec2b)
        gf_row.pack(fill="x", pady=(0, 4))
        chk_gf = ttk.Checkbutton(gf_row, text="🎯 Guided Filter", variable=self.use_guided)
        chk_gf.pack(side="left")
        ToolTip(chk_gf, "Edge-preserving smoothing (better than bilateral)")
        ttk.Label(gf_row, text="Radius:").pack(side="left", padx=(8, 2))
        ttk.Entry(gf_row, width=3, textvariable=self.guided_radius).pack(side="left")
        ttk.Label(gf_row, text="ε:").pack(side="left", padx=(8, 2))
        ttk.Entry(gf_row, width=5, textvariable=self.guided_eps).pack(side="left")

        # --- Difference of Gaussians ---
        self.use_dog = tk.BooleanVar(value=False)
        self.dog_sigma1 = tk.DoubleVar(value=1.0)
        self.dog_sigma2 = tk.DoubleVar(value=3.0)
        self.dog_blend = tk.DoubleVar(value=0.3)

        dog_row = ttk.Frame(sec2b)
        dog_row.pack(fill="x", pady=(0, 4))
        chk_dog = ttk.Checkbutton(dog_row, text="🔍 DoG Edge", variable=self.use_dog)
        chk_dog.pack(side="left")
        ToolTip(chk_dog, "Difference of Gaussians edge enhancement")
        ttk.Label(dog_row, text="σ1:").pack(side="left", padx=(8, 2))
        ttk.Entry(dog_row, width=4, textvariable=self.dog_sigma1).pack(side="left")
        ttk.Label(dog_row, text="σ2:").pack(side="left", padx=(4, 2))
        ttk.Entry(dog_row, width=4, textvariable=self.dog_sigma2).pack(side="left")
        ttk.Label(dog_row, text="Blend:").pack(side="left", padx=(4, 2))
        ttk.Entry(dog_row, width=4, textvariable=self.dog_blend).pack(side="left")

        # --- Shadow/Highlight Correction ---
        self.use_shadow_highlight = tk.BooleanVar(value=False)
        self.shadow_amount = tk.DoubleVar(value=0.3)
        self.highlight_amount = tk.DoubleVar(value=0.3)

        sh_row = ttk.Frame(sec2b)
        sh_row.pack(fill="x", pady=(0, 4))
        chk_sh = ttk.Checkbutton(sh_row, text="🌓 Shadow/Highlight", variable=self.use_shadow_highlight)
        chk_sh.pack(side="left")
        ToolTip(chk_sh, "Lift shadows and reduce highlights")
        ttk.Label(sh_row, text="Shadow:").pack(side="left", padx=(8, 2))
        ttk.Scale(sh_row, from_=0.0, to=1.0, variable=self.shadow_amount, orient="horizontal", length=60).pack(side="left")
        ttk.Label(sh_row, text="Highlight:").pack(side="left", padx=(4, 2))
        ttk.Scale(sh_row, from_=0.0, to=1.0, variable=self.highlight_amount, orient="horizontal", length=60).pack(side="left")

        # --- Local Contrast ---
        self.use_local_contrast = tk.BooleanVar(value=False)
        self.local_contrast_size = tk.IntVar(value=31)

        lc_row = ttk.Frame(sec2b)
        lc_row.pack(fill="x", pady=(0, 4))
        chk_lc = ttk.Checkbutton(lc_row, text="📊 Local Contrast", variable=self.use_local_contrast)
        chk_lc.pack(side="left")
        ToolTip(chk_lc, "Normalize local contrast for detail enhancement")
        ttk.Label(lc_row, text="Window:").pack(side="left", padx=(8, 2))
        ttk.Entry(lc_row, width=4, textvariable=self.local_contrast_size).pack(side="left")

        # --- Adaptive Gamma ---
        self.use_adaptive_gamma = tk.BooleanVar(value=False)

        ag_row = ttk.Frame(sec2b)
        ag_row.pack(fill="x", pady=(0, 2))
        chk_ag = ttk.Checkbutton(ag_row, text="🌈 Adaptive Gamma", variable=self.use_adaptive_gamma)
        chk_ag.pack(side="left")
        ToolTip(chk_ag, "Automatically adjust gamma based on image brightness")

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 3: SAM2 Parameters
        # ═══════════════════════════════════════════════════════════════════════
        sec3 = ttk.LabelFrame(container, text=f"  {self.icons['sam']} SAM2 Parameters  ", padding=(10, 8), style='Options.TLabelframe')
        self._add_left_pane(container, sec3, weight=1, fill="x", expand=False)

        self.m_points_per_side  = tk.IntVar(value=16)
        self.m_points_per_batch = tk.IntVar(value=16)
        self.m_pred_iou_thresh  = tk.DoubleVar(value=0.90)
        self.m_stability_score_thresh = tk.DoubleVar(value=0.80)
        self.m_crop_n_layers    = tk.IntVar(value=1)
        self.m_crop_overlap_ratio = tk.DoubleVar(value=0.30)
        self.m_crop_n_points_downscale_factor = tk.IntVar(value=2)
        self.m_box_nms_thresh   = tk.DoubleVar(value=0.60)
        self.m_min_mask_region_area = tk.IntVar(value=800)
        self.m_use_m2m          = tk.BooleanVar(value=True)
        self.m_output_mode      = tk.StringVar(value="binary_mask")

        # SAM2 params in clean rows
        sam_params = [
            [("pts/side", self.m_points_per_side), ("pts/batch", self.m_points_per_batch)],
            [("IoU thresh", self.m_pred_iou_thresh), ("Stability", self.m_stability_score_thresh)],
            [("Crop layers", self.m_crop_n_layers), ("Overlap", self.m_crop_overlap_ratio)],
            [("NMS thresh", self.m_box_nms_thresh), ("Min area", self.m_min_mask_region_area)],
        ]

        for row_data in sam_params:
            row = ttk.Frame(sec3)
            row.pack(fill="x", pady=1)
            for lbl, var in row_data:
                ttk.Label(row, text=lbl, width=10).pack(side="left")
                ttk.Entry(row, width=6, textvariable=var).pack(side="left", padx=(0, 12))

        # Checkbox row
        opt_row = ttk.Frame(sec3)
        opt_row.pack(fill="x", pady=(4, 0))
        ttk.Checkbutton(opt_row, text="use_m2m", variable=self.m_use_m2m).pack(side="left")
        ttk.Label(opt_row, text="Output:").pack(side="left", padx=(12, 4))
        ttk.Combobox(opt_row, width=12, state="readonly", textvariable=self.m_output_mode,
                     values=("binary_mask", "coco_rle", "uncompressed_rle", "polygons")).pack(side="left")
        ttk.Button(opt_row, text="?", width=2, command=self.explain_mask_params).pack(side="left", padx=(8, 0))

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 5: Phenotypes
        # ═══════════════════════════════════════════════════════════════════════
        sec5 = ttk.LabelFrame(container, text=f"  {self.icons['phenotype']} Phenotypes  ", padding=(10, 8), style='Options.TLabelframe')
        self._add_left_pane(container, sec5, weight=1, fill="x", expand=False)

        self.ph_all    = tk.BooleanVar(value=True)
        self.ph_area   = tk.BooleanVar(value=True)
        self.ph_len    = tk.BooleanVar(value=True)
        self.ph_wid    = tk.BooleanVar(value=True)
        self.ph_color  = tk.BooleanVar(value=True)
        self.ph_hsv    = tk.BooleanVar(value=True)
        self.ph_shape  = tk.BooleanVar(value=True)   # perimeter/hull/solidity/extent/circularity/eq_diam
        self.ph_comp   = tk.BooleanVar(value=True)   # component count
        self.ph_veg    = tk.BooleanVar(value=True)   # vegetation indices + green fraction
        self.ph_hsvvar = tk.BooleanVar(value=True)   # hue/sat variance
        self.ph_none   = tk.BooleanVar(value=False)  # select none

        def _sync_ph(*_):
            if self.ph_none.get():
                # None overrides everything else
                self.ph_all.set(False)
                self.ph_area.set(False); self.ph_len.set(False); self.ph_wid.set(False)
                self.ph_color.set(False); self.ph_hsv.set(False)
                self.ph_shape.set(False); self.ph_comp.set(False); self.ph_veg.set(False); self.ph_hsvvar.set(False)
                return
            if self.ph_all.get():
                self.ph_area.set(True); self.ph_len.set(True); self.ph_wid.set(True)
                self.ph_color.set(True); self.ph_hsv.set(True)
                self.ph_shape.set(True); self.ph_comp.set(True); self.ph_veg.set(True); self.ph_hsvvar.set(True)
                self.ph_none.set(False)
            else:
                if all(v.get() for v in (self.ph_area, self.ph_len, self.ph_wid, self.ph_color,
                                         self.ph_hsv, self.ph_shape, self.ph_comp, self.ph_veg, self.ph_hsvvar)):
                    self.ph_all.set(True)

        ph_checks = ttk.Frame(sec5)
        ph_checks.pack(fill="x")

        row1 = ttk.Frame(ph_checks)
        row1.pack(fill="x")
        ttk.Checkbutton(row1, text="All", variable=self.ph_all, command=_sync_ph).pack(side="left")
        ttk.Checkbutton(row1, text="None", variable=self.ph_none, command=_sync_ph).pack(side="left", padx=(6, 0))
        for txt, var in [("Area", self.ph_area), ("Length", self.ph_len), ("Width", self.ph_wid), ("Color", self.ph_color)]:
            ttk.Checkbutton(row1, text=txt, variable=var,
                           command=lambda: (self.ph_all.set(False), self.ph_none.set(False))).pack(side="left", padx=(6, 0))

        row2 = ttk.Frame(ph_checks)
        row2.pack(fill="x", pady=(4, 0))
        for txt, var in [("HSV", self.ph_hsv), ("Shape", self.ph_shape), ("Components", self.ph_comp),
                         ("VegIdx", self.ph_veg), ("HSV Var", self.ph_hsvvar)]:
            ttk.Checkbutton(row2, text=txt, variable=var,
                           command=lambda: (self.ph_all.set(False), self.ph_none.set(False))).pack(side="left", padx=(6, 0))

        ph_help = ttk.Frame(sec5)
        ph_help.pack(fill="x", pady=(6, 0))
        ttk.Button(ph_help, text="?", width=2, command=self.explain_phenotypes).pack(side="right")

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 4: Actions (moved below Phenotypes)
        # ═══════════════════════════════════════════════════════════════════════
        sec4 = ttk.LabelFrame(container, text=f"  {self.icons['action']} Actions  ", padding=(10, 8), style='TLabelframe')
        self._add_left_pane(container, sec4, weight=1, fill="x", expand=False)

        # Main action buttons
        action_row = ttk.Frame(sec4)
        action_row.pack(fill="x", pady=(0, 8))
        btn_preview = ttk.Button(action_row, text="👁 Preview Enhance", command=self.preview_enhance)
        btn_preview.pack(side="left")
        ToolTip(btn_preview, "Preview enhancement settings (Ctrl+E)")
        btn_segment = ttk.Button(action_row, text="✂️ Segment", command=self.segment, style='Accent.TButton')
        btn_segment.pack(side="left", padx=(8, 0))
        ToolTip(btn_segment, "Run SAM2 segmentation (Ctrl+Enter)")
        btn_segment_all = ttk.Button(action_row, text="Segment ALL…", command=self.segment_all_batch)
        btn_segment_all.pack(side="left", padx=(8, 0))
        ToolTip(btn_segment_all, "Run SAM2 on all images in the opened folder")

        # Save buttons
        save_row = ttk.Frame(sec4)
        save_row.pack(fill="x", pady=(0, 4))
        btn_save_all = ttk.Button(save_row, text="💾 Save ALL…", command=self.save_all_masks, style='Secondary.TButton')
        btn_save_all.pack(side="left")
        ToolTip(btn_save_all, "Save all masks to a folder")
        btn_save_sel = ttk.Button(save_row, text="💾 Save Selected…", command=self.save_selected_masks, style='Secondary.TButton')
        btn_save_sel.pack(side="left", padx=(8, 0))
        ToolTip(btn_save_sel, "Save selected masks only (Ctrl+S)")
        btn_save_batch = ttk.Button(save_row, text="💾 Save Batch…", command=self.save_all_batch_results, style='Secondary.TButton')
        btn_save_batch.pack(side="left", padx=(8, 0))
        ToolTip(btn_save_batch, "Save cached batch masks to a folder")
        btn_save_out = ttk.Button(save_row, text="📊 Outputs…", command=self.save_all_outputs, style='Secondary.TButton')
        btn_save_out.pack(side="left", padx=(8, 0))
        ToolTip(btn_save_out, "Save masks with phenotype CSV")
        btn_load = ttk.Button(save_row, text="📂 Load…", command=self.load_masks, style='Secondary.TButton')
        btn_load.pack(side="left", padx=(8, 0))
        ToolTip(btn_load, "Load previously saved masks")

        # Settings save/load row
        settings_row = ttk.Frame(sec4)
        settings_row.pack(fill="x", pady=(4, 0))
        ttk.Label(settings_row, text="Settings:", width=8).pack(side="left")
        btn_save_settings = ttk.Button(settings_row, text="⬇ Save…", command=self.save_settings, style='Secondary.TButton')
        btn_save_settings.pack(side="left", padx=(4, 0))
        ToolTip(btn_save_settings, "Save all enhancement parameters to a JSON file")
        btn_load_settings = ttk.Button(settings_row, text="⬆ Load…", command=self.load_settings, style='Secondary.TButton')
        btn_load_settings.pack(side="left", padx=(8, 0))
        ToolTip(btn_load_settings, "Load enhancement parameters from a JSON file")
        btn_reset_settings = ttk.Button(settings_row, text="↺ Reset", command=self.reset_settings, style='Secondary.TButton')
        btn_reset_settings.pack(side="left", padx=(8, 0))
        ToolTip(btn_reset_settings, "Reset all parameters to defaults")

        # TIF Conversion utility row
        convert_row = ttk.Frame(sec4)
        convert_row.pack(fill="x", pady=(8, 0))
        ttk.Label(convert_row, text="Convert:", width=8).pack(side="left")
        btn_convert_tif = ttk.Button(convert_row, text="📁 TIF → …", command=self._convert_tif_folder, style='Secondary.TButton')
        btn_convert_tif.pack(side="left", padx=(4, 0))
        ToolTip(btn_convert_tif, "Convert all TIF/TIFF files in a folder to PNG or JPEG")
        ttk.Label(convert_row, text="Format:").pack(side="left", padx=(12, 4))
        self._convert_format = tk.StringVar(value="PNG")
        fmt_combo = ttk.Combobox(convert_row, textvariable=self._convert_format, values=["PNG", "JPEG"], width=6, state="readonly")
        fmt_combo.pack(side="left")
        ToolTip(fmt_combo, "Output format (PNG=lossless, JPEG=smaller)")
        ttk.Label(convert_row, text="Max size:").pack(side="left", padx=(12, 4))
        self._convert_max_size = tk.StringVar(value="0")
        size_entry = ttk.Entry(convert_row, textvariable=self._convert_max_size, width=6)
        size_entry.pack(side="left")
        ToolTip(size_entry, "Max dimension in pixels (0=no resize)")



        
    def make_preview_frame(self, paned_parent):
        # Preview frame - added to PanedWindow
        f = ttk.LabelFrame(paned_parent, text=f"  {self.icons['preview']} Preview  ", padding=(5, 5))
        paned_parent.add(f, weight=3)  # Give preview more weight

        # Toolbar with modern styling
        bar = ttk.Frame(f)
        bar.pack(fill="x", pady=(0, 4))

        # Zoom controls
        btn_zoom_out = ttk.Button(bar, text="−", width=2, command=lambda: self._zoom_by(0.8), style='Icon.TButton')
        btn_zoom_out.pack(side="left")
        ToolTip(btn_zoom_out, "Zoom out (Ctrl+-)")
        btn_fit = ttk.Button(bar, text="Fit", width=3, command=self._zoom_fit, style='Icon.TButton')
        btn_fit.pack(side="left", padx=2)
        ToolTip(btn_fit, "Fit to window (Ctrl+0)")
        btn_zoom_in = ttk.Button(bar, text="+", width=2, command=lambda: self._zoom_by(1.25), style='Icon.TButton')
        btn_zoom_in.pack(side="left")
        ToolTip(btn_zoom_in, "Zoom in (Ctrl++)")

        ttk.Separator(bar, orient="vertical").pack(side="left", padx=6, fill="y")

        # Crop tools
        ttk.Checkbutton(bar, text="Crop", style="Toolbutton",
                        variable=self._crop_mode, command=self._set_crop_mode).pack(side="left")
        self._btn_crop_apply  = ttk.Button(bar, text="Apply", width=5, command=self._apply_crop, state="disabled")
        self._btn_crop_cancel = ttk.Button(bar, text="Cancel", width=5, command=self._cancel_crop, state="disabled")
        self._btn_crop_apply.pack(side="left", padx=(4, 2))
        self._btn_crop_cancel.pack(side="left")

        ttk.Separator(bar, orient="vertical").pack(side="left", padx=6, fill="y")

        # Pick mode toggle (for selecting masks directly on the preview)
        ttk.Checkbutton(
            bar,
            text="Pick",
            style="Toolbutton",
            variable=self._pick_mode,
            command=self._toggle_pick_mode,
        ).pack(side="left")

        ttk.Separator(bar, orient="vertical").pack(side="left", padx=6, fill="y")

        # Navigation
        ttk.Button(bar, text="◀", width=2, command=self.prev_image).pack(side="left")
        ttk.Button(bar, text="▶", width=2, command=self.next_image).pack(side="left", padx=(2, 0))
        self._batch_status = ttk.Label(bar, text="", width=8, anchor="w")
        self._batch_status.pack(side="left", padx=(6, 0))

        # Canvas
        self.canvas = tk.Canvas(f, width=500, height=400, bg=self.colors['canvas_bg'], highlightthickness=0, cursor="tcross")
        self.canvas.pack(fill="both", expand=True)

        # Pick controls bar
        self.pickbar = ttk.Frame(f)
        self.pickbar.pack(fill="x", pady=(4, 0))

        ttk.Button(self.pickbar, text="Deselect", width=7,
                   command=lambda: self._set_edit_mode("deselect")).pack(side="left")
        ttk.Button(self.pickbar, text="Select", width=7,
                   command=lambda: self._set_edit_mode("select")).pack(side="left", padx=(4, 0))
        ttk.Button(self.pickbar, text="Reset", width=5,
                   command=self._reset_pick).pack(side="left", padx=(8, 0))
        ttk.Button(self.pickbar, text="Apply", width=5,
                   command=self._apply_pick).pack(side="left", padx=(4, 0))
        ttk.Button(self.pickbar, text="Combine", width=6,
                   command=self._apply_combine_from_picks).pack(side="left", padx=(4, 0))

        self._pick_status = ttk.Label(self.pickbar, text="", anchor="w")
        self._pick_status.pack(side="left", padx=(8, 0))

        # Mouse bindings for zoom/pan
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._on_wheel(e, delta=+120))
        self.canvas.bind("<Button-5>", lambda e: self._on_wheel(e, delta=-120))
        self.canvas.bind("<ButtonPress-1>", self._pan_start)
        self.canvas.bind("<B1-Motion>", self._pan_move)

        # Pick mode state
        self._edit_mode = tk.StringVar(value="none")
        self._picks: set[int] = set()
        self._picks_action: str | None = None

        # Re-render on resize
        self.canvas.bind("<Configure>", lambda e: self._render_preview())

        # Ensure correct mouse behavior
        self._bind_canvas_events()

    # ---------- Crop: mode switching & bindings ----------
    def _set_crop_mode(self):
        """Toggle crop tool and (re)bind canvas events."""
        on = bool(self._crop_mode.get())
        # buttons reflect selection state only when in crop mode
        self._update_crop_buttons()
        # clear any previous overlay when toggling
        if not on:
            self._clear_crop_overlay()
        # rebind LMB behaviour
        self._bind_canvas_events()

    def _bind_canvas_events(self):
        # Clear old bindings
        for seq in ("<ButtonPress-1>", "<B1-Motion>", "<ButtonRelease-1>",
                "<Motion>", "<Leave>", "<MouseWheel>", "<Button-4>", "<Button-5>"):
            try:
                self.canvas.unbind(seq)
            except Exception:
                pass

        # 1) Pick mode (our click-to-select/deselect)
        if getattr(self, "_edit_mode", None) and self._edit_mode.get() != "none":
            self.canvas.bind("<ButtonPress-1>", self._on_pick_click)
            self.canvas.configure(cursor="hand2")
            return

        # 2) Crop mode
        if getattr(self, "_crop_mode", None) and self._crop_mode.get():
            self.canvas.bind("<ButtonPress-1>", self._crop_start)
            self.canvas.bind("<B1-Motion>",    self._crop_drag)
            self.canvas.bind("<ButtonRelease-1>", self._crop_end)
            self.canvas.configure(cursor="tcross")
            return

        # 3) Default: pan + wheel zoom
        self.canvas.bind("<ButtonPress-1>", self._pan_start)
        self.canvas.bind("<B1-Motion>",     self._pan_move)
        self.canvas.bind("<MouseWheel>",    lambda e: self._on_wheel(e))
        self.canvas.bind("<Button-4>",      lambda e: self._on_wheel(e, +120))  # Linux
        self.canvas.bind("<Button-5>",      lambda e: self._on_wheel(e, -120))  # Linux
        self.canvas.configure(cursor="")




    def make_masks_frame(self, paned_parent):
        # Masks frame - added to PanedWindow below Preview
        f = ttk.LabelFrame(paned_parent, text=f"  {self.icons['masks']} Masks  ", padding=(5, 5))
        paned_parent.add(f, weight=1)

        # Toolbar with modern buttons
        bar = ttk.Frame(f)
        bar.pack(fill="x", pady=(0, 4))

        btn_delete = ttk.Button(bar, text="🗑", width=3, command=self.delete_selected_masks, style='Icon.TButton')
        btn_delete.pack(side="left")
        ToolTip(btn_delete, "Delete selected masks (Del)")
        btn_clear = ttk.Button(bar, text="✖", width=3, command=self.clear_all_masks, style='Icon.TButton')
        btn_clear.pack(side="left", padx=(4, 0))
        ToolTip(btn_clear, "Clear all masks")
        btn_combine = ttk.Button(bar, text="🔗", width=3, command=self.combine_selected_masks, style='Icon.TButton')
        btn_combine.pack(side="left", padx=(4, 0))
        ToolTip(btn_combine, "Combine selected masks into one")
        btn_refine = ttk.Button(bar, text="🔍", width=3, command=self.refine_selected_masks, style='Icon.TButton')
        btn_refine.pack(side="left", padx=(4, 0))
        ToolTip(btn_refine, "Re-segment within selected mask regions")
        btn_extend = ttk.Button(bar, text="↗", width=3, command=self.on_predict_extend, style='Icon.TButton')
        btn_extend.pack(side="left", padx=(4, 0))
        ToolTip(btn_extend, "Extend/predict occluded parts of mask")

        ttk.Separator(bar, orient="vertical").pack(side="left", padx=6, fill="y")

        ttk.Label(bar, text="Extend mode:").pack(side="left", padx=(0, 4))
        self.extend_mode = tk.StringVar(value="auto")
        combo_mode = ttk.Combobox(bar, width=8, state="readonly",
                     textvariable=self.extend_mode,
                     values=("auto", "rosette", "blade", "ml"))
        combo_mode.pack(side="left")
        ToolTip(combo_mode, "auto: detect shape type\nrosette: for circular plants\nblade: for elongated leaves\nml: use trained SAM2 weights")

        # Listbox with modern styling and scrollbar
        wrap = ttk.Frame(f)
        wrap.pack(fill="both", expand=True)

        self.lb = tk.Listbox(wrap, width=30, height=12, selectmode="extended",
                             bg=self.colors['canvas_bg'], fg=self.colors['text_light'],
                             selectbackground=self.colors['accent'],
                             selectforeground=self.colors['text_light'],
                             highlightthickness=0,
                             bd=0,
                             font=("Menlo", 10))
        self.lb.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.lb.yview)
        sb.pack(side="right", fill="y")
        self.lb.config(yscrollcommand=sb.set)

        # Bindings
        self.lb.bind("<<ListboxSelect>>", self.on_select_mask)
        self.lb.bind("<Delete>", lambda e: self.delete_selected_masks())
        self.lb.bind("<BackSpace>", lambda e: self.delete_selected_masks())
        self.lb.bind("<Control-a>", lambda e: (self.lb.select_set(0, tk.END), "break"))
        self.lb.bind("<Command-a>", lambda e: (self.lb.select_set(0, tk.END), "break"))
        # Toggle-select with Ctrl/Cmd click (multi-select without clearing)
        self.lb.bind("<Control-Button-1>", self._toggle_listbox_selection)
        self.lb.bind("<Command-Button-1>", self._toggle_listbox_selection)

    def make_training_frame(self, parent):
        """Create training panel with two tabs: Fine-tune and Hidden Structure."""
        tf = ttk.LabelFrame(parent, text=f"  {self.icons['training']} Training  ", padding=(10, 5))

        # Create notebook for tabs
        notebook = ttk.Notebook(tf)
        notebook.pack(fill="both", expand=True)

        # ════════════════════════════════════════════════════════════════════
        # TAB 1: Fine-tune (existing functionality)
        # ════════════════════════════════════════════════════════════════════
        tab1 = ttk.Frame(notebook, padding=8)
        notebook.add(tab1, text="  Fine-tune  ")

        # --- dataset root
        r = 0
        ttk.Label(tab1, text="Dataset folder:").grid(row=r, column=0, sticky="w")
        self.train_root_var = tk.StringVar(value="")
        ttk.Entry(tab1, textvariable=self.train_root_var, width=64).grid(row=r, column=1, sticky="ew", padx=4)
        ttk.Button(tab1, text="Choose...", command=self._pick_train_root).grid(row=r, column=2)
        tab1.grid_columnconfigure(1, weight=1)

        # --- example collection bar
        r += 1
        self.train_msg = ttk.Label(tab1, text="0 examples", anchor="w")
        self.train_msg.grid(row=r, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(tab1, text="Prompt me after each Segment", variable=self.train_auto_prompt).grid(row=r, column=2, sticky="e")

        r += 1
        bar = ttk.Frame(tab1); bar.grid(row=r, column=0, columnspan=3, sticky="w", pady=(2,4))
        ttk.Button(bar, text="Add current segmentation as example", command=self._add_current_to_training).pack(side="left")
        ttk.Button(bar, text="Import RGBA crops…", command=self._import_rgba_crops).pack(side="left", padx=(8,0))
        ttk.Button(bar, text="Open dataset folder", command=self._open_train_root).pack(side="left", padx=(8,0))
        ttk.Button(bar, text="Clear examples", command=self._clear_training_set).pack(side="left", padx=(8,0))

        # --- training params
        r += 1
        grid = ttk.Frame(tab1); grid.grid(row=r, column=0, columnspan=3, sticky="ew")
        ttk.Label(grid, text="Checkpoint (.pt)").grid(row=0, column=0, sticky="e")
        ttk.Entry(grid, textvariable=self.train_ckpt_var, width=48).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(grid, text="...", command=lambda: self._browse_file_into(self.train_ckpt_var)).grid(row=0, column=2)

        ttk.Label(grid, text="Config (.yaml)").grid(row=1, column=0, sticky="e")
        ttk.Entry(grid, textvariable=self.train_cfg_var, width=48).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(grid, text="...", command=lambda: self._browse_file_into(self.train_cfg_var, ("YAML","*.yaml *.yml"))).grid(row=1, column=2)

        ttk.Label(grid, text="Save to (.pth)").grid(row=2, column=0, sticky="e")
        ttk.Entry(grid, textvariable=self.train_out_var, width=48).grid(row=2, column=1, sticky="ew", padx=4)
        ttk.Button(grid, text="...", command=lambda: self._browse_save_into(self.train_out_var, default_ext=".pth")).grid(row=2, column=2)

        ttk.Label(grid, text="Steps").grid(row=3, column=0, sticky="e")
        ttk.Spinbox(grid, from_=100, to=200000, increment=100, textvariable=self.train_steps_var, width=10).grid(row=3, column=1, sticky="w", padx=4)
        ttk.Label(grid, text="LR").grid(row=3, column=2, sticky="e")
        ttk.Entry(grid, textvariable=self.train_lr_var, width=10).grid(row=3, column=3, sticky="w", padx=4)

        ttk.Label(grid, text="Image size").grid(row=4, column=0, sticky="e")
        ttk.Spinbox(grid, from_=256, to=2048, increment=128, textvariable=self.train_size_var, width=10).grid(row=4, column=1, sticky="w", padx=4)
        ttk.Label(grid, text="Device").grid(row=4, column=2, sticky="e")
        ttk.Combobox(grid, textvariable=self.train_device_var, values=["mps", "cuda", "cpu"], width=8).grid(row=4, column=3, sticky="w", padx=4)
        grid.grid_columnconfigure(1, weight=1)

        # --- actions
        r += 1
        tbar = ttk.Frame(tab1); tbar.grid(row=r, column=0, columnspan=3, sticky="w", pady=(4,0))
        ttk.Button(tbar, text="Train NOW", command=self._launch_training).pack(side="left")
        ttk.Button(tbar, text="Load fine-tuned into predictor", command=self._load_finetuned_into_predictor).pack(side="left", padx=(8,0))

        # ════════════════════════════════════════════════════════════════════
        # TAB 2: Hidden Structure (Occlusion Augmentation)
        # ════════════════════════════════════════════════════════════════════
        tab2 = ttk.Frame(notebook, padding=8)
        notebook.add(tab2, text="  Hidden Structure  ")

        # Description
        desc = ttk.Label(tab2, text="Train the model to predict complete leaf shapes from partially occluded views.\n"
                                    "Uses synthetic occlusions during training to learn hidden structure.",
                         wraplength=700, justify="left")
        desc.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        # --- Data folder (folder of leaf images)
        r = 1
        ttk.Label(tab2, text="Leaf images folder:").grid(row=r, column=0, sticky="e", padx=(0,4))
        ttk.Entry(tab2, textvariable=self.occ_data_var, width=50).grid(row=r, column=1, sticky="ew", padx=4)
        ttk.Button(tab2, text="Choose...", command=self._pick_occ_data_folder).grid(row=r, column=2)
        tab2.grid_columnconfigure(1, weight=1)

        # --- Checkpoint
        r += 1
        ttk.Label(tab2, text="Base checkpoint (.pt):").grid(row=r, column=0, sticky="e", padx=(0,4))
        ttk.Entry(tab2, textvariable=self.occ_ckpt_var, width=50).grid(row=r, column=1, sticky="ew", padx=4)
        ttk.Button(tab2, text="...", command=lambda: self._browse_file_into(self.occ_ckpt_var)).grid(row=r, column=2)

        # --- Config
        r += 1
        ttk.Label(tab2, text="Config (.yaml):").grid(row=r, column=0, sticky="e", padx=(0,4))
        ttk.Entry(tab2, textvariable=self.occ_cfg_var, width=50).grid(row=r, column=1, sticky="ew", padx=4)
        ttk.Button(tab2, text="...", command=lambda: self._browse_file_into(self.occ_cfg_var, ("YAML","*.yaml *.yml"))).grid(row=r, column=2)

        # --- Output path
        r += 1
        ttk.Label(tab2, text="Save to (.pth):").grid(row=r, column=0, sticky="e", padx=(0,4))
        ttk.Entry(tab2, textvariable=self.occ_out_var, width=50).grid(row=r, column=1, sticky="ew", padx=4)
        ttk.Button(tab2, text="...", command=lambda: self._browse_save_into(self.occ_out_var, default_ext=".pth")).grid(row=r, column=2)

        # --- Training parameters (two columns)
        r += 1
        params_frame = ttk.LabelFrame(tab2, text=" Training Parameters ", padding=8)
        params_frame.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(8,4))

        # Left column
        ttk.Label(params_frame, text="Image size:").grid(row=0, column=0, sticky="e", padx=(0,4))
        ttk.Spinbox(params_frame, from_=256, to=1024, increment=64, textvariable=self.occ_size_var, width=8).grid(row=0, column=1, sticky="w")

        ttk.Label(params_frame, text="Steps:").grid(row=1, column=0, sticky="e", padx=(0,4))
        ttk.Spinbox(params_frame, from_=1000, to=100000, increment=1000, textvariable=self.occ_steps_var, width=8).grid(row=1, column=1, sticky="w")

        ttk.Label(params_frame, text="Learning rate:").grid(row=2, column=0, sticky="e", padx=(0,4))
        ttk.Entry(params_frame, textvariable=self.occ_lr_var, width=10).grid(row=2, column=1, sticky="w")

        ttk.Label(params_frame, text="Device:").grid(row=3, column=0, sticky="e", padx=(0,4))
        dev_combo = ttk.Combobox(params_frame, textvariable=self.occ_device_var, values=["cuda", "mps", "cpu"], width=8)
        dev_combo.grid(row=3, column=1, sticky="w")

        # Right column - Occlusion parameters
        ttk.Label(params_frame, text="Min occlusion %:").grid(row=0, column=2, sticky="e", padx=(20,4))
        ttk.Scale(params_frame, from_=0.05, to=0.50, variable=self.occ_min_var, orient="horizontal", length=100).grid(row=0, column=3, sticky="w")
        self.occ_min_lbl = ttk.Label(params_frame, text="15%", width=5)
        self.occ_min_lbl.grid(row=0, column=4, sticky="w")

        ttk.Label(params_frame, text="Max occlusion %:").grid(row=1, column=2, sticky="e", padx=(20,4))
        ttk.Scale(params_frame, from_=0.20, to=0.80, variable=self.occ_max_var, orient="horizontal", length=100).grid(row=1, column=3, sticky="w")
        self.occ_max_lbl = ttk.Label(params_frame, text="50%", width=5)
        self.occ_max_lbl.grid(row=1, column=4, sticky="w")

        ttk.Label(params_frame, text="Occluder count min:").grid(row=2, column=2, sticky="e", padx=(20,4))
        ttk.Spinbox(params_frame, from_=0, to=5, textvariable=self.occ_count_min_var, width=5).grid(row=2, column=3, sticky="w")

        ttk.Label(params_frame, text="Occluder count max:").grid(row=3, column=2, sticky="e", padx=(20,4))
        ttk.Spinbox(params_frame, from_=1, to=10, textvariable=self.occ_count_max_var, width=5).grid(row=3, column=3, sticky="w")

        # Update occlusion percentage labels
        def update_occ_labels(*_):
            self.occ_min_lbl.configure(text=f"{int(self.occ_min_var.get()*100)}%")
            self.occ_max_lbl.configure(text=f"{int(self.occ_max_var.get()*100)}%")
        self.occ_min_var.trace_add("write", update_occ_labels)
        self.occ_max_var.trace_add("write", update_occ_labels)

        # --- Actions
        r += 1
        occ_bar = ttk.Frame(tab2); occ_bar.grid(row=r, column=0, columnspan=3, sticky="w", pady=(8,0))
        ttk.Button(occ_bar, text="Train Hidden Structure", command=self._launch_occlusion_training).pack(side="left")
        ttk.Button(occ_bar, text="Load trained model", command=self._load_occlusion_model).pack(side="left", padx=(8,0))

        # ════════════════════════════════════════════════════════════════════
        # TAB 3: Target Segment (class-specific fine-tune + filter)
        # ════════════════════════════════════════════════════════════════════
        tab3 = ttk.Frame(notebook, padding=8)
        notebook.add(tab3, text="  Target Segment  ")

        desc3 = ttk.Label(tab3, text="Train a tip-only segmentation model (no SAM) using your saved target masks.\n"
                                     "When enabled, the main Segment / Segment ALL will run this model directly and output only the tip mask.",
                          wraplength=700, justify="left")
        desc3.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        r = 1
        ttk.Label(tab3, text="Dataset folder:").grid(row=r, column=0, sticky="e", padx=(0,4))
        ttk.Entry(tab3, textvariable=self.target_root_var, width=50).grid(row=r, column=1, sticky="ew", padx=4)
        ttk.Button(tab3, text="Choose...", command=self._pick_target_root).grid(row=r, column=2)
        tab3.grid_columnconfigure(1, weight=1)

        r += 1
        self.target_msg = ttk.Label(tab3, text="0 examples", anchor="w")
        self.target_msg.grid(row=r, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(tab3, text="Resume from existing tip model",
                        variable=self.target_resume_var).grid(row=r, column=2, sticky="e")

        r += 1
        tbar = ttk.Frame(tab3); tbar.grid(row=r, column=0, columnspan=3, sticky="w", pady=(2,4))
        ttk.Button(tbar, text="Add selected masks as target", command=self._add_current_to_target).pack(side="left")
        ttk.Button(tbar, text="Mark image as NO target", command=self._add_negative_target).pack(side="left", padx=(8,0))
        ttk.Button(tbar, text="Open dataset folder", command=self._open_target_root).pack(side="left", padx=(8,0))
        ttk.Button(tbar, text="Scan dataset", command=self._scan_target_dataset).pack(side="left", padx=(8,0))
        ttk.Button(tbar, text="Clear target dataset", command=self._clear_target_set).pack(side="left", padx=(8,0))

        # --- training params
        r += 1
        grid3 = ttk.Frame(tab3); grid3.grid(row=r, column=0, columnspan=3, sticky="ew")
        ttk.Label(grid3, text="Save to (.pth)").grid(row=0, column=0, sticky="e")
        ttk.Entry(grid3, textvariable=self.target_out_var, width=48).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(grid3, text="...", command=lambda: self._browse_save_into(self.target_out_var, default_ext=".pth")).grid(row=0, column=2)

        ttk.Label(grid3, text="Steps").grid(row=1, column=0, sticky="e")
        ttk.Spinbox(grid3, from_=100, to=200000, increment=100, textvariable=self.target_steps_var, width=10).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(grid3, text="LR").grid(row=1, column=2, sticky="e")
        ttk.Entry(grid3, textvariable=self.target_lr_var, width=10).grid(row=1, column=3, sticky="w", padx=4)

        ttk.Label(grid3, text="Image size").grid(row=2, column=0, sticky="e")
        ttk.Spinbox(grid3, from_=256, to=2048, increment=128, textvariable=self.target_size_var, width=10).grid(row=2, column=1, sticky="w", padx=4)
        ttk.Label(grid3, text="Device").grid(row=2, column=2, sticky="e")
        ttk.Combobox(grid3, textvariable=self.target_device_var, values=["mps", "cuda", "cpu"], width=8).grid(row=2, column=3, sticky="w", padx=4)

        ttk.Label(grid3, text="Batch size").grid(row=3, column=0, sticky="e")
        ttk.Spinbox(grid3, from_=1, to=32, increment=1, textvariable=self.target_batch_var, width=8).grid(row=3, column=1, sticky="w", padx=4)
        ttk.Label(grid3, text="Model").grid(row=3, column=2, sticky="e")
        ttk.Combobox(grid3, textvariable=self.target_arch_var, values=["unet_resnet18", "unet_small"], width=14)\
            .grid(row=3, column=3, sticky="w", padx=4)
        ttk.Checkbutton(grid3, text="Pretrained encoder", variable=self.target_pretrained_var)\
            .grid(row=3, column=4, sticky="w", padx=(6,0))
        ttk.Checkbutton(grid3, text="Allow empty (no-target) images", variable=self.target_allow_empty_var)\
            .grid(row=3, column=2, columnspan=2, sticky="w", padx=4)
        grid3.grid_columnconfigure(1, weight=1)

        r += 1
        tbar2 = ttk.Frame(tab3); tbar2.grid(row=r, column=0, columnspan=3, sticky="w", pady=(4,0))
        ttk.Button(tbar2, text="Train Tip Model", command=self._launch_target_training).pack(side="left")
        ttk.Button(tbar2, text="Load tip model", command=self._load_target_model).pack(side="left", padx=(8,0))

        # --- Tip Segmentation (no SAM)
        r += 1
        tipf = ttk.LabelFrame(tab3, text=" Tip Segmentation ", padding=8)
        tipf.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(8,4))
        ttk.Checkbutton(
            tipf,
            text="Use tip model for segmentation (no SAM)",
            variable=self.target_use_tipseg,
        ).grid(row=0, column=0, columnspan=4, sticky="w")

        ttk.Label(tipf, text="Threshold").grid(row=1, column=0, sticky="e", pady=(4,0))
        ttk.Spinbox(tipf, from_=0.05, to=0.95, increment=0.05, textvariable=self.target_tipseg_thresh, width=6)\
            .grid(row=1, column=1, sticky="w", pady=(4,0))
        ttk.Label(tipf, text="Min area (px)").grid(row=1, column=2, sticky="e", padx=(16,4), pady=(4,0))
        ttk.Spinbox(tipf, from_=0, to=200000, increment=50, textvariable=self.target_tipseg_min_area, width=8)\
            .grid(row=1, column=3, sticky="w", pady=(4,0))
        ttk.Checkbutton(tipf, text="Keep largest component", variable=self.target_tipseg_keep_largest)\
            .grid(row=2, column=0, columnspan=3, sticky="w", pady=(4,0))

        ttk.Checkbutton(tipf, text="Sliding window (large images)", variable=self.tipseg_use_tiles)\
            .grid(row=3, column=0, columnspan=3, sticky="w", pady=(6,0))
        ttk.Label(tipf, text="Tile").grid(row=4, column=0, sticky="e", pady=(4,0))
        ttk.Spinbox(tipf, from_=128, to=2048, increment=64, textvariable=self.tipseg_tile_size, width=8)\
            .grid(row=4, column=1, sticky="w", pady=(4,0))
        ttk.Label(tipf, text="Stride").grid(row=4, column=2, sticky="e", padx=(16,4), pady=(4,0))
        ttk.Spinbox(tipf, from_=64, to=2048, increment=64, textvariable=self.tipseg_stride, width=8)\
            .grid(row=4, column=3, sticky="w", pady=(4,0))
        ttk.Checkbutton(tipf, text="Color-guided scan (faster)", variable=self.tipseg_color_guided)\
            .grid(row=5, column=0, columnspan=3, sticky="w", pady=(4,0))
        ttk.Label(tipf, text="Color area min").grid(row=5, column=2, sticky="e", padx=(16,4), pady=(4,0))
        ttk.Spinbox(tipf, from_=0, to=50000, increment=50, textvariable=self.tipseg_color_min_area, width=8)\
            .grid(row=5, column=3, sticky="w", pady=(4,0))

        ttk.Label(tipf, text="Hue low/high").grid(row=6, column=0, sticky="e", pady=(4,0))
        ttk.Spinbox(tipf, from_=0, to=179, increment=1, textvariable=self.tipseg_hue_low, width=6)\
            .grid(row=6, column=1, sticky="w", pady=(4,0))
        ttk.Spinbox(tipf, from_=0, to=179, increment=1, textvariable=self.tipseg_hue_high, width=6)\
            .grid(row=6, column=2, sticky="w", pady=(4,0))
        ttk.Label(tipf, text="Sat min").grid(row=6, column=3, sticky="e", padx=(8,4), pady=(4,0))
        ttk.Spinbox(tipf, from_=0, to=255, increment=5, textvariable=self.tipseg_sat_min, width=6)\
            .grid(row=6, column=4, sticky="w", pady=(4,0))

        ttk.Label(tipf, text="Val min").grid(row=7, column=0, sticky="e", pady=(4,0))
        ttk.Spinbox(tipf, from_=0, to=255, increment=5, textvariable=self.tipseg_val_min, width=6)\
            .grid(row=7, column=1, sticky="w", pady=(4,0))
        ttk.Label(tipf, text="Brown V max").grid(row=7, column=2, sticky="e", padx=(8,4), pady=(4,0))
        ttk.Spinbox(tipf, from_=0, to=255, increment=5, textvariable=self.tipseg_val_brown_max, width=6)\
            .grid(row=7, column=3, sticky="w", pady=(4,0))

        ttk.Label(tipf, text="Min leaf %").grid(row=8, column=0, sticky="e", pady=(4,0))
        ttk.Spinbox(tipf, from_=0.0, to=100.0, increment=0.5, textvariable=self.tipseg_min_leaf_pct, width=6)\
            .grid(row=8, column=1, sticky="w", pady=(4,0))
        ttk.Label(tipf, text="Min stress %").grid(row=8, column=2, sticky="e", padx=(8,4), pady=(4,0))
        ttk.Spinbox(tipf, from_=0.0, to=100.0, increment=0.5, textvariable=self.tipseg_min_stress_pct, width=6)\
            .grid(row=8, column=3, sticky="w", pady=(4,0))
        ttk.Checkbutton(tipf, text="Stop after first hit", variable=self.tipseg_stop_after_first)\
            .grid(row=9, column=0, columnspan=3, sticky="w", pady=(4,0))

        # ════════════════════════════════════════════════════════════════════
        # SHARED LOG AREA (below tabs)
        # ════════════════════════════════════════════════════════════════════
        log_frame = ttk.LabelFrame(tf, text=" Training Log ", padding=4)
        log_frame.pack(fill="both", expand=True, pady=(8, 0))

        self.train_log = tk.Text(log_frame, height=8, width=100, bg=self.colors['bg_pale'], fg=self.colors['text_dark'])
        self.train_log.pack(fill="both", expand=True)
        return tf

    def make_status_bar(self, root):
        """Create a modern status bar at the bottom of the window."""
        c = self.colors

        # Status bar frame
        status_frame = tk.Frame(root, bg=c['bg_dark'], height=28)
        status_frame.grid(row=1, column=0, sticky="ew")
        status_frame.grid_propagate(False)

        # Left section: Status message
        self._status_icon = tk.Label(
            status_frame, text="", bg=c['bg_dark'], fg=c['text_light'],
            font=("Helvetica", 11), padx=8
        )
        self._status_icon.pack(side="left")

        self._status_label = tk.Label(
            status_frame, text="Ready", bg=c['bg_dark'], fg=c['text_light'],
            font=("Helvetica", 10), anchor="w"
        )
        self._status_label.pack(side="left", fill="x", expand=True)

        # Right section: Active weights + shortcuts
        self._weights_label = tk.Label(
            status_frame, text="Weights: (none)",
            bg=c['bg_dark'], fg=c['bg_light'],
            font=("Helvetica", 9), padx=10
        )
        self._weights_label.pack(side="right")

        shortcuts_text = "⌨ Ctrl+O: Open  |  Ctrl+S: Save  |  Del: Delete mask  |  Ctrl+Z: Undo"
        self._shortcuts_label = tk.Label(
            status_frame, text=shortcuts_text,
            bg=c['bg_dark'], fg=c['bg_light'],
            font=("Helvetica", 9), padx=10
        )
        self._shortcuts_label.pack(side="right")

        # Separator line above status bar
        sep = tk.Frame(root, bg=c['border'], height=1)
        sep.grid(row=2, column=0, columnspan=2, sticky="new")

    def set_status(self, message, status_type="info"):
        """Update the status bar with a message and icon.

        status_type: 'info', 'success', 'warning', 'error', 'processing'
        """
        icons = {
            'info': 'ℹ️',
            'success': '✓',
            'warning': '⚠️',
            'error': '✗',
            'processing': '⏳',
        }
        colors = {
            'info': self.colors['text_light'],
            'success': self.colors['success'],
            'warning': self.colors['warning'],
            'error': self.colors['error'],
            'processing': self.colors['accent'],
        }

        icon = icons.get(status_type, '')
        color = colors.get(status_type, self.colors['text_light'])

        try:
            self._status_icon.configure(text=icon, fg=color)
            self._status_label.configure(text=message, fg=color)
        except Exception:
            pass

    def _refresh_weights_badge(self):
        """Update the bottom-right weights badge (SAM weights + tip model presence)."""
        try:
            sam_tag = getattr(self, "_sam_weights_tag", "(none)")
            tip_tag = " | Tip: loaded" if getattr(self, "tipseg_model", None) is not None else ""
            if hasattr(self, "_weights_label"):
                self._weights_label.configure(text=f"Weights: {sam_tag}{tip_tag}")
        except Exception:
            pass

    def _set_sam_weights_tag(self, tag: str):
        self._sam_weights_tag = str(tag)
        self._refresh_weights_badge()

    def _bind_global_shortcuts(self):
        """Bind keyboard shortcuts for common actions."""
        # Open image
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Command-o>", lambda e: self.open_image())

        # Save masks
        self.root.bind("<Control-s>", lambda e: self.save_selected_masks())
        self.root.bind("<Command-s>", lambda e: self.save_selected_masks())

        # Segment
        self.root.bind("<Control-Return>", lambda e: self.segment())
        self.root.bind("<Command-Return>", lambda e: self.segment())

        # Preview enhance
        self.root.bind("<Control-e>", lambda e: self.preview_enhance())
        self.root.bind("<Command-e>", lambda e: self.preview_enhance())

        # Zoom controls
        self.root.bind("<Control-plus>", lambda e: self._zoom_by(1.25))
        self.root.bind("<Control-minus>", lambda e: self._zoom_by(0.8))
        self.root.bind("<Control-0>", lambda e: self._zoom_fit())

        # Navigation
        self.root.bind("<Control-Left>", lambda e: self.prev_image())
        self.root.bind("<Control-Right>", lambda e: self.next_image())

    def _browse_file_into(self, var, ftypes=("All","*.*")):
        p = filedialog.askopenfilename(filetypes=[ftypes] if isinstance(ftypes, tuple) else [("All","*.*")])
        if p: var.set(p)

    def _browse_save_into(self, var, default_ext=".pth"):
        p = filedialog.asksaveasfilename(defaultextension=default_ext,
                                        filetypes=[("Torch","*.pth *.pt"), ("All","*.*")])
        if p: var.set(p)

    # ═══════════════════════════════════════════════════════════════════════
    # Settings Save/Load/Reset
    # ═══════════════════════════════════════════════════════════════════════

    def _get_all_settings(self) -> dict:
        """Collect all enhancement parameters into a dictionary."""
        settings = {
            "_version": "1.0",
            "_description": "Leaf Segmenter Enhancement Settings",

            # Basic enhancement
            "use_green": self.use_green.get(),
            "use_classic": self.use_classic.get(),
            "brightness": self.s_brightness.get(),
            "contrast": self.s_contrast.get(),
            "gamma": self.s_gamma.get(),
            "unsharp": self.chk_unsharp.get(),
            "laplacian": self.chk_laplacian.get(),
            "whiten_bg": self.chk_whiten.get(),

            # Whiten parameters
            "val_min": self.s_val_min.get(),
            "sat_max": self.s_sat_max.get(),
            "close_iters": self.s_close_iters.get(),
            "halo_erode": self.s_halo_erode.get(),
            "halo_feather": self.s_halo_feather.get(),

            # Denoise
            "median_on": self.dn_median_on.get(),
            "median_ksize": self.dn_median_ksize.get(),
            "mean_on": self.dn_mean_on.get(),
            "mean_ksize": self.dn_mean_ksize.get(),

            # Edge darken
            "edge_darken_on": self.ed_on.get(),
            "edge_darken_width": self.ed_width.get(),
            "edge_darken_amount": self.ed_amount.get(),

            # SAM2 parameters
            "sam_points_per_side": self.m_points_per_side.get(),
            "sam_points_per_batch": self.m_points_per_batch.get(),
            "sam_pred_iou_thresh": self.m_pred_iou_thresh.get(),
            "sam_stability_score_thresh": self.m_stability_score_thresh.get(),
            "sam_crop_n_layers": self.m_crop_n_layers.get(),
            "sam_crop_overlap_ratio": self.m_crop_overlap_ratio.get(),
            "sam_box_nms_thresh": self.m_box_nms_thresh.get(),
            "sam_min_mask_region_area": self.m_min_mask_region_area.get(),
            "sam_use_m2m": self.m_use_m2m.get(),
            "sam_output_mode": self.m_output_mode.get(),

            # Phenotypes
            "ph_all": self.ph_all.get(),
            "ph_area": self.ph_area.get(),
            "ph_len": self.ph_len.get(),
            "ph_wid": self.ph_wid.get(),
            "ph_color": self.ph_color.get(),
            "ph_hsv": self.ph_hsv.get(),
            "ph_shape": self.ph_shape.get(),
            "ph_comp": self.ph_comp.get(),
            "ph_veg": self.ph_veg.get(),
            "ph_hsvvar": self.ph_hsvvar.get(),

            # Rotation
            "rotation_angle": self.rot_angle.get(),
        }

        # Advanced enhancement options (check if they exist)
        advanced_params = [
            ("use_veg_index", "use_veg_index"),
            ("veg_index_type", "veg_index_type"),
            ("veg_index_blend", "veg_index_blend"),
            ("use_white_balance", "use_white_balance"),
            ("white_balance_type", "white_balance_type"),
            ("use_retinex", "use_retinex"),
            ("retinex_type", "retinex_type"),
            ("retinex_sigma", "retinex_sigma"),
            ("use_lab", "use_lab"),
            ("lab_l_factor", "lab_l_factor"),
            ("lab_a_shift", "lab_a_shift"),
            ("use_nlm", "use_nlm"),
            ("nlm_h", "nlm_h"),
            ("use_tophat", "use_tophat"),
            ("tophat_size", "tophat_size"),
            ("use_guided", "use_guided"),
            ("guided_radius", "guided_radius"),
            ("guided_eps", "guided_eps"),
            ("use_dog", "use_dog"),
            ("dog_sigma1", "dog_sigma1"),
            ("dog_sigma2", "dog_sigma2"),
            ("dog_blend", "dog_blend"),
            ("use_shadow_highlight", "use_shadow_highlight"),
            ("shadow_amount", "shadow_amount"),
            ("highlight_amount", "highlight_amount"),
            ("use_local_contrast", "use_local_contrast"),
            ("local_contrast_size", "local_contrast_size"),
            ("use_adaptive_gamma", "use_adaptive_gamma"),
        ]

        for key, attr in advanced_params:
            if hasattr(self, attr):
                var = getattr(self, attr)
                settings[key] = var.get()

        return settings

    def _apply_settings(self, settings: dict):
        """Apply settings from a dictionary to the UI."""
        # Helper to safely set a variable
        def safe_set(var_name, key, default=None):
            if hasattr(self, var_name):
                var = getattr(self, var_name)
                val = settings.get(key, default)
                if val is not None:
                    try:
                        var.set(val)
                    except Exception:
                        pass

        # Basic enhancement
        safe_set("use_green", "use_green", True)
        safe_set("use_classic", "use_classic", False)
        safe_set("s_brightness", "brightness", -25)
        safe_set("s_contrast", "contrast", 1.0)
        safe_set("s_gamma", "gamma", 1.2)
        safe_set("chk_unsharp", "unsharp", True)
        safe_set("chk_laplacian", "laplacian", False)
        safe_set("chk_whiten", "whiten_bg", False)

        # Whiten parameters
        safe_set("s_val_min", "val_min", 200)
        safe_set("s_sat_max", "sat_max", 35)
        safe_set("s_close_iters", "close_iters", 1)
        safe_set("s_halo_erode", "halo_erode", 1)
        safe_set("s_halo_feather", "halo_feather", 2)

        # Denoise
        safe_set("dn_median_on", "median_on", False)
        safe_set("dn_median_ksize", "median_ksize", 5)
        safe_set("dn_mean_on", "mean_on", False)
        safe_set("dn_mean_ksize", "mean_ksize", 3)

        # Edge darken
        safe_set("ed_on", "edge_darken_on", False)
        safe_set("ed_width", "edge_darken_width", 3)
        safe_set("ed_amount", "edge_darken_amount", 0.35)

        # SAM2 parameters
        safe_set("m_points_per_side", "sam_points_per_side", 16)
        safe_set("m_points_per_batch", "sam_points_per_batch", 16)
        safe_set("m_pred_iou_thresh", "sam_pred_iou_thresh", 0.90)
        safe_set("m_stability_score_thresh", "sam_stability_score_thresh", 0.80)
        safe_set("m_crop_n_layers", "sam_crop_n_layers", 1)
        safe_set("m_crop_overlap_ratio", "sam_crop_overlap_ratio", 0.30)
        safe_set("m_box_nms_thresh", "sam_box_nms_thresh", 0.60)
        safe_set("m_min_mask_region_area", "sam_min_mask_region_area", 800)
        safe_set("m_use_m2m", "sam_use_m2m", True)
        safe_set("m_output_mode", "sam_output_mode", "binary_mask")

        # Phenotypes
        safe_set("ph_all", "ph_all", True)
        safe_set("ph_area", "ph_area", True)
        safe_set("ph_len", "ph_len", True)
        safe_set("ph_wid", "ph_wid", True)
        safe_set("ph_color", "ph_color", True)
        safe_set("ph_hsv", "ph_hsv", True)
        safe_set("ph_shape", "ph_shape", True)
        safe_set("ph_comp", "ph_comp", True)
        safe_set("ph_veg", "ph_veg", True)
        safe_set("ph_hsvvar", "ph_hsvvar", True)

        # Rotation
        safe_set("rot_angle", "rotation_angle", 0.0)

        # Advanced enhancement options
        advanced_params = [
            ("use_veg_index", "use_veg_index", False),
            ("veg_index_type", "veg_index_type", "ExG"),
            ("veg_index_blend", "veg_index_blend", 0.3),
            ("use_white_balance", "use_white_balance", False),
            ("white_balance_type", "white_balance_type", "grayworld"),
            ("use_retinex", "use_retinex", False),
            ("retinex_type", "retinex_type", "multi"),
            ("retinex_sigma", "retinex_sigma", 80),
            ("use_lab", "use_lab", False),
            ("lab_l_factor", "lab_l_factor", 1.0),
            ("lab_a_shift", "lab_a_shift", -10),
            ("use_nlm", "use_nlm", False),
            ("nlm_h", "nlm_h", 10),
            ("use_tophat", "use_tophat", False),
            ("tophat_size", "tophat_size", 50),
            ("use_guided", "use_guided", False),
            ("guided_radius", "guided_radius", 8),
            ("guided_eps", "guided_eps", 0.04),
            ("use_dog", "use_dog", False),
            ("dog_sigma1", "dog_sigma1", 1.0),
            ("dog_sigma2", "dog_sigma2", 3.0),
            ("dog_blend", "dog_blend", 0.3),
            ("use_shadow_highlight", "use_shadow_highlight", False),
            ("shadow_amount", "shadow_amount", 0.3),
            ("highlight_amount", "highlight_amount", 0.3),
            ("use_local_contrast", "use_local_contrast", False),
            ("local_contrast_size", "local_contrast_size", 31),
            ("use_adaptive_gamma", "use_adaptive_gamma", False),
        ]

        for attr, key, default in advanced_params:
            safe_set(attr, key, default)

        # Update knob display if rotation changed
        if hasattr(self, '_draw_knob'):
            self._draw_knob()

    def save_settings(self):
        """Save all enhancement settings to a JSON file."""
        p = filedialog.asksaveasfilename(
            title="Save Enhancement Settings",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="enhancement_settings.json"
        )
        if not p:
            return

        try:
            settings = self._get_all_settings()
            with open(p, 'w') as f:
                json.dump(settings, f, indent=2)
            self.set_status(f"Settings saved to {Path(p).name}", "success")
            messagebox.showinfo("Settings Saved", f"Enhancement settings saved to:\n{p}")
        except Exception as e:
            self.set_status(f"Failed to save settings", "error")
            messagebox.showerror("Save Failed", str(e))

    def load_settings(self):
        """Load enhancement settings from a JSON file."""
        p = filedialog.askopenfilename(
            title="Load Enhancement Settings",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not p:
            return

        try:
            with open(p, 'r') as f:
                settings = json.load(f)

            # Check version compatibility
            version = settings.get("_version", "unknown")
            if version != "1.0":
                if not messagebox.askyesno("Version Mismatch",
                    f"Settings file version ({version}) may not be fully compatible.\nContinue loading?"):
                    return

            self._apply_settings(settings)
            self.set_status(f"Settings loaded from {Path(p).name}", "success")
            messagebox.showinfo("Settings Loaded",
                f"Enhancement settings loaded from:\n{p}\n\nClick 'Preview Enhance' to see the effect.")
        except json.JSONDecodeError as e:
            self.set_status("Invalid settings file", "error")
            messagebox.showerror("Load Failed", f"Invalid JSON file:\n{e}")
        except Exception as e:
            self.set_status("Failed to load settings", "error")
            messagebox.showerror("Load Failed", str(e))

    def reset_settings(self):
        """Reset all enhancement settings to defaults."""
        if not messagebox.askyesno("Reset Settings",
            "Reset all enhancement parameters to their default values?"):
            return

        # Default settings
        defaults = {
            "_version": "1.0",
            "use_green": True,
            "use_classic": False,
            "brightness": -25,
            "contrast": 1.0,
            "gamma": 1.2,
            "unsharp": True,
            "laplacian": False,
            "whiten_bg": False,
            "val_min": 200,
            "sat_max": 35,
            "close_iters": 1,
            "halo_erode": 1,
            "halo_feather": 2,
            "median_on": False,
            "median_ksize": 5,
            "mean_on": False,
            "mean_ksize": 3,
            "edge_darken_on": False,
            "edge_darken_width": 3,
            "edge_darken_amount": 0.35,
            "sam_points_per_side": 16,
            "sam_points_per_batch": 16,
            "sam_pred_iou_thresh": 0.90,
            "sam_stability_score_thresh": 0.80,
            "sam_crop_n_layers": 1,
            "sam_crop_overlap_ratio": 0.30,
            "sam_box_nms_thresh": 0.60,
            "sam_min_mask_region_area": 800,
            "sam_use_m2m": True,
            "sam_output_mode": "binary_mask",
            "ph_all": True,
            "ph_area": True,
            "ph_len": True,
            "ph_wid": True,
            "ph_color": True,
            "ph_hsv": True,
            "ph_shape": True,
            "ph_comp": True,
            "ph_veg": True,
            "ph_hsvvar": True,
            "rotation_angle": 0.0,
            # Advanced defaults (all off)
            "use_veg_index": False,
            "veg_index_type": "ExG",
            "veg_index_blend": 0.3,
            "use_white_balance": False,
            "white_balance_type": "grayworld",
            "use_retinex": False,
            "retinex_type": "multi",
            "retinex_sigma": 80,
            "use_lab": False,
            "lab_l_factor": 1.0,
            "lab_a_shift": -10,
            "use_nlm": False,
            "nlm_h": 10,
            "use_tophat": False,
            "tophat_size": 50,
            "use_guided": False,
            "guided_radius": 8,
            "guided_eps": 0.04,
            "use_dog": False,
            "dog_sigma1": 1.0,
            "dog_sigma2": 3.0,
            "dog_blend": 0.3,
            "use_shadow_highlight": False,
            "shadow_amount": 0.3,
            "highlight_amount": 0.3,
            "use_local_contrast": False,
            "local_contrast_size": 31,
            "use_adaptive_gamma": False,
        }

        self._apply_settings(defaults)
        self.set_status("Settings reset to defaults", "success")

    def _convert_tif_folder(self):
        """Convert all TIF/TIFF files in a folder to PNG or JPEG."""
        # Ask for input folder
        in_folder = filedialog.askdirectory(title="Select folder containing TIF files")
        if not in_folder:
            return

        in_path = Path(in_folder)
        tif_files = list(in_path.glob("*.tif")) + list(in_path.glob("*.tiff")) + \
                    list(in_path.glob("*.TIF")) + list(in_path.glob("*.TIFF"))

        if not tif_files:
            messagebox.showwarning("No TIF Files", f"No .tif or .tiff files found in:\n{in_folder}")
            return

        # Ask for output folder
        out_folder = filedialog.askdirectory(title=f"Select output folder for converted files ({len(tif_files)} TIFs found)")
        if not out_folder:
            return

        out_path = Path(out_folder)
        out_path.mkdir(parents=True, exist_ok=True)

        # Get format and max size
        fmt = self._convert_format.get().upper()
        ext = ".png" if fmt == "PNG" else ".jpg"
        try:
            max_size = int(self._convert_max_size.get())
        except ValueError:
            max_size = 0

        # Conversion in thread to avoid blocking GUI
        def convert_worker():
            converted = 0
            failed = []
            for i, tif_path in enumerate(tif_files):
                try:
                    self.root.after(0, lambda p=tif_path, idx=i: self.set_status(
                        f"Converting {idx+1}/{len(tif_files)}: {p.name}", "info"))

                    # Read TIF (cv2 handles most TIF formats)
                    img = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
                    if img is None:
                        # Try with PIL for more exotic TIFs
                        from PIL import Image
                        pil_img = Image.open(tif_path)
                        img = np.array(pil_img)
                        if len(img.shape) == 2:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        elif img.shape[2] == 4:
                            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                        elif img.shape[2] == 3:
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    if img is None:
                        failed.append(tif_path.name)
                        continue

                    # Handle 16-bit images
                    if img.dtype == np.uint16:
                        img = (img / 256).astype(np.uint8)
                    elif img.dtype == np.float32 or img.dtype == np.float64:
                        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

                    # Resize if max_size specified
                    if max_size > 0:
                        h, w = img.shape[:2]
                        if max(h, w) > max_size:
                            scale = max_size / max(h, w)
                            new_w, new_h = int(w * scale), int(h * scale)
                            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    # Save
                    out_file = out_path / (tif_path.stem + ext)
                    if fmt == "PNG":
                        cv2.imwrite(str(out_file), img)
                    else:
                        cv2.imwrite(str(out_file), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    converted += 1
                except Exception as e:
                    failed.append(f"{tif_path.name}: {e}")

            # Done - report results
            def show_result():
                msg = f"Converted {converted}/{len(tif_files)} files to {fmt}"
                if failed:
                    msg += f"\n\nFailed ({len(failed)}):\n" + "\n".join(failed[:10])
                    if len(failed) > 10:
                        msg += f"\n... and {len(failed)-10} more"
                    messagebox.showwarning("Conversion Complete", msg)
                else:
                    messagebox.showinfo("Conversion Complete", msg + f"\n\nOutput: {out_folder}")
                self.set_status(f"Converted {converted} TIF files", "success")

            self.root.after(0, show_result)

        import threading
        threading.Thread(target=convert_worker, daemon=True).start()
        self.set_status(f"Converting {len(tif_files)} TIF files...", "info")

    def _pick_train_root(self):
        d = filedialog.askdirectory(title="Choose dataset root (will create images/ and masks/)")
        if not d: return
        self.train_root = d
        self.train_root_var.set(d)
        self._ensure_train_dirs()
        self._maybe_use_dataset_images_for_batch(self.train_images_dir)

    def _ensure_train_dirs(self):
        if not self.train_root:
            return False
        root = Path(self.train_root)
        (root/"images").mkdir(parents=True, exist_ok=True)
        (root/"masks").mkdir(parents=True, exist_ok=True)
        self.train_images_dir = root/"images"
        self.train_masks_dir  = root/"masks"
        return True

    def _open_train_root(self):
        if not self.train_root:
            messagebox.showwarning("Dataset", "Pick a dataset folder first.")
            return
        try:
            import webbrowser
            webbrowser.open(Path(self.train_root).as_uri())
        except Exception:
            messagebox.showinfo("Dataset", str(self.train_root))

    def _update_train_msg(self):
        n = len(self.train_examples)
        self.train_msg.configure(text=f"{n} example{'s' if n!=1 else ''}")

    def _current_selected_mask_indices(self):
        if not self.sr or not self.sr.masks:
            return []
        sel = list(self.lb.curselection())
        return sel if sel else list(range(len(self.sr.masks)))

    def _add_current_to_training(self):
        if not self._ensure_train_dirs():
            messagebox.showwarning("Dataset", "Pick a dataset folder first."); return
        if not self.sr or not self.sr.masks:
            messagebox.showwarning("Training", "Run Segmentation first."); return
        idxs = self._current_selected_mask_indices()
        if not idxs:
            messagebox.showwarning("Training", "No masks to add."); return

        base = self.sr.img_color           # same geometry as masks
        stem = Path(self.img_path).stem if self.img_path else f"Image_{int(time.time())}"

        # ensure unique stem
        out_img = self.train_images_dir / f"{stem}.png"
        k = 1
        while out_img.exists():
            k += 1
            out_img = self.train_images_dir / f"{stem}_{k}.png"

        # save image
        cv2.imwrite(str(out_img), cv2.cvtColor(base, cv2.COLOR_RGB2BGR))

        # save per-instance masks: <stem>_inst01.png, inst02.png, ...
        saved = []
        for j, idx in enumerate(idxs, start=1):
            seg = self.sr.masks[idx]["segmentation"].astype(np.uint8)
            mp = self.train_masks_dir / f"{out_img.stem}_inst{j:02d}.png"
            cv2.imwrite(str(mp), seg * 255)
            saved.append(str(mp))

        # record example + light manifest
        self.train_examples.append({"image": str(out_img), "masks": saved})
        with open(Path(self.train_root)/"manifest.json", "w") as f:
            json.dump({"examples": self.train_examples}, f, indent=2)

        self._update_train_msg()
        messagebox.showinfo("Training", f"Added {len(saved)} mask(s) from this image.")

    def _import_rgba_crops(self):
        """Import RGBA PNG files where alpha channel is the mask."""
        if not self._ensure_train_dirs():
            messagebox.showwarning("Dataset", "Pick a dataset folder first.")
            return

        # Ask for folder with RGBA files
        src_folder = filedialog.askdirectory(title="Select folder with RGBA PNG crops")
        if not src_folder:
            return

        src_path = Path(src_folder)
        png_files = list(src_path.glob("*.png")) + list(src_path.glob("*.PNG"))

        if not png_files:
            messagebox.showwarning("Import", f"No PNG files found in:\n{src_folder}")
            return

        imported = 0
        skipped = 0

        for png_path in png_files:
            try:
                # Read with alpha channel
                img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    skipped += 1
                    continue

                # Check if RGBA
                if len(img.shape) != 3 or img.shape[2] != 4:
                    # Not RGBA - try to use as regular image with auto-mask from non-black pixels
                    if len(img.shape) == 2:
                        # Grayscale - treat as mask itself
                        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        mask = (img > 0).astype(np.uint8)
                    elif img.shape[2] == 3:
                        # RGB - create mask from non-black pixels
                        rgb = img
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        mask = (gray > 10).astype(np.uint8)
                    else:
                        skipped += 1
                        continue
                else:
                    # RGBA - split into RGB and mask from alpha
                    rgb = img[:, :, :3]  # BGR channels
                    alpha = img[:, :, 3]  # Alpha channel
                    mask = (alpha > 127).astype(np.uint8)

                # Skip if mask is empty
                if mask.sum() < 10:
                    skipped += 1
                    continue

                # Generate unique stem
                stem = png_path.stem
                out_img = self.train_images_dir / f"{stem}.png"
                k = 1
                while out_img.exists():
                    k += 1
                    out_img = self.train_images_dir / f"{stem}_{k}.png"

                # Save RGB image
                cv2.imwrite(str(out_img), rgb)

                # Save mask
                mask_path = self.train_masks_dir / f"{out_img.stem}_inst01.png"
                cv2.imwrite(str(mask_path), mask * 255)

                # Record example
                self.train_examples.append({
                    "image": str(out_img),
                    "masks": [str(mask_path)]
                })
                imported += 1

            except Exception as e:
                print(f"Error importing {png_path.name}: {e}")
                skipped += 1

        # Save manifest
        if imported > 0:
            with open(Path(self.train_root) / "manifest.json", "w") as f:
                json.dump({"examples": self.train_examples}, f, indent=2)

        self._update_train_msg()
        msg = f"Imported {imported} RGBA crops as training examples."
        if skipped > 0:
            msg += f"\nSkipped {skipped} files (not RGBA or empty mask)."
        messagebox.showinfo("Import Complete", msg)

    def _clear_training_set(self):
        if not self.train_root: return
        if not messagebox.askyesno("Clear dataset", "Delete ALL files under images/ and masks/?"):
            return
        for sub in ("images","masks"):
            p = Path(self.train_root)/sub
            if p.exists():
                for q in p.iterdir():
                    try: q.unlink()
                    except Exception: pass
        self.train_examples.clear()
        self._update_train_msg()
        messagebox.showinfo("Dataset", "Cleared.")

    # ===== Target Segment dataset helpers =====
    def _pick_target_root(self):
        d = filedialog.askdirectory(title="Choose target dataset root (will create images/ and masks/)")
        if not d:
            return
        self.target_root = d
        self.target_root_var.set(d)
        self._ensure_target_dirs()
        self._scan_target_dataset()
        self._maybe_use_dataset_images_for_batch(self.target_images_dir)

    def _ensure_target_dirs(self):
        if not self.target_root:
            return False
        root = Path(self.target_root)
        (root/"images").mkdir(parents=True, exist_ok=True)
        (root/"masks").mkdir(parents=True, exist_ok=True)
        self.target_images_dir = root/"images"
        self.target_masks_dir  = root/"masks"
        return True

    def _open_target_root(self):
        if not self.target_root:
            messagebox.showwarning("Dataset", "Pick a dataset folder first.")
            return
        try:
            import webbrowser
            webbrowser.open(Path(self.target_root).as_uri())
        except Exception:
            messagebox.showinfo("Dataset", str(self.target_root))

    def _update_target_msg(self):
        n = len(self.target_examples)
        self.target_msg.configure(text=f"{n} example{'s' if n!=1 else ''}")

    def _scan_target_dataset(self):
        """Scan dataset folders on disk and update the UI message."""
        if not self._ensure_target_dirs():
            return
        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        root = Path(self.target_root)
        images_dir = Path(self.target_images_dir)
        masks_dir = Path(self.target_masks_dir)

        # Count images in images/; if empty, also look at root
        imgs = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        if not imgs:
            imgs = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]

        pos_masks = list(masks_dir.glob("*_inst*.png"))
        no_targets = list(masks_dir.glob("*_nomask.txt"))
        msg = f"images: {len(imgs)} | pos masks: {len(pos_masks)} | no-target: {len(no_targets)}"
        try:
            self.target_msg.configure(text=msg)
        except Exception:
            pass

    def _maybe_use_dataset_images_for_batch(self, images_dir: Path, allow_empty: bool = False):
        """If dataset images exist, load them into the batch viewer."""
        try:
            exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

            def _collect(p: Path):
                if not p or not p.exists() or not p.is_dir():
                    return []
                return [str(x) for x in sorted(p.iterdir())
                        if x.suffix.lower() in exts and x.is_file()]

            images_dir = Path(images_dir)
            imgs = _collect(images_dir)

            # If empty, try common fallbacks: sibling 'images' or parent
            if not imgs:
                # If user picked root, prefer root/images if present
                sub = images_dir / "images"
                if sub.exists():
                    imgs = _collect(sub)
                    if imgs:
                        images_dir = sub
                if not imgs:
                    parent = images_dir.parent
                    parent_imgs = _collect(parent)
                    if parent_imgs:
                        images_dir = parent
                        imgs = parent_imgs

            if not imgs:
                if not allow_empty:
                    messagebox.showwarning("Dataset", "No images found in dataset folder (or its images/ subfolder).")
                return

            self.batch_dir = str(images_dir)
            self.batch_images = imgs
            self.batch_idx = 0
            self._load_batch_index(0)
            self.set_status(f"Loaded dataset images: {len(imgs)}", "success")
        except Exception:
            pass

    def _add_current_to_target(self):
        if not self._ensure_target_dirs():
            messagebox.showwarning("Dataset", "Pick a dataset folder first."); return
        if not self.sr or not self.sr.masks:
            messagebox.showwarning("Target", "Run Segmentation first."); return
        idxs = list(self.lb.curselection()) if self.lb else []
        if not idxs:
            messagebox.showwarning("Target", "Select the target mask(s) first."); return

        base = self.sr.img_color
        stem = Path(self.img_path).stem if self.img_path else f"Image_{int(time.time())}"

        out_img = self.target_images_dir / f"{stem}.png"
        k = 1
        while out_img.exists():
            k += 1
            out_img = self.target_images_dir / f"{stem}_{k}.png"

        cv2.imwrite(str(out_img), cv2.cvtColor(base, cv2.COLOR_RGB2BGR))

        saved = []
        for j, idx in enumerate(idxs, start=1):
            seg = self.sr.masks[idx]["segmentation"].astype(np.uint8)
            mp = self.target_masks_dir / f"{out_img.stem}_inst{j:02d}.png"
            cv2.imwrite(str(mp), seg * 255)
            saved.append(str(mp))

        self.target_examples.append({"image": str(out_img), "masks": saved})
        with open(Path(self.target_root)/"manifest.json", "w") as f:
            json.dump({"examples": self.target_examples}, f, indent=2)

        self._update_target_msg()
        messagebox.showinfo("Target", f"Added {len(saved)} target mask(s).")

    def _add_negative_target(self):
        if not self._ensure_target_dirs():
            messagebox.showwarning("Dataset", "Pick a dataset folder first."); return
        if self.sr is None or self.img is None:
            messagebox.showwarning("Target", "Open an image first."); return

        base = self.sr.img_color if self.sr else self.img
        stem = Path(self.img_path).stem if self.img_path else f"Image_{int(time.time())}"

        out_img = self.target_images_dir / f"{stem}.png"
        k = 1
        while out_img.exists():
            k += 1
            out_img = self.target_images_dir / f"{stem}_{k}.png"

        cv2.imwrite(str(out_img), cv2.cvtColor(base, cv2.COLOR_RGB2BGR))
        marker = self.target_masks_dir / f"{out_img.stem}_nomask.txt"
        try:
            marker.write_text("no target")
        except Exception:
            pass

        self.target_examples.append({"image": str(out_img), "masks": [], "no_target": True})
        with open(Path(self.target_root)/"manifest.json", "w") as f:
            json.dump({"examples": self.target_examples}, f, indent=2)

        self._update_target_msg()
        messagebox.showinfo("Target", "Added NO-target example.")

    def _clear_target_set(self):
        if not self.target_root:
            return
        if not messagebox.askyesno("Clear dataset", "Delete ALL files under images/ and masks/?"):
            return
        for sub in ("images","masks"):
            p = Path(self.target_root)/sub
            if p.exists():
                for q in p.iterdir():
                    try: q.unlink()
                    except Exception: pass
        self.target_examples.clear()
        self._update_target_msg()
        messagebox.showinfo("Dataset", "Cleared.")

    def _append_train_log(self, line: str):
        try:
            self.train_log.insert("end", line + "\n"); self.train_log.see("end")
        except Exception:
            print(line)

    def _launch_training(self):
        if not self._ensure_train_dirs():
            messagebox.showwarning("Training", "Pick a dataset folder first."); return
        # must have at least one image saved as example
        if len(list((Path(self.train_images_dir)).glob("*.png"))) == 0:
            messagebox.showwarning("Training", "No images in dataset. Add examples first."); return

        # allow falling back to Model panel entries if Training fields are blank
        ckpt = (self.train_ckpt_var.get().strip() or self.e_ckpt.get().strip())
        if not ckpt or not os.path.exists(ckpt):
            messagebox.showwarning("Training", "Pick a valid SAM2 checkpoint (.pt)."); return
        cfg = (self.train_cfg_var.get().strip() or self.e_cfg.get().strip())
        outp= self.train_out_var.get().strip() or str(Path.home()/ "sam2_arabidopsis.pth")
        steps=int(self.train_steps_var.get()); lr=float(self.train_lr_var.get())
        img_size = int(self.train_size_var.get())
        device = (self.train_device_var.get().strip() or self.e_dev.get().strip() or "cpu")

        py = shlex.quote(sys.executable)
        cmd = (
            f"{py} sam2_trainer_arabidopsis.py "
            f"--images {shlex.quote(str(self.train_images_dir))} "
            f"--masks {shlex.quote(str(self.train_masks_dir))} "
            f"--checkpoint {shlex.quote(ckpt)} "
            f"--config {shlex.quote(cfg)} "
            f"--save_to {shlex.quote(outp)} --steps {steps} --lr {lr} "
            f"--size {img_size} --device {shlex.quote(device)}"
        )

        self._append_train_log("")
        self._append_train_log("Launching training:")
        self._append_train_log(f"  Images: {self.train_images_dir}")
        self._append_train_log(f"  Masks:  {self.train_masks_dir}")
        self._append_train_log(f"  Checkpoint: {ckpt}")
        self._append_train_log(f"  Config: {cfg}")
        self._append_train_log(f"  Output: {outp}")
        self._append_train_log(f"  Steps: {steps}, LR: {lr}, Size: {img_size}, Device: {device}")
        self._append_train_log("Command: " + cmd)
        try:
            self._train_proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            threading.Thread(target=self._train_reader_thread, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Training", str(e))

    def _train_reader_thread(self):
        p = getattr(self, "_train_proc", None)
        if not p: return
        for raw in iter(p.stdout.readline, b""):
            line = raw.decode(errors="replace").rstrip()
            self.root.after(0, lambda s=line: self._append_train_log(s))
        p.wait()
        code = p.returncode
        self.root.after(0, lambda: self._append_train_log(f"Training finished with code {code}"))

    def _load_finetuned_into_predictor(self):
        try:
            if self.sam2_model is None:
                messagebox.showwarning("No base model", "Load the SAM2 model first (Load Model or Load Bundle).")
                return
            ckpt_path = self.train_out_var.get().strip()
            if not ckpt_path or not os.path.exists(ckpt_path):
                messagebox.showwarning("Missing file", "Pick a valid fine-tuned .pth file.")
                return
            map_dev = (self.e_dev.get().strip() or "cpu")
            try:
                ckpt = torch.load(ckpt_path, map_location=map_dev, weights_only=True)
            except TypeError:
                ckpt = torch.load(ckpt_path, map_location=map_dev)
            if isinstance(ckpt, dict):
                if "state_dict" in ckpt:
                    sd = ckpt["state_dict"]
                elif "model" in ckpt:
                    sd = ckpt["model"]
                else:
                    sd = ckpt
            else:
                sd = ckpt
            self.sam2_model.load_state_dict(sd, strict=False)
            self.sam2_model.eval()
            messagebox.showinfo("Model", "Fine-tuned weights loaded.")
            try:
                self._set_sam_weights_tag("fine-tuned")
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def _launch_target_training(self):
        if not self._ensure_target_dirs():
            messagebox.showwarning("Target", "Pick a dataset folder first."); return
        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        if len([p for p in Path(self.target_images_dir).iterdir() if p.is_file() and p.suffix.lower() in exts]) == 0:
            messagebox.showwarning("Target", "No images in dataset. Add examples first."); return

        outp = self.target_out_var.get().strip() or str(Path.home()/ "sam2_target_segment.pth")
        steps = int(self.target_steps_var.get())
        lr = float(self.target_lr_var.get())
        img_size = int(self.target_size_var.get())
        device = (self.target_device_var.get().strip() or self.e_dev.get().strip() or "cpu")
        batch_size = int(self.target_batch_var.get())
        allow_empty = bool(self.target_allow_empty_var.get())
        resume = bool(self.target_resume_var.get()) and bool(outp) and os.path.exists(outp)
        arch = (self.target_arch_var.get().strip() or "unet_resnet18")
        pretrained = bool(self.target_pretrained_var.get())

        # Use unbuffered output so progress prints show up in the GUI log in real-time.
        py = shlex.quote(sys.executable) + " -u"
        cmd = (
            f"{py} tip_segmenter_trainer.py "
            f"--images {shlex.quote(str(self.target_images_dir))} "
            f"--masks {shlex.quote(str(self.target_masks_dir))} "
            f"--out {shlex.quote(outp)} --steps {steps} --lr {lr} "
            f"--size {img_size} --batch {batch_size} --device {shlex.quote(device)} "
            f"--arch {shlex.quote(arch)} "
        )
        if allow_empty:
            cmd += " --allow-empty"
        if pretrained:
            cmd += " --pretrained"
        if resume:
            cmd += f" --resume {shlex.quote(outp)}"

        self._append_train_log("")
        self._append_train_log("Launching Tip Segmenter training (no SAM):")
        self._append_train_log(f"  Images: {self.target_images_dir}")
        self._append_train_log(f"  Masks:  {self.target_masks_dir}")
        self._append_train_log(f"  Output: {outp}")
        self._append_train_log(f"  Steps: {steps}, LR: {lr}, Size: {img_size}, Device: {device}")
        self._append_train_log(f"  Batch: {batch_size}, Allow empty: {allow_empty}")
        self._append_train_log(f"  Arch: {arch}, Pretrained: {pretrained}")
        if resume:
            self._append_train_log("  Resume: True")
        self._append_train_log("Command: " + cmd)

        try:
            self._train_proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                self._append_train_log(f"Process started (pid={self._train_proc.pid})")
            except Exception:
                pass
            threading.Thread(target=self._train_reader_thread, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Target Training", str(e))

    def _load_target_model(self):
        try:
            ckpt_path = self.target_out_var.get().strip()
            if not ckpt_path or not os.path.exists(ckpt_path):
                messagebox.showwarning("Missing file", "Pick a valid tip model .pth file.")
                return
            map_dev = (self.target_device_var.get().strip() or self.e_dev.get().strip() or "cpu")
            from tip_segmenter_model import load_tipseg_checkpoint

            model, meta = load_tipseg_checkpoint(ckpt_path, device=map_dev)
            self.tipseg_model = model
            self.tipseg_meta = meta
            try:
                self.target_use_tipseg.set(True)
                # sync UI threshold if checkpoint provides one
                if "threshold" in meta:
                    self.target_tipseg_thresh.set(float(meta["threshold"]))
            except Exception:
                pass

            messagebox.showinfo("Model", "Tip segmenter loaded.")
            try:
                self._refresh_weights_badge()
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    # ----- Target filter helpers -----
    def _mask_features(self, mask_bool: np.ndarray, img_rgb: np.ndarray):
        ys, xs = np.nonzero(mask_bool)
        if xs.size == 0:
            return None
        H, W = mask_bool.shape[:2]
        area = float(xs.size)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        w = max(1, x2 - x1 + 1)
        h = max(1, y2 - y1 + 1)
        aspect = max(w / h, h / w)
        area_frac = area / float(H * W)

        if img_rgb.dtype != np.uint8:
            rgb8 = np.clip(img_rgb, 0, 255).astype(np.uint8)
        else:
            rgb8 = img_rgb
        hsv = cv2.cvtColor(rgb8, cv2.COLOR_RGB2HSV)
        vals = hsv[mask_bool]
        h_mean, s_mean, v_mean = vals.mean(axis=0)
        return {
            "area_frac": float(area_frac),
            "aspect": float(aspect),
            "h": float(h_mean),
            "s": float(s_mean),
            "v": float(v_mean),
        }

    def _classify_masks(self, masks, img_rgb):
        if self.target_clf is None:
            return None
        meta = self.target_clf_meta or {}
        input_size = int(meta.get("input_size", 224))
        mean = meta.get("mean", [0.485, 0.456, 0.406])
        std = meta.get("std", [0.229, 0.224, 0.225])
        device = meta.get("device", "cpu")

        try:
            from torchvision import transforms
        except Exception:
            return None

        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size)),
            transforms.Normalize(mean=mean, std=std),
        ])

        scores = []
        for m in masks:
            seg = m.get("segmentation")
            if seg is None:
                scores.append(0.0)
                continue
            mask = seg.astype(bool)
            ys, xs = np.nonzero(mask)
            if xs.size == 0:
                scores.append(0.0)
                continue
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            # pad by 20%
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            pad = int(max(w, h) * 0.2)
            H, W = img_rgb.shape[:2]
            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
            x2 = min(W - 1, x2 + pad); y2 = min(H - 1, y2 + pad)
            crop = img_rgb[y1:y2 + 1, x1:x2 + 1].copy()
            m_crop = mask[y1:y2 + 1, x1:x2 + 1]
            crop = np.where(m_crop[..., None], crop, 0)

            x = tfm(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                logit = self.target_clf(x).squeeze(0).squeeze(0)
                score = float(torch.sigmoid(logit).item())
            scores.append(score)
        return scores

    def _learn_target_filter(self):
        if not self.sr or not self.sr.masks:
            messagebox.showwarning("Target Filter", "Run segmentation first.")
            return
        sel = list(self.lb.curselection())
        if not sel:
            messagebox.showwarning("Target Filter", "Select one or more target masks.")
            return
        feats = []
        img = self.sr.img_color
        for idx in sel:
            m = self.sr.masks[idx]["segmentation"].astype(bool)
            f = self._mask_features(m, img)
            if f:
                feats.append(f)
        if not feats:
            messagebox.showwarning("Target Filter", "No valid masks selected.")
            return
        # compute mean/std
        def _mean_std(key):
            vals = np.array([f[key] for f in feats], dtype=np.float32)
            return float(vals.mean()), float(vals.std())
        stats = {
            "area_mean": _mean_std("area_frac")[0],
            "area_std": _mean_std("area_frac")[1],
            "aspect_mean": _mean_std("aspect")[0],
            "aspect_std": _mean_std("aspect")[1],
            "h_mean": _mean_std("h")[0],
            "h_std": _mean_std("h")[1],
            "s_mean": _mean_std("s")[0],
            "s_std": _mean_std("s")[1],
            "v_mean": _mean_std("v")[0],
            "v_std": _mean_std("v")[1],
            "count": len(feats),
        }
        self.target_filter_stats = stats
        messagebox.showinfo("Target Filter", f"Learned filter from {len(feats)} mask(s).")

    def _reset_target_filter(self):
        self.target_filter_stats = None
        messagebox.showinfo("Target Filter", "Filter reset.")

    def _apply_target_filter(self, masks, img_rgb):
        # If classifier is enabled and loaded, use it to filter
        if bool(self.target_use_classifier.get()) and self.target_clf is not None:
            scores = self._classify_masks(masks, img_rgb)
            if scores is None:
                return masks
            thresh = float(self.target_cls_thresh.get())
            kept = [m for m, s in zip(masks, scores) if s >= thresh]
            if not kept and bool(self.target_cls_keep_best.get()) and masks:
                best_idx = int(np.argmax(scores))
                kept = [masks[best_idx]]
            return kept

        stats = self.target_filter_stats
        if not stats:
            return masks
        k = float(self.target_filter_k.get())
        # floors to avoid overly strict filters with tiny std
        area_std = max(stats.get("area_std", 0.0), 0.003)
        aspect_std = max(stats.get("aspect_std", 0.0), 0.25)
        h_std = max(stats.get("h_std", 0.0), 5.0)
        s_std = max(stats.get("s_std", 0.0), 8.0)
        v_std = max(stats.get("v_std", 0.0), 8.0)

        kept = []
        for m in masks:
            seg = m.get("segmentation")
            if seg is None:
                continue
            f = self._mask_features(seg.astype(bool), img_rgb)
            if not f:
                continue
            if abs(f["area_frac"] - stats["area_mean"]) > k * area_std:
                continue
            if abs(f["aspect"] - stats["aspect_mean"]) > k * aspect_std:
                continue
            if abs(f["h"] - stats["h_mean"]) > k * h_std:
                continue
            if abs(f["s"] - stats["s_mean"]) > k * s_std:
                continue
            if abs(f["v"] - stats["v_mean"]) > k * v_std:
                continue
            kept.append(m)
        return kept

    def _after_segment_prompt(self):
        """After each Segment, optionally ask to add example, then auto-advance & segment next."""
        if not self.train_auto_prompt.get():
            return
        try:
            if messagebox.askyesno("Training", "Add this segmentation as a training example? (selected masks if any, else all)"):
                self._add_current_to_training()
            if self.batch_images and messagebox.askyesno("Continue", "Segment next image with same settings?"):
                self.next_image()
                self.segment()
        except Exception:
            pass

    # ════════════════════════════════════════════════════════════════════════
    # Hidden Structure (Occlusion Augmentation) Training Methods
    # ════════════════════════════════════════════════════════════════════════

    def _pick_occ_data_folder(self):
        """Pick folder containing leaf images for occlusion training."""
        d = filedialog.askdirectory(title="Select folder with leaf images")
        if d:
            self.occ_data_var.set(d)
            # Count images
            count = len(list(Path(d).rglob("*.png"))) + len(list(Path(d).rglob("*.jpg")))
            self._append_train_log(f"Selected folder: {d} ({count} images found)")

    def _launch_occlusion_training(self):
        """Launch occlusion_augmentation.py training script."""
        data_dir = self.occ_data_var.get().strip()
        if not data_dir or not os.path.isdir(data_dir):
            messagebox.showwarning("Hidden Structure", "Pick a valid folder with leaf images.")
            return

        # Check for images
        img_count = len(list(Path(data_dir).rglob("*.png"))) + len(list(Path(data_dir).rglob("*.jpg")))
        if img_count == 0:
            messagebox.showwarning("Hidden Structure", "No PNG/JPG images found in the selected folder.")
            return

        # Get checkpoint (fallback to Model panel)
        ckpt = self.occ_ckpt_var.get().strip() or self.e_ckpt.get().strip()
        if not ckpt or not os.path.exists(ckpt):
            messagebox.showwarning("Hidden Structure", "Pick a valid SAM2 checkpoint (.pt).")
            return

        cfg = self.occ_cfg_var.get().strip()
        # Skip auto-detect placeholder or empty
        if cfg in ["(auto-detect)", "", "auto"]:
            cfg = None
        out_path = self.occ_out_var.get().strip() or str(Path.home() / "sam2_hidden_structure.pth")
        img_size = int(self.occ_size_var.get())
        steps = int(self.occ_steps_var.get())
        lr = float(self.occ_lr_var.get())
        occ_min = float(self.occ_min_var.get())
        occ_max = float(self.occ_max_var.get())
        occ_count_min = int(self.occ_count_min_var.get())
        occ_count_max = int(self.occ_count_max_var.get())
        device = self.occ_device_var.get().strip() or "cuda"

        # Build command to run occlusion_augmentation.py
        script_path = Path(__file__).parent / "occlusion_augmentation.py"
        if not script_path.exists():
            messagebox.showerror("Hidden Structure", f"Cannot find occlusion_augmentation.py at:\n{script_path}")
            return

        py = shlex.quote(sys.executable)
        cmd = (
            f"{py} {shlex.quote(str(script_path))} "
            f"--data {shlex.quote(data_dir)} "
            f"--ckpt {shlex.quote(ckpt)} "
            f"--out {shlex.quote(out_path)} "
            f"--size {img_size} "
            f"--steps {steps} "
            f"--lr {lr} "
            f"--occ-min {occ_min:.2f} "
            f"--occ-max {occ_max:.2f} "
            f"--occ-count-min {occ_count_min} "
            f"--occ-count-max {occ_count_max} "
            f"--device {device}"
        )

        # Add config if provided
        if cfg:
            cmd += f" --config {shlex.quote(cfg)}"

        self._append_train_log("")
        self._append_train_log("=" * 60)
        self._append_train_log("Launching Hidden Structure Training:")
        self._append_train_log(f"  Data: {data_dir} ({img_count} images)")
        self._append_train_log(f"  Checkpoint: {ckpt}")
        self._append_train_log(f"  Config: {cfg if cfg else '(auto-detect from checkpoint)'}")
        self._append_train_log(f"  Output: {out_path}")
        self._append_train_log(f"  Occlusion: {occ_min*100:.0f}%-{occ_max*100:.0f}%, {occ_count_min}-{occ_count_max} occluders")
        self._append_train_log(f"  Steps: {steps}, LR: {lr}, Size: {img_size}, Device: {device}")
        self._append_train_log("=" * 60)
        self._append_train_log(f"Command: {cmd}")
        self._append_train_log("")

        try:
            self._occ_train_proc = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).parent)
            )
            threading.Thread(target=self._occ_train_reader_thread, daemon=True).start()
            self.set_status("Hidden Structure training started...", "info")
        except Exception as e:
            messagebox.showerror("Hidden Structure", f"Failed to start training:\n{e}")

    def _occ_train_reader_thread(self):
        """Read output from occlusion training process."""
        p = getattr(self, "_occ_train_proc", None)
        if not p:
            return
        for raw in iter(p.stdout.readline, b""):
            line = raw.decode(errors="replace").rstrip()
            self.root.after(0, lambda s=line: self._append_train_log(s))
        p.wait()
        code = p.returncode
        self.root.after(0, lambda: self._append_train_log(f"\nHidden Structure training finished with code {code}"))
        if code == 0:
            self.root.after(0, lambda: self.set_status("Hidden Structure training completed!", "success"))
        else:
            self.root.after(0, lambda: self.set_status(f"Training failed with code {code}", "error"))

    def _load_occlusion_model(self):
        """Load the trained hidden structure model into the predictor."""
        out_path = self.occ_out_var.get().strip()
        if not out_path or not os.path.exists(out_path):
            # Ask user to pick a file
            out_path = filedialog.askopenfilename(
                title="Select trained model",
                filetypes=[("PyTorch model", "*.pth *.pt"), ("All files", "*.*")]
            )
            if not out_path:
                return
            self.occ_out_var.set(out_path)

        try:
            device = self.e_dev.get().strip() or "cpu"
            checkpoint = torch.load(out_path, map_location=device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            self.sam2_model.load_state_dict(state_dict, strict=False)
            self.sam2_model.eval()

            meta = checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}
            self._append_train_log(f"Loaded hidden structure model from: {out_path}")
            if meta:
                self._append_train_log(f"  Model info: {meta}")

            messagebox.showinfo("Model Loaded", f"Hidden structure model loaded successfully!\n\n{out_path}")
            self.set_status("Hidden structure model loaded", "success")

        except Exception as e:
            messagebox.showerror("Load Failed", f"Failed to load model:\n{e}")
            self._append_train_log(f"ERROR loading model: {e}")

    def _rebuild_mask_list(self):
        self.lb.delete(0, tk.END)
        if not self.sr or not self.sr.masks:
            return
        for i, m in enumerate(self.sr.masks):
            meta = m.get("meta", {}) or {}
            tag = ""
            if meta.get("predicted"):
                tag += " [PRED]"
            if meta.get("split"):
                tag += " [SPLIT]"
            if "refined_from" in meta:
                tag += f" [REF:{int(meta['refined_from']):03d}]"
            self.lb.insert(tk.END, f"[{i:03d}]{tag} area={int(m['area'])} bbox={list(map(int, m['bbox']))}")



    def delete_selected_masks(self, event=None):
        """Remove currently selected masks from the result and refresh UI."""
        if not self.sr or not self.sr.masks:
            messagebox.showwarning("No masks", "Run segmentation first.")
            return
        sel = list(self.lb.curselection())
        if not sel:
            messagebox.showwarning("No selection", "Select one or more masks to delete.")
            return

        # delete from the end to avoid index shifts
        for idx in sorted(sel, reverse=True):
            if 0 <= idx < len(self.sr.masks):
                del self.sr.masks[idx]

        # refresh list + preview
        if hasattr(self, "_picks"):
            self._picks.clear()
            if hasattr(self, "_pick_status"):
                self._pick_status.configure(text="")
        self._rebuild_mask_list()
        self._sync_listbox_selection_from_picks()
        # Clear preview or show current base/enhanced image again
        if self.img_preview is not None:
            self.show_image(self.img_preview)
        elif self.img is not None:
            self.show_image(self.img)
        else:
            self.canvas.delete("all")

    def _combine_masks(self, idxs):
        """
        Union a list of mask indices into the first index; delete the rest.
        Recomputes segmentation, bbox and area. Keeps other fields from the
        'kept' (first) mask when present.
        """
        if not self.sr or not self.sr.masks or len(idxs) < 2:
            return False

        # Sort and choose one to keep (smallest idx)
        idxs = sorted(set(int(i) for i in idxs if 0 <= int(i) < len(self.sr.masks)))
        if len(idxs) < 2:
            return False

        keep = idxs[0]
        others = idxs[1:]

        # Union the boolean masks
        base = self.sr.masks[keep]
        union = base["segmentation"].astype(bool)
        for j in others:
            union |= self.sr.masks[j]["segmentation"].astype(bool)

        # Recompute bbox + area
        ys, xs = np.nonzero(union)
        if xs.size:
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            bbox = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
        else:
            bbox = [0, 0, 0, 0]

        # Update the kept mask
        base["segmentation"] = union.astype(np.uint8)
        base["bbox"] = bbox
        base["area"] = float(union.sum())

        # Delete others (highest first to avoid reindex issues)
        for j in sorted(others, reverse=True):
            del self.sr.masks[j]

        return True
    

    def on_predict_extend(self):
        sel = list(self.lb.curselection())
        if len(sel) != 1:
            messagebox.showwarning("Pick one", "Select exactly one mask to extend.")
            return
        i = sel[0]
        base = self.sr.masks[i]["segmentation"].astype(bool)

        # avoid invading other masks
        forbid = np.zeros_like(base, dtype=bool)
        for j, mm in enumerate(self.sr.masks):
            if j != i:
                forbid |= mm["segmentation"].astype(bool)

        mode = getattr(self, "extend_mode", tk.StringVar(value="auto")).get().lower()
        if mode in ("ml", "model", "sam2"):
            pred = self._predict_extend_ml(base, forbid)
        else:
            pred = predict_extend_mask(base, method=mode, strength=1.0, forbid_mask=forbid)

        if pred is None or not pred.any():
            return

        added = np.logical_and(pred, ~base)

        ys, xs = np.nonzero(pred)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bbox = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]

        self.sr.masks.append({
            "segmentation": pred.astype(np.uint8),
            "bbox": bbox,
            "area": float(pred.sum()),
            "meta": {"predicted": True, "pred_mode": mode, "extended_bool": added.astype(np.uint8)}
        })
        self._rebuild_mask_list()
        self.lb.selection_clear(0, tk.END)
        self.lb.selection_set(tk.END)
        self.on_select_mask()

    def _sample_points_in_mask(self, mask_bool: np.ndarray, n: int = 3):
        ys, xs = np.nonzero(mask_bool)
        if xs.size == 0:
            return None
        pts = []
        cx, cy = int(xs.mean()), int(ys.mean())
        pts.append([cx, cy])
        if n > 1:
            count = n - 1
            idx = np.random.choice(xs.size, size=count, replace=xs.size < count)
            for k in idx:
                pts.append([int(xs[k]), int(ys[k])])
        return np.asarray(pts, dtype=np.float32)

    def _keep_components_touching_base(self, mask_bool: np.ndarray, base_bool: np.ndarray):
        m = (mask_bool.astype(np.uint8) > 0).astype(np.uint8)
        num, labels = cv2.connectedComponents(m, connectivity=4)
        if num <= 1:
            return mask_bool
        keep = np.zeros_like(mask_bool, dtype=bool)
        for lbl in range(1, num):
            comp = labels == lbl
            if np.logical_and(comp, base_bool).any():
                keep |= comp
        if not keep.any():
            return base_bool
        return keep

    def _predict_extend_ml(self, base_mask_bool: np.ndarray, forbid_mask: np.ndarray | None):
        if self.sam2_model is None:
            messagebox.showwarning("No model", "Load the SAM2 model first.")
            return None
        if SAM2ImagePredictor is None:
            messagebox.showwarning("Missing SAM2 predictor", "SAM2ImagePredictor is not available in this environment.")
            return None

        base_img = self._base_image()
        if base_img is None:
            return None

        seg_img = self._enhance_pipeline(self._apply_denoise(base_img))
        predictor = getattr(self, "_ml_predictor", None)
        if predictor is None or getattr(predictor, "model", None) is not self.sam2_model:
            predictor = SAM2ImagePredictor(self.sam2_model)
            self._ml_predictor = predictor

        try:
            predictor.set_image(seg_img)
        except Exception as e:
            _show_err("ML Extend: set_image", e)
            return None

        pts = self._sample_points_in_mask(base_mask_bool, n=5)
        if pts is None:
            return None
        labels = np.ones((pts.shape[0],), dtype=np.int32)

        ys, xs = np.nonzero(base_mask_bool)
        if xs.size == 0:
            return None
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        pad = int(0.5 * max(w, h))
        H, W = seg_img.shape[:2]
        bx1 = max(0, x1 - pad)
        by1 = max(0, y1 - pad)
        bx2 = min(W - 1, x2 + pad)
        by2 = min(H - 1, y2 + pad)
        box = np.array([bx1, by1, bx2, by2], dtype=np.float32)

        try:
            masks, ious, _ = predictor.predict(
                point_coords=pts,
                point_labels=labels,
                box=box,
                multimask_output=True,
                return_logits=False,
            )
        except Exception as e:
            _show_err("ML Extend: predict", e)
            return None

        if masks is None:
            return None

        if masks.ndim == 2:
            best = masks
        else:
            best_idx = 0
            scores = []
            for k in range(masks.shape[0]):
                mk = masks[k].astype(bool)
                inter = np.logical_and(mk, base_mask_bool).sum()
                union = np.logical_or(mk, base_mask_bool).sum()
                scores.append(inter / max(1, union))
            if scores and max(scores) > 0:
                best_idx = int(np.argmax(scores))
            elif ious is not None and np.size(ious) > 0:
                best_idx = int(np.argmax(ious))
            best = masks[best_idx]

        pred = best.astype(bool)
        pred = self._keep_components_touching_base(pred, base_mask_bool)
        if forbid_mask is not None:
            pred = np.logical_and(pred, ~forbid_mask)
        pred = np.logical_or(pred, base_mask_bool)
        return pred




    def _apply_combine_from_picks(self):
        """Combine using the indices the user clicked in Preview."""
        if not getattr(self, "_picks", None):
            messagebox.showwarning("No picks", "Click two or more segments, then press Combine.")
            return
        idxs = sorted(self._picks)
        if len(idxs) < 2:
            messagebox.showwarning("Not enough", "Pick at least two segments to combine.")
            return

        if self._combine_masks(idxs):
            # Clear picks and refresh UI
            keep = idxs[0]
            self._picks.clear()
            self._rebuild_mask_list()
            self.lb.selection_clear(0, tk.END)
            if 0 <= keep < len(self.sr.masks):
                self.lb.selection_set(keep)
            if hasattr(self, "_pick_status"):
                self._pick_status.configure(text="")
            self._render_preview()
        else:
            messagebox.showwarning("Combine", "Couldn’t combine those selections.")

    def _toggle_listbox_selection(self, event):
        """Toggle listbox item selection with Ctrl/Cmd click."""
        try:
            idx = self.lb.nearest(event.y)
            if idx < 0:
                return "break"
            if idx in self.lb.curselection():
                self.lb.selection_clear(idx)
            else:
                self.lb.selection_set(idx)
            self.on_select_mask()
            return "break"
        except Exception:
            return "break"


    def combine_selected_masks(self):
        """Combine rows selected in the Masks listbox."""
        if not self.sr or not self.sr.masks:
            messagebox.showwarning("No masks", "Run segmentation first.")
            return

        sel = list(self.lb.curselection())
        if len(sel) < 2:
            messagebox.showwarning("Not enough", "Select two or more masks to combine.")
            return

        if self._combine_masks(sel):
            # Replace selection with the single kept index
            kept = min(sel)
            self._rebuild_mask_list()
            self.lb.selection_clear(0, tk.END)
            if 0 <= kept < len(self.sr.masks):
                self.lb.selection_set(kept)
            if hasattr(self, "_picks"):
                self._picks.clear()
                if hasattr(self, "_pick_status"):
                    self._pick_status.configure(text="")
            self._render_preview()
        else:
            messagebox.showwarning("Combine", "Couldn’t combine those selections.")

    def refine_selected_masks(self):
        """Re-run segmentation on selected mask bboxes and add new masks (keeps originals)."""
        if not self.sr or not self.sr.masks:
            messagebox.showwarning("No masks", "Run segmentation first.")
            return
        if self.sam2_model is None:
            messagebox.showwarning("No model", "Load the SAM2 model first.")
            return
        sel = list(self.lb.curselection())
        if not sel:
            messagebox.showwarning("No selection", "Select one or more masks to refine.")
            return
        self._show_busy("Refining selected masks…")
        threading.Thread(target=self._refine_selected_worker, args=(sel,), daemon=True).start()

    def _refine_selected_worker(self, sel_idxs):
        try:
            base = self._base_image()
            if base is None:
                raise RuntimeError("No image loaded.")
            seg_img = self._apply_enhance_pipeline(base)
            full_h, full_w = seg_img.shape[:2]
            gen = self.build_mask_generator()

            new_masks = []
            # re-segment within each selected mask bbox (with padding)
            for idx in sel_idxs:
                if idx < 0 or idx >= len(self.sr.masks):
                    continue
                m = self.sr.masks[idx]
                x, y, w, h = map(int, m.get("bbox", (0, 0, 0, 0)))
                if w <= 1 or h <= 1:
                    continue
                pad = max(4, int(0.10 * max(w, h)))
                x1 = max(0, x - pad); y1 = max(0, y - pad)
                x2 = min(full_w, x + w + pad); y2 = min(full_h, y + h + pad)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                roi = seg_img[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                roi_masks = gen.generate(roi)
                roi_masks = dedupe_by_mask_iou(roi_masks, iou_thresh=0.80)
                split_min = max(20, int(self.m_min_mask_region_area.get() * 0.10))
                roi_masks = split_masks_by_cc(roi_masks, min_area=split_min)

                for rm in roi_masks:
                    seg = rm.get("segmentation")
                    if not isinstance(seg, np.ndarray):
                        continue
                    seg_u8 = (seg > 0).astype(np.uint8)
                    if seg_u8.ndim != 2:
                        continue
                    rh, rw = seg_u8.shape[:2]
                    full_seg = np.zeros((full_h, full_w), dtype=np.uint8)
                    full_seg[y1:y1 + rh, x1:x1 + rw] = seg_u8

                    bx, by, bw, bh = rm.get("bbox", (0, 0, 0, 0))
                    new_bbox = [int(x1 + bx), int(y1 + by), int(bw), int(bh)]

                    meta = dict(rm.get("meta", {}))
                    meta["refined_from"] = int(idx)
                    meta["roi_bbox"] = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

                    nm = dict(rm)
                    nm["segmentation"] = full_seg
                    nm["bbox"] = new_bbox
                    nm["area"] = float(seg_u8.sum())
                    nm["meta"] = meta
                    new_masks.append(nm)

            def _update():
                if not new_masks:
                    self._hide_busy()
                    messagebox.showinfo("Refine", "No new masks produced for the selected regions.")
                    return
                start = len(self.sr.masks)
                self.sr.masks.extend(new_masks)
                self._rebuild_mask_list()
                self.lb.selection_clear(0, tk.END)
                for i in range(start, len(self.sr.masks)):
                    self.lb.selection_set(i)
                if hasattr(self, "_pick_status"):
                    self._pick_status.configure(text=f"Added {len(new_masks)} masks")
                self._render_preview()
                self._hide_busy()

            self.root.after(0, _update)
        except Exception as e:
            self.root.after(0, lambda: (self._hide_busy(), messagebox.showerror("Refine failed", str(e))))


    def clear_all_masks(self):
        if not self.sr:
            return
        self.sr.masks = []
        self._rebuild_mask_list()
        if self.img_preview is not None:
            self.show_image(self.img_preview)
        elif self.img is not None:
            self.show_image(self.img)
        else:
            self.canvas.delete("all")


    # ---- File/model ----
    def pick_ckpt(self):
        p = filedialog.askopenfilename(title="Select SAM2 checkpoint (.pt)")
        if p:
            self.e_ckpt.delete(0, tk.END)
            self.e_ckpt.insert(0, p)

    def pick_cfg(self):
        p = filedialog.askopenfilename(title="Select SAM2 config YAML",
                                       filetypes=[("YAML","*.yaml *.yml"), ("All","*.*")])
        if p:
            self.e_cfg.delete(0, tk.END)
            self.e_cfg.insert(0, p)

    def open_image(self):
        """Pick an image, remember original, apply current rotation, and show it."""
        p = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Images", "*.tif *.tiff *.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        if not p:
            return

        self.set_status(f"Loading {Path(p).name}...", "processing")
        self.img_path = p
        arr = ensure_uint8_rgb(Image.open(p))
        self.img_orig = arr                 # keep the unmodified original
        self.img_preview = None
        self.sr = None
        self.lb.delete(0, tk.END)

        # apply current angle to produce the working image
        self.img = self._base_image()
        self.show_image(self.img)

        # Update status with image info
        h, w = arr.shape[:2]
        self.set_status(f"Loaded: {Path(p).name} ({w}×{h})", "success")

    def open_folder(self):
        d = filedialog.askdirectory(title="Open folder with images")
        if not d:
            return
        exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
        imgs = [str(p) for p in sorted(Path(d).iterdir())
                if p.suffix.lower() in exts and p.is_file()]
        if not imgs:
            messagebox.showwarning("No images", "That folder has no supported images.")
            return

        self.batch_dir = d
        self.batch_images = imgs
        self.batch_idx = 0
        self._load_batch_index(0)

    def load_bundle(self):
            """Load a .pt/.pth bundle that contains cfg + weights. If cfg is a dict/DictConfig,
            dump it to a temp YAML and point Hydra to that folder so build_sam2() gets a string name."""
            if _sam2_import_error is not None:
                _show_err("SAM2 import error", f"Couldn't import sam2 modules:\n{_sam2_import_error}")
                return

            p = filedialog.askopenfilename(
                title="Open SAM2 bundle",
                filetypes=[("Torch files","*.pt *.pth"), ("All files","*.*")]
            )
            if not p:
                return

            dev = (self.e_dev.get().strip() or "cpu")

            try:
                bundle = torch.load(p, map_location="cpu", weights_only=False)  # explicit to silence warning
                logging.info("Bundle keys: %s", list(bundle.keys()))
            except Exception as e:
                _show_err("load_bundle/read", e)
                return

            tmp_ckpt = None
            tmp_cfg_dir = None
            try:
                apply_pp = bool(bundle.get("apply_postprocessing", self.chk_post.get()))
                meta     = bundle.get("meta") or {}
                cfg_in   = bundle.get("cfg")               # may be dict/DictConfig/None
                ck_bytes = bundle.get("ckpt_bytes") or bundle.get("checkpoint_bytes")
                ck_path  = bundle.get("ckpt_path") if isinstance(bundle.get("ckpt_path"), str) else None

                # ---- prepare checkpoint path ----
                if isinstance(ck_bytes, (bytes, bytearray)):
                    import tempfile, os
                    fd, tmp_path = tempfile.mkstemp(suffix=".pt")
                    with os.fdopen(fd, "wb") as fh:
                        fh.write(ck_bytes)
                    tmp_ckpt = tmp_path
                    logging.info("Using checkpoint from bytes -> %s", tmp_ckpt)
                    use_ckpt_path = tmp_ckpt
                elif ck_path and os.path.exists(ck_path):
                    logging.info("Using checkpoint at %s", ck_path)
                    use_ckpt_path = ck_path
                elif "state_dict" in bundle:
                    use_ckpt_path = None   # we’ll load state_dict after constructing model
                else:
                    raise RuntimeError("Bundle has neither 'ckpt_bytes', 'ckpt_path', nor 'state_dict'.")

                # ---- resolve config into something build_sam2() accepts (a STRING name) ----
                cfg_name_for_build = None

                # Case A: cfg provided as dict/DictConfig -> dump to temp YAML + point Hydra there
                if cfg_in is not None and not isinstance(cfg_in, str):
                    try:
                        cfg_dc = OmegaConf.create(cfg_in) if not ('DictConfig' in str(type(cfg_in))) else cfg_in
                        import tempfile, os
                        tmp_cfg_dir = tempfile.mkdtemp(prefix="sam2cfg_")
                        tmp_yaml = os.path.join(tmp_cfg_dir, "bundle_cfg.yaml")
                        with open(tmp_yaml, "w") as f:
                            f.write(OmegaConf.to_yaml(cfg_dc))
                        _hydra_reinit_to_dir(tmp_cfg_dir)
                        cfg_name_for_build = "bundle_cfg"  # stem of the temp yaml
                        logging.info("Using cfg from bundle (dict) via temp YAML at %s", tmp_yaml)
                    except Exception as e:
                        _show_err("load_bundle/cfg-dump", e)
                        return

                # Case B: bundle recorded a short config name -> use it
                if cfg_name_for_build is None:
                    cfg_short = meta.get("config_name") or bundle.get("cfg_short_name") \
                                or (self.e_cfg.get().strip() or "sam2.1_hiera_l")
                    # If user set SAM2_CONFIG_DIR, honor it so Hydra can find the yaml
                    cfg_dir_env = os.environ.get("SAM2_CONFIG_DIR")
                    if cfg_dir_env and os.path.isdir(cfg_dir_env):
                        _hydra_reinit_to_dir(cfg_dir_env)
                    cfg_name_for_build = cfg_short
                    logging.info("Using cfg short name: %s", cfg_name_for_build)

                # ---- build the model ----
                model = build_sam2(cfg_name_for_build, use_ckpt_path, device=dev, apply_postprocessing=apply_pp)

                # If only state_dict was in bundle, load it now
                if use_ckpt_path is None and "state_dict" in bundle:
                    model.load_state_dict(bundle["state_dict"], strict=False)

                self.sam2_model = model
                self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2_model)

                # reflect in UI
                self.e_ckpt.delete(0, tk.END); self.e_ckpt.insert(0, "[bundle]")
                self.e_cfg.delete(0, tk.END);  self.e_cfg.insert(0, str(meta.get("config_name") or "bundle_cfg"))
                _show_info(f"Loaded SAM2 bundle on device '{dev}'.", title="Model")
                try:
                    self._set_sam_weights_tag("base")
                except Exception:
                    pass

            except Exception as e:
                _show_err("load_bundle/build", e)
                self.sam2_model = None
                self.mask_generator = None
            finally:
                if tmp_ckpt and os.path.exists(tmp_ckpt):
                    try: os.remove(tmp_ckpt)
                    except Exception: pass
                if tmp_cfg_dir:
                    try:
                        import shutil
                        shutil.rmtree(tmp_cfg_dir, ignore_errors=True)
                    except Exception:
                        pass



    def _load_batch_index(self, i: int):
        if not self.batch_images:
            return
        # cache current masks before switching
        if getattr(self, "img_path", None) and self.sr is not None:
            try:
                self._batch_mask_cache[self.img_path] = self.sr
            except Exception:
                pass
        i = max(0, min(len(self.batch_images) - 1, int(i)))
        self.batch_idx = i
        p = self.batch_images[i]
        self.img_path = p

        # if we have cached masks, restore them
        cached = self._batch_mask_cache.get(p)
        if cached is not None:
            self.sr = cached
            self.img_orig = cached.img_color
            self.img = cached.img_color
            self.img_preview = None
            self._rebuild_mask_list()
            # show segmented image if available
            try:
                self.show_image(cached.img_seg if cached.img_seg is not None else self.img)
            except Exception:
                self.show_image(self.img)
        else:
            # same as open_image, but from a known path
            arr = ensure_uint8_rgb(Image.open(p))
            self.img_orig = arr
            self.img_preview = None
            self.sr = None
            self.lb.delete(0, tk.END)
            self.img = self._base_image()
            self.show_image(self.img)
        self._update_batch_status()

    def _update_batch_status(self):
        if not self.batch_images:
            self._batch_status.configure(text="")
        else:
            self._batch_status.configure(text=f"({self.batch_idx+1} / {len(self.batch_images)})")

    def next_image(self):
        if not self.batch_images:
            return
        self._load_batch_index(self.batch_idx + 1)

    def prev_image(self):
        if not self.batch_images:
            return
        self._load_batch_index(self.batch_idx - 1)


    ##### Segment one image
    def _segment_sync_for_array(self, arr_rgb_uint8, apply_filter: bool = True):
        arr_rgb_uint8 = self._apply_denoise(arr_rgb_uint8)
        seg_img = self._enhance_pipeline(arr_rgb_uint8)

        gen = self.build_mask_generator()
        masks = gen.generate(seg_img)
        masks = dedupe_by_mask_iou(masks, iou_thresh=0.80)
        split_min = max(20, int(self.m_min_mask_region_area.get() * 0.10))
        masks = split_masks_by_cc(masks, min_area=split_min)
        if apply_filter and bool(self.target_filter_enable.get()) and (self.target_filter_stats or self.target_clf is not None):
            masks = self._apply_target_filter(masks, arr_rgb_uint8)
        return masks, seg_img, arr_rgb_uint8

    ### batch segmenter

    def segment_all_batch(self):
        if not self.batch_images:
            # try to use dataset images if available
            if getattr(self, "target_images_dir", None):
                self._maybe_use_dataset_images_for_batch(self.target_images_dir, allow_empty=True)
            elif getattr(self, "train_images_dir", None):
                self._maybe_use_dataset_images_for_batch(self.train_images_dir, allow_empty=True)
            if not self.batch_images:
                messagebox.showwarning("No folder", "Open a folder or choose a dataset folder first.")
                return

        # Tip-only segmentation path (no SAM)
        if bool(self.target_use_tipseg.get()):
            if self.tipseg_model is None:
                messagebox.showwarning("No tip model", "Load the tip model first (Target Segment → Load tip model).")
                return

            angle = float(self.rot_angle.get())
            print(f"[Batch] TipSeg on {len(self.batch_images)} images with angle={angle:.2f}° …")
            if not messagebox.askyesno(
                "Segment ALL (Tip)",
                f"Run the tip model on {len(self.batch_images)} images using current settings?\n\n"
                f"Results will be kept in memory. Use 'Save Batch…' to write to disk.",
            ):
                return

            meta = self.tipseg_meta or {}
            input_size = int(meta.get("input_size", int(self.target_size_var.get() or 512)))
            device = str(meta.get("device", self.target_device_var.get().strip() or "cpu"))
            thr = float(self.target_tipseg_thresh.get())
            min_area = int(self.target_tipseg_min_area.get())
            keep_largest = bool(self.target_tipseg_keep_largest.get())
            use_tiles = bool(self.tipseg_use_tiles.get())
            tile_size = int(self.tipseg_tile_size.get())
            stride = int(self.tipseg_stride.get())
            color_guided = bool(self.tipseg_color_guided.get())
            color_min_area = int(self.tipseg_color_min_area.get())
            hue_low = int(self.tipseg_hue_low.get())
            hue_high = int(self.tipseg_hue_high.get())
            sat_min = int(self.tipseg_sat_min.get())
            val_min = int(self.tipseg_val_min.get())
            val_brown_max = int(self.tipseg_val_brown_max.get())
            min_leaf_pct = float(self.tipseg_min_leaf_pct.get())
            min_stress_pct = float(self.tipseg_min_stress_pct.get())
            stop_after_first = bool(self.tipseg_stop_after_first.get())

            total_masks = 0
            for idx, p in enumerate(self.batch_images, 1):
                try:
                    rgb = ensure_uint8_rgb(Image.open(p))
                    if abs(angle) > 1e-6:
                        rgb = self._rotate_any(rgb, angle)

                    if use_tiles:
                        mb = self._tipseg_sliding_window(
                            rgb,
                            input_size=input_size,
                            device=device,
                            threshold=thr,
                            min_area=min_area,
                            keep_largest=keep_largest,
                            tile_size=tile_size,
                            stride=stride,
                            color_guided=color_guided,
                            color_min_area=color_min_area,
                            hue_low=hue_low,
                            hue_high=hue_high,
                            sat_min=sat_min,
                            val_min=val_min,
                            val_brown_max=val_brown_max,
                            min_leaf_pct=min_leaf_pct,
                            min_stress_pct=min_stress_pct,
                            stop_after_first=stop_after_first,
                        )
                    else:
                        from tip_segmenter_model import predict_tip_mask
                        mb = predict_tip_mask(
                            self.tipseg_model,
                            rgb,
                            input_size=input_size,
                            device=device,
                            threshold=thr,
                            min_area=min_area,
                            keep_largest=keep_largest,
                        )

                    masks = []
                    if mb is not None and mb.any():
                        bbox, area = self._bbox_area_from_mask(mb)
                        masks = [{
                            "segmentation": mb.astype(np.uint8),
                            "area": float(area),
                            "bbox": bbox,
                            "predicted_iou": 1.0,
                            "stability_score": 1.0,
                            "meta": {"source": "tipseg"},
                        }]

                    self._batch_mask_cache[p] = SegResult(
                        masks=masks,
                        img_color=rgb,
                        img_seg=rgb,
                        rotate_applied=self.chk_rotate.get()
                    )
                    total_masks += len(masks)
                    print(f"  [{idx}/{len(self.batch_images)}] {Path(p).name}: {len(masks)} mask(s)")
                except Exception as e:
                    print(f"  [skip] {p}: {e}")

            messagebox.showinfo(
                "Batch done",
                f"Processed {len(self.batch_images)} images.\nTotal masks: {total_masks}\n\n"
                f"Results are cached in memory.\nUse 'Save Batch…' to write to disk.",
            )
            return

        # Default: SAM2 batch segmentation
        if self.sam2_model is None:
            messagebox.showwarning("No model", "Load the SAM2 model first.")
            return

        # reuse the current rotation/params everywhere
        angle = float(self.rot_angle.get())

        # small progress dialog in terminal
        print(f"[Batch] Processing {len(self.batch_images)} images with angle={angle:.2f}° …")

        # optional simple “are you sure”
        if not messagebox.askyesno("Segment ALL", f"Run SAM2 on {len(self.batch_images)} images using current settings?\n\nResults will be kept in memory. Use 'Save Batch…' to write to disk."):
            return

        total_masks = 0
        filter_fallbacks = 0
        for idx, p in enumerate(self.batch_images, 1):
            try:
                rgb = ensure_uint8_rgb(Image.open(p))
                # apply the SAME rotation picked in the UI
                if abs(angle) > 1e-6:
                    rgb = self._rotate_any(rgb, angle)

                masks, seg_img, color_img = self._segment_sync_for_array(rgb, apply_filter=True)
                # If target filter wiped everything, fall back to raw masks
                if (not masks) and bool(self.target_filter_enable.get()) and (self.target_filter_stats or self.target_clf is not None):
                    masks, seg_img, color_img = self._segment_sync_for_array(rgb, apply_filter=False)
                    filter_fallbacks += 1
                # cache per-image outputs in memory
                self._batch_mask_cache[p] = SegResult(
                    masks=masks,
                    img_color=color_img,
                    img_seg=seg_img,
                    rotate_applied=self.chk_rotate.get()
                )

                total_masks += len(masks)
                print(f"  [{idx}/{len(self.batch_images)}] {Path(p).name}: {len(masks)} masks")

            except Exception as e:
                print(f"  [skip] {p}: {e}")

        msg = f"Processed {len(self.batch_images)} images.\nTotal masks: {total_masks}"
        if filter_fallbacks:
            msg += f"\nFilter fallback used on {filter_fallbacks} images."
        msg += "\n\nResults are cached in memory.\nUse 'Save Batch…' to write to disk."
        messagebox.showinfo("Batch done", msg)

    def save_all_batch_results(self):
        """Save cached batch masks to disk."""
        if not self.batch_images:
            messagebox.showwarning("No folder", "Open a folder or choose a dataset folder first.")
            return
        if not self._batch_mask_cache:
            messagebox.showwarning("No cached masks", "Run Segment ALL first to cache results.")
            return

        out_root = filedialog.askdirectory(title="Choose output folder for batch masks")
        if not out_root:
            return
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        total_masks = 0
        for p in self.batch_images:
            sr = self._batch_mask_cache.get(p)
            if sr is None:
                continue
            masks = sr.masks or []
            color_img = sr.img_color

            stem = Path(p).stem
            img_dir = out_root / stem
            img_dir.mkdir(exist_ok=True, parents=True)

            rows = []
            erode_px = max(0, int(self.s_halo_erode.get())) if hasattr(self, "s_halo_erode") else 1
            feather_px = max(0, int(self.s_halo_feather.get())) if hasattr(self, "s_halo_feather") else 2

            for k, m in enumerate(masks, 1):
                seg_bool = m["segmentation"].astype(bool)
                mask_path = img_dir / f"{stem}_{k}.png"
                crop_path = img_dir / f"{stem}_crop_{k}.png"
                save_binary_mask(seg_bool, mask_path)
                save_masked_crop_rgba(color_img, seg_bool, m["bbox"], crop_path,
                                      erode_px=erode_px, feather_px=feather_px)
                rows.append({
                    "mask_idx": k,
                    "area_px": int(m["area"]),
                    "bbox": list(map(int, m["bbox"])),
                    "mask_png": str(mask_path),
                    "crop_png": str(crop_path),
                })

            csv_path = img_dir / f"{stem}_mask_manifest.csv"
            with open(csv_path, "w", newline="") as f:
                import csv as _csv
                w = _csv.DictWriter(f, fieldnames=["mask_idx","area_px","bbox","mask_png","crop_png"])
                w.writeheader(); w.writerows(rows)

            total_masks += len(masks)

        messagebox.showinfo("Batch saved", f"Saved masks for {len(self.batch_images)} images.\nTotal masks: {total_masks}\n\nSaved under:\n{out_root}")


        


    # ===== Rotation / knob helpers =====
    def _rotate_any(self, arr, deg):
        """Rotate arr by deg (CCW), expanding canvas so nothing gets clipped."""
        if abs(deg) < 1e-6:
            return arr
        h, w = arr.shape[:2]
        c = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(c, deg, 1.0)  # CCW positive
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw = int(h * sin + w * cos)
        nh = int(h * cos + w * sin)
        # translate to keep image centered
        M[0, 2] += (nw / 2) - c[0]
        M[1, 2] += (nh / 2) - c[1]
        return cv2.warpAffine(arr, M, (nw, nh),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    def _base_image(self):
        arr = self.img_orig if self.img_orig is not None else self.img
        if arr is None:
            return None
        ang = float(self.rot_angle.get())
        # guard with the new toggle
        if abs(ang) > 0.01 and bool(getattr(self, "chk_rotate", tk.BooleanVar(value=True)).get()):
            arr = self._rotate_any(arr, ang) # Rotate
        return arr


    def _set_angle(self, deg):
        """Set angle (clamped), redraw knob, and refresh preview of rotated base image."""
        deg = max(-180.0, min(180.0, float(deg)))
        self.rot_angle.set(deg)
        self._draw_knob()
        if self.img_orig is not None:
            self.img = self._base_image()
            self.img_preview = None
            self.show_image(self.img)

    def _angle_from_spin(self):
        try:
            self._set_angle(float(self.rot_angle.get()))
        except Exception:
            pass

    def _angle_from_xy(self, x, y):
        cx, cy = self._knob_center
        import math
        ang = math.degrees(math.atan2(cy - y, x - cx))  # 0° at +x axis, CCW positive
        return max(-180.0, min(180.0, ang))

    def _knob_down(self, e):
        self._set_angle(self._angle_from_xy(e.x, e.y))

    def _knob_drag(self, e):
        self._set_angle(self._angle_from_xy(e.x, e.y))

    def _draw_knob(self):
        if not self._knob:
            return
        cv = self._knob
        cv.delete("all")
        cx, cy = self._knob_center
        r = self._knob_r
        c = self.colors

        # Outer shadow ring (subtle 3D effect)
        cv.create_oval(cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2,
                       fill=c['bg_medium'], outline="")

        # Main knob face with gradient effect (simulated with rings)
        cv.create_oval(cx - r, cy - r, cx + r, cy + r,
                       fill=c['bg_pale'], outline=c['accent'], width=2)

        # Inner circle for depth
        inner_r = r - 4
        cv.create_oval(cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r,
                       fill="", outline=c['bg_light'], width=1)

        # Tick marks every 45° with varying intensity
        import math
        tick_len = max(4, r // 3)
        for i, a in enumerate(range(-180, 181, 45)):
            rad = math.radians(a)
            # Major ticks at 0, 90, -90, 180
            is_major = a % 90 == 0
            x0 = cx + (r - tick_len) * math.cos(rad)
            y0 = cy - (r - tick_len) * math.sin(rad)
            x1 = cx + (r - 2) * math.cos(rad)
            y1 = cy - (r - 2) * math.sin(rad)
            tick_color = c['accent'] if is_major else c['bg_light']
            tick_width = 2 if is_major else 1
            cv.create_line(x0, y0, x1, y1, fill=tick_color, width=tick_width)

        # Center dot
        center_r = 3
        cv.create_oval(cx - center_r, cy - center_r, cx + center_r, cy + center_r,
                       fill=c['bg_medium'], outline="")

        # Indicator line (like a real knob)
        deg = float(self.rot_angle.get())
        rad = math.radians(deg)
        line_start = center_r + 2
        line_end = r - 4
        x0 = cx + line_start * math.cos(rad)
        y0 = cy - line_start * math.sin(rad)
        x1 = cx + line_end * math.cos(rad)
        y1 = cy - line_end * math.sin(rad)
        cv.create_line(x0, y0, x1, y1, fill=c['accent'], width=3, capstyle="round")

        # Indicator tip dot
        dot_r = 3
        hx = cx + (r - 6) * math.cos(rad)
        hy = cy - (r - 6) * math.sin(rad)
        cv.create_oval(hx - dot_r, hy - dot_r, hx + dot_r, hy + dot_r,
                       fill=c['accent'], outline=c['text_light'], width=1)

    def _edge_darken(self, rgb):
        """Darken pixels near edges; helps SAM pick boundaries."""
        try:
            if not self.ed_on.get():
                return rgb
        except Exception:
            return rgb

        width  = max(1, int(self.ed_width.get()))
        amount = float(self.ed_amount.get())  # 0..1

        gray  = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 80, 180)

        if width > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width*2+1, width*2+1))
            edges = cv2.dilate(edges, k, iterations=1)

        mask = (edges > 0)[..., None]
        out  = rgb.astype(np.float32)
        out[mask] *= (1.0 - amount)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _enhance_pipeline(self, arr_rgb_uint8):
        """Apply selected enhancement stages in sequence."""
        x = arr_rgb_uint8
        if self.use_green.get():
            x = enhance_leaf_edges_rgb(x)  # stage 1
        if self.use_classic.get():
            x = preprocess_for_edges(      # stage 2
                x,
                brightness=self.s_brightness.get(),
                contrast=self.s_contrast.get(),
                use_unsharp=self.chk_unsharp.get(),
                unsharp_kernel_size=9, unsharp_sigma=10.0, unsharp_amount=1.5,
                use_laplacian=self.chk_laplacian.get(),
                gamma=self.s_gamma.get(),
            )
            # edge darken (last)
        x = self._edge_darken(x)
        return x

    # ---- Preview ----
    def preview_enhance(self):
        arr = self._base_image()
        if arr is None:
            self.set_status("No image loaded", "warning")
            messagebox.showwarning("No image", "Open an image first.")
            return

        self.set_status("Applying enhancements...", "processing")
        self.root.update_idletasks()

        arr2 = self._apply_enhance_pipeline(arr)
        self.img_preview = arr2
        self.show_image(arr2)

        # Count active enhancements for status
        active = []
        if self.use_green.get(): active.append("Green")
        if self.use_classic.get(): active.append("Classic")
        if getattr(self, 'use_white_balance', None) and self.use_white_balance.get(): active.append("WB")
        if getattr(self, 'use_lab', None) and self.use_lab.get(): active.append("LAB")
        if getattr(self, 'use_retinex', None) and self.use_retinex.get(): active.append("Retinex")
        if getattr(self, 'use_veg_index', None) and self.use_veg_index.get(): active.append(self.veg_index_type.get())
        if getattr(self, 'use_nlm', None) and self.use_nlm.get(): active.append("NLM")
        if getattr(self, 'use_guided', None) and self.use_guided.get(): active.append("Guided")
        if getattr(self, 'use_dog', None) and self.use_dog.get(): active.append("DoG")
        if getattr(self, 'use_tophat', None) and self.use_tophat.get(): active.append("TopHat")
        if getattr(self, 'use_shadow_highlight', None) and self.use_shadow_highlight.get(): active.append("S/H")
        if getattr(self, 'use_local_contrast', None) and self.use_local_contrast.get(): active.append("LocalC")
        if getattr(self, 'use_adaptive_gamma', None) and self.use_adaptive_gamma.get(): active.append("AdaptG")

        if active:
            self.set_status(f"Preview: {', '.join(active)}", "success")
        else:
            self.set_status("Preview: No enhancements active", "info")




    def show_image(self, arr):
        # keep the raw image; the renderer handles fit/zoom/pan
        self._img_for_preview = arr
        if getattr(self, "_fit_mode", True):
            # when in Fit mode, always re-center
            self._pan = [0, 0]
        self._render_preview()
    
    def _apply_denoise(self, x: np.ndarray) -> np.ndarray:
        """Optionally apply median/mean blur (RGB). Keeps dtype/shape."""
        img = x
        try:
            if bool(self.dn_median_on.get()):
                k = int(self.dn_median_ksize.get())
                if k < 3: k = 3
                if k % 2 == 0: k += 1
                k = min(k, 31)                       # keep it reasonable
                img = cv2.medianBlur(img, k)         # great for salt&pepper
            if bool(self.dn_mean_on.get()):
                k = int(self.dn_mean_ksize.get())
                if k < 3: k = 3
                if k % 2 == 0: k += 1
                k = min(k, 31)
                img = cv2.blur(img, (k, k))          # simple mean (Gaussian also fine)
        except Exception:
            # fail-safe: return original if user types odd values mid-edit
            pass
        return img
    
    def _darken_edges_rgb(self, img_rgb: np.ndarray, width_px: int, strength: float) -> np.ndarray:
        """
        Darken a narrow band around intensity edges.
        width_px: thickness of the edge band (1..31)
        strength: how much to darken (0..0.9), applied to V in HSV (safer than raw RGB).
        """
        if img_rgb is None or img_rgb.ndim != 3:
            return img_rgb
        h, w = img_rgb.shape[:2]
        width_px = max(1, min(int(width_px), 31))
        strength = max(0.0, min(float(strength), 0.95))

        # 1) Edge map (Sobel magnitude)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gxf  = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gyf  = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag  = cv2.magnitude(gxf, gyf)  # float
        # normalize to [0,1]
        mmin, mmax = float(mag.min()), float(mag.max())
        if mmax > mmin:
            mag = (mag - mmin) / (mmax - mmin)
        else:
            mag = np.zeros_like(mag, dtype=np.float32)

        # 2) Thicken the band to width_px (dilate)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width_px, width_px))
        band = cv2.dilate(mag, k)
        # soften edges a bit so the darkening fades
        band = cv2.GaussianBlur(band, (0, 0), sigmaX=width_px * 0.5)
        band = np.clip(band, 0.0, 1.0)

        # 3) Darken only the Value channel in HSV (reduces halo artifacts)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        H, S, V = cv2.split(hsv)
        # Darken: V' = V * (1 - strength * band)
        V = V * (1.0 - strength * band.astype(np.float32))
        hsv_out = cv2.merge([H, S, V])
        out = cv2.cvtColor(np.clip(hsv_out, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        return out

    
    def _apply_enhance_pipeline(self, arr_rgb_uint8):
        x = arr_rgb_uint8

        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 1: Color Correction (apply first for consistent colors)
        # ═══════════════════════════════════════════════════════════════════════

        # White Balance
        if getattr(self, 'use_white_balance', None) and self.use_white_balance.get():
            wb_type = self.white_balance_type.get()
            if wb_type == "grayworld":
                x = white_balance_grayworld(x)
            elif wb_type == "max_white":
                x = white_balance_max_white(x)

        # LAB Color Enhancement
        if getattr(self, 'use_lab', None) and self.use_lab.get():
            x = enhance_lab_green(x,
                                  l_factor=self.lab_l_factor.get(),
                                  a_shift=self.lab_a_shift.get())

        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 2: Illumination Correction
        # ═══════════════════════════════════════════════════════════════════════

        # Retinex
        if getattr(self, 'use_retinex', None) and self.use_retinex.get():
            ret_type = self.retinex_type.get()
            sigma = self.retinex_sigma.get()
            if ret_type == "single":
                x = single_scale_retinex(x, sigma=sigma)
            else:
                x = multi_scale_retinex(x, sigmas=(sigma//4, sigma, sigma*3))

        # Shadow/Highlight Correction
        if getattr(self, 'use_shadow_highlight', None) and self.use_shadow_highlight.get():
            x = shadow_highlight_correction(x,
                                            shadow_amount=self.shadow_amount.get(),
                                            highlight_amount=self.highlight_amount.get())

        # Morphological Top-hat
        if getattr(self, 'use_tophat', None) and self.use_tophat.get():
            x = morphological_tophat(x, kernel_size=self.tophat_size.get())

        # Adaptive Gamma
        if getattr(self, 'use_adaptive_gamma', None) and self.use_adaptive_gamma.get():
            x = adaptive_gamma(x)

        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 3: Original Enhancement Methods
        # ═══════════════════════════════════════════════════════════════════════

        # Green-aware enhancement
        if bool(self.use_green.get()):
            x = enhance_leaf_edges_rgb(x)

        # Classic preprocess
        if bool(self.use_classic.get()):
            x = preprocess_for_edges(
                x,
                brightness=self.s_brightness.get(),
                contrast=self.s_contrast.get(),
                use_unsharp=self.chk_unsharp.get(),
                unsharp_kernel_size=9, unsharp_sigma=10.0, unsharp_amount=1.5,
                use_laplacian=self.chk_laplacian.get(),
                gamma=self.s_gamma.get(),
            )

        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 4: Vegetation Index Enhancement
        # ═══════════════════════════════════════════════════════════════════════

        if getattr(self, 'use_veg_index', None) and self.use_veg_index.get():
            x = enhance_with_vegetation_index(x,
                                              index_type=self.veg_index_type.get(),
                                              blend=self.veg_index_blend.get())

        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 5: Denoising
        # ═══════════════════════════════════════════════════════════════════════

        # NLM Denoising (better edge preservation)
        if getattr(self, 'use_nlm', None) and self.use_nlm.get():
            x = denoise_nlm(x, h=self.nlm_h.get())

        # Original denoise (median / mean)
        x = self._apply_denoise(x)

        # Guided Filter (edge-preserving smoothing)
        if getattr(self, 'use_guided', None) and self.use_guided.get():
            x = guided_filter_enhance(x,
                                      radius=self.guided_radius.get(),
                                      eps=self.guided_eps.get())

        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 6: Edge & Contrast Enhancement (apply last)
        # ═══════════════════════════════════════════════════════════════════════

        # Local Contrast Normalization
        if getattr(self, 'use_local_contrast', None) and self.use_local_contrast.get():
            x = local_contrast_normalization(x, kernel_size=self.local_contrast_size.get())

        # Difference of Gaussians
        if getattr(self, 'use_dog', None) and self.use_dog.get():
            x = difference_of_gaussians(x,
                                        sigma1=self.dog_sigma1.get(),
                                        sigma2=self.dog_sigma2.get(),
                                        blend=self.dog_blend.get())

        # Edge darken as the final step
        if bool(self.ed_on.get()):
            x = self._darken_edges_rgb(x, self.ed_width.get(), self.ed_amount.get())

        return x

    # ----- Zoom/Pan helpers -----
    def _zoom_fit(self):
        self._fit_mode = True
        self._zoom = 1.0
        self._pan = [0, 0]
        self._render_preview()

    def _zoom_by(self, factor):
        # switch to custom zoom, scale around center
        self._fit_mode = False
        self._zoom = float(getattr(self, "_zoom", 1.0)) * float(factor)
        self._zoom = max(0.05, min(self._zoom, 20.0))  # sane bounds
        self._render_preview()

    def _on_wheel(self, event, delta=None):
        d = delta if delta is not None else event.delta
        self._zoom_by(1.1 if d > 0 else 1/1.1)

    def _pan_start(self, event):
        if getattr(self, "_img_for_preview", None) is None:
            return
        self._drag_start = (event.x, event.y)

    def _pan_move(self, event):
        if getattr(self, "_img_for_preview", None) is None or getattr(self, "_drag_start", None) is None:
            return
        if getattr(self, "_fit_mode", True):
            # first drag takes you out of Fit mode so panning sticks
            self._fit_mode = False
            self._zoom = max(getattr(self, "_zoom", 1.0), 1.0)
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._drag_start = (event.x, event.y)
        self._pan[0] += dx
        self._pan[1] += dy
        self._render_preview()

    def _render_preview(self):
        if getattr(self, "_img_for_preview", None) is None or self.canvas.winfo_width() <= 1:
            return
        arr = self._img_for_preview
        H, W = arr.shape[:2]
        cw = int(self.canvas.winfo_width())
        ch = int(self.canvas.winfo_height())

        # base scale that fits the image into the canvas
        fit = min(cw / max(1, W), ch / max(1, H))
        scale = fit if getattr(self, "_fit_mode", True) else fit * getattr(self, "_zoom", 1.0)

        new_w = max(1, int(W * scale))
        new_h = max(1, int(H * scale))

        from PIL import Image, ImageTk
        im = Image.fromarray(arr).resize((new_w, new_h), Image.BILINEAR)

        # keep a reference so Tk doesn't GC the image
        self.tk_img = ImageTk.PhotoImage(im)
        self.canvas.delete("all")
        cx, cy = cw // 2 + self._pan[0], ch // 2 + self._pan[1]
        self.canvas.create_image(cx, cy, image=self.tk_img, anchor="center")

        # draw selection overlay on top (if any)
        self._draw_crop_overlay()

        # (optional) replace inline drawing with helper:
        self._draw_pick_overlays()   # <-- add this line

        if getattr(self, "_picks", None) and self.sr is not None:
            # we'll draw in canvas coords, so transform bboxes
            for i in self._picks:
                if i < 0 or i >= len(self.sr.masks):
                    continue
                x, y, w, h = map(int, self.sr.masks[i]["bbox"])
                # image → canvas
                cx1, cy1 = self._image_to_canvas_xy(x, y)
                cx2, cy2 = self._image_to_canvas_xy(x + w, y + h)
                self.canvas.create_rectangle(
                    cx1, cy1, cx2, cy2,
                    outline="black", dash=(4, 3), width=2, tags=("pickbox",)
                )

    
    # ---------- Crop helpers (INSIDE the class) ----------
    def _canvas_geometry(self):
        """Return (W,H,scale,cx,cy,cw,ch) for current preview image on the canvas."""
        if getattr(self, "_img_for_preview", None) is None:
            return None
        arr = self._img_for_preview
        H, W = arr.shape[:2]
        cw = int(self.canvas.winfo_width())
        ch = int(self.canvas.winfo_height())
        fit = min(cw / max(1, W), ch / max(1, H))
        scale = fit if getattr(self, "_fit_mode", True) else fit * getattr(self, "_zoom", 1.0)
        cx = cw // 2 + self._pan[0]
        cy = ch // 2 + self._pan[1]
        return W, H, scale, cx, cy, cw, ch

    def _canvas_to_image_xy(self, x, y):
        """Canvas -> image coords (clamped)."""
        geom = self._canvas_geometry()
        if geom is None:
            return 0, 0
        W, H, scale, cx, cy, *_ = geom
        s = max(1e-6, scale)
        ix = int(round((x - cx) / s + W / 2))
        iy = int(round((y - cy) / s + H / 2))
        ix = max(0, min(W - 1, ix))
        iy = max(0, min(H - 1, iy))
        return ix, iy

    def _image_to_canvas_xy(self, ix, iy):
        """Image -> canvas coords."""
        geom = self._canvas_geometry()
        if geom is None:
            return 0, 0
        W, H, scale, cx, cy, *_ = geom
        x = cx + (ix - W / 2) * scale
        y = cy + (iy - H / 2) * scale
        return int(round(x)), int(round(y))

    def _draw_crop_overlay(self):
        """Draw/refresh the crop rectangle overlay (border only, no fill)."""
        # clear any previous overlay
        self.canvas.delete("crop")
        if self._crop_rect_img is None:
            return

        x1, y1, x2, y2 = self._crop_rect_img
        x1, x2 = sorted((int(x1), int(x2)))
        y1, y2 = sorted((int(y1), int(y2)))

        c1x, c1y = self._image_to_canvas_xy(x1, y1)
        c2x, c2y = self._image_to_canvas_xy(x2, y2)

        # outline only — dashed cyan border, no fill
        self._crop_canvas_id = self.canvas.create_rectangle(
            c1x, c1y, c2x, c2y,
            outline="#00d7ff",
            width=2,
            dash=(6, 3),
            tags="crop",
        )


    def _clear_crop_overlay(self):
        self.canvas.delete("crop")
        self._crop_canvas_id = None
        self._crop_rect_img = None
        self._update_crop_buttons()

    def _update_crop_buttons(self):
        have_sel = bool(self._crop_mode.get()) and (self._crop_rect_img is not None)
        self._btn_crop_apply.configure(state=("normal" if have_sel else "disabled"))
        self._btn_crop_cancel.configure(state=("normal" if self._crop_mode.get() else "disabled"))

    # ---------- Crop interactions ----------
    def _crop_start(self, event):
        if getattr(self, "_img_for_preview", None) is None:
            return
        ix, iy = self._canvas_to_image_xy(event.x, event.y)
        self._crop_rect_img = [ix, iy, ix, iy]
        self._draw_crop_overlay()
        self._update_crop_buttons()

    def _crop_drag(self, event):
        if self._crop_rect_img is None:
            return
        ix, iy = self._canvas_to_image_xy(event.x, event.y)
        self._crop_rect_img[2] = ix
        self._crop_rect_img[3] = iy
        self._draw_crop_overlay()
        self._update_crop_buttons()

    def _crop_end(self, event):
        if self._crop_rect_img is None:
            return
        # finalize one more time
        self._crop_drag(event)
        x1, y1, x2, y2 = self._crop_rect_img
        if abs(x2 - x1) < 3 or abs(y2 - y1) < 3:
            # too small → drop
            self._clear_crop_overlay()
        self._update_crop_buttons()

    def _apply_crop(self):
        """Apply crop to current rotated base image and make it the new original."""
        if self._crop_rect_img is None or self.img_orig is None:
            return

        arr = self._base_image()
        if arr is None:
            return

        H, W = arr.shape[:2]
        x1, y1, x2, y2 = self._crop_rect_img
        x1, x2 = sorted((int(round(x1)), int(round(x2))))
        y1, y2 = sorted((int(round(y1)), int(round(y2))))
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H, y2))

        if x2 - x1 < 2 or y2 - y1 < 2:
            self._clear_crop_overlay()
            return

        crop = arr[y1:y2, x1:x2].copy()

        # Make the crop the new "original"; rotation and enhancement continue from here
        self.img_orig = crop
        self.img_preview = None
        self.img = crop

        # reset view to fit
        self._fit_mode = True
        self._zoom = 1.0
        self._pan = [0, 0]

        self._clear_crop_overlay()
        self._crop_mode.set(False)
        self._bind_canvas_events()
        self.show_image(crop)

    def _cancel_crop(self):
        self._clear_crop_overlay()

    
    # ---------- Edit-mode toggles (select/deselect/none) ----------
    def _toggle_pick_mode(self):
        """Toolbar toggle to enable/disable click-to-pick on the preview."""
        if bool(getattr(self, "_pick_mode", tk.BooleanVar(value=False)).get()):
            self._set_edit_mode("select")
        else:
            # leave current picks intact, just stop capturing clicks
            self._set_edit_mode("none")

    def _set_edit_mode(self, mode: str):
        """
        mode is 'select' or 'deselect'. Activates click-to-pick,
        clears current picks, rebinds canvas, and refreshes preview.
        """
        if not hasattr(self, "_edit_mode"):
            # safety: ensure these exist (they should be initialized in __init__)
            self._edit_mode = tk.StringVar(value="none")
        if not hasattr(self, "_picks"):
            self._picks = set()
        if not hasattr(self, "_pick_mode"):
            self._pick_mode = tk.BooleanVar(value=False)

        # normalize & set
        mode = (mode or "none").lower()
        if mode not in ("select", "deselect"):
            mode = "none"
        self._edit_mode.set(mode)
        # keep toolbar toggle in sync
        self._pick_mode.set(mode != "none")
        if hasattr(self, "_pick_status"):
            if mode == "none":
                self._pick_status.configure(text="")
            else:
                self._pick_status.configure(text=f"{len(getattr(self, '_picks', []))} selected")

        # fresh selection slate when entering a mode
        if mode != "none":
            self._picks.clear()

        # rebind + redraw
        self._bind_canvas_events()
        self._render_preview()

    def _reset_pick(self):
        self._picks.clear()
        self._edit_mode.set("none")
        if hasattr(self, "_pick_mode"):
            self._pick_mode.set(False)
        self._pick_status.configure(text="")
        self._sync_listbox_selection_from_picks()
        self._bind_canvas_events()
        self._render_preview()


    def _apply_pick(self):
        if not self.sr or not self.sr.masks:
            return
        if not self._picks:
            self._set_edit_mode("none")
            return
        action = self._edit_mode.get()
        if action == "deselect":
            for idx in sorted(self._picks, reverse=True):
                if 0 <= idx < len(self.sr.masks):
                    del self.sr.masks[idx]
        elif action == "select":
            kept = [self.sr.masks[i] for i in sorted(self._picks) if 0 <= i < len(self.sr.masks)]
            self.sr.masks = kept
        self._picks.clear()
        self._edit_mode.set("none")
        if hasattr(self, "_pick_mode"):
            self._pick_mode.set(False)
        self._pick_status.configure(text="")
        self._rebuild_mask_list()
        self._sync_listbox_selection_from_picks()
        self._bind_canvas_events()
        self._render_preview()

    def _show_busy(self, text: str = "Working…"):
        """Show a modern busy indicator with animated spinner on the canvas."""
        # Update status bar
        self.set_status(text, "processing")

        # Show spinner on canvas if available
        if hasattr(self, '_spinner'):
            try:
                self._spinner.show(text)
            except Exception:
                pass

        # Fallback: also show modal for blocking operations
        if self._busy_win is not None:
            try:
                self._busy_win.lift()
                return
            except Exception:
                self._busy_win = None

        win = tk.Toplevel(self.root)
        win.title("")
        win.transient(self.root)
        win.resizable(False, False)
        win.overrideredirect(True)  # No title bar for modern look
        win.configure(bg=self.colors.get("bg_dark", "#1a2f1a"))

        frame = tk.Frame(win, bg=self.colors.get("bg_dark", "#1a2f1a"), padx=20, pady=15)
        frame.pack(fill="both", expand=True)

        # Icon
        icon_label = tk.Label(frame, text="⏳", font=("Helvetica", 20),
                             bg=self.colors.get("bg_dark", "#1a2f1a"),
                             fg=self.colors.get("accent", "#4caf50"))
        icon_label.pack()

        # Text
        text_label = tk.Label(frame, text=text, font=("Helvetica", 11),
                             bg=self.colors.get("bg_dark", "#1a2f1a"),
                             fg=self.colors.get("text_light", "#f1f8e9"))
        text_label.pack(pady=(8, 10))

        # Progress bar
        pb = ttk.Progressbar(frame, mode="indeterminate", length=200)
        pb.pack(fill="x")
        pb.start(15)

        # Center on root
        try:
            win.update_idletasks()
            rx = self.root.winfo_rootx()
            ry = self.root.winfo_rooty()
            rw = self.root.winfo_width()
            rh = self.root.winfo_height()
            ww = win.winfo_reqwidth()
            wh = win.winfo_reqheight()
            win.geometry(f"+{rx + rw//2 - ww//2}+{ry + rh//2 - wh//2}")
        except Exception:
            pass

        self._busy_win = win
        self._busy_bar = pb

    def _hide_busy(self):
        """Hide busy indicator."""
        # Update status bar
        self.set_status("Ready", "success")

        # Hide spinner
        if hasattr(self, '_spinner'):
            try:
                self._spinner.hide()
            except Exception:
                pass

        # Hide modal
        if self._busy_win is None:
            return
        try:
            if hasattr(self, "_busy_bar") and self._busy_bar is not None:
                self._busy_bar.stop()
            self._busy_win.destroy()
        except Exception:
            pass
        self._busy_win = None
        self._busy_bar = None


    def _mask_candidates_at(self, x_img: int, y_img: int):
        """Return candidate mask indices under the cursor (sorted smallest area first)."""
        if not self.sr or not self.sr.masks:
            return []
        # Skip "full-image" masks that swallow everything (often index 0)
        img_h, img_w = None, None
        try:
            if self.sr.img_color is not None:
                img_h, img_w = self.sr.img_color.shape[:2]
        except Exception:
            pass
        full_mask_area = None
        if img_h and img_w:
            full_mask_area = float(img_h * img_w)

        hits = []
        # 1) Try exact hit-test on the segmentation mask
        for i, m in enumerate(self.sr.masks):
            if i in getattr(self, "_pick_blacklist", set()):
                continue
            seg = m.get("segmentation")
            if seg is None:
                continue
            try:
                if 0 <= y_img < seg.shape[0] and 0 <= x_img < seg.shape[1] and seg[y_img, x_img]:
                    a = m.get("area", 0)
                    if full_mask_area and a >= 0.90 * full_mask_area:
                        # ignore near-full-image masks when picking
                        continue
                    hits.append((i, float(a)))
            except Exception:
                # segmentation may be RLE/polygons; fall back to bbox below
                pass
        if hits:
            hits.sort(key=lambda z: z[1])  # smallest area first
            return [i for i, _ in hits]

        # 2) Fallback: bbox hit-test (works even if segmentation isn't a numpy mask)
        hits = []
        for i, m in enumerate(self.sr.masks):
            if i in getattr(self, "_pick_blacklist", set()):
                continue
            try:
                x, y, w, h = map(int, m.get("bbox", (0, 0, 0, 0)))
            except Exception:
                continue
            if full_mask_area and (w * h) >= 0.90 * full_mask_area:
                # ignore near-full-image masks when picking
                continue
            if x <= x_img < x + w and y <= y_img < y + h:
                a = m.get("area", 0)
                hits.append((i, float(a)))
        hits.sort(key=lambda z: z[1])
        return [i for i, _ in hits]

    def _on_pick_click(self, event):
        if not self.sr or not self.sr.masks:
            return
        pt = self._canvas_to_image_xy(event.x, event.y)
        if not pt:
            return
        candidates = self._mask_candidates_at(*pt)
        if not candidates:
            return
        # cycle through candidates on repeated clicks at nearly the same spot
        cycle_idx = 0
        last = getattr(self, "_last_pick_candidates", None)
        last_xy = getattr(self, "_last_pick_xy", None)
        if last is not None and last_xy is not None and last == candidates:
            dx = abs(pt[0] - last_xy[0]); dy = abs(pt[1] - last_xy[1])
            if dx <= 2 and dy <= 2 and len(candidates) > 1:
                cycle_idx = (int(getattr(self, "_last_pick_cycle_idx", 0)) + 1) % len(candidates)
        idx = candidates[cycle_idx]
        self._last_pick_candidates = candidates
        self._last_pick_xy = pt
        self._last_pick_cycle_idx = cycle_idx
        if idx in getattr(self, "_pick_blacklist", set()):
            return
        if idx in self._picks:
            self._picks.remove(idx)
        else:
            self._picks.add(idx)
        self._sync_listbox_selection_from_picks()
        if hasattr(self, "_pick_status"):
            self._pick_status.configure(text=f"{len(self._picks)} selected")
        self._render_preview()

    def _draw_pick_overlays(self):
        if not self.sr or not self.sr.masks or not self._picks:
            return
        for idx in self._picks:
            if 0 <= idx < len(self.sr.masks):
                x, y, w, h = map(int, self.sr.masks[idx]["bbox"])
                p1 = self._image_to_canvas_xy(x,     y)
                p2 = self._image_to_canvas_xy(x + w, y + h)
                if p1 and p2:
                    x1, y1 = p1; x2, y2 = p2
                    self.canvas.create_rectangle(x1, y1, x2, y2,
                                                outline="black", dash=(4, 2), width=2)

    def _sync_listbox_selection_from_picks(self):
        """Keep listbox selection in sync with preview picks without swapping preview."""
        if not hasattr(self, "lb"):
            return
        self._suppress_listbox_select = True
        try:
            self.lb.selection_clear(0, tk.END)
            for i in sorted(self._picks):
                if i in getattr(self, "_pick_blacklist", set()):
                    continue
                if 0 <= i < self.lb.size():
                    self.lb.selection_set(i)
            if self._picks:
                visible = [i for i in sorted(self._picks) if i not in getattr(self, "_pick_blacklist", set())]
                if visible:
                    self.lb.see(visible[-1])
        finally:
            self._suppress_listbox_select = False

    def build_mask_generator(self):
        return SAM2AutomaticMaskGenerator(
            self.sam2_model,
            points_per_side=self.m_points_per_side.get(),
            points_per_batch=self.m_points_per_batch.get(),
            pred_iou_thresh=float(self.m_pred_iou_thresh.get()),
            stability_score_thresh=float(self.m_stability_score_thresh.get()),
            crop_n_layers=self.m_crop_n_layers.get(),
            crop_overlap_ratio=float(self.m_crop_overlap_ratio.get()),
            crop_n_points_downscale_factor=self.m_crop_n_points_downscale_factor.get(),
            box_nms_thresh=float(self.m_box_nms_thresh.get()),
            min_mask_region_area=int(self.m_min_mask_region_area.get()),
            use_m2m=bool(self.m_use_m2m.get()),
            output_mode=self.m_output_mode.get(),
        )
    

    # ---- Segment ----
    def segment(self):
        if self.img is None:
            self.set_status("No image loaded", "warning")
            messagebox.showwarning("No image", "Open an image first.")
            return
        # If enabled, run tip-only segmentation (no SAM needed)
        if bool(self.target_use_tipseg.get()):
            if self.tipseg_model is None:
                self.set_status("No tip model loaded", "warning")
                messagebox.showwarning("No tip model", "Load the tip model first (Target Segment → Load tip model).")
                return
            self._show_busy("Running tip segmentation…")
            threading.Thread(target=self._tipseg_worker, daemon=True).start()
            return

        # Default: SAM2 segmentation
        if self.sam2_model is None:
            self.set_status("No model loaded", "warning")
            messagebox.showwarning("No model", "Load the SAM2 model first.")
            return
        self._show_busy("Running SAM2 segmentation…")
        threading.Thread(target=self._segment_worker, daemon=True).start()

    def _segment_worker(self):
        """Run SAM2 on the enhanced *rotated* image and update the UI."""
        try:
            arr = self.img.copy()
            if arr is None:
                raise RuntimeError("No image loaded.")
            seg_img = self._apply_enhance_pipeline(arr)

            gen = self.build_mask_generator()
            masks = gen.generate(seg_img)
            masks = dedupe_by_mask_iou(masks, iou_thresh=0.80)
            split_min = max(20, int(self.m_min_mask_region_area.get() * 0.10))
            masks = split_masks_by_cc(masks, min_area=split_min)
            if bool(self.target_filter_enable.get()) and (self.target_filter_stats or self.target_clf is not None):
                masks = self._apply_target_filter(masks, self.img.copy())

            # Store results (this is safe from thread)
            self.sr = SegResult(
                masks=masks,
                img_color=self.img.copy(),
                img_seg=seg_img,
                rotate_applied=self.chk_rotate.get()
            )

            # Schedule GUI updates on main thread
            def _update_gui():
                self.lb.delete(0, tk.END)
                for i, m in enumerate(masks):
                    self.lb.insert(tk.END, f"[{i:03d}] area={int(m['area'])} bbox={list(map(int, m['bbox']))}")
                self.show_image(seg_img)
                self._hide_busy()
                if bool(self.target_filter_enable.get()) and (self.target_filter_stats or self.target_clf is not None):
                    self.set_status(f"✓ Segmentation complete: {len(masks)} masks kept (filtered)", "success")
                else:
                    self.set_status(f"✓ Segmentation complete: {len(masks)} masks found", "success")
                messagebox.showinfo("Segmentation", f"Found {len(masks)} masks.")
                # cache for batch navigation
                try:
                    if self.img_path:
                        self._batch_mask_cache[self.img_path] = self.sr
                except Exception:
                    pass
                self._after_segment_prompt()

            self.root.after(0, _update_gui)

        except Exception as e:
            def _show_error():
                self._hide_busy()
                self.set_status(f"Segmentation failed: {str(e)[:50]}", "error")
                messagebox.showerror("Segmentation error", str(e))
            self.root.after(0, _show_error)

    def _tipseg_worker(self):
        """Run the tip-only segmenter on the current image (no SAM)."""
        try:
            arr = self.img.copy()
            if arr is None:
                raise RuntimeError("No image loaded.")

            from tip_segmenter_model import predict_tip_mask

            meta = self.tipseg_meta or {}
            input_size = int(meta.get("input_size", int(self.target_size_var.get() or 512)))
            device = str(meta.get("device", self.target_device_var.get().strip() or "cpu"))
            thr = float(self.target_tipseg_thresh.get())
            min_area = int(self.target_tipseg_min_area.get())
            keep_largest = bool(self.target_tipseg_keep_largest.get())

            if bool(self.tipseg_use_tiles.get()):
                mask_bool = self._tipseg_sliding_window(
                    arr,
                    input_size=input_size,
                    device=device,
                    threshold=thr,
                    min_area=min_area,
                    keep_largest=keep_largest,
                    tile_size=int(self.tipseg_tile_size.get()),
                    stride=int(self.tipseg_stride.get()),
                    color_guided=bool(self.tipseg_color_guided.get()),
                    color_min_area=int(self.tipseg_color_min_area.get()),
                    hue_low=int(self.tipseg_hue_low.get()),
                    hue_high=int(self.tipseg_hue_high.get()),
                    sat_min=int(self.tipseg_sat_min.get()),
                    val_min=int(self.tipseg_val_min.get()),
                    val_brown_max=int(self.tipseg_val_brown_max.get()),
                    min_leaf_pct=float(self.tipseg_min_leaf_pct.get()),
                    min_stress_pct=float(self.tipseg_min_stress_pct.get()),
                    stop_after_first=bool(self.tipseg_stop_after_first.get()),
                )
            else:
                mask_bool = predict_tip_mask(
                    self.tipseg_model,
                    arr,
                    input_size=input_size,
                    device=device,
                    threshold=thr,
                    min_area=min_area,
                    keep_largest=keep_largest,
                )

            masks = []
            if mask_bool is not None and mask_bool.any():
                bbox, area = self._bbox_area_from_mask(mask_bool)
                masks = [{
                    "segmentation": mask_bool.astype(np.uint8),
                    "area": float(area),
                    "bbox": bbox,
                    "predicted_iou": 1.0,
                    "stability_score": 1.0,
                    "meta": {"source": "tipseg"},
                }]

            self.sr = SegResult(
                masks=masks,
                img_color=arr.copy(),
                img_seg=arr.copy(),
                rotate_applied=self.chk_rotate.get()
            )

            def _update_gui():
                self.lb.delete(0, tk.END)
                for i, m in enumerate(masks):
                    self.lb.insert(tk.END, f"[{i:03d}] area={int(m['area'])} bbox={list(map(int, m['bbox']))}")
                if masks:
                    # show overlay by default so the mask is visible
                    try:
                        self.show_image(self._overlay_all_masks_colored(arr, alpha=0.5, outline=True))
                    except Exception:
                        self.show_image(arr)
                else:
                    self.show_image(arr)
                self._hide_busy()
                self.set_status(f"✓ Tip segmentation complete: {len(masks)} mask(s)", "success")
                messagebox.showinfo("Tip segmentation", f"Found {len(masks)} tip mask(s).")
                try:
                    if self.img_path:
                        self._batch_mask_cache[self.img_path] = self.sr
                except Exception:
                    pass
                self._after_segment_prompt()

            self.root.after(0, _update_gui)
        except Exception as e:
            def _show_error():
                self._hide_busy()
                self.set_status(f"Tip segmentation failed: {str(e)[:50]}", "error")
                messagebox.showerror("Tip segmentation error", str(e))
            self.root.after(0, _show_error)

    def _tipseg_sliding_window(
        self,
        img_rgb,
        input_size: int,
        device: str,
        threshold: float,
        min_area: int,
        keep_largest: bool,
        tile_size: int,
        stride: int,
        color_guided: bool,
        color_min_area: int,
        hue_low: int,
        hue_high: int,
        sat_min: int,
        val_min: int,
        val_brown_max: int,
        min_leaf_pct: float = 0.0,
        min_stress_pct: float = 0.0,
        stop_after_first: bool = False,
    ):
        """Run tip model over overlapping tiles; optionally focus on color-guided ROIs."""
        from tip_segmenter_model import predict_tip_mask

        H, W = img_rgb.shape[:2]
        tile = int(tile_size) if tile_size > 0 else int(input_size)
        tile = max(64, min(tile, max(H, W)))
        st = int(stride) if stride > 0 else max(32, tile // 2)
        st = max(32, min(st, tile))

        # Precompute color masks if needed (for filtering tiles or ROIs)
        leaf_u8 = None
        stress_u8 = None
        need_color_masks = color_guided or (min_leaf_pct > 0.0) or (min_stress_pct > 0.0)
        if need_color_masks:
            leaf_u8, stress_u8 = self._tipseg_color_masks(
                img_rgb,
                hue_low=hue_low,
                hue_high=hue_high,
                sat_min=sat_min,
                val_min=val_min,
                val_brown_max=val_brown_max,
            )

        # Determine scan regions
        if color_guided:
            rois = self._tipseg_color_rois(
                img_rgb,
                min_area=color_min_area,
                pad=max(16, tile // 4),
                hue_low=hue_low,
                hue_high=hue_high,
                sat_min=sat_min,
                val_min=val_min,
                val_brown_max=val_brown_max,
                stress_u8=stress_u8,
            )
            if not rois:
                rois = [(0, 0, W, H)]
        else:
            rois = [(0, 0, W, H)]

        # Build a set of tile positions
        positions = set()
        for (x1, y1, x2, y2) in rois:
            x1 = max(0, min(W - 1, int(x1)))
            y1 = max(0, min(H - 1, int(y1)))
            x2 = max(0, min(W, int(x2)))
            y2 = max(0, min(H, int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            ys = list(range(y1, y2, st))
            xs = list(range(x1, x2, st))
            if not ys:
                ys = [y1]
            if not xs:
                xs = [x1]
            for yy in ys:
                for xx in xs:
                    xx = min(xx, max(0, W - tile))
                    yy = min(yy, max(0, H - tile))
                    positions.add((xx, yy))

        full_mask = np.zeros((H, W), dtype=np.uint8)
        min_leaf_frac = max(0.0, float(min_leaf_pct) / 100.0)
        min_stress_frac = max(0.0, float(min_stress_pct) / 100.0)
        for (x, y) in sorted(positions, key=lambda p: (p[1], p[0])):
            tile_rgb = img_rgb[y:y + tile, x:x + tile]
            if tile_rgb.shape[0] == 0 or tile_rgb.shape[1] == 0:
                continue
            th = tile_rgb.shape[0]
            tw = tile_rgb.shape[1]

            # Skip tiles with too little non-white (leaf) coverage
            if leaf_u8 is not None and min_leaf_frac > 0.0:
                leaf_tile = leaf_u8[y:y + th, x:x + tw]
                if leaf_tile.size == 0:
                    continue
                leaf_frac = float(np.count_nonzero(leaf_tile)) / float(leaf_tile.size)
                if leaf_frac < min_leaf_frac:
                    continue

            # Skip tiles with too little stress-like color (if requested)
            if stress_u8 is not None and min_stress_frac > 0.0:
                stress_tile = stress_u8[y:y + th, x:x + tw]
                if stress_tile.size == 0:
                    continue
                stress_frac = float(np.count_nonzero(stress_tile)) / float(stress_tile.size)
                if stress_frac < min_stress_frac:
                    continue

            # Predict on tile (no per-tile filtering)
            tile_mask = predict_tip_mask(
                self.tipseg_model,
                tile_rgb,
                input_size=input_size,
                device=device,
                threshold=threshold,
                min_area=0,
                keep_largest=False,
            )
            if tile_mask is None or not tile_mask.any():
                continue
            th, tw = tile_mask.shape[:2]
            h = min(th, H - y)
            w = min(tw, W - x)
            full_mask[y:y + h, x:x + w] = np.maximum(full_mask[y:y + h, x:x + w], tile_mask[:h, :w].astype(np.uint8))
            if stop_after_first:
                break

        # Final component filtering
        if (min_area > 0) or keep_largest:
            full_mask = self._filter_mask_components(full_mask, min_area=min_area, keep_largest=keep_largest)
        return full_mask.astype(bool)

    def _filter_mask_components(self, mask_u8, min_area: int = 0, keep_largest: bool = True):
        mask_u8 = (mask_u8 > 0).astype(np.uint8)
        if mask_u8.sum() == 0:
            return mask_u8
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if num <= 1:
            return mask_u8
        comps = []
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < int(min_area):
                continue
            comps.append((area, i))
        if not comps:
            return np.zeros_like(mask_u8)
        if keep_largest:
            _, idx = max(comps, key=lambda t: t[0])
            return (labels == idx).astype(np.uint8)
        out = np.zeros_like(mask_u8)
        for _, idx in comps:
            out[labels == idx] = 1
        return out

    def _tipseg_color_masks(
        self,
        img_rgb,
        hue_low: int = 10,
        hue_high: int = 40,
        sat_min: int = 35,
        val_min: int = 40,
        val_brown_max: int = 200,
    ):
        """Return leaf and stress-like color masks (uint8 0/255)."""
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        h = hsv[..., 0]
        s = hsv[..., 1]
        v = hsv[..., 2]

        # Broad leaf mask (exclude white background)
        leaf = (s > 20) & (v > 40)

        # Stress-like colors: yellow/brown-ish within leaf
        h_low = int(max(0, min(179, hue_low)))
        h_high = int(max(0, min(179, hue_high)))
        if h_low <= h_high:
            hmask = (h >= h_low) & (h <= h_high)
        else:
            # wraparound (e.g., 170..10)
            hmask = (h >= h_low) | (h <= h_high)

        sat_min = int(max(0, min(255, sat_min)))
        val_min = int(max(0, min(255, val_min)))
        val_brown_max = int(max(0, min(255, val_brown_max)))

        # allow either decent saturation OR darker brown values
        stress = leaf & hmask & (v >= val_min) & ((s >= sat_min) | (v <= val_brown_max))

        leaf_u8 = (leaf.astype(np.uint8) * 255)
        stress_u8 = (stress.astype(np.uint8) * 255)
        return leaf_u8, stress_u8

    def _tipseg_color_rois(
        self,
        img_rgb,
        min_area: int = 600,
        pad: int = 64,
        hue_low: int = 10,
        hue_high: int = 40,
        sat_min: int = 35,
        val_min: int = 40,
        val_brown_max: int = 200,
        stress_u8=None,
    ):
        """Find candidate ROIs using a simple color heuristic (yellow/brown within leaf)."""
        if stress_u8 is None:
            _, stress_u8 = self._tipseg_color_masks(
                img_rgb,
                hue_low=hue_low,
                hue_high=hue_high,
                sat_min=sat_min,
                val_min=val_min,
                val_brown_max=val_brown_max,
            )
        H, W = stress_u8.shape[:2]
        if stress_u8.sum() == 0:
            return []

        k = np.ones((3, 3), np.uint8)
        stress_u8 = cv2.morphologyEx(stress_u8, cv2.MORPH_OPEN, k, iterations=1)
        stress_u8 = cv2.morphologyEx(stress_u8, cv2.MORPH_CLOSE, k, iterations=2)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(stress_u8, connectivity=8)
        rois = []
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < int(min_area):
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(W, x + w + pad)
            y2 = min(H, y + h + pad)
            rois.append((x1, y1, x2, y2))
        return rois




    # ---- Select mask and preview crop ----
    def on_select_mask(self, event=None):
        if getattr(self, "_suppress_listbox_select", False):
            return
        if not self.sr:
            return
        sel = self.lb.curselection()
        if not sel:
            return
        idx = sel[-1]

        # NEW: when the first list item is selected, show a colored overlay of ALL masks
        if idx == 0:
            # Use the original color image as the base (so nothing about crops changes)
            colored = self._overlay_all_masks_colored(self.sr.img_color, alpha=0.45, outline=True)
            self.show_image(colored)
            return

        # OLD behavior for any other item (keep your individual crop preview)
        m = self.sr.masks[idx]
        mask_bool = m["segmentation"].astype(bool)
        x, y, w, h = map(int, m["bbox"])
        x2, y2 = x + w, y + h
        crop = self.sr.img_color[y:y2, x:x2, :].copy()
        msk  = mask_bool[y:y2, x:x2]

        # over checkerboard (unchanged)
        alpha = (msk.astype(np.uint8) * 255)[..., None]
        rgba  = np.dstack([crop, alpha])
        tile  = 16
        Hc, Wc = rgba.shape[:2]
        if Hc == 0 or Wc == 0:
            # nothing to show
            self.show_image(crop)  # or return
            return
        chk = np.indices((Hc, Wc)).sum(axis=0) // tile
        bg  = np.where((chk % 2)[..., None], 200, 160).astype(np.uint8)
        a   = alpha.astype(np.float32) / 255.0
        comp = (rgba[..., :3] * a + bg * (1 - a)).astype(np.uint8)
        
        # If this mask was extended, draw base vs added outlines
        meta = m.get("meta", {})
        ext_full = meta.get("extended_bool", None)

        if ext_full is not None:
            # ext_full is full-image sized; crop to this mask's bbox
            ext_crop = (ext_full.astype(np.uint8) > 0)[y:y2, x:x2]
            # shapes must match
            if ext_crop.shape == msk.shape and ext_crop.any():
                comp_bgr = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
                # outline: whole mask (yellow)
                cnts_all, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(comp_bgr, cnts_all, -1, (0, 220, 255), 1)
                # outline: added part only (magenta)
                #cnts_add, _ = cv2.findContours(ext_crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #cv2.drawContours(comp_bgr, cnts_add, -1, (255, 0, 255), 1)
                #comp = cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2RGB)
                cnts_add, _ = cv2.findContours(ext_crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                comp_bgr = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
                cv2.drawContours(comp_bgr, cnts_add, -1, (255, 0, 255), 1)  # thin magenta
                comp = cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2RGB)

        self.show_image(comp)


    # ---- Save selected ----
    def save_selected(self):
        if not self.sr:
            messagebox.showwarning("Nothing to save", "Run segmentation first.")
            return
        sel = self.lb.curselection()
        if not sel:
            messagebox.showwarning("No selection", "Select one or more masks in the list.")
            return
        out_dir = filedialog.askdirectory(title="Choose output folder")
        if not out_dir:
            return
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        rows = []
        erode_px = 1
        feather_px = 2
        close_iters = 1

        for idx in sel:
            m = self.sr.masks[idx]
            mask_bool = m["segmentation"].astype(bool)

            if close_iters > 0:
                k = np.ones((3, 3), np.uint8)
                mask_bool = cv2.morphologyEx(mask_bool.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=close_iters).astype(bool)

            bbox = m["bbox"]
            base = f"mask_{idx:03d}"
            mask_path = out / f"{base}.png"
            crop_path = out / f"crop_{idx:03d}.png"
            save_binary_mask(mask_bool, mask_path)
            save_masked_crop_rgba(self.sr.img_color, mask_bool, bbox, crop_path, erode_px=erode_px, feather_px=feather_px)

            rows.append({
                "mask_idx": int(idx),
                "area_px": int(m["area"]),
                "bbox": list(map(int, m["bbox"])),
                "mask_png": str(mask_path),
                "crop_png": str(crop_path),
            })

        csv_path = out / "mask_manifest.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["mask_idx","area_px","bbox","mask_png","crop_png"])
            writer.writeheader()
            writer.writerows(rows)
        messagebox.showinfo("Saved", f"Saved {len(rows)} items to:\n{out}\n\nManifest: {csv_path}")

    def save_selected_masks(self):
        # Reuse your existing save_selected()
        return self.save_selected()

    def save_all_masks(self):
        if not self.sr or not self.sr.masks:
            messagebox.showwarning("Nothing to save", "Run segmentation first.")
            return
        # select everything, save, then restore selection
        old = self.lb.curselection()
        self.lb.selection_clear(0, tk.END)
        self.lb.selection_set(0, tk.END)
        try:
            self.save_selected()
        finally:
            self.lb.selection_clear(0, tk.END)
            for i in old:
                self.lb.selection_set(i)

    def export_individual_phenotypes(self):
        # hook you’ll fill in with the per-segment metrics export
        messagebox.showinfo("Phenotypes", "Individual phenotypes export (to be implemented).")

    def export_joint_phenotypes(self):
        # hook you’ll fill in with the combined metrics export
        messagebox.showinfo("Phenotypes", "Joint phenotypes export (to be implemented).")

    # ---------- phenotype flag collector ----------
    def _phen_flags(self):
        """Return which groups are enabled."""
        all_on = bool(self.ph_all.get())
        return dict(
            area   = all_on or bool(self.ph_area.get()),
            length = all_on or bool(self.ph_len.get()),
            width  = all_on or bool(self.ph_wid.get()),
            color  = all_on or bool(self.ph_color.get()),
            hsv    = all_on or bool(self.ph_hsv.get()),
            shape  = all_on or bool(self.ph_shape.get()),
            comp   = all_on or bool(self.ph_comp.get()),
            veg    = all_on or bool(self.ph_veg.get()),
            hsvvar = all_on or bool(self.ph_hsvvar.get()),
        )

    # ---------- mask/crop extraction for an index ----------
    def _mask_and_rgb_for_idx(self, idx: int):
        """Return (rgb_crop, mask_crop_bool, bbox) for mask index."""
        m = self.sr.masks[idx]
        mask_bool_full = m["segmentation"].astype(bool)
        x, y, w, h = map(int, m["bbox"])
        x2, y2 = x + w, y + h
        rgb = self.sr.img_color[y:y2, x:x2, :].copy()
        mask_crop = mask_bool_full[y:y2, x:x2]
        return rgb, mask_crop, (x, y, w, h)

    # ---------- measure one mask according to flags ----------
    def _measure_one_mask(self, idx: int, flags: dict):
        rgb, mask, bbox = self._mask_and_rgb_for_idx(idx)
        res = {}
        geom_mask = mask.astype(bool)
        color_mask = geom_mask.copy()

        mrec = self.sr.masks[idx]; meta = mrec.get("meta", {})
        ext_full = meta.get("extended_bool")
        if ext_full is not None:
            x, y, w, h = map(int, mrec["bbox"]); x2, y2 = x+w, y+h
            ext_crop = (ext_full.astype(np.uint8) > 0)[y:y2, x:x2]
            if ext_crop.shape == color_mask.shape:
                color_mask = np.logical_and(color_mask, ~ext_crop)
        if color_mask.sum() == 0:
            color_mask = geom_mask

        if flags.get("area"):
            res["area_px2"] = int(geom_mask.sum())
        if flags.get("comp"):
            try:
                cc = cv2.connectedComponents(geom_mask.astype(np.uint8), connectivity=8)[0] - 1
                res["components"] = int(max(cc, 0))
            except Exception:
                res["components"] = 0
        if flags.get("length") or flags.get("width"):
            maj, minw, axis_w, axis_h = _pca_major_minor(geom_mask)
            if flags.get("length"): res["length_major_px"] = round(maj, 2)
            if flags.get("width"):  res["width_minor_px"]  = round(minw, 2)
            paper = _length_width_after_deskew(geom_mask)
            if flags.get("length"):
                res["length_bbox_px"]   = round(paper["length_px"], 2)
                res["deskew_angle_deg"] = round(paper["angle_deg"], 2)
            if flags.get("width"):
                res["width_row_max_px"] = round(paper["width_px_max"], 2)
                res["width_row_p95_px"] = round(paper["width_px_p95"], 2)
            if flags.get("width"):  res["axis_width_px"]  = int(axis_w)
            if flags.get("length"): res["axis_height_px"] = int(axis_h)

        if flags.get("shape"):
            area = float(geom_mask.sum())
            perim = 0.0
            hull_area = 0.0
            hull_perim = 0.0
            try:
                m8 = (geom_mask.astype(np.uint8) * 255)
                cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    cnt = max(cnts, key=cv2.contourArea)
                    perim = float(cv2.arcLength(cnt, True))
                    hull = cv2.convexHull(cnt)
                    hull_area = float(cv2.contourArea(hull))
                    hull_perim = float(cv2.arcLength(hull, True))
            except Exception:
                pass

            bbox_area = float(bbox[2] * bbox[3]) if bbox else 0.0
            solidity = (area / hull_area) if hull_area > 0 else 0.0
            extent = (area / bbox_area) if bbox_area > 0 else 0.0
            circularity = (4.0 * math.pi * area / (perim * perim)) if perim > 0 else 0.0
            equiv_d = (math.sqrt(4.0 * area / math.pi)) if area > 0 else 0.0

            res.update({
                "perimeter_px": round(perim, 2),
                "hull_area_px2": round(hull_area, 2),
                "hull_perimeter_px": round(hull_perim, 2),
                "solidity": round(solidity, 4),
                "extent": round(extent, 4),
                "circularity": round(circularity, 4),
                "equiv_diameter_px": round(equiv_d, 2),
            })

        if flags.get("color"):
            R, G, B = _color_stats(rgb, color_mask)
            res.update({
                "mean_R": round(R["mean"],3), "mean_G": round(G["mean"],3), "mean_B": round(B["mean"],3),
                "median_R": round(R["median"],3), "median_G": round(G["median"],3), "median_B": round(B["median"],3),
                "sum_R": round(R["sum"],1), "sum_G": round(G["sum"],1), "sum_B": round(B["sum"],1),
                "std_R": round(R["std"],3), "std_G": round(G["std"],3), "std_B": round(B["std"],3),
            })
            if flags.get("veg"):
                # vegetation indices (masked pixels only)
                cm = color_mask.astype(bool)
                if cm.any():
                    r = rgb[..., 0].astype(np.float32)[cm]
                    g = rgb[..., 1].astype(np.float32)[cm]
                    b = rgb[..., 2].astype(np.float32)[cm]
                    exg = 2 * g - r - b
                    exr = 1.4 * r - g
                    exgr = exg - exr
                    denom = (2 * g + r + b)
                    gli = np.divide((2 * g - r - b), denom, out=np.zeros_like(denom), where=denom != 0)
                    green_frac = np.mean((g > r) & (g > b)) if g.size else 0.0
                    res.update({
                        "exg_mean": round(float(np.mean(exg)), 3),
                        "exr_mean": round(float(np.mean(exr)), 3),
                        "exgr_mean": round(float(np.mean(exgr)), 3),
                        "gli_mean": round(float(np.mean(gli)), 4),
                        "green_frac": round(float(green_frac), 4),
                    })
        if flags.get("hsv"):
            Hstats, Sstats, Vstats = _color_stats_hsv(rgb, color_mask)
            res.update({
                "mean_H": round(Hstats["mean"],3), "mean_S": round(Sstats["mean"],3), "mean_V": round(Vstats["mean"],3),
                "median_H": round(Hstats["median"],3), "median_S": round(Sstats["median"],3), "median_V": round(Vstats["median"],3),
                "sum_H": round(Hstats["sum"],1), "sum_S": round(Sstats["sum"],1), "sum_V": round(Vstats["sum"],1),
                "std_H": round(Hstats["std"],3), "std_S": round(Sstats["std"],3), "std_V": round(Vstats["std"],3),
            })
            if flags.get("hsvvar"):
                res.update({
                    "var_H": round(float(Hstats["std"]) ** 2, 4),
                    "var_S": round(float(Sstats["std"]) ** 2, 4),
                })
        return res



    # ---------- common mask saver ----------
    def _export_masks(self, indices, out_dir: Path, crop_dir: Path | None = None):
        """Save mask PNG + RGBA crop for given indices. Returns manifest rows."""
        base_name = Path(self.img_path).stem if self.img_path else "Image"
        rows = []
        erode_px = 1
        feather_px = 2
        close_iters = int(self.s_close_iters.get() if hasattr(self, "s_close_iters") else 0)
        crop_dir = crop_dir or out_dir
        crop_dir.mkdir(parents=True, exist_ok=True)

        for k, idx in enumerate(indices, start=1):
            m = self.sr.masks[idx]
            mask_bool = m["segmentation"].astype(bool)

            if close_iters > 0:
                k3 = np.ones((3, 3), np.uint8)
                mask_bool = cv2.morphologyEx(mask_bool.astype(np.uint8), cv2.MORPH_CLOSE, k3,
                                             iterations=close_iters).astype(bool)

            bbox = m["bbox"]
            seg_id = f"{base_name}_{k}"          # 1-based numbering
            mask_path = out_dir / f"{seg_id}.mask.png"
            crop_path = crop_dir / f"{seg_id}.crop.png"
            save_binary_mask(mask_bool, mask_path)
            save_masked_crop_rgba(self.sr.img_color, mask_bool, bbox, crop_path,
                                  erode_px=erode_px, feather_px=feather_px)

            rows.append({
                "file": base_name,
                "segment_id": seg_id,
                "mask_png": str(mask_path),
                "crop_png": str(crop_path),
                "area_px2": int(m["area"]),
                "bbox": list(map(int, m["bbox"])),
            })
        return rows

    def _parse_bbox_field(self, bbox_val):
        if bbox_val is None:
            return None
        if isinstance(bbox_val, (list, tuple)) and len(bbox_val) == 4:
            try:
                return [int(x) for x in bbox_val]
            except Exception:
                return None
        s = str(bbox_val).strip()
        if not s:
            return None
        s = s.strip("[]()")
        parts = [p for p in re.split(r"[,\s]+", s) if p]
        if len(parts) != 4:
            return None
        try:
            return [int(float(p)) for p in parts]
        except Exception:
            return None

    def _bbox_area_from_mask(self, mask_bool: np.ndarray):
        ys, xs = np.nonzero(mask_bool)
        if xs.size:
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            bbox = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
        else:
            bbox = [0, 0, 0, 0]
        area = float(mask_bool.sum())
        return bbox, area

    def _resolve_manifest_path(self, base_dir: Path, p):
        if not p:
            return None
        q = Path(p)
        if not q.is_absolute():
            cand = base_dir / q
            if cand.exists():
                return cand
            alt = base_dir.parent / q
            if alt.exists():
                return alt
            q = cand
        return q

    def _save_mask_bundle_manifest(self, out_dir: Path, rows: list, base_name: str, rel_base: Path | None = None):
        image_png = None
        segmented_png = None
        try:
            if self.sr and self.sr.img_color is not None:
                image_png = out_dir / f"{base_name}.image.png"
                cv2.imwrite(str(image_png), cv2.cvtColor(self.sr.img_color, cv2.COLOR_RGB2BGR))
            if self.sr and self.sr.img_seg is not None:
                segmented_png = out_dir / f"{base_name}.segmented.png"
                cv2.imwrite(str(segmented_png), cv2.cvtColor(self.sr.img_seg, cv2.COLOR_RGB2BGR))
        except Exception:
            pass

        rel_base = rel_base or out_dir

        def _rel(p):
            try:
                return str(Path(p).relative_to(rel_base))
            except Exception:
                return str(p)

        masks = []
        for r in rows:
            masks.append({
                "segment_id": r.get("segment_id") or r.get("mask_idx"),
                "mask_png": _rel(r.get("mask_png")) if r.get("mask_png") else "",
                "crop_png": _rel(r.get("crop_png")) if r.get("crop_png") else "",
                "area_px2": r.get("area_px2") or r.get("area_px"),
                "bbox": r.get("bbox"),
            })

        bundle = {
            "version": 1,
            "source_image": str(self.img_path) if self.img_path else "",
            "image_png": _rel(image_png) if image_png else "",
            "segmented_png": _rel(segmented_png) if segmented_png else "",
            "rotate_applied": bool(self.sr.rotate_applied) if self.sr else False,
            "masks": masks,
        }

        bundle_path = out_dir / f"{base_name}.mask_bundle.json"
        with open(bundle_path, "w") as f:
            json.dump(bundle, f, indent=2)
        return bundle_path

    def _prompt_for_base_image(self):
        p = filedialog.askopenfilename(
            title="Select base image for masks",
            filetypes=[("Images", "*.tif *.tiff *.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        if not p:
            return None, None
        try:
            return ensure_uint8_rgb(Image.open(p)), p
        except Exception:
            return None, None

    def _load_masks_from_json(self, manifest_path: Path):
        with open(manifest_path, "r") as f:
            data = json.load(f)
        base_dir = manifest_path.parent

        image_png = self._resolve_manifest_path(base_dir, data.get("image_png") or data.get("image"))
        segmented_png = self._resolve_manifest_path(base_dir, data.get("segmented_png") or data.get("segmented"))
        source_image = self._resolve_manifest_path(base_dir, data.get("source_image") or data.get("image_path"))

        base_img = None
        base_img_path = None
        for cand in (image_png, source_image):
            if cand and cand.exists():
                base_img = ensure_uint8_rgb(Image.open(cand))
                base_img_path = str(cand)
                break

        seg_img = None
        if segmented_png and segmented_png.exists():
            seg_img = ensure_uint8_rgb(Image.open(segmented_png))

        masks = []
        for item in data.get("masks", []):
            mask_path = self._resolve_manifest_path(base_dir, item.get("mask_png") or item.get("mask"))
            if not mask_path or not mask_path.exists():
                continue
            mask_u8 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_u8 is None:
                continue
            mask_bool = mask_u8 > 0

            bbox = self._parse_bbox_field(item.get("bbox"))
            if base_img is not None and mask_bool.shape != base_img.shape[:2]:
                if bbox and mask_bool.shape == (int(bbox[3]), int(bbox[2])):
                    full = np.zeros(base_img.shape[:2], dtype=bool)
                    x, y, w, h = map(int, bbox)
                    full[y:y+h, x:x+w] = mask_bool
                    mask_bool = full

            if bbox is None or len(bbox) != 4:
                bbox, area = self._bbox_area_from_mask(mask_bool)
            else:
                area = float(item.get("area_px2") or item.get("area_px") or mask_bool.sum())

            masks.append({
                "segmentation": mask_bool.astype(np.uint8),
                "bbox": bbox,
                "area": float(area),
                "meta": {"loaded_from": str(mask_path)},
            })

        return base_img, seg_img, masks, base_img_path

    def _load_masks_from_csv(self, manifest_path: Path):
        base_dir = manifest_path.parent
        rows = []
        with open(manifest_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

        base_name = None
        if rows and rows[0].get("file"):
            base_name = rows[0].get("file")

        base_img = None
        base_img_path = None
        if base_name:
            cand = base_dir / f"{base_name}.image.png"
            if cand.exists():
                base_img = ensure_uint8_rgb(Image.open(cand))
                base_img_path = str(cand)

        seg_img = None
        if base_name:
            cand = base_dir / f"{base_name}.segmented.png"
            if cand.exists():
                seg_img = ensure_uint8_rgb(Image.open(cand))

        masks = []
        for item in rows:
            mask_path = self._resolve_manifest_path(base_dir, item.get("mask_png") or item.get("mask"))
            if not mask_path or not mask_path.exists():
                continue
            mask_u8 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_u8 is None:
                continue
            mask_bool = mask_u8 > 0

            bbox = self._parse_bbox_field(item.get("bbox"))
            if base_img is not None and mask_bool.shape != base_img.shape[:2]:
                if bbox and mask_bool.shape == (int(bbox[3]), int(bbox[2])):
                    full = np.zeros(base_img.shape[:2], dtype=bool)
                    x, y, w, h = map(int, bbox)
                    full[y:y+h, x:x+w] = mask_bool
                    mask_bool = full

            if bbox is None or len(bbox) != 4:
                bbox, area = self._bbox_area_from_mask(mask_bool)
            else:
                area = float(item.get("area_px2") or item.get("area_px") or mask_bool.sum())

            masks.append({
                "segmentation": mask_bool.astype(np.uint8),
                "bbox": bbox,
                "area": float(area),
                "meta": {"loaded_from": str(mask_path)},
            })

        return base_img, seg_img, masks, base_img_path

    # ---------- UI actions: save masks ----------
    def save_all_masks(self):
        if not self.sr:
            messagebox.showwarning("Nothing to save", "Run segmentation first.")
            return
        out = filedialog.askdirectory(title="Choose output folder")
        if not out: return
        out_dir = Path(out); out_dir.mkdir(parents=True, exist_ok=True)

        idxs = list(range(len(self.sr.masks)))
        rows = self._export_masks(idxs, out_dir)

        # optional manifest
        csv_path = out_dir / "mask_manifest.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file","segment_id","mask_png","crop_png","area_px2","bbox"])
            writer.writeheader(); writer.writerows(rows)

        base_name = Path(self.img_path).stem if self.img_path else "Image"
        bundle_path = self._save_mask_bundle_manifest(out_dir, rows, base_name)

        messagebox.showinfo(
            "Saved",
            f"Exported {len(rows)} masks to:\n{out_dir}\n\nBundle: {bundle_path}"
        )

    def save_selected_masks(self):
        if not self.sr:
            messagebox.showwarning("Nothing to save", "Run segmentation first.")
            return
        sel = list(self.lb.curselection())
        if not sel:
            if messagebox.askyesno("No selection", "No segments selected. Save ALL instead?"):
                return self.save_all_masks()
            return
        out = filedialog.askdirectory(title="Choose output folder")
        if not out: return
        out_dir = Path(out); out_dir.mkdir(parents=True, exist_ok=True)

        rows = self._export_masks(sel, out_dir)

        csv_path = out_dir / "mask_manifest_selected.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file","segment_id","mask_png","crop_png","area_px2","bbox"])
            writer.writeheader(); writer.writerows(rows)

        base_name = Path(self.img_path).stem if self.img_path else "Image"
        bundle_path = self._save_mask_bundle_manifest(out_dir, rows, base_name)

        messagebox.showinfo(
            "Saved",
            f"Exported {len(rows)} selected masks to:\n{out_dir}\n\nBundle: {bundle_path}"
        )

    def save_all_outputs(self):
        if not self.sr:
            messagebox.showwarning("Nothing to save", "Run segmentation first.")
            return
        flags = self._phen_flags()

        sel = list(self.lb.curselection())
        idxs = sel if sel else list(range(len(self.sr.masks)))
        if not sel:
            if not messagebox.askyesno("No selection", "No segments selected. Export ALL segments?"):
                return

        out_root = filedialog.askdirectory(title="Choose output folder")
        if not out_root:
            return
        root = Path(out_root)
        phen_dir = root / "phenotypes"
        seg_dir = root / "segments"
        mask_dir = root / "masks"
        phen_dir.mkdir(parents=True, exist_ok=True)
        seg_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        rows = self._export_masks(idxs, mask_dir, crop_dir=seg_dir)

        csv_path = mask_dir / "mask_manifest.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file","segment_id","mask_png","crop_png","area_px2","bbox"])
            writer.writeheader(); writer.writerows(rows)

        base_name = Path(self.img_path).stem if self.img_path else "Image"
        bundle_path = self._save_mask_bundle_manifest(mask_dir, rows, base_name, rel_base=root)

        ind_csv = phen_dir / f"{base_name}_phenotypes_individual.csv"
        joint_csv = phen_dir / f"{base_name}_phenotypes_joint.csv"
        self._write_individual_phenotypes(idxs, flags, ind_csv)
        self._write_joint_phenotypes(idxs, flags, joint_csv)

        messagebox.showinfo(
            "Saved",
            "Saved outputs to:\n"
            f"{root}\n\n"
            f"Phenotypes: {phen_dir}\n"
            f"Segments: {seg_dir}\n"
            f"Masks + bundle: {mask_dir}\n\n"
            f"Bundle: {bundle_path}"
        )

    def load_masks(self):
        p = filedialog.askopenfilename(
            title="Load masks (manifest JSON/CSV)",
            filetypes=[("Mask bundle/manifest", "*.json *.csv"), ("All files", "*.*")]
        )
        if not p:
            return
        path = Path(p)

        try:
            if path.suffix.lower() == ".json":
                base_img, seg_img, masks, base_img_path = self._load_masks_from_json(path)
            else:
                base_img, seg_img, masks, base_img_path = self._load_masks_from_csv(path)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return

        if not masks:
            messagebox.showwarning("Load masks", "No masks found in that manifest.")
            return

        if base_img is None:
            base_img, base_img_path = self._prompt_for_base_image()
        if base_img is None:
            h, w = masks[0]["segmentation"].shape[:2]
            base_img = np.zeros((h, w, 3), dtype=np.uint8)

        if seg_img is None:
            seg_img = base_img.copy()

        self.img_path = base_img_path
        self.img_orig = base_img
        self.img = base_img
        self.img_preview = None
        self.sr = SegResult(
            masks=masks,
            img_color=base_img,
            img_seg=seg_img,
            rotate_applied=False
        )

        if hasattr(self, "_picks"):
            self._picks.clear()
        if hasattr(self, "_pick_status"):
            self._pick_status.configure(text="")

        self._rebuild_mask_list()
        self.show_image(seg_img)


    def load_model(self):
        """Load from checkpoint path + config (short name, YAML path, or config dir)."""
        if _sam2_import_error is not None:
            messagebox.showerror("SAM2 import error", f"Couldn't import sam2 modules:\n{_sam2_import_error}")
            return

        ckpt = self.e_ckpt.get().strip()
        if not ckpt or not os.path.exists(ckpt):
            messagebox.showerror("Missing checkpoint", "Please pick a valid .pt checkpoint file.")
            return

        cfg_field = (self.e_cfg.get().strip() or "sam2.1_hiera_l")
        dev      = (self.e_dev.get().strip() or "cpu")
        apply_pp = bool(self.chk_post.get())

        try:
            # Robust resolver: accepts short name ("sam2.1_hiera_l"), a full YAML path,
            # or a configs directory. It also searches near the checkpoint and $SAM2_CONFIG_DIR.
            cfg_resolved = _resolve_sam2_cfg(cfg_field, ckpt_path=ckpt)

            self.sam2_model = build_sam2(cfg_resolved, ckpt, device=dev, apply_postprocessing=apply_pp)
            self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2_model)
            messagebox.showinfo("Model", f"Loaded model on device '{dev}'.")
            try:
                self._set_sam_weights_tag("base")
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            self.sam2_model = None
            self.mask_generator = None

    def _write_individual_phenotypes(self, idxs, flags, out_csv):
        base_name = Path(self.img_path).stem if self.img_path else "Image"
        rows = []
        for j, idx in enumerate(idxs, start=1):
            r = {"FileName": base_name, "Segment": f"{base_name}_{j}"}
            r.update(self._measure_one_mask(idx, flags))
            rows.append(r)

        # headers: stable order
        cols = ["FileName","Segment"]
        if flags["area"]:   cols += ["area_px2"]
        if flags["length"]: cols += ["length_major_px","length_bbox_px","deskew_angle_deg","axis_height_px"]
        if flags["width"]:  cols += ["width_minor_px","width_row_max_px","width_row_p95_px","axis_width_px"]
        if flags["shape"]:  cols += ["perimeter_px","hull_area_px2","hull_perimeter_px","solidity",
                                     "extent","circularity","equiv_diameter_px"]
        if flags["comp"]:   cols += ["components"]
        if flags["color"]:  cols += ["mean_R","mean_G","mean_B","median_R","median_G","median_B",
                                     "sum_R","sum_G","sum_B","std_R","std_G","std_B"]
        if flags["veg"]:    cols += ["exg_mean","exr_mean","exgr_mean","gli_mean","green_frac"]
        if flags["hsv"]:    cols += ["mean_H","mean_S","mean_V","median_H","median_S","median_V",
                                     "sum_H","sum_S","sum_V","std_H","std_S","std_V"]
        if flags["hsvvar"]: cols += ["var_H","var_S"]

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader(); writer.writerows(rows)

    def _write_joint_phenotypes(self, idxs, flags, out_csv):
        # accumulate
        agg = {}
        n = len(idxs)
        for idx in idxs:
            r = self._measure_one_mask(idx, flags)
            for k, v in r.items():
                if isinstance(v, (int, float)):
                    agg[k] = agg.get(k, 0.0) + float(v)

        base_name = Path(self.img_path).stem if self.img_path else "Image"
        row = {"FileName": base_name, "n_segments": n}

        # write sums and means for numeric fields selected
        def _emit(name):
            if name in agg:
                row[name + "_total"] = round(agg[name], 3)
                row[name + "_mean"]  = round(agg[name] / max(1, n), 3)

        if flags["area"]:
            _emit("area_px2")

        if flags["length"]:
            for key in ("length_major_px", "length_bbox_px", "axis_height_px"):
                _emit(key)

        if flags["width"]:
            for key in ("width_minor_px", "width_row_max_px", "width_row_p95_px", "axis_width_px"):
                _emit(key)

        if flags["shape"]:
            for key in ("perimeter_px","hull_area_px2","hull_perimeter_px","solidity",
                        "extent","circularity","equiv_diameter_px"):
                _emit(key)

        if flags["comp"]:
            _emit("components")

        if flags["color"]:
            # for color stats, averaging the means makes sense; sums we also sum/mean
            for key in ("mean_R","mean_G","mean_B","median_R","median_G","median_B",
                        "sum_R","sum_G","sum_B","std_R","std_G","std_B"):
                if key in agg:
                    row[key + "_mean"]  = round(agg[key]/max(1,n), 3)
                    row[key + "_total"] = round(agg[key], 3)
        if flags["veg"]:
            for key in ("exg_mean","exr_mean","exgr_mean","gli_mean","green_frac"):
                if key in agg:
                    row[key + "_mean"]  = round(agg[key]/max(1,n), 4)
                    row[key + "_total"] = round(agg[key], 4)
        if flags["hsv"]:
            for key in ("mean_H","mean_S","mean_V","median_H","median_S","median_V",
                        "sum_H","sum_S","sum_V","std_H","std_S","std_V"):
                if key in agg:
                    row[key + "_mean"]  = round(agg[key]/max(1,n), 3)
                    row[key + "_total"] = round(agg[key], 3)
        if flags["hsvvar"]:
            for key in ("var_H","var_S"):
                if key in agg:
                    row[key + "_mean"]  = round(agg[key]/max(1,n), 4)
                    row[key + "_total"] = round(agg[key], 4)

        # column order
        cols = ["FileName","n_segments"]
        for group in (("area_px2",), 
                      ("length_major_px","length_bbox_px","axis_height_px"),
                      ("width_minor_px","width_row_max_px","width_row_p95_px","axis_width_px")):
            for k in group:
                if k+"_total" in row:
                    cols += [k+"_total", k+"_mean"]
        for k in ("perimeter_px","hull_area_px2","hull_perimeter_px","solidity",
                  "extent","circularity","equiv_diameter_px","components",
                  "exg_mean","exr_mean","exgr_mean","gli_mean","green_frac",
                  "var_H","var_S"):
            if k+"_total" in row:
                cols += [k+"_total", k+"_mean"]
        if "mean_R_mean" in {k for k in row}:  # color included
            for k in ("mean_R","mean_G","mean_B","median_R","median_G","median_B",
                      "sum_R","sum_G","sum_B","std_R","std_G","std_B"):
                if k+"_mean" in row:
                    cols += [k+"_total", k+"_mean"]
        if "mean_H_mean" in {k for k in row}:  # HSV included
            for k in ("mean_H","mean_S","mean_V","median_H","median_S","median_V",
                      "sum_H","sum_S","sum_V","std_H","std_S","std_V"):
                if k+"_mean" in row:
                    cols += [k+"_total", k+"_mean"]

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader(); writer.writerow(row)

    # ---------- INDIVIDUAL phenotypes ----------
    def export_individual_phenotypes(self):
        if not self.sr:
            messagebox.showwarning("Nothing to export", "Run segmentation first.")
            return
        flags = self._phen_flags()

        sel = list(self.lb.curselection())
        idxs = sel if sel else list(range(len(self.sr.masks)))
        if not sel:
            if not messagebox.askyesno("No selection", "No segments selected. Export ALL segments?"):
                return

        out_csv = filedialog.asksaveasfilename(
            title="Save individual phenotypes CSV", defaultextension=".csv",
            filetypes=[("CSV","*.csv")]
        )
        if not out_csv: return

        self._write_individual_phenotypes(idxs, flags, out_csv)
        messagebox.showinfo("Saved", f"Individual phenotypes written to:\n{out_csv}")

    # ---------- JOINT phenotypes (sum/mean over segments) ----------
    def export_joint_phenotypes(self):
        if not self.sr:
            messagebox.showwarning("Nothing to export", "Run segmentation first.")
            return
        flags = self._phen_flags()

        sel = list(self.lb.curselection())
        idxs = sel if sel else list(range(len(self.sr.masks)))
        if not sel:
            if not messagebox.askyesno("No selection", "No segments selected. Use ALL segments for the joint row?"):
                return

        out_csv = filedialog.asksaveasfilename(
            title="Save JOINT phenotypes CSV", defaultextension=".csv",
            filetypes=[("CSV","*.csv")]
        )
        if not out_csv: return

        self._write_joint_phenotypes(idxs, flags, out_csv)
        messagebox.showinfo("Saved", f"Joint phenotypes written to:\n{out_csv}")

    def explain_phenotypes(self):
        msg = (
            "Phenotype groups:\n"
            "• Area: pixel area of the mask.\n"
            "• Length/Width: PCA major/minor axes + deskewed length/row widths.\n"
            "• Color: RGB mean/median/sum/std within the mask.\n"
            "• HSV: HSV mean/median/sum/std within the mask.\n"
            "• Shape: perimeter, convex hull area/perimeter, solidity, extent, circularity, equiv. diameter.\n"
            "• Components: number of connected components in the mask.\n"
            "• VegIdx: ExG, ExR, ExGR, GLI, green fraction.\n"
            "• HSV Var: variance of H and S (from std²).\n"
            "\n"
            "These are the main easy, robust phenotypes. We can add more (eccentricity, "
            "fractal/roughness, skeleton length, etc.) if you want."
        )
        messagebox.showinfo("Phenotypes", msg)

    def _mask_color(self, k: int) -> tuple[int, int, int]:
        """
        Stable, vivid color per index (HSV wheel → RGB).
        """
        import colorsys
        h = (k * 0.61803398875) % 1.0  # golden-ratio spacing for distinct hues
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 1.0)
        return int(r * 255), int(g * 255), int(b * 255)

    def _overlay_all_masks_colored(self, base_img, alpha: float = 0.45, outline: bool = True):
        """
        Return a copy of base_img where EVERY mask is painted with a translucent color.
        """
        import numpy as np, cv2
        out = base_img.copy().astype(np.float32)
        # If we only have 1 mask, use it; otherwise skip index 0 (often “everything”).
        masks = self.sr.masks if len(self.sr.masks) <= 1 else self.sr.masks[1:]
        start_i = 0 if len(self.sr.masks) <= 1 else 1
        for i, m in enumerate(masks, start=start_i):
            seg = m["segmentation"]
            if not hasattr(seg, "dtype"):  # skip non-binary formats
                continue
            m_bool = seg.astype(bool)
            col = np.asarray(self._mask_color(i), dtype=np.float32)
            out[m_bool] = out[m_bool] * (1.0 - alpha) + col * alpha
            if outline:
                cnts, _ = cv2.findContours(m_bool.astype(np.uint8),
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, cnts, -1, tuple(int(c) for c in col.tolist()), 2)
        return out.astype(np.uint8)
    
    def _exit_edit_mode(self):
        """Leave pick mode; restore default pan/zoom bindings and redraw."""
        if not hasattr(self, "_edit_mode"):
            self._edit_mode = tk.StringVar(value="none")
        self._edit_mode.set("none")
        self._bind_canvas_events()
        self._render_preview()

           

    def explain_mask_params(self):
        txt = (
            "Quick guide:\n\n"
            "• points_per_side — grid resolution per crop. ↑ = more proposals, slower.\n"
            "• points_per_batch — batch size for those probes. ↑ uses more VRAM.\n"
            "• pred_iou_thresh — model’s quality score cutoff. Lower (0.5–0.7) finds more, "
            "higher (0.8–0.9) is cleaner.\n"
            "• stability_score_thresh — rejects masks that wobble under perturbations. Lower if thin/low-contrast leaves vanish.\n"
            "• crop_n_layers — # of multi-scale crops. More helps small objects; costs time.\n"
            "• crop_overlap_ratio — overlap between crops. More overlap reduces splits; slower.\n"
            "• crop_points_downscale — fewer points on deeper crop layers (keeps runtime sane).\n"
            "• box_nms_thresh — IoU threshold to suppress duplicate masks (on boxes).\n"
            "• min_mask_region_area — drops tiny noisy regions (px^2).\n"
            "• use_m2m — extra mask-to-mask refinement/merging.\n"
            "• output_mode — 'binary_mask' for boolean arrays (best for PNGs).\n"
        )
        try:
            messagebox.showinfo("Mask Generator parameters", txt)
        except Exception:
            print(txt)


    




if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.3)
    except Exception:
        pass
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    app = LeafSegmenterGUI(root)
    root.mainloop()
