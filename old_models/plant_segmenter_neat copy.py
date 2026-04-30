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
    edges = cv2.Canny(gray, 80, 180)  # tweak if needed

    if width > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width*2+1, width*2+1))
        edges = cv2.dilate(edges, k, iterations=1)

    mask = (edges > 0)[..., None]
    out  = rgb.astype(np.float32)
    out[mask] *= (1.0 - amount)
    return np.clip(out, 0, 255).astype(np.uint8)


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

class LeafSegmenterGUI:
    def __init__(self, root):
        self.root = root
        root.title("Leaf Segmenter (SAM2)")

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

        self.make_model_frame(root)
        self.make_options_frame(root)
        self.make_actions_frame(root)
        self.make_preview_frame(root)
        self.make_masks_frame(root)
        

        # click-to-pick editing state
        self._edit_mode = tk.StringVar(value="none")   # 'none' | 'deselect' | 'select'
        self._picks = set()                            # mask indices clicked on canvas

        
        # --- batch mode state ---
        self.batch_dir: str | None = None
        self.batch_images: list[str] = []
        self.batch_idx: int = -1

        # UI vars for the Training panel
        self.train_auto_prompt = tk.BooleanVar(value=True)
        self.train_ckpt_var = tk.StringVar(value="")            # falls back to Model panel if left empty
        self.train_cfg_var  = tk.StringVar(value="sam2_hiera_s.yaml")
        self.train_out_var  = tk.StringVar(value=str(Path.home()/ "sam2_arabidopsis.pth"))
        self.train_steps_var= tk.IntVar(value=10000)
        self.train_lr_var   = tk.DoubleVar(value=1e-5)

        # Buil the training panel
        self.make_training_frame(root)

        # click-to-pick editing state
        self._edit_mode = tk.StringVar(value="none")  # 'none' | 'deselect' | 'select'
        self._picks = set()                            # set of mask indices picked on canvas

        
        # --- training state ---
        self.train_root = None              # dataset root containing images/ and masks/
        self.train_images_dir = None
        self.train_masks_dir = None
        self.train_examples = []            # [{ "image": path, "masks": [paths...] }]

        # UI vars for the Training panel
        self.train_auto_prompt = tk.BooleanVar(value=True)
        self.train_ckpt_var = tk.StringVar(value="")            # will default to your Model panel if empty
        self.train_cfg_var  = tk.StringVar(value="sam2_hiera_s.yaml")
        self.train_out_var  = tk.StringVar(value=str(Path.home()/ "sam2_arabidopsis.pth"))
        self.train_steps_var= tk.IntVar(value=10000)
        self.train_lr_var   = tk.DoubleVar(value=1e-5)



        

    # ---- Frames ----
    def make_model_frame(self, root):
        f = ttk.LabelFrame(root, text="Model")
        f.grid(row=0, column=0, padx=8, pady=6, sticky="ew")

        ttk.Label(f, text="Checkpoint:").grid(row=0, column=0, sticky="w")
        self.e_ckpt = ttk.Entry(f, width=54)
        self.e_ckpt.grid(row=0, column=1, padx=4)
        ttk.Button(f, text="…", command=self.pick_ckpt).grid(row=0, column=2)

        ttk.Label(f, text="Config (name / YAML / dir):").grid(row=1, column=0, sticky="w")
        self.e_cfg = ttk.Entry(f, width=54)
        self.e_cfg.insert(0, "sam2.1_hiera_l")
        self.e_cfg.grid(row=1, column=1, padx=4)
        ttk.Button(f, text="Pick Config…", command=self.pick_cfg).grid(row=1, column=2)

        ttk.Label(f, text="Device:").grid(row=2, column=0, sticky="w")
        self.e_dev = ttk.Entry(f, width=12)
        self.e_dev.insert(0, "cpu")  # or "cuda" / "mps"
        self.e_dev.grid(row=2, column=1, sticky="w")
        self.chk_post = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="apply_postprocessing", variable=self.chk_post).grid(row=2, column=1, sticky="e")

        ttk.Button(f, text="Load Model",   command=self.load_model).grid(row=0, column=3, rowspan=3, padx=6)
        ttk.Button(f, text="Load Bundle…", command=self.load_bundle).grid(row=0, column=4, rowspan=3, padx=6)


    def make_options_frame(self, root):
        f = ttk.LabelFrame(root, text="Enhancement & Segmentation Options")
        f.grid(row=1, column=0, padx=8, pady=6, sticky="ew")

        openbar = ttk.Frame(f)
        openbar.grid(row=0, column=0, padx=2, pady=2, sticky="w")

        ttk.Button(openbar, text="Open Image…",  command=self.open_image).pack(side="left")
        ttk.Button(openbar, text="Open Folder…", command=self.open_folder).pack(side="left", padx=(6, 0))

        #ttk.Button(f, text="Open Image…", command=self.open_image).grid(row=0, column=0, padx=2, pady=2, sticky="w")
        rot = ttk.Labelframe(f, text="Rotate")
        rot.grid(row=0, column=1, columnspan=2, padx=4, sticky="w")

        # the circular knob
        self._knob = tk.Canvas(rot, width=72, height=72, bg="#ddd", highlightthickness=0)
        self._knob.grid(row=0, column=0, rowspan=2, padx=(2,6), pady=2)
        self._knob.bind("<Button-1>", self._knob_down)
        self._knob.bind("<B1-Motion>", self._knob_drag)

        # add this line – a backing var that says "apply rotation"
        self.chk_rotate = getattr(self, "chk_rotate", tk.BooleanVar(value=True))

        # degree spinbox
        self.spin_angle = ttk.Spinbox(rot, from_=-180, to=180, increment=1,
                                    textvariable=self.rot_angle, width=6,
                                    command=self._angle_from_spin)
        self.spin_angle.grid(row=0, column=1, sticky="w")
        ttk.Button(rot, text="Reset", command=lambda: self._set_angle(0)).grid(row=1, column=1, sticky="w", pady=(2,0))

        # also update when user types then leaves/presses Enter
        self.spin_angle.bind("<Return>", lambda e: self._angle_from_spin())
        self.spin_angle.bind("<FocusOut>", lambda e: self._angle_from_spin())
        self._draw_knob()  # initial paint

        
        # --- Image Enhancements (boxed, tidy 3-per-row) --------------------------------
        # checkboxes so we can stack both pipelines
        self.use_green   = getattr(self, "use_green",   tk.BooleanVar(value=True))
        self.use_classic = getattr(self, "use_classic", tk.BooleanVar(value=False))
        
        enh = ttk.Labelframe(f, text="Image Enhancements")
        enh.grid(row=1, column=0, columnspan=6, sticky="ew", padx=2, pady=(8, 4))

        # Make columns 1,3,5 stretchy so entries/scales breathe
        for c in (1, 3, 5):
            enh.grid_columnconfigure(c, weight=1)

        # Mode
        ttk.Checkbutton(enh, text="Green-aware enhancer", variable=self.use_green).grid(row=0, column=0, columnspan=3, sticky="w", padx=2, pady=(2,0))
        ttk.Checkbutton(enh, text="Classic preprocess",   variable=self.use_classic).grid(row=0, column=3, columnspan=3, sticky="w", padx=2, pady=(2,0))

        # Sliders (full width)
        self.s_brightness = getattr(self, "s_brightness", tk.IntVar(value=-25))
        self.s_contrast   = getattr(self, "s_contrast",   tk.DoubleVar(value=1.0))
        self.s_gamma      = getattr(self, "s_gamma",      tk.DoubleVar(value=1.2))

        ttk.Label(enh, text="Brightness").grid(row=1, column=0, sticky="w")
        ttk.Scale(enh, from_=-100, to=100, variable=self.s_brightness,
                orient="horizontal").grid(row=1, column=1, columnspan=5, sticky="ew")

        ttk.Label(enh, text="Contrast").grid(row=2, column=0, sticky="w")
        ttk.Scale(enh, from_=0.5, to=2.0, variable=self.s_contrast,
                orient="horizontal").grid(row=2, column=1, columnspan=5, sticky="ew")

        ttk.Label(enh, text="Gamma").grid(row=3, column=0, sticky="w")
        ttk.Scale(enh, from_=0.5, to=2.0, variable=self.s_gamma,
                orient="horizontal").grid(row=3, column=1, columnspan=5, sticky="ew")

        # Row A: three toggles
        self.chk_unsharp   = getattr(self, "chk_unsharp",   tk.BooleanVar(value=True))
        self.chk_laplacian = getattr(self, "chk_laplacian", tk.BooleanVar(value=False))
        self.chk_whiten    = getattr(self, "chk_whiten",    tk.BooleanVar(value=False))

        ttk.Checkbutton(enh, text="Unsharp",            variable=self.chk_unsharp)  .grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(enh, text="Laplacian",          variable=self.chk_laplacian).grid(row=4, column=2, columnspan=2, sticky="w")
        ttk.Checkbutton(enh, text="Whiten Background",  variable=self.chk_whiten)   .grid(row=4, column=4, columnspan=2, sticky="w")

        # Row B: val_min / sat_max / Close iters
        self.s_val_min     = getattr(self, "s_val_min",     tk.IntVar(value=200))
        self.s_sat_max     = getattr(self, "s_sat_max",     tk.IntVar(value=35))
        self.s_close_iters = getattr(self, "s_close_iters", tk.IntVar(value=1))

        ttk.Label(enh, text="val_min").grid(row=5, column=0, sticky="e")
        ttk.Entry(enh, width=6, textvariable=self.s_val_min).grid(row=5, column=1, sticky="w")
        ttk.Label(enh, text="sat_max").grid(row=5, column=2, sticky="e")
        ttk.Entry(enh, width=6, textvariable=self.s_sat_max).grid(row=5, column=3, sticky="w")
        ttk.Label(enh, text="Close iters").grid(row=5, column=4, sticky="e")
        ttk.Entry(enh, width=6, textvariable=self.s_close_iters).grid(row=5, column=5, sticky="w")

        # Row C: Median denoise / Mean denoise / Halo erode
        self.dn_median_on    = getattr(self, "dn_median_on",    tk.BooleanVar(value=False))
        self.dn_median_ksize = getattr(self, "dn_median_ksize", tk.IntVar(value=5))
        self.dn_mean_on      = getattr(self, "dn_mean_on",      tk.BooleanVar(value=False))
        self.dn_mean_ksize   = getattr(self, "dn_mean_ksize",   tk.IntVar(value=3))
        self.s_halo_erode    = getattr(self, "s_halo_erode",    tk.IntVar(value=1))

        ttk.Checkbutton(enh, text="Median denoise", variable=self.dn_median_on).grid(row=6, column=0, sticky="w")
        ttk.Entry(enh, width=4, textvariable=self.dn_median_ksize).grid(row=6, column=1, sticky="w")
        ttk.Checkbutton(enh, text="Mean denoise",   variable=self.dn_mean_on).grid(row=6, column=2, sticky="w")
        ttk.Entry(enh, width=4, textvariable=self.dn_mean_ksize).grid(row=6, column=3, sticky="w")
        ttk.Label(enh, text="Halo erode px").grid(row=6, column=4, sticky="e")
        ttk.Entry(enh, width=6, textvariable=self.s_halo_erode).grid(row=6, column=5, sticky="w")

        # Row D: Feather px
        self.s_halo_feather = getattr(self, "s_halo_feather", tk.IntVar(value=2))
        ttk.Label(enh, text="Feather px").grid(row=7, column=0, sticky="e")
        ttk.Entry(enh, width=6, textvariable=self.s_halo_feather).grid(row=7, column=1, sticky="w")
        ttk.Separator(enh, orient="horizontal").grid(row=8, column=0, columnspan=6, sticky="ew", pady=(6,0))
        
        
        # --- Edge darken (3 per row): toggle, width, strength ---
        self.ed_on      = getattr(self, "ed_on",      tk.BooleanVar(value=False))
        self.ed_width   = getattr(self, "ed_width",   tk.IntVar(value=3))      # px band around edges
        self.ed_amount  = getattr(self, "ed_amount",  tk.DoubleVar(value=0.35))# 0..1 darkness

        ed_row = 9  # next free row inside the "enh" box (we used 0..8 above)
        ttk.Checkbutton(enh, text="Edge darken", variable=self.ed_on)\
            .grid(row=ed_row, column=0, sticky="w")
        ttk.Label(enh, text="width (px)").grid(row=ed_row, column=2, sticky="e")
        ttk.Entry(enh, width=6, textvariable=self.ed_width).grid(row=ed_row, column=3, sticky="w")
        ttk.Label(enh, text="strength").grid(row=ed_row, column=4, sticky="e")
        ttk.Entry(enh, width=6, textvariable=self.ed_amount).grid(row=ed_row, column=5, sticky="w")


        
        # --- end Image Enhancements ----------------------------------------------------


        # --- Mask Generator controls ---
        row =2
        mg = ttk.Labelframe(f, text="Mask Generator (SAM2)")
        mg.grid(row=row, column=0, columnspan=6, sticky="ew", padx=2, pady=(8, 4))

        # backing variables
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
        self.m_output_mode      = tk.StringVar(value="binary_mask")  # other: coco_rle, uncompressed_rle, polygons

        r = 0
        ttk.Label(mg, text="points_per_side").grid(row=r, column=0, sticky="e")
        ttk.Entry(mg, width=7, textvariable=self.m_points_per_side).grid(row=r, column=1, sticky="w", padx=3)
        ttk.Label(mg, text="points_per_batch").grid(row=r, column=2, sticky="e")
        ttk.Entry(mg, width=7, textvariable=self.m_points_per_batch).grid(row=r, column=3, sticky="w", padx=3)
        ttk.Checkbutton(mg, text="use_m2m", variable=self.m_use_m2m).grid(row=r, column=4, sticky="w")
        r += 1

        ttk.Label(mg, text="pred_iou_thresh").grid(row=r, column=0, sticky="e")
        ttk.Entry(mg, width=7, textvariable=self.m_pred_iou_thresh).grid(row=r, column=1, sticky="w", padx=3)
        ttk.Label(mg, text="stability_score_thresh").grid(row=r, column=2, sticky="e")
        ttk.Entry(mg, width=7, textvariable=self.m_stability_score_thresh).grid(row=r, column=3, sticky="w", padx=3)
        ttk.Label(mg, text="output_mode").grid(row=r, column=4, sticky="e")
        ttk.Combobox(mg, width=14, state="readonly",
                     textvariable=self.m_output_mode,
                     values=("binary_mask", "coco_rle", "uncompressed_rle", "polygons")).grid(row=r, column=5, sticky="w")
        r += 1

        ttk.Label(mg, text="crop_n_layers").grid(row=r, column=0, sticky="e")
        ttk.Entry(mg, width=7, textvariable=self.m_crop_n_layers).grid(row=r, column=1, sticky="w", padx=3)
        ttk.Label(mg, text="crop_overlap_ratio").grid(row=r, column=2, sticky="e")
        ttk.Entry(mg, width=7, textvariable=self.m_crop_overlap_ratio).grid(row=r, column=3, sticky="w", padx=3)
        ttk.Label(mg, text="crop_points_downscale").grid(row=r, column=4, sticky="e")
        ttk.Entry(mg, width=7, textvariable=self.m_crop_n_points_downscale_factor).grid(row=r, column=5, sticky="w", padx=3)
        r += 1

        ttk.Label(mg, text="box_nms_thresh").grid(row=r, column=0, sticky="e")
        ttk.Entry(mg, width=7, textvariable=self.m_box_nms_thresh).grid(row=r, column=1, sticky="w", padx=3)
        ttk.Label(mg, text="min_mask_region_area").grid(row=r, column=2, sticky="e")
        ttk.Entry(mg, width=7, textvariable=self.m_min_mask_region_area).grid(row=r, column=3, sticky="w", padx=3)
        ttk.Button(mg, text="Explain…", command=self.explain_mask_params).grid(row=r, column=5, sticky="e", padx=2)
        # --- end Mask Generator controls ---

        # buttons
        btnrow = row + 1
        ttk.Button(f, text="Preview Enhance", command=self.preview_enhance)\
        .grid(row=btnrow, column=0, pady=4, sticky="w")
        ttk.Button(f, text="Segment", command=self.segment)\
        .grid(row=btnrow, column=1, pady=4, sticky="w")

        # --- Phenotype options ---
        row = btnrow + 1   # <— move phenotypes to the next row so they don't overlap the buttons
        ph = ttk.Labelframe(f, text="Phenotypes")
        ph.grid(row=row, column=0, columnspan=6, sticky="ew", padx=2, pady=(8, 4))


        self.ph_all    = tk.BooleanVar(value=True)
        self.ph_area   = tk.BooleanVar(value=True)
        self.ph_len    = tk.BooleanVar(value=True)   # length-like metrics
        self.ph_wid    = tk.BooleanVar(value=True)   # width-like metrics
        self.ph_color  = tk.BooleanVar(value=True)  # RGB
        self.ph_hsv   = tk.BooleanVar(value=True)   # HSV


        def _sync_ph(*_):
            # "All" mirrors other boxes; manual changes flip All off/on
            if self.ph_all.get():
                self.ph_area.set(True); self.ph_len.set(True); self.ph_wid.set(True); self.ph_color.set(True); self.ph_hsv.set(True)
            else:
                if all(v.get() for v in (self.ph_area, self.ph_len, self.ph_wid, self.ph_color, self.ph_hsv)):
                    self.ph_all.set(True)

        ttk.Checkbutton(ph, text="All",     variable=self.ph_all,   command=_sync_ph).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(ph, text="Area",    variable=self.ph_area,  command=lambda: self.ph_all.set(False)).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(ph, text="Length",  variable=self.ph_len,   command=lambda: self.ph_all.set(False)).grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(ph, text="Width",   variable=self.ph_wid,   command=lambda: self.ph_all.set(False)).grid(row=0, column=3, sticky="w")
        ttk.Checkbutton(ph, text="Color",   variable=self.ph_color, command=lambda: self.ph_all.set(False)).grid(row=0, column=4, sticky="w")
        ttk.Checkbutton(ph, text="HSV",     variable=self.ph_hsv, command=lambda: self.ph_all.set(False)).grid(row=0, column=5, sticky="w")


        
    def make_actions_frame(self, root):
        f = ttk.LabelFrame(root, text="Save")
        f.grid(row=2, column=0, padx=8, pady=6, sticky="ew")

        row = 0
        ttk.Button(
            f, text="Save ALL Masks…", width=22,
            command=self.save_all_masks
        ).grid(row=row, column=0, padx=4, pady=4, sticky="w")

        ttk.Button(
            f, text="Save Selected Masks…", width=22,
            command=self.save_selected_masks
        ).grid(row=row, column=1, padx=4, pady=4, sticky="w")

        row += 1
        ttk.Button(
            f, text="Save INDIVIDUAL phenotypes…", width=28,
            command=self.export_individual_phenotypes
        ).grid(row=row, column=0, padx=4, pady=4, sticky="w")

        ttk.Button(
            f, text="Save JOINT phenotypes…", width=24,
            command=self.export_joint_phenotypes
        ).grid(row=row, column=1, padx=4, pady=4, sticky="w")


    def make_preview_frame(self, root):
        f = ttk.LabelFrame(root, text="Preview")
        f.grid(row=0, column=1, rowspan=3, padx=8, pady=6, sticky="nsew")
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(2, weight=1)

        # toolbar
        bar = ttk.Frame(f)
        bar.pack(fill="x", padx=4, pady=(4,0))

        ttk.Button(bar, text="−", width=3, command=lambda: self._zoom_by(0.8)).pack(side="left")
        ttk.Button(bar, text="Fit", width=4, command=self._zoom_fit).pack(side="left", padx=4)
        ttk.Button(bar, text="+", width=3, command=lambda: self._zoom_by(1.25)).pack(side="left")

        # crop tools
        ttk.Checkbutton(bar, text="Crop", width=6, style="Toolbutton",
                        variable=self._crop_mode, command=self._set_crop_mode).pack(side="left", padx=(12,4))
        self._btn_crop_apply  = ttk.Button(bar, text="Apply",  width=6, command=self._apply_crop, state="disabled")
        self._btn_crop_cancel = ttk.Button(bar, text="Cancel", width=6, command=self._cancel_crop, state="disabled")
        self._btn_crop_apply.pack(side="left", padx=2)
        self._btn_crop_cancel.pack(side="left", padx=(2,0))

        # canvas
        self.canvas = tk.Canvas(f, width=640, height=640, bg="#202020", highlightthickness=0, cursor="tcross")
        self.canvas.pack(fill="both", expand=True)

        # --- second row under the canvas (interactive pick controls) ---
        self.pickbar = ttk.Frame(f)
        self.pickbar.pack(fill="x", padx=4, pady=(4, 6))

        ttk.Button(self.pickbar, text="Deselect", width=9,
                command=lambda: self._set_edit_mode("deselect")).pack(side="left")
        ttk.Button(self.pickbar, text="Select",   width=9,
                command=lambda: self._set_edit_mode("select")).pack(side="left", padx=(6, 0))
        ttk.Button(self.pickbar, text="Reset",    width=9,
                command=self._reset_pick).pack(side="left", padx=(12, 0))
        ttk.Button(self.pickbar, text="Apply",    width=9,
                command=self._apply_pick).pack(side="left", padx=(6, 0))
        ttk.Button(self.pickbar, text="Combine", width=9,
           command=self._apply_combine_from_picks).pack(side="left", padx=(6, 0))

        self._pick_status = ttk.Label(self.pickbar, text="", anchor="w")
        self._pick_status.pack(side="left", padx=(12, 0))


        # mouse: wheel = zoom, drag = pan
        self.canvas.bind("<MouseWheel>", self._on_wheel)       # Windows / macOS
        self.canvas.bind("<Button-4>",  lambda e: self._on_wheel(e, delta=+120))  # X11
        self.canvas.bind("<Button-5>",  lambda e: self._on_wheel(e, delta=-120))  # X11
        self.canvas.bind("<ButtonPress-1>", self._pan_start)
        self.canvas.bind("<B1-Motion>",    self._pan_move)

        ttk.Separator(bar, orient="vertical").pack(side="left", padx=6, fill="y")
        ttk.Button(bar, text="Prev", width=5, command=self.prev_image).pack(side="left")
        ttk.Button(bar, text="Next", width=5, command=self.next_image).pack(side="left", padx=(4, 0))
        
        # second toolbar row under the canvas…
        

        
        # small status label: “(3 / 42)”
        self._batch_status = ttk.Label(bar, text="", width=10, anchor="w")
        self._batch_status.pack(side="left", padx=(8, 0))

        ttk.Separator(bar, orient="vertical").pack(side="left", padx=6, fill="y")
        ttk.Button(bar, text="Segment ALL…", command=self.segment_all_batch).pack(side="left")

        # --- pick mode state (for click-to-select on the canvas) ---
        self._edit_mode = tk.StringVar(value="none")   # "select" | "deselect" | "none"
        self._picks: set[int] = set()
        self._picks_action: str | None = None          # remembers last action for Apply


        # re-render when canvas resizes
        self.canvas.bind("<Configure>", lambda e: self._render_preview())



        # ensure correct LMB behavior at startup
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




    def make_masks_frame(self, root):
        f = ttk.LabelFrame(root, text="Masks")
        # give the frame room so the list can grow
        f.grid(row=0, column=2, rowspan=3, padx=8, pady=6, sticky="nsew")

        # --- TOP TOOLBAR ---
        bar = ttk.Frame(f)
        bar.pack(fill="x", padx=4, pady=(4, 0))
        ttk.Button(bar, text="Delete selected", command=self.delete_selected_masks).pack(side="left")
        ttk.Button(bar, text="Clear all",       command=self.clear_all_masks).pack(side="left", padx=(6, 0))
        ttk.Button(bar, text="Combine selected",command=self.combine_selected_masks).pack(side="left", padx=(6, 0))


        # --- SCROLLABLE LIST AREA (wrapped) ---
        wrap = ttk.Frame(f)
        wrap.pack(fill="both", expand=True, padx=4, pady=(4, 6))

        self.lb = tk.Listbox(wrap, width=38, height=32, selectmode="extended")
        self.lb.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.lb.yview)
        sb.pack(side="right", fill="y")
        self.lb.config(yscrollcommand=sb.set)

        # selection + keyboard helpers
        self.lb.bind("<<ListboxSelect>>", self.on_select_mask)
        self.lb.bind("<Delete>",     lambda e: self.delete_selected_masks())
        self.lb.bind("<BackSpace>",  lambda e: self.delete_selected_masks())
        self.lb.bind("<Control-a>",  lambda e: (self.lb.select_set(0, tk.END), "break"))
        self.lb.bind("<Command-a>",  lambda e: (self.lb.select_set(0, tk.END), "break"))  # macOS

        

        # Small button bar under the list
        #btns = ttk.Frame(f)
        #btns.pack(fill="x", padx=2, pady=(4, 0))
        #ttk.Button(btns, text="Delete selected", command=self.delete_selected_masks).pack(side="left")
        #ttk.Button(btns, text="Clear all",       command=self.clear_all_masks).pack(side="left", padx=(6, 0))

    def make_training_frame(self, root):
        tf = ttk.LabelFrame(root, text="Training")
        tf.grid(row=3, column=0, columnspan=3, padx=8, pady=(4,8), sticky="ew")
        root.grid_rowconfigure(3, weight=0)

        # --- dataset root
        r = 0
        ttk.Label(tf, text="Dataset folder:").grid(row=r, column=0, sticky="w")
        self.train_root_var = tk.StringVar(value="")
        ttk.Entry(tf, textvariable=self.train_root_var, width=64).grid(row=r, column=1, sticky="ew", padx=4)
        ttk.Button(tf, text="Choose…", command=self._pick_train_root).grid(row=r, column=2)
        tf.grid_columnconfigure(1, weight=1)

        # --- example collection bar
        r += 1
        self.train_msg = ttk.Label(tf, text="0 examples", anchor="w")
        self.train_msg.grid(row=r, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(tf, text="Prompt me after each Segment", variable=self.train_auto_prompt).grid(row=r, column=2, sticky="e")

        r += 1
        bar = ttk.Frame(tf); bar.grid(row=r, column=0, columnspan=3, sticky="w", pady=(2,4))
        ttk.Button(bar, text="Add current segmentation as example", command=self._add_current_to_training).pack(side="left")
        ttk.Button(bar, text="Open dataset folder", command=self._open_train_root).pack(side="left", padx=(8,0))
        ttk.Button(bar, text="Clear examples", command=self._clear_training_set).pack(side="left", padx=(8,0))

        # --- training params
        r += 1
        grid = ttk.Frame(tf); grid.grid(row=r, column=0, columnspan=3, sticky="ew")
        ttk.Label(grid, text="Checkpoint (.pt)").grid(row=0, column=0, sticky="e")
        ttk.Entry(grid, textvariable=self.train_ckpt_var, width=48).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(grid, text="…", command=lambda: self._browse_file_into(self.train_ckpt_var)).grid(row=0, column=2)

        ttk.Label(grid, text="Config (.yaml)").grid(row=1, column=0, sticky="e")
        ttk.Entry(grid, textvariable=self.train_cfg_var, width=48).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(grid, text="…", command=lambda: self._browse_file_into(self.train_cfg_var, ("YAML","*.yaml *.yml"))).grid(row=1, column=2)

        ttk.Label(grid, text="Save to (.pth)").grid(row=2, column=0, sticky="e")
        ttk.Entry(grid, textvariable=self.train_out_var, width=48).grid(row=2, column=1, sticky="ew", padx=4)
        ttk.Button(grid, text="…", command=lambda: self._browse_save_into(self.train_out_var, default_ext=".pth")).grid(row=2, column=2)

        ttk.Label(grid, text="Steps").grid(row=3, column=0, sticky="e")
        ttk.Spinbox(grid, from_=100, to=200000, increment=100, textvariable=self.train_steps_var, width=10).grid(row=3, column=1, sticky="w")
        ttk.Label(grid, text="LR").grid(row=3, column=2, sticky="e")
        ttk.Entry(grid, textvariable=self.train_lr_var, width=10).grid(row=3, column=3, sticky="w")
        grid.grid_columnconfigure(1, weight=1)

        # --- actions + log
        r += 1
        tbar = ttk.Frame(tf); tbar.grid(row=r, column=0, columnspan=3, sticky="w")
        ttk.Button(tbar, text="Train NOW", command=self._launch_training).pack(side="left")
        ttk.Button(tbar, text="Load fine-tuned into predictor", command=self._load_finetuned_into_predictor).pack(side="left", padx=(8,0))

        r += 1
        self.train_log = tk.Text(tf, height=10, width=100)
        self.train_log.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(4,2))

    def _browse_file_into(self, var, ftypes=("All","*.*")):
        p = filedialog.askopenfilename(filetypes=[ftypes] if isinstance(ftypes, tuple) else [("All","*.*")])
        if p: var.set(p)

    def _browse_save_into(self, var, default_ext=".pth"):
        p = filedialog.asksaveasfilename(defaultextension=default_ext,
                                        filetypes=[("Torch","*.pth *.pt"), ("All","*.*")])
        if p: var.set(p)

    def _pick_train_root(self):
        d = filedialog.askdirectory(title="Choose dataset root (will create images/ and masks/)")
        if not d: return
        self.train_root = d
        self.train_root_var.set(d)
        self._ensure_train_dirs()

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

        py = shlex.quote(sys.executable)
        cmd = (
            f"{py} sam2_trainer_arabidopsis.py "
            f"--images {shlex.quote(str(self.train_images_dir))} "
            f"--masks {shlex.quote(str(self.train_masks_dir))} "
            f"--checkpoint {shlex.quote(ckpt)} "
            f"--config {shlex.quote(cfg)} "
            f"--save_to {shlex.quote(outp)} --steps {steps} --lr {lr}"
        )

        self._append_train_log("")
        self._append_train_log("Launching training:\n" + cmd)
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
            sd = torch.load(self.train_out_var.get().strip(), map_location=(self.e_dev.get().strip() or "cpu"))
            self.sam2_model.load_state_dict(sd, strict=False)
            self.sam2_model.eval()
            messagebox.showinfo("Model", "Fine-tuned weights loaded.")
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

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


    def _rebuild_mask_list(self):
        """Refresh the right-list labels to current indices/areas/bboxes."""
        self.lb.delete(0, tk.END)
        if not self.sr or not self.sr.masks:
            return
        for i, m in enumerate(self.sr.masks):
            self.lb.insert(tk.END, f"[{i:03d}] area={int(m['area'])} bbox={list(map(int, m['bbox']))}")

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
        self._rebuild_mask_list()
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
            self._picks.clear()
            self._rebuild_mask_list()
            self._render_preview()
        else:
            messagebox.showwarning("Combine", "Couldn’t combine those selections.")


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
            self._render_preview()
        else:
            messagebox.showwarning("Combine", "Couldn’t combine those selections.")



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

        self.img_path = p
        arr = ensure_uint8_rgb(Image.open(p))
        self.img_orig = arr                 # keep the unmodified original
        self.img_preview = None
        self.sr = None
        self.lb.delete(0, tk.END)

        # apply current angle to produce the working image
        self.img = self._base_image()
        self.show_image(self.img)

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
                bundle = torch.load(p, map_location="cpu")  # ok to leave weights_only default
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
        i = max(0, min(len(self.batch_images) - 1, int(i)))
        self.batch_idx = i
        p = self.batch_images[i]
        # same as open_image, but from a known path
        self.img_path = p
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
    def _segment_sync_for_array(self, arr_rgb_uint8):
        arr_rgb_uint8 = self._apply_denoise(arr_rgb_uint8)
        seg_img = self._enhance_pipeline(arr_rgb_uint8)

        gen = self.build_mask_generator()
        masks = gen.generate(seg_img)
        masks = dedupe_by_mask_iou(masks, iou_thresh=0.80)
        return masks, seg_img, arr_rgb_uint8

    ### batch segmenter

    def segment_all_batch(self):
        if not self.batch_images:
            messagebox.showwarning("No folder", "Open a folder first.")
            return
        if self.sam2_model is None:
            messagebox.showwarning("No model", "Load the SAM2 model first.")
            return

        out_root = filedialog.askdirectory(title="Choose output folder for all masks/crops")
        if not out_root:
            return
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        # reuse the current rotation/params everywhere
        angle = float(self.rot_angle.get())

        # small progress dialog in terminal
        print(f"[Batch] Processing {len(self.batch_images)} images with angle={angle:.2f}° …")

        # optional simple “are you sure”
        if not messagebox.askyesno("Segment ALL", f"Run SAM2 on {len(self.batch_images)} images using current settings?\n\nMasks and crops will be saved under:\n{out_root}"):
            return

        total_masks = 0
        for idx, p in enumerate(self.batch_images, 1):
            try:
                rgb = ensure_uint8_rgb(Image.open(p))
                # apply the SAME rotation picked in the UI
                if abs(angle) > 1e-6:
                    rgb = self._rotate_any(rgb, angle)

                masks, seg_img, color_img = self._segment_sync_for_array(rgb)

                # save per-image outputs
                stem = Path(p).stem
                img_dir = out_root / stem
                img_dir.mkdir(exist_ok=True, parents=True)

                rows = []
                erode_px = max(0, int(self.s_halo_erode.get())) if hasattr(self, "s_halo_erode") else 1
                feather_px = max(0, int(self.s_halo_feather.get())) if hasattr(self, "s_halo_feather") else 2

                for k, m in enumerate(masks, 1):
                    seg_bool = m["segmentation"].astype(bool)
                    # mask & crop names: FileStem_1.png, FileStem_crop_1.png, etc.
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

                # manifest per image
                csv_path = img_dir / f"{stem}_mask_manifest.csv"
                with open(csv_path, "w", newline="") as f:
                    import csv as _csv
                    w = _csv.DictWriter(f, fieldnames=["mask_idx","area_px","bbox","mask_png","crop_png"])
                    w.writeheader(); w.writerows(rows)

                total_masks += len(masks)
                print(f"  [{idx}/{len(self.batch_images)}] {Path(p).name}: {len(masks)} masks")

            except Exception as e:
                print(f"  [skip] {p}: {e}")

        messagebox.showinfo("Batch done", f"Processed {len(self.batch_images)} images.\nTotal masks: {total_masks}\n\nSaved under:\n{out_root}")


        


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

        # face
        cv.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#fafafa", outline="#888")

        # tick marks every 30°
        import math
        for a in range(-180, 181, 30):
            rad = math.radians(a)
            x0 = cx + (r - 10) * math.cos(rad); y0 = cy - (r - 10) * math.sin(rad)
            x1 = cx + (r - 2)  * math.cos(rad); y1 = cy - (r - 2)  * math.sin(rad)
            cv.create_line(x0, y0, x1, y1, fill="#aaa")

        # indicator dot
        deg = float(self.rot_angle.get())
        rad = math.radians(deg)
        hx = cx + (r - 6) * math.cos(rad)
        hy = cy - (r - 6) * math.sin(rad)
        cv.create_oval(hx - 4, hy - 4, hx + 4, hy + 4, fill="#333", outline="")

        # current angle text
        cv.create_text(cx, cy + r + 10, text=f"{deg:.0f}°", fill="#333")


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
            messagebox.showwarning("No image", "Open an image first.")
            return
        arr2 = self._apply_enhance_pipeline(arr)
        self.img_preview = arr2
        self.show_image(arr2)




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
        # 1) Green-aware (optional)
        if bool(self.use_green.get()):
            x = enhance_leaf_edges_rgb(x)
        # 2) Classic preprocess (optional)
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
        # 3) Denoise toggles (median / mean) if you have them
        x = self._apply_denoise(x)

        # 4) Edge darken as the last step before returning
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

        # draw selection overlay on top (if any)
        self._draw_crop_overlay()

        # (optional) replace inline drawing with helper:
        self._draw_pick_overlays()   # <-- add this line

        if getattr(self, "_picks", None):
            # we’ll draw in canvas coords, so transform bboxes
            for i in self._picks:
                if i <= 0 or i >= len(self.sr.masks):
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

        # normalize & set
        mode = (mode or "none").lower()
        if mode not in ("select", "deselect"):
            mode = "none"
        self._edit_mode.set(mode)

        # fresh selection slate when entering a mode
        if mode != "none":
            self._picks.clear()

        # rebind + redraw
        self._bind_canvas_events()
        self._render_preview()

    def _reset_pick(self):
        self._picks.clear()
        self._edit_mode.set("none")
        self._pick_status.configure(text="")
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
        self._pick_status.configure(text="")
        self._rebuild_mask_list()
        self._bind_canvas_events()
        self._render_preview()


    def _mask_index_at(self, x_img: int, y_img: int):
        if not self.sr or not self.sr.masks:
            return None
        hit = None
        best_area = -1
        for i, m in enumerate(self.sr.masks):
            seg = m.get("segmentation")
            if seg is None:
                continue
            try:
                if 0 <= y_img < seg.shape[0] and 0 <= x_img < seg.shape[1] and seg[y_img, x_img]:
                    a = m.get("area", 0)
                    if a > best_area:
                        best_area = a
                        hit = i
            except Exception:
                pass
        return hit

    def _on_pick_click(self, event):
        if not self.sr or not self.sr.masks:
            return
        pt = self._canvas_to_image_xy(event.x, event.y)
        if not pt:
            return
        idx = self._mask_index_at(*pt)
        if idx is None:
            return
        if idx in self._picks:
            self._picks.remove(idx)
        else:
            self._picks.add(idx)
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
            messagebox.showwarning("No image", "Open an image first.")
            return
        if self.sam2_model is None:
            messagebox.showwarning("No model", "Load the SAM2 model first.")
            return
        threading.Thread(target=self._segment_worker, daemon=True).start()

    def _segment_worker(self):
        """Run SAM2 on the enhanced *rotated* image and update the UI."""
        try:
            arr = self.img.copy()  # or self._base_image(), whichever you use
            if arr is None:
                raise RuntimeError("No image loaded.")
            seg_img = self._apply_enhance_pipeline(arr)  # <— one path for both preview & segment
            #seg_img = self._apply_enhance_pipeline(arr)

            gen = make_mask_generator(self.sam2_model)
            masks = gen.generate(seg_img)
            masks = dedupe_by_mask_iou(masks, iou_thresh=0.80)

            self.sr = SegResult(masks=masks, img_color=self.img.copy(),
                                img_seg=seg_img, rotate_applied=self.chk_rotate.get())

            self.lb.delete(0, tk.END)
            for i, m in enumerate(masks):
                self.lb.insert(tk.END, f"[{i:03d}] area={int(m['area'])} bbox={list(map(int, m['bbox']))}")
            self.show_image(seg_img)
            messagebox.showinfo("Segmentation", f"Found {len(masks)} masks.")
        except Exception as e:
            messagebox.showerror("Segmentation error", str(e))

            gen = self.build_mask_generator()
            masks = gen.generate(seg_img)
            masks = dedupe_by_mask_iou(masks, iou_thresh=0.80)

            # store results against the rotated color image
            self.sr = SegResult(
                masks=masks,
                img_color=arr.copy(),
                img_seg=seg_img,
                rotate_applied=abs(self.rot_angle.get()) > 1e-3,
            )

            # refresh UI
            self.lb.delete(0, tk.END)
            for i, m in enumerate(masks):
                self.lb.insert(tk.END, f"[{i:03d}] area={int(m['area'])} bbox={list(map(int, m['bbox']))}")

            self.show_image(seg_img)
            messagebox.showinfo("Segmentation", f"Found {len(masks)} masks.")

        except Exception as e:
            messagebox.showerror("Segmentation error", str(e))
        
        self.root.after(0, self._after_segment_prompt)




    # ---- Select mask and preview crop ----
    def on_select_mask(self, event=None):
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
        rgba = np.dstack([crop, alpha])
        tile = 16
        H, W = rgba.shape[:2]
        chk = np.indices((H, W)).sum(axis=0) // tile
        bg = np.where((chk % 2)[..., None], 200, 160).astype(np.uint8)
        comp = rgba[..., :3].copy()
        a = alpha.astype(np.float32) / 255.0
        comp = (comp * a + bg * (1 - a)).astype(np.uint8)
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
        res = dict()

        if flags["area"]:
            res["area_px2"] = int(mask.sum())

        if flags["length"] or flags["width"]:
            maj, minw, axis_w, axis_h = _pca_major_minor(mask)
            if flags["length"]:
                res["length_major_px"] = round(maj, 2)
                # also paper-style length
            if flags["width"]:
                res["width_minor_px"]  = round(minw, 2)

            paper = _length_width_after_deskew(mask)
            if flags["length"]:
                res["length_bbox_px"] = round(paper["length_px"], 2)
                res["deskew_angle_deg"] = round(paper["angle_deg"], 2)
            if flags["width"]:
                res["width_row_max_px"] = round(paper["width_px_max"], 2)
                res["width_row_p95_px"] = round(paper["width_px_p95"], 2)

            # simple axis-aligned bbox for reference
            if flags["width"]:
                res["axis_width_px"]  = int(axis_w)
            if flags["length"]:
                res["axis_height_px"] = int(axis_h)

        if flags["color"]:
            R, G, B = _color_stats(rgb, mask)
            # full stats
            res.update({
                "mean_R": round(R["mean"],3), "mean_G": round(G["mean"],3), "mean_B": round(B["mean"],3),
                "median_R": round(R["median"],3), "median_G": round(G["median"],3), "median_B": round(B["median"],3),
                "sum_R": round(R["sum"],1), "sum_G": round(G["sum"],1), "sum_B": round(B["sum"],1),
                "std_R": round(R["std"],3), "std_G": round(G["std"],3), "std_B": round(B["std"],3),
            })
        if flags.get("hsv"):
            Hstats, Sstats, Vstats = _color_stats_hsv(rgb, mask_bool)
            out.update({
                "mean_H": round(Hstats["mean"], 3), "mean_S": round(Sstats["mean"], 3), "mean_V": round(Vstats["mean"], 3),
                "median_H": round(Hstats["median"], 3), "median_S": round(Sstats["median"], 3), "median_V": round(Vstats["median"], 3),
                "sum_H": round(Hstats["sum"], 1), "sum_S": round(Sstats["sum"], 1), "sum_V": round(Vstats["sum"], 1),
                "std_H": round(Hstats["std"], 3), "std_S": round(Sstats["std"], 3), "std_V": round(Vstats["std"], 3),
            })
        return res

    # ---------- common mask saver ----------
    def _export_masks(self, indices, out_dir: Path):
        """Save mask PNG + RGBA crop for given indices. Returns manifest rows."""
        base_name = Path(self.img_path).stem if self.img_path else "Image"
        rows = []
        erode_px = 1
        feather_px = 2
        close_iters = int(self.s_close_iters.get() if hasattr(self, "s_close_iters") else 0)

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
            crop_path = out_dir / f"{seg_id}.crop.png"
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
        messagebox.showinfo("Saved", f"Exported {len(rows)} masks to:\n{out_dir}")

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
        messagebox.showinfo("Saved", f"Exported {len(rows)} selected masks to:\n{out_dir}")


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
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            self.sam2_model = None
            self.mask_generator = None

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
        if flags["color"]:  cols += ["mean_R","mean_G","mean_B","median_R","median_G","median_B",
                                     "sum_R","sum_G","sum_B","std_R","std_G","std_B"]
        if flags["hsv"]:    cols += ["mean_H","mean_S","mean_V","median_H","median_S","median_V",
                                     "sum_H","sum_S","sum_V","std_H","std_S","std_V"]

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader(); writer.writerows(rows)
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

        if flags["color"]:
            # for color stats, averaging the means makes sense; sums we also sum/mean
            for key in ("mean_R","mean_G","mean_B","median_R","median_G","median_B",
                        "sum_R","sum_G","sum_B","std_R","std_G","std_B"):
                if key in agg:
                    row[key + "_mean"]  = round(agg[key]/max(1,n), 3)
                    row[key + "_total"] = round(agg[key], 3)
        if flags["hsv"]:
            for key in ("mean_H","mean_S","mean_V","median_H","median_S","median_V",
                        "sum_H","sum_S","sum_V","std_H","std_S","std_V"):
                if key in agg:
                    row[key + "_mean"]  = round(agg[key]/max(1,n), 3)
                    row[key + "_total"] = round(agg[key], 3)


        # column order
        cols = ["FileName","n_segments"]
        for group in (("area_px2",), 
                      ("length_major_px","length_bbox_px","axis_height_px"),
                      ("width_minor_px","width_row_max_px","width_row_p95_px","axis_width_px")):
            for k in group:
                if k+"_total" in row:
                    cols += [k+"_total", k+"_mean"]
        if "mean_R_mean" in {k for k in row}:  # color included
            for k in ("mean_R","mean_G","mean_B","median_R","median_G","median_B",
                      "sum_R","sum_G","sum_B","std_R","std_G","std_B"):
                if k+"_mean" in row:
                    cols += [k+"_total", k+"_mean"]

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader(); writer.writerow(row)
        messagebox.showinfo("Saved", f"Joint phenotypes written to:\n{out_csv}")

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
        # paint from mask 1 onward so we ignore the big “everything” mask if it exists
        for i, m in enumerate(self.sr.masks[1:], start=1):
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

