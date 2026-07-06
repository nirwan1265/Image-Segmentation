#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phenotyping.py
==============
All plant phenotype measurement functions and CSV export logic.
No tkinter, no GUI state — pure numpy/cv2/math.

Includes:
  - Color statistics  : _color_stats (RGB), _color_stats_hsv
  - Shape geometry    : _pca_angle_deg, _rotate_mask, _pca_major_minor,
                        _pca_orientation_full, _length_width_after_deskew
  - Vegetation indices: _vegetation_indices_stats
  - Full phenotype    : compute_phenotypes  ← main public entry point
  - CSV helpers       : build_individual_row, build_joint_row
"""

import math
import csv
import numpy as np
import cv2

from mask_utils import _ensure_mask_2d, _resize_mask_to_image


# =============================================================================
# Color statistics
# =============================================================================

def _color_stats(rgb: np.ndarray, mask_bool: np.ndarray) -> tuple:
    """
    Compute per-channel (R, G, B) stats within the mask.
    Returns three dicts, each with keys: mean, median, sum, std.
    """
    mask_bool = _ensure_mask_2d(mask_bool)
    if mask_bool is None:
        def _empty(): return dict(mean=0.0, median=0.0, sum=0.0, std=0.0)
        return _empty(), _empty(), _empty()
    mask_bool = _resize_mask_to_image(mask_bool, rgb)
    R = rgb[..., 0][mask_bool].astype(np.float32)
    G = rgb[..., 1][mask_bool].astype(np.float32)
    B = rgb[..., 2][mask_bool].astype(np.float32)

    def _stats(ch):
        if ch.size == 0:
            return dict(mean=0.0, median=0.0, sum=0.0, std=0.0)
        return dict(
            mean=float(ch.mean()),
            median=float(np.median(ch)),
            sum=float(ch.sum()),
            std=float(ch.std()),
        )
    return _stats(R), _stats(G), _stats(B)


def _color_stats_hsv(rgb: np.ndarray, mask_bool: np.ndarray) -> tuple:
    """
    Compute per-channel (H, S, V) stats within the mask.
    OpenCV HSV: H∈[0,179], S,V∈[0,255] for uint8.
    Returns three dicts with keys: mean, median, sum, std.
    """
    rgb8 = np.clip(rgb, 0, 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
    hsv = cv2.cvtColor(rgb8, cv2.COLOR_RGB2HSV)
    mask_bool = _ensure_mask_2d(mask_bool)
    if mask_bool is None:
        def _empty(): return dict(mean=0.0, median=0.0, sum=0.0, std=0.0)
        return _empty(), _empty(), _empty()
    mask_bool = _resize_mask_to_image(mask_bool, hsv)
    H = hsv[..., 0][mask_bool].astype(np.float32)
    S = hsv[..., 1][mask_bool].astype(np.float32)
    V = hsv[..., 2][mask_bool].astype(np.float32)

    def _stats(ch):
        if ch.size == 0:
            return dict(mean=0.0, median=0.0, sum=0.0, std=0.0)
        return dict(
            mean=float(ch.mean()),
            median=float(np.median(ch)),
            sum=float(ch.sum()),
            std=float(ch.std()),
        )
    return _stats(H), _stats(S), _stats(V)


# =============================================================================
# Shape / geometry measurements
# =============================================================================

def _pca_angle_deg(mask_bool: np.ndarray) -> float:
    """Angle (degrees) of the major PCA axis of the mask pixel coordinates."""
    ys, xs = np.nonzero(mask_bool)
    if xs.size < 2:
        return 0.0
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    pts -= pts.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(pts, full_matrices=False)
    vx, vy = Vt[0, 0], Vt[0, 1]
    return math.degrees(math.atan2(vy, vx))


def _rotate_mask(mask_bool: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a boolean mask around its centre by angle_deg degrees."""
    h, w = mask_bool.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    m_u8 = mask_bool.astype(np.uint8) * 255
    m_rot = cv2.warpAffine(
        m_u8, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return m_rot > 0


def _length_width_after_deskew(mask_bool: np.ndarray) -> dict:
    """
    Deskew by PCA angle then measure:
      angle_deg     – rotation applied (degrees)
      length_px     – vertical span after deskew
      width_px_max  – maximum row width
      width_px_p95  – 95th-percentile row width
    """
    ang = _pca_angle_deg(mask_bool)
    m_rot = _rotate_mask(mask_bool, -ang)

    rows_present = np.any(m_rot, axis=1)
    y_idx = np.where(rows_present)[0]
    if y_idx.size == 0:
        return dict(angle_deg=-ang, length_px=0.0,
                    width_px_max=0.0, width_px_p95=0.0)

    y_top, y_bot = int(y_idx.min()), int(y_idx.max())
    length_px = float(y_bot - y_top + 1)

    widths = []
    for y in range(y_top, y_bot + 1):
        xs = np.where(m_rot[y])[0]
        if xs.size:
            widths.append(xs.max() - xs.min() + 1)
    widths = (
        np.array(widths, dtype=np.float32)
        if widths
        else np.array([0.0], dtype=np.float32)
    )
    return dict(
        angle_deg=-ang,
        length_px=float(length_px),
        width_px_max=float(widths.max()),
        width_px_p95=float(np.percentile(widths, 95)),
    )


def _pca_major_minor(mask_bool: np.ndarray) -> tuple:
    """
    PCA-based major/minor axis lengths and bounding-box axis sizes.
    Returns (length_major, width_minor, axis_w, axis_h).
    """
    ys, xs = np.nonzero(mask_bool)
    if xs.size < 2:
        axis_w = int(xs.max() - xs.min() + 1) if xs.size else 0
        axis_h = int(ys.max() - ys.min() + 1) if ys.size else 0
        return 0.0, 0.0, axis_w, axis_h
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mu = pts.mean(axis=0, keepdims=True)
    X = pts - mu
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    proj = X @ V
    length_major = proj[:, 0].max() - proj[:, 0].min()
    width_minor = proj[:, 1].max() - proj[:, 1].min()
    axis_w = xs.max() - xs.min() + 1
    axis_h = ys.max() - ys.min() + 1
    return float(length_major), float(width_minor), int(axis_w), int(axis_h)


# =============================================================================
# Vegetation-index stats (scalar summaries over masked region)
# =============================================================================

def _vegetation_indices_stats(rgb: np.ndarray, mask_bool: np.ndarray) -> dict:
    """
    Compute scalar vegetation index statistics within the mask.
    Returns dict with keys: exg_mean, exr_mean, exgr_mean, gli_mean, green_frac.
    """
    mask_bool = _ensure_mask_2d(mask_bool)
    if mask_bool is None or not mask_bool.any():
        return dict(exg_mean=0.0, exr_mean=0.0, exgr_mean=0.0,
                    gli_mean=0.0, green_frac=0.0)
    mask_bool = _resize_mask_to_image(mask_bool, rgb)
    R = rgb[..., 0][mask_bool].astype(np.float32) / 255.0
    G = rgb[..., 1][mask_bool].astype(np.float32) / 255.0
    B = rgb[..., 2][mask_bool].astype(np.float32) / 255.0

    ExG  = 2 * G - R - B
    ExR  = 1.4 * R - G
    ExGR = ExG - ExR
    GLI  = (2 * G - R - B) / (2 * G + R + B + 1e-6)

    hsv_mask = rgb[..., 1][mask_bool].astype(np.float32)  # green channel proxy
    green_frac = float((G > 0.3).sum()) / max(1, G.size)

    return dict(
        exg_mean=float(ExG.mean()),
        exr_mean=float(ExR.mean()),
        exgr_mean=float(ExGR.mean()),
        gli_mean=float(GLI.mean()),
        green_frac=round(green_frac, 4),
    )


# =============================================================================
# Full per-mask phenotype computation
# =============================================================================

def compute_phenotypes(
    rgb: np.ndarray,
    mask_bool: np.ndarray,
    flags: dict = None,
) -> dict:
    """
    Compute all phenotype measurements for a single mask.

    flags: dict of bool toggles (default all True):
      area, length, width, shape, comp, color, hsv, veg, hsvvar

    Returns a flat dict of float/int values keyed by phenotype name.
    """
    if flags is None:
        flags = {k: True for k in
                 ("area", "length", "width", "shape", "comp",
                  "color", "hsv", "veg", "hsvvar")}

    mask_2d = _ensure_mask_2d(mask_bool)
    if mask_2d is None:
        return {}
    mask_2d = _resize_mask_to_image(mask_2d, rgb)

    row = {}

    # ── Area ──────────────────────────────────────────────────────────────────
    if flags.get("area"):
        row["area_px2"] = float(mask_2d.sum())

    # ── Length / Width ─────────────────────────────────────────────────────────
    if flags.get("length") or flags.get("width"):
        lw = _length_width_after_deskew(mask_2d)
        maj, minw, axis_w, axis_h = _pca_major_minor(mask_2d)
        if flags.get("length"):
            row["length_major_px"]  = round(maj, 2)
            row["length_bbox_px"]   = float(axis_h)
            row["axis_height_px"]   = float(axis_h)
        if flags.get("width"):
            row["width_minor_px"]   = round(minw, 2)
            row["width_row_max_px"] = round(lw["width_px_max"], 2)
            row["width_row_p95_px"] = round(lw["width_px_p95"], 2)
            row["axis_width_px"]    = float(axis_w)

    # ── Shape descriptors ─────────────────────────────────────────────────────
    if flags.get("shape"):
        m8 = mask_2d.astype(np.uint8)
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            perimeter = cv2.arcLength(cnt, True)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            hull_perim = cv2.arcLength(hull, True)
            area = float(mask_2d.sum())
            bbox_area = float(mask_2d.shape[0] * mask_2d.shape[1])
            circ = (4 * math.pi * area / (perimeter ** 2 + 1e-6))
            equiv_d = math.sqrt(4 * area / math.pi) if area > 0 else 0.0
            row.update(
                perimeter_px=round(perimeter, 2),
                hull_area_px2=round(hull_area, 2),
                hull_perimeter_px=round(hull_perim, 2),
                solidity=round(area / (hull_area + 1e-6), 4),
                extent=round(area / (bbox_area + 1e-6), 4),
                circularity=round(circ, 4),
                equiv_diameter_px=round(equiv_d, 2),
            )

    # ── Connected components ──────────────────────────────────────────────────
    if flags.get("comp"):
        num, _ = cv2.connectedComponents(mask_2d.astype(np.uint8), connectivity=8)
        row["components"] = max(0, num - 1)

    # ── RGB color stats ───────────────────────────────────────────────────────
    if flags.get("color"):
        sr, sg, sb = _color_stats(rgb, mask_2d)
        for ch, st in zip(("R", "G", "B"), (sr, sg, sb)):
            for k, v in st.items():
                row[f"{k}_{ch}"] = round(v, 3)

    # ── HSV color stats ───────────────────────────────────────────────────────
    if flags.get("hsv"):
        sh, ss, sv = _color_stats_hsv(rgb, mask_2d)
        for ch, st in zip(("H", "S", "V"), (sh, ss, sv)):
            for k, v in st.items():
                row[f"{k}_{ch}"] = round(v, 3)

    # ── HSV variance ─────────────────────────────────────────────────────────
    if flags.get("hsvvar"):
        sh, ss, _ = _color_stats_hsv(rgb, mask_2d)
        row["var_H"] = round(sh["std"] ** 2, 4)
        row["var_S"] = round(ss["std"] ** 2, 4)

    # ── Vegetation indices ────────────────────────────────────────────────────
    if flags.get("veg"):
        row.update(_vegetation_indices_stats(rgb, mask_2d))

    return row


# =============================================================================
# CSV export helpers
# =============================================================================

def build_individual_rows(
    masks: list,
    rgb: np.ndarray,
    idxs: list,
    flags: dict,
    filename: str = "",
) -> list[dict]:
    """
    Build one CSV row per selected mask (individual phenotypes).
    Returns a list of dicts ready for csv.DictWriter.
    """
    rows = []
    for i in idxs:
        m = masks[i]
        seg = m.get("segmentation")
        if seg is None:
            continue
        phen = compute_phenotypes(rgb, seg, flags)
        phen["FileName"] = filename
        phen["segment_idx"] = i
        rows.append(phen)
    return rows


def build_joint_row(
    masks: list,
    rgb: np.ndarray,
    idxs: list,
    flags: dict,
    filename: str = "",
) -> dict:
    """
    Aggregate phenotypes across selected masks into a single joint row.
    Scalar geometry metrics are summed and also stored as per-segment means.
    """
    n = 0
    agg: dict = {}

    def _add(key, val):
        agg[key] = agg.get(key, 0.0) + float(val)

    for i in idxs:
        m = masks[i]
        seg = m.get("segmentation")
        if seg is None:
            continue
        phen = compute_phenotypes(rgb, seg, flags)
        n += 1
        for k, v in phen.items():
            if isinstance(v, (int, float)):
                _add(k, v)

    row = {"FileName": filename, "n_segments": n}

    scalar_keys = [
        "area_px2", "length_major_px", "length_bbox_px", "axis_height_px",
        "width_minor_px", "width_row_max_px", "width_row_p95_px",
        "axis_width_px", "perimeter_px", "hull_area_px2", "hull_perimeter_px",
        "solidity", "extent", "circularity", "equiv_diameter_px",
        "components", "exg_mean", "exr_mean", "exgr_mean", "gli_mean",
        "green_frac", "var_H", "var_S",
    ]
    for k in scalar_keys:
        if k in agg:
            row[k + "_total"] = round(agg[k], 3)
            row[k + "_mean"] = round(agg[k] / max(1, n), 3)

    # Color / HSV: average the means
    color_keys = [
        "mean_R", "mean_G", "mean_B", "median_R", "median_G", "median_B",
        "sum_R", "sum_G", "sum_B", "std_R", "std_G", "std_B",
        "mean_H", "mean_S", "mean_V", "median_H", "median_S", "median_V",
        "sum_H", "sum_S", "sum_V", "std_H", "std_S", "std_V",
    ]
    for k in color_keys:
        if k in agg:
            row[k + "_mean"] = round(agg[k] / max(1, n), 3)
            row[k + "_total"] = round(agg[k], 3)

    return row


def write_individual_csv(rows: list[dict], out_path: str) -> None:
    """Write a list of per-mask phenotype rows to a CSV file."""
    if not rows:
        return
    all_keys = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_joint_csv(row: dict, out_path: str) -> None:
    """Write a single joint-phenotype row to a CSV file."""
    if not row:
        return
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
