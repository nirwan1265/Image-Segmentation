#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mask_utils.py
=============
Pure mask-math utilities — no tkinter, no GUI state.
All functions take/return numpy arrays or plain Python dicts.

Includes:
  - Normalisation    : _ensure_mask_2d, _resize_mask_to_image
  - IO               : save_binary_mask, save_masked_crop_rgba
  - Comparison       : mask_iou, dedupe_by_mask_iou
  - Splitting        : split_masks_by_cc
  - Geometry helpers : _mask_to_contour_pts, _pca_orientation_full,
                       _largest_component_bool, _convex_hull_fill
  - Shape completion : _rosette_circle_extend, _rosette_hull_wedge_extend,
                       _rosette_ellipse_scale_extend, _tapered_extension,
                       predict_extend_mask  ← main public entry point
"""

import numpy as np
import cv2
from pathlib import Path


# =============================================================================
# Normalisation helpers
# =============================================================================

def _ensure_mask_2d(mask_bool: np.ndarray) -> np.ndarray | None:
    """
    Normalise mask to a 2-D boolean array.
    Handles shapes: HxW, HxWx1, HxWx3, 1xHxW, 1x1xHxW.
    """
    if mask_bool is None:
        return None
    m = np.asarray(mask_bool)
    if m.ndim == 3:
        if m.shape[0] == 1:
            m = m[0]
        elif m.shape[2] == 1:
            m = m[..., 0]
        else:
            m = np.max(m, axis=2)
    elif m.ndim == 4:
        m = m.squeeze()
    return (m > 0)


def _resize_mask_to_image(
    mask_bool: np.ndarray, img: np.ndarray
) -> np.ndarray | None:
    """Resize mask to match image (H, W) if they differ."""
    if mask_bool is None:
        return None
    img_H, img_W = img.shape[:2]
    mask_H, mask_W = mask_bool.shape[:2]
    if mask_H != img_H or mask_W != img_W:
        mask_bool = cv2.resize(
            mask_bool.astype(np.uint8), (img_W, img_H),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    return mask_bool


# =============================================================================
# Mask I/O
# =============================================================================

def save_binary_mask(mask_bool: np.ndarray, out_path) -> None:
    """Save a boolean mask as a white-on-black PNG."""
    cv2.imwrite(str(out_path), (mask_bool.astype(np.uint8) * 255))


def save_masked_crop_rgba(
    image_rgb_uint8: np.ndarray,
    mask_bool: np.ndarray,
    bbox,
    out_path,
    erode_px: int = 0,
    feather_px: int = 0,
) -> None:
    """
    Save a transparent-background crop of the masked region as RGBA PNG.
    erode_px  : erode mask edge before saving (removes fringe artefacts)
    feather_px: soft-edge distance-transform alpha (0 = hard edge)
    """
    x, y, w, h = map(int, bbox)
    x2, y2 = x + w, y + h
    x, y = max(0, x), max(0, y)

    crop_img = image_rgb_uint8[y:y2, x:x2, :]
    crop_msk = mask_bool[y:y2, x:x2].astype(np.uint8)

    if erode_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (max(1, 2 * erode_px + 1),) * 2
        )
        crop_msk = cv2.erode(crop_msk, k, iterations=1)

    if feather_px > 0:
        dist = cv2.distanceTransform(
            (crop_msk > 0).astype(np.uint8), cv2.DIST_L2, 3
        )
        alpha = np.clip(dist / float(feather_px), 0, 1) * 255.0
        alpha = alpha.astype(np.uint8)
    else:
        alpha = crop_msk * 255

    rgba = np.dstack([crop_img, alpha])
    cv2.imwrite(str(out_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))


# =============================================================================
# Comparison / deduplication
# =============================================================================

def mask_iou(a_bool: np.ndarray, b_bool: np.ndarray) -> float:
    """Intersection-over-Union for two boolean masks."""
    inter = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum() + 1e-6
    return inter / union


def dedupe_by_mask_iou(masks: list, iou_thresh: float = 0.80) -> list:
    """
    Remove duplicate masks (keeps the largest by area).
    Two masks are considered duplicates if their IoU > iou_thresh.
    """
    kept = []
    for m in sorted(masks, key=lambda z: z["area"], reverse=True):
        seg = m["segmentation"].astype(bool)
        if any(
            mask_iou(seg, k["segmentation"].astype(bool)) > iou_thresh
            for k in kept
        ):
            continue
        kept.append(m)
    return kept


# =============================================================================
# Connected-component splitting
# =============================================================================

def split_masks_by_cc(
    masks: list,
    min_area: int = 50,
    max_components: int = None,
) -> list:
    """
    Split multi-blob masks into one mask per connected component.
    Components smaller than min_area are discarded.
    If max_components is set, only the largest N components are kept.
    """
    out = []
    for m in masks:
        seg = m.get("segmentation")
        if not isinstance(seg, np.ndarray):
            out.append(m)
            continue
        seg_u8 = (seg > 0).astype(np.uint8)
        if seg_u8.ndim != 2:
            out.append(m)
            continue

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            seg_u8, connectivity=8
        )
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
            h_s = int(stats[label, cv2.CC_STAT_HEIGHT])
            comp = dict(m)
            comp["segmentation"] = (labels == label).astype(np.uint8)
            comp["bbox"] = [x, y, w, h_s]
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
                comps = sorted(
                    comps, key=lambda z: z.get("area", 0), reverse=True
                )[: int(max_components)]
            out.extend(comps)
    return out


# =============================================================================
# Geometry helpers (internal)
# =============================================================================

def _mask_to_contour_pts(mask_bool: np.ndarray) -> np.ndarray | None:
    """Return the largest contour of the mask as an Nx2 int32 array."""
    m8 = (mask_bool.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    pts = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.int32)
    return np.ascontiguousarray(pts)


def _largest_component_bool(mask_bool: np.ndarray) -> np.ndarray | None:
    """Return the largest 4-connected component as a boolean mask, or None."""
    m = (mask_bool.astype(np.uint8) > 0).astype(np.uint8)
    num, lab = cv2.connectedComponents(m, connectivity=4)
    if num <= 1:
        return None
    counts = np.bincount(lab.ravel())
    lbl = int(np.argmax(counts[1:]) + 1)
    return lab == lbl


def _convex_hull_fill(mask_bool: np.ndarray) -> np.ndarray:
    """Return a boolean mask of the convex hull of the input mask."""
    pts = _mask_to_contour_pts(mask_bool)
    if pts is None or pts.shape[0] < 3:
        return mask_bool
    hull = np.ascontiguousarray(
        cv2.convexHull(pts).reshape(-1, 2).astype(np.int32)
    )
    H, W = mask_bool.shape[:2]
    hull_mask = np.ascontiguousarray(np.zeros((H, W), dtype=np.uint8))
    cv2.fillConvexPoly(hull_mask, hull, 1)
    return hull_mask.astype(bool)


def _pca_orientation_full(mask_bool: np.ndarray):
    """
    PCA decomposition of mask pixel coordinates.
    Returns (mu, vmaj, vperp, length, width, X, proj) or None if too few pixels.
    """
    ys, xs = np.nonzero(mask_bool)
    if xs.size < 10:
        return None
    X = np.column_stack([xs, ys]).astype(np.float32)
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(1, len(Xc) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    vmaj = eigvecs[:, order[0]]
    vmaj = vmaj / (np.linalg.norm(vmaj) + 1e-8)
    vperp = np.array([-vmaj[1], vmaj[0]], dtype=np.float32)
    proj = Xc @ vmaj
    length = float(proj.max() - proj.min())
    width = float((Xc @ vperp).max() - (Xc @ vperp).min())
    return mu, vmaj, vperp, length, width, X, proj


# =============================================================================
# Shape-completion strategies (internal)
# =============================================================================

def _rosette_circle_extend(mask_bool: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Arabidopsis rosette completion: fit a minimum enclosing circle, grow it slightly.
    strength ~ 0.7–1.5  (1.0 = gentle ~10% grow)
    """
    pts = _mask_to_contour_pts(mask_bool)
    if pts is None or pts.shape[0] < 3:
        return mask_bool
    (cx, cy), r = cv2.minEnclosingCircle(pts.astype(np.float32))
    r2 = float(r) * (1.10 + 0.20 * (float(strength) - 1.0))
    H, W = mask_bool.shape
    out = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(r2)), 1,
               thickness=-1)
    return np.logical_or(mask_bool, out.astype(bool))


def _rosette_hull_wedge_extend(
    mask_bool: np.ndarray, strength: float = 1.0
) -> np.ndarray:
    """
    Fill only the largest convex-hull 'wedge' (hull minus mask).
    Good for partially-occluded rosette leaves.
    """
    hull_bool = _convex_hull_fill(mask_bool)
    added = np.logical_and(hull_bool, ~mask_bool)
    if not added.any():
        return mask_bool
    wedge = _largest_component_bool(added)
    if wedge is None or not wedge.any():
        return mask_bool
    if strength > 1.01:
        k = max(1, int(round(2 * (strength - 1.0))))
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        wedge = cv2.morphologyEx(
            wedge.astype(np.uint8), cv2.MORPH_DILATE, se
        ).astype(bool)
    return np.logical_or(mask_bool, wedge)


def _rosette_ellipse_scale_extend(
    mask_bool: np.ndarray, scale: float = 1.12
) -> np.ndarray:
    """Fit an ellipse to the mask contour and scale it up slightly."""
    pts = _mask_to_contour_pts(mask_bool)
    if pts is None or pts.shape[0] < 5:
        return mask_bool
    try:
        (cx, cy), (MA, ma), ang = cv2.fitEllipse(pts.astype(np.float32))
    except cv2.error:
        return mask_bool
    MA2, ma2 = max(3, MA * scale), max(3, ma * scale)
    H, W = mask_bool.shape
    out = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(
        out,
        (int(round(cx)), int(round(cy))),
        (int(round(MA2 / 2)), int(round(ma2 / 2))),
        ang, 0, 360, 1, thickness=-1,
    )
    return np.logical_or(mask_bool, out.astype(bool))


def _tapered_extension(mask_bool: np.ndarray, k_extend: float = 0.6) -> np.ndarray:
    """
    Extend a blade-shaped leaf along its major PCA axis with a triangular taper.
    k_extend: fraction of current length to extend by.
    """
    H, W = mask_bool.shape
    info = _pca_orientation_full(mask_bool)
    if info is None:
        return mask_bool
    mu, vmaj, vperp, length, width, X, proj = info

    tip_is_plus = proj.mean() > 0
    direction = vmaj if tip_is_plus else -vmaj
    extend_len = max(8.0, k_extend * max(20.0, length))
    front_max = proj.max() if tip_is_plus else -proj.min()
    base_center = mu + direction * front_max
    half_base = 0.5 * max(4.0, 0.5 * width)

    tip = mu + direction * (front_max + extend_len)
    p1 = base_center + vperp * half_base
    p2 = base_center - vperp * half_base

    poly = np.stack([p1, p2, tip]).astype(np.int32)
    ext = np.zeros_like(mask_bool, dtype=np.uint8)
    cv2.fillConvexPoly(ext, poly, 1)
    ext = cv2.morphologyEx(
        ext, cv2.MORPH_DILATE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )
    return mask_bool | (ext > 0)


# =============================================================================
# Public entry point for shape completion
# =============================================================================

def predict_extend_mask(
    mask_bool: np.ndarray,
    method: str = "auto",
    strength: float = 1.0,
    forbid_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """
    Non-ML mask completion/extension.

    method:
      'auto'    – choose based on PCA aspect ratio (blade if ratio ≥ 2.2)
      'rosette' – hull-wedge fill, fallback to circle grow
      'blade'   – tapered triangular extension along major axis

    strength: 0.7–1.5 (controls how aggressively to extend)
    forbid_mask: optional boolean mask of pixels that must not be added
                 (e.g. pixels already belonging to another leaf)

    Returns a new boolean mask of the same shape.
    """
    if mask_bool is None:
        return None
    base = mask_bool.astype(bool)
    mode = (method or "auto").lower().strip()

    if mode == "auto":
        from phenotyping import _pca_major_minor
        maj, minw, *_ = _pca_major_minor(base)
        ratio = (maj / (minw + 1e-6)) if minw >= 0 else 0.0
        mode = "blade" if ratio >= 2.2 else "rosette"

    if mode == "blade":
        pred = _tapered_extension(base, k_extend=0.6 * float(strength))
    else:
        pred = _rosette_hull_wedge_extend(base, strength=float(strength))
        if pred is None or np.array_equal(pred, base):
            pred = _rosette_circle_extend(base, strength=float(strength))

    pred = pred.astype(bool)
    if forbid_mask is not None:
        pred = np.logical_and(pred, ~forbid_mask.astype(bool))
        pred = np.logical_or(base, pred)
    return pred
