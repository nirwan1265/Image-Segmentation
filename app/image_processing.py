#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image_processing.py
====================
Pure image-processing functions — no tkinter, no GUI state.
All functions take numpy arrays (RGB uint8) in and return numpy arrays out.

Includes:
  - Basic helpers       : ensure_uint8_rgb, rotate_left_90
  - Enhancement         : preprocess_for_edges, enhance_leaf_edges_rgb,
                          flatten_background_whiten
  - Vegetation indices  : compute_vegetation_indices, enhance_with_vegetation_index
  - Advanced enhancers  : denoise_nlm, single_scale_retinex, multi_scale_retinex,
                          morphological_tophat, guided_filter_enhance,
                          enhance_lab_green, white_balance_grayworld,
                          white_balance_max_white, difference_of_gaussians,
                          local_contrast_normalization, adaptive_gamma,
                          shadow_highlight_correction
"""

import numpy as np
import cv2


# =============================================================================
# Basic helpers
# =============================================================================

def ensure_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert any array to RGB uint8 (handles grayscale and RGBA input)."""
    arr = np.array(arr)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def rotate_left_90(img_rgb_uint8: np.ndarray) -> np.ndarray:
    """Rotate image 90° counter-clockwise."""
    return cv2.rotate(img_rgb_uint8, cv2.ROTATE_90_COUNTERCLOCKWISE)


# =============================================================================
# Core enhancement pipeline
# =============================================================================

def preprocess_for_edges(
    img_rgb_uint8: np.ndarray,
    brightness: float = 0,
    contrast: float = 1.0,
    use_unsharp: bool = True,
    unsharp_kernel_size: int = 9,
    unsharp_sigma: float = 10.0,
    unsharp_amount: float = 1.5,
    use_laplacian: bool = False,
    gamma: float = None,
) -> np.ndarray:
    """
    Brighten/contrast → unsharp mask → optional Laplacian → optional gamma.
    Returns RGB uint8.
    """
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
        tab = np.clip(
            (np.arange(256) / 255.0) ** (1.0 / gamma) * 255.0, 0, 255
        ).astype(np.uint8)
        x = cv2.LUT(x, tab)

    return np.ascontiguousarray(x)


def enhance_leaf_edges_rgb(
    img_rgb_uint8: np.ndarray,
    hsv_h_low: int = 25,
    hsv_h_high: int = 95,
    hsv_s_min: int = 40,
    hsv_v_min: int = 40,
    clahe_clip: float = 2.0,
    clahe_tiles: int = 8,
    bilateral_d: int = 7,
    bilateral_sigma: float = 50,
    unsharp_amount: float = 1.5,
    unsharp_sigma: float = 10,
    unsharp_ksize: int = 9,
    sobel_blend: float = 0.12,
) -> np.ndarray:
    """
    Plant-specific edge enhancement pipeline:
      HSV green-channel CLAHE → bilateral filter → Sobel edge blend → unsharp.
    """
    x = img_rgb_uint8.astype(np.uint8)

    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    green = (
        (h >= hsv_h_low) & (h <= hsv_h_high) &
        (s >= hsv_s_min) & (v >= hsv_v_min)
    )
    green = cv2.morphologyEx(
        green.astype(np.uint8), cv2.MORPH_OPEN,
        np.ones((5, 5), np.uint8), iterations=1
    ).astype(bool)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                             tileGridSize=(clahe_tiles, clahe_tiles))
    v_clahe = clahe.apply(v)
    v_eq = v.copy()
    v_eq[green] = v_clahe[green] if green.any() else v_eq[green]
    if not green.any():
        v_eq = v_clahe

    hsv2 = hsv.copy()
    hsv2[..., 2] = v_eq
    rgb_eq = cv2.cvtColor(hsv2, cv2.COLOR_HSV2RGB)

    rgb_bi = cv2.bilateralFilter(
        rgb_eq, d=int(bilateral_d),
        sigmaColor=bilateral_sigma, sigmaSpace=bilateral_sigma
    )

    gray = cv2.cvtColor(rgb_bi, cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel = cv2.normalize(
        cv2.magnitude(sx, sy), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    hsv3 = cv2.cvtColor(rgb_bi, cv2.COLOR_RGB2HSV)
    hsv3[..., 2] = np.clip(
        hsv3[..., 2].astype(np.float32) + sobel_blend * sobel, 0, 255
    ).astype(np.uint8)
    rgb_edge = cv2.cvtColor(hsv3, cv2.COLOR_HSV2RGB)

    blur = cv2.GaussianBlur(
        rgb_edge, (int(unsharp_ksize), int(unsharp_ksize)), unsharp_sigma
    )
    sharp = cv2.addWeighted(rgb_edge, unsharp_amount, blur,
                             -(unsharp_amount - 1.0), 0)

    return np.ascontiguousarray(sharp)


def flatten_background_whiten(
    img_rgb_uint8: np.ndarray,
    val_min: int = 200,
    sat_max: int = 35,
    morph_open: int = 3,
    morph_close: int = 5,
) -> np.ndarray:
    """
    Detect bright/desaturated background pixels in HSV and paint them white.
    Useful for images on white trays/paper.
    """
    hsv = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bg = (v >= val_min) & (s <= sat_max)
    if morph_open > 0:
        k = np.ones((morph_open, morph_open), np.uint8)
        bg = cv2.morphologyEx(bg.astype(np.uint8), cv2.MORPH_OPEN, k,
                              iterations=1).astype(bool)
    if morph_close > 0:
        k = np.ones((morph_close, morph_close), np.uint8)
        bg = cv2.morphologyEx(bg.astype(np.uint8), cv2.MORPH_CLOSE, k,
                              iterations=1).astype(bool)
    out = img_rgb_uint8.copy()
    out[bg] = 255
    return out


# =============================================================================
# Vegetation indices
# =============================================================================

def compute_vegetation_indices(rgb: np.ndarray) -> dict:
    """
    Compute plant-specific vegetation indices from an RGB image.

    Returns a dict of normalized uint8 single-channel images:
      'ExG'  – Excess Green Index          (highlights green vegetation)
      'GRVI' – Green-Red Vegetation Index
      'VARI' – Visible Atmospherically Resistant Index
      'TGI'  – Triangular Greenness Index
      'GLI'  – Green Leaf Index
    """
    R = rgb[..., 0].astype(np.float32)
    G = rgb[..., 1].astype(np.float32)
    B = rgb[..., 2].astype(np.float32)

    r, g, b = R / 255.0, G / 255.0, B / 255.0

    ExG  = 2 * g - r - b
    GRVI = (g - r) / (g + r + 1e-6)
    VARI = (g - r) / (g + r - b + 1e-6)
    TGI  = g - 0.39 * r - 0.61 * b
    GLI  = (2 * g - r - b) / (2 * g + r + b + 1e-6)

    def _norm(arr):
        arr = np.clip(arr, np.percentile(arr, 1), np.percentile(arr, 99))
        return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return {
        'ExG': _norm(ExG),
        'GRVI': _norm(GRVI),
        'VARI': _norm(VARI),
        'TGI': _norm(TGI),
        'GLI': _norm(GLI),
    }


def enhance_with_vegetation_index(
    rgb: np.ndarray,
    index_type: str = 'ExG',
    blend: float = 0.3,
) -> np.ndarray:
    """
    Blend original image with a vegetation-index overlay.
    index_type: 'ExG' | 'GRVI' | 'VARI' | 'TGI' | 'GLI'
    blend: 0.0–1.0 (how strongly to blend the index in)
    """
    indices = compute_vegetation_indices(rgb)
    idx_img = indices.get(index_type, indices['ExG'])
    idx_rgb = cv2.cvtColor(idx_img, cv2.COLOR_GRAY2RGB)
    enhanced = cv2.addWeighted(rgb, 1.0 - blend, idx_rgb, blend, 0)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


# =============================================================================
# Advanced enhancement functions
# =============================================================================

def denoise_nlm(
    rgb: np.ndarray,
    h: float = 10,
    template_size: int = 7,
    search_size: int = 21,
) -> np.ndarray:
    """
    Non-local means denoising — preserves edges better than median/mean.
    h: filter strength (higher = more denoising)
    """
    template_size = template_size if template_size % 2 == 1 else template_size + 1
    search_size = search_size if search_size % 2 == 1 else search_size + 1
    return cv2.fastNlMeansDenoisingColored(
        rgb, None, h, h, template_size, search_size
    )


def single_scale_retinex(rgb: np.ndarray, sigma: float = 80) -> np.ndarray:
    """
    Single-Scale Retinex — removes illumination effects and enhances detail.
    sigma: Gaussian blur sigma (higher = more illumination removal)
    """
    rgb_f = rgb.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(rgb_f, (0, 0), sigma)
    retinex = np.log10(rgb_f) - np.log10(blur + 1.0)

    result = np.zeros_like(rgb, dtype=np.uint8)
    for c in range(3):
        result[..., c] = cv2.normalize(
            retinex[..., c], None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
    return result


def multi_scale_retinex(
    rgb: np.ndarray,
    sigmas: tuple = (15, 80, 250),
    weights: list = None,
) -> np.ndarray:
    """
    Multi-Scale Retinex with Color Restoration (MSRCR).
    Combines multiple scales for better illumination correction.
    """
    if weights is None:
        weights = [1.0 / len(sigmas)] * len(sigmas)

    rgb_f = rgb.astype(np.float32) + 1.0
    log_rgb = np.log10(rgb_f)

    msr = np.zeros_like(rgb_f)
    for sigma, w in zip(sigmas, weights):
        blur = cv2.GaussianBlur(rgb_f, (0, 0), sigma)
        msr += w * (log_rgb - np.log10(blur + 1.0))

    intensity = np.mean(rgb_f, axis=2, keepdims=True)
    color_restoration = np.log10(125.0 * rgb_f / (intensity + 1.0) + 1.0)
    msr = msr * color_restoration

    result = np.zeros_like(rgb, dtype=np.uint8)
    for c in range(3):
        result[..., c] = cv2.normalize(
            msr[..., c], None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
    return result


def morphological_tophat(rgb: np.ndarray, kernel_size: int = 50) -> np.ndarray:
    """
    Top-hat + Black-hat transform for illumination normalization.
    Removes uneven background illumination.
    kernel_size: size of structuring element (larger = removes larger variations)
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    result = np.zeros_like(rgb)
    for c in range(3):
        ch = rgb[..., c]
        tophat   = cv2.morphologyEx(ch, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(ch, cv2.MORPH_BLACKHAT, kernel)
        result[..., c] = cv2.subtract(cv2.add(ch, tophat), blackhat)
    return result


def guided_filter_enhance(
    rgb: np.ndarray,
    radius: int = 8,
    eps: float = 0.04,
) -> np.ndarray:
    """
    Edge-preserving smoothing using guided filter (falls back to bilateral).
    Uses the green channel as guide — best for plants.
    """
    try:
        rgb_f = rgb.astype(np.float32) / 255.0
        guide = rgb_f[..., 1]
        filtered = np.zeros_like(rgb_f)
        for c in range(3):
            filtered[..., c] = cv2.ximgproc.guidedFilter(
                guide, rgb_f[..., c], radius, eps
            )
        return (filtered * 255).clip(0, 255).astype(np.uint8)
    except AttributeError:
        return cv2.bilateralFilter(rgb, radius, eps * 1000, eps * 1000)


def enhance_lab_green(
    rgb: np.ndarray,
    l_factor: float = 1.0,
    a_shift: float = -10,
    b_shift: float = 0,
) -> np.ndarray:
    """
    Enhance in LAB color space.
    The 'a' channel is the red-green axis; negative a_shift boosts green.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] = np.clip(lab[..., 0] * l_factor, 0, 255)
    lab[..., 1] = np.clip(lab[..., 1] + a_shift, 0, 255)
    lab[..., 2] = np.clip(lab[..., 2] + b_shift, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def white_balance_grayworld(rgb: np.ndarray) -> np.ndarray:
    """
    Gray World white balance — assumes average color should be neutral gray.
    Good for standardizing colours across lighting conditions.
    """
    result = rgb.astype(np.float32)
    avg_r = np.mean(result[..., 0])
    avg_g = np.mean(result[..., 1])
    avg_b = np.mean(result[..., 2])
    avg_all = (avg_r + avg_g + avg_b) / 3.0
    if avg_r > 0:
        result[..., 0] *= avg_all / avg_r
    if avg_g > 0:
        result[..., 1] *= avg_all / avg_g
    if avg_b > 0:
        result[..., 2] *= avg_all / avg_b
    return np.clip(result, 0, 255).astype(np.uint8)


def white_balance_max_white(rgb: np.ndarray, percentile: float = 99) -> np.ndarray:
    """
    Max-White white balance — treats the brightest pixels as white.
    percentile: use this percentile as 'white' (99 avoids outliers)
    """
    result = rgb.astype(np.float32)
    for c in range(3):
        max_val = np.percentile(result[..., c], percentile)
        if max_val > 0:
            result[..., c] = result[..., c] * (255.0 / max_val)
    return np.clip(result, 0, 255).astype(np.uint8)


def difference_of_gaussians(
    rgb: np.ndarray,
    sigma1: float = 1.0,
    sigma2: float = 3.0,
    blend: float = 0.3,
) -> np.ndarray:
    """
    Difference of Gaussians — enhances edges (similar to biological vision).
    sigma1: fine detail, sigma2: coarser context
    """
    g1 = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), sigma1)
    g2 = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), sigma2)
    dog = cv2.normalize(g1 - g2, None, -128, 128, cv2.NORM_MINMAX)
    enhanced = rgb.astype(np.float32) + blend * dog
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def local_contrast_normalization(
    rgb: np.ndarray, kernel_size: int = 31
) -> np.ndarray:
    """
    Local contrast normalization — divides by local standard deviation
    to enhance fine detail across uneven backgrounds.
    """
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    result = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        ch = rgb[..., c].astype(np.float32)
        local_mean = cv2.GaussianBlur(ch, (kernel_size, kernel_size), 0)
        local_sq_mean = cv2.GaussianBlur(ch ** 2, (kernel_size, kernel_size), 0)
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0) + 1e-6)
        normalized = (ch - local_mean) / local_std
        result[..., c] = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)


def adaptive_gamma(rgb: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Adaptive gamma correction based on mean luminance.
    Dark images get gamma < 1 (brightened); bright images get gamma > 1.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[..., 0].astype(np.float32) / 255.0
    mean_l = np.mean(l_channel)
    gamma = (0.5 + mean_l) if mean_l < 0.5 else (mean_l + 0.5)
    l_corrected = np.power(l_channel, 1.0 / gamma)
    lab[..., 0] = (l_corrected * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def shadow_highlight_correction(
    rgb: np.ndarray,
    shadow_amount: float = 0.3,
    highlight_amount: float = 0.3,
) -> np.ndarray:
    """
    Lift shadows and recover highlights separately in LAB luminance.
    shadow_amount: 0–1 (how much to lift dark areas)
    highlight_amount: 0–1 (how much to pull back bright areas)
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l = lab[..., 0].astype(np.float32) / 255.0
    shadow_mask = np.power(1.0 - l, 2)
    highlight_mask = np.power(l, 2)
    l_corrected = l + shadow_amount * shadow_mask * (1.0 - l)
    l_corrected = l_corrected - highlight_amount * highlight_mask * l_corrected
    lab[..., 0] = (l_corrected * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
