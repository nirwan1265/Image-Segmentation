#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leaf Segmenter – interactive GUI for enhancement + SAM2 auto-masking + selective saving.

Run:  python leaf_segmenter_gui.py

You can:
  1) Open image (TIFF/JPG/PNG)
  2) Choose enhancement path:
       - Green-aware enhancer (CLAHE + bilateral + sobel + unsharp; only on green-ish pixels)
       - OR Classic preprocess (brightness/contrast/gamma + optional Laplacian + unsharp)
     Tweak a few knobs, then Preview.
  3) Load SAM2 model (checkpoint + cfg + device).
  4) Segment → see how many masks.
  5) Select masks in the list → Preview crop for any mask.
  6) Save Selected… → choose folder, saves mask_XXX.png + crop_XXX.png + manifest CSV.

Tested on plain Python + Tkinter. Requires:
  numpy, Pillow, opencv-python, sam2 (installed), torch.
"""

import os
import csv
import json
import math
import threading
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import cv2
from PIL import Image, ImageTk, ImageOps

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# --- SAM2 imports (make sure your repo is installed or PYTHONPATHed) ---
try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    # user’s builder name may vary; adapt to your local function
    from sam2.build_sam import build_sam2
except Exception as e:
    SAM2AutomaticMaskGenerator = None
    build_sam2 = None
    _sam2_import_error = e
else:
    _sam2_import_error = None


# =========================
# Image helpers (RGB uint8)
# =========================

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

    # tighten edge (optional)
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, 2*erode_px+1),)*2)
        crop_msk = cv2.erode(crop_msk, k, iterations=1)

    # feather alpha (optional)
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

    hsv2 = hsv.copy()
    hsv2[..., 2] = v_eq
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

def flatten_background_whiten(
    img_rgb_uint8,
    val_min=200, sat_max=35, morph_open=3, morph_close=5
):
    """Force near-white paper to pure white to suppress halos."""
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

        # state
        self.img_path = None
        self.img = None            # original RGB
        self.img_preview = None    # enhanced preview (for display)
        self.sr: SegResult|None = None
        self.sam2_model = None

        # ---------- Layout ----------
        self.make_model_frame(root)
        self.make_options_frame(root)
        self.make_actions_frame(root)
        self.make_preview_frame(root)
        self.make_masks_frame(root)

        # initial states
        self.update_buttons_state()

    # ---- Frames ----
    def make_model_frame(self, root):
        f = ttk.LabelFrame(root, text="Model")
        f.grid(row=0, column=0, padx=8, pady=6, sticky="ew")

        ttk.Label(f, text="Checkpoint:").grid(row=0, column=0, sticky="w")
        self.e_ckpt = ttk.Entry(f, width=54)
        self.e_ckpt.grid(row=0, column=1, padx=4)
        ttk.Button(f, text="…", command=self.pick_ckpt).grid(row=0, column=2)

        ttk.Label(f, text="Config (name or .yaml):").grid(row=1, column=0, sticky="w")
        self.e_cfg = ttk.Entry(f, width=54)
        self.e_cfg.insert(0, "sam2.1_hiera_l")
        self.e_cfg.grid(row=1, column=1, padx=4)

        ttk.Label(f, text="Device:").grid(row=2, column=0, sticky="w")
        self.e_dev = ttk.Entry(f, width=12)
        self.e_dev.insert(0, "cpu")  # change to "cuda" or "mps" if you want
        self.e_dev.grid(row=2, column=1, sticky="w")
        self.chk_post = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="apply_postprocessing", variable=self.chk_post).grid(row=2, column=1, sticky="e")

        ttk.Button(f, text="Load Model", command=self.load_model).grid(row=0, column=3, rowspan=3, padx=6)

    def make_options_frame(self, root):
        f = ttk.LabelFrame(root, text="Enhancement & Segmentation Options")
        f.grid(row=1, column=0, padx=8, pady=6, sticky="ew")

        # File ops
        ttk.Button(f, text="Open Image…", command=self.open_image).grid(row=0, column=0, padx=2, pady=2, sticky="w")
        self.chk_rotate = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Rotate 90° CCW", variable=self.chk_rotate).grid(row=0, column=1, padx=4)

        # Enhancer choice (radio)
        self.enh_mode = tk.StringVar(value="green")
        ttk.Radiobutton(f, text="Green-aware enhancer", value="green", variable=self.enh_mode).grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(f, text="Classic preprocess", value="classic", variable=self.enh_mode).grid(row=1, column=1, sticky="w")

        # Classic knobs
        self.s_brightness = tk.IntVar(value=-25)
        self.s_contrast = tk.DoubleVar(value=1.0)
        self.s_gamma = tk.DoubleVar(value=1.2)
        self.chk_unsharp = tk.BooleanVar(value=True)
        self.chk_laplacian = tk.BooleanVar(value=False)

        row = 2
        ttk.Label(f, text="Brightness").grid(row=row, column=0, sticky="w")
        ttk.Scale(f, from_=-100, to=100, variable=self.s_brightness, orient="horizontal", length=180).grid(row=row, column=1, sticky="ew")
        row += 1
        ttk.Label(f, text="Contrast").grid(row=row, column=0, sticky="w")
        ttk.Scale(f, from_=0.5, to=2.0, variable=self.s_contrast, orient="horizontal", length=180).grid(row=row, column=1, sticky="ew")
        row += 1
        ttk.Label(f, text="Gamma").grid(row=row, column=0, sticky="w")
        ttk.Scale(f, from_=0.5, to=2.0, variable=self.s_gamma, orient="horizontal", length=180).grid(row=row, column=1, sticky="ew")
        row += 1
        ttk.Checkbutton(f, text="Unsharp", variable=self.chk_unsharp).grid(row=row, column=0, sticky="w")
        ttk.Checkbutton(f, text="Laplacian", variable=self.chk_laplacian).grid(row=row, column=1, sticky="w")
        row += 1

        # Background flatten
        self.chk_whiten = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Whiten Background", variable=self.chk_whiten).grid(row=row, column=0, sticky="w")
        self.s_val_min = tk.IntVar(value=200)
        self.s_sat_max = tk.IntVar(value=35)
        ttk.Label(f, text="val_min").grid(row=row, column=1, sticky="e")
        ttk.Entry(f, width=6, textvariable=self.s_val_min).grid(row=row, column=2, sticky="w", padx=2)
        ttk.Label(f, text="sat_max").grid(row=row, column=3, sticky="e")
        ttk.Entry(f, width=6, textvariable=self.s_sat_max).grid(row=row, column=4, sticky="w", padx=2)
        row += 1

        # Post
        self.s_close_iters = tk.IntVar(value=1)
        self.s_halo_erode = tk.IntVar(value=1)
        self.s_halo_feather = tk.IntVar(value=2)
        ttk.Label(f, text="Close iters").grid(row=row, column=0, sticky="w")
        ttk.Entry(f, width=6, textvariable=self.s_close_iters).grid(row=row, column=1, sticky="w")
        ttk.Label(f, text="Halo erode px").grid(row=row, column=2, sticky="e")
        ttk.Entry(f, width=6, textvariable=self.s_halo_erode).grid(row=row, column=3, sticky="w")
        ttk.Label(f, text="Feather px").grid(row=row, column=4, sticky="e")
        ttk.Entry(f, width=6, textvariable=self.s_halo_feather).grid(row=row, column=5, sticky="w")

        # Buttons
        ttk.Button(f, text="Preview Enhance", command=self.preview_enhance).grid(row=row+1, column=0, pady=4, sticky="w")
        ttk.Button(f, text="Segment", command=self.segment).grid(row=row+1, column=1, pady=4, sticky="w")

    def make_actions_frame(self, root):
        f = ttk.LabelFrame(root, text="Save")
        f.grid(row=2, column=0, padx=8, pady=6, sticky="ew")
        ttk.Button(f, text="Save Selected Masks…", command=self.save_selected).grid(row=0, column=0, padx=4, pady=4, sticky="w")

    def make_preview_frame(self, root):
        f = ttk.LabelFrame(root, text="Preview")
        f.grid(row=0, column=1, rowspan=3, padx=8, pady=6, sticky="nsew")
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(2, weight=1)

        self.canvas = tk.Canvas(f, width=640, height=640, bg="#202020", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

    def make_masks_frame(self, root):
        f = ttk.LabelFrame(root, text="Masks")
        f.grid(row=0, column=2, rowspan=3, padx=8, pady=6, sticky="ns")
        self.lb = tk.Listbox(f, width=38, height=32, selectmode="extended")
        self.lb.pack(side="left", fill="y")
        sb = ttk.Scrollbar(f, orient="vertical", command=self.lb.yview)
        sb.pack(side="right", fill="y")
        self.lb.config(yscrollcommand=sb.set)
        self.lb.bind("<<ListboxSelect>>", self.on_select_mask)

    # ---- File/model ----
    def pick_ckpt(self):
        p = filedialog.askopenfilename(title="Select SAM2 checkpoint (.pt)")
        if p:
            self.e_ckpt.delete(0, tk.END)
            self.e_ckpt.insert(0, p)

    def open_image(self):
        p = filedialog.askopenfilename(title="Open image", filetypes=[("Images","*.tif *.tiff *.png *.jpg *.jpeg")])
        if not p:
            return
        self.img_path = p
        arr = ensure_uint8_rgb(Image.open(p))
        if self.chk_rotate.get():
            arr = rotate_left_90(arr)
        self.img = arr
        self.sr = None
        self.lb.delete(0, tk.END)
        self.show_image(arr)
        self.update_buttons_state()

    def load_model(self):
        if _sam2_import_error is not None:
            messagebox.showerror("SAM2 import error",
                                 f"Couldn't import sam2 modules:\n{_sam2_import_error}")
            return
        ckpt = self.e_ckpt.get().strip()
        cfg  = self.e_cfg.get().strip()
        dev  = self.e_dev.get().strip() or "cpu"
        if not ckpt or not os.path.exists(ckpt):
            messagebox.showerror("Missing checkpoint", "Please pick a valid SAM2 checkpoint (.pt).")
            return
        try:
            self.sam2_model = build_sam2(cfg, ckpt, device=dev, apply_postprocessing=self.chk_post.get())
            messagebox.showinfo("Model", f"Loaded SAM2 on device '{dev}'.")
        except Exception as e:
            messagebox.showerror("Model load failed", str(e))
            self.sam2_model = None
        self.update_buttons_state()

    # ---- UI state ----
    def update_buttons_state(self):
        has_img = self.img is not None
        has_model = self.sam2_model is not None
        # Preview needs image
        # Segment needs image + model
        # Save needs segmentation results and selection
        # handled in method bodies for clarity.

    # ---- Preview ----
    def preview_enhance(self):
        if self.img is None:
            messagebox.showwarning("No image", "Open an image first.")
            return

        arr = self.img.copy()

        # background whiten (on original color)
        if self.chk_whiten.get():
            arr = flatten_background_whiten(arr, val_min=self.s_val_min.get(), sat_max=self.s_sat_max.get())

        if self.enh_mode.get() == "green":
            arr2 = enhance_leaf_edges_rgb(arr)
        else:
            arr2 = preprocess_for_edges(
                arr,
                brightness=self.s_brightness.get(),
                contrast=self.s_contrast.get(),
                use_unsharp=self.chk_unsharp.get(),
                unsharp_kernel_size=9, unsharp_sigma=10.0, unsharp_amount=1.5,
                use_laplacian=self.chk_laplacian.get(),
                gamma=self.s_gamma.get()
            )

        self.img_preview = arr2
        self.show_image(arr2)

    def show_image(self, arr):
        """Fit image to canvas while keeping aspect."""
        H, W = arr.shape[:2]
        canvas_w = int(self.canvas.winfo_width() or 640)
        canvas_h = int(self.canvas.winfo_height() or 640)

        # resize to fit
        scale = min(canvas_w / W, canvas_h / H)
        new_w = max(1, int(W * scale))
        new_h = max(1, int(H * scale))

        im = Image.fromarray(arr)
        im = im.resize((new_w, new_h), Image.BILINEAR)
        # center on canvas
        bg = Image.new("RGB", (canvas_w, canvas_h), (32, 32, 32))
        bg.paste(im, ((canvas_w - new_w) // 2, (canvas_h - new_h) // 2))
        self.tk_img = ImageTk.PhotoImage(bg)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    # ---- Segment ----
    def segment(self):
        if self.img is None:
            messagebox.showwarning("No image", "Open an image first.")
            return
        if self.sam2_model is None:
            messagebox.showwarning("No model", "Load the SAM2 model first.")
            return

        # run in a thread to keep UI alive
        threading.Thread(target=self._segment_worker, daemon=True).start()

    def _segment_worker(self):
        try:
            arr = self.img.copy()
            if self.chk_whiten.get():
                arr = flatten_background_whiten(arr, val_min=self.s_val_min.get(), sat_max=self.s_sat_max.get())

            if self.enh_mode.get() == "green":
                seg_img = enhance_leaf_edges_rgb(arr)
            else:
                seg_img = preprocess_for_edges(
                    arr,
                    brightness=self.s_brightness.get(),
                    contrast=self.s_contrast.get(),
                    use_unsharp=self.chk_unsharp.get(),
                    unsharp_kernel_size=9, unsharp_sigma=10.0, unsharp_amount=1.5,
                    use_laplacian=self.chk_laplacian.get(),
                    gamma=self.s_gamma.get()
                )

            gen = make_mask_generator(self.sam2_model)
            masks = gen.generate(seg_img)
            masks = dedupe_by_mask_iou(masks, iou_thresh=0.80)

            self.sr = SegResult(masks=masks, img_color=self.img.copy(), img_seg=seg_img, rotate_applied=self.chk_rotate.get())

            # populate listbox
            self.lb.delete(0, tk.END)
            for i, m in enumerate(masks):
                self.lb.insert(tk.END, f"[{i:03d}] area={int(m['area'])} bbox={list(map(int, m['bbox']))}")
            self.show_image(seg_img)
            messagebox.showinfo("Segmentation", f"Found {len(masks)} masks.")
        except Exception as e:
            messagebox.showerror("Segmentation error", str(e))

    # ---- Select mask and preview crop ----
    def on_select_mask(self, event=None):
        if not self.sr:
            return
        sel = self.lb.curselection()
        if not sel:
            return
        idx = sel[-1]
        m = self.sr.masks[idx]
        mask_bool = m["segmentation"].astype(bool)
        x, y, w, h = map(int, m["bbox"])
        x2, y2 = x + w, y + h
        crop = self.sr.img_color[y:y2, x:x2, :].copy()
        msk  = mask_bool[y:y2, x:x2]

        # compose preview with transparent bg over gray checker
        alpha = (msk.astype(np.uint8) * 255)[..., None]
        rgba = np.dstack([crop, alpha])
        # make checkerboard background
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
        erode_px = max(0, int(self.s_halo_erode.get()))
        feather_px = max(0, int(self.s_halo_feather.get()))
        close_iters = max(0, int(self.s_close_iters.get()))

        for idx in sel:
            m = self.sr.masks[idx]
            mask_bool = m["segmentation"].astype(bool)

            # optional close
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
                "crop_png": str(crop_path)
            })

        # manifest
        csv_path = out / "mask_manifest.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["mask_idx","area_px","bbox","mask_png","crop_png"])
            writer.writeheader()
            writer.writerows(rows)

        messagebox.showinfo("Saved", f"Saved {len(rows)} items to:\n{out}\n\nManifest: {csv_path}")

# ---- main ----
if __name__ == "__main__":
    root = tk.Tk()
    # nicer defaults
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
