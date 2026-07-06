#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gui_app.py
==========
LeafSegmenterGUI — complete application class.

Pure-function work is delegated to:
  app_visuals.py      ← colours, styles, ToolTip, AnimatedSpinner
  image_processing.py ← image enhancement (no GUI)
  mask_utils.py       ← mask math / shape completion (no GUI)
  phenotyping.py      ← measurements and CSV export (no GUI)
  sam2_utils.py       ← SAM2 model loading helpers

Run:
    python plant_segmenter.py
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os
import sys
import csv
import json
import math
import re
import shlex
import shutil
import signal
import tempfile
import threading
import traceback
import logging
import time
import colorsys
from dataclasses import dataclass
from pathlib import Path

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import cv2
import torch
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ── Hydra / OmegaConf ─────────────────────────────────────────────────────────
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
try:
    from omegaconf import DictConfig
except Exception:
    class DictConfig:
        pass

# ── SAM2 (optional) ───────────────────────────────────────────────────────────
try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    _sam2_import_error = None
except Exception as _e:
    SAM2AutomaticMaskGenerator = None
    build_sam2 = None
    _sam2_import_error = _e

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception:
    SAM2ImagePredictor = None

# ── local modules ─────────────────────────────────────────────────────────────
from app_visuals import COLORS, ICONS, apply_theme, ToolTip, AnimatedSpinner, status_color
from image_processing import (
    ensure_uint8_rgb, rotate_left_90,
    preprocess_for_edges, enhance_leaf_edges_rgb, flatten_background_whiten,
    compute_vegetation_indices, enhance_with_vegetation_index,
    denoise_nlm, single_scale_retinex, multi_scale_retinex,
    morphological_tophat, guided_filter_enhance, enhance_lab_green,
    white_balance_grayworld, white_balance_max_white,
    difference_of_gaussians, local_contrast_normalization,
    adaptive_gamma, shadow_highlight_correction,
)
from mask_utils import (
    _ensure_mask_2d, _resize_mask_to_image,
    save_binary_mask, save_masked_crop_rgba,
    mask_iou, dedupe_by_mask_iou, split_masks_by_cc,
    predict_extend_mask,
)
from phenotyping import (
    _color_stats, _color_stats_hsv,
    _pca_angle_deg, _pca_major_minor, _length_width_after_deskew,
    compute_phenotypes, build_individual_rows, build_joint_row,
    write_individual_csv, write_joint_csv,
)
from sam2_utils import (
    _hydra_reinit_to_dir, _compose_from_yaml, _resolve_sam2_cfg,
    make_mask_generator,
)

import tab_train
import tab_leaf_completion
import tab_leaf_unfolding
import color_filter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ── Shared data structures ─────────────────────────────────────────────────────
@dataclass
class SegResult:
    masks: list
    img_color: np.ndarray
    img_seg: np.ndarray
    rotate_applied: bool

# ── GUI error helpers ──────────────────────────────────────────────────────────
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

class LeafSegmenterGUI:
    def __init__(self, root):
        self.root = root
        root.title("🌿 Leaf Segmenter — SAM2 Plant Phenotyping Tool")

        # ── Colors + icons come from app_visuals.py ───────────────────────────
        self.colors = COLORS
        self.icons  = ICONS

        # Apply the full theme (all ttk styles) in one call
        apply_theme(root)
        root.configure(bg=COLORS['bg_dark'])

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

        # auto-preview enhancement
        self.auto_preview = tk.BooleanVar(value=True)
        self._auto_preview_job = None
        self._auto_preview_delay_ms = 200

        # save-mask sizing options
        self.save_mask_full = tk.BooleanVar(value=True)
        self.save_mask_crop = tk.BooleanVar(value=False)

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

        # Configure main_top grid (left + right) - approximately 35% / 65% split
        main_top.grid_columnconfigure(0, weight=35, minsize=480)  # left panel ~35%
        main_top.grid_columnconfigure(1, weight=65)  # right panel ~65%
        main_top.grid_rowconfigure(0, weight=1)

        # ═══════════════════════════════════════════════════════════════════════
        # LEFT PANEL: Scrollable container for all controls
        # ═══════════════════════════════════════════════════════════════════════
        left_container = ttk.Frame(main_top)
        left_container.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=6)

        # Canvas + Scrollbar for scrolling
        self._left_canvas = tk.Canvas(left_container, width=480, highlightthickness=0,
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

        # ── Global scroll router ─────────────────────────────────────────────
        # One bind_all handler catches every scroll event in the app and
        # routes it to the nearest *content-scrollable* ancestor of the
        # widget under the mouse.  "Content-scrollable" means the widget
        # has a yscrollcommand set — this excludes the image-preview canvas
        # (which has no scrollbar) while including the left-panel canvas,
        # training-tab canvases, and the masks listbox.

        def _delta_to_units(e):
            if e.delta:
                return int(-1 * (e.delta / 120)) if abs(e.delta) > 10 else -e.delta
            return 0

        def _has_yscroll(w):
            try:
                cls = w.winfo_class()
                if cls == "Listbox":
                    return True
                if cls in ("Canvas", "Text"):
                    return bool(w.configure("yscrollcommand")[4])
            except Exception:
                pass
            return False

        def _find_scrollable(widget):
            try:
                w = widget
                while w:
                    if _has_yscroll(w):
                        return w
                    pname = w.winfo_parent()
                    if not pname:
                        break
                    w = w.nametowidget(pname)
            except Exception:
                pass
            return None

        def _global_scroll(e):
            units = _delta_to_units(e)
            if units == 0:
                return
            target = _find_scrollable(e.widget)
            if target:
                target.yview_scroll(units, "units")
                return "break"

        def _global_scroll_linux(e, direction):
            target = _find_scrollable(e.widget)
            if target:
                target.yview_scroll(direction, "units")
                return "break"

        root.bind_all("<MouseWheel>", _global_scroll)
        root.bind_all("<Button-4>",   lambda e: _global_scroll_linux(e, -1))
        root.bind_all("<Button-5>",   lambda e: _global_scroll_linux(e,  1))

        # no-op — kept so later self._bind_left_scroll() calls don't crash
        self._bind_left_scroll = lambda: None

        # ── Undo stack ────────────────────────────────────────────────────────
        # Each entry is a deep-copy snapshot of self.sr.masks taken just
        # before a destructive operation. Max 50 states kept.
        self._undo_stack: list = []
        self._undo_max: int = 50

        # ═══════════════════════════════════════════════════════════════════════
        # RIGHT PANEL: scrollable canvas + vertical PanedWindow with 3 panes:
        #   1. Preview   (draggable)
        #   2. Masks     (draggable)
        #   3. Color Filter (draggable)
        # A scrollbar on the right lets you reach all three even at small sizes.
        # ═══════════════════════════════════════════════════════════════════════

        # Container holds [canvas | scrollbar] side by side
        _rc = tk.Frame(main_top, bg=self.colors["bg_dark"])
        _rc.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=6)
        _rc.grid_rowconfigure(0, weight=1)
        _rc.grid_columnconfigure(0, weight=1)
        _rc.grid_columnconfigure(1, weight=0)

        # The canvas is what scrolls
        self._right_scroll_canvas = tk.Canvas(
            _rc, highlightthickness=0, bg=self.colors["bg_dark"])
        self._right_scrollbar = ttk.Scrollbar(
            _rc, orient="vertical",
            command=self._right_scroll_canvas.yview)
        self._right_scroll_canvas.configure(
            yscrollcommand=self._right_scrollbar.set)
        self._right_scroll_canvas.grid(row=0, column=0, sticky="nsew")
        self._right_scrollbar.grid(row=0, column=1, sticky="ns")

        # Inner frame inside the canvas — the PanedWindow lives here
        self._right_inner = ttk.Frame(self._right_scroll_canvas)
        self._right_win_id = self._right_scroll_canvas.create_window(
            (0, 0), window=self._right_inner, anchor="nw")

        # Keep inner frame width = canvas width
        def _rc_resize(e):
            self._right_scroll_canvas.itemconfig(
                self._right_win_id, width=e.width)
        self._right_scroll_canvas.bind("<Configure>", _rc_resize)

        # Update scroll region whenever inner frame changes size
        def _rc_inner_configure(e):
            self._right_scroll_canvas.configure(
                scrollregion=self._right_scroll_canvas.bbox("all"))
        self._right_inner.bind("<Configure>", _rc_inner_configure)

        # PanedWindow with 3 draggable panes
        self.right_panel = ttk.PanedWindow(self._right_inner, orient="vertical")
        self.right_panel.pack(fill="both", expand=True)

        # Model type selection variables (must be defined before make_model_frame)
        self.model_type_var = tk.StringVar(value="sam2")
        self.tip_model_path_var = tk.StringVar(value="")

        # Build the frames
        self.make_model_frame(self.left_panel)
        self._on_model_type_change()
        self.make_options_frame(self.left_panel)
        self.make_preview_frame(self.right_panel)
        self.make_masks_frame(self.right_panel)

        # ── Color Filter — third pane (draggable like the others) ─────────────
        self._cf_pane = ttk.LabelFrame(
            self.right_panel,
            text="  🎨 Color Filter  ",
            padding=(6, 4))
        self.right_panel.add(self._cf_pane, weight=1)
        color_filter.attach(self, self._cf_pane)

        # Give each pane a sensible default height after first layout
        def _set_initial_sash(*_):
            try:
                total = self._right_scroll_canvas.winfo_height()
                if total < 100:
                    total = 700
                # Preview gets ~50%, Masks ~30%, Color Filter ~20%
                self.right_panel.sashpos(0, int(total * 0.50))
                self.right_panel.sashpos(1, int(total * 0.80))
            except Exception:
                pass
        self.root.after(200, _set_initial_sash)

        # Bind scrolling to all left panel children (must be done after frames are built)
        self._bind_left_scroll()

        # click-to-pick editing state
        self._edit_mode = tk.StringVar(value="none")   # 'none' | 'deselect' | 'select'
        self._picks = set()                            # mask indices clicked on canvas

        
        # --- batch mode state ---
        self.batch_dir: str | None = None
        self.batch_images: list[str] = []
        self.batch_idx: int = -1
        self._batch_mask_cache = {}  # img_path -> SegResult
        self._batch_out_dir = None

        # UI vars for the Shape Completion tab (mask → mask completion)
        self.shape_masks_var = tk.StringVar(value="")  # Folder with complete leaf masks for training
        self.shape_test_masks_var = tk.StringVar(value="")  # Folder with masks for testing
        self.shape_occ_min_var = tk.DoubleVar(value=0.15)
        self.shape_occ_max_var = tk.DoubleVar(value=0.50)
        self.shape_out_var = tk.StringVar(value=str(Path.home() / "shape_completion.pth"))
        self.shape_steps_var = tk.IntVar(value=2000)
        self.shape_size_var = tk.IntVar(value=128)
        self.shape_batch_var = tk.IntVar(value=4)
        self.shape_device_var = tk.StringVar(value="mps")
        self.shape_model = None
        self.shape_meta = {}

        # UI vars for the Train Custom Model tab
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
        # Default off: "Segment" should run SAM2 unless user explicitly enables tip-only segmentation.
        self.target_use_tipseg = tk.BooleanVar(value=False)
        # Note: model_type_var and tip_model_path_var are defined earlier (before make_model_frame)
        self.target_tipseg_thresh = tk.DoubleVar(value=0.99)
        self.target_tipseg_min_area = tk.IntVar(value=1500)
        self.target_tipseg_keep_largest = tk.BooleanVar(value=False)
        self.tipseg_use_tiles = tk.BooleanVar(value=True)
        self.tipseg_tile_size = tk.IntVar(value=512)
        self.tipseg_stride = tk.IntVar(value=64)
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
        self.tipseg_remove_white = tk.BooleanVar(value=True)
        self.tipseg_white_sat_max = tk.IntVar(value=50)
        self.tipseg_white_val_min = tk.IntVar(value=205)
        self.tipseg_remove_green = tk.BooleanVar(value=True)
        self.tipseg_green_hue_low = tk.IntVar(value=30)
        self.tipseg_green_hue_high = tk.IntVar(value=120)
        self.tipseg_green_sat_min = tk.IntVar(value=25)
        self.tipseg_green_val_min = tk.IntVar(value=50)
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
        self._spinner = AnimatedSpinner(root)

        # click-to-pick editing state
        self._edit_mode = tk.StringVar(value="none")  # 'none' | 'deselect' | 'select'
        self._picks = set()                            # set of mask indices picked on canvas

        # Bind keyboard shortcuts
        self._bind_global_shortcuts()

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

        # ═══════════════════════════════════════════════════════════════════
        # Model Type Selection
        # ═══════════════════════════════════════════════════════════════════
        type_row = ttk.Frame(f)
        type_row.pack(fill="x", pady=(0, 8))
        ttk.Label(type_row, text="Mode:", width=12).pack(side="left")
        ttk.Radiobutton(type_row, text="SAM2", variable=self.model_type_var, value="sam2",
                        command=self._on_model_type_change).pack(side="left")
        ttk.Radiobutton(type_row, text="Custom Model (no SAM)", variable=self.model_type_var, value="tip",
                        command=self._on_model_type_change).pack(side="left", padx=(12, 0))

        # ═══════════════════════════════════════════════════════════════════
        # SAM2 Options (shown when SAM2 selected)
        # ═══════════════════════════════════════════════════════════════════
        self.sam2_frame = ttk.Frame(f)
        self.sam2_frame.pack(fill="x")

        # Row 0: Checkpoint
        row0 = ttk.Frame(self.sam2_frame)
        row0.pack(fill="x", pady=3)
        ttk.Label(row0, text="Checkpoint:", width=12).pack(side="left")
        self.e_ckpt = ttk.Entry(row0, width=38)
        self.e_ckpt.pack(side="left", fill="x", expand=True, padx=(0, 6))
        btn_ckpt = ttk.Button(row0, text="…", width=3, command=self.pick_ckpt, style='Icon.TButton')
        btn_ckpt.pack(side="left")
        ToolTip(btn_ckpt, "Browse for SAM2 checkpoint file (.pt)")

        # Row 1: Config
        row1 = ttk.Frame(self.sam2_frame)
        row1.pack(fill="x", pady=3)
        ttk.Label(row1, text="Config:", width=12).pack(side="left")
        self.e_cfg = ttk.Entry(row1, width=38)
        self.e_cfg.insert(0, "sam2.1_hiera_l")
        self.e_cfg.pack(side="left", fill="x", expand=True, padx=(0, 6))
        btn_cfg = ttk.Button(row1, text="…", width=3, command=self.pick_cfg, style='Icon.TButton')
        btn_cfg.pack(side="left")
        ToolTip(btn_cfg, "Browse for SAM2 config YAML file")

        # Row 2: Device + postprocessing
        row2 = ttk.Frame(self.sam2_frame)
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
        row3 = ttk.Frame(self.sam2_frame)
        row3.pack(fill="x", pady=(10, 4))
        btn_load = ttk.Button(row3, text="⬆ Load Model", command=self.load_model, style='Accent.TButton')
        btn_load.pack(side="left", padx=(0, 8))
        ToolTip(btn_load, "Load SAM2 model from checkpoint and config")
        btn_bundle = ttk.Button(row3, text="📦 Load Bundle…", command=self.load_bundle, style='Accent.TButton')
        btn_bundle.pack(side="left")
        ToolTip(btn_bundle, "Load a pre-packaged SAM2 bundle (.pt file with embedded config)")

        # ═══════════════════════════════════════════════════════════════════
        # Custom Model Options (shown when Custom Model selected)
        # ═══════════════════════════════════════════════════════════════════
        self.tip_frame = ttk.Frame(f)
        # Initially hidden - will be shown when tip model is selected

        # Row 0: Model path
        tip_row0 = ttk.Frame(self.tip_frame)
        tip_row0.pack(fill="x", pady=3)
        ttk.Label(tip_row0, text="Model file:", width=12).pack(side="left")
        self.e_tip_model = ttk.Entry(tip_row0, textvariable=self.tip_model_path_var, width=38)
        self.e_tip_model.pack(side="left", fill="x", expand=True, padx=(0, 6))
        btn_tip = ttk.Button(tip_row0, text="…", width=3, command=self._browse_tip_model, style='Icon.TButton')
        btn_tip.pack(side="left")
        ToolTip(btn_tip, "Browse for tip model (.pth)")

        # Row 1: Device
        tip_row1 = ttk.Frame(self.tip_frame)
        tip_row1.pack(fill="x", pady=3)
        ttk.Label(tip_row1, text="Device:", width=12).pack(side="left")
        self.e_tip_dev = ttk.Entry(tip_row1, width=8)
        self.e_tip_dev.insert(0, "mps")
        self.e_tip_dev.pack(side="left")
        ToolTip(self.e_tip_dev, "Device to run model on (cpu, cuda, mps)")

        # Row 2: Load button
        tip_row2 = ttk.Frame(self.tip_frame)
        tip_row2.pack(fill="x", pady=(10, 4))
        btn_load_tip = ttk.Button(tip_row2, text="⬆ Load Custom Model", command=self._load_tip_model_main, style='Accent.TButton')
        btn_load_tip.pack(side="left")
        ToolTip(btn_load_tip, "Load the tip segmentation model")

        # Status label
        self.tip_status_lbl = ttk.Label(self.tip_frame, text="No model loaded", anchor="w")
        self.tip_status_lbl.pack(fill="x", pady=(4, 0))


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

        # Rotation preset buttons
        preset_row = ttk.Frame(sec1)
        preset_row.pack(fill="x", pady=(2, 0))
        ttk.Label(preset_row, text="Quick:").pack(side="left", padx=(0, 6))
        for label, angle in [
            ("90°", 90), ("180°", 180), ("270°", 270),
            ("-90°", -90), ("45°", 45), ("-45°", -45),
        ]:
            btn = ttk.Button(
                preset_row, text=label, width=4,
                command=lambda a=angle: self._set_angle(a),
                style="Icon.TButton",
            )
            btn.pack(side="left", padx=(0, 2))
            ToolTip(btn, f"Set rotation to {label}")

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
        self._br_scale = ttk.Scale(br_row, from_=-100, to=100, variable=self.s_brightness, orient="horizontal",
                                    command=lambda _: self._schedule_auto_preview())
        self._br_scale.pack(side="left", fill="x", expand=True)
        self._br_val = ttk.Label(br_row, text="0", width=4)
        self._br_val.pack(side="left")
        self.s_brightness.trace_add("write", lambda *_: self._br_val.configure(text=str(self.s_brightness.get())))

        # Contrast with value display
        ct_row = ttk.Frame(sec2)
        ct_row.pack(fill="x", pady=1)
        ttk.Label(ct_row, text="Contrast", width=9).pack(side="left")
        self._ct_scale = ttk.Scale(ct_row, from_=0.5, to=2.0, variable=self.s_contrast, orient="horizontal",
                                    command=lambda _: self._schedule_auto_preview())
        self._ct_scale.pack(side="left", fill="x", expand=True)
        self._ct_val = ttk.Label(ct_row, text="1.0", width=4)
        self._ct_val.pack(side="left")
        self.s_contrast.trace_add("write", lambda *_: self._ct_val.configure(text=f"{self.s_contrast.get():.1f}"))

        # Gamma with value display
        gm_row = ttk.Frame(sec2)
        gm_row.pack(fill="x", pady=1)
        ttk.Label(gm_row, text="Gamma", width=9).pack(side="left")
        self._gm_scale = ttk.Scale(gm_row, from_=0.5, to=2.5, variable=self.s_gamma, orient="horizontal",
                                    command=lambda _: self._schedule_auto_preview())
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
        ttk.Scale(us_row1, from_=0.5, to=3.0, variable=self.unsharp_amount, orient="horizontal", length=60,
                  command=lambda _: self._schedule_auto_preview()).pack(side="left")
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
        ttk.Scale(ed_row, from_=0.0, to=1.0, variable=self.ed_amount, orient="horizontal", length=60,
                  command=lambda _: self._schedule_auto_preview()).pack(side="left")

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
        ttk.Scale(veg_row, from_=0.0, to=1.0, variable=self.veg_index_blend, orient="horizontal", length=80,
                  command=lambda _: self._schedule_auto_preview()).pack(side="left")

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
        ttk.Scale(nlm_row, from_=1, to=30, variable=self.nlm_h, orient="horizontal", length=80,
                  command=lambda _: self._schedule_auto_preview()).pack(side="left")

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
        ttk.Scale(sh_row, from_=0.0, to=1.0, variable=self.shadow_amount, orient="horizontal", length=60,
                  command=lambda _: self._schedule_auto_preview()).pack(side="left")
        ttk.Label(sh_row, text="Highlight:").pack(side="left", padx=(4, 2))
        ttk.Scale(sh_row, from_=0.0, to=1.0, variable=self.highlight_amount, orient="horizontal", length=60,
                  command=lambda _: self._schedule_auto_preview()).pack(side="left")

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

        # Wire auto-preview to enhancement controls (once)
        self._wire_auto_preview()

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

        # SAM2 params — each label and entry gets a hover tooltip
        _SAM_TIPS = {
            "pts/side": (
                "points_per_side — grid density of prompt points.\n"
                "Higher = more mask proposals, slower segmentation.\n"
                "Good range: 16–64. Try 32 for dense rosettes."
            ),
            "pts/batch": (
                "points_per_batch — how many prompt points are processed\n"
                "together. Higher uses more VRAM but is faster overall.\n"
                "Lower if you get out-of-memory errors. Default: 16."
            ),
            "IoU thresh": (
                "pred_iou_thresh — model's own quality score cutoff.\n"
                "Higher (0.85–0.95) = only confident masks kept, fewer but cleaner.\n"
                "Lower (0.5–0.7) = more masks found, more noise. Default: 0.90."
            ),
            "Stability": (
                "stability_score_thresh — rejects masks that change shape\n"
                "under small input perturbations (i.e. wobbly / uncertain masks).\n"
                "Lower this if thin or low-contrast leaves keep vanishing."
            ),
            "Crop layers": (
                "crop_n_layers — number of multi-scale crop passes.\n"
                "0 = single pass. 1+ = also segments zoomed-in crops.\n"
                "Use 1–2 to catch small leaves; costs extra time."
            ),
            "Overlap": (
                "crop_overlap_ratio — how much adjacent crops overlap (0–1).\n"
                "More overlap reduces split masks at crop boundaries\n"
                "but makes segmentation slower. Default: 0.30."
            ),
            "NMS thresh": (
                "box_nms_thresh — IoU threshold for suppressing duplicate masks.\n"
                "Lower = more aggressive deduplication (fewer overlapping masks).\n"
                "Raise if touching leaves are incorrectly merged. Default: 0.60."
            ),
            "Min area": (
                "min_mask_region_area — drop masks smaller than this (px²).\n"
                "Removes small noise blobs. Raise for noisy images,\n"
                "lower if small seedlings are being missed. Default: 800."
            ),
            "use_m2m": (
                "use_m2m — extra mask-to-mask refinement pass.\n"
                "Improves boundary quality, especially where leaves overlap.\n"
                "Slightly slower. Recommended to keep ON."
            ),
            "Output": (
                "output_mode — format of the returned mask data.\n"
                "'binary_mask' = boolean numpy array (best for PNG export).\n"
                "Other modes (coco_rle etc.) are for external annotation tools."
            ),
        }

        sam_params = [
            [("pts/side",    self.m_points_per_side),  ("pts/batch", self.m_points_per_batch)],
            [("IoU thresh",  self.m_pred_iou_thresh),  ("Stability", self.m_stability_score_thresh)],
            [("Crop layers", self.m_crop_n_layers),    ("Overlap",   self.m_crop_overlap_ratio)],
            [("NMS thresh",  self.m_box_nms_thresh),   ("Min area",  self.m_min_mask_region_area)],
        ]

        for row_data in sam_params:
            row = ttk.Frame(sec3)
            row.pack(fill="x", pady=1)
            for lbl, var in row_data:
                lbl_w = ttk.Label(row, text=lbl, width=10)
                lbl_w.pack(side="left")
                ent_w = ttk.Entry(row, width=6, textvariable=var)
                ent_w.pack(side="left", padx=(0, 12))
                if lbl in _SAM_TIPS:
                    ToolTip(lbl_w, _SAM_TIPS[lbl])
                    ToolTip(ent_w, _SAM_TIPS[lbl])

        # Checkbox + output row
        opt_row = ttk.Frame(sec3)
        opt_row.pack(fill="x", pady=(4, 0))
        chk_m2m = ttk.Checkbutton(opt_row, text="use_m2m", variable=self.m_use_m2m)
        chk_m2m.pack(side="left")
        ToolTip(chk_m2m, _SAM_TIPS["use_m2m"])

        out_lbl = ttk.Label(opt_row, text="Output:")
        out_lbl.pack(side="left", padx=(12, 4))
        out_cb = ttk.Combobox(opt_row, width=12, state="readonly",
                              textvariable=self.m_output_mode,
                              values=("binary_mask", "coco_rle",
                                      "uncompressed_rle", "polygons"))
        out_cb.pack(side="left")
        ToolTip(out_lbl, _SAM_TIPS["Output"])
        ToolTip(out_cb,  _SAM_TIPS["Output"])

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
        chk_auto = ttk.Checkbutton(action_row, text="Auto Preview", variable=self.auto_preview,
                                   command=self._schedule_auto_preview)
        chk_auto.pack(side="left", padx=(8, 0))
        ToolTip(chk_auto, "Automatically preview enhancements when settings change")
        btn_segment = ttk.Button(action_row, text="✂️ Segment", command=self.segment, style='Accent.TButton')
        btn_segment.pack(side="left", padx=(12, 0))
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

        # Save mask size options
        save_opts = ttk.Frame(sec4)
        save_opts.pack(fill="x", pady=(0, 4))
        ttk.Label(save_opts, text="Mask size:", width=8).pack(side="left")
        chk_full = ttk.Checkbutton(save_opts, text="Full (original)", variable=self.save_mask_full)
        chk_full.pack(side="left")
        ToolTip(chk_full, "Save full-size masks (original image dimensions)")
        chk_crop = ttk.Checkbutton(save_opts, text="Crop (bbox)", variable=self.save_mask_crop)
        chk_crop.pack(side="left", padx=(8, 0))
        ToolTip(chk_crop, "Save cropped masks (bbox dimensions)")

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

        btn_edit = ttk.Button(bar, text="✏", width=3, command=self.edit_mask_mode, style='Icon.TButton')
        btn_edit.pack(side="left", padx=(4, 0))
        ToolTip(btn_edit, "Edit mask boundary")

        btn_dup = ttk.Button(bar, text="⧉", width=3, command=self.duplicate_selected_masks, style='Icon.TButton')
        btn_dup.pack(side="left", padx=(4, 0))
        ToolTip(btn_dup, "Duplicate selected masks")

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

    def _make_scrollable_tab(self, notebook, tab_text):
        """Create a scrollable tab frame with Canvas + Scrollbar."""
        # Outer frame to hold canvas + scrollbar
        outer = ttk.Frame(notebook)
        notebook.add(outer, text=tab_text)

        # Canvas for scrolling
        canvas = tk.Canvas(outer, highlightthickness=0, bg=self.colors['bg_pale'])
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Inner frame that holds the actual content
        inner = ttk.Frame(canvas, padding=8)
        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")

        # Update scroll region when inner frame changes size
        def _on_inner_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind("<Configure>", _on_inner_configure)

        # Make canvas resize inner frame width
        def _on_canvas_configure(e):
            canvas.itemconfig(canvas_window, width=e.width)
        canvas.bind("<Configure>", _on_canvas_configure)

        # Scrolling is handled globally by root.bind_all in __init__.
        # _bind_scroll is kept as a no-op so existing call sites don't crash.
        inner._bind_scroll = lambda: None

        return inner

    def make_training_frame(self, parent):
        """Build the Training panel — each tab is in its own module."""
        tf = ttk.LabelFrame(parent,
                            text=f"  {self.icons['training']} Training  ",
                            padding=(10, 5))

        notebook = ttk.Notebook(tf)
        notebook.pack(fill="both", expand=True)

        # ── Tab 1: Train Custom Model ─────────────────────────────────────────
        tab1 = self._make_scrollable_tab(notebook, "  Train Custom Model  ")
        tab_train.build(self, tab1)
        tab1._bind_scroll()

        # ── Tab 2: Leaf Completion ────────────────────────────────────────────
        tab2 = self._make_scrollable_tab(notebook, "  Leaf Completion  ")
        tab_leaf_completion.build(self, tab2)
        tab2._bind_scroll()

        # ── Tab 3: Leaf Unfolding ─────────────────────────────────────────────
        tab3 = self._make_scrollable_tab(notebook, "  Leaf Unfolding  ")
        tab_leaf_unfolding.build(self, tab3)
        tab3._bind_scroll()

        # ── Shared Training Log ───────────────────────────────────────────────
        log_frame = ttk.LabelFrame(tf, text=" Training Log ", padding=4)
        log_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.train_log = tk.Text(
            log_frame, height=8, width=100,
            bg=self.colors['bg_pale'], fg=self.colors['text_dark'])
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
            bg=c['bg_dark'], fg=c['text_muted'],
            font=("Helvetica", 9), padx=10
        )
        self._weights_label.pack(side="right")

        shortcuts_text = "⌨ Ctrl+O: Open  |  Ctrl+S: Save  |  Del: Delete mask  |  Ctrl+Z: Undo"
        self._shortcuts_label = tk.Label(
            status_frame, text=shortcuts_text,
            bg=c['bg_dark'], fg=c['text_muted'],
            font=("Helvetica", 9), padx=10
        )

        # Undo counter badge
        self._undo_label = tk.Label(
            status_frame, text="Undo: —",
            bg=c['bg_dark'], fg=c['text_muted'],
            font=("Helvetica", 9, "bold"), padx=10,
            cursor="hand2",
        )
        self._undo_label.pack(side="right")
        self._undo_label.bind("<Button-1>", self.undo)
        self._shortcuts_label.pack(side="right")

        # Current file label — always visible centre of status bar
        self._file_label = tk.Label(
            status_frame,
            text="No file loaded",
            bg=c['bg_dark'], fg=c['accent'],
            font=("Helvetica", 9, "bold"), padx=12,
        )
        self._file_label.pack(side="left", padx=(20, 0))

        # Separator line above status bar
        sep = tk.Frame(root, bg=c['border'], height=1)
        sep.grid(row=2, column=0, columnspan=2, sticky="new")

    def _update_file_label(self) -> None:
        """Update the filename badge in the status bar."""
        if not hasattr(self, "_file_label"):
            return
        if self.img_path:
            name = Path(self.img_path).name
            # Show batch context if in folder mode
            if self.batch_images and self.batch_idx >= 0:
                n = len(self.batch_images)
                i = self.batch_idx + 1
                self._file_label.configure(
                    text=f"📄 {name}  [{i} / {n}]",
                    fg=self.colors["accent"])
            else:
                self._file_label.configure(
                    text=f"📄 {name}",
                    fg=self.colors["accent"])
        else:
            self._file_label.configure(
                text="No file loaded",
                fg=self.colors["text_muted"])

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

        # Undo
        self.root.bind("<Control-z>", self.undo)
        self.root.bind("<Command-z>", self.undo)   # macOS

        # Navigation
        self.root.bind("<Control-Left>", lambda e: self.prev_image())
        self.root.bind("<Control-Right>", lambda e: self.next_image())

    # =========================================================================
    # Undo system
    # =========================================================================

    def _push_undo(self, label: str = "action") -> None:
        """Snapshot current masks list onto the undo stack before a mutation."""
        if not self.sr:
            return
        import copy
        snapshot = copy.deepcopy(self.sr.masks)
        self._undo_stack.append((label, snapshot))
        if len(self._undo_stack) > self._undo_max:
            self._undo_stack.pop(0)
        self._update_undo_status()

    def undo(self, event=None) -> None:
        """Restore the most recent snapshot from the undo stack."""
        if not self._undo_stack:
            self.set_status("Nothing to undo", "info")
            return
        if not self.sr:
            return
        label, snapshot = self._undo_stack.pop()
        import copy
        self.sr.masks = copy.deepcopy(snapshot)
        if hasattr(self, "_picks"):
            self._picks.clear()
        self._rebuild_mask_list()
        self._sync_listbox_selection_from_picks()
        if self.img_preview is not None:
            self.show_image(self.img_preview)
        elif self.img is not None:
            self.show_image(self.img)
        self._update_undo_status()
        self.set_status(f"Undid: {label}  ({len(self._undo_stack)} left)", "success")

    def _update_undo_status(self) -> None:
        """Update the undo count badge in the status bar if the label exists."""
        n = len(self._undo_stack)
        if hasattr(self, "_undo_label"):
            self._undo_label.configure(
                text=f"Undo: {n}" if n else "Undo: —",
                fg=self.colors["accent"] if n else self.colors["text_muted"],
            )

    def _browse_file_into(self, var, ftypes=("All","*.*")):
        p = filedialog.askopenfilename(filetypes=[ftypes] if isinstance(ftypes, tuple) else [("All","*.*")])
        if p: var.set(p)

    def _browse_folder_into(self, var, title="Choose folder"):
        p = filedialog.askdirectory(title=title)
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

    # ===== Train Custom Model dataset helpers =====
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

    def _train_reader_thread(self):
        """Read training subprocess output and display in log."""
        p = getattr(self, "_train_proc", None)
        if not p:
            return
        for raw in iter(p.stdout.readline, b""):
            line = raw.decode(errors="replace").rstrip()
            self.root.after(0, lambda s=line: self._append_train_log(s))
        p.wait()
        code = p.returncode
        self.root.after(0, lambda: self._append_train_log(f"Training finished with code {code}"))

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
        mask_bool = _ensure_mask_2d(mask_bool)
        if mask_bool is None:
            return None
        ys, xs = np.nonzero(mask_bool)
        if xs.size == 0:
            return None
        H, W = mask_bool.shape[:2]

        # Check if mask dimensions match image dimensions
        img_H, img_W = img_rgb.shape[:2]
        if H != img_H or W != img_W:
            # Resize mask to match image dimensions
            mask_resized = cv2.resize(
                mask_bool.astype(np.uint8), (img_W, img_H),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            mask_bool = mask_resized
            H, W = img_H, img_W
            ys, xs = np.nonzero(mask_bool)
            if xs.size == 0:
                return None

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
        if vals.size == 0:
            return None
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
        img_H, img_W = img_rgb.shape[:2]
        for m in masks:
            seg = m.get("segmentation")
            if seg is None:
                scores.append(0.0)
                continue
            mask = _ensure_mask_2d(seg).astype(bool)
            if mask is None:
                scores.append(0.0)
                continue
            # Resize mask if dimensions don't match image
            mask_H, mask_W = mask.shape[:2]
            if mask_H != img_H or mask_W != img_W:
                mask = cv2.resize(
                    mask.astype(np.uint8), (img_W, img_H),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
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
            H, W = img_H, img_W
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

    # ════════════════════════════════════════════════════════════════════════
    # Shape Completion (Mask → Mask U-Net) Methods
    # ════════════════════════════════════════════════════════════════════════

    def _pick_shape_masks_folder(self):
        """Pick folder containing complete leaf masks for training."""
        d = filedialog.askdirectory(title="Select folder with complete leaf masks")
        if d:
            self.shape_masks_var.set(d)
            # Count masks
            exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
            count = sum(1 for p in Path(d).rglob("*") if p.suffix.lower() in exts)
            self.shape_count_lbl.configure(text=f"({count} masks)")
            self._append_train_log(f"Selected masks folder: {d} ({count} masks found)")

    def _launch_shape_training(self):
        """Launch mask_completion.py train for shape completion."""
        masks_dir = self.shape_masks_var.get().strip()
        if not masks_dir or not os.path.isdir(masks_dir):
            messagebox.showwarning("Shape Completion", "Pick a valid folder with complete leaf masks.")
            return

        out_path = self.shape_out_var.get().strip() or str(Path.home() / "shape_completion.pth")
        steps = int(self.shape_steps_var.get())
        img_size = int(self.shape_size_var.get())
        batch = int(self.shape_batch_var.get())
        device = self.shape_device_var.get().strip() or "mps"
        occ_min = float(self.shape_occ_min_var.get())
        occ_max = float(self.shape_occ_max_var.get())

        script_path = Path(__file__).parent / "mask_completion.py"
        if not script_path.exists():
            messagebox.showerror("Shape Completion", f"Cannot find mask_completion.py at:\n{script_path}")
            return

        py = shlex.quote(sys.executable)
        cmd = (
            f"{py} {shlex.quote(str(script_path))} train "
            f"--masks {shlex.quote(masks_dir)} "
            f"--output {shlex.quote(out_path)} "
            f"--steps {steps} "
            f"--size {img_size} "
            f"--batch {batch} "
            f"--device {device} "
            f"--occ-min {occ_min:.2f} "
            f"--occ-max {occ_max:.2f}"
        )

        self._append_train_log("")
        self._append_train_log("=" * 60)
        self._append_train_log("Launching Shape Completion Training (mask → mask):")
        self._append_train_log(f"  Masks: {masks_dir}")
        self._append_train_log(f"  Output: {out_path}")
        self._append_train_log(f"  Steps: {steps}, Size: {img_size}, Batch: {batch}, Device: {device}")
        self._append_train_log(f"  Occlusion: {occ_min*100:.0f}%-{occ_max*100:.0f}%")
        self._append_train_log("=" * 60)
        self._append_train_log(f"Command: {cmd}")
        self._append_train_log("")

        try:
            self._train_job_name = "Shape Completion"
            self.shape_status_lbl.configure(text="Training started...")
            preexec = os.setsid if hasattr(os, "setsid") else None
            self._occ_train_proc = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).parent),
                preexec_fn=preexec,
            )
            threading.Thread(target=self._shape_train_reader_thread, daemon=True).start()
            self.set_status("Shape completion training started...", "info")
        except Exception as e:
            messagebox.showerror("Shape Completion", f"Failed to start training:\n{e}")

    def _shape_train_reader_thread(self):
        """Read output from the shape completion training subprocess."""
        p = getattr(self, "_occ_train_proc", None)
        if not p:
            return
        for raw in iter(p.stdout.readline, b""):
            line = raw.decode(errors="replace").rstrip()
            self.root.after(0, lambda s=line: self._append_train_log(s))
            # Update status label with progress
            if "loss=" in line.lower() or "iou=" in line.lower():
                self.root.after(0, lambda s=line: self.shape_status_lbl.configure(text=s[:80]))
        p.wait()
        code = p.returncode
        job = getattr(self, "_train_job_name", "Training")
        self.root.after(0, lambda: self._append_train_log(f"\n{job} finished with code {code}"))
        if code == 0:
            self.root.after(0, lambda: self.shape_status_lbl.configure(text="Training completed!"))
            self.root.after(0, lambda: self.set_status(f"{job} completed!", "success"))
        else:
            self.root.after(0, lambda: self.shape_status_lbl.configure(text=f"Training failed (code {code})"))
            self.root.after(0, lambda: self.set_status(f"Training failed with code {code}", "error"))

    def _load_shape_model(self):
        """Load a trained shape completion model."""
        out_path = self.shape_out_var.get().strip()
        if not out_path or not os.path.exists(out_path):
            # Ask user to pick file
            out_path = filedialog.askopenfilename(
                title="Select shape completion model",
                filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")]
            )
            if not out_path:
                return
            self.shape_out_var.set(out_path)

        try:
            from mask_completion import load_model
            device = self.shape_device_var.get().strip() or "mps"
            model, meta = load_model(out_path, device=device)
            self.shape_model = model
            self.shape_meta = meta or {}
            self._append_train_log(f"Loaded shape model from: {out_path}")
            self.shape_status_lbl.configure(text=f"Model loaded: {Path(out_path).name}")
            messagebox.showinfo("Shape Model", f"Shape completion model loaded.\n\n{out_path}")
            self.set_status("Shape model loaded", "success")
        except Exception as e:
            messagebox.showerror("Shape Completion", f"Failed to load shape model:\n{e}")

    def _pick_shape_test_masks_folder(self):
        """Pick folder containing masks for testing."""
        d = filedialog.askdirectory(title="Select folder with test masks (complete masks)")
        if d:
            self.shape_test_masks_var.set(d)
            exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
            count = sum(1 for p in Path(d).rglob("*") if p.suffix.lower() in exts)
            self._append_train_log(f"Selected test masks folder: {d} ({count} masks found)")

    def _test_shape_model(self):
        """Test the loaded shape model on masks."""
        if self.shape_model is None:
            messagebox.showwarning("Shape Completion", "Load a shape model first.")
            return

        # Use test masks folder if set, otherwise fall back to training masks
        masks_dir = self.shape_test_masks_var.get().strip()
        if not masks_dir or not os.path.isdir(masks_dir):
            masks_dir = self.shape_masks_var.get().strip()
        if not masks_dir or not os.path.isdir(masks_dir):
            messagebox.showwarning("Shape Completion", "Pick a test masks folder first.")
            return

        out_path = self.shape_out_var.get().strip()
        script_path = Path(__file__).parent / "mask_completion.py"
        if not script_path.exists():
            messagebox.showerror("Shape Completion", f"Cannot find mask_completion.py")
            return

        # Create output directory
        test_out = Path(masks_dir).parent / "shape_test_results"
        test_out.mkdir(exist_ok=True)

        py = shlex.quote(sys.executable)
        cmd = (
            f"{py} {shlex.quote(str(script_path))} test "
            f"--model {shlex.quote(out_path)} "
            f"--masks {shlex.quote(masks_dir)} "
            f"--output {shlex.quote(str(test_out))} "
            f"--num 6"
        )

        self._append_train_log(f"Running test: {cmd}")
        self.shape_status_lbl.configure(text="Running test...")

        def run_test():
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=str(Path(__file__).parent))
                self.root.after(0, lambda: self._append_train_log(result.stdout))
                if result.stderr:
                    self.root.after(0, lambda: self._append_train_log(result.stderr))
                if result.returncode == 0:
                    self.root.after(0, lambda: self.shape_status_lbl.configure(text=f"Test done! Results in {test_out}"))
                    self.root.after(0, lambda: messagebox.showinfo("Test Complete", f"Results saved to:\n{test_out}"))
                    # Open the results folder
                    if sys.platform == "darwin":
                        subprocess.run(["open", str(test_out)])
                else:
                    self.root.after(0, lambda: self.shape_status_lbl.configure(text="Test failed"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Test", f"Test failed:\n{e}"))

        threading.Thread(target=run_test, daemon=True).start()

    def _stop_training(self):
        """Stop the currently running training process, if any."""
        p = getattr(self, "_occ_train_proc", None)
        if not p or p.poll() is not None:
            messagebox.showinfo("Training", "No active training process.")
            return
        try:
            self._append_train_log("Stop requested by user.")
            if hasattr(os, "killpg") and hasattr(os, "setsid"):
                os.killpg(p.pid, signal.SIGTERM)
            else:
                p.terminate()
            self.set_status("Stopping training…", "warning")
        except Exception as e:
            messagebox.showerror("Training", f"Failed to stop training:\n{e}")

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

        self._push_undo("delete mask")

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

        self._push_undo("combine masks")

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
        self._push_undo("extend mask")
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

    def _load_shape_model_from_masks(self):
        """Load shape completion model from the Masks panel."""
        out_path = filedialog.askopenfilename(
            title="Select shape completion model",
            filetypes=[("PyTorch model", "*.pth *.pt"), ("All files", "*.*")]
        )
        if not out_path:
            return

        try:
            from mask_completion import load_model
            # Try to get device from shape_device_var if available, otherwise default to mps
            device = getattr(self, 'shape_device_var', None)
            device = device.get().strip() if device else "mps"
            if not device:
                device = "mps"

            model, meta = load_model(out_path, device=device)
            self.shape_model = model
            self.shape_meta = meta or {}

            # Update the shape_out_var if it exists (for consistency with Training tab)
            if hasattr(self, 'shape_out_var'):
                self.shape_out_var.set(out_path)
            if hasattr(self, 'shape_status_lbl'):
                self.shape_status_lbl.configure(text=f"Model loaded: {Path(out_path).name}")

            messagebox.showinfo("Shape Model", f"Shape completion model loaded!\n\n{Path(out_path).name}")
            self.set_status("Shape model loaded", "success")
        except Exception as e:
            messagebox.showerror("Shape Completion", f"Failed to load shape model:\n{e}")

    def on_extend_shape(self):
        """Extend selected mask using geometric shape fitting (no model needed)."""
        sel = self.lb.curselection()
        if not sel:
            messagebox.showwarning("Extend Shape", "Select a mask first.")
            return
        if len(sel) > 1:
            messagebox.showwarning("Extend Shape", "Select only one mask.")
            return
        if self.sr is None or not self.sr.masks:
            messagebox.showwarning("Extend Shape", "No masks available.")
            return

        idx = sel[0]
        if idx >= len(self.sr.masks):
            return

        # Get the mask segmentation data
        mask_data = self.sr.masks[idx]
        base = mask_data["segmentation"].copy()
        shape_type = self.shape_extend_var.get()

        # Fit shape and get completed mask
        completed = self._fit_shape_to_mask(base, shape_type)
        if completed is None:
            messagebox.showwarning("Extend Shape", "Could not fit shape to mask.")
            return

        # Calculate added area
        added = np.logical_and(completed, ~base)
        base_area = int(base.sum())
        completed_area = int(completed.sum())
        added_area = int(added.sum())

        if added_area == 0:
            messagebox.showinfo("Extend Shape", f"Shape fitting did not extend the mask.\nThe {shape_type} shape fits within the original mask.")
            return

        # Calculate average RGB color from original segment area
        avg_color = None
        if self.sr is not None and self.sr.img_color is not None:
            img_arr = self.sr.img_color
            if img_arr.ndim == 3 and img_arr.shape[2] >= 3:
                mask_pixels = img_arr[base]
                if len(mask_pixels) > 0:
                    avg_r = int(mask_pixels[:, 0].mean())
                    avg_g = int(mask_pixels[:, 1].mean())
                    avg_b = int(mask_pixels[:, 2].mean())
                    avg_color = (avg_r, avg_g, avg_b)

        # Update the mask with completed version
        self.sr.masks[idx]["segmentation"] = completed
        self.sr.masks[idx]["area"] = completed_area

        # Store extended info for visualization
        if not hasattr(self, '_mask_extended_info'):
            self._mask_extended_info = {}
        self._mask_extended_info[idx] = {
            'extended_bool': added,
            'avg_color': avg_color,
            'shape_type': shape_type,
        }

        print(f"[Extend Shape] {shape_type}: Base area: {base_area}, Completed: {completed_area}, Added: {added_area}")

        # Update display
        self._rebuild_mask_list()
        self.lb.selection_clear(0, tk.END)
        self.lb.selection_set(idx)
        self.on_select_mask(None)
        self.set_status(f"Extended with {shape_type}: +{added_area} pixels", "success")

    def _update_leaf_completion_preview(self, mask: np.ndarray = None, idx: int = None):
        """Update the original mask preview in the Leaf Completion tab."""
        if not hasattr(self, '_leaf_orig_canvas'):
            return

        # Clear canvases
        self._leaf_orig_canvas.delete("all")
        self._leaf_comp_canvas.delete("all")

        if mask is None:
            self._leaf_stats_label.configure(text="Select a mask to preview")
            return

        # Draw original mask on the canvas
        zoom = getattr(self, '_leaf_preview_zoom', 1.0)
        self._draw_mask_on_canvas(self._leaf_orig_canvas, mask, 250, 250, zoom=zoom)

        # Store the current mask index for later
        self._leaf_preview_idx = idx
        self._leaf_preview_orig_mask = mask.copy()

        # Update stats
        area = int(mask.sum())
        self._leaf_stats_label.configure(text=f"Original area: {area:,} pixels")

        # Clear any pending completion
        self._pending_leaf_completion = None
        if hasattr(self, '_leaf_update_btn'):
            self._leaf_update_btn.configure(state="disabled")

    def _draw_mask_on_canvas(self, canvas, mask: np.ndarray, canvas_w: int, canvas_h: int,
                              completed_mask: np.ndarray = None, zoom: float = 1.0):
        """Draw a mask on a canvas, optionally showing completion overlay.

        Args:
            canvas: Tkinter canvas to draw on
            mask: Binary mask array
            canvas_w, canvas_h: Canvas dimensions
            completed_mask: Optional completed mask to show overlay
            zoom: Zoom factor (1.0 = fit to canvas, 2.0 = 200% zoom, etc.)
        """
        from PIL import Image, ImageTk

        canvas.delete("all")
        h, w = mask.shape

        # Create RGB image from mask
        if completed_mask is not None:
            # Show original in one color, added area in another
            img_arr = np.zeros((h, w, 3), dtype=np.uint8)
            # Original area in teal
            img_arr[mask] = [117, 178, 178]  # Teal for original
            # Added area in green (or red if shrunk)
            added = np.logical_and(completed_mask, ~mask)
            removed = np.logical_and(mask, ~completed_mask)
            img_arr[added] = [100, 220, 100]  # Green for added
            img_arr[removed] = [220, 100, 100]  # Red for removed (when shrinking)
        else:
            # Just show mask in teal
            img_arr = np.zeros((h, w, 3), dtype=np.uint8)
            img_arr[mask] = [117, 178, 178]

        # Resize to fit canvas while maintaining aspect ratio, then apply zoom
        img = Image.fromarray(img_arr)
        fit_scale = min(canvas_w / w, canvas_h / h) * 0.9
        scale = fit_scale * zoom
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w > 0 and new_h > 0:
            img = img.resize((new_w, new_h), Image.NEAREST)

        # Convert to PhotoImage and draw
        photo = ImageTk.PhotoImage(img)
        canvas.create_image(canvas_w // 2, canvas_h // 2, image=photo, anchor="center")
        canvas._photo = photo  # Keep reference to prevent garbage collection

    def _preview_leaf_completion(self, scale: float = 1.0):
        """Preview the leaf completion without applying it."""
        sel = self.lb.curselection()
        if not sel:
            messagebox.showwarning("Leaf Completion", "Select a mask first.")
            return
        if len(sel) > 1:
            messagebox.showwarning("Leaf Completion", "Select only one mask for preview.")
            return
        if self.sr is None or not self.sr.masks:
            messagebox.showwarning("Leaf Completion", "No masks available.")
            return

        idx = sel[0]
        if idx >= len(self.sr.masks):
            return

        # Get the mask
        mask_data = self.sr.masks[idx]
        base = mask_data["segmentation"].copy()
        shape_type = self.shape_extend_var.get()

        # Reset scale factor, offset, and rotation when doing a fresh preview
        if scale == 1.0:
            self._leaf_scale_factor = 1.0
            self._leaf_scale_label.configure(text="100%")
            self._leaf_shape_offset = [0, 0]
            if hasattr(self, '_leaf_offset_label'):
                self._leaf_offset_label.configure(text="(0, 0)")
            self._leaf_rotation_offset = 0
            if hasattr(self, '_leaf_rotation_label'):
                self._leaf_rotation_label.configure(text="0°")

        # Fit shape with scale, offset, and rotation
        offset = tuple(getattr(self, '_leaf_shape_offset', [0, 0]))
        rotation = getattr(self, '_leaf_rotation_offset', 0)
        completed = self._fit_shape_to_mask(base, shape_type, scale=self._leaf_scale_factor,
                                            offset=offset, rotation=rotation)
        if completed is None:
            messagebox.showwarning("Leaf Completion", "Could not fit shape to mask.")
            return

        # Calculate stats
        added = np.logical_and(completed, ~base)
        base_area = int(base.sum())
        completed_area = int(completed.sum())
        added_area = int(added.sum())

        # Update the original canvas
        zoom = getattr(self, '_leaf_preview_zoom', 1.0)
        self._draw_mask_on_canvas(self._leaf_orig_canvas, base, 250, 250, zoom=zoom)

        # Update the completed canvas with overlay
        self._draw_mask_on_canvas(self._leaf_comp_canvas, base, 250, 250, completed_mask=completed, zoom=zoom)

        # Enable +/- buttons and edit button
        self._leaf_shrink_btn.configure(state="normal")
        self._leaf_grow_btn.configure(state="normal")
        self._leaf_edit_btn.configure(state="normal")

        # Store base info for scaling
        self._leaf_preview_base = base
        self._leaf_preview_idx = idx

        # Update stats
        if added_area > 0:
            self._leaf_stats_label.configure(
                text=f"Original: {base_area:,} px  →  Completed: {completed_area:,} px  (+{added_area:,} px)")
            # Enable update button
            self._leaf_update_btn.configure(state="normal")
            # Store pending completion
            self._pending_leaf_completion = {
                'idx': idx,
                'completed': completed,
                'base': base,
                'shape_type': shape_type,
                'added_area': added_area,
                'completed_area': completed_area,
            }
        elif added_area < 0:
            # Shrunk smaller than original
            removed_area = -added_area
            self._leaf_stats_label.configure(
                text=f"Original: {base_area:,} px  →  Reduced: {completed_area:,} px  (-{removed_area:,} px)")
            self._leaf_update_btn.configure(state="normal")
            self._pending_leaf_completion = {
                'idx': idx,
                'completed': completed,
                'base': base,
                'shape_type': shape_type,
                'added_area': added_area,
                'completed_area': completed_area,
            }
        else:
            self._leaf_stats_label.configure(
                text=f"No change - {shape_type} matches original mask ({base_area:,} px)")
            self._leaf_update_btn.configure(state="disabled")
            self._pending_leaf_completion = None

        scale_pct = int(self._leaf_scale_factor * 100)
        self.set_status(f"Preview: {shape_type} at {scale_pct}% scale", "info")

    def _grow_leaf_shape(self):
        """Increase the shape size by 5%."""
        if not hasattr(self, '_leaf_preview_base'):
            return
        self._leaf_scale_factor += 0.05
        self._leaf_scale_label.configure(text=f"{int(self._leaf_scale_factor * 100)}%")
        self._update_scaled_preview()

    def _shrink_leaf_shape(self):
        """Decrease the shape size by 5%."""
        if not hasattr(self, '_leaf_preview_base'):
            return
        self._leaf_scale_factor = max(0.5, self._leaf_scale_factor - 0.05)  # Min 50%
        self._leaf_scale_label.configure(text=f"{int(self._leaf_scale_factor * 100)}%")
        self._update_scaled_preview()

    def _update_scaled_preview(self):
        """Update the preview with the current scale factor."""
        if not hasattr(self, '_leaf_preview_base') or self._leaf_preview_base is None:
            return

        base = self._leaf_preview_base
        shape_type = self.shape_extend_var.get()

        # Fit shape with current scale
        completed = self._fit_shape_to_mask(base, shape_type, scale=self._leaf_scale_factor)
        if completed is None:
            return

        # Calculate stats
        added = np.logical_and(completed, ~base)
        base_area = int(base.sum())
        completed_area = int(completed.sum())
        added_area = int(added.sum())

        # Update the completed canvas
        zoom = getattr(self, '_leaf_preview_zoom', 1.0)
        self._draw_mask_on_canvas(self._leaf_comp_canvas, base, 250, 250, completed_mask=completed, zoom=zoom)

        # Update stats and pending completion
        if added_area > 0:
            self._leaf_stats_label.configure(
                text=f"Original: {base_area:,} px  →  Completed: {completed_area:,} px  (+{added_area:,} px)")
            self._leaf_update_btn.configure(state="normal")
        elif added_area < 0:
            removed_area = -added_area
            self._leaf_stats_label.configure(
                text=f"Original: {base_area:,} px  →  Reduced: {completed_area:,} px  (-{removed_area:,} px)")
            self._leaf_update_btn.configure(state="normal")
        else:
            self._leaf_stats_label.configure(
                text=f"No change - shape matches original ({base_area:,} px)")

        # Update pending completion
        self._pending_leaf_completion = {
            'idx': self._leaf_preview_idx,
            'completed': completed,
            'base': base,
            'shape_type': shape_type,
            'added_area': added_area,
            'completed_area': completed_area,
        }

    def _apply_leaf_completion(self):
        """Apply the pending leaf completion to the mask."""
        self._push_undo("leaf completion")
        if not self._pending_leaf_completion:
            messagebox.showwarning("Leaf Completion", "No pending completion. Click 'Preview' first.")
            return

        pending = self._pending_leaf_completion
        idx = pending['idx']
        completed = pending['completed']
        shape_type = pending['shape_type']
        added_area = pending['added_area']
        completed_area = pending['completed_area']
        base = pending['base']

        # Calculate average RGB color from original segment area
        avg_color = None
        if self.sr is not None and self.sr.img_color is not None:
            img_arr = self.sr.img_color
            if img_arr.ndim == 3 and img_arr.shape[2] >= 3:
                mask_pixels = img_arr[base]
                if len(mask_pixels) > 0:
                    avg_r = int(mask_pixels[:, 0].mean())
                    avg_g = int(mask_pixels[:, 1].mean())
                    avg_b = int(mask_pixels[:, 2].mean())
                    avg_color = (avg_r, avg_g, avg_b)

        # Update the mask
        self.sr.masks[idx]["segmentation"] = completed
        self.sr.masks[idx]["area"] = completed_area

        # Store extended info for visualization
        added = np.logical_and(completed, ~base)
        if not hasattr(self, '_mask_extended_info'):
            self._mask_extended_info = {}
        self._mask_extended_info[idx] = {
            'extended_bool': added,
            'avg_color': avg_color,
            'shape_type': shape_type,
        }

        print(f"[Leaf Completion] Applied {shape_type}: +{added_area} pixels")

        # Clear pending
        self._pending_leaf_completion = None
        self._leaf_update_btn.configure(state="disabled")

        # Update display
        self._rebuild_mask_list()
        self.lb.selection_clear(0, tk.END)
        self.lb.selection_set(idx)
        self.on_select_mask(None)
        self.set_status(f"Applied {shape_type}: +{added_area} pixels", "success")

        # Update the preview to show final result
        self._leaf_stats_label.configure(text=f"✓ Mask updated: {completed_area:,} px (+{added_area:,} added)")

    def _cancel_leaf_completion(self):
        """Cancel the pending leaf completion."""
        self._pending_leaf_completion = None
        self._leaf_update_btn.configure(state="disabled")

        # Reset scale factor and disable +/- buttons
        self._leaf_scale_factor = 1.0
        if hasattr(self, '_leaf_scale_label'):
            self._leaf_scale_label.configure(text="100%")
        if hasattr(self, '_leaf_shrink_btn'):
            self._leaf_shrink_btn.configure(state="disabled")
        if hasattr(self, '_leaf_grow_btn'):
            self._leaf_grow_btn.configure(state="disabled")
        if hasattr(self, '_leaf_edit_btn'):
            self._leaf_edit_btn.configure(state="disabled")

        # Reset position offset
        self._leaf_shape_offset = [0, 0]
        if hasattr(self, '_leaf_offset_label'):
            self._leaf_offset_label.configure(text="(0, 0)")

        # Reset rotation offset
        self._leaf_rotation_offset = 0
        if hasattr(self, '_leaf_rotation_label'):
            self._leaf_rotation_label.configure(text="0°")

        # Clear stored base mask
        self._leaf_preview_base = None

        # Clear the completed canvas
        if hasattr(self, '_leaf_comp_canvas'):
            self._leaf_comp_canvas.delete("all")

        # Restore original preview if we have it
        if hasattr(self, '_leaf_preview_orig_mask') and self._leaf_preview_orig_mask is not None:
            self._draw_mask_on_canvas(self._leaf_orig_canvas, self._leaf_preview_orig_mask, 250, 250)
            area = int(self._leaf_preview_orig_mask.sum())
            self._leaf_stats_label.configure(text=f"Original area: {area:,} pixels")
        else:
            self._leaf_stats_label.configure(text="Completion cancelled")

        self.set_status("Leaf completion cancelled", "info")

    def _edit_leaf_shape(self):
        """Open contour editor to fine-tune the completed leaf shape."""
        if not self._pending_leaf_completion:
            messagebox.showwarning("Edit Shape", "No shape to edit. Click 'Preview' first.")
            return

        completed = self._pending_leaf_completion['completed']
        base = self._pending_leaf_completion['base']
        idx = self._pending_leaf_completion['idx']

        # Open the contour editor for the completed shape
        self._show_leaf_contour_editor(idx, completed, base)

    def _show_leaf_contour_editor(self, idx: int, completed: np.ndarray, base: np.ndarray):
        """Show contour editor dialog for the completed leaf shape."""
        # Create toplevel window
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Completed Shape")
        dialog.geometry("900x700")
        dialog.transient(self.root)
        dialog.grab_set()

        self._leaf_edit_dialog = dialog
        self._leaf_edit_idx = idx
        self._leaf_edit_completed_original = completed.copy()
        self._leaf_edit_base = base.copy()

        # Extract contour from completed shape
        contours, _ = cv2.findContours(completed.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            messagebox.showwarning("Edit Shape", "Could not extract shape contour.")
            dialog.destroy()
            return

        # Use largest contour
        contour = max(contours, key=cv2.contourArea)

        # Simplify contour to get control points
        epsilon = 0.008 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        pts = approx.reshape(-1, 2).tolist()
        if len(pts) > 60:
            step = len(pts) // 50
            pts = pts[::step]
        elif len(pts) < 10:
            pts = contour.reshape(-1, 2).tolist()
            step = max(1, len(pts) // 40)
            pts = pts[::step]

        self._leaf_edit_points = pts
        self._leaf_edit_dragging_idx = None
        self._leaf_edit_hover_idx = None

        # Get bbox for cropping with padding
        ys, xs = np.nonzero(completed)
        if len(ys) == 0:
            dialog.destroy()
            return
        x1_c, x2_c = int(xs.min()), int(xs.max())
        y1_c, y2_c = int(ys.min()), int(ys.max())

        pad = 50
        img_h, img_w = completed.shape
        y1 = max(0, y1_c - pad)
        y2 = min(img_h, y2_c + pad)
        x1 = max(0, x1_c - pad)
        x2 = min(img_w, x2_c + pad)
        self._leaf_edit_crop_bounds = (y1, y2, x1, x2)

        # Main layout
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Left: Instructions and buttons
        tools_frame = ttk.LabelFrame(main_frame, text=" Edit Shape ", padding=8)
        tools_frame.pack(side="left", fill="y", padx=(0, 10))

        ttk.Label(tools_frame, text="Instructions:", font=("Helvetica", 10, "bold")).pack(anchor="w")
        ttk.Label(tools_frame, text="• Drag points to reshape", wraplength=150).pack(anchor="w", pady=2)
        ttk.Label(tools_frame, text="• Ctrl/Cmd+click to add", wraplength=150).pack(anchor="w", pady=2)
        ttk.Label(tools_frame, text="• Shift+click to delete", wraplength=150).pack(anchor="w", pady=2)

        ttk.Separator(tools_frame, orient="horizontal").pack(fill="x", pady=10)

        # Point count display
        self._leaf_edit_point_label = ttk.Label(tools_frame, text=f"Points: {len(pts)}")
        self._leaf_edit_point_label.pack(anchor="w", pady=5)

        # Add/Delete point buttons
        point_btn_frame = ttk.Frame(tools_frame)
        point_btn_frame.pack(fill="x", pady=5)
        self._leaf_edit_add_mode = tk.BooleanVar(value=False)
        self._leaf_edit_delete_mode = tk.BooleanVar(value=False)

        self._leaf_edit_add_btn = ttk.Checkbutton(point_btn_frame, text="➕ Add Point",
                                                   variable=self._leaf_edit_add_mode,
                                                   command=self._leaf_edit_toggle_add)
        self._leaf_edit_add_btn.pack(fill="x", pady=2)

        self._leaf_edit_del_btn = ttk.Checkbutton(point_btn_frame, text="➖ Delete Point",
                                                   variable=self._leaf_edit_delete_mode,
                                                   command=self._leaf_edit_toggle_delete)
        self._leaf_edit_del_btn.pack(fill="x", pady=2)

        ttk.Separator(tools_frame, orient="horizontal").pack(fill="x", pady=10)

        ttk.Button(tools_frame, text="Smooth", command=self._leaf_edit_smooth).pack(fill="x", pady=2)
        ttk.Button(tools_frame, text="Reset", command=self._leaf_edit_reset).pack(fill="x", pady=2)

        ttk.Separator(tools_frame, orient="horizontal").pack(fill="x", pady=10)

        ttk.Button(tools_frame, text="Apply", command=self._leaf_edit_apply).pack(fill="x", pady=2)
        ttk.Button(tools_frame, text="Cancel", command=self._leaf_edit_cancel).pack(fill="x", pady=2)

        # Right: Canvas for editing
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side="right", fill="both", expand=True)

        self._leaf_edit_canvas = tk.Canvas(canvas_frame, bg=self.colors['canvas_bg'],
                                            highlightthickness=1, highlightbackground=self.colors['border'])
        self._leaf_edit_canvas.pack(fill="both", expand=True)

        # Bind mouse events
        self._leaf_edit_canvas.bind("<Button-1>", self._leaf_edit_click)
        self._leaf_edit_canvas.bind("<B1-Motion>", self._leaf_edit_drag)
        self._leaf_edit_canvas.bind("<ButtonRelease-1>", self._leaf_edit_release)
        self._leaf_edit_canvas.bind("<Motion>", self._leaf_edit_hover)
        self._leaf_edit_canvas.bind("<Control-Button-1>", self._leaf_edit_add_point)
        self._leaf_edit_canvas.bind("<Command-Button-1>", self._leaf_edit_add_point)
        self._leaf_edit_canvas.bind("<Shift-Button-1>", self._leaf_edit_delete_point)

        # Initialize
        self._leaf_edit_scale = 1.0
        self._leaf_edit_offset = (0, 0)

        # Initial draw
        dialog.update_idletasks()
        dialog.after(100, self._leaf_edit_refresh_canvas)

        dialog.protocol("WM_DELETE_WINDOW", self._leaf_edit_cancel)

    def _leaf_edit_refresh_canvas(self):
        """Refresh the leaf shape editor canvas."""
        if not hasattr(self, '_leaf_edit_canvas') or not self._leaf_edit_canvas.winfo_exists():
            return

        y1, y2, x1, x2 = self._leaf_edit_crop_bounds
        base = self._leaf_edit_base

        # Get image crop if available
        if self.sr is not None and self.sr.img_color is not None:
            img_crop = self.sr.img_color[y1:y2, x1:x2].copy()
        else:
            # Create gray background
            Hc, Wc = y2 - y1, x2 - x1
            img_crop = np.full((Hc, Wc, 3), 128, dtype=np.uint8)

        Hc, Wc = img_crop.shape[:2]

        # Create visualization
        vis = img_crop.copy()

        # Show original mask in blue tint
        base_crop = base[y1:y2, x1:x2]
        vis[base_crop > 0] = (vis[base_crop > 0] * 0.5 + np.array([100, 100, 200]) * 0.5).astype(np.uint8)

        # Draw filled polygon from control points in green
        pts_array = np.array(self._leaf_edit_points, dtype=np.int32)
        pts_crop = pts_array - np.array([x1, y1])

        overlay = np.zeros_like(vis)
        if len(pts_crop) >= 3:
            cv2.fillPoly(overlay, [pts_crop], (100, 200, 100))
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

        # Draw contour line
        if len(pts_crop) >= 2:
            pts_draw = pts_crop.reshape((-1, 1, 2))
            cv2.polylines(vis, [pts_draw], True, (255, 255, 0), 2)

        # Scale to fit canvas
        canvas_w = self._leaf_edit_canvas.winfo_width()
        canvas_h = self._leaf_edit_canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            canvas_w, canvas_h = 700, 600

        scale = min(canvas_w / Wc, canvas_h / Hc) * 0.9
        self._leaf_edit_scale = scale

        disp_w = int(Wc * scale)
        disp_h = int(Hc * scale)
        resized = cv2.resize(vis, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

        offset_x = (canvas_w - disp_w) // 2
        offset_y = (canvas_h - disp_h) // 2
        self._leaf_edit_offset = (offset_x, offset_y)

        # Convert to PhotoImage
        img = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(img)
        self._leaf_edit_canvas._photo = photo

        # Draw image
        self._leaf_edit_canvas.delete("all")
        self._leaf_edit_canvas.create_image(canvas_w // 2, canvas_h // 2, image=photo, anchor="center", tags="img")

        # Draw control points
        point_radius = 6
        for i, pt in enumerate(self._leaf_edit_points):
            cx = (pt[0] - x1) * scale + offset_x
            cy = (pt[1] - y1) * scale + offset_y

            if i == self._leaf_edit_hover_idx:
                color = "#FFD700"
                r = point_radius + 2
            else:
                color = "#00FF00"
                r = point_radius

            self._leaf_edit_canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                fill=color, outline="white", width=2, tags=f"pt_{i}"
            )

        # Update point count
        if hasattr(self, '_leaf_edit_point_label'):
            self._leaf_edit_point_label.configure(text=f"Points: {len(self._leaf_edit_points)}")

    def _leaf_edit_canvas_to_image(self, cx, cy):
        """Convert canvas coords to image coords for leaf editor."""
        y1, y2, x1, x2 = self._leaf_edit_crop_bounds
        offset_x, offset_y = self._leaf_edit_offset
        scale = self._leaf_edit_scale

        crop_x = (cx - offset_x) / scale
        crop_y = (cy - offset_y) / scale

        img_x = crop_x + x1
        img_y = crop_y + y1

        return img_x, img_y

    def _leaf_edit_find_nearest(self, cx, cy, threshold=15):
        """Find nearest control point."""
        y1, y2, x1, x2 = self._leaf_edit_crop_bounds
        offset_x, offset_y = self._leaf_edit_offset
        scale = self._leaf_edit_scale

        min_dist = float('inf')
        nearest_idx = None

        for i, pt in enumerate(self._leaf_edit_points):
            pcx = (pt[0] - x1) * scale + offset_x
            pcy = (pt[1] - y1) * scale + offset_y

            dist = np.sqrt((cx - pcx)**2 + (cy - pcy)**2)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                nearest_idx = i

        return nearest_idx

    def _leaf_edit_toggle_add(self):
        """Toggle add point mode."""
        if self._leaf_edit_add_mode.get():
            self._leaf_edit_delete_mode.set(False)
            self._leaf_edit_canvas.configure(cursor="plus")
        else:
            self._leaf_edit_canvas.configure(cursor="")

    def _leaf_edit_toggle_delete(self):
        """Toggle delete point mode."""
        if self._leaf_edit_delete_mode.get():
            self._leaf_edit_add_mode.set(False)
            self._leaf_edit_canvas.configure(cursor="X_cursor")
        else:
            self._leaf_edit_canvas.configure(cursor="")

    def _leaf_edit_hover(self, event):
        """Handle hover in leaf editor."""
        old_hover = self._leaf_edit_hover_idx
        self._leaf_edit_hover_idx = self._leaf_edit_find_nearest(event.x, event.y)

        if self._leaf_edit_add_mode.get():
            self._leaf_edit_canvas.configure(cursor="plus")
        elif self._leaf_edit_delete_mode.get():
            self._leaf_edit_canvas.configure(cursor="X_cursor")
        elif self._leaf_edit_hover_idx is not None:
            self._leaf_edit_canvas.configure(cursor="hand2")
        else:
            self._leaf_edit_canvas.configure(cursor="")

        if old_hover != self._leaf_edit_hover_idx:
            self._leaf_edit_refresh_canvas()

    def _leaf_edit_click(self, event):
        """Handle click in leaf editor."""
        if self._leaf_edit_add_mode.get():
            self._leaf_edit_add_point(event)
            return
        if self._leaf_edit_delete_mode.get():
            self._leaf_edit_delete_point(event)
            return

        self._leaf_edit_dragging_idx = self._leaf_edit_find_nearest(event.x, event.y)

    def _leaf_edit_drag(self, event):
        """Handle drag in leaf editor."""
        if self._leaf_edit_dragging_idx is None:
            return

        img_x, img_y = self._leaf_edit_canvas_to_image(event.x, event.y)

        img_h, img_w = self._leaf_edit_completed_original.shape
        img_x = max(0, min(img_w - 1, img_x))
        img_y = max(0, min(img_h - 1, img_y))

        self._leaf_edit_points[self._leaf_edit_dragging_idx] = [int(img_x), int(img_y)]
        self._leaf_edit_refresh_canvas()

    def _leaf_edit_release(self, event):
        """Handle release in leaf editor."""
        self._leaf_edit_dragging_idx = None

    def _leaf_edit_add_point(self, event):
        """Add a control point."""
        img_x, img_y = self._leaf_edit_canvas_to_image(event.x, event.y)

        if len(self._leaf_edit_points) < 2:
            self._leaf_edit_points.append([int(img_x), int(img_y)])
        else:
            min_dist = float('inf')
            insert_idx = len(self._leaf_edit_points)

            for i in range(len(self._leaf_edit_points)):
                p1 = np.array(self._leaf_edit_points[i])
                p2 = np.array(self._leaf_edit_points[(i + 1) % len(self._leaf_edit_points)])
                pt = np.array([img_x, img_y])

                line_vec = p2 - p1
                line_len = np.linalg.norm(line_vec)
                if line_len < 1:
                    continue
                line_unit = line_vec / line_len
                proj_len = np.dot(pt - p1, line_unit)
                proj_len = max(0, min(line_len, proj_len))
                closest = p1 + proj_len * line_unit
                dist = np.linalg.norm(pt - closest)

                if dist < min_dist:
                    min_dist = dist
                    insert_idx = i + 1

            self._leaf_edit_points.insert(insert_idx, [int(img_x), int(img_y)])

        self._leaf_edit_refresh_canvas()

    def _leaf_edit_delete_point(self, event):
        """Delete a control point."""
        if len(self._leaf_edit_points) <= 3:
            messagebox.showwarning("Edit Shape", "Need at least 3 points.")
            return

        idx = self._leaf_edit_find_nearest(event.x, event.y, threshold=20)
        if idx is not None:
            del self._leaf_edit_points[idx]
            self._leaf_edit_hover_idx = None
            self._leaf_edit_refresh_canvas()

    def _leaf_edit_smooth(self):
        """Smooth the contour."""
        if len(self._leaf_edit_points) < 5:
            return

        pts = np.array(self._leaf_edit_points, dtype=np.float32)
        n = len(pts)

        smoothed = []
        for i in range(n):
            prev_pt = pts[(i - 1) % n]
            curr_pt = pts[i]
            next_pt = pts[(i + 1) % n]
            new_pt = (prev_pt * 0.25 + curr_pt * 0.5 + next_pt * 0.25)
            smoothed.append([int(new_pt[0]), int(new_pt[1])])

        self._leaf_edit_points = smoothed
        self._leaf_edit_refresh_canvas()

    def _leaf_edit_reset(self):
        """Reset to original completed shape."""
        completed = self._leaf_edit_completed_original
        contours, _ = cv2.findContours(completed.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            epsilon = 0.008 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            pts = approx.reshape(-1, 2).tolist()
            if len(pts) > 60:
                step = len(pts) // 50
                pts = pts[::step]
            self._leaf_edit_points = pts
            self._leaf_edit_refresh_canvas()

    def _leaf_edit_apply(self):
        """Apply the edited shape to pending completion."""
        if len(self._leaf_edit_points) < 3:
            messagebox.showwarning("Edit Shape", "Need at least 3 points.")
            return

        # Create new mask from contour
        pts = np.array(self._leaf_edit_points, dtype=np.int32)
        new_completed = np.zeros_like(self._leaf_edit_completed_original, dtype=np.uint8)
        cv2.fillPoly(new_completed, [pts], 1)

        # Update pending completion
        base = self._leaf_edit_base
        added = np.logical_and(new_completed, ~base)
        completed_area = int(new_completed.sum())
        added_area = int(added.sum())

        self._pending_leaf_completion['completed'] = new_completed.astype(bool)
        self._pending_leaf_completion['added_area'] = added_area
        self._pending_leaf_completion['completed_area'] = completed_area

        # Update the completed canvas preview
        zoom = getattr(self, '_leaf_preview_zoom', 1.0)
        self._draw_mask_on_canvas(self._leaf_comp_canvas, base, 250, 250,
                                  completed_mask=new_completed.astype(bool), zoom=zoom)

        # Update stats
        base_area = int(base.sum())
        if added_area > 0:
            self._leaf_stats_label.configure(
                text=f"Original: {base_area:,} px  →  Completed: {completed_area:,} px  (+{added_area:,} px)")
        elif added_area < 0:
            self._leaf_stats_label.configure(
                text=f"Original: {base_area:,} px  →  Reduced: {completed_area:,} px  ({added_area:,} px)")
        else:
            self._leaf_stats_label.configure(
                text=f"No change ({completed_area:,} px)")

        # Close dialog
        self._leaf_edit_dialog.destroy()
        self.set_status("Shape edited - click 'Update Mask' to apply", "success")

    def _leaf_edit_cancel(self):
        """Cancel leaf shape editing."""
        self._leaf_edit_dialog.destroy()
        self.set_status("Shape edit cancelled", "info")

    def _leaf_zoom_by(self, factor: float):
        """Zoom the leaf completion preview by a factor."""
        self._leaf_preview_zoom = getattr(self, '_leaf_preview_zoom', 1.0) * factor
        self._leaf_preview_zoom = max(0.25, min(self._leaf_preview_zoom, 5.0))  # bounds: 25% - 500%
        self._leaf_zoom_label.configure(text=f"{int(self._leaf_preview_zoom * 100)}%")
        self._refresh_leaf_previews()

    def _leaf_zoom_fit(self):
        """Reset leaf completion preview zoom to fit."""
        self._leaf_preview_zoom = 1.0
        self._leaf_zoom_label.configure(text="100%")
        self._refresh_leaf_previews()

    def _leaf_canvas_mousewheel(self, event):
        """Handle mousewheel zoom on leaf completion canvases."""
        # macOS uses event.delta, Windows/Linux use different values
        if event.delta > 0:
            self._leaf_zoom_by(1.1)
        else:
            self._leaf_zoom_by(0.9)

    def _refresh_leaf_previews(self):
        """Refresh both leaf completion preview canvases with current zoom."""
        if not hasattr(self, '_leaf_preview_base') or self._leaf_preview_base is None:
            # No preview active, try to use the original mask
            if hasattr(self, '_leaf_preview_orig_mask') and self._leaf_preview_orig_mask is not None:
                self._draw_mask_on_canvas(self._leaf_orig_canvas, self._leaf_preview_orig_mask, 250, 250,
                                          zoom=self._leaf_preview_zoom)
            return

        base = self._leaf_preview_base

        # Redraw original canvas
        self._draw_mask_on_canvas(self._leaf_orig_canvas, base, 250, 250, zoom=self._leaf_preview_zoom)

        # Redraw completed canvas if we have pending completion
        if self._pending_leaf_completion:
            completed = self._pending_leaf_completion['completed']
            self._draw_mask_on_canvas(self._leaf_comp_canvas, base, 250, 250,
                                      completed_mask=completed, zoom=self._leaf_preview_zoom)

    def _leaf_drag_start(self, event):
        """Start dragging the fitted shape."""
        if not self._pending_leaf_completion:
            return
        self._leaf_drag_start_pos = (event.x, event.y)

    def _leaf_drag_move(self, event):
        """Handle dragging to move the fitted shape."""
        if not self._pending_leaf_completion or not self._leaf_drag_start_pos:
            return

        # Calculate delta in canvas coordinates
        dx = event.x - self._leaf_drag_start_pos[0]
        dy = event.y - self._leaf_drag_start_pos[1]

        # Convert canvas delta to mask pixel delta (accounting for zoom and scale)
        base = self._leaf_preview_base
        if base is None:
            return

        h, w = base.shape
        canvas_w, canvas_h = 250, 250
        zoom = getattr(self, '_leaf_preview_zoom', 1.0)
        fit_scale = min(canvas_w / w, canvas_h / h) * 0.9
        scale = fit_scale * zoom

        # Convert canvas pixels to mask pixels
        mask_dx = int(dx / scale)
        mask_dy = int(dy / scale)

        # Update offset
        self._leaf_shape_offset[0] += mask_dx
        self._leaf_shape_offset[1] += mask_dy

        # Update drag start position
        self._leaf_drag_start_pos = (event.x, event.y)

        # Update offset label
        self._leaf_offset_label.configure(text=f"({self._leaf_shape_offset[0]}, {self._leaf_shape_offset[1]})")

        # Regenerate completed mask with new offset and update preview
        self._update_shape_with_offset()

    def _leaf_drag_end(self, event):
        """End dragging."""
        self._leaf_drag_start_pos = None

    def _leaf_reset_position(self):
        """Reset the shape position offset to (0, 0)."""
        self._leaf_shape_offset = [0, 0]
        self._leaf_offset_label.configure(text="(0, 0)")

        # Regenerate preview if we have one active
        if self._pending_leaf_completion and hasattr(self, '_leaf_preview_base'):
            self._update_shape_with_offset()

    def _leaf_rotate(self, degrees: float):
        """Rotate the shape by the specified degrees."""
        self._leaf_rotation_offset = getattr(self, '_leaf_rotation_offset', 0) + degrees
        # Normalize to -180 to 180
        while self._leaf_rotation_offset > 180:
            self._leaf_rotation_offset -= 360
        while self._leaf_rotation_offset < -180:
            self._leaf_rotation_offset += 360

        self._leaf_rotation_label.configure(text=f"{int(self._leaf_rotation_offset)}°")

        # Regenerate preview if we have one active
        if self._pending_leaf_completion and hasattr(self, '_leaf_preview_base'):
            self._update_shape_with_offset()

    def _leaf_reset_rotation(self):
        """Reset the shape rotation to 0°."""
        self._leaf_rotation_offset = 0
        self._leaf_rotation_label.configure(text="0°")

        # Regenerate preview if we have one active
        if self._pending_leaf_completion and hasattr(self, '_leaf_preview_base'):
            self._update_shape_with_offset()

    def _update_shape_with_offset(self):
        """Update the completed mask with the current offset and rotation."""
        if not hasattr(self, '_leaf_preview_base') or self._leaf_preview_base is None:
            return

        base = self._leaf_preview_base
        shape_type = self.shape_extend_var.get()
        offset = tuple(self._leaf_shape_offset)
        rotation = getattr(self, '_leaf_rotation_offset', 0)

        # Fit shape with current scale, offset, and rotation
        completed = self._fit_shape_to_mask(base, shape_type,
                                            scale=self._leaf_scale_factor,
                                            offset=offset,
                                            rotation=rotation)
        if completed is None:
            return

        # Calculate stats
        added = np.logical_and(completed, ~base)
        removed = np.logical_and(base, ~completed)
        base_area = int(base.sum())
        completed_area = int(completed.sum())
        added_area = int(added.sum()) - int(removed.sum())

        # Update the completed canvas
        zoom = getattr(self, '_leaf_preview_zoom', 1.0)
        self._draw_mask_on_canvas(self._leaf_comp_canvas, base, 250, 250,
                                  completed_mask=completed, zoom=zoom)

        # Update stats
        if added_area > 0:
            self._leaf_stats_label.configure(
                text=f"Original: {base_area:,} px  →  Completed: {completed_area:,} px  (+{added_area:,} px)")
        elif added_area < 0:
            self._leaf_stats_label.configure(
                text=f"Original: {base_area:,} px  →  Reduced: {completed_area:,} px  ({added_area:,} px)")
        else:
            self._leaf_stats_label.configure(
                text=f"Original: {base_area:,} px  →  Completed: {completed_area:,} px")

        # Update pending completion
        self._pending_leaf_completion = {
            'idx': self._leaf_preview_idx,
            'completed': completed,
            'base': base,
            'shape_type': shape_type,
            'added_area': added_area,
            'completed_area': completed_area,
        }

    def _draw_leaf_shape_on_canvas(self, canvas, shape: str, w: int, h: int, color: str = None):
        """Draw a leaf shape icon on a canvas of given size."""
        if color is None:
            color = self.colors.get('accent', '#75E6DA')

        canvas.delete("all")
        cx, cy = w // 2, h // 2
        margin = int(w * 0.12)

        if shape == "Ellipse":
            # Simple oval
            canvas.create_oval(margin, margin + 2, w - margin, h - margin - 2, fill=color, outline="")

        elif shape == "Orbicular":
            # Round with stem notch at bottom
            canvas.create_oval(margin, margin, w - margin, h - margin - 4, fill=color, outline="")
            # Small notch for stem
            notch_w = int(w * 0.12)
            canvas.create_rectangle(cx - notch_w, h - margin - 2, cx + notch_w, h - 2,
                                    fill=self.colors['bg_dark'], outline="")

        elif shape == "Ovate":
            # Egg shape, wider at base
            pts = [
                (cx, margin),           # top point
                (margin + 2, cy - 2),   # left upper
                (margin, h - margin - 4),  # left lower (wider)
                (cx, h - margin),       # bottom
                (w - margin, h - margin - 4),  # right lower (wider)
                (w - margin - 2, cy - 2),  # right upper
            ]
            canvas.create_polygon(pts, fill=color, outline="", smooth=True)

        elif shape == "Obovate":
            # Egg shape, wider at top
            pts = [
                (cx, margin),           # top
                (margin, cy - 4),       # left upper (wider)
                (margin + 4, h - margin - 4),  # left lower
                (cx, h - margin),       # bottom point
                (w - margin - 4, h - margin - 4),  # right lower
                (w - margin, cy - 4),   # right upper (wider)
            ]
            canvas.create_polygon(pts, fill=color, outline="", smooth=True)

        elif shape == "Lanceolate":
            # Narrow, pointed - lance shape
            narrow = int(w * 0.22)
            pts = [
                (cx, margin),           # top point
                (cx - narrow, cy - 4),  # left
                (cx - narrow + 2, h - margin - 4),  # left lower
                (cx, h - margin),       # bottom point
                (cx + narrow - 2, h - margin - 4),  # right lower
                (cx + narrow, cy - 4),  # right
            ]
            canvas.create_polygon(pts, fill=color, outline="", smooth=True)

        elif shape == "Ensiform":
            # Sword shape - very narrow, long
            narrow = int(w * 0.12)
            pts = [
                (cx, margin),           # top point (sharp)
                (cx - narrow - 2, cy - 8),  # left upper
                (cx - narrow, h - margin - 4),  # left lower
                (cx, h - margin),       # bottom
                (cx + narrow, h - margin - 4),  # right lower
                (cx + narrow + 2, cy - 8),  # right upper
            ]
            canvas.create_polygon(pts, fill=color, outline="", smooth=True)

    def _on_leaf_shape_selected(self, event=None):
        """Handle leaf shape selection - update gallery highlights and preview."""
        selected = self.shape_extend_var.get()

        # Update gallery canvas borders to show selection
        if hasattr(self, '_leaf_shape_canvases'):
            for shape, canvas in self._leaf_shape_canvases.items():
                if shape == selected:
                    canvas.configure(highlightbackground=self.colors['accent'],
                                     highlightthickness=3)
                else:
                    canvas.configure(highlightbackground=self.colors['bg_medium'],
                                     highlightthickness=2)

        # Update the selected shape label
        if hasattr(self, '_selected_shape_label'):
            self._selected_shape_label.configure(text=selected)

        # Update the small preview canvas
        self._update_leaf_shape_preview()

    def _update_leaf_shape_preview(self, event=None):
        """Draw the selected leaf shape on the small preview canvas."""
        if not hasattr(self, '_leaf_shape_canvas'):
            return

        shape = self.shape_extend_var.get()
        self._draw_leaf_shape_on_canvas(self._leaf_shape_canvas, shape, 32, 32)

    def _fit_shape_to_mask(self, mask: np.ndarray, shape_type: str, scale: float = 1.0,
                           offset: tuple = (0, 0), rotation: float = 0) -> np.ndarray | None:
        """Fit a geometric shape to the mask and return the completed mask.

        Alignment strategy (automatic, no user effort needed):
          1. CENTROID  — shape center = pixel centroid of the mask
          2. ORIENTATION — shape angle = PCA major axis of the mask pixels
          3. SCALE — shape axes = PCA major/minor lengths × scale factor

        Args:
            mask: Binary mask (bool array)
            shape_type: Shape name string (matches LEAF_SHAPES in tab_leaf_completion.py)
            scale: Scale factor (1.0 = fit tightly to mask extents)
            offset: (dx, dy) manual drag offset in pixels
            rotation: Additional rotation in degrees (from rotate buttons)

        Returns:
            Completed binary mask (bool) or None if fitting failed
        """
        H, W = mask.shape[:2]
        completed = np.zeros((H, W), dtype=np.uint8)

        # ── 1. CENTROID — pixel centre of mass ───────────────────────────────
        ys, xs = np.nonzero(mask)
        if len(xs) < 5:
            return None
        cx = float(xs.mean()) + offset[0]
        cy = float(ys.mean()) + offset[1]

        # ── 2. PCA ORIENTATION — major axis angle ────────────────────────────
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        pts_c = pts - pts.mean(axis=0)
        cov = (pts_c.T @ pts_c) / max(1, len(pts_c) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]   # largest eigenvalue first
        vmaj = eigvecs[:, order[0]]          # major axis direction vector

        # cv2.ellipse angle convention: degrees from vertical (Y-axis), clockwise
        pca_angle = float(np.degrees(np.arctan2(vmaj[0], vmaj[1]))) + rotation

        # ── 3. SCALE — PCA-based axis lengths ────────────────────────────────
        proj_maj = pts_c @ vmaj
        proj_min = pts_c @ np.array([-vmaj[1], vmaj[0]])
        half_maj = float((proj_maj.max() - proj_maj.min()) / 2.0) * scale
        half_min = float((proj_min.max() - proj_min.min()) / 2.0) * scale
        half_maj = max(half_maj, 3.0)
        half_min = max(half_min, 3.0)

        # cv2 ellipse axes = (half_minor, half_major) when angle aligns with Y
        axes = (half_min, half_maj)
        center_i = (int(round(cx)), int(round(cy)))

        # ── Helper: draw rotated polygon on `completed` ───────────────────────
        def _rot_pts(local_pts_nm):
            """Convert normalised [-1..1] local coords → pixel coords.
            local_pts_nm: list of (u, v) where u=minor axis, v=major axis.
            """
            ang_r = np.radians(pca_angle)
            cos_a, sin_a = np.cos(ang_r), np.sin(ang_r)
            # major axis direction in image space
            maj_x =  sin_a;  maj_y = cos_a
            min_x =  cos_a;  min_y = -sin_a
            out = []
            for u, v in local_pts_nm:
                px = cx + u * half_min * min_x + v * half_maj * maj_x
                py = cy + u * half_min * min_y + v * half_maj * maj_y
                out.append((int(round(px)), int(round(py))))
            return np.array(out, dtype=np.int32)

        s = shape_type.lower().replace("-", "").replace(" ", "")

        # ── Draw each shape ───────────────────────────────────────────────────

        if s in ("ellipse", "elliptical"):
            cv2.ellipse(completed, center_i, (int(half_min), int(half_maj)),
                        pca_angle, 0, 360, 255, -1)

        elif s == "orbicular":
            r = int(max(half_maj, half_min))
            cv2.circle(completed, center_i, r, 255, -1)

        elif s == "ovate":
            # Egg: wide rounded base, tapered apex
            # Shift centroid slightly toward base so widest part ~40% from bottom
            pts_poly = _rot_pts([
                ( 0.0,  -1.00),  # apex
                ( 0.55, -0.40),  # upper right
                ( 0.90,  0.10),  # widest right
                ( 0.72,  0.65),  # lower right
                ( 0.28,  1.00),  # base right
                ( 0.0,   0.85),  # base centre
                (-0.28,  1.00),
                (-0.72,  0.65),
                (-0.90,  0.10),
                (-0.55, -0.40),
            ])
            cv2.fillPoly(completed, [pts_poly], 255)

        elif s == "obovate":
            # Reversed egg: wide top, narrow base
            pts_poly = _rot_pts([
                ( 0.0,  -1.00),  # narrow tip top
                ( 0.40, -0.55),
                ( 0.88, -0.10),  # widest right (above mid)
                ( 0.75,  0.50),
                ( 0.35,  1.00),  # narrow base right
                ( 0.0,   0.80),
                (-0.35,  1.00),
                (-0.75,  0.50),
                (-0.88, -0.10),
                (-0.40, -0.55),
            ])
            cv2.fillPoly(completed, [pts_poly], 255)

        elif s == "cordate":
            # Heart: two lobes at top, pointed tip at base
            # Left lobe
            lobe_cx = cx - half_min * 0.30
            lobe_cy = cy - half_maj * 0.35
            lobe_r  = (half_min * 0.58, half_maj * 0.45)
            cv2.ellipse(completed,
                        (int(round(lobe_cx)), int(round(lobe_cy))),
                        (int(lobe_r[0]), int(lobe_r[1])),
                        pca_angle, 0, 360, 255, -1)
            # Right lobe
            lobe_cx2 = cx + half_min * 0.30
            cv2.ellipse(completed,
                        (int(round(lobe_cx2)), int(round(lobe_cy))),
                        (int(lobe_r[0]), int(lobe_r[1])),
                        pca_angle, 0, 360, 255, -1)
            # Lower body to pointed tip
            pts_poly = _rot_pts([
                (-0.90, -0.20),
                ( 0.90, -0.20),
                ( 0.0,   1.00),   # pointed tip
                ( 0.0,  -0.10),   # inner notch
            ])
            cv2.fillPoly(completed, [pts_poly], 255)

        elif s == "reniform":
            # Kidney: wide, shallow, concave at base centre
            pts_poly = _rot_pts([
                ( 0.0,  -0.65),
                ( 0.60, -0.90),
                ( 0.95, -0.20),
                ( 0.88,  0.55),
                ( 0.35,  0.90),
                ( 0.08,  0.60),   # base indentation right
                ( 0.0,   0.45),   # base centre notch
                (-0.08,  0.60),
                (-0.35,  0.90),
                (-0.88,  0.55),
                (-0.95, -0.20),
                (-0.60, -0.90),
            ])
            cv2.fillPoly(completed, [pts_poly], 255)

        elif s == "lanceolate":
            # Lance: widest at ~35% from base, sharp both ends
            pts_poly = _rot_pts([
                ( 0.0,  -1.00),   # pointed apex
                ( 0.52, -0.10),   # right shoulder (widest)
                ( 0.28,  1.00),   # base right
                ( 0.0,   0.75),   # rounded base
                (-0.28,  1.00),
                (-0.52, -0.10),
            ])
            cv2.fillPoly(completed, [pts_poly], 255)

        elif s == "oblanceolate":
            # Reversed lance: widest near apex, narrower base
            pts_poly = _rot_pts([
                ( 0.0,  -1.00),   # narrow apex
                ( 0.50, -0.45),   # right shoulder (widest, near top)
                ( 0.20,  1.00),   # base right (narrow)
                ( 0.0,   0.85),
                (-0.20,  1.00),
                (-0.50, -0.45),
            ])
            cv2.fillPoly(completed, [pts_poly], 255)

        elif s == "oblong":
            # Rectangular body, parallel sides, rounded ends
            pts_poly = _rot_pts([
                (-0.50, -1.00),
                ( 0.0,  -0.88),
                ( 0.50, -1.00),
                ( 0.50,  1.00),
                ( 0.0,   0.88),
                (-0.50,  1.00),
            ])
            cv2.fillPoly(completed, [pts_poly], 255)

        elif s == "linear":
            # Very narrow strap, parallel sides, rounded ends
            pts_poly = _rot_pts([
                (-0.18, -1.00),
                ( 0.0,  -0.90),
                ( 0.18, -1.00),
                ( 0.18,  1.00),
                ( 0.0,   0.90),
                (-0.18,  1.00),
            ])
            cv2.fillPoly(completed, [pts_poly], 255)

        elif s in ("ensiform", "sword"):
            # Sword: narrow uniform width, sharp tip
            pts_poly = _rot_pts([
                (-0.14, -1.00),
                ( 0.0,  -0.88),
                ( 0.14, -1.00),
                ( 0.10,  0.90),
                ( 0.0,   1.00),   # sharp tip
                (-0.10,  0.90),
            ])
            cv2.fillPoly(completed, [pts_poly], 255)

        elif s in ("cuneate", "wedge"):
            # Wedge: narrow pointed base, wide flat top
            pts_poly = _rot_pts([
                (-0.92, -0.85),
                ( 0.92, -0.85),
                ( 0.92, -1.00),
                (-0.92, -1.00),
                ( 0.0,   1.00),   # pointed base
            ])
            cv2.fillPoly(completed, [pts_poly], 255)

        elif s == "spathulate":
            # Spoon: narrow stalk, wide round blade at top
            # Stalk (lower half)
            pts_stalk = _rot_pts([
                (-0.18,  0.10),
                ( 0.18,  0.10),
                ( 0.18,  1.00),
                (-0.18,  1.00),
            ])
            cv2.fillPoly(completed, [pts_stalk], 255)
            # Round blade (upper half)
            blade_cx = int(round(cx + (-half_maj * 0.35) * np.sin(np.radians(pca_angle))))
            blade_cy = int(round(cy + (-half_maj * 0.35) * np.cos(np.radians(pca_angle))))
            cv2.ellipse(completed,
                        (blade_cx, blade_cy),
                        (int(half_min * 0.88), int(half_maj * 0.55)),
                        pca_angle, 0, 360, 255, -1)

        elif s in ("deltoid", "triangular"):
            # Triangle: wide base, pointed apex
            pts_poly = _rot_pts([
                ( 0.0,  -1.00),   # apex
                ( 0.95,  1.00),   # base right
                ( 0.0,   0.70),   # slight base concavity
                (-0.95,  1.00),
            ])
            cv2.fillPoly(completed, [pts_poly], 255)

        elif s == "rhomboid":
            # Diamond: widest at mid-height
            pts_poly = _rot_pts([
                ( 0.0,  -1.00),
                ( 0.88,  0.0 ),
                ( 0.0,   1.00),
                (-0.88,  0.0 ),
            ])
            cv2.fillPoly(completed, [pts_poly], 255)

        elif s in ("sagittate", "sagittata"):
            # Arrowhead: pointed apex, basal lobes pointing downward
            pts_poly = _rot_pts([
                ( 0.0,  -1.00),   # apex
                ( 0.58, -0.30),   # right shoulder
                ( 0.30,  0.15),   # right waist
                ( 0.72,  1.00),   # right lobe tip
                ( 0.20,  0.55),   # right lobe inner
                ( 0.0,   0.45),   # base notch
                (-0.20,  0.55),
                (-0.72,  1.00),
                (-0.30,  0.15),
                (-0.58, -0.30),
            ])
            cv2.fillPoly(completed, [pts_poly], 255)

        elif s in ("hastate", "alabardata"):
            # Halberd: pointed apex, basal lobes pointing sideways
            pts_poly = _rot_pts([
                ( 0.0,  -1.00),   # apex
                ( 0.50, -0.35),   # right shoulder
                ( 0.28,  0.10),   # right waist (pinch)
                ( 0.95,  0.55),   # right lobe tip (sideways)
                ( 0.28,  1.00),   # right lobe base
                ( 0.0,   0.75),   # base centre
                (-0.28,  1.00),
                (-0.95,  0.55),
                (-0.28,  0.10),
                (-0.50, -0.35),
            ])
            cv2.fillPoly(completed, [pts_poly], 255)

        else:
            # Fallback: plain ellipse
            cv2.ellipse(completed, center_i, (int(half_min), int(half_maj)),
                        pca_angle, 0, 360, 255, -1)

        # ── Merge with original mask (never remove existing pixels) ──────────
        completed_bool = (completed > 127)
        completed_bool = np.logical_or(completed_bool, mask)
        return completed_bool

    def on_complete_selected_mask(self):
        """Complete selected mask using shape completion model (no SAM).

        The extended/completed area is filled with the average RGB color of the image.
        """
        sel = list(self.lb.curselection())
        if len(sel) != 1:
            messagebox.showwarning("Pick one", "Select exactly one mask to complete.")
            return
        if self.shape_model is None:
            messagebox.showwarning("Shape Completion", "Load the shape completion model first.\n\nUse 'Load Model' button or go to Training > Shape Completion tab.")
            return

        i = sel[0]
        base = self.sr.masks[i]["segmentation"].astype(bool)

        # avoid invading other masks
        forbid = np.zeros_like(base, dtype=bool)
        for j, mm in enumerate(self.sr.masks):
            if j != i:
                forbid |= mm["segmentation"].astype(bool)

        pred = self._predict_extend_completion(base, forbid)
        if pred is None or not pred.any():
            messagebox.showwarning("Shape Completion", "Prediction failed - model returned empty mask.")
            return

        added = np.logical_and(pred, ~base)

        # Debug info
        base_area = int(base.sum())
        pred_area = int(pred.sum())
        added_area = int(added.sum())
        print(f"[Shape Completion] Base area: {base_area}, Predicted area: {pred_area}, Added: {added_area}")

        # DEBUG: Save comparison image showing input vs output
        debug_dir = Path("/tmp/shape_completion_debug")
        debug_dir.mkdir(exist_ok=True)

        # Get bbox of base mask
        ys, xs = np.nonzero(base)
        if len(ys) > 0:
            by1, by2 = int(ys.min()), int(ys.max()) + 1
            bx1, bx2 = int(xs.min()), int(xs.max()) + 1
            pad = 20
            by1, bx1 = max(0, by1-pad), max(0, bx1-pad)
            by2, bx2 = min(base.shape[0], by2+pad), min(base.shape[1], bx2+pad)

            # Crop both masks to same region
            base_crop = base[by1:by2, bx1:bx2].astype(np.uint8) * 255
            pred_crop = pred[by1:by2, bx1:bx2].astype(np.uint8) * 255

            # Create side-by-side comparison
            h, w = base_crop.shape
            comparison = np.zeros((h, w * 3 + 20, 3), dtype=np.uint8)
            comparison[:, :, :] = 50  # dark gray background

            # Input (green)
            comparison[:h, :w, 1] = base_crop
            # Output (magenta)
            comparison[:h, w+10:w*2+10, 0] = pred_crop
            comparison[:h, w+10:w*2+10, 2] = pred_crop
            # Difference (cyan = added, red = removed)
            diff_added = np.logical_and(pred[by1:by2, bx1:bx2], ~base[by1:by2, bx1:bx2])
            diff_removed = np.logical_and(base[by1:by2, bx1:bx2], ~pred[by1:by2, bx1:bx2])
            comparison[:h, w*2+20:, 1] = pred_crop  # base green
            comparison[:h, w*2+20:, 0][diff_added] = 255  # added = cyan
            comparison[:h, w*2+20:, 2][diff_added] = 255
            comparison[:h, w*2+20:, 0][diff_removed] = 255  # removed = red
            comparison[:h, w*2+20:, 1][diff_removed] = 0
            comparison[:h, w*2+20:, 2][diff_removed] = 0

            cv2.imwrite(str(debug_dir / "comparison.png"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(debug_dir / "input_base.png"), base_crop)
            cv2.imwrite(str(debug_dir / "output_pred.png"), pred_crop)
            print(f"[DEBUG] Saved comparison to {debug_dir}/comparison.png")

        if added_area == 0:
            # Model didn't extend the mask - might already be complete
            messagebox.showinfo("Shape Completion",
                f"Model prediction has same shape as input.\n\n"
                f"Base area: {base_area} px\n"
                f"Predicted area: {pred_area} px\n"
                f"Added: {added_area} px\n\n"
                "The mask may already be complete, or the model doesn't see anything to extend.")

        # Calculate average RGB color from the ORIGINAL SEGMENT area for filling the extended part
        avg_color = None
        if self.sr is not None and self.sr.img_color is not None:
            img_arr = self.sr.img_color
            if img_arr.ndim == 3 and img_arr.shape[2] >= 3:
                # Get pixels only within the original mask (base)
                mask_pixels = img_arr[base]  # shape: (N, 3) where N is number of pixels in mask
                if len(mask_pixels) > 0:
                    avg_r = int(mask_pixels[:, 0].mean())
                    avg_g = int(mask_pixels[:, 1].mean())
                    avg_b = int(mask_pixels[:, 2].mean())
                    avg_color = (avg_r, avg_g, avg_b)
                    print(f"[Shape Completion] Fill color (avg RGB of segment): {avg_color}")

        ys, xs = np.nonzero(pred)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bbox = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]

        self.sr.masks.append({
            "segmentation": pred.astype(np.uint8),
            "bbox": bbox,
            "area": float(pred.sum()),
            "meta": {
                "predicted": True,
                "pred_mode": "completion",
                "extended_bool": added.astype(np.uint8),
                "fill_color": avg_color,  # Average RGB color for the extended area
                "base_area": base_area,
                "added_area": added_area,
            }
        })
        self._rebuild_mask_list()
        self.lb.selection_clear(0, tk.END)
        self.lb.selection_set(tk.END)
        self.on_select_mask()

        if added_area > 0:
            self.set_status(f"Shape completed: +{added_area} pixels", "success")

    # ════════════════════════════════════════════════════════════════════════
    # LEAF UNFOLDING METHODS
    # ════════════════════════════════════════════════════════════════════════

    def _preview_leaf_unfolding(self):
        """Preview unfolding of selected masks."""
        sel = list(self.lb.curselection())
        if len(sel) < 2:
            messagebox.showwarning("Leaf Unfolding", "Select at least 2 masks to unfold.")
            return
        if self.sr is None or not self.sr.masks:
            messagebox.showwarning("Leaf Unfolding", "No masks available.")
            return

        # Reset state
        self._unfold_rotation_angle = 0
        self._unfold_offset = [0, 0]
        self._unfold_flip_h = False
        self._unfold_flip_v = False
        self._unfold_angle_label.configure(text="0°")
        self._unfold_offset_label.configure(text="(0, 0)")

        # Store selected masks AND their image data
        self._unfold_masks = []
        img = self.sr.img_color  # Original image
        for idx in sel:
            mask = self.sr.masks[idx]["segmentation"].astype(bool)
            # Extract image pixels for this mask (full image, masked)
            img_masked = img.copy()
            img_masked[~mask] = 0  # Zero out pixels outside mask
            self._unfold_masks.append({
                'idx': idx,
                'mask': mask.copy(),
                'original': mask.copy(),
                'img': img_masked,  # Image with only this mask's pixels
                'img_original': img_masked.copy(),  # Keep original for reset
            })

        # Populate mask selection radio buttons
        for widget in self._unfold_mask_frame.winfo_children():
            widget.destroy()

        self._unfold_mask_var.set("mask_0")  # Default to first mask
        for i, m in enumerate(self._unfold_masks):
            area = int(m['mask'].sum())
            rb = ttk.Radiobutton(self._unfold_mask_frame, text=f"Mask {i+1} ({area:,} px)",
                                  variable=self._unfold_mask_var, value=f"mask_{i}",
                                  command=self._unfold_update_preview)
            rb.pack(anchor="w")

        # Draw original (combined masks with different colors)
        self._draw_unfold_original()

        # Draw unfolded preview
        self._unfold_update_preview()

        # Enable update button
        self._unfold_update_btn.configure(state="normal")

        # Update stats
        total_area = sum(int(m['mask'].sum()) for m in self._unfold_masks)
        self._unfold_stats_label.configure(text=f"Selected {len(sel)} masks, total area: {total_area:,} px")

        self.set_status(f"Previewing unfold of {len(sel)} masks", "info")

    def _draw_unfold_original(self):
        """Draw the original masks on the unfold original canvas."""
        if not self._unfold_masks:
            return

        # Get combined bounding box
        combined = np.zeros_like(self._unfold_masks[0]['original'], dtype=bool)
        for m in self._unfold_masks:
            combined |= m['original']

        ys, xs = np.nonzero(combined)
        if len(ys) == 0:
            return

        h, w = combined.shape

        # Combine actual image data from all masks
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        for m in self._unfold_masks:
            # Each mask's img_original has pixels only where mask is True
            vis[m['original']] = m['img_original'][m['original']]

        # Crop to bounding box with padding
        pad = 20
        y1, y2 = max(0, ys.min() - pad), min(h, ys.max() + pad)
        x1, x2 = max(0, xs.min() - pad), min(w, xs.max() + pad)
        vis_crop = vis[y1:y2, x1:x2].copy()
        combined_crop = combined[y1:y2, x1:x2]

        # Add checkerboard background
        tile = 16
        Hc, Wc = vis_crop.shape[:2]
        if Hc > 0 and Wc > 0:
            chk = np.indices((Hc, Wc)).sum(axis=0) // tile
            bg = np.where((chk % 2)[..., None], 200, 160).astype(np.uint8)
            bg = np.dstack([bg, bg, bg])
            vis_crop = np.where(combined_crop[..., None], vis_crop, bg)

        self._unfold_crop_bounds = (y1, y2, x1, x2)

        # Draw on canvas
        self._draw_rgb_on_canvas(self._unfold_orig_canvas, vis_crop, 250, 250)

    def _unfold_update_preview(self):
        """
        Update the unfolded preview.

        Core concept: a folded leaf is its mirror image across the fold line.
        We detect the fold line as the junction boundary between the two masks,
        fit a line through it, then reflect the selected (folded) mask across
        that line.  The result is placed beside the stationary mask — giving the
        realistic unfolded appearance.

        The rotation control now fine-tunes the fold-line angle.
        The offset control lets you shift the mirrored piece.
        """
        if not self._unfold_masks:
            return

        mask_idx_str = self._unfold_mask_var.get()
        try:
            fold_idx = int(mask_idx_str.split("_")[1])
        except (IndexError, ValueError):
            fold_idx = 0

        h, w = self._unfold_masks[0]['original'].shape
        result_mask = np.zeros((h, w), dtype=bool)
        result_img  = np.zeros((h, w, 3), dtype=np.uint8)

        fold_mask   = self._unfold_masks[fold_idx]['original']
        static_mask = np.zeros((h, w), dtype=bool)
        for i, m in enumerate(self._unfold_masks):
            if i != fold_idx:
                static_mask |= m['original']

        # ── 1. Find fold line ──────────────────────────────────────────────────
        # Dilate both masks and find their overlap zone (junction)
        k = np.ones((7, 7), np.uint8)
        d_fold   = cv2.dilate(fold_mask.astype(np.uint8),   k, iterations=4)
        d_static = cv2.dilate(static_mask.astype(np.uint8), k, iterations=4)
        junction = np.logical_and(d_fold > 0, d_static > 0)

        if junction.any():
            jys, jxs = np.nonzero(junction)
            pts = np.stack([jxs, jys], axis=1).astype(np.float32)
            mu  = pts.mean(axis=0)
            pts_c = pts - mu
            _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
            fold_dir = Vt[0]   # unit vector along fold line
        else:
            # Fallback: use PCA major axis of the folded mask itself
            fys, fxs = np.nonzero(fold_mask)
            pts = np.stack([fxs, fys], axis=1).astype(np.float32)
            mu  = pts.mean(axis=0)
            pts_c = pts - mu
            _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
            fold_dir = Vt[0]
            mu = pts.mean(axis=0)

        # ── 2. Apply user rotation tweak to fold line angle ───────────────────
        extra_rad = np.radians(self._unfold_rotation_angle)
        cos_e, sin_e = np.cos(extra_rad), np.sin(extra_rad)
        fold_dir = np.array([
            fold_dir[0] * cos_e - fold_dir[1] * sin_e,
            fold_dir[0] * sin_e + fold_dir[1] * cos_e,
        ])
        fold_dir /= np.linalg.norm(fold_dir) + 1e-9

        # ── 3. Reflect the folded mask across the fold line ───────────────────
        pivot = mu.astype(float)          # point on the fold line
        nx, ny = -fold_dir[1], fold_dir[0]  # normal to fold line

        # Build reflection matrix:  P' = P - 2*(P-pivot)·n̂ * n̂
        fys2, fxs2 = np.nonzero(fold_mask)
        if len(fxs2) == 0:
            return

        coords = np.stack([fxs2.astype(float), fys2.astype(float)], axis=1)
        diff   = coords - pivot
        proj   = (diff[:, 0] * nx + diff[:, 1] * ny)[:, None]
        reflected = coords - 2 * proj * np.array([nx, ny])

        # Apply user offset
        reflected[:, 0] += self._unfold_offset[0]
        reflected[:, 1] += self._unfold_offset[1]

        rx = np.clip(np.round(reflected[:, 0]).astype(int), 0, w - 1)
        ry = np.clip(np.round(reflected[:, 1]).astype(int), 0, h - 1)

        mirrored_mask = np.zeros((h, w), dtype=bool)
        mirrored_mask[ry, rx] = True

        # Close small gaps from rounding
        mirrored_mask = cv2.morphologyEx(
            mirrored_mask.astype(np.uint8),
            cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        ).astype(bool)

        # ── 4. Build mirrored image pixels ────────────────────────────────────
        img_orig = self._unfold_masks[fold_idx]['img_original']
        src_coords = (fys2, fxs2)
        dst_ry = np.clip(np.round(reflected[:, 1]).astype(int), 0, h - 1)
        dst_rx = np.clip(np.round(reflected[:, 0]).astype(int), 0, w - 1)
        mirrored_img = np.zeros((h, w, 3), dtype=np.uint8)
        mirrored_img[dst_ry, dst_rx] = img_orig[fys2, fxs2]

        # ── 5. Compose result: static + mirrored ──────────────────────────────
        result_mask = static_mask | mirrored_mask
        for i, m in enumerate(self._unfold_masks):
            if i != fold_idx:
                result_img[m['original']] = m['img_original'][m['original']]
        result_img[mirrored_mask] = mirrored_img[mirrored_mask]

        # Store pending for apply
        self._unfold_pending = {
            'masks':      self._unfold_masks,
            'result':     result_mask,
            'result_img': result_img,
            'fold_idx':   fold_idx,
            'fold_dir':   fold_dir,
            'pivot':      pivot,
            'mirrored_mask': mirrored_mask,
        }

        self._draw_unfold_result(result_mask, fold_idx)

    def _get_unfold_pivot(self, rotate_idx: int) -> tuple:
        """Get the pivot point for rotation."""
        if not self._unfold_masks:
            return (0, 0)

        mask_to_rotate = self._unfold_masks[rotate_idx]['original']
        pivot_mode = self._unfold_pivot_var.get()

        if pivot_mode == "center":
            # Use center of the mask to rotate
            ys, xs = np.nonzero(mask_to_rotate)
            if len(ys) > 0:
                return (int(xs.mean()), int(ys.mean()))
            return (0, 0)

        else:  # junction mode - find closest point between masks
            # Find the junction/overlap area between the rotating mask and others
            other_combined = np.zeros_like(mask_to_rotate, dtype=bool)
            for i, m in enumerate(self._unfold_masks):
                if i != rotate_idx:
                    other_combined |= m['original']

            # Dilate both masks slightly and find intersection
            kernel = np.ones((5, 5), np.uint8)
            dilated_rotate = cv2.dilate(mask_to_rotate.astype(np.uint8), kernel, iterations=3)
            dilated_other = cv2.dilate(other_combined.astype(np.uint8), kernel, iterations=3)

            junction = np.logical_and(dilated_rotate > 0, dilated_other > 0)

            if junction.any():
                ys, xs = np.nonzero(junction)
                return (int(xs.mean()), int(ys.mean()))

            # Fallback to center of rotating mask
            ys, xs = np.nonzero(mask_to_rotate)
            if len(ys) > 0:
                return (int(xs.mean()), int(ys.mean()))
            return (0, 0)

    def _rotate_mask(self, mask: np.ndarray, angle: float, pivot: tuple,
                     offset: tuple = (0, 0)) -> np.ndarray:
        """Rotate a mask around a pivot point."""
        if angle == 0 and offset == (0, 0):
            return mask.copy()

        h, w = mask.shape
        px, py = pivot

        # Create rotation matrix
        M = cv2.getRotationMatrix2D((px, py), -angle, 1.0)  # Negative for clockwise

        # Add translation offset
        M[0, 2] += offset[0]
        M[1, 2] += offset[1]

        # Apply transformation
        rotated = cv2.warpAffine(mask.astype(np.uint8) * 255, M, (w, h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        return rotated > 127

    def _rotate_image(self, img: np.ndarray, angle: float, pivot: tuple,
                      offset: tuple = (0, 0)) -> np.ndarray:
        """Rotate an RGB image around a pivot point."""
        if angle == 0 and offset == (0, 0):
            return img.copy()

        h, w = img.shape[:2]
        px, py = pivot

        # Create rotation matrix
        M = cv2.getRotationMatrix2D((px, py), -angle, 1.0)  # Negative for clockwise

        # Add translation offset
        M[0, 2] += offset[0]
        M[1, 2] += offset[1]

        # Apply transformation with bilinear interpolation for smooth results
        rotated = cv2.warpAffine(img, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        return rotated

    def _flip_mask(self, mask: np.ndarray, pivot: tuple, flip_h: bool, flip_v: bool) -> np.ndarray:
        """Flip a mask around a pivot point."""
        if not flip_h and not flip_v:
            return mask.copy()

        h, w = mask.shape
        px, py = pivot

        result = mask.copy()

        if flip_h:
            # Flip horizontally around pivot
            # Create coordinate grid
            coords_x = np.arange(w)
            new_x = 2 * px - coords_x
            new_x = np.clip(new_x, 0, w - 1).astype(int)
            result = result[:, new_x]

        if flip_v:
            # Flip vertically around pivot
            coords_y = np.arange(h)
            new_y = 2 * py - coords_y
            new_y = np.clip(new_y, 0, h - 1).astype(int)
            result = result[new_y, :]

        return result

    def _flip_image(self, img: np.ndarray, pivot: tuple, flip_h: bool, flip_v: bool) -> np.ndarray:
        """Flip an RGB image around a pivot point."""
        if not flip_h and not flip_v:
            return img.copy()

        h, w = img.shape[:2]
        px, py = pivot

        result = img.copy()

        if flip_h:
            # Flip horizontally around pivot
            coords_x = np.arange(w)
            new_x = 2 * px - coords_x
            new_x = np.clip(new_x, 0, w - 1).astype(int)
            result = result[:, new_x, :]

        if flip_v:
            # Flip vertically around pivot
            coords_y = np.arange(h)
            new_y = 2 * py - coords_y
            new_y = np.clip(new_y, 0, h - 1).astype(int)
            result = result[new_y, :, :]

        return result

    def _draw_unfold_result(self, result: np.ndarray, rotate_idx: int):
        """Draw the unfolded result on the canvas."""
        if not hasattr(self, '_unfold_crop_bounds'):
            return

        y1, y2, x1, x2 = self._unfold_crop_bounds

        # Expand bounds if needed to capture rotated mask
        ys, xs = np.nonzero(result)
        if len(ys) > 0:
            pad = 20
            h, w = result.shape
            y1 = max(0, min(y1, ys.min() - pad))
            y2 = min(h, max(y2, ys.max() + pad))
            x1 = max(0, min(x1, xs.min() - pad))
            x2 = min(w, max(x2, xs.max() + pad))

        # Create visualization using actual transformed image data
        # Get the combined result image from pending
        if hasattr(self, '_unfold_pending') and self._unfold_pending and 'result_img' in self._unfold_pending:
            result_img = self._unfold_pending['result_img']
            vis = result_img[y1:y2, x1:x2].copy()

            # Add checkerboard background for transparency
            result_crop = result[y1:y2, x1:x2]
            tile = 16
            Hc, Wc = vis.shape[:2]
            if Hc > 0 and Wc > 0:
                chk = np.indices((Hc, Wc)).sum(axis=0) // tile
                bg = np.where((chk % 2)[..., None], 200, 160).astype(np.uint8)
                bg = np.dstack([bg, bg, bg])  # Make it 3-channel
                # Composite: show image where mask is, checkerboard where not
                vis = np.where(result_crop[..., None], vis, bg)
        else:
            # Fallback to colored masks
            vis = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
            colors = [
                (255, 100, 100),  # Red
                (100, 255, 100),  # Green
                (100, 100, 255),  # Blue
                (255, 255, 100),  # Yellow
                (255, 100, 255),  # Magenta
                (100, 255, 255),  # Cyan
            ]
            for i, m in enumerate(self._unfold_masks):
                color = colors[i % len(colors)]
                mask_crop = m['mask'][y1:y2, x1:x2]
                vis[mask_crop] = color

        # Apply zoom
        zoom = getattr(self, '_unfold_zoom', 1.0)
        self._draw_rgb_on_canvas(self._unfold_result_canvas, vis, 250, 250, zoom=zoom)

        # Store for drag calculations
        self._unfold_vis_bounds = (y1, y2, x1, x2)

    def _draw_rgb_on_canvas(self, canvas, rgb_arr: np.ndarray, canvas_w: int, canvas_h: int,
                            zoom: float = 1.0):
        """Draw an RGB array on a canvas."""
        canvas.delete("all")

        h, w = rgb_arr.shape[:2]
        if h == 0 or w == 0:
            return

        # Calculate display size maintaining aspect ratio
        scale = min(canvas_w / w, canvas_h / h) * zoom
        disp_w = int(w * scale)
        disp_h = int(h * scale)

        # Resize
        resized = cv2.resize(rgb_arr, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

        # Convert to PhotoImage
        img = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(img)

        # Store reference to prevent garbage collection
        canvas._photo = photo
        canvas._scale = scale

        # Draw centered
        cx = canvas_w // 2
        cy = canvas_h // 2
        canvas.create_image(cx, cy, image=photo, anchor="center")

    def _unfold_rotate(self, degrees: float):
        """Rotate by specified degrees."""
        self._unfold_rotation_angle = getattr(self, '_unfold_rotation_angle', 0) + degrees

        # Normalize to -180 to 180
        while self._unfold_rotation_angle > 180:
            self._unfold_rotation_angle -= 360
        while self._unfold_rotation_angle < -180:
            self._unfold_rotation_angle += 360

        self._unfold_angle_label.configure(text=f"{int(self._unfold_rotation_angle)}°")
        self._unfold_update_preview()

    def _unfold_set_rotation(self, degrees: float):
        """Set rotation to a specific angle."""
        self._unfold_rotation_angle = degrees
        self._unfold_angle_label.configure(text=f"{int(self._unfold_rotation_angle)}°")
        self._unfold_update_preview()

    def _unfold_reset_rotation(self):
        """Reset rotation to 0."""
        self._unfold_rotation_angle = 0
        self._unfold_angle_label.configure(text="0°")
        self._unfold_update_preview()

    def _unfold_flip_horizontal(self):
        """Rotate fold line 90° — changes which axis the mirror reflects across."""
        self._unfold_rotation_angle = (self._unfold_rotation_angle + 90) % 360
        if hasattr(self, '_unfold_angle_label'):
            self._unfold_angle_label.configure(
                text=f"{self._unfold_rotation_angle:.0f}°")
        self._unfold_update_preview()

    def _unfold_flip_vertical(self):
        """Rotate fold line -90°."""
        self._unfold_rotation_angle = (self._unfold_rotation_angle - 90) % 360
        if hasattr(self, '_unfold_angle_label'):
            self._unfold_angle_label.configure(
                text=f"{self._unfold_rotation_angle:.0f}°")
        self._unfold_update_preview()

    def _unfold_reset_position(self):
        """Reset position offset to (0, 0)."""
        self._unfold_offset = [0, 0]
        self._unfold_offset_label.configure(text="(0, 0)")
        self._unfold_update_preview()

    def _unfold_drag_start(self, event):
        """Start dragging to reposition."""
        self._unfold_drag_start_pos = (event.x, event.y)

    def _unfold_drag_move(self, event):
        """Handle drag to reposition mask."""
        if not hasattr(self, '_unfold_drag_start_pos') or self._unfold_drag_start_pos is None:
            return

        # Calculate delta in canvas coordinates
        dx = event.x - self._unfold_drag_start_pos[0]
        dy = event.y - self._unfold_drag_start_pos[1]

        # Convert to mask coordinates (account for zoom and scale)
        scale = getattr(self._unfold_result_canvas, '_scale', 1.0)
        if scale > 0:
            mask_dx = dx / scale
            mask_dy = dy / scale
        else:
            mask_dx, mask_dy = dx, dy

        # Update offset
        self._unfold_offset[0] += int(mask_dx)
        self._unfold_offset[1] += int(mask_dy)

        self._unfold_offset_label.configure(text=f"({self._unfold_offset[0]}, {self._unfold_offset[1]})")

        # Update drag start position
        self._unfold_drag_start_pos = (event.x, event.y)

        # Update preview
        self._unfold_update_preview()

    def _unfold_drag_end(self, event):
        """End dragging."""
        self._unfold_drag_start_pos = None

    def _unfold_zoom_by(self, factor: float):
        """Zoom the unfold preview by a factor."""
        self._unfold_zoom = getattr(self, '_unfold_zoom', 1.0) * factor
        self._unfold_zoom = max(0.25, min(self._unfold_zoom, 5.0))
        self._unfold_zoom_label.configure(text=f"{int(self._unfold_zoom * 100)}%")
        self._unfold_update_preview()

    def _unfold_zoom_fit(self):
        """Reset unfold preview zoom to 100%."""
        self._unfold_zoom = 1.0
        self._unfold_zoom_label.configure(text="100%")
        self._unfold_update_preview()

    def _apply_leaf_unfolding(self):
        self._push_undo("leaf unfolding")
        """Apply the unfolding - merge masks into one."""
        if not hasattr(self, '_unfold_pending') or self._unfold_pending is None:
            messagebox.showwarning("Leaf Unfolding", "No pending unfold. Click 'Preview Selected' first.")
            return

        pending = self._unfold_pending
        result = pending['result']
        result_img = pending.get('result_img', None)  # Combined/transformed image
        original_indices = [m['idx'] for m in pending['masks']]

        # Calculate area
        result_area = int(result.sum())

        # Calculate bounding box
        ys, xs = np.nonzero(result)
        if len(ys) == 0:
            messagebox.showwarning("Leaf Unfolding", "Result mask is empty.")
            return

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bbox = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]

        # Crop the combined image to bbox for storage
        unfolded_img_crop = None
        if result_img is not None:
            unfolded_img_crop = result_img[y1:y2+1, x1:x2+1].copy()

        # Add new combined mask
        self.sr.masks.append({
            "segmentation": result.astype(np.uint8),
            "bbox": bbox,
            "area": float(result_area),
            "meta": {
                "unfolded": True,
                "source_indices": original_indices,
                "rotation_angle": self._unfold_rotation_angle,
                "offset": tuple(self._unfold_offset),
                "unfolded_image": unfolded_img_crop,  # Store the transformed image
            }
        })

        # Optionally remove original masks (in reverse order to maintain indices)
        # Ask user first
        remove = messagebox.askyesno("Leaf Unfolding",
            f"Unfolded mask created ({result_area:,} px).\n\n"
            f"Remove the original {len(original_indices)} masks?")

        if remove:
            for idx in sorted(original_indices, reverse=True):
                del self.sr.masks[idx]

        # Rebuild mask list
        self._rebuild_mask_list()

        # Select the new mask
        self.lb.selection_clear(0, tk.END)
        if remove:
            self.lb.selection_set(len(self.sr.masks) - 1)
        else:
            self.lb.selection_set(tk.END)
        self.on_select_mask()

        # Clear state
        self._cancel_leaf_unfolding()

        self.set_status(f"Unfolded {len(original_indices)} masks: {result_area:,} px", "success")

    def _cancel_leaf_unfolding(self):
        """Cancel/reset the unfold preview."""
        self._unfold_masks = []
        self._unfold_pending = None
        self._unfold_rotation_angle = 0
        self._unfold_offset = [0, 0]

        self._unfold_angle_label.configure(text="0°")
        self._unfold_offset_label.configure(text="(0, 0)")
        self._unfold_update_btn.configure(state="disabled")
        self._unfold_stats_label.configure(text="Select 2+ masks to unfold")

        # Clear canvases
        self._unfold_orig_canvas.delete("all")
        self._unfold_result_canvas.delete("all")

        # Clear radio buttons
        for widget in self._unfold_mask_frame.winfo_children():
            widget.destroy()

        self.set_status("Unfold cancelled", "info")

    # ════════════════════════════════════════════════════════════════════════
    # SPLIT MASK - Draw line to split one mask into two
    # ════════════════════════════════════════════════════════════════════════

    # ════════════════════════════════════════════════════════════════════════
    # EDIT MASK - Contour-based boundary editing
    # ════════════════════════════════════════════════════════════════════════

    def edit_mask_mode(self):
        """Enter mask edit mode - drag control points to reshape the boundary."""
        sel = list(self.lb.curselection())
        if len(sel) != 1:
            messagebox.showwarning("Edit Mask", "Select exactly one mask to edit.")
            return
        if self.sr is None or not self.sr.masks:
            messagebox.showwarning("Edit Mask", "No masks available.")
            return

        idx = sel[0]
        mask = self.sr.masks[idx]["segmentation"].astype(bool)

        # Create contour edit dialog
        self._show_contour_edit_dialog(idx, mask)

    def _show_contour_edit_dialog(self, idx: int, mask: np.ndarray):
        """Show the mask contour edit dialog - drag points to reshape boundary."""
        # Create toplevel window
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Mask Boundary")
        dialog.geometry("900x700")
        dialog.transient(self.root)
        dialog.grab_set()

        self._edit_dialog = dialog
        self._edit_idx = idx
        self._edit_mask_original = mask.copy()
        self._edit_mask = mask.astype(np.uint8).copy()

        # Extract contour and create control points
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            messagebox.showwarning("Edit Mask", "Could not extract mask contour.")
            dialog.destroy()
            return

        # Use largest contour
        contour = max(contours, key=cv2.contourArea)

        # Simplify contour to get control points (approx every 15 pixels)
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Further sample if too many points
        pts = approx.reshape(-1, 2).tolist()
        if len(pts) > 50:
            step = len(pts) // 40
            pts = pts[::step]
        elif len(pts) < 8:
            # If too few points, use more from original contour
            pts = contour.reshape(-1, 2).tolist()
            step = max(1, len(pts) // 30)
            pts = pts[::step]

        self._edit_control_points = pts  # List of [x, y] in image coords
        self._edit_dragging_idx = None
        self._edit_hover_idx = None

        # Get bbox for cropping with padding
        x, y, w, h = map(int, self.sr.masks[idx]["bbox"])
        pad = 50
        img_h, img_w = mask.shape
        y1 = max(0, y - pad)
        y2 = min(img_h, y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(img_w, x + w + pad)
        self._edit_crop_bounds = (y1, y2, x1, x2)

        # Main layout
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Left: Instructions and buttons
        tools_frame = ttk.LabelFrame(main_frame, text=" Edit Boundary ", padding=8)
        tools_frame.pack(side="left", fill="y", padx=(0, 10))

        ttk.Label(tools_frame, text="Instructions:", font=("Helvetica", 10, "bold")).pack(anchor="w")
        ttk.Label(tools_frame, text="• Drag points to reshape", wraplength=150).pack(anchor="w", pady=2)
        ttk.Label(tools_frame, text="• Ctrl/Cmd+click to add", wraplength=150).pack(anchor="w", pady=2)
        ttk.Label(tools_frame, text="• Shift+click to delete", wraplength=150).pack(anchor="w", pady=2)

        ttk.Separator(tools_frame, orient="horizontal").pack(fill="x", pady=10)

        # Point count display
        self._edit_point_label = ttk.Label(tools_frame, text=f"Points: {len(pts)}")
        self._edit_point_label.pack(anchor="w", pady=5)

        # Add/Delete point buttons
        point_btn_frame = ttk.Frame(tools_frame)
        point_btn_frame.pack(fill="x", pady=5)
        self._edit_add_mode = tk.BooleanVar(value=False)
        self._edit_delete_mode = tk.BooleanVar(value=False)

        self._edit_add_btn = ttk.Checkbutton(point_btn_frame, text="➕ Add Point",
                                              variable=self._edit_add_mode,
                                              command=self._edit_toggle_add_mode)
        self._edit_add_btn.pack(fill="x", pady=2)

        self._edit_del_btn = ttk.Checkbutton(point_btn_frame, text="➖ Delete Point",
                                              variable=self._edit_delete_mode,
                                              command=self._edit_toggle_delete_mode)
        self._edit_del_btn.pack(fill="x", pady=2)

        ttk.Separator(tools_frame, orient="horizontal").pack(fill="x", pady=10)

        ttk.Button(tools_frame, text="Smooth", command=self._edit_smooth_contour).pack(fill="x", pady=2)
        ttk.Button(tools_frame, text="Reset", command=self._edit_reset_contour).pack(fill="x", pady=2)

        ttk.Separator(tools_frame, orient="horizontal").pack(fill="x", pady=10)

        ttk.Button(tools_frame, text="Apply", command=self._edit_apply_contour).pack(fill="x", pady=2)
        ttk.Button(tools_frame, text="Cancel", command=self._edit_cancel_contour).pack(fill="x", pady=2)

        # Right: Canvas for editing
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side="right", fill="both", expand=True)

        self._edit_canvas = tk.Canvas(canvas_frame, bg=self.colors['canvas_bg'],
                                       highlightthickness=1, highlightbackground=self.colors['border'])
        self._edit_canvas.pack(fill="both", expand=True)

        # Bind mouse events
        self._edit_canvas.bind("<Button-1>", self._edit_contour_click)
        self._edit_canvas.bind("<B1-Motion>", self._edit_contour_drag)
        self._edit_canvas.bind("<ButtonRelease-1>", self._edit_contour_release)
        self._edit_canvas.bind("<Motion>", self._edit_contour_hover)
        # Ctrl+click / Cmd+click for add point
        self._edit_canvas.bind("<Control-Button-1>", self._edit_contour_add_point)
        self._edit_canvas.bind("<Command-Button-1>", self._edit_contour_add_point)
        # Shift+click for delete point
        self._edit_canvas.bind("<Shift-Button-1>", self._edit_contour_delete_point)

        # Initialize
        self._edit_scale = 1.0

        # Initial draw after canvas is sized
        dialog.update_idletasks()
        dialog.after(100, self._edit_refresh_contour_canvas)

        # Handle window close
        dialog.protocol("WM_DELETE_WINDOW", self._edit_cancel_contour)

    def _edit_refresh_contour_canvas(self):
        """Refresh the canvas showing image with contour and control points."""
        if not hasattr(self, '_edit_canvas') or not self._edit_canvas.winfo_exists():
            return

        y1, y2, x1, x2 = self._edit_crop_bounds
        img_crop = self.sr.img_color[y1:y2, x1:x2].copy()
        Hc, Wc = img_crop.shape[:2]

        # Create filled mask from control points
        pts_array = np.array(self._edit_control_points, dtype=np.int32)
        # Offset points to crop coordinates
        pts_crop = pts_array - np.array([x1, y1])

        # Draw filled polygon on visualization
        vis = img_crop.copy()

        # Semi-transparent mask overlay
        overlay = np.zeros_like(vis)
        if len(pts_crop) >= 3:
            cv2.fillPoly(overlay, [pts_crop], (100, 200, 100))
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        # Draw contour line
        if len(pts_crop) >= 2:
            pts_draw = pts_crop.reshape((-1, 1, 2))
            cv2.polylines(vis, [pts_draw], True, (255, 255, 0), 2)

        # Scale to fit canvas
        canvas_w = self._edit_canvas.winfo_width()
        canvas_h = self._edit_canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            canvas_w, canvas_h = 700, 600

        scale = min(canvas_w / Wc, canvas_h / Hc) * 0.9
        self._edit_scale = scale

        disp_w = int(Wc * scale)
        disp_h = int(Hc * scale)
        resized = cv2.resize(vis, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

        # Center offset
        offset_x = (canvas_w - disp_w) // 2
        offset_y = (canvas_h - disp_h) // 2
        self._edit_offset = (offset_x, offset_y)

        # Convert to PhotoImage
        img = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(img)
        self._edit_canvas._photo = photo

        # Draw image
        self._edit_canvas.delete("all")
        self._edit_canvas.create_image(canvas_w // 2, canvas_h // 2, image=photo, anchor="center", tags="img")

        # Draw control points
        point_radius = 6
        for i, pt in enumerate(self._edit_control_points):
            # Convert image coords to canvas coords
            cx = (pt[0] - x1) * scale + offset_x
            cy = (pt[1] - y1) * scale + offset_y

            # Highlight if hovering
            if i == self._edit_hover_idx:
                color = "#FFD700"  # Gold
                r = point_radius + 2
            else:
                color = "#00FF00"  # Green
                r = point_radius

            self._edit_canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                fill=color, outline="white", width=2, tags=f"pt_{i}"
            )

        # Update point count label
        if hasattr(self, '_edit_point_label'):
            self._edit_point_label.configure(text=f"Points: {len(self._edit_control_points)}")

    def _edit_canvas_to_image(self, cx, cy):
        """Convert canvas coords to image coords."""
        y1, y2, x1, x2 = self._edit_crop_bounds
        offset_x, offset_y = self._edit_offset
        scale = self._edit_scale

        # Canvas to crop coords
        crop_x = (cx - offset_x) / scale
        crop_y = (cy - offset_y) / scale

        # Crop to full image coords
        img_x = crop_x + x1
        img_y = crop_y + y1

        return img_x, img_y

    def _edit_find_nearest_point(self, cx, cy, threshold=15):
        """Find nearest control point within threshold distance."""
        y1, y2, x1, x2 = self._edit_crop_bounds
        offset_x, offset_y = self._edit_offset
        scale = self._edit_scale

        min_dist = float('inf')
        nearest_idx = None

        for i, pt in enumerate(self._edit_control_points):
            # Convert point to canvas coords
            pcx = (pt[0] - x1) * scale + offset_x
            pcy = (pt[1] - y1) * scale + offset_y

            dist = np.sqrt((cx - pcx)**2 + (cy - pcy)**2)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                nearest_idx = i

        return nearest_idx

    def _edit_toggle_add_mode(self):
        """Toggle add point mode - disable delete mode if enabling add."""
        if self._edit_add_mode.get():
            self._edit_delete_mode.set(False)
            self._edit_canvas.configure(cursor="plus")
        else:
            self._edit_canvas.configure(cursor="")

    def _edit_toggle_delete_mode(self):
        """Toggle delete point mode - disable add mode if enabling delete."""
        if self._edit_delete_mode.get():
            self._edit_add_mode.set(False)
            self._edit_canvas.configure(cursor="X_cursor")
        else:
            self._edit_canvas.configure(cursor="")

    def _edit_contour_hover(self, event):
        """Handle mouse hover - highlight nearest point."""
        old_hover = self._edit_hover_idx
        self._edit_hover_idx = self._edit_find_nearest_point(event.x, event.y)

        # Update cursor based on mode
        if self._edit_add_mode.get():
            self._edit_canvas.configure(cursor="plus")
        elif self._edit_delete_mode.get():
            self._edit_canvas.configure(cursor="X_cursor")
        elif self._edit_hover_idx is not None:
            self._edit_canvas.configure(cursor="hand2")
        else:
            self._edit_canvas.configure(cursor="")

        # Refresh if hover changed
        if old_hover != self._edit_hover_idx:
            self._edit_refresh_contour_canvas()

    def _edit_contour_click(self, event):
        """Handle click - add/delete point if mode active, otherwise drag."""
        # Check if in add mode (from button)
        if self._edit_add_mode.get():
            self._edit_contour_add_point(event)
            return

        # Check if in delete mode (from button)
        if self._edit_delete_mode.get():
            self._edit_contour_delete_point(event)
            return

        # Normal mode - start dragging if near a point
        self._edit_dragging_idx = self._edit_find_nearest_point(event.x, event.y)

    def _edit_contour_drag(self, event):
        """Handle drag - move the selected point."""
        if self._edit_dragging_idx is None:
            return

        # Convert canvas coords to image coords
        img_x, img_y = self._edit_canvas_to_image(event.x, event.y)

        # Clamp to image bounds
        img_h, img_w = self._edit_mask_original.shape
        img_x = max(0, min(img_w - 1, img_x))
        img_y = max(0, min(img_h - 1, img_y))

        # Update point
        self._edit_control_points[self._edit_dragging_idx] = [int(img_x), int(img_y)]

        # Refresh canvas
        self._edit_refresh_contour_canvas()

    def _edit_contour_release(self, event):
        """Handle mouse release - stop dragging."""
        self._edit_dragging_idx = None

    def _edit_contour_add_point(self, event):
        """Add a new control point at click location (Ctrl/Cmd+click or Add mode)."""
        img_x, img_y = self._edit_canvas_to_image(event.x, event.y)

        # Find best position to insert (between two nearest points)
        if len(self._edit_control_points) < 2:
            self._edit_control_points.append([int(img_x), int(img_y)])
        else:
            # Find which edge is closest
            min_dist = float('inf')
            insert_idx = len(self._edit_control_points)

            for i in range(len(self._edit_control_points)):
                p1 = np.array(self._edit_control_points[i])
                p2 = np.array(self._edit_control_points[(i + 1) % len(self._edit_control_points)])
                pt = np.array([img_x, img_y])

                # Distance from point to line segment
                line_vec = p2 - p1
                line_len = np.linalg.norm(line_vec)
                if line_len < 1:
                    continue
                line_unit = line_vec / line_len
                proj_len = np.dot(pt - p1, line_unit)
                proj_len = max(0, min(line_len, proj_len))
                closest = p1 + proj_len * line_unit
                dist = np.linalg.norm(pt - closest)

                if dist < min_dist:
                    min_dist = dist
                    insert_idx = i + 1

            self._edit_control_points.insert(insert_idx, [int(img_x), int(img_y)])

        self._edit_refresh_contour_canvas()

    def _edit_contour_delete_point(self, event):
        """Delete the nearest control point (Shift+click or Delete mode)."""
        if len(self._edit_control_points) <= 3:
            messagebox.showwarning("Edit Mask", "Need at least 3 points for a valid shape.")
            return

        idx = self._edit_find_nearest_point(event.x, event.y, threshold=20)
        if idx is not None:
            del self._edit_control_points[idx]
            self._edit_hover_idx = None
            self._edit_refresh_contour_canvas()

    def _edit_smooth_contour(self):
        """Smooth the contour by averaging nearby points."""
        if len(self._edit_control_points) < 5:
            return

        pts = np.array(self._edit_control_points, dtype=np.float32)
        n = len(pts)

        # Simple smoothing: average with neighbors
        smoothed = []
        for i in range(n):
            prev_pt = pts[(i - 1) % n]
            curr_pt = pts[i]
            next_pt = pts[(i + 1) % n]
            new_pt = (prev_pt * 0.25 + curr_pt * 0.5 + next_pt * 0.25)
            smoothed.append([int(new_pt[0]), int(new_pt[1])])

        self._edit_control_points = smoothed
        self._edit_refresh_contour_canvas()

    def _edit_reset_contour(self):
        """Reset to original contour."""
        mask = self._edit_mask_original
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            pts = approx.reshape(-1, 2).tolist()
            if len(pts) > 50:
                step = len(pts) // 40
                pts = pts[::step]
            self._edit_control_points = pts
            self._edit_refresh_contour_canvas()

    def _edit_apply_contour(self):
        """Apply the edited contour as the new mask."""
        if len(self._edit_control_points) < 3:
            messagebox.showwarning("Edit Mask", "Need at least 3 points.")
            return

        # Create new mask from contour
        pts = np.array(self._edit_control_points, dtype=np.int32)
        mask = np.zeros_like(self._edit_mask_original, dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)

        # Update bounding box
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            messagebox.showwarning("Edit Mask", "Resulting mask is empty!")
            return

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bbox = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
        area = float(mask.sum())

        # Update the mask
        self._push_undo("edit boundary")
        self.sr.masks[self._edit_idx]["segmentation"] = mask
        self.sr.masks[self._edit_idx]["bbox"] = bbox
        self.sr.masks[self._edit_idx]["area"] = area

        # Add edit metadata
        meta = self.sr.masks[self._edit_idx].get("meta", {}) or {}
        meta["edited"] = True
        self.sr.masks[self._edit_idx]["meta"] = meta

        # Close dialog and refresh
        self._edit_dialog.destroy()
        self._rebuild_mask_list()
        self.lb.selection_clear(0, tk.END)
        self.lb.selection_set(self._edit_idx)
        self.on_select_mask()

        self.set_status("Mask boundary updated", "success")

    def _edit_cancel_contour(self):
        """Cancel editing and close dialog."""
        self._edit_dialog.destroy()
        self.set_status("Edit cancelled", "info")

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
        # Prefer shape completion model if loaded
        if getattr(self, "shape_model", None) is not None:
            return self._predict_extend_completion(base_mask_bool, forbid_mask)

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

    def _predict_extend_completion(self, base_mask_bool: np.ndarray, forbid_mask: np.ndarray | None):
        """Predict completed mask from partial mask using shape completion model.

        IMPORTANT: The model expects a CROPPED mask (just the leaf region), not the full image.
        We crop to bbox, predict, then place back into full image coordinates.
        """
        try:
            from mask_completion import predict_mask
        except Exception as e:
            messagebox.showwarning("Shape Completion", f"Shape completion module not available: {e}")
            return None

        if self.shape_model is None:
            messagebox.showwarning("Shape Completion", "Load the shape completion model first.")
            return None

        meta = self.shape_meta or {}
        input_size = int(meta.get("size", 128))

        # Get bounding box of the mask
        ys, xs = np.nonzero(base_mask_bool)
        if len(ys) == 0:
            return None
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1

        # Add small padding around the crop - just enough for model to extend edges
        # Training masks fill ~80% of canvas, so keep padding minimal
        h, w = y2 - y1, x2 - x1
        pad_y = max(5, int(h * 0.1))  # 10% padding - keeps mask large in canvas
        pad_x = max(5, int(w * 0.1))

        # Expand bbox with padding, clamp to image bounds
        img_h, img_w = base_mask_bool.shape
        y1_pad = max(0, y1 - pad_y)
        y2_pad = min(img_h, y2 + pad_y)
        x1_pad = max(0, x1 - pad_x)
        x2_pad = min(img_w, x2 + pad_x)

        # Crop the mask to bbox (this is what the model expects!)
        mask_crop = base_mask_bool[y1_pad:y2_pad, x1_pad:x2_pad]

        print(f"[DEBUG] Full mask shape: {base_mask_bool.shape}, area: {base_mask_bool.sum()}")
        print(f"[DEBUG] Cropped mask shape: {mask_crop.shape}, area: {mask_crop.sum()}")
        print(f"[DEBUG] Bbox: x={x1_pad}:{x2_pad}, y={y1_pad}:{y2_pad}")

        # DEBUG: Save input mask to file
        debug_input = mask_crop.astype(np.uint8) * 255
        cv2.imwrite('/tmp/debug_input_mask.png', debug_input)
        print(f"[DEBUG] Saved input mask to /tmp/debug_input_mask.png")

        # Predict on the CROPPED mask
        pred_crop_raw = predict_mask(
            self.shape_model,
            mask_crop.astype(np.uint8) * 255,
            size=input_size,
            threshold=0.5,
        )
        pred_crop = (pred_crop_raw > 127).astype(bool)

        # DEBUG: Save output mask to file
        cv2.imwrite('/tmp/debug_output_mask.png', pred_crop_raw)
        print(f"[DEBUG] Saved output mask to /tmp/debug_output_mask.png")

        print(f"[DEBUG] Predicted crop area: {pred_crop.sum()}")

        # Place prediction back into full image coordinates
        pred_full = np.zeros_like(base_mask_bool, dtype=bool)
        pred_full[y1_pad:y2_pad, x1_pad:x2_pad] = pred_crop

        # Ensure base mask is always included
        pred_full = np.logical_or(pred_full, base_mask_bool)

        print(f"[DEBUG] Full prediction area: {pred_full.sum()}")
        print(f"[DEBUG] Added pixels: {np.logical_and(pred_full, ~base_mask_bool).sum()}")

        # Keep only components touching the original mask
        pred_full = self._keep_components_touching_base(pred_full, base_mask_bool)

        return pred_full




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
            messagebox.showwarning("Combine", "Couldn't combine those selections.")

    def duplicate_selected_masks(self):
        """Duplicate the selected masks."""
        if not self.sr or not self.sr.masks:
            messagebox.showwarning("No masks", "Run segmentation first.")
            return

        sel = list(self.lb.curselection())
        if not sel:
            messagebox.showwarning("No selection", "Select one or more masks to duplicate.")
            return

        # Duplicate each selected mask
        new_indices = []
        for idx in sel:
            mask_data = self.sr.masks[idx]
            # Deep copy the mask data
            new_mask = {
                "segmentation": mask_data["segmentation"].copy(),
                "bbox": list(mask_data["bbox"]),
                "area": mask_data["area"],
                "meta": {"duplicated_from": idx}
            }
            self.sr.masks.append(new_mask)
            new_indices.append(len(self.sr.masks) - 1)

        # Rebuild list and select the new duplicates
        self._rebuild_mask_list()
        self.lb.selection_clear(0, tk.END)
        for new_idx in new_indices:
            self.lb.selection_set(new_idx)

        self._render_preview()
        self.set_status(f"Duplicated {len(sel)} mask(s)", "success")

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
        self._push_undo("clear all masks")
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
        self._update_file_label()
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

    # =========================================================================
    # Model Type Selection (SAM2 vs Custom Model)
    # =========================================================================
    def _on_model_type_change(self):
        """Show/hide SAM2 or Custom Model options based on selection."""
        mode = self.model_type_var.get()
        if mode == "sam2":
            self.sam2_frame.pack(fill="x")
            self.tip_frame.pack_forget()
        else:  # tip
            self.sam2_frame.pack_forget()
            self.tip_frame.pack(fill="x")

    def _browse_tip_model(self):
        """Browse for a tip model .pth file."""
        p = filedialog.askopenfilename(
            title="Select Custom Model",
            filetypes=[("PyTorch model", "*.pth *.pt"), ("All files", "*.*")]
        )
        if p:
            self.tip_model_path_var.set(p)

    def _load_tip_model_main(self):
        """Load the tip model for segmentation (no SAM)."""
        try:
            ckpt_path = self.tip_model_path_var.get().strip()
            if not ckpt_path or not os.path.exists(ckpt_path):
                messagebox.showwarning("Missing file", "Pick a valid tip model .pth file.")
                return
            dev = self.e_tip_dev.get().strip() or "mps"

            from tip_segmenter_model import load_tipseg_checkpoint
            model, meta = load_tipseg_checkpoint(ckpt_path, device=dev)
            self.tipseg_model = model
            self.tipseg_meta = meta

            # Update status label
            self.tip_status_lbl.configure(text=f"✓ Model loaded on {dev}")

            # Update threshold if available
            if "threshold" in meta:
                try:
                    self.target_tipseg_thresh.set(float(meta["threshold"]))
                except Exception:
                    pass

            messagebox.showinfo("Model", f"Tip model loaded on '{dev}'.")
            try:
                self._refresh_weights_badge()
            except Exception:
                pass
        except Exception as e:
            self.tip_status_lbl.configure(text="✗ Load failed")
            messagebox.showerror("Load failed", str(e))

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
        self._update_file_label()
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
            if not self.batch_images:
                messagebox.showwarning("No folder", "Open a folder or choose a dataset folder first.")
                return

        # Tip-only segmentation path (no SAM)
        if bool(self.target_use_tipseg.get()):
            if self.tipseg_model is None:
                use_sam = messagebox.askyesno(
                    "Tip model not loaded",
                    "Tip model isn't loaded.\n\nUse SAM2 segmentation instead?",
                )
                if use_sam:
                    self.target_use_tipseg.set(False)
                else:
                    messagebox.showwarning("No model", "Load the model first (Train Custom Model → Load Model).")
                    return
            if bool(self.target_use_tipseg.get()):
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
                remove_white = bool(self.tipseg_remove_white.get())
                white_sat_max = int(self.tipseg_white_sat_max.get())
                white_val_min = int(self.tipseg_white_val_min.get())
                remove_green = bool(self.tipseg_remove_green.get())
                green_hue_low = int(self.tipseg_green_hue_low.get())
                green_hue_high = int(self.tipseg_green_hue_high.get())
                green_sat_min = int(self.tipseg_green_sat_min.get())
                green_val_min = int(self.tipseg_green_val_min.get())

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
                                remove_white=remove_white,
                                white_sat_max=white_sat_max,
                                white_val_min=white_val_min,
                                remove_green=remove_green,
                                green_hue_low=green_hue_low,
                                green_hue_high=green_hue_high,
                                green_sat_min=green_sat_min,
                                green_val_min=green_val_min,
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
                            if remove_white and mb is not None and mb.any():
                                white = self._tipseg_white_mask(rgb, sat_max=white_sat_max, val_min=white_val_min)
                                mb = mb & (~white)
                            if remove_green and mb is not None and mb.any():
                                green = self._tipseg_green_mask(
                                    rgb,
                                    hue_low=green_hue_low,
                                    hue_high=green_hue_high,
                                    sat_min=green_sat_min,
                                    val_min=green_val_min,
                                )
                                mb = mb & (~green)

                        masks = []
                        if mb is not None and mb.any():
                            mask_u8 = mb.astype(np.uint8)
                            min_area = int(self.target_tipseg_min_area.get())
                            keep_largest = bool(self.target_tipseg_keep_largest.get())
                            comps = split_masks_by_cc([{"segmentation": mask_u8}], min_area=max(1, min_area))
                            if keep_largest and comps:
                                comps = [max(comps, key=lambda m: m.get("area", 0))]
                            masks = []
                            for m in comps:
                                seg = m["segmentation"].astype(np.uint8)
                                bbox, area = self._bbox_area_from_mask(seg)
                                masks.append({
                                    "segmentation": seg,
                                    "area": float(area),
                                    "bbox": bbox,
                                    "predicted_iou": 1.0,
                                    "stability_score": 1.0,
                                    "meta": {"source": "tipseg"},
                                })

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
                import traceback
                print(f"  [skip] {p}: {e}")
                traceback.print_exc()

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
        save_full, save_crop = self._get_mask_save_flags()
        if not (save_full or save_crop):
            messagebox.showwarning("Nothing to save", "Select at least one mask size: Full or Crop.")
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
                bbox = m.get("bbox")
                if not bbox or len(bbox) != 4:
                    bbox, _ = self._bbox_area_from_mask(seg_bool)
                mask_path = None
                mask_crop_path = None
                crop_path = img_dir / f"{stem}_crop_{k}.png"
                if save_full:
                    mask_path = img_dir / f"{stem}_{k}.png"
                    save_binary_mask(seg_bool, mask_path)
                if save_crop:
                    x, y, w, h = map(int, bbox)
                    x2, y2 = x + w, y + h
                    x = max(0, x); y = max(0, y)
                    x2 = min(seg_bool.shape[1], x2)
                    y2 = min(seg_bool.shape[0], y2)
                    mask_crop = seg_bool[y:y2, x:x2]
                    mask_crop_path = img_dir / f"{stem}_{k}.mask.crop.png"
                    save_binary_mask(mask_crop, mask_crop_path)
                save_masked_crop_rgba(color_img, seg_bool, bbox, crop_path,
                                      erode_px=erode_px, feather_px=feather_px)
                rows.append({
                    "mask_idx": k,
                    "area_px": int(m["area"]),
                    "bbox": list(map(int, bbox)) if bbox else [],
                    "mask_png": str(mask_path or mask_crop_path) if (mask_path or mask_crop_path) else "",
                    "mask_crop_png": str(mask_crop_path) if mask_crop_path else "",
                    "crop_png": str(crop_path),
                })

            csv_path = img_dir / f"{stem}_mask_manifest.csv"
            with open(csv_path, "w", newline="") as f:
                import csv as _csv
                w = _csv.DictWriter(f, fieldnames=["mask_idx","area_px","bbox","mask_png","mask_crop_png","crop_png"])
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

        mask = (edges > 0)
        out  = rgb.astype(np.float32)
        # Apply darkening to all 3 channels where mask is True
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
    def _wire_auto_preview(self):
        if getattr(self, "_auto_preview_wired", False):
            return
        self._auto_preview_wired = True

        vars_to_watch = [
            # Pipeline + basic adjustments
            self.enhance_pipeline, self.use_green, self.use_classic,
            self.s_brightness, self.s_contrast, self.s_gamma,
            # Sharpening
            self.chk_unsharp, self.unsharp_amount, self.unsharp_sigma, self.unsharp_ksize,
            self.chk_laplacian,
            # Background
            self.chk_whiten, self.chk_darken_bg, self.s_val_min, self.s_sat_max,
            # Denoising
            self.dn_median_on, self.dn_median_ksize, self.dn_mean_on, self.dn_mean_ksize,
            # Edge enhancement
            self.ed_on, self.ed_width, self.ed_amount,
            # Advanced enhancement
            self.use_veg_index, self.veg_index_type, self.veg_index_blend,
            self.use_white_balance, self.white_balance_type,
            self.use_retinex, self.retinex_type, self.retinex_sigma,
            self.use_lab, self.lab_l_factor, self.lab_a_shift,
            self.use_nlm, self.nlm_h,
            self.use_tophat, self.tophat_size,
            self.use_guided, self.guided_radius, self.guided_eps,
            self.use_dog, self.dog_sigma1, self.dog_sigma2, self.dog_blend,
            self.use_shadow_highlight, self.shadow_amount, self.highlight_amount,
            self.use_local_contrast, self.local_contrast_size,
            self.use_adaptive_gamma,
        ]

        for var in vars_to_watch:
            try:
                var.trace_add("write", self._schedule_auto_preview)
            except Exception:
                pass

    def _schedule_auto_preview(self, *_):
        if not getattr(self, "auto_preview", None) or not self.auto_preview.get():
            return
        if self.img is None and self.img_orig is None:
            return
        if self._auto_preview_job is not None:
            try:
                self.root.after_cancel(self._auto_preview_job)
            except Exception:
                pass
        self._auto_preview_job = self.root.after(self._auto_preview_delay_ms, self._run_auto_preview)

    def _run_auto_preview(self):
        self._auto_preview_job = None
        if not self.auto_preview.get():
            return
        if self.img is None and self.img_orig is None:
            return
        try:
            self.preview_enhance()
        except (tk.TclError, ValueError, TypeError):
            # Ignore intermediate invalid entry values while typing.
            return

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

        # Color filter highlights (drawn on top of everything)
        color_filter.draw_highlights(self)

    
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
        if hasattr(self, "_pick_status"):
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
        if hasattr(self, "_pick_status"):
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

        # Check if Custom Model mode is selected in the main Model panel
        use_tip_mode = (getattr(self, 'model_type_var', None) and
                        self.model_type_var.get() == "tip")

        # If enabled via checkbox OR model type selection, run tip-only segmentation (no SAM needed)
        if use_tip_mode or bool(self.target_use_tipseg.get()):
            if self.tipseg_model is None:
                self.set_status("No tip model loaded", "warning")
                use_sam = messagebox.askyesno(
                    "Tip model not loaded",
                    "Tip model isn't loaded.\n\nUse SAM2 segmentation instead?",
                )
                if use_sam:
                    self.target_use_tipseg.set(False)
                    if use_tip_mode:
                        self.model_type_var.set("sam2")
                        self._on_model_type_change()
                else:
                    return
            # Re-check after potential fallback
            use_tip_mode = (getattr(self, 'model_type_var', None) and
                            self.model_type_var.get() == "tip")
            if use_tip_mode or bool(self.target_use_tipseg.get()):
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
                    remove_white=bool(self.tipseg_remove_white.get()),
                    white_sat_max=int(self.tipseg_white_sat_max.get()),
                    white_val_min=int(self.tipseg_white_val_min.get()),
                    remove_green=bool(self.tipseg_remove_green.get()),
                    green_hue_low=int(self.tipseg_green_hue_low.get()),
                    green_hue_high=int(self.tipseg_green_hue_high.get()),
                    green_sat_min=int(self.tipseg_green_sat_min.get()),
                    green_val_min=int(self.tipseg_green_val_min.get()),
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
                if bool(self.tipseg_remove_white.get()) and mask_bool is not None and mask_bool.any():
                    white = self._tipseg_white_mask(
                        arr,
                        sat_max=int(self.tipseg_white_sat_max.get()),
                        val_min=int(self.tipseg_white_val_min.get()),
                    )
                    mask_bool = mask_bool & (~white)
                if bool(self.tipseg_remove_green.get()) and mask_bool is not None and mask_bool.any():
                    green = self._tipseg_green_mask(
                        arr,
                        hue_low=int(self.tipseg_green_hue_low.get()),
                        hue_high=int(self.tipseg_green_hue_high.get()),
                        sat_min=int(self.tipseg_green_sat_min.get()),
                        val_min=int(self.tipseg_green_val_min.get()),
                    )
                    mask_bool = mask_bool & (~green)

            masks = []
            if mask_bool is not None and mask_bool.any():
                mask_u8 = mask_bool.astype(np.uint8)
                min_area = int(self.target_tipseg_min_area.get())
                keep_largest = bool(self.target_tipseg_keep_largest.get())
                # Always split into connected components for tipseg output
                comps = split_masks_by_cc([{"segmentation": mask_u8}], min_area=max(1, min_area))
                if keep_largest and comps:
                    comps = [max(comps, key=lambda m: m.get("area", 0))]
                masks = []
                for m in comps:
                    seg = m["segmentation"].astype(np.uint8)
                    bbox, area = self._bbox_area_from_mask(seg)
                    masks.append({
                        "segmentation": seg,
                        "area": float(area),
                        "bbox": bbox,
                        "predicted_iou": 1.0,
                        "stability_score": 1.0,
                        "meta": {"source": "tipseg"},
                    })

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
        remove_white: bool = False,
        white_sat_max: int = 25,
        white_val_min: int = 210,
        remove_green: bool = False,
        green_hue_low: int = 35,
        green_hue_high: int = 90,
        green_sat_min: int = 35,
        green_val_min: int = 40,
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
        if remove_white and full_mask is not None and full_mask.any():
            white = self._tipseg_white_mask(img_rgb, sat_max=white_sat_max, val_min=white_val_min)
            full_mask = (full_mask.astype(bool) & (~white)).astype(np.uint8)
        if remove_green and full_mask is not None and full_mask.any():
            green = self._tipseg_green_mask(
                img_rgb,
                hue_low=green_hue_low,
                hue_high=green_hue_high,
                sat_min=green_sat_min,
                val_min=green_val_min,
            )
            full_mask = (full_mask.astype(bool) & (~green)).astype(np.uint8)
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

    def _tipseg_white_mask(self, img_rgb, sat_max: int = 25, val_min: int = 210):
        """Return boolean mask for near-white background (low saturation, high value)."""
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        s = hsv[..., 1]
        v = hsv[..., 2]
        sat_max = int(max(0, min(255, sat_max)))
        val_min = int(max(0, min(255, val_min)))
        return (s <= sat_max) & (v >= val_min)

    def _tipseg_green_mask(
        self,
        img_rgb,
        hue_low: int = 35,
        hue_high: int = 90,
        sat_min: int = 35,
        val_min: int = 40,
    ):
        """Return boolean mask for green leaf pixels (HSV threshold)."""
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        h = hsv[..., 0]
        s = hsv[..., 1]
        v = hsv[..., 2]

        h_low = int(max(0, min(179, hue_low)))
        h_high = int(max(0, min(179, hue_high)))
        if h_low <= h_high:
            hmask = (h >= h_low) & (h <= h_high)
        else:
            hmask = (h >= h_low) | (h <= h_high)

        sat_min = int(max(0, min(255, sat_min)))
        val_min = int(max(0, min(255, val_min)))
        return hmask & (s >= sat_min) & (v >= val_min)

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

        # Check if this is an unfolded mask with stored transformed image
        meta = m.get("meta", {})
        unfolded_img = meta.get("unfolded_image", None)

        if unfolded_img is not None:
            # Use the stored transformed/combined image instead of original
            crop = unfolded_img.copy()
        else:
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

        # If this mask was extended, fill the extended area with average RGB color
        # (meta already obtained above for unfolded check)
        ext_full = meta.get("extended_bool", None)
        fill_color = meta.get("fill_color", None)

        if ext_full is not None:
            # ext_full is full-image sized; crop to this mask's bbox
            ext_crop = (ext_full.astype(np.uint8) > 0)[y:y2, x:x2]
            # shapes must match
            if ext_crop.shape == msk.shape and ext_crop.any():
                # Fill the extended area with the average segment color (SOLID fill, not blended)
                if fill_color is not None:
                    fill_rgb = np.array(fill_color, dtype=np.uint8)
                else:
                    fill_rgb = np.array([100, 180, 100], dtype=np.uint8)  # default green if no color stored

                # SOLID fill for the extended/predicted area
                comp[ext_crop] = fill_rgb

                # Draw outline around the whole mask (yellow) and extended part (cyan for visibility)
                comp_bgr = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
                cnts_all, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(comp_bgr, cnts_all, -1, (0, 220, 255), 2)  # yellow outline (thicker)
                cnts_add, _ = cv2.findContours(ext_crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(comp_bgr, cnts_add, -1, (255, 255, 0), 2)  # cyan outline for extended (thicker)
                comp = cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2RGB)

        self.show_image(comp)

        # Update Leaf Completion preview when a mask is selected
        if hasattr(self, '_update_leaf_completion_preview'):
            self._update_leaf_completion_preview(mask_bool, idx)


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

    def _get_mask_save_flags(self):
        """Return (save_full, save_crop) mask size options."""
        save_full = True
        save_crop = False
        try:
            save_full = bool(self.save_mask_full.get())
        except Exception:
            pass
        try:
            save_crop = bool(self.save_mask_crop.get())
        except Exception:
            pass
        return save_full, save_crop

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
                cm = _ensure_mask_2d(color_mask)
                if cm is not None:
                    cm = _resize_mask_to_image(cm, rgb)
                if cm is not None and cm.any():
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
        save_full, save_crop = self._get_mask_save_flags()
        crop_dir = crop_dir or out_dir
        crop_dir.mkdir(parents=True, exist_ok=True)

        for k, idx in enumerate(indices, start=1):
            m = self.sr.masks[idx]
            mask_bool = m["segmentation"].astype(bool)

            if close_iters > 0:
                k3 = np.ones((3, 3), np.uint8)
                mask_bool = cv2.morphologyEx(mask_bool.astype(np.uint8), cv2.MORPH_CLOSE, k3,
                                             iterations=close_iters).astype(bool)

            bbox = m.get("bbox")
            if not bbox or len(bbox) != 4:
                bbox, _ = self._bbox_area_from_mask(mask_bool)
            seg_id = f"{base_name}_{k}"          # 1-based numbering
            mask_path = None
            mask_crop_path = None
            crop_path = crop_dir / f"{seg_id}.crop.png"
            if save_full:
                mask_path = out_dir / f"{seg_id}.mask.png"
                save_binary_mask(mask_bool, mask_path)
            if save_crop:
                x, y, w, h = map(int, bbox)
                x2, y2 = x + w, y + h
                x = max(0, x); y = max(0, y)
                x2 = min(mask_bool.shape[1], x2)
                y2 = min(mask_bool.shape[0], y2)
                mask_crop = mask_bool[y:y2, x:x2]
                mask_crop_path = out_dir / f"{seg_id}.mask.crop.png"
                save_binary_mask(mask_crop, mask_crop_path)
            save_masked_crop_rgba(self.sr.img_color, mask_bool, bbox, crop_path,
                                  erode_px=erode_px, feather_px=feather_px)

            rows.append({
                "file": base_name,
                "segment_id": seg_id,
                "mask_png": str(mask_path or mask_crop_path) if (mask_path or mask_crop_path) else "",
                "mask_crop_png": str(mask_crop_path) if mask_crop_path else "",
                "crop_png": str(crop_path),
                "area_px2": int(m["area"]),
                "bbox": list(map(int, bbox)) if bbox else [],
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
                "mask_crop_png": _rel(r.get("mask_crop_png")) if r.get("mask_crop_png") else "",
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
            mask_path = self._resolve_manifest_path(
                base_dir,
                item.get("mask_png") or item.get("mask") or item.get("mask_crop_png")
            )
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
            mask_path = self._resolve_manifest_path(
                base_dir,
                item.get("mask_png") or item.get("mask") or item.get("mask_crop_png")
            )
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
        save_full, save_crop = self._get_mask_save_flags()
        if not (save_full or save_crop):
            messagebox.showwarning("Nothing to save", "Select at least one mask size: Full or Crop.")
            return
        out = filedialog.askdirectory(title="Choose output folder")
        if not out: return
        out_dir = Path(out); out_dir.mkdir(parents=True, exist_ok=True)

        idxs = list(range(len(self.sr.masks)))
        rows = self._export_masks(idxs, out_dir)

        # optional manifest
        csv_path = out_dir / "mask_manifest.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file","segment_id","mask_png","mask_crop_png","crop_png","area_px2","bbox"])
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
        save_full, save_crop = self._get_mask_save_flags()
        if not (save_full or save_crop):
            messagebox.showwarning("Nothing to save", "Select at least one mask size: Full or Crop.")
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
            writer = csv.DictWriter(f, fieldnames=["file","segment_id","mask_png","mask_crop_png","crop_png","area_px2","bbox"])
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
        save_full, save_crop = self._get_mask_save_flags()
        if not (save_full or save_crop):
            messagebox.showwarning("Nothing to save", "Select at least one mask size: Full or Crop.")
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
            writer = csv.DictWriter(f, fieldnames=["file","segment_id","mask_png","mask_crop_png","crop_png","area_px2","bbox"])
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


    






# =============================================================================
# Entry point
# =============================================================================

def main():
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

if __name__ == "__main__":
    main()
