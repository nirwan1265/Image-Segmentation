#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tab_train.py
============
Builds the "Train Custom Model" tab content.

Public function:
    build(app, parent_frame)
        app          — the LeafSegmenterGUI instance (for self.* vars + commands)
        parent_frame — the scrollable inner frame returned by _make_scrollable_tab()
"""

import tkinter as tk
from tkinter import ttk


def build(app, tab):
    """Populate the Train Custom Model tab."""

    # ── Description ──────────────────────────────────────────────────────────
    ttk.Label(
        tab,
        text=(
            "Train a custom segmentation model (no SAM needed at inference).\n"
            "Collect masks, train the model, then use it directly for segmentation."
        ),
        wraplength=700, justify="left",
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

    tab.grid_columnconfigure(1, weight=1)

    # ── Dataset folder ────────────────────────────────────────────────────────
    r = 1
    ttk.Label(tab, text="Dataset folder:").grid(row=r, column=0, sticky="e", padx=(0, 4))
    ttk.Entry(tab, textvariable=app.target_root_var, width=50).grid(
        row=r, column=1, sticky="ew", padx=4)
    ttk.Button(tab, text="Choose...", command=app._pick_target_root).grid(row=r, column=2)

    # ── Status + resume ───────────────────────────────────────────────────────
    r += 1
    app.target_msg = ttk.Label(tab, text="0 examples", anchor="w")
    app.target_msg.grid(row=r, column=0, columnspan=2, sticky="w")
    ttk.Checkbutton(tab, text="Resume from existing model",
                    variable=app.target_resume_var).grid(row=r, column=2, sticky="e")

    # ── Action toolbar ────────────────────────────────────────────────────────
    r += 1
    tbar = ttk.Frame(tab)
    tbar.grid(row=r, column=0, columnspan=3, sticky="w", pady=(2, 4))
    ttk.Button(tbar, text="Add selected masks as target",
               command=app._add_current_to_target).pack(side="left")
    ttk.Button(tbar, text="Mark image as NO target",
               command=app._add_negative_target).pack(side="left", padx=(8, 0))
    ttk.Button(tbar, text="Open dataset folder",
               command=app._open_target_root).pack(side="left", padx=(8, 0))
    ttk.Button(tbar, text="Scan dataset",
               command=app._scan_target_dataset).pack(side="left", padx=(8, 0))
    ttk.Button(tbar, text="Clear dataset",
               command=app._clear_target_set).pack(side="left", padx=(8, 0))

    # ── Training hyperparameters grid ─────────────────────────────────────────
    r += 1
    grid1 = ttk.Frame(tab)
    grid1.grid(row=r, column=0, columnspan=3, sticky="ew")

    ttk.Label(grid1, text="Save to (.pth)").grid(row=0, column=0, sticky="e")
    ttk.Entry(grid1, textvariable=app.target_out_var, width=48).grid(
        row=0, column=1, sticky="ew", padx=4)
    ttk.Button(grid1, text="...",
               command=lambda: app._browse_save_into(
                   app.target_out_var, default_ext=".pth")
               ).grid(row=0, column=2)

    ttk.Label(grid1, text="Steps").grid(row=1, column=0, sticky="e")
    ttk.Spinbox(grid1, from_=100, to=200000, increment=100,
                textvariable=app.target_steps_var, width=10).grid(
        row=1, column=1, sticky="w", padx=4)
    ttk.Label(grid1, text="LR").grid(row=1, column=2, sticky="e")
    ttk.Entry(grid1, textvariable=app.target_lr_var, width=10).grid(
        row=1, column=3, sticky="w", padx=4)

    ttk.Label(grid1, text="Image size").grid(row=2, column=0, sticky="e")
    ttk.Spinbox(grid1, from_=256, to=2048, increment=128,
                textvariable=app.target_size_var, width=10).grid(
        row=2, column=1, sticky="w", padx=4)
    ttk.Label(grid1, text="Device").grid(row=2, column=2, sticky="e")
    ttk.Combobox(grid1, textvariable=app.target_device_var,
                 values=["mps", "cuda", "cpu"], width=8).grid(
        row=2, column=3, sticky="w", padx=4)

    ttk.Label(grid1, text="Batch size").grid(row=3, column=0, sticky="e")
    ttk.Spinbox(grid1, from_=1, to=32, increment=1,
                textvariable=app.target_batch_var, width=8).grid(
        row=3, column=1, sticky="w", padx=4)
    ttk.Label(grid1, text="Model").grid(row=3, column=2, sticky="e")
    ttk.Combobox(grid1, textvariable=app.target_arch_var,
                 values=["unet_resnet18", "unet_small"], width=14).grid(
        row=3, column=3, sticky="w", padx=4)
    ttk.Checkbutton(grid1, text="Pretrained encoder",
                    variable=app.target_pretrained_var).grid(
        row=3, column=4, sticky="w", padx=(6, 0))
    ttk.Checkbutton(grid1, text="Allow empty (no-target) images",
                    variable=app.target_allow_empty_var).grid(
        row=3, column=2, columnspan=2, sticky="w", padx=4)
    grid1.grid_columnconfigure(1, weight=1)

    # ── Train / Load buttons ──────────────────────────────────────────────────
    r += 1
    tbar2 = ttk.Frame(tab)
    tbar2.grid(row=r, column=0, columnspan=3, sticky="w", pady=(4, 0))
    ttk.Button(tbar2, text="Train Model",
               command=app._launch_target_training).pack(side="left")
    ttk.Button(tbar2, text="Load Model",
               command=app._load_target_model).pack(side="left", padx=(8, 0))

    # ── Inference Settings ────────────────────────────────────────────────────
    r += 1
    tipf = ttk.LabelFrame(tab, text=" Inference Settings ", padding=8)
    tipf.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(8, 4))

    ttk.Checkbutton(tipf, text="Use custom model for segmentation (no SAM)",
                    variable=app.target_use_tipseg).grid(
        row=0, column=0, columnspan=4, sticky="w")

    ttk.Label(tipf, text="Threshold").grid(row=1, column=0, sticky="e", pady=(4, 0))
    ttk.Spinbox(tipf, from_=0.05, to=0.95, increment=0.05,
                textvariable=app.target_tipseg_thresh, width=6).grid(
        row=1, column=1, sticky="w", pady=(4, 0))
    ttk.Label(tipf, text="Min area (px)").grid(
        row=1, column=2, sticky="e", padx=(16, 4), pady=(4, 0))
    ttk.Spinbox(tipf, from_=0, to=200000, increment=50,
                textvariable=app.target_tipseg_min_area, width=8).grid(
        row=1, column=3, sticky="w", pady=(4, 0))
    ttk.Checkbutton(tipf, text="Keep largest component",
                    variable=app.target_tipseg_keep_largest).grid(
        row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))

    ttk.Checkbutton(tipf, text="Sliding window (large images)",
                    variable=app.tipseg_use_tiles).grid(
        row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))
    ttk.Label(tipf, text="Tile").grid(row=4, column=0, sticky="e", pady=(4, 0))
    ttk.Spinbox(tipf, from_=128, to=2048, increment=64,
                textvariable=app.tipseg_tile_size, width=8).grid(
        row=4, column=1, sticky="w", pady=(4, 0))
    ttk.Label(tipf, text="Stride").grid(
        row=4, column=2, sticky="e", padx=(16, 4), pady=(4, 0))
    ttk.Spinbox(tipf, from_=64, to=2048, increment=64,
                textvariable=app.tipseg_stride, width=8).grid(
        row=4, column=3, sticky="w", pady=(4, 0))
    ttk.Checkbutton(tipf, text="Color-guided scan (faster)",
                    variable=app.tipseg_color_guided).grid(
        row=5, column=0, columnspan=3, sticky="w", pady=(4, 0))
    ttk.Label(tipf, text="Color area min").grid(
        row=5, column=2, sticky="e", padx=(16, 4), pady=(4, 0))
    ttk.Spinbox(tipf, from_=0, to=50000, increment=50,
                textvariable=app.tipseg_color_min_area, width=8).grid(
        row=5, column=3, sticky="w", pady=(4, 0))

    ttk.Label(tipf, text="Hue low/high").grid(row=6, column=0, sticky="e", pady=(4, 0))
    ttk.Spinbox(tipf, from_=0, to=179, increment=1,
                textvariable=app.tipseg_hue_low, width=6).grid(
        row=6, column=1, sticky="w", pady=(4, 0))
    ttk.Spinbox(tipf, from_=0, to=179, increment=1,
                textvariable=app.tipseg_hue_high, width=6).grid(
        row=6, column=2, sticky="w", pady=(4, 0))
    ttk.Label(tipf, text="Sat min").grid(
        row=6, column=3, sticky="e", padx=(8, 4), pady=(4, 0))
    ttk.Spinbox(tipf, from_=0, to=255, increment=5,
                textvariable=app.tipseg_sat_min, width=6).grid(
        row=6, column=4, sticky="w", pady=(4, 0))

    ttk.Label(tipf, text="Val min").grid(row=7, column=0, sticky="e", pady=(4, 0))
    ttk.Spinbox(tipf, from_=0, to=255, increment=5,
                textvariable=app.tipseg_val_min, width=6).grid(
        row=7, column=1, sticky="w", pady=(4, 0))
    ttk.Label(tipf, text="Brown V max").grid(
        row=7, column=2, sticky="e", padx=(8, 4), pady=(4, 0))
    ttk.Spinbox(tipf, from_=0, to=255, increment=5,
                textvariable=app.tipseg_val_brown_max, width=6).grid(
        row=7, column=3, sticky="w", pady=(4, 0))

    ttk.Label(tipf, text="Min leaf %").grid(row=8, column=0, sticky="e", pady=(4, 0))
    ttk.Spinbox(tipf, from_=0.0, to=100.0, increment=0.5,
                textvariable=app.tipseg_min_leaf_pct, width=6).grid(
        row=8, column=1, sticky="w", pady=(4, 0))
    ttk.Label(tipf, text="Min stress %").grid(
        row=8, column=2, sticky="e", padx=(8, 4), pady=(4, 0))
    ttk.Spinbox(tipf, from_=0.0, to=100.0, increment=0.5,
                textvariable=app.tipseg_min_stress_pct, width=6).grid(
        row=8, column=3, sticky="w", pady=(4, 0))
    ttk.Checkbutton(tipf, text="Stop after first hit",
                    variable=app.tipseg_stop_after_first).grid(
        row=9, column=0, columnspan=3, sticky="w", pady=(4, 0))

    ttk.Checkbutton(tipf, text="Remove white background",
                    variable=app.tipseg_remove_white).grid(
        row=10, column=0, columnspan=3, sticky="w", pady=(4, 0))
    ttk.Label(tipf, text="White sat max").grid(row=11, column=0, sticky="e", pady=(4, 0))
    ttk.Spinbox(tipf, from_=0, to=255, increment=5,
                textvariable=app.tipseg_white_sat_max, width=6).grid(
        row=11, column=1, sticky="w", pady=(4, 0))
    ttk.Label(tipf, text="White val min").grid(
        row=11, column=2, sticky="e", padx=(8, 4), pady=(4, 0))
    ttk.Spinbox(tipf, from_=0, to=255, increment=5,
                textvariable=app.tipseg_white_val_min, width=6).grid(
        row=11, column=3, sticky="w", pady=(4, 0))

    ttk.Checkbutton(tipf, text="Remove green leaf",
                    variable=app.tipseg_remove_green).grid(
        row=12, column=0, columnspan=3, sticky="w", pady=(4, 0))
    ttk.Label(tipf, text="Green hue low/high").grid(
        row=13, column=0, sticky="e", pady=(4, 0))
    ttk.Spinbox(tipf, from_=0, to=179, increment=1,
                textvariable=app.tipseg_green_hue_low, width=6).grid(
        row=13, column=1, sticky="w", pady=(4, 0))
    ttk.Spinbox(tipf, from_=0, to=179, increment=1,
                textvariable=app.tipseg_green_hue_high, width=6).grid(
        row=13, column=2, sticky="w", pady=(4, 0))
    ttk.Label(tipf, text="Green sat min").grid(
        row=13, column=3, sticky="e", padx=(8, 4), pady=(4, 0))
    ttk.Spinbox(tipf, from_=0, to=255, increment=5,
                textvariable=app.tipseg_green_sat_min, width=6).grid(
        row=13, column=4, sticky="w", pady=(4, 0))
    ttk.Label(tipf, text="Green val min").grid(row=14, column=0, sticky="e", pady=(4, 0))
    ttk.Spinbox(tipf, from_=0, to=255, increment=5,
                textvariable=app.tipseg_green_val_min, width=6).grid(
        row=14, column=1, sticky="w", pady=(4, 0))
