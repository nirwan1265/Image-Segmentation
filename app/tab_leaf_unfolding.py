#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tab_leaf_unfolding.py
=====================
Builds the "Leaf Unfolding" tab content.

Unfolding works by MIRRORING the selected (folded) mask across the fold line,
not by rotating it. The fold line is auto-detected as the junction boundary
between the two masks. The user can fine-tune the fold line angle and shift
the mirrored piece with drag.

Public function:
    build(app, parent_frame)
"""

import tkinter as tk
from tkinter import ttk


def build(app, tab) -> None:
    """Populate the Leaf Unfolding tab."""

    unfold_container = ttk.Frame(tab)
    unfold_container.grid(row=0, column=0, sticky="nsew")
    tab.grid_columnconfigure(0, weight=1)
    tab.grid_rowconfigure(0, weight=1)

    unfold_container.grid_columnconfigure(0, weight=0)
    unfold_container.grid_columnconfigure(1, weight=1)
    unfold_container.grid_columnconfigure(2, weight=1)
    unfold_container.grid_rowconfigure(0, weight=1)

    # ── COLUMN 0: Options ─────────────────────────────────────────────────────
    opt_frame = ttk.Frame(unfold_container, padding=(0, 4, 10, 4))
    opt_frame.grid(row=0, column=0, sticky="ns")

    ttk.Label(
        opt_frame,
        text=(
            "How it works:\n"
            "1. Select 2+ masks (the folded leaf parts).\n"
            "2. Choose which part is the folded flap.\n"
            "3. The app mirrors it across the fold line.\n"
            "4. Fine-tune angle & position, then Apply."
        ),
        font=("Helvetica", 8), justify="left",
        wraplength=170,
    ).pack(anchor="w", pady=(0, 8))

    app._unfold_preview_btn = ttk.Button(
        opt_frame, text="▶ Preview Unfolding",
        command=app._preview_leaf_unfolding,
        style="Accent.TButton")
    app._unfold_preview_btn.pack(fill="x", pady=(0, 10))

    ttk.Separator(opt_frame, orient="horizontal").pack(fill="x", pady=6)

    # ── Which mask is the folded flap ─────────────────────────────────────────
    ttk.Label(opt_frame, text="Folded part (to mirror):",
              font=("Helvetica", 9, "bold")).pack(anchor="w")
    app._unfold_mask_var = tk.StringVar(value="mask_1")
    app._unfold_mask_frame = ttk.Frame(opt_frame)
    app._unfold_mask_frame.pack(fill="x", pady=(4, 8))
    # Radio buttons populated dynamically

    ttk.Separator(opt_frame, orient="horizontal").pack(fill="x", pady=6)

    # ── Fold line angle (fine-tune) ───────────────────────────────────────────
    ttk.Label(opt_frame, text="Fold line angle:",
              font=("Helvetica", 9, "bold")).pack(anchor="w")
    ttk.Label(opt_frame,
              text="(auto-detected from junction;\nadjust if result looks off)",
              font=("Helvetica", 7), justify="left").pack(anchor="w")

    rot_frame = ttk.Frame(opt_frame)
    rot_frame.pack(fill="x", pady=4)
    ttk.Button(rot_frame, text=" ↶ ", width=3,
               command=lambda: app._unfold_rotate(-5)).pack(side="left", padx=2)
    app._unfold_angle_label = ttk.Label(rot_frame, text="0°", width=6, anchor="center")
    app._unfold_angle_label.pack(side="left", padx=4)
    ttk.Button(rot_frame, text=" ↷ ", width=3,
               command=lambda: app._unfold_rotate(5)).pack(side="left", padx=2)

    quick_frame = ttk.Frame(opt_frame)
    quick_frame.pack(fill="x", pady=2)
    for label, angle in [("90°", 90), ("180°", 180), ("-90°", -90)]:
        ttk.Button(quick_frame, text=label, width=4,
                   command=lambda a=angle: app._unfold_set_rotation(a)
                   ).pack(side="left", padx=2)

    ttk.Button(opt_frame, text="Reset angle",
               command=app._unfold_reset_rotation).pack(fill="x", pady=(6, 4))

    ttk.Separator(opt_frame, orient="horizontal").pack(fill="x", pady=6)

    # ── Position (drag offset) ────────────────────────────────────────────────
    ttk.Label(opt_frame, text="Position offset:",
              font=("Helvetica", 9, "bold")).pack(anchor="w")
    ttk.Label(opt_frame, text="(drag the preview to adjust)",
              font=("Helvetica", 7)).pack(anchor="w")
    app._unfold_offset_label = ttk.Label(opt_frame, text="(0, 0)",
                                          font=("Helvetica", 9))
    app._unfold_offset_label.pack(anchor="w", pady=2)
    ttk.Button(opt_frame, text="Reset Position",
               command=app._unfold_reset_position).pack(fill="x", pady=4)

    ttk.Separator(opt_frame, orient="horizontal").pack(fill="x", pady=6)

    # ── Action buttons ────────────────────────────────────────────────────────
    btn_frame = ttk.Frame(opt_frame)
    btn_frame.pack(fill="x", pady=6)
    app._unfold_update_btn = ttk.Button(
        btn_frame, text="✓ Apply",
        state="disabled",
        command=app._apply_leaf_unfolding,
        style="Accent.TButton")
    app._unfold_update_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
    ttk.Button(btn_frame, text="✗ Cancel",
               command=app._cancel_leaf_unfolding).pack(
        side="left", fill="x", expand=True)

    # ── COLUMN 1: Original ────────────────────────────────────────────────────
    orig_lf = ttk.LabelFrame(unfold_container,
                              text=" Original (Selected Masks) ", padding=4)
    orig_lf.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
    orig_lf.grid_rowconfigure(0, weight=1)
    orig_lf.grid_columnconfigure(0, weight=1)
    app._unfold_orig_canvas = tk.Canvas(
        orig_lf, bg=app.colors['canvas_bg'],
        highlightthickness=1,
        highlightbackground=app.colors['border'])
    app._unfold_orig_canvas.grid(row=0, column=0, sticky="nsew")

    # ── COLUMN 2: Unfolded result ─────────────────────────────────────────────
    unfold_lf = ttk.LabelFrame(unfold_container,
                                text=" Unfolded Preview (mirror of fold) ", padding=4)
    unfold_lf.grid(row=0, column=2, sticky="nsew", padx=4, pady=4)
    unfold_lf.grid_rowconfigure(0, weight=1)
    unfold_lf.grid_columnconfigure(0, weight=1)
    app._unfold_result_canvas = tk.Canvas(
        unfold_lf, bg=app.colors['canvas_bg'],
        highlightthickness=1,
        highlightbackground=app.colors['border'])
    app._unfold_result_canvas.grid(row=0, column=0, sticky="nsew")

    # Drag bindings on result canvas to shift the mirrored piece
    app._unfold_result_canvas.bind("<Button-1>",       app._unfold_drag_start)
    app._unfold_result_canvas.bind("<B1-Motion>",       app._unfold_drag_move)
    app._unfold_result_canvas.bind("<ButtonRelease-1>", app._unfold_drag_end)

    # ── Zoom controls ─────────────────────────────────────────────────────────
    zoom_f = ttk.Frame(unfold_container)
    zoom_f.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(0, 4))
    ttk.Label(zoom_f, text="Zoom:", font=("Helvetica", 9)).pack(
        side="left", padx=(4, 8))
    ttk.Button(zoom_f, text=" − ", width=3,
               command=lambda: app._unfold_zoom_by(0.8)).pack(side="left", padx=2)
    app._unfold_zoom_label = ttk.Label(zoom_f, text="100%", width=5, anchor="center")
    app._unfold_zoom_label.pack(side="left", padx=4)
    ttk.Button(zoom_f, text=" + ", width=3,
               command=lambda: app._unfold_zoom_by(1.25)).pack(side="left", padx=2)
    ttk.Button(zoom_f, text="Fit", width=4,
               command=app._unfold_zoom_fit).pack(side="left", padx=(8, 2))

    # ── Stats ─────────────────────────────────────────────────────────────────
    app._unfold_stats_label = ttk.Label(
        unfold_container,
        text="Select 2+ masks — the smaller one is usually the folded flap",
        font=("Helvetica", 9))
    app._unfold_stats_label.grid(row=2, column=0, columnspan=3,
                                  sticky="w", pady=(4, 0))

    # ── State init ────────────────────────────────────────────────────────────
    app._unfold_rotation_angle = 0
    app._unfold_offset         = [0, 0]
    app._unfold_zoom           = 1.0
    app._unfold_masks          = []
    app._unfold_pending        = None
    app._unfold_drag_start_pos = None
    # pivot_var still needed by _get_unfold_pivot for fold line detection
    app._unfold_pivot_var      = tk.StringVar(value="junction")
