#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
color_filter.py
===============
Color-based mask filter with an interactive 2D HSV picker.

The picker shows a Hue × Saturation plane (with a separate Value slider).
Drag to draw selection rectangles on the plane — multiple rectangles are
supported (OR logic).  Each selection is shown as a colour chip below.

Public API (unchanged — gui_app.py needs no changes):
  attach(app, parent)               — build the panel
  on_canvas_click_pick(app, event)  — eyedropper pick from preview canvas
  draw_highlights(app)              — overlay bounding boxes on preview
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2

# ── Constants ────────────────────────────────────────────────────────────────
PICKER_W   = 360   # width  of HSV plane canvas (maps to Hue 0–179 × 2)
PICKER_H   = 180   # height of HSV plane canvas (maps to Sat 0–255)
CHIP_SIZE  = 20    # colour chip square size in px


# =============================================================================
# Public API
# =============================================================================

def attach(app, parent) -> None:
    _init_state(app)
    _build_panel(app, parent)
    _render_picker(app)


def on_canvas_click_pick(app, event) -> None:
    """Eyedropper: sample pixel from preview canvas → add as new selection."""
    if not getattr(app, "_color_pick_active", False):
        return
    app._color_pick_active = False

    ix, iy = app._canvas_to_image_xy(event.x, event.y)
    img = getattr(app, "img_orig", None)
    if img is None:
        _restore_canvas_mode(app)
        return

    H, W = img.shape[:2]
    ix = max(0, min(W - 1, ix))
    iy = max(0, min(H - 1, iy))
    r, g, b = int(img[iy, ix, 0]), int(img[iy, ix, 1]), int(img[iy, ix, 2])

    _add_selection_from_rgb(app, r, g, b)
    _restore_canvas_mode(app)
    _refresh_preview(app)


def draw_highlights(app) -> None:
    """Draw coloured bounding boxes on the preview canvas."""
    highlighted = getattr(app, "_cf_highlighted", set())
    if not highlighted or not getattr(app, "sr", None):
        return
    mode = getattr(app, "_cf_highlight_mode", "remove")
    col_match = "#ef5350" if mode == "remove" else "#40c074"
    col_other = "#40c074" if mode == "remove" else "#ef5350"
    for i, m in enumerate(app.sr.masks):
        x, y, w, h = map(int, m["bbox"])
        cx1, cy1 = app._image_to_canvas_xy(x,     y)
        cx2, cy2 = app._image_to_canvas_xy(x + w, y + h)
        color = col_match if i in highlighted else col_other
        app.canvas.create_rectangle(
            cx1, cy1, cx2, cy2,
            outline=color, width=2, dash=(5, 3),
            tags=("cf_highlight",))


# =============================================================================
# State
# =============================================================================

def _init_state(app) -> None:
    # Each selection: dict with keys h0,h1,s0,s1 (all 0-1 normalised)
    app._cf_selections: list[dict] = []
    app._cf_mode       = tk.StringVar(value="remove")
    app._cf_v_low      = tk.IntVar(value=0)
    app._cf_v_high     = tk.IntVar(value=255)
    app._color_pick_active = False
    app._cf_highlighted: set = set()
    app._cf_highlight_mode  = "remove"

    # Drag state for the picker canvas
    app._cf_drag_start = None   # (x, y) in canvas coords
    app._cf_drag_rect  = None   # canvas item id of the rubber-band rect

    # Last eyedropper pick position (normalised 0-1 h, s) for the dot marker
    app._cf_last_pick_hs = None   # tuple (h_norm, s_norm) or None


# =============================================================================
# Panel builder
# =============================================================================

def _build_panel(app, parent) -> None:
    c = app.colors

    outer = tk.Frame(parent, bg=c["bg_medium"])
    outer.pack(fill="both", expand=True, padx=4, pady=4)

    # ── Row 0: eyedropper + reset ─────────────────────────────────────────────
    row0 = tk.Frame(outer, bg=c["bg_medium"])
    row0.pack(fill="x", pady=(0, 6))

    ttk.Button(row0, text="⛏ Pick from image",
               command=lambda: _enter_pick_mode(app)).pack(side="left")
    ttk.Button(row0, text="✖ Clear all",
               command=lambda: _clear_all_selections(app)).pack(
        side="left", padx=(8, 0))

    app._cf_count_label = tk.Label(
        row0, text="", bg=c["bg_medium"],
        fg=c["text_muted"], font=("Helvetica", 9))
    app._cf_count_label.pack(side="right", padx=(0, 4))

    # ── Row 1: 2D HSV picker canvas ───────────────────────────────────────────
    picker_label = tk.Label(outer, text="Hue →                  Saturation ↑",
                             bg=c["bg_medium"], fg=c["text_muted"],
                             font=("Helvetica", 8))
    picker_label.pack(anchor="w")

    picker_frame = tk.Frame(outer, bg=c["bg_dark"],
                             highlightthickness=1,
                             highlightbackground=c["border"])
    picker_frame.pack(fill="x", pady=(0, 4))

    app._cf_picker = tk.Canvas(
        picker_frame,
        width=PICKER_W, height=PICKER_H,
        highlightthickness=0, cursor="crosshair")
    app._cf_picker.pack(fill="x", expand=True)

    # Axis labels
    label_row = tk.Frame(outer, bg=c["bg_medium"])
    label_row.pack(fill="x")
    tk.Label(label_row, text="0°", bg=c["bg_medium"],
             fg=c["text_muted"], font=("Helvetica", 7)).pack(side="left")
    tk.Label(label_row, text="Red", bg=c["bg_medium"],
             fg=c["text_muted"], font=("Helvetica", 7)).pack(side="left", padx=(4,0))
    tk.Label(label_row, text="Yellow", bg=c["bg_medium"],
             fg=c["text_muted"], font=("Helvetica", 7)).pack(side="left", padx=(18,0))
    tk.Label(label_row, text="Green", bg=c["bg_medium"],
             fg=c["text_muted"], font=("Helvetica", 7)).pack(side="left", padx=(18,0))
    tk.Label(label_row, text="Cyan", bg=c["bg_medium"],
             fg=c["text_muted"], font=("Helvetica", 7)).pack(side="left", padx=(12,0))
    tk.Label(label_row, text="Blue", bg=c["bg_medium"],
             fg=c["text_muted"], font=("Helvetica", 7)).pack(side="left", padx=(10,0))
    tk.Label(label_row, text="179°", bg=c["bg_medium"],
             fg=c["text_muted"], font=("Helvetica", 7)).pack(side="right")

    # Picker mouse bindings
    app._cf_picker.bind("<ButtonPress-1>",   lambda e: _drag_start(app, e))
    app._cf_picker.bind("<B1-Motion>",        lambda e: _drag_move(app, e))
    app._cf_picker.bind("<ButtonRelease-1>",  lambda e: _drag_end(app, e))

    # ── Row 2: Value (brightness) range ──────────────────────────────────────
    vrow = tk.Frame(outer, bg=c["bg_medium"])
    vrow.pack(fill="x", pady=(6, 2))
    tk.Label(vrow, text="Brightness (V):", bg=c["bg_medium"],
             fg=c["text_light"], font=("Helvetica", 9)).pack(side="left")
    ttk.Spinbox(vrow, from_=0, to=255, width=4,
                textvariable=app._cf_v_low,
                command=lambda: _refresh_preview(app)).pack(side="left", padx=(6,2))
    tk.Label(vrow, text="–", bg=c["bg_medium"],
             fg=c["text_muted"]).pack(side="left")
    ttk.Spinbox(vrow, from_=0, to=255, width=4,
                textvariable=app._cf_v_high,
                command=lambda: _refresh_preview(app)).pack(side="left", padx=(2,8))

    # Visual bar for V range
    app._cf_v_bar = tk.Canvas(vrow, width=120, height=12,
                               bg=c["bg_dark"], highlightthickness=0)
    app._cf_v_bar.pack(side="left")

    def _redraw_v(*_):
        app._cf_v_bar.delete("all")
        x0 = app._cf_v_low.get()  / 255 * 120
        x1 = app._cf_v_high.get() / 255 * 120
        app._cf_v_bar.create_rectangle(0, 1, 120, 11,
                                        fill=c["bg_medium"], outline="")
        app._cf_v_bar.create_rectangle(x0, 1, max(x0+2, x1), 11,
                                        fill=c["accent"], outline="")
    app._cf_v_low.trace_add("write",  lambda *_: _redraw_v())
    app._cf_v_high.trace_add("write", lambda *_: _redraw_v())
    _redraw_v()

    # ── Row 3: Active selections chips ───────────────────────────────────────
    tk.Label(outer, text="Active selections  (drag on picker to add):",
             bg=c["bg_medium"], fg=c["text_muted"],
             font=("Helvetica", 8)).pack(anchor="w", pady=(8, 2))

    app._cf_chips_frame = tk.Frame(outer, bg=c["bg_medium"])
    app._cf_chips_frame.pack(fill="x", pady=(0, 6))
    _rebuild_chips(app)

    # ── Row 4: Mode ───────────────────────────────────────────────────────────
    row4 = tk.Frame(outer, bg=c["bg_medium"])
    row4.pack(fill="x", pady=(0, 4))
    tk.Label(row4, text="Action:", bg=c["bg_medium"],
             fg=c["text_light"], font=("Helvetica", 9)).pack(side="left")
    ttk.Radiobutton(row4, text="Remove matching",
                    variable=app._cf_mode, value="remove").pack(
        side="left", padx=(6, 12))
    ttk.Radiobutton(row4, text="Keep matching only",
                    variable=app._cf_mode, value="keep").pack(side="left")

    # ── Row 5: Action buttons ─────────────────────────────────────────────────
    row5 = tk.Frame(outer, bg=c["bg_medium"])
    row5.pack(fill="x", pady=(4, 0))
    ttk.Button(row5, text="👁 Preview",
               command=lambda: _refresh_preview(app)).pack(side="left")
    ttk.Button(row5, text="✓ Apply",
               command=lambda: _apply_filter(app),
               style="Accent.TButton").pack(side="left", padx=(8, 0))
    ttk.Button(row5, text="✗ Clear highlights",
               command=lambda: _clear_highlights(app)).pack(
        side="left", padx=(8, 0))


# =============================================================================
# HSV picker canvas rendering
# =============================================================================

def _render_picker(app) -> None:
    """Draw the Hue × Saturation gradient onto the picker canvas."""
    try:
        w = app._cf_picker.winfo_width()
        h = app._cf_picker.winfo_height()
        if w < 10:
            w, h = PICKER_W, PICKER_H
    except Exception:
        w, h = PICKER_W, PICKER_H

    # Build HSV image: x=Hue (0-179), y=Saturation (255→0 top to bottom)
    hue_row  = np.linspace(0, 179, w, dtype=np.uint8)
    sat_col  = np.linspace(255, 30, h, dtype=np.uint8)
    H_plane  = np.tile(hue_row,  (h, 1))
    S_plane  = np.tile(sat_col,  (w, 1)).T
    V_plane  = np.full((h, w), 220, dtype=np.uint8)

    hsv_img = np.stack([H_plane, S_plane, V_plane], axis=2).astype(np.uint8)
    rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

    # Convert to PhotoImage via PPM bytes (no PIL needed)
    rows, cols = rgb_img.shape[:2]
    ppm = (f"P6\n{cols} {rows}\n255\n").encode() + rgb_img.tobytes()
    app._cf_picker_img = tk.PhotoImage(data=ppm)
    app._cf_picker.delete("all")
    app._cf_picker.create_image(0, 0, image=app._cf_picker_img, anchor="nw",
                                 tags="bg")

    # Redraw selections on top
    _redraw_selection_rects(app)

    # Redraw on resize
    app._cf_picker.bind("<Configure>", lambda e: _render_picker(app))


def _redraw_selection_rects(app) -> None:
    """Draw all selection rectangles over the picker."""
    app._cf_picker.delete("sel_rect")
    try:
        w = max(1, app._cf_picker.winfo_width())
        h = max(1, app._cf_picker.winfo_height())
    except Exception:
        w, h = PICKER_W, PICKER_H

    for i, sel in enumerate(app._cf_selections):
        x0 = sel["h0"] * w
        x1 = sel["h1"] * w
        y0 = (1 - sel["s1"]) * h   # s=1 → top, s=0 → bottom
        y1 = (1 - sel["s0"]) * h
        col = _sel_color(sel)
        app._cf_picker.create_rectangle(
            x0, y0, x1, y1,
            outline=col, width=2, fill=col,
            stipple="gray25", tags="sel_rect")
        # Index label
        app._cf_picker.create_text(
            (x0 + x1) / 2, (y0 + y1) / 2,
            text=str(i + 1),
            fill="white", font=("Helvetica", 9, "bold"),
            tags="sel_rect")

    # Draw eyedropper crosshair dot if a pick has been made
    _draw_pick_dot(app)


# =============================================================================
# Drag interaction on picker
# =============================================================================

def _draw_pick_dot(app) -> None:
    """Draw a crosshair marker on the picker at the last eyedropper position."""
    app._cf_picker.delete("pick_dot")
    pos = getattr(app, "_cf_last_pick_hs", None)
    if pos is None:
        return
    h_n, s_n = pos
    try:
        w = max(1, app._cf_picker.winfo_width())
        h = max(1, app._cf_picker.winfo_height())
    except Exception:
        w, h = PICKER_W, PICKER_H

    px = h_n * w
    py = (1 - s_n) * h   # s=1 → top

    r = 7   # crosshair radius
    # Outer white ring
    app._cf_picker.create_oval(
        px - r, py - r, px + r, py + r,
        outline="white", width=2, tags="pick_dot")
    # Inner black ring (contrast on bright backgrounds)
    app._cf_picker.create_oval(
        px - r + 2, py - r + 2, px + r - 2, py + r - 2,
        outline="black", width=1, tags="pick_dot")
    # Centre dot
    app._cf_picker.create_oval(
        px - 2, py - 2, px + 2, py + 2,
        fill="white", outline="black", width=1, tags="pick_dot")
    # Crosshair lines
    app._cf_picker.create_line(
        px - r - 3, py, px + r + 3, py,
        fill="white", width=1, dash=(3, 2), tags="pick_dot")
    app._cf_picker.create_line(
        px, py - r - 3, px, py + r + 3,
        fill="white", width=1, dash=(3, 2), tags="pick_dot")


def _drag_start(app, e) -> None:
    app._cf_drag_start = (e.x, e.y)
    app._cf_drag_rect = app._cf_picker.create_rectangle(
        e.x, e.y, e.x, e.y,
        outline="#ffffff", width=2, dash=(4, 2), tags="rubber")


def _drag_move(app, e) -> None:
    if app._cf_drag_start is None or app._cf_drag_rect is None:
        return
    x0, y0 = app._cf_drag_start
    app._cf_picker.coords(app._cf_drag_rect, x0, y0, e.x, e.y)


def _drag_end(app, e) -> None:
    if app._cf_drag_start is None:
        return
    x0, y0 = app._cf_drag_start
    x1, y1 = e.x, e.y
    app._cf_drag_start = None
    app._cf_drag_rect  = None
    app._cf_picker.delete("rubber")

    # Ignore tiny clicks (< 5px in either axis)
    if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
        return

    # Convert canvas coords → normalised h, s
    try:
        w = max(1, app._cf_picker.winfo_width())
        h = max(1, app._cf_picker.winfo_height())
    except Exception:
        w, h = PICKER_W, PICKER_H

    h0_n = max(0.0, min(1.0, min(x0, x1) / w))
    h1_n = max(0.0, min(1.0, max(x0, x1) / w))
    # y=0 → top → high saturation
    s0_n = max(0.0, min(1.0, 1 - max(y0, y1) / h))
    s1_n = max(0.0, min(1.0, 1 - min(y0, y1) / h))

    # Sample representative colour from centre of selection
    h_mid = (h0_n + h1_n) / 2
    s_mid = (s0_n + s1_n) / 2
    h_cv  = int(h_mid * 179)
    s_cv  = int(s_mid * 255)
    v_cv  = 200
    rgb   = cv2.cvtColor(
        np.array([[[h_cv, s_cv, v_cv]]], dtype=np.uint8),
        cv2.COLOR_HSV2RGB)[0, 0]
    rep_color = f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"

    sel = dict(h0=h0_n, h1=h1_n, s0=s0_n, s1=s1_n, color=rep_color)
    app._cf_selections.append(sel)

    _redraw_selection_rects(app)
    _rebuild_chips(app)
    _refresh_preview(app)


# =============================================================================
# Selection chips (below the picker)
# =============================================================================

def _sel_color(sel: dict) -> str:
    return sel.get("color", "#40c074")


def _rebuild_chips(app) -> None:
    """Rebuild the row of colour chips, one per selection."""
    c = app.colors
    for child in app._cf_chips_frame.winfo_children():
        child.destroy()

    if not app._cf_selections:
        tk.Label(app._cf_chips_frame, text="None — drag on the picker above",
                 bg=c["bg_medium"], fg=c["text_muted"],
                 font=("Helvetica", 8)).pack(side="left")
        return

    for i, sel in enumerate(app._cf_selections):
        chip = tk.Frame(app._cf_chips_frame, bg=c["bg_medium"])
        chip.pack(side="left", padx=(0, 6))

        # Colour square
        sq = tk.Canvas(chip, width=CHIP_SIZE, height=CHIP_SIZE,
                        highlightthickness=1,
                        highlightbackground=c["border"])
        sq.configure(bg=_sel_color(sel))
        sq.pack(side="left")

        # Label showing hue range
        h_lo = int(sel["h0"] * 179)
        h_hi = int(sel["h1"] * 179)
        s_lo = int(sel["s0"] * 255)
        s_hi = int(sel["s1"] * 255)
        lbl = tk.Label(chip,
                        text=f"H{h_lo}–{h_hi} S{s_lo}–{s_hi}",
                        bg=c["bg_medium"], fg=c["text_light"],
                        font=("Helvetica", 7))
        lbl.pack(side="left", padx=(3, 0))

        # Remove button
        idx = i
        rm = tk.Label(chip, text="✕", bg=c["bg_medium"],
                       fg=c["error"], font=("Helvetica", 9),
                       cursor="hand2")
        rm.pack(side="left", padx=(3, 0))
        rm.bind("<Button-1>", lambda e, j=idx: _remove_selection(app, j))


def _remove_selection(app, idx: int) -> None:
    if 0 <= idx < len(app._cf_selections):
        app._cf_selections.pop(idx)
    _redraw_selection_rects(app)
    _rebuild_chips(app)
    _refresh_preview(app)


def _clear_all_selections(app) -> None:
    app._cf_selections.clear()
    _redraw_selection_rects(app)
    _rebuild_chips(app)
    _clear_highlights(app)


# =============================================================================
# Pick mode (eyedropper from preview canvas)
# =============================================================================

def _enter_pick_mode(app) -> None:
    app._color_pick_active = True
    app.canvas.configure(cursor="crosshair")
    app.canvas.bind("<ButtonPress-1>",
                    lambda e: on_canvas_click_pick(app, e))
    app.set_status("Click on the image to pick a color…", "info")


def _restore_canvas_mode(app) -> None:
    app.canvas.configure(cursor="tcross")
    app.canvas.bind("<ButtonPress-1>", app._pan_start)
    app.set_status("Color picked — selection added.", "success")


def _add_selection_from_rgb(app, r: int, g: int, b: int) -> None:
    """Convert picked RGB → HSV and add a selection rect + crosshair dot."""
    hsv = cv2.cvtColor(
        np.array([[[r, g, b]]], dtype=np.uint8),
        cv2.COLOR_RGB2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    tol_h, tol_s = 20, 60
    h0 = max(0,   h - tol_h) / 179
    h1 = min(179, h + tol_h) / 179
    s0 = max(0,   s - tol_s) / 255
    s1 = min(255, s + tol_s) / 255

    rep_color = f"#{r:02x}{g:02x}{b:02x}"
    app._cf_selections.append(
        dict(h0=h0, h1=h1, s0=s0, s1=s1, color=rep_color))

    # Store normalised H/S position so the crosshair dot can be drawn
    app._cf_last_pick_hs = (h / 179, s / 255)

    # Also set V range
    app._cf_v_low.set(max(0,   v - 60))
    app._cf_v_high.set(min(255, v + 60))

    _redraw_selection_rects(app)   # redraws rects AND the dot
    _rebuild_chips(app)


# =============================================================================
# Matching logic
# =============================================================================

def _mask_matches(mask_dict: dict, img_rgb: np.ndarray,
                  selections: list, v_low: int, v_high: int) -> bool:
    """Return True if mask mean colour falls in ANY of the selections (OR logic)."""
    seg = mask_dict.get("segmentation")
    if seg is None:
        return False
    seg_bool = (np.asarray(seg) > 0)
    if not seg_bool.any():
        return False

    ih, iw = img_rgb.shape[:2]
    mh, mw = seg_bool.shape[:2]
    if mh != ih or mw != iw:
        seg_bool = cv2.resize(
            seg_bool.astype(np.uint8), (iw, ih),
            interpolation=cv2.INTER_NEAREST).astype(bool)

    pixels = img_rgb[seg_bool]
    if len(pixels) == 0:
        return False

    mean_rgb = pixels.mean(axis=0).astype(np.uint8)
    mean_hsv = cv2.cvtColor(
        mean_rgb.reshape(1, 1, 3),
        cv2.COLOR_RGB2HSV)[0, 0]
    h, s, v = int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])

    # V filter applies to ALL selections
    if not (v_low <= v <= v_high):
        return False

    # Match if inside ANY selection rectangle
    for sel in selections:
        h0 = sel["h0"] * 179
        h1 = sel["h1"] * 179
        s0 = sel["s0"] * 255
        s1 = sel["s1"] * 255
        if h0 <= h <= h1 and s0 <= s <= s1:
            return True
    return False


def _get_matching_indices(app) -> list:
    if not getattr(app, "sr", None) or not app.sr.masks:
        return []
    img = getattr(app, "img_orig", None)
    if img is None:
        return []
    if not app._cf_selections:
        return []

    v_low  = app._cf_v_low.get()
    v_high = app._cf_v_high.get()

    return [
        i for i, m in enumerate(app.sr.masks)
        if _mask_matches(m, img, app._cf_selections, v_low, v_high)
    ]


# =============================================================================
# Preview
# =============================================================================

def _refresh_preview(app) -> None:
    matching = _get_matching_indices(app)
    mode = app._cf_mode.get()
    n_total = len(app.sr.masks) if getattr(app, "sr", None) and app.sr.masks else 0

    if not app._cf_selections:
        app._cf_count_label.configure(text="No selections — drag on picker")
        return

    if mode == "remove":
        label = f"Would remove {len(matching)} / {n_total} masks"
    else:
        label = f"Would keep {len(matching)} / {n_total} masks"
    app._cf_count_label.configure(text=label)

    app._cf_highlighted    = set(matching)
    app._cf_highlight_mode = mode

    if hasattr(app, "_render_preview"):
        app._render_preview()


# =============================================================================
# Apply
# =============================================================================

def _apply_filter(app) -> None:
    if not getattr(app, "sr", None) or not app.sr.masks:
        return
    if not app._cf_selections:
        app.set_status("No colour selections — drag on the picker first.", "info")
        return

    matching = _get_matching_indices(app)
    if not matching:
        app.set_status("No masks match the current selections.", "info")
        return

    mode = app._cf_mode.get()
    if hasattr(app, "_push_undo"):
        app._push_undo("color filter")

    to_delete = (set(matching) if mode == "remove"
                 else set(range(len(app.sr.masks))) - set(matching))

    for i in sorted(to_delete, reverse=True):
        if 0 <= i < len(app.sr.masks):
            del app.sr.masks[i]

    _clear_highlights(app)
    app._rebuild_mask_list()
    if hasattr(app, "_sync_listbox_selection_from_picks"):
        app._sync_listbox_selection_from_picks()
    if getattr(app, "img_preview", None) is not None:
        app.show_image(app.img_preview)
    elif getattr(app, "img", None) is not None:
        app.show_image(app.img)
    app.set_status(
        f"Color filter: removed {len(to_delete)} masks, "
        f"{len(app.sr.masks)} remaining.", "success")


# =============================================================================
# Helpers
# =============================================================================

def _clear_highlights(app) -> None:
    app._cf_highlighted = set()
    app._cf_count_label.configure(text="")
    if hasattr(app, "_render_preview"):
        app._render_preview()
