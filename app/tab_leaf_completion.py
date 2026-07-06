#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tab_leaf_completion.py
======================
Builds the "Leaf Completion" tab content.

Two shape galleries in a sub-notebook:
  • Geometric Shapes  — classic botanical forms (Ovate, Lanceolate, …)
  • Crop Species      — real plant leaf silhouettes (Maize, Tomato, …)

Public function:
    build(app, parent_frame)
"""

import tkinter as tk
from tkinter import ttk


# =============================================================================
# Geometric shape registry
# =============================================================================

LEAF_SHAPES = [
    "Orbicular",    "Elliptical",   "Ovate",
    "Obovate",      "Cordate",      "Reniform",
    "Lanceolate",   "Oblanceolate", "Oblong",
    "Linear",       "Ensiform",     "Cuneate",
    "Spathulate",   "Deltoid",      "Rhomboid",
    "Sagittate",    "Hastate",      "Ellipse",
]

# =============================================================================
# Crop species registry
# =============================================================================

CROP_SHAPES = [
    "Maize",        "Wheat",        "Sorghum",
    "Rice",         "Barley",       "Arabidopsis",
    "Tomato",       "Soybean",      "Sunflower",
    "Tobacco",      "Cassava",      "Cotton",
    "Grape",        "CommonBean",   "Cucumber",
    "Potato",       "Pepper",       "Lettuce",
]

GALLERY_COLS = 3


# =============================================================================
# Geometric shape drawing
# =============================================================================

def _draw_shape(canvas: tk.Canvas, shape: str, W: int, H: int,
                fill: str = "#40c074", outline: str = "#2ea85e") -> None:
    canvas.delete("all")
    cx, cy = W / 2.0, H / 2.0
    mx, my = W * 0.11, H * 0.08
    hw = cx - mx
    hh = cy - my

    def poly(*pts, smooth=True):
        canvas.create_polygon(*pts, fill=fill, outline=outline,
                              width=1, smooth=smooth)

    def oval(x0, y0, x1, y1):
        canvas.create_oval(x0, y0, x1, y1, fill=fill, outline=outline, width=1)

    s = shape.lower().replace("-", "").replace(" ", "")

    if s in ("orbicular",):
        r = min(hw, hh)
        oval(cx - r, cy - r, cx + r, cy + r)
    elif s in ("elliptical", "ellipse"):
        oval(mx, my, W - mx, H - my)
    elif s == "ovate":
        poly(cx, my,
             cx+hw*0.55, my+hh*0.35, cx+hw*0.90, cy,
             cx+hw*0.72, cy+hh*0.55, cx+hw*0.30, H-my,
             cx, H-my*0.6,
             cx-hw*0.30, H-my, cx-hw*0.72, cy+hh*0.55,
             cx-hw*0.90, cy, cx-hw*0.55, my+hh*0.35)
    elif s == "obovate":
        poly(cx, my*0.7,
             cx+hw*0.40, my+hh*0.30, cx+hw*0.88, cy-hh*0.10,
             cx+hw*0.75, cy+hh*0.50, cx+hw*0.38, H-my,
             cx, H-my*1.3,
             cx-hw*0.38, H-my, cx-hw*0.75, cy+hh*0.50,
             cx-hw*0.88, cy-hh*0.10, cx-hw*0.40, my+hh*0.30)
    elif s == "cordate":
        oval(mx, my, cx+hw*0.15, cy*0.80)
        oval(cx-hw*0.15, my, W-mx, cy*0.80)
        poly(mx+hw*0.10, cy*0.55, cx, H-my, W-mx-hw*0.10, cy*0.55,
             cx, cy*0.35, smooth=False)
    elif s == "reniform":
        poly(cx, my+hh*0.35,
             cx+hw*0.60, my, cx+hw*0.95, cy-hh*0.10,
             cx+hw*0.88, cy+hh*0.45, cx+hw*0.35, H-my,
             cx+hw*0.08, H-my*0.65, cx, H-my*0.45,
             cx-hw*0.08, H-my*0.65, cx-hw*0.35, H-my,
             cx-hw*0.88, cy+hh*0.45, cx-hw*0.95, cy-hh*0.10,
             cx-hw*0.60, my)
    elif s == "lanceolate":
        poly(cx, my,
             cx+hw*0.52, my+hh*0.55, cx+hw*0.30, H-my,
             cx, H-my*0.5,
             cx-hw*0.30, H-my, cx-hw*0.52, my+hh*0.55)
    elif s == "oblanceolate":
        poly(cx, my*0.6,
             cx+hw*0.50, my+hh*0.42, cx+hw*0.20, H-my,
             cx, H-my*1.5,
             cx-hw*0.20, H-my, cx-hw*0.50, my+hh*0.42)
    elif s == "oblong":
        w2 = hw * 0.50
        poly(cx-w2, my+hh*0.25, cx, my, cx+w2, my+hh*0.25,
             cx+w2, H-my-hh*0.25, cx, H-my, cx-w2, H-my-hh*0.25)
    elif s == "linear":
        w2 = hw * 0.18
        poly(cx-w2, my+hh*0.15, cx, my, cx+w2, my+hh*0.15,
             cx+w2, H-my-hh*0.15, cx, H-my, cx-w2, H-my-hh*0.15)
    elif s in ("ensiform", "sword"):
        w2 = hw * 0.14
        poly(cx-w2, my+hh*0.10, cx, my, cx+w2, my+hh*0.10,
             cx+w2*0.55, H-my, cx, H-my*0.3, cx-w2*0.55, H-my)
    elif s in ("cuneate", "wedge"):
        poly(cx-hw*0.92, my, cx+hw*0.92, my,
             cx+hw*0.92, my+hh*0.30, cx, H-my,
             cx-hw*0.92, my+hh*0.30, smooth=False)
    elif s == "spathulate":
        sw = hw * 0.18
        poly(cx-sw, cy+hh*0.10, cx-sw, H-my,
             cx+sw, H-my, cx+sw, cy+hh*0.10, smooth=False)
        oval(cx-hw*0.82, my, cx+hw*0.82, my+hh*1.05)
    elif s in ("deltoid", "triangular"):
        poly(cx, my, cx+hw*0.95, H-my,
             cx, H-my*0.55, cx-hw*0.95, H-my, smooth=False)
    elif s == "rhomboid":
        poly(cx, my, cx+hw*0.88, cy,
             cx, H-my, cx-hw*0.88, cy, smooth=False)
    elif s in ("sagittate", "sagittata"):
        poly(cx, my,
             cx+hw*0.58, cy*0.80, cx+hw*0.30, cy+hh*0.30,
             cx+hw*0.72, H-my, cx+hw*0.20, cy+hh*0.55,
             cx, cy+hh*0.65,
             cx-hw*0.20, cy+hh*0.55, cx-hw*0.72, H-my,
             cx-hw*0.30, cy+hh*0.30, cx-hw*0.58, cy*0.80)
    elif s in ("hastate", "alabardata"):
        poly(cx, my,
             cx+hw*0.50, cy*0.75, cx+hw*0.28, cy+hh*0.25,
             cx+hw*0.95, cy+hh*0.65, cx+hw*0.28, H-my*0.80,
             cx, H-my*0.50,
             cx-hw*0.28, H-my*0.80, cx-hw*0.95, cy+hh*0.65,
             cx-hw*0.28, cy+hh*0.25, cx-hw*0.50, cy*0.75)
    else:
        oval(mx, my, W-mx, H-my)


# =============================================================================
# Crop species shape drawing
# =============================================================================

def _draw_crop(canvas: tk.Canvas, species: str, W: int, H: int,
               fill: str = "#40c074", outline: str = "#2ea85e") -> None:
    """Draw a species-specific leaf silhouette traced from real plant morphology."""
    canvas.delete("all")
    cx, cy = W / 2.0, H / 2.0
    mx, my = W * 0.10, H * 0.07
    hw = cx - mx
    hh = cy - my

    def poly(*pts, smooth=True):
        canvas.create_polygon(*pts, fill=fill, outline=outline,
                              width=1, smooth=smooth)

    def oval(x0, y0, x1, y1):
        canvas.create_oval(x0, y0, x1, y1, fill=fill, outline=outline, width=1)

    def rect(x0, y0, x1, y1):
        canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=outline, width=1)

    s = species.lower().replace(" ", "").replace("(", "").replace(")", "")

    # ── Monocots — narrow, strap-like ────────────────────────────────────────

    if s in ("maize", "corn", "maizecorn"):
        # Long curved strap, wider in middle, slight curve (asymmetric)
        poly(cx-hw*0.08, my,
             cx+hw*0.35, my+hh*0.20,
             cx+hw*0.55, cy,
             cx+hw*0.40, cy+hh*0.50,
             cx+hw*0.15, H-my,
             cx-hw*0.05, H-my*0.60,
             cx-hw*0.30, H-my,
             cx-hw*0.50, cy+hh*0.40,
             cx-hw*0.42, cy-hh*0.10,
             cx-hw*0.18, my+hh*0.10)

    elif s == "wheat":
        # Very narrow, tapered both ends, slight twist
        w2 = hw * 0.20
        poly(cx-w2*0.6, my,
             cx+w2*0.6, my,
             cx+w2, cy-hh*0.20,
             cx+w2*0.5, H-my,
             cx, H-my*0.5,
             cx-w2*0.5, H-my,
             cx-w2, cy-hh*0.20)

    elif s == "sorghum":
        # Similar to maize but wider, more oval cross-section
        poly(cx-hw*0.15, my,
             cx+hw*0.45, my+hh*0.25,
             cx+hw*0.65, cy,
             cx+hw*0.45, cy+hh*0.55,
             cx+hw*0.12, H-my,
             cx, H-my*0.70,
             cx-hw*0.25, H-my,
             cx-hw*0.55, cy+hh*0.45,
             cx-hw*0.50, cy-hh*0.05,
             cx-hw*0.22, my+hh*0.12)

    elif s == "rice":
        # Very narrow, almost linear, long
        w2 = hw * 0.14
        poly(cx-w2*0.5, my,
             cx+w2*0.5, my,
             cx+w2, cy,
             cx+w2*0.4, H-my,
             cx, H-my*0.4,
             cx-w2*0.4, H-my,
             cx-w2, cy)

    elif s == "barley":
        # Multiple narrow blades (tiller cluster) — simplified as 3 thin straps
        for dx, tilt in [(-hw*0.32, -0.18), (0, 0), (hw*0.32, 0.18)]:
            w2 = hw * 0.10
            pts = [
                cx + dx - w2,     my + hh*0.15,
                cx + dx + w2,     my + hh*0.15,
                cx + dx + w2 + tilt*hh, H-my,
                cx + dx + tilt*hh, H-my*0.50,
                cx + dx - w2 + tilt*hh, H-my,
            ]
            canvas.create_polygon(*pts, fill=fill, outline=outline,
                                   width=1, smooth=True)

    # ── Dicots — broad, lobed ─────────────────────────────────────────────────

    elif s == "arabidopsis":
        # Rosette: central petiole hub with 5 ovate leaves radiating out
        # Petiole hub
        canvas.create_oval(cx-hw*0.08, cy-hh*0.08,
                           cx+hw*0.08, cy+hh*0.08,
                           fill=fill, outline=outline)
        # 5 leaves at different angles
        import math
        for i, (ang_deg, lr, lh) in enumerate([
            (270, 0.55, 0.40),   # top
            (340, 0.50, 0.38),   # top-right
            (200, 0.50, 0.38),   # top-left
            (40,  0.45, 0.35),   # right
            (140, 0.45, 0.35),   # left
        ]):
            a = math.radians(ang_deg)
            # leaf centre offset from hub
            lx = cx + math.cos(a) * hw * 0.42
            ly = cy + math.sin(a) * hh * 0.42
            # draw small oval
            canvas.create_oval(
                lx - hw*lr*0.48, ly - hh*lh*0.55,
                lx + hw*lr*0.48, ly + hh*lh*0.55,
                fill=fill, outline=outline, width=1)
            # connecting petiole
            canvas.create_line(cx, cy, lx, ly,
                                fill=fill, width=max(1, int(W*0.03)))

    elif s == "tomato":
        # Pinnate compound: central rachis + 4 pairs of leaflets + terminal
        import math
        # rachis
        rect(cx-hw*0.04, my+hh*0.30, cx+hw*0.04, H-my)
        # terminal leaflet
        canvas.create_oval(cx-hw*0.28, my, cx+hw*0.28, my+hh*0.55,
                           fill=fill, outline=outline)
        # lateral leaflets — 3 pairs decreasing in size
        for i, (yoff, xoff, lw, lh, flip) in enumerate([
            (0.45, 0.38, 0.25, 0.32, 1),
            (0.70, 0.40, 0.22, 0.28, 1),
            (0.90, 0.32, 0.16, 0.22, 1),
        ]):
            for side in [-1, 1]:
                lx = cx + side * hw * xoff
                ly = my + hh * yoff
                canvas.create_oval(lx - hw*lw, ly - hh*lh*0.55,
                                   lx + hw*lw, ly + hh*lh*0.45,
                                   fill=fill, outline=outline, width=1)

    elif s == "soybean":
        # Trifoliate: 3 oval leaflets on short petiolules
        import math
        # central petiole stub
        rect(cx-hw*0.03, cy+hh*0.15, cx+hw*0.03, H-my)
        # terminal leaflet (top)
        canvas.create_oval(cx-hw*0.42, my, cx+hw*0.42, my+hh*0.80,
                           fill=fill, outline=outline, width=1)
        # two lateral leaflets
        for side in [-1, 1]:
            lx = cx + side * hw * 0.52
            ly = cy + hh * 0.05
            canvas.create_oval(lx-hw*0.38, ly-hh*0.50,
                               lx+hw*0.38, ly+hh*0.50,
                               fill=fill, outline=outline, width=1)

    elif s == "sunflower":
        # Broad cordate with serrated edge — approximate with large cordate + notches
        # Main body
        poly(cx, my,
             cx+hw*0.55, my+hh*0.20,
             cx+hw*0.92, cy-hh*0.10,
             cx+hw*0.88, cy+hh*0.40,
             cx+hw*0.55, H-my,
             cx, H-my*0.70,
             cx-hw*0.55, H-my,
             cx-hw*0.88, cy+hh*0.40,
             cx-hw*0.92, cy-hh*0.10,
             cx-hw*0.55, my+hh*0.20)
        # Serration notches on right side (simplified)
        for yi in [0.25, 0.45, 0.65]:
            nx = cx + hw * 0.92
            ny = cy - hh * 0.10 + (hh * 0.50) * yi
            canvas.create_oval(nx-hw*0.06, ny-hh*0.05,
                               nx+hw*0.06, ny+hh*0.05,
                               fill=canvas["bg"], outline=canvas["bg"])

    elif s in ("tobacco", "nicotiana", "tobacconicotiana"):
        # Large oval/elliptic, slightly pointed tip
        poly(cx, my*0.6,
             cx+hw*0.50, my+hh*0.25,
             cx+hw*0.92, cy,
             cx+hw*0.80, cy+hh*0.55,
             cx+hw*0.35, H-my,
             cx, H-my*0.75,
             cx-hw*0.35, H-my,
             cx-hw*0.80, cy+hh*0.55,
             cx-hw*0.92, cy,
             cx-hw*0.50, my+hh*0.25)

    elif s == "cassava":
        # Palmate: 5-7 deep lobes radiating from centre
        import math
        # Draw 7 lanceolate lobes
        for i in range(7):
            ang = math.radians(-90 + i * (180 / 6))
            lx = cx + math.cos(ang) * hw * 0.85
            ly = cy + math.sin(ang) * hh * 0.85
            # lobe as narrow oval
            canvas.create_oval(lx-hw*0.14, ly-hh*0.30,
                               lx+hw*0.14, ly+hh*0.30,
                               fill=fill, outline=outline, width=1)
            canvas.create_line(cx, cy, lx, ly,
                                fill=fill, width=max(1, int(W*0.04)))
        # centre hub
        canvas.create_oval(cx-hw*0.12, cy-hh*0.12,
                           cx+hw*0.12, cy+hh*0.12,
                           fill=fill, outline=outline)

    elif s == "cotton":
        # 3-5 pointed lobes, maple-like
        poly(cx, my,
             cx+hw*0.28, cy-hh*0.30,
             cx+hw*0.90, cy-hh*0.15,
             cx+hw*0.55, cy+hh*0.10,
             cx+hw*0.80, H-my,
             cx+hw*0.20, cy+hh*0.45,
             cx, H-my*0.60,
             cx-hw*0.20, cy+hh*0.45,
             cx-hw*0.80, H-my,
             cx-hw*0.55, cy+hh*0.10,
             cx-hw*0.90, cy-hh*0.15,
             cx-hw*0.28, cy-hh*0.30)

    elif s == "grape":
        # Deeply 5-lobed, sinuses between lobes
        poly(cx, my,
             cx+hw*0.42, cy-hh*0.55,
             cx+hw*0.90, cy-hh*0.25,
             cx+hw*0.60, cy+hh*0.05,
             cx+hw*0.88, cy+hh*0.55,
             cx+hw*0.30, cy+hh*0.40,
             cx+hw*0.18, H-my,
             cx, H-my*0.70,
             cx-hw*0.18, H-my,
             cx-hw*0.30, cy+hh*0.40,
             cx-hw*0.88, cy+hh*0.55,
             cx-hw*0.60, cy+hh*0.05,
             cx-hw*0.90, cy-hh*0.25,
             cx-hw*0.42, cy-hh*0.55)

    elif s in ("commonbean", "bean"):
        # Broad cordate-ovate
        poly(cx, my,
             cx+hw*0.60, my+hh*0.15,
             cx+hw*0.92, cy,
             cx+hw*0.78, cy+hh*0.55,
             cx+hw*0.30, H-my,
             cx, H-my*0.80,
             cx-hw*0.30, H-my,
             cx-hw*0.78, cy+hh*0.55,
             cx-hw*0.92, cy,
             cx-hw*0.60, my+hh*0.15)

    elif s == "cucumber":
        # Broad pentagonal with shallow lobes and serrated edge
        poly(cx, my,
             cx+hw*0.48, my+hh*0.10,
             cx+hw*0.90, cy-hh*0.20,
             cx+hw*0.82, cy+hh*0.35,
             cx+hw*0.50, H-my,
             cx, H-my*0.82,
             cx-hw*0.50, H-my,
             cx-hw*0.82, cy+hh*0.35,
             cx-hw*0.90, cy-hh*0.20,
             cx-hw*0.48, my+hh*0.10)

    elif s == "potato":
        # Pinnate compound (similar to tomato, fewer leaflets)
        import math
        rect(cx-hw*0.04, my+hh*0.20, cx+hw*0.04, H-my)
        # terminal leaflet
        canvas.create_oval(cx-hw*0.32, my, cx+hw*0.32, my+hh*0.50,
                           fill=fill, outline=outline)
        # 2 pairs lateral
        for i, (yoff, xoff, lw, lh) in enumerate([
            (0.40, 0.38, 0.22, 0.28),
            (0.68, 0.35, 0.18, 0.22),
        ]):
            for side in [-1, 1]:
                lx = cx + side * hw * xoff
                ly = my + hh * yoff
                canvas.create_oval(lx-hw*lw, ly-hh*lh*0.55,
                                   lx+hw*lw, ly+hh*lh*0.55,
                                   fill=fill, outline=outline, width=1)

    elif s == "pepper":
        # Smooth ovate, pointed tip — clean simple shape
        poly(cx, my*0.5,
             cx+hw*0.48, my+hh*0.28,
             cx+hw*0.88, cy+hh*0.05,
             cx+hw*0.70, cy+hh*0.60,
             cx+hw*0.28, H-my,
             cx, H-my*0.80,
             cx-hw*0.28, H-my,
             cx-hw*0.70, cy+hh*0.60,
             cx-hw*0.88, cy+hh*0.05,
             cx-hw*0.48, my+hh*0.28)

    elif s == "lettuce":
        # Broad, ruffled/wavy edge — approximate with wide obovate + bumpy outline
        poly(cx, my*0.8,
             cx+hw*0.38, my,
             cx+hw*0.72, my+hh*0.18,
             cx+hw*0.55, my+hh*0.38,
             cx+hw*0.90, cy-hh*0.05,
             cx+hw*0.75, cy+hh*0.28,
             cx+hw*0.92, cy+hh*0.55,
             cx+hw*0.58, H-my,
             cx+hw*0.25, H-my*0.75,
             cx, H-my*0.60,
             cx-hw*0.25, H-my*0.75,
             cx-hw*0.58, H-my,
             cx-hw*0.92, cy+hh*0.55,
             cx-hw*0.75, cy+hh*0.28,
             cx-hw*0.90, cy-hh*0.05,
             cx-hw*0.55, my+hh*0.38,
             cx-hw*0.72, my+hh*0.18,
             cx-hw*0.38, my)

    else:
        # fallback ellipse
        oval(mx, my, W-mx, H-my)


# =============================================================================
# Shared gallery builder
# =============================================================================

def _build_gallery(parent, app, shapes, draw_fn, cols=GALLERY_COLS):
    """Build a grid of shape thumbnails with radio buttons."""
    c = app.colors
    for i, shape in enumerate(shapes):
        row_i, col_i = divmod(i, cols)
        cell = ttk.Frame(parent)
        cell.grid(row=row_i, column=col_i, padx=4, pady=3)

        cv = tk.Canvas(cell, width=54, height=54,
                       bg=c['bg_dark'],
                       highlightthickness=2,
                       highlightbackground=c['bg_medium'])
        cv.pack()
        app._leaf_shape_canvases[shape] = cv
        draw_fn(cv, shape, 54, 54,
                fill=c['accent'], outline=c['accent_active'])

        # Highlight border when selected
        def _on_select(s=shape, cv=cv):
            # Reset all borders
            for name, other_cv in app._leaf_shape_canvases.items():
                other_cv.configure(
                    highlightbackground=app.colors['bg_medium'],
                    highlightthickness=2)
            # Highlight selected
            cv.configure(
                highlightbackground=app.colors['accent'],
                highlightthickness=2)
            app.shape_extend_var.set(s)
            app._on_leaf_shape_selected()

        cv.bind("<Button-1>", lambda e, fn=_on_select: fn())

        nf = ttk.Frame(cell)
        nf.pack(pady=(2, 0))
        rb = ttk.Radiobutton(nf, text="", variable=app.shape_extend_var,
                             value=shape, command=_on_select)
        rb.pack(side="left")
        # Truncate long names
        display = shape if len(shape) <= 10 else shape[:9] + "…"
        ttk.Label(nf, text=display,
                  font=("Helvetica", 7, "bold")).pack(side="left")


# =============================================================================
# Tab builder
# =============================================================================

def build(app, tab) -> None:
    """Populate the Leaf Completion tab."""

    # 3-column layout: options | original | completed
    main = ttk.Frame(tab)
    main.grid(row=0, column=0, sticky="nsew")
    tab.grid_columnconfigure(0, weight=1)
    tab.grid_rowconfigure(0, weight=1)
    main.grid_columnconfigure(0, weight=0)
    main.grid_columnconfigure(1, weight=1)
    main.grid_columnconfigure(2, weight=1)
    main.grid_rowconfigure(0, weight=1)

    # ── COLUMN 0: Options ────────────────────────────────────────────────────
    opts = ttk.Frame(main)
    opts.grid(row=0, column=0, sticky="ns", padx=(0, 8), pady=4)

    # Shared state
    app._leaf_shape_canvases = {}
    app.shape_extend_var = tk.StringVar(value="Ellipse")

    # ── Gallery notebook: Geometric | Crop Species ───────────────────────────
    gallery_outer = ttk.LabelFrame(opts, text=" Shape Gallery ", padding=6)
    gallery_outer.pack(fill="x", pady=(0, 8))

    gallery_nb = ttk.Notebook(gallery_outer)
    gallery_nb.pack(fill="both", expand=True)

    # Tab A — Geometric shapes
    geo_outer = ttk.Frame(gallery_nb)
    gallery_nb.add(geo_outer, text=" Geometric ")
    geo_scroll_canvas = tk.Canvas(geo_outer, highlightthickness=0,
                                   bg=app.colors['bg_dark'], height=280)
    geo_sb = ttk.Scrollbar(geo_outer, orient="vertical",
                            command=geo_scroll_canvas.yview)
    geo_scroll_canvas.configure(yscrollcommand=geo_sb.set)
    geo_sb.pack(side="right", fill="y")
    geo_scroll_canvas.pack(side="left", fill="both", expand=True)
    geo_inner = ttk.Frame(geo_scroll_canvas)
    geo_win = geo_scroll_canvas.create_window((0, 0), window=geo_inner, anchor="nw")
    geo_inner.bind("<Configure>",
                   lambda e: geo_scroll_canvas.configure(
                       scrollregion=geo_scroll_canvas.bbox("all")))
    geo_scroll_canvas.bind("<Configure>",
                            lambda e: geo_scroll_canvas.itemconfig(
                                geo_win, width=e.width))
    _build_gallery(geo_inner, app, LEAF_SHAPES, _draw_shape)

    # Tab B — Crop species
    crop_outer = ttk.Frame(gallery_nb)
    gallery_nb.add(crop_outer, text=" Crop Species ")
    crop_scroll_canvas = tk.Canvas(crop_outer, highlightthickness=0,
                                    bg=app.colors['bg_dark'], height=280)
    crop_sb = ttk.Scrollbar(crop_outer, orient="vertical",
                             command=crop_scroll_canvas.yview)
    crop_scroll_canvas.configure(yscrollcommand=crop_sb.set)
    crop_sb.pack(side="right", fill="y")
    crop_scroll_canvas.pack(side="left", fill="both", expand=True)
    crop_inner = ttk.Frame(crop_scroll_canvas)
    crop_win = crop_scroll_canvas.create_window((0, 0), window=crop_inner, anchor="nw")
    crop_inner.bind("<Configure>",
                    lambda e: crop_scroll_canvas.configure(
                        scrollregion=crop_scroll_canvas.bbox("all")))
    crop_scroll_canvas.bind("<Configure>",
                             lambda e: crop_scroll_canvas.itemconfig(
                                 crop_win, width=e.width))
    _build_gallery(crop_inner, app, CROP_SHAPES, _draw_crop)

    # ── Apply Completion controls ────────────────────────────────────────────
    ctrl = ttk.LabelFrame(opts, text=" Apply Completion ", padding=8)
    ctrl.pack(fill="x", pady=(0, 8))

    sel_row = ttk.Frame(ctrl)
    sel_row.pack(fill="x", pady=(0, 6))
    ttk.Label(sel_row, text="Selected:").pack(side="left", padx=(0, 6))
    app._leaf_shape_canvas = tk.Canvas(
        sel_row, width=28, height=28,
        bg=app.colors['bg_dark'],
        highlightthickness=1,
        highlightbackground=app.colors['accent'])
    app._leaf_shape_canvas.pack(side="left", padx=(0, 6))
    app._selected_shape_label = ttk.Label(sel_row, text="Ellipse",
                                           font=("Helvetica", 10, "bold"))
    app._selected_shape_label.pack(side="left")

    ttk.Button(ctrl, text="▶ Preview",
               command=app._preview_leaf_completion,
               style="Accent.TButton").pack(fill="x", pady=(0, 4))

    size_row = ttk.Frame(ctrl)
    size_row.pack(fill="x", pady=(0, 4))
    ttk.Label(size_row, text="Size:").pack(side="left", padx=(0, 4))
    app._leaf_shrink_btn = ttk.Button(size_row, text=" − ", width=3,
                                       command=app._shrink_leaf_shape,
                                       state="disabled")
    app._leaf_shrink_btn.pack(side="left", padx=(0, 2))
    app._leaf_scale_label = ttk.Label(size_row, text="100%", width=5, anchor="center")
    app._leaf_scale_label.pack(side="left", padx=2)
    app._leaf_grow_btn = ttk.Button(size_row, text=" + ", width=3,
                                     command=app._grow_leaf_shape,
                                     state="disabled")
    app._leaf_grow_btn.pack(side="left", padx=(2, 0))
    app._leaf_scale_factor = 1.0

    app._leaf_edit_btn = ttk.Button(ctrl, text="✏ Edit Shape",
                                     command=app._edit_leaf_shape,
                                     state="disabled")
    app._leaf_edit_btn.pack(fill="x", pady=(0, 4))

    app._leaf_update_btn = ttk.Button(ctrl, text="✓ Update Mask",
                                       command=app._apply_leaf_completion,
                                       state="disabled")
    app._leaf_update_btn.pack(fill="x", pady=(0, 4))

    ttk.Button(ctrl, text="✗ Cancel",
               command=app._cancel_leaf_completion).pack(fill="x")

    ttk.Label(opts, text="Select mask → Pick shape → Preview → Update",
              font=("Helvetica", 8), wraplength=200).pack(anchor="w", pady=(4, 0))

    # ── COLUMN 1: Original mask ──────────────────────────────────────────────
    orig_lf = ttk.LabelFrame(main, text=" Original Mask ", padding=8)
    orig_lf.grid(row=0, column=1, sticky="nsew", padx=5, pady=4)
    orig_lf.grid_columnconfigure(0, weight=1)
    orig_lf.grid_rowconfigure(0, weight=1)
    app._leaf_orig_canvas = tk.Canvas(
        orig_lf, width=250, height=250,
        bg=app.colors['canvas_bg'],
        highlightthickness=2,
        highlightbackground=app.colors['bg_medium'])
    app._leaf_orig_canvas.pack(expand=True, fill="both", padx=10, pady=(10, 5))

    # ── COLUMN 2: Completed mask ─────────────────────────────────────────────
    comp_lf = ttk.LabelFrame(main, text=" Completed Mask ", padding=8)
    comp_lf.grid(row=0, column=2, sticky="nsew", padx=(5, 0), pady=4)
    comp_lf.grid_columnconfigure(0, weight=1)
    comp_lf.grid_rowconfigure(0, weight=1)
    app._leaf_comp_canvas = tk.Canvas(
        comp_lf, width=250, height=250,
        bg=app.colors['canvas_bg'],
        highlightthickness=2,
        highlightbackground=app.colors['bg_medium'])
    app._leaf_comp_canvas.pack(expand=True, fill="both", padx=10, pady=(10, 5))

    # ── Zoom controls ────────────────────────────────────────────────────────
    zoom_f = ttk.Frame(main)
    zoom_f.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(0, 4))
    ttk.Label(zoom_f, text="Zoom:", font=("Helvetica", 9)).pack(
        side="left", padx=(10, 5))
    ttk.Button(zoom_f, text=" − ", width=3,
               command=lambda: app._leaf_zoom_by(0.8)).pack(side="left", padx=2)
    app._leaf_zoom_label = ttk.Label(zoom_f, text="100%", width=5, anchor="center")
    app._leaf_zoom_label.pack(side="left", padx=2)
    ttk.Button(zoom_f, text=" + ", width=3,
               command=lambda: app._leaf_zoom_by(1.25)).pack(side="left", padx=2)
    ttk.Button(zoom_f, text="Fit", width=4,
               command=app._leaf_zoom_fit).pack(side="left", padx=(10, 2))
    app._leaf_preview_zoom = 1.0

    # ── Canvas bindings ──────────────────────────────────────────────────────
    app._leaf_orig_canvas.bind("<MouseWheel>", app._leaf_canvas_mousewheel)
    app._leaf_comp_canvas.bind("<MouseWheel>", app._leaf_canvas_mousewheel)
    app._leaf_orig_canvas.bind("<Button-4>", lambda e: app._leaf_zoom_by(1.1))
    app._leaf_orig_canvas.bind("<Button-5>", lambda e: app._leaf_zoom_by(0.9))
    app._leaf_comp_canvas.bind("<Button-4>", lambda e: app._leaf_zoom_by(1.1))
    app._leaf_comp_canvas.bind("<Button-5>", lambda e: app._leaf_zoom_by(0.9))
    app._leaf_comp_canvas.bind("<ButtonPress-1>",  app._leaf_drag_start)
    app._leaf_comp_canvas.bind("<B1-Motion>",       app._leaf_drag_move)
    app._leaf_comp_canvas.bind("<ButtonRelease-1>", app._leaf_drag_end)

    # ── Position + Rotation controls ─────────────────────────────────────────
    adj_f = ttk.Frame(main)
    adj_f.grid(row=2, column=1, columnspan=2, sticky="ew", pady=(0, 4))
    ttk.Label(adj_f, text="Position:", font=("Helvetica", 9)).pack(
        side="left", padx=(10, 5))
    app._leaf_offset_label = ttk.Label(adj_f, text="(0, 0)", width=8, anchor="center")
    app._leaf_offset_label.pack(side="left", padx=2)
    ttk.Button(adj_f, text="Reset", width=4,
               command=app._leaf_reset_position).pack(side="left", padx=(5, 2))
    ttk.Separator(adj_f, orient="vertical").pack(side="left", padx=8, fill="y", pady=2)
    ttk.Label(adj_f, text="Rotate:", font=("Helvetica", 9)).pack(side="left", padx=(0, 5))
    ttk.Button(adj_f, text=" ↶ ", width=3,
               command=lambda: app._leaf_rotate(-5)).pack(side="left", padx=2)
    app._leaf_rotation_label = ttk.Label(adj_f, text="0°", width=5, anchor="center")
    app._leaf_rotation_label.pack(side="left", padx=2)
    ttk.Button(adj_f, text=" ↷ ", width=3,
               command=lambda: app._leaf_rotate(5)).pack(side="left", padx=2)
    ttk.Button(adj_f, text="Reset", width=4,
               command=app._leaf_reset_rotation).pack(side="left", padx=(5, 2))
    ttk.Label(adj_f, text="(Drag to move)",
              font=("Helvetica", 8)).pack(side="left", padx=(12, 0))

    # ── Stats label ──────────────────────────────────────────────────────────
    app._leaf_stats_label = ttk.Label(main, text="Select a mask to preview",
                                       font=("Helvetica", 9))
    app._leaf_stats_label.grid(row=1, column=0, columnspan=3,
                                sticky="w", pady=(4, 0))

    # ── State init ───────────────────────────────────────────────────────────
    app._leaf_shape_offset    = [0, 0]
    app._leaf_drag_start_pos  = None
    app._leaf_rotation_offset = 0
    app._pending_leaf_completion = None

    app.root.after(100, app._update_leaf_shape_preview)
    app.root.after(100, app._on_leaf_shape_selected)
