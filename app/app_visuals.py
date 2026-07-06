#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app_visuals.py
==============
Everything that defines how the app *looks* — in one place.

Contains:
  COLORS      – full plant-inspired color palette (dict)
  ICONS       – unicode emoji icon map (dict)
  apply_theme(root)  – configure all ttk styles on the root window
  ToolTip     – fade-in hover tooltip widget
  AnimatedSpinner – spinning overlay shown during long operations
"""

import tkinter as tk
from tkinter import ttk


# =============================================================================
# Color palette
# =============================================================================

COLORS: dict[str, str] = {
    # ── Backgrounds — dark charcoal/slate, only 2 shades ─────────────────────
    'bg_dark':        '#1e1e2e',   # Deep charcoal  — main window background
    'bg_medium':      '#2a2a3e',   # Slate panel    — panel / labelframe backgrounds
    'bg_light':       '#35354f',   # Lifted slate   — section headers, masks panel
    'bg_pale':        '#e8eaf6',   # Cool near-white — entry / combobox fields
    'canvas_bg':      '#12121c',   # Almost black    — image-viewer canvas

    # ── Accent — green used ONLY for interactive elements ─────────────────────
    'accent':         '#40c074',   # Vivid mint-green  — primary buttons, active states
    'accent_hover':   '#56d68a',   # Lighter mint      — hover
    'accent_active':  '#2ea85e',   # Pressed / active
    'highlight':      '#40c074',   # Selection / focus ring

    # ── Text ──────────────────────────────────────────────────────────────────
    'text_light':     '#e0e0f0',   # Soft white  — text on dark backgrounds
    'text_dark':      '#12121c',   # Near-black  — text on light input fields
    'text_muted':     '#8888aa',   # Dimmed grey — secondary labels, hints

    # ── Status colours ────────────────────────────────────────────────────────
    'success':        '#40c074',   # Green  (matches accent)
    'warning':        '#ffb74d',   # Amber
    'error':          '#ef5350',   # Red
    'info':           '#42a5f5',   # Blue

    # ── Structural ────────────────────────────────────────────────────────────
    'border':         '#3a3a55',   # Subtle border / separator
}


# =============================================================================
# Icon map
# =============================================================================

ICONS: dict[str, str] = {
    'model':    '🧠',
    'image':    '🖼️',
    'enhance':  '✨',
    'sam':      '🎯',
    'phenotype':'📊',
    'action':   '⚡',
    'masks':    '🎭',
    'preview':  '👁️',
    'training': '🔬',
    'folder':   '📁',
    'save':     '💾',
    'load':     '📂',
    'segment':  '✂️',
    'success':  '✓',
    'warning':  '⚠️',
    'info':     'ℹ️',
}


# =============================================================================
# Theme application
# =============================================================================

def apply_theme(root: tk.Tk) -> None:
    """
    Apply the dark scientific / VS Code-style theme to *root*.
    Charcoal/slate backgrounds, white text everywhere, green ONLY on
    interactive elements (buttons, focus rings, selections).
    Call once during app initialisation, before building any widgets.
    """
    c = COLORS
    style = ttk.Style(root)
    style.theme_use('clam')

    # ── Base widget styles ────────────────────────────────────────────────────
    style.configure('TFrame',
                    background=c['bg_dark'])
    style.configure('TLabel',
                    background=c['bg_dark'],
                    foreground=c['text_light'],
                    font=('Helvetica', 10))
    style.configure('TCheckbutton',
                    background=c['bg_dark'],
                    foreground=c['text_light'],
                    focuscolor=c['accent'],
                    font=('Helvetica', 10))
    style.configure('TRadiobutton',
                    background=c['bg_dark'],
                    foreground=c['text_light'],
                    focuscolor=c['accent'],
                    font=('Helvetica', 10))
    style.configure('TSeparator',
                    background=c['border'])
    style.configure('TPanedwindow',
                    background=c['bg_dark'])

    # Notebook tabs — slate bg, green underline on active
    style.configure('TNotebook',
                    background=c['bg_dark'],
                    bordercolor=c['border'])
    style.configure('TNotebook.Tab',
                    background=c['bg_medium'],
                    foreground=c['text_muted'],
                    padding=(10, 4),
                    font=('Helvetica', 10))
    style.map('TNotebook.Tab',
              background=[('selected', c['bg_light'])],
              foreground=[('selected', c['text_light'])])

    # ── LabelFrames — slate panels, green border on left edge ────────────────
    style.configure('TLabelframe',
                    background=c['bg_medium'],
                    bordercolor=c['border'],
                    relief='flat')
    style.configure('TLabelframe.Label',
                    background=c['bg_medium'],
                    foreground=c['accent'],        # green section title
                    font=('Helvetica', 10, 'bold'))

    # All named panel variants — same slate bg, green titles
    for name in ('Model', 'Options', 'Preview'):
        style.configure(f'{name}.TLabelframe',
                        background=c['bg_medium'],
                        bordercolor=c['border'],
                        relief='flat')
        style.configure(f'{name}.TLabelframe.Label',
                        background=c['bg_medium'],
                        foreground=c['accent'],
                        font=('Helvetica', 10, 'bold'))

    # Masks panel — slightly lighter slate so it reads as a distinct zone
    style.configure('Masks.TLabelframe',
                    background=c['bg_light'],
                    bordercolor=c['border'],
                    relief='flat')
    style.configure('Masks.TLabelframe.Label',
                    background=c['bg_light'],
                    foreground=c['accent'],
                    font=('Helvetica', 10, 'bold'))

    # Training panel — same slate as medium panels
    style.configure('Training.TLabelframe',
                    background=c['bg_medium'],
                    bordercolor=c['border'],
                    relief='flat')
    style.configure('Training.TLabelframe.Label',
                    background=c['bg_medium'],
                    foreground=c['accent'],
                    font=('Helvetica', 10, 'bold'))

    # ── Buttons ───────────────────────────────────────────────────────────────
    # Default — slate background, plain text; green only on hover
    style.configure('TButton',
                    background=c['bg_light'],
                    foreground=c['text_light'],
                    borderwidth=0,
                    focuscolor=c['accent'],
                    padding=(10, 5),
                    font=('Helvetica', 10),
                    relief='flat')
    style.map('TButton',
              background=[('active', c['accent']),
                          ('pressed', c['accent_active'])],
              foreground=[('active', c['text_dark']),
                          ('pressed', c['text_dark'])])

    # Accent — solid green, used for primary actions (Load Model, Segment)
    style.configure('Accent.TButton',
                    background=c['accent'],
                    foreground=c['text_dark'],
                    borderwidth=0,
                    focuscolor=c['accent_hover'],
                    padding=(12, 7),
                    font=('Helvetica', 10, 'bold'),
                    relief='flat')
    style.map('Accent.TButton',
              background=[('active', c['accent_hover']),
                          ('pressed', c['accent_active'])],
              foreground=[('active', c['text_dark'])])

    # Secondary — very subtle, for non-critical actions
    style.configure('Secondary.TButton',
                    background=c['bg_medium'],
                    foreground=c['text_muted'],
                    borderwidth=0,
                    padding=(9, 4),
                    font=('Helvetica', 10),
                    relief='flat')
    style.map('Secondary.TButton',
              background=[('active', c['bg_light'])],
              foreground=[('active', c['text_light'])])

    # Icon — small toolbar buttons (mask panel)
    style.configure('Icon.TButton',
                    background=c['bg_light'],
                    foreground=c['text_light'],
                    borderwidth=0,
                    font=('Helvetica', 12),
                    padding=(6, 4),
                    width=3,
                    relief='flat')
    style.map('Icon.TButton',
              background=[('active', c['accent'])],
              foreground=[('active', c['text_dark'])])

    # Danger — red for destructive actions
    style.configure('Danger.TButton',
                    background=c['error'],
                    foreground=c['text_light'],
                    borderwidth=0,
                    padding=(10, 5),
                    font=('Helvetica', 10),
                    relief='flat')
    style.map('Danger.TButton',
              background=[('active', '#c62828')],
              foreground=[('active', c['text_light'])])

    # ── Input widgets — cool near-white fields, dark text ─────────────────────
    style.configure('TEntry',
                    fieldbackground=c['bg_pale'],
                    foreground=c['text_dark'],
                    insertcolor=c['text_dark'],
                    bordercolor=c['border'],
                    font=('Helvetica', 10))
    style.configure('TCombobox',
                    fieldbackground=c['bg_pale'],
                    background=c['bg_pale'],
                    foreground=c['text_dark'],
                    bordercolor=c['border'],
                    font=('Helvetica', 10))
    style.configure('TSpinbox',
                    fieldbackground=c['bg_pale'],
                    foreground=c['text_dark'],
                    bordercolor=c['border'],
                    font=('Helvetica', 10))

    # ── Sliders & scrollbars ──────────────────────────────────────────────────
    style.configure('TScale',
                    background=c['bg_medium'],
                    troughcolor=c['bg_dark'],
                    sliderthickness=14)
    style.map('TScale',
              troughcolor=[('active', c['accent'])])

    style.configure('TScrollbar',
                    background=c['bg_light'],
                    troughcolor=c['bg_dark'],
                    bordercolor=c['bg_medium'],
                    arrowcolor=c['text_muted'],
                    relief='flat')
    style.map('TScrollbar',
              background=[('active', c['accent'])])

    # ── Progressbar ──────────────────────────────────────────────────────────
    style.configure('TProgressbar',
                    background=c['accent'],
                    troughcolor=c['bg_dark'],
                    bordercolor=c['border'])

    # ── Treeview ─────────────────────────────────────────────────────────────
    style.configure('Treeview',
                    background=c['bg_medium'],
                    foreground=c['text_light'],
                    fieldbackground=c['bg_medium'],
                    bordercolor=c['border'],
                    rowheight=26,
                    font=('Helvetica', 10))
    style.configure('Treeview.Heading',
                    background=c['bg_light'],
                    foreground=c['text_muted'],
                    font=('Helvetica', 10, 'bold'),
                    relief='flat')
    style.map('Treeview',
              background=[('selected', c['accent'])],
              foreground=[('selected', c['text_dark'])])


# =============================================================================
# ToolTip  — fade-in hover tooltip
# =============================================================================

class ToolTip:
    """
    Attaches a fade-in tooltip to any tkinter widget.

    Usage:
        ToolTip(my_button, "Click to segment the image")
    """

    def __init__(self, widget: tk.Widget, text: str, delay: int = 400) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tip_window: tk.Toplevel | None = None
        self._after_id = None
        self.alpha = 0.0

        widget.bind("<Enter>",       self._schedule)
        widget.bind("<Leave>",       self._hide)
        widget.bind("<ButtonPress>", self._hide)

    # ── Scheduling ────────────────────────────────────────────────────────────

    def _schedule(self, _event=None) -> None:
        self._hide()
        self._after_id = self.widget.after(self.delay, self._show)

    # ── Show / fade-in ────────────────────────────────────────────────────────

    def _show(self) -> None:
        if self.tip_window:
            return
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4

        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_attributes("-topmost", True)
        try:
            tw.wm_attributes("-alpha", 0.0)
        except Exception:
            pass

        frame = tk.Frame(tw, bg=COLORS['bg_dark'], bd=0, relief="flat")
        frame.pack()
        tk.Label(
            frame,
            text=self.text,
            bg=COLORS['bg_dark'],
            fg=COLORS['text_light'],
            font=("Helvetica", 10),
            padx=10,
            pady=6,
        ).pack()

        tw.update_idletasks()
        tw_w = tw.winfo_reqwidth()
        tw.wm_geometry(f"+{x - tw_w // 2}+{y}")
        self._fade_in()

    def _fade_in(self) -> None:
        if not self.tip_window:
            return
        self.alpha = min(1.0, self.alpha + 0.15)
        try:
            self.tip_window.wm_attributes("-alpha", self.alpha)
        except Exception:
            pass
        if self.alpha < 1.0:
            self.widget.after(20, self._fade_in)

    # ── Hide ─────────────────────────────────────────────────────────────────

    def _hide(self, _event=None) -> None:
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None
        self.alpha = 0.0


# =============================================================================
# AnimatedSpinner  — full-canvas loading overlay
# =============================================================================

class AnimatedSpinner:
    """
    Animated spinning-arc overlay, displayed on top of a parent widget
    while a long operation is running.

    Usage:
        spinner = AnimatedSpinner(parent_frame)
        spinner.show("Segmenting…")
        # … do work …
        spinner.hide()
    """

    def __init__(self, parent: tk.Widget) -> None:
        self.parent = parent
        self.canvas: tk.Canvas | None = None
        self.angle = 0
        self.animating = False
        self.label_text = "Processing…"

    def show(self, text: str = "Processing…") -> None:
        self.label_text = text
        if self.canvas:
            return   # already visible

        self.canvas = tk.Canvas(
            self.parent,
            bg=COLORS['canvas_bg'],
            highlightthickness=0,
        )
        self.canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Semi-transparent dark overlay
        self.canvas.create_rectangle(
            0, 0, 2000, 2000,
            fill=COLORS['bg_dark'],
            stipple="gray50",
            tags="overlay",
        )

        self.animating = True
        self._draw()

    def _draw(self) -> None:
        if not self.canvas or not self.animating:
            return

        self.canvas.delete("spinner")
        self.canvas.delete("text")

        cx = max(10, self.canvas.winfo_width() // 2)
        cy = max(10, self.canvas.winfo_height() // 2)
        if cx == 10:    # not yet laid out
            cx, cy = 200, 150

        r = 30
        for i in range(8):
            start = self.angle + i * 45
            alpha = max(0.05, 1.0 - i * 0.12)
            color = _blend_hex(COLORS['accent'], COLORS['bg_dark'], alpha)
            self.canvas.create_arc(
                cx - r, cy - r, cx + r, cy + r,
                start=start, extent=30,
                style="arc", width=4, outline=color,
                tags="spinner",
            )

        self.canvas.create_text(
            cx, cy + r + 25,
            text=self.label_text,
            fill=COLORS['text_light'],
            font=("Helvetica", 12, "bold"),
            tags="text",
        )

        self.angle = (self.angle + 15) % 360
        self.parent.after(50, self._draw)

    def hide(self) -> None:
        self.animating = False
        if self.canvas:
            self.canvas.destroy()
            self.canvas = None


# =============================================================================
# Internal colour helpers
# =============================================================================

def _blend_hex(c1: str, c2: str, alpha: float) -> str:
    """Linearly blend two hex colours; alpha=1.0 → fully c1."""
    try:
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
        r = int(r1 * alpha + r2 * (1 - alpha))
        g = int(g1 * alpha + g2 * (1 - alpha))
        b = int(b1 * alpha + b2 * (1 - alpha))
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return c1


def status_color(status: str) -> str:
    """Map a status string to its display colour."""
    return {
        'success': COLORS['success'],
        'warning': COLORS['warning'],
        'error':   COLORS['error'],
        'info':    COLORS['info'],
    }.get(status.lower(), COLORS['text_light'])
