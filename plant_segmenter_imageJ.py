#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ImageJ-style launcher for Plant Segmenter.

This keeps the core segmentation behavior from plant_segmenter.py, but
reorganizes the UI with a classic menu bar and hides the heavy enhancement
("image edit") panel by default.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

from plant_segmenter import LeafSegmenterGUI


class PlantSegmenterImageJGUI(LeafSegmenterGUI):
    def __init__(self, root: tk.Tk):
        super().__init__(root)
        self.root.title("Plant Segmenter - ImageJ Style")

        self._enhancement_frame = None
        self._actions_frame = None
        self._enhancement_panel_visible = tk.BooleanVar(value=False)

        self._discover_left_sections()
        self._set_enhancement_panel_visible(False)
        self._build_imagej_menu()

        self.set_status(
            "ImageJ-style window ready. Enhancement controls are hidden (Window -> Enhancement Panel).",
            "info",
        )

    def _iter_widgets(self, parent):
        yield parent
        for child in parent.winfo_children():
            yield from self._iter_widgets(child)

    def _discover_left_sections(self):
        for widget in self._iter_widgets(self.left_panel):
            try:
                text = str(widget.cget("text")).lower()
            except Exception:
                continue

            if "enhancement" in text and self._enhancement_frame is None:
                self._enhancement_frame = widget
            if "action" in text and self._actions_frame is None:
                self._actions_frame = widget

    def _set_enhancement_panel_visible(self, visible: bool):
        if self._enhancement_frame is None:
            return

        if visible:
            if not self._enhancement_frame.winfo_ismapped():
                pack_kwargs = dict(fill="x", expand=False, pady=(0, 8))
                if self._actions_frame is not None and self._actions_frame.winfo_exists():
                    self._enhancement_frame.pack(before=self._actions_frame, **pack_kwargs)
                else:
                    self._enhancement_frame.pack(**pack_kwargs)
        else:
            if self._enhancement_frame.winfo_ismapped():
                self._enhancement_frame.pack_forget()

        self._enhancement_panel_visible.set(visible)

    def _toggle_enhancement_panel(self):
        self._set_enhancement_panel_visible(not self._enhancement_panel_visible.get())

    def _toggle_crop_from_menu(self):
        self._crop_mode.set(not self._crop_mode.get())
        self._set_crop_mode()

    def _toggle_pick_from_menu(self):
        self._pick_mode.set(not self._pick_mode.get())
        self._toggle_pick_mode()

    def _build_imagej_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image...", command=self.open_image)
        file_menu.add_command(label="Open Folder...", command=self.open_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Load Masks...", command=self.load_masks)
        file_menu.add_command(label="Save All Masks...", command=self.save_all_masks)
        file_menu.add_command(label="Save Selected Masks...", command=self.save_selected_masks)
        file_menu.add_command(label="Save Batch Masks...", command=self.save_all_batch_results)
        file_menu.add_command(label="Save Outputs (PNG+CSV)...", command=self.save_all_outputs)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Intentionally lightweight to match your request:
        # "image edit" controls are out of the way in this UI.
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="No edit tools in this build", state="disabled")
        menubar.add_cascade(label="Edit", menu=edit_menu)

        image_menu = tk.Menu(menubar, tearoff=0)
        image_menu.add_command(label="Image edits moved to Window -> Enhancement Panel", state="disabled")
        menubar.add_cascade(label="Image", menu=image_menu)

        segment_menu = tk.Menu(menubar, tearoff=0)
        segment_menu.add_command(label="Load SAM2 Model...", command=self.load_model)
        segment_menu.add_command(label="Load SAM2 Bundle...", command=self.load_bundle)
        segment_menu.add_separator()
        segment_menu.add_command(label="Preview Enhancement", command=self.preview_enhance)
        segment_menu.add_command(label="Segment Current Image", command=self.segment)
        segment_menu.add_command(label="Segment All Images...", command=self.segment_all_batch)
        menubar.add_cascade(label="Segment", menu=segment_menu)

        masks_menu = tk.Menu(menubar, tearoff=0)
        masks_menu.add_command(label="Delete Selected", command=self.delete_selected_masks)
        masks_menu.add_command(label="Clear All", command=self.clear_all_masks)
        masks_menu.add_separator()
        masks_menu.add_command(label="Combine Selected", command=self.combine_selected_masks)
        masks_menu.add_command(label="Refine Selected", command=self.refine_selected_masks)
        masks_menu.add_command(label="Complete Selected", command=self.on_complete_selected_mask)
        menubar.add_cascade(label="Masks", menu=masks_menu)

        analyze_menu = tk.Menu(menubar, tearoff=0)
        analyze_menu.add_command(label="Phenotype Metrics Help", command=self.explain_phenotypes)
        analyze_menu.add_command(label="Mask Parameter Guide", command=self.explain_mask_params)
        menubar.add_cascade(label="Analyze", menu=analyze_menu)

        window_menu = tk.Menu(menubar, tearoff=0)
        window_menu.add_command(label="Fit to Window", command=self._zoom_fit)
        window_menu.add_command(label="Zoom In", command=lambda: self._zoom_by(1.1))
        window_menu.add_command(label="Zoom Out", command=lambda: self._zoom_by(1 / 1.1))
        window_menu.add_separator()
        window_menu.add_checkbutton(
            label="Enhancement Panel",
            variable=self._enhancement_panel_visible,
            command=self._toggle_enhancement_panel,
        )
        window_menu.add_command(label="Toggle Crop Tool", command=self._toggle_crop_from_menu)
        window_menu.add_command(label="Toggle Pick Tool", command=self._toggle_pick_from_menu)
        menubar.add_cascade(label="Window", menu=window_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about_dialog)
        menubar.add_cascade(label="Help", menu=help_menu)

    def _show_about_dialog(self):
        messagebox.showinfo(
            "About Plant Segmenter - ImageJ Style",
            "Plant Segmenter - ImageJ Style\n\n"
            "Core segmentation engine: plant_segmenter.py\n"
            "UI mode: ImageJ-like menus, with image-edit panel hidden by default.",
        )


if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.3)
    except Exception:
        pass
    app = PlantSegmenterImageJGUI(root)
    root.mainloop()
