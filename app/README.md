# app/  —  Modular source package

This folder is the refactored version of the original `plant_segmenter.py`
(which remains untouched in the project root).

## How to run

```bash
# from the app/ directory
cd app
python plant_segmenter.py

# or from project root
python app/plant_segmenter.py
```

## File map

```
app/
│
├── plant_segmenter.py   ← thin entry point  (just calls main())
├── gui_app.py           ← LeafSegmenterGUI class  (layout + event handlers)
│
├── app_visuals.py       ← everything visual / style related
│   # COLORS dict         — full plant-inspired palette
│   # ICONS dict          — unicode emoji map
│   # apply_theme(root)   — configure all ttk styles in one call
│   # ToolTip             — fade-in hover tooltip widget
│   # AnimatedSpinner     — spinning overlay for long operations
│   # status_color()      — map 'success'|'warning'|'error' → hex colour
│
├── image_processing.py  ← pure image math  (numpy/cv2, no GUI)
│   # ensure_uint8_rgb, rotate_left_90
│   # preprocess_for_edges, enhance_leaf_edges_rgb
│   # flatten_background_whiten
│   # compute_vegetation_indices, enhance_with_vegetation_index
│   # denoise_nlm, single_scale_retinex, multi_scale_retinex
│   # morphological_tophat, guided_filter_enhance, enhance_lab_green
│   # white_balance_grayworld, white_balance_max_white
│   # difference_of_gaussians, local_contrast_normalization
│   # adaptive_gamma, shadow_highlight_correction
│
├── mask_utils.py        ← pure mask math  (numpy/cv2, no GUI)
│   # _ensure_mask_2d, _resize_mask_to_image
│   # save_binary_mask, save_masked_crop_rgba
│   # mask_iou, dedupe_by_mask_iou
│   # split_masks_by_cc
│   # predict_extend_mask  ← main public entry for shape completion
│   #   internally: rosette (circle/hull-wedge/ellipse) + blade (tapered)
│
├── phenotyping.py       ← measurements + CSV export  (no GUI)
│   # _color_stats (RGB), _color_stats_hsv
│   # _pca_angle_deg, _pca_major_minor, _length_width_after_deskew
│   # _vegetation_indices_stats
│   # compute_phenotypes  ← main per-mask measurement function
│   # build_individual_rows, build_joint_row
│   # write_individual_csv, write_joint_csv
│
├── sam2_utils.py        ← SAM2 loading + Hydra config helpers  (no GUI)
│   # _hydra_reinit_to_dir, _compose_from_yaml
│   # _resolve_sam2_cfg   ← handles short names / YAML paths / directories
│   # make_mask_generator ← SAM2AutomaticMaskGenerator with plant defaults
│   # load_sam2_model     ← from checkpoint + config
│   # load_sam2_bundle    ← from a bundled .pt file
│
├── mask_completion.py   ← ML-based leaf completion  (already separate ✓)
├── tip_segmenter_trainer.py  ← custom U-Net training  (already separate ✓)
├── tip_segmenter_model.py    ← custom U-Net model def (already separate ✓)
│
└── __init__.py          ← package init, re-exports public API
```

## Migration note for `gui_app.py`

`gui_app.py` contains stub `raise NotImplementedError` for the four panel
builder methods:

- `make_model_frame()`
- `make_options_frame()`
- `make_preview_frame()`
- `make_masks_frame()`
- `_on_model_type_change()`

These are the large tkinter layout blocks from the original file.
To complete the migration, copy those method bodies verbatim from
`plant_segmenter.py` (project root) into the matching methods in `gui_app.py`.
The imports at the top of `gui_app.py` are already wired up so all the
helper functions they call will resolve correctly.

## Design principles

1. **Pure functions are never in the GUI file.**
   Every numpy/cv2 function that doesn't touch `self` or tkinter lives in
   `image_processing.py`, `mask_utils.py`, or `phenotyping.py`.
   They are independently testable without launching the GUI.

2. **All visual decisions live in `app_visuals.py`.**
   To retheme the app (colours, fonts, button sizes), edit one file.

3. **SAM2 optional at import time.**
   `sam2_utils.py` catches `ImportError` gracefully — the GUI still launches
   and shows a friendly error if SAM2 is not installed.

4. **The original `plant_segmenter.py` is never modified.**
   The `app/` folder is an additive parallel structure.
