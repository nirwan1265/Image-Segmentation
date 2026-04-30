# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an image segmentation project focused on plant/leaf segmentation using SAM (Segment Anything Model) and SAM2. The primary use case is phenotyping - extracting morphological measurements from plant images.

## Key Components

### Main Applications

- **`plant_segmenter.py` / `plant_segmenter_neat.py`**: Interactive tkinter GUI applications for leaf segmentation using SAM2. Features include:
  - Image enhancement controls
  - SAM2 automatic mask generation
  - Selective mask saving with phenotyping metrics export
  - PCA-based leaf orientation and dimension analysis
  - Color statistics (RGB/HSV) extraction

- **`occlusion_augmentation.py`**: Data augmentation module for training SAM on occluded leaf images. Supports both SAM2 and SAM(v1) with automatic fallback.

- **`PSAM_segment_fixed_panels_corrected.py`**: Alternative GUI segmenter with similar functionality.

### FastSAM Integration

The `FastSAM/` directory contains a local clone of the FastSAM project (CNN-based faster alternative to SAM). Used via:
```python
from fastsam import FastSAM, FastSAMPrompt
```

## Dependencies

Core requirements (from FastSAM/requirements.txt):
- PyTorch >= 1.7.0, TorchVision >= 0.8.1
- OpenCV >= 4.6.0
- Pillow, NumPy, SciPy
- Hydra-core, OmegaConf (for SAM2 config)

SAM2-specific:
- SAM2 repo must be in PYTHONPATH for `sam2.build_sam` and `sam2.automatic_mask_generator` imports
- Set `SAM2_CONFIG_DIR` environment variable to point to SAM2 configs directory

## Running the Applications

```bash
# Main GUI segmenter (requires SAM2)
python plant_segmenter_neat.py
# Then click "Load Bundle…" and select sam2_bundle.pt to load the model

# FastSAM inference
python FastSAM/Inference.py --model_path ./weights/FastSAM.pt --img_path <image>

# FastSAM with text prompt
python FastSAM/Inference.py --model_path ./weights/FastSAM.pt --img_path <image> --text_prompt "the leaf"
```

## Architecture Notes

### Phenotyping Pipeline

The segmenters compute these metrics per mask:
1. **Geometry**: area (pixels), bounding box, PCA-derived length/width
2. **Color**: RGB and HSV channel statistics (mean, median, std)
3. **Orientation**: PCA angle for de-skewing measurements

### Mask Processing Helpers

Key functions in the segmenter files:
- `_pca_major_minor()`: PCA-based axis-aligned length/width
- `_length_width_after_deskew()`: Rotate mask to align major axis, measure span
- `_convex_hull_fill()`, `_rosette_*_extend()`: Shape completion for partial leaves
- `dedupe_by_mask_iou()`: Remove duplicate overlapping masks

### SAM2 Config Resolution

The `_resolve_sam2_cfg()` function handles config loading flexibly:
- Full YAML path
- Directory containing configs
- Short name like "sam2.1_hiera_l" (searches near checkpoint or `$SAM2_CONFIG_DIR`)

## Model Checkpoints

- SAM2: `.pt` files (e.g., `sam2_bundle.pt`)
- FastSAM: `FastSAM/weights/FastSAM.pt`
- SAM v1: `sam_vit_h_4b8939.pth` (ViT-H model)

## Recent GUI Improvements (plant_segmenter_neat.py)

### Layout Changes
- **4-panel layout**: Left scrollable controls | Right stacked Preview+Masks | Training at bottom
- **Scrollable left panel**: Canvas + Scrollbar with mousewheel support for the controls panel
- **Vertical stacking**: Preview and Masks panels use `ttk.PanedWindow` with vertical orientation

### Color Theme (Teal)
Applied a cohesive teal color palette:
```python
self.colors = {
    'bg_dark': '#05445E',       # Dark teal - main background
    'bg_medium': '#088395',     # Medium teal - panels
    'bg_light': '#7AB2B2',      # Light teal - sections
    'bg_pale': '#EBF4F6',       # Pale - inputs/entries
    'accent': '#189AB4',        # Accent for buttons
    'accent_hover': '#75E6DA',  # Button hover
    'text_light': '#FFFFFF',    # White text on dark
    'text_dark': '#05445E',     # Dark text on light
    'canvas_bg': '#0A2A3A',     # Dark canvas background
}
```

### Thread Safety Fix
The segmentation worker was causing GUI hangs. Fixed by wrapping all GUI updates in `self.root.after(0, callback)` to ensure they run on the main thread:
```python
def _segment_worker(self):
    # ... segmentation code ...
    def _update_gui():
        self.lb.delete(0, tk.END)
        # ... update listbox, show image, etc.
    self.root.after(0, _update_gui)
```

### Key UI Components
- **Rotation knob**: 50x50px circular dial for image rotation
- **Enhancement sliders**: Brightness, contrast, saturation, sharpness
- **SAM2 parameters**: Points-per-side, IOU threshold, stability score
- **Phenotype checkboxes**: Length/width, color stats, export to CSV
