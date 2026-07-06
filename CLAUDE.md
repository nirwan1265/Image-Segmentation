# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an image segmentation project focused on plant/leaf segmentation using SAM (Segment Anything Model) and SAM2. The primary use case is phenotyping - extracting morphological measurements from plant images.

## Key Components

### Main Applications

- **`plant_segmenter.py`**: Interactive tkinter GUI application for leaf segmentation. Features include:
  - Image enhancement controls
  - SAM2 automatic mask generation
  - Custom model segmentation (no SAM needed)
  - Selective mask saving with phenotyping metrics export
  - PCA-based leaf orientation and dimension analysis
  - Color statistics (RGB/HSV) extraction
  - Model training (Train Custom Model, Shape Completion)

- **`occlusion_augmentation.py`**: Data augmentation module for training SAM on occluded leaf images. Supports both SAM2 and SAM(v1) with automatic fallback.

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

## Running the Application

```bash
# Main GUI segmenter
python plant_segmenter.py
# Then click "Load Bundle…" and select sam2_bundle.pt to load the model

# FastSAM inference
python FastSAM/Inference.py --model_path ./weights/FastSAM.pt --img_path <image>

# FastSAM with text prompt
python FastSAM/Inference.py --model_path ./weights/FastSAM.pt --img_path <image> --text_prompt "the leaf"
```

## GUI Structure

### Model Panel

The Model panel offers two segmentation modes:
- **SAM2:** Load SAM2 checkpoint/bundle for general-purpose segmentation
- **Tip Model:** Load a custom-trained lightweight model (no SAM needed at inference)

### Mask Panel Tools

The Masks panel toolbar provides editing tools:
- **Delete/Clear** - Remove selected or all masks
- **Combine** - Merge multiple masks into one
- **Refine** - Re-segment within mask regions using SAM2
- **Split** - Draw a line to split one mask into two (`split_mask_mode()`)
- **Edit** - Open draw/erase dialog for manual pixel editing (`edit_mask_mode()`)

### Training Panel (3 tabs)

1. **Train Custom Model:** Train a lightweight U-Net for specific segmentation tasks
   - Uses `tip_segmenter_trainer.py` + `tip_segmenter_model.py`
   - Dataset: `images/` + `masks/` folders
   - Output: `.pth` model file

2. **Leaf Completion:** Complete partial/occluded leaves using geometric shape fitting
   - Uses `cv2.fitEllipse()` to fit ellipse to mask contour
   - Interactive adjustment: size (+/−), position (drag), rotation (↶/↷)
   - Zoom controls for detailed preview
   - 3-column layout: Options | Original | Completed

3. **Leaf Unfolding:** Unfold folded leaves by rotating mask segments
   - Select 2+ masks representing parts of a folded leaf
   - Rotate one mask around a pivot point (auto-detected junction or mask center)
   - Position adjustment via drag, rotation via buttons
   - Merges masks into single unfolded mask on Update

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
- Custom Model: `.pth` files from Train Custom Model tab
- Mask Completion: `mask_completion.pth`

## Leaf Completion (GUI - Geometric Fitting)

**Purpose:** Complete partial/occluded leaf masks using geometric shape fitting.

**Location:** Training Panel → Leaf Completion tab

### How It Works
1. Fits an ellipse to the mask contour using `cv2.fitEllipse()`
2. User can adjust: size, position (drag), rotation
3. Preview shows original vs completed side-by-side
4. "Update" applies the completed shape to the mask

### Key Methods
- `_fit_shape_to_mask()`: Fits ellipse with scale, offset, rotation parameters
- `_preview_leaf_completion()`: Shows original and completed previews
- `_update_leaf_completion()`: Applies the completed mask
- `_leaf_rotate()`, `_leaf_reset_position()`: Adjustment handlers

### UI Controls
- **Size**: `+`/`−` buttons (5% increments)
- **Position**: Drag on completed preview canvas
- **Rotation**: `↶`/`↷` buttons (5° steps)
- **Zoom**: `+`/`−` buttons, mousewheel, Fit button

---

## Split Mask (GUI - Draw Line to Split)

**Purpose:** Split a single mask into multiple masks by drawing a dividing line.

**Location:** Masks panel → ✂ button

### How It Works
1. User selects one mask and clicks Split
2. Canvas enters split mode - user draws a line through the mask
3. Line creates a "cut" through the mask
4. Connected components become separate masks

### Key Methods
- `split_mask_mode()`: Enter split mode, bind mouse events
- `_apply_split()`: Use cv2.connectedComponents to separate parts
- `_canvas_to_image_coords()`: Convert canvas clicks to image coordinates

---

## Edit Mask (GUI - Draw/Erase Pixels)

**Purpose:** Manually add or remove pixels from a mask with brush tools.

**Location:** Masks panel → ✏ button

### How It Works
1. Opens a dialog with the mask displayed
2. User can Draw (add pixels) or Erase (remove pixels)
3. Adjustable brush size
4. Apply saves changes, Cancel discards

### Key Methods
- `edit_mask_mode()`: Open edit dialog
- `_edit_draw_at()`, `_edit_draw_line()`: Apply brush strokes using cv2.circle/cv2.line
- `_edit_apply()`: Update mask in sr.masks with edited version

---

## Leaf Unfolding (GUI - Rotate & Merge Masks)

**Purpose:** Unfold folded leaves by rotating mask segments and merging them.

**Location:** Training Panel → Leaf Unfolding tab

### How It Works
1. Select 2+ masks in the mask list (parts of a folded leaf)
2. Click "Preview Selected" to see them with different colors
3. Select which mask to rotate
4. Adjust rotation angle and position
5. "Update" merges all masks into one unfolded mask

### Key Methods
- `_preview_leaf_unfolding()`: Loads selected masks and shows preview
- `_rotate_mask()`: Rotates a mask around a pivot point using cv2.warpAffine
- `_get_unfold_pivot()`: Finds pivot point (junction or mask center)
- `_apply_leaf_unfolding()`: Merges masks and optionally removes originals

### UI Controls
- **Rotation**: `↶`/`↷` buttons (5° steps), quick buttons (90°, 180°, -90°)
- **Pivot**: Auto (junction between masks) or mask center
- **Position**: Drag on preview canvas
- **Zoom**: `+`/`−` buttons, Fit button

---

## Mask Completion (CLI - ML-based Amodal Segmentation)

**Purpose:** Predict full leaf shape from partial/occluded leaf masks using ML.

**Script:** `mask_completion.py`

> **Note:** The GUI now uses geometric fitting for leaf completion. This ML approach is available as a CLI tool for batch processing.

### Architecture
- Simple U-Net (mask → mask, no RGB)
- Input: 1-channel binary mask (partial/occluded leaf)
- Output: 1-channel binary mask (completed leaf)
- Size: 128x128 internal processing

### Why Mask-Only Works Better
Previous attempts (`leaf_completion_v2.py`, `leaf_completion_v3.py`) used RGB+mask input. Issues:
1. Train/inference domain gap (synthetic occlusion vs real SAM masks)
2. Model tried to hallucinate pixels without enough data
3. Edge-cropped leaves vs occluded leaves are different problems

The mask-only approach (`mask_completion.py`) works because:
1. No RGB domain issues
2. Model learns leaf SHAPE, not appearance
3. Simpler = fewer things to go wrong

### Training Data Setup
```
leaf_completion/
├── complete_mask/      # Binary masks of complete leaves (white on black)
└── incomplete/         # Real incomplete leaves for testing (optional)
```

### Commands
```bash
# Train (synthetic occlusion 15-50%)
python mask_completion.py train \
    --masks /path/to/complete_mask \
    --output mask_completion.pth \
    --device mps \
    --steps 2000

# Test with synthetic occlusion
python mask_completion.py test \
    --model mask_completion.pth \
    --masks /path/to/complete_mask \
    --output ./test_results

# Inference on real partial mask
python mask_completion.py infer \
    --model mask_completion.pth \
    --mask partial_leaf.png \
    --output completed_leaf.png
```

### Results
- 96-99% IoU on synthetic occlusion test
- Trained on 46 Arabidopsis leaf masks

### Extracting Boundary After Completion
```python
import cv2
mask = cv2.imread('completed.png', cv2.IMREAD_GRAYSCALE)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boundary_points = contours[0]  # numpy array of (x,y) points
```

## GUI Features

### Layout
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

### Thread Safety
The segmentation worker uses `self.root.after(0, callback)` to ensure GUI updates run on the main thread:
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
