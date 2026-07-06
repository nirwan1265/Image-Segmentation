# Image-Segmentation

Interactive plant image segmentation and phenotype-focused model training.

This project provides a desktop GUI app for:

- Loading SAM2 checkpoints/bundles and segmenting plant images
- Training custom segmentation models (no SAM needed at inference)
- Browsing image folders and running `Segment` / `Segment ALL`
- Saving selected masks, batch outputs, and converted formats
- Leaf shape completion model training (occlusion-aware)

## Main App

- Entry point: `plant_segmenter.py`
- Run:

```bash
python plant_segmenter.py
```

## What The App Does

The GUI combines classic segmentation workflows with trainable modules:

- **Base segmentation (SAM2):** General-purpose mask generation from loaded image(s)
- **Custom Model segmentation:** Use your own trained lightweight model (no SAM needed)
- **Shape Completion:** Predict full leaf masks from partial/occluded observations
- **Batch workflow:** Process folder images, review masks, and export results

## Model Panel

The Model panel lets you choose between two segmentation modes:

- **SAM2:** Load a SAM2 checkpoint/bundle for general-purpose segmentation
- **Custom Model:** Load a custom-trained lightweight model that doesn't require SAM

## Mask Editing Tools

The Masks panel toolbar includes editing tools:

- **🗑 Delete** - Remove selected masks
- **✖ Clear** - Remove all masks
- **🔗 Combine** - Merge selected masks into one
- **🔍 Refine** - Re-segment within mask regions
- **✏ Edit** - Edit mask boundary by dragging control points

### Edit Mask Boundary
1. Select a single mask
2. Click the ✏ button - opens boundary editor
3. **Drag points** to reshape the mask boundary
4. **Add points**: Ctrl/Cmd+click OR toggle "➕ Add Point" button then click
5. **Delete points**: Shift+click OR toggle "➖ Delete Point" button then click
6. Click **Smooth** to smooth the contour
7. Click **Apply** to save or **Cancel** to discard

## Training Panel

The Training panel has three tabs:

### Train Custom Model

Train a lightweight U-Net segmentation model using your own masks:

1. Choose a dataset folder (will create `images/` and `masks/` subfolders)
2. Add segmentation examples by selecting masks and clicking "Add selected masks as target"
3. Configure training parameters (steps, learning rate, image size, etc.)
4. Click "Train Model" to start training
5. Load the trained model to use it for segmentation without SAM

### Leaf Completion

Complete partial/occluded leaf masks using geometric shape fitting:

1. Select a mask from the mask list
2. Click "Preview" to see the fitted shape (ellipse/oval)
3. Adjust the shape using:
   - **Size**: `+`/`−` buttons to grow/shrink by 5%
   - **Position**: Drag on the preview to reposition
   - **Rotation**: `↶`/`↷` buttons to rotate by 5°
   - **Zoom**: `+`/`−` buttons or mousewheel to zoom preview
4. Click "✏ Edit Shape" to fine-tune the boundary by dragging control points
5. Click "Update" to apply the completed shape to the mask
6. Or click "Cancel" to discard changes

The geometric approach uses OpenCV's `cv2.fitEllipse()` to fit an ellipse to the mask contour, then allows manual refinement with the contour editor for precise control.

### Leaf Unfolding

Unfold folded leaves by rotating mask segments:

1. Select 2+ masks in the mask list (parts of a folded leaf)
2. Click "Preview Selected" to see the masks combined
3. Select which mask to rotate using the radio buttons
4. Adjust using:
   - **Rotation**: `↶`/`↷` buttons (5° steps) or quick buttons (90°, 180°, -90°)
   - **Pivot point**: Auto (junction between masks) or mask center
   - **Position**: Drag on the preview to reposition
   - **Zoom**: `+`/`−` buttons to zoom preview
5. Click "Update" to merge into a single unfolded mask
6. Optionally remove the original masks

## Training Scripts

These scripts are launched internally by `plant_segmenter.py`:

- `tip_segmenter_trainer.py` + `tip_segmenter_model.py` — Custom model training + inference
- `mask_completion.py` — Shape completion training + inference

---

## Mask Completion (CLI - Amodal Leaf Segmentation)

**Goal:** Predict the full leaf shape from a partial/occluded leaf mask using ML.

**Script:** `mask_completion.py`

> **Note:** The GUI now uses geometric shape fitting (ellipse) for leaf completion, which is faster and more interactive. The ML-based approach below is still available as a CLI tool for batch processing or cases where geometric fitting isn't sufficient.

### How It Works

```
Complete leaf mask → Synthetic occlusion (15-50% removed) → Partial mask
                                    ↓
                            U-Net learns to predict
                                    ↓
                        Partial mask → Full mask
```

The model is a simple U-Net that takes a binary mask as input and outputs the completed binary mask. No RGB needed - just mask to mask.

### Data Setup

```
leaf_completion/
├── complete/           # Complete leaf RGB images (optional, for reference)
├── complete_mask/      # Complete leaf binary masks (white leaf on black)
└── incomplete/         # Real incomplete leaves for testing (optional)
```

Mask naming: `ara2012_plant001_mask_crop_2.png`

### Training

```bash
python mask_completion.py train \
    --masks /path/to/complete_mask \
    --output mask_completion.pth \
    --device mps \
    --steps 2000
```

### Testing (with synthetic occlusion)

```bash
python mask_completion.py test \
    --model mask_completion.pth \
    --masks /path/to/complete_mask \
    --output ./test_results \
    --num 6 \
    --occlusion 0.3
```

Output shows: Original → Occluded → Predicted → Ground Truth → Difference

### Inference (on real incomplete masks)

```bash
python mask_completion.py infer \
    --model mask_completion.pth \
    --mask /path/to/partial_mask.png \
    --output completed.png
```

### Extract Boundary

After getting the completed mask, extract the leaf boundary:

```python
import cv2
mask = cv2.imread('completed.png', cv2.IMREAD_GRAYSCALE)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boundary = contours[0]  # Leaf outline points
```

### Results

- IoU: 96-99% on synthetic occlusion test
- Trained on 46 complete Arabidopsis leaf masks
- Model file: `mask_completion.pth`

---

## Notes

- Legacy GUI variants were moved to `old_models/`.
- Main development target is now `plant_segmenter.py`.
- Old leaf completion attempts (`leaf_completion_v2.py`, `leaf_completion_v3.py`) used RGB+mask input - didn't work well. The mask-only approach (`mask_completion.py`) works better.
