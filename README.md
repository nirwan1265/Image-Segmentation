# Image-Segmentation

Interactive plant image segmentation and phenotype-focused model training.

This project provides a desktop GUI app for:

- Loading SAM2 checkpoints/bundles and segmenting plant images
- Browsing image folders and running `Segment` / `Segment ALL`
- Saving selected masks, batch outputs, and converted formats
- Target phenotype segmentation (tip/region-of-interest model, no SAM at inference)
- Leaf shape completion model training (occlusion-aware, no SAM)
- Fine-tuning workflows connected from the GUI

## Main App

- Entry point: `plant_segmenter.py`
- Run:

```bash
python plant_segmenter.py
```

## What The App Does

The GUI combines classic segmentation workflows with trainable modules:

- **Base segmentation (SAM2):** general-purpose mask generation from loaded image(s)
- **Target Segment tab:** trains/loads a tip or ROI model using saved masks
- **Completion model tools:** trains/loads a leaf completion model to predict full shape from partial observations
- **Batch workflow:** process folder images, review masks, and export results

## Training Backends Used By The GUI

These scripts are launched internally by `plant_segmenter.py`:

- `sam2_trainer_arabidopsis.py` — SAM2 fine-tuning
- `tip_segmenter_trainer.py` + `tip_segmenter_model.py` — target ROI/tip training + inference
- `leaf_completion_trainer.py` + `leaf_completion_model.py` — completion training + inference

## Notes

- Legacy GUI variants were moved to `old_models/`.
- Main development target is now `plant_segmenter.py`.
