# app/__init__.py
# Makes `app` importable as a package.
# Core public API re-exported for convenience.

from .app_visuals      import COLORS, ICONS, apply_theme, ToolTip, AnimatedSpinner
from .image_processing import ensure_uint8_rgb, enhance_leaf_edges_rgb
from .mask_utils       import predict_extend_mask, dedupe_by_mask_iou, split_masks_by_cc
from .phenotyping      import compute_phenotypes, write_individual_csv, write_joint_csv
from .sam2_utils       import load_sam2_model, load_sam2_bundle, make_mask_generator

__all__ = [
    "COLORS", "ICONS", "apply_theme", "ToolTip", "AnimatedSpinner",
    "ensure_uint8_rgb", "enhance_leaf_edges_rgb",
    "predict_extend_mask", "dedupe_by_mask_iou", "split_masks_by_cc",
    "compute_phenotypes", "write_individual_csv", "write_joint_csv",
    "load_sam2_model", "load_sam2_bundle", "make_mask_generator",
]
