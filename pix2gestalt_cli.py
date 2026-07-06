#!/usr/bin/env python3
"""
pix2gestalt CLI - Amodal segmentation for plant leaves.

Given an image and a visible (modal) mask, predicts the complete (amodal) shape.

Setup (one-time):
    git clone https://github.com/cvlab-columbia/pix2gestalt.git
    cd pix2gestalt
    pip install -r requirements.txt
    git clone https://github.com/CompVis/taming-transformers.git
    pip install -e taming-transformers/
    git clone https://github.com/openai/CLIP.git
    pip install -e CLIP/

    # Download weights from HuggingFace:
    # https://huggingface.co/cvlab/pix2gestalt-weights
    # Place epoch=000005.ckpt in pix2gestalt/checkpoints/

Usage:
    python pix2gestalt_cli.py --image leaf.png --mask visible_mask.png --output completed.png

    # With SAM2 to get visible mask first:
    python pix2gestalt_cli.py --image leaf.png --use-sam --output completed.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

# Check if pix2gestalt is available
PIX2GESTALT_PATH = Path(__file__).parent / "pix2gestalt"
if PIX2GESTALT_PATH.exists():
    sys.path.insert(0, str(PIX2GESTALT_PATH))


def _print(msg: str):
    print(msg, flush=True)


def load_and_preprocess_image(path: str, size: int = 256) -> np.ndarray:
    """Load image and resize to square, return RGB uint8."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Letterbox to square
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = img_resized

    return canvas, {"scale": scale, "x0": x0, "y0": y0, "new_w": new_w, "new_h": new_h, "orig_w": w, "orig_h": h}


def load_and_preprocess_mask(path: str, size: int = 256, meta: dict = None) -> np.ndarray:
    """Load binary mask and resize to match image preprocessing."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {path}")

    mask = (mask > 127).astype(np.uint8) * 255

    if meta:
        # Apply same letterbox transform
        mask_resized = cv2.resize(mask, (meta["new_w"], meta["new_h"]), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((size, size), dtype=np.uint8)
        canvas[meta["y0"]:meta["y0"] + meta["new_h"], meta["x0"]:meta["x0"] + meta["new_w"]] = mask_resized
        mask = canvas
    else:
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

    # Convert to RGB format (pix2gestalt expects 3-channel mask)
    mask_rgb = np.stack([mask, mask, mask], axis=-1)
    return mask_rgb


def unletterbox_mask(mask: np.ndarray, meta: dict) -> np.ndarray:
    """Convert mask back to original image size."""
    cropped = mask[meta["y0"]:meta["y0"] + meta["new_h"], meta["x0"]:meta["x0"] + meta["new_w"]]
    return cv2.resize(cropped, (meta["orig_w"], meta["orig_h"]), interpolation=cv2.INTER_NEAREST)


def run_with_pix2gestalt(
    image_rgb: np.ndarray,
    visible_mask_rgb: np.ndarray,
    model,
    device: str,
    n_samples: int = 4,
    ddim_steps: int = 200,
    guidance_scale: float = 2.0,
) -> list[np.ndarray]:
    """Run pix2gestalt inference."""
    try:
        from pix2gestalt.inference import run_pix2gestalt
    except ImportError:
        raise ImportError(
            "pix2gestalt not found. Please clone it:\n"
            "  git clone https://github.com/cvlab-columbia/pix2gestalt.git"
        )

    results = run_pix2gestalt(
        model=model,
        device=device,
        input_im=image_rgb,
        visible_mask=visible_mask_rgb,
        scale=guidance_scale,
        n_samples=n_samples,
        ddim_steps=ddim_steps,
        ddim_eta=1.0,
        precision="autocast",
        h=256,
        w=256,
    )
    return results


def load_pix2gestalt_model(checkpoint_path: str, device: str = "cuda"):
    """Load pix2gestalt model from checkpoint."""
    try:
        from pix2gestalt.inference import load_model_from_config
        from omegaconf import OmegaConf
    except ImportError:
        raise ImportError(
            "pix2gestalt dependencies not found. Please install:\n"
            "  pip install omegaconf\n"
            "  And clone pix2gestalt repo"
        )

    # Default config path
    config_path = PIX2GESTALT_PATH / "configs" / "sd-finetune-pix2gestalt-c_concat-256.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    _print(f"Loading config from: {config_path}")
    config = OmegaConf.load(str(config_path))

    _print(f"Loading model from: {checkpoint_path}")
    model = load_model_from_config(config, str(checkpoint_path), device)

    return model


def extract_amodal_mask_from_completion(
    original_rgb: np.ndarray,
    completion_rgb: np.ndarray,
    visible_mask: np.ndarray,
    threshold: float = 30.0,
) -> np.ndarray:
    """
    Extract amodal mask from the completed image.

    The completion shows the full object. We find pixels that differ
    significantly from background or match the visible object colors.
    """
    # Simple approach: threshold difference from white background
    # or use color similarity to visible region

    # Convert to float for comparison
    comp_f = completion_rgb.astype(np.float32)

    # Method 1: Non-white pixels in completion
    white_diff = np.abs(comp_f - 255.0).max(axis=-1)
    mask1 = white_diff > threshold

    # Method 2: Union with visible mask
    vis_binary = visible_mask[:, :, 0] > 127 if visible_mask.ndim == 3 else visible_mask > 127

    # Combine: completion mask OR visible mask
    amodal = np.logical_or(mask1, vis_binary).astype(np.uint8) * 255

    # Clean up with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    amodal = cv2.morphologyEx(amodal, cv2.MORPH_CLOSE, kernel)
    amodal = cv2.morphologyEx(amodal, cv2.MORPH_OPEN, kernel)

    return amodal


def main():
    parser = argparse.ArgumentParser(
        description="pix2gestalt CLI for amodal leaf segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--mask", required=False, help="Visible/modal mask path (binary)")
    parser.add_argument("--output", required=True, help="Output path for amodal mask")
    parser.add_argument("--output-vis", default=None, help="Output path for visualization")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to pix2gestalt checkpoint (default: pix2gestalt/checkpoints/epoch=000005.ckpt)",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--n-samples", type=int, default=4, help="Number of completion samples")
    parser.add_argument("--ddim-steps", type=int, default=200, help="DDIM sampling steps")
    parser.add_argument("--guidance-scale", type=float, default=2.0, help="Guidance scale")
    parser.add_argument("--size", type=int, default=256, help="Processing size")

    # SAM option for getting visible mask
    parser.add_argument("--use-sam", action="store_true", help="Use SAM2 to generate visible mask")
    parser.add_argument("--sam-checkpoint", default=None, help="SAM2 checkpoint path")

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.image).exists():
        _print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)

    if not args.mask and not args.use_sam:
        _print("ERROR: Either --mask or --use-sam is required")
        sys.exit(1)

    # Load and preprocess image
    _print(f"Loading image: {args.image}")
    image_rgb, meta = load_and_preprocess_image(args.image, args.size)
    _print(f"  Original size: {meta['orig_w']}x{meta['orig_h']}")
    _print(f"  Preprocessed to: {args.size}x{args.size}")

    # Get visible mask
    if args.use_sam:
        _print("SAM2 mask generation not yet implemented in this CLI")
        _print("Please provide a mask with --mask")
        sys.exit(1)
    else:
        _print(f"Loading mask: {args.mask}")
        visible_mask_rgb = load_and_preprocess_mask(args.mask, args.size, meta)

    # Find checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        # Default locations
        candidates = [
            PIX2GESTALT_PATH / "checkpoints" / "epoch=000005.ckpt",
            PIX2GESTALT_PATH / "epoch=000005.ckpt",
            Path("epoch=000005.ckpt"),
        ]
        for c in candidates:
            if c.exists():
                checkpoint_path = str(c)
                break

    if checkpoint_path is None or not Path(checkpoint_path).exists():
        _print("ERROR: Checkpoint not found. Please download from HuggingFace:")
        _print("  https://huggingface.co/cvlab/pix2gestalt-weights")
        _print("  Place epoch=000005.ckpt in pix2gestalt/checkpoints/")
        sys.exit(1)

    # Check device
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        _print("WARN: CUDA not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        _print("WARN: MPS not available, falling back to CPU")
        args.device = "cpu"

    _print(f"Using device: {args.device}")

    # Load model
    _print("Loading pix2gestalt model...")
    try:
        model = load_pix2gestalt_model(checkpoint_path, args.device)
    except Exception as e:
        _print(f"ERROR loading model: {e}")
        _print("\nMake sure pix2gestalt is properly installed:")
        _print("  git clone https://github.com/cvlab-columbia/pix2gestalt.git")
        _print("  cd pix2gestalt && pip install -r requirements.txt")
        sys.exit(1)

    # Run inference
    _print(f"Running pix2gestalt (n_samples={args.n_samples}, steps={args.ddim_steps})...")
    completions = run_with_pix2gestalt(
        image_rgb=image_rgb,
        visible_mask_rgb=visible_mask_rgb,
        model=model,
        device=args.device,
        n_samples=args.n_samples,
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
    )

    _print(f"Generated {len(completions)} completion samples")

    # Extract amodal mask from best completion (use first for now)
    best_completion = completions[0]
    amodal_mask_sq = extract_amodal_mask_from_completion(
        image_rgb, best_completion, visible_mask_rgb
    )

    # Unletterbox to original size
    amodal_mask = unletterbox_mask(amodal_mask_sq, meta)

    # Save outputs
    _print(f"Saving amodal mask to: {args.output}")
    cv2.imwrite(args.output, amodal_mask)

    if args.output_vis:
        # Create visualization grid
        vis_mask_gray = visible_mask_rgb[:, :, 0]
        row1 = np.concatenate([image_rgb, np.stack([vis_mask_gray]*3, axis=-1)], axis=1)
        row2 = np.concatenate([best_completion, np.stack([amodal_mask_sq]*3, axis=-1)], axis=1)
        grid = np.concatenate([row1, row2], axis=0)
        grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.output_vis, grid_bgr)
        _print(f"Saved visualization to: {args.output_vis}")

    _print("Done!")


if __name__ == "__main__":
    main()
