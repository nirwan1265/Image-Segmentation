#!/usr/bin/env python3
"""
SAM2 Fine-tuning Script for Plant Segmentation

Based on: https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code
Tutorial: https://medium.com/@sagieppel/train-fine-tune-segment-anything-2-sam-2-in-60-lines-of-code-928dd29a63b3

This script fine-tunes SAM2 on collected segmentation examples.
Called by the Training panel in plant_segmenter_neat.py.

Usage:
    python sam2_trainer_arabidopsis.py \
        --images /path/to/images \
        --masks /path/to/masks \
        --checkpoint /path/to/sam2.pt \
        --config sam2.1_hiera_l \
        --save_to /path/to/output.pth \
        --steps 10000 \
        --lr 1e-5
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path

import numpy as np
import cv2
import torch

# ----------------------------
# SAM2 imports
# ----------------------------
SAM2_AVAILABLE = True
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    SAM2_AVAILABLE = False
    print("WARNING: SAM2 not available. Install SAM2 repo.")


def setup_hydra():
    """Initialize Hydra for SAM2 config resolution."""
    try:
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        candidates = [
            os.environ.get("SAM2_CONFIG_DIR", ""),
            os.path.expanduser("~/Documents/Github/sam2/configs/sam2.1"),
            "/Users/nirwantandukar/Documents/Github/sam2/configs/sam2.1",
        ]

        for c in candidates:
            if c and os.path.isdir(c):
                GlobalHydra.instance().clear()
                initialize_config_dir(config_dir=os.path.abspath(c), job_name="finetune")
                print(f"Hydra initialized: {c}")
                return True
    except Exception as e:
        print(f"Hydra setup warning: {e}")
    return False


def load_model_from_bundle_or_checkpoint(checkpoint_path: str, config: str, device: str):
    """
    Load SAM2 model from bundle format or regular checkpoint.
    Returns (predictor, model)
    """
    print(f"Loading model from: {checkpoint_path}")

    # Load checkpoint to inspect
    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Check if bundle format
    is_bundle = isinstance(bundle, dict) and (
        "ckpt_bytes" in bundle or "checkpoint_bytes" in bundle or
        "state_dict" in bundle or "cfg" in bundle
    )

    state_dict = None

    if is_bundle:
        print("Detected bundle format")
        meta = bundle.get("meta", {})

        # Get config from bundle if available
        bundle_config = meta.get("config_name") or bundle.get("cfg_short_name")
        if bundle_config:
            print(f"Bundle specifies config: {bundle_config}")
            config = bundle_config

        # Extract checkpoint bytes or state dict
        ck_bytes = bundle.get("ckpt_bytes") or bundle.get("checkpoint_bytes")
        if ck_bytes:
            fd, tmp_path = tempfile.mkstemp(suffix=".pt")
            with os.fdopen(fd, "wb") as fh:
                fh.write(ck_bytes)
            inner = torch.load(tmp_path, map_location="cpu", weights_only=False)
            os.remove(tmp_path)
            if isinstance(inner, dict) and "model" in inner:
                state_dict = inner["model"]
            else:
                state_dict = inner
        elif "state_dict" in bundle:
            state_dict = bundle["state_dict"]
    else:
        # Regular checkpoint
        if isinstance(bundle, dict) and "model" in bundle:
            state_dict = bundle["model"]
        else:
            state_dict = bundle

    # Auto-detect model size if needed
    if not config or config in ["sam2_hiera_s.yaml", "sam2_hiera_s", "(auto-detect)"]:
        if state_dict:
            block_keys = [k for k in state_dict.keys() if "image_encoder.trunk.blocks." in k and ".norm1.weight" in k]
            num_blocks = len(block_keys)
            if num_blocks >= 40:
                config = "sam2.1_hiera_l"
            elif num_blocks >= 20:
                config = "sam2.1_hiera_b+"
            elif num_blocks >= 14:
                config = "sam2.1_hiera_s"
            else:
                config = "sam2.1_hiera_t"
            print(f"Auto-detected config: {config} ({num_blocks} blocks)")

    # Clean config name
    if config and config.endswith(".yaml"):
        config = config[:-5]

    # Setup Hydra
    setup_hydra()

    # Create temp checkpoint and build model
    print(f"Building model with config: {config}")
    fd, tmp_ckpt = tempfile.mkstemp(suffix=".pt")
    torch.save({"model": state_dict}, tmp_ckpt)
    os.close(fd)

    try:
        sam2_model = build_sam2(config, tmp_ckpt, device=device)
    finally:
        os.remove(tmp_ckpt)

    predictor = SAM2ImagePredictor(sam2_model)
    print("Model loaded successfully!")

    return predictor, sam2_model


def read_batch_from_dataset(images_dir: Path, masks_dir: Path, image_size: int = 1024,
                            neg_points: int = 0, allow_empty: bool = False):
    """
    Read a random image and its masks from the dataset.
    Returns (image, masks, points, labels) or None if no valid data.
    """
    # Find all images
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    if not image_files:
        return None

    # Pick random image
    img_path = image_files[np.random.randint(len(image_files))]
    stem = img_path.stem

    # Find corresponding masks
    mask_files = sorted(masks_dir.glob(f"{stem}_inst*.png"))
    if not mask_files:
        # Try without _inst suffix
        mask_files = sorted(masks_dir.glob(f"{stem}*.png"))
        mask_files = [m for m in mask_files if m.stem != stem]  # exclude if same name
    # Marker for "no target" images
    no_target_marker = masks_dir / f"{stem}_nomask.txt"
    if not mask_files and not (allow_empty and no_target_marker.exists()):
        return None

    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = img[..., ::-1]  # BGR to RGB

    # Resize
    r = min(image_size / img.shape[1], image_size / img.shape[0])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))

    # Load masks and generate points
    masks = []
    points = []
    labels = []

    def _rand_points(mask_bool: np.ndarray, n: int):
        coords = np.argwhere(mask_bool > 0)
        if coords.shape[0] == 0:
            return []
        idx = np.random.choice(coords.shape[0], size=min(n, coords.shape[0]), replace=coords.shape[0] < n)
        pts = coords[idx][:, [1, 0]]  # x, y
        return pts.tolist()

    def _rand_points_bg(mask_bool: np.ndarray, n: int):
        coords = np.argwhere(mask_bool == 0)
        if coords.shape[0] == 0:
            return []
        idx = np.random.choice(coords.shape[0], size=min(n, coords.shape[0]), replace=coords.shape[0] < n)
        pts = coords[idx][:, [1, 0]]  # x, y
        return pts.tolist()

    for mask_path in mask_files:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # Resize mask
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)

        if mask.sum() < 10:  # Skip tiny masks
            continue

        masks.append(mask)

        # Positive + negative points for this mask
        pos = _rand_points(mask, 1)
        neg = _rand_points_bg(mask, max(0, int(neg_points)))
        pts = (pos + neg) if pos else neg
        if not pts:
            continue
        points.append(pts)
        lbls = [1] * len(pos) + [0] * len(neg)
        labels.append(lbls)

    # Allow empty images as negative-only samples
    if not masks and allow_empty and no_target_marker.exists():
        empty = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        masks = [empty]
        neg_n = max(1, int(neg_points))
        neg = _rand_points_bg(empty, neg_n)
        if not neg:
            return None
        points = [neg]
        labels = [[0] * len(neg)]

    if not masks:
        return None

    return img, np.array(masks), np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)


def train(
    images_dir: str,
    masks_dir: str,
    checkpoint: str,
    config: str,
    save_to: str,
    steps: int = 10000,
    lr: float = 1e-5,
    device: str = "cpu",
    image_size: int = 1024,
    neg_points: int = 0,
    allow_empty: bool = False
):
    """Main training loop."""

    if not SAM2_AVAILABLE:
        print("ERROR: SAM2 not available!")
        return

    device = (device or "cpu").strip().lower()
    if device.startswith("mps"):
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            print("MPS fallback enabled (PYTORCH_ENABLE_MPS_FALLBACK=1) for unsupported ops.")

    images_path = Path(images_dir)
    masks_path = Path(masks_dir)

    # Check dataset
    img_count = len(list(images_path.glob("*.png"))) + len(list(images_path.glob("*.jpg")))
    if img_count == 0:
        print(f"ERROR: No images found in {images_dir}")
        return
    print(f"Found {img_count} images in dataset")

    # Load model
    predictor, model = load_model_from_bundle_or_checkpoint(checkpoint, config, device)

    # Enable training
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)
    # Note: image_encoder stays frozen by default (much faster, less memory)

    # Optimizer
    optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=lr, weight_decay=4e-5)

    # Mixed precision (for CUDA)
    use_amp = device.startswith("cuda")
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    print(f"\nStarting training for {steps} steps...")
    print(f"  Learning rate: {lr}")
    print(f"  Image size: {image_size}")
    print(f"  Device: {device}")
    print(f"  Mixed precision: {use_amp}")
    print()

    mean_iou = 0
    best_iou = 0

    for itr in range(steps):
        # Read batch
        batch = read_batch_from_dataset(images_path, masks_path, image_size,
                                        neg_points=neg_points, allow_empty=allow_empty)
        if batch is None:
            continue

        image, masks, input_points, input_labels = batch
        if masks.shape[0] == 0:
            continue

        try:
            if use_amp:
                with torch.cuda.amp.autocast():
                    loss, iou = train_step(predictor, image, masks, input_points, input_labels, device)

                predictor.model.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, iou = train_step(predictor, image, masks, input_points, input_labels, device)

                predictor.model.zero_grad()
                loss.backward()
                optimizer.step()

            # Update metrics
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())

            # Log progress
            if itr % 100 == 0:
                print(f"[{itr}/{steps}] loss={float(loss):.4f}, IOU={mean_iou:.4f}")
                sys.stdout.flush()

            # Save checkpoint
            if itr % 1000 == 0 and itr > 0:
                if mean_iou > best_iou:
                    best_iou = mean_iou
                    save_checkpoint(predictor.model, save_to, itr, mean_iou, config)
                    print(f"  Saved checkpoint (IOU={mean_iou:.4f})")

        except Exception as e:
            print(f"  Error at step {itr}: {e}")
            continue

    # Final save
    save_checkpoint(predictor.model, save_to, steps, mean_iou, config)
    print(f"\nTraining complete! Final IOU={mean_iou:.4f}")
    print(f"Saved to: {save_to}")


def train_step(predictor, image, masks, input_points, input_labels, device):
    """Single training step."""

    # Set image (runs image encoder)
    predictor.set_image(image)

    # Prepare prompts
    mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
        input_points, input_labels, box=None, mask_logits=None, normalize_coords=True
    )

    # Prompt encoding
    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
        points=(unnorm_coords, labels),
        boxes=None,
        masks=None,
    )

    # Mask decoder
    batched_mode = unnorm_coords.shape[0] > 1
    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]

    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
        repeat_image=batched_mode,
        high_res_features=high_res_features,
    )

    # Upscale masks
    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

    # Ground truth
    gt_mask = torch.tensor(masks.astype(np.float32)).to(device)
    prd_mask = torch.sigmoid(prd_masks[:, 0])

    # Segmentation loss (cross entropy)
    seg_loss = (
        -gt_mask * torch.log(prd_mask + 1e-5)
        - (1 - gt_mask) * torch.log(1 - prd_mask + 1e-5)
    ).mean()

    # IOU score loss
    inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
    iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter + 1e-5)
    score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

    # Combined loss
    loss = seg_loss + score_loss * 0.05

    return loss, iou


def save_checkpoint(model, path, step, iou, config):
    """Save model checkpoint."""
    save_dict = {
        "state_dict": model.state_dict(),
        "meta": {
            "step": step,
            "iou": float(iou),
            "config": config,
            "sam2": True,
        }
    }
    torch.save(save_dict, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SAM2 for plant segmentation")
    parser.add_argument("--images", required=True, help="Directory containing training images")
    parser.add_argument("--masks", required=True, help="Directory containing mask images")
    parser.add_argument("--checkpoint", required=True, help="Path to SAM2 checkpoint or bundle")
    parser.add_argument("--config", default=None, help="SAM2 config (auto-detected if not provided)")
    parser.add_argument("--save_to", required=True, help="Output path for fine-tuned model")
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--device", default="cpu", help="Device (cuda/mps/cpu)")
    parser.add_argument("--size", type=int, default=1024, help="Training image size")
    parser.add_argument("--neg-points", type=int, default=0, help="Negative points per mask (background)")
    parser.add_argument("--allow-empty", action="store_true", help="Allow images with no target (negative-only)")

    args = parser.parse_args()

    train(
        images_dir=args.images,
        masks_dir=args.masks,
        checkpoint=args.checkpoint,
        config=args.config,
        save_to=args.save_to,
        steps=args.steps,
        lr=args.lr,
        device=args.device,
        image_size=args.size,
        neg_points=args.neg_points,
        allow_empty=args.allow_empty
    )
