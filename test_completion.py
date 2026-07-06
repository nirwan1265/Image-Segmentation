#!/usr/bin/env python3
"""
Proper test for leaf completion - with ground truth comparison.

Takes complete leaves, artificially occludes them, predicts completion,
and shows side-by-side comparison with ground truth.
"""
import sys
import numpy as np
import cv2
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
from leaf_completion_v2 import (
    UNetCompletion, letterbox, unletterbox, auto_generate_mask,
    create_random_shape, LetterboxInfo
)
import torch


def erode_mask_from_edges(mask: np.ndarray, erosion_percent: float = 0.3) -> np.ndarray:
    """
    Erode mask from edges to simulate occlusion.
    Removes a percentage of the mask from the borders.
    """
    h, w = mask.shape

    # Calculate erosion kernel size based on mask size
    mask_pixels = (mask > 0).sum()
    target_removal = int(mask_pixels * erosion_percent)

    # Use distance transform to find edge pixels
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)

    # Find threshold that removes approximately target_removal pixels
    sorted_dists = np.sort(dist[mask > 0])
    if len(sorted_dists) > 0:
        threshold_idx = min(target_removal, len(sorted_dists) - 1)
        threshold = sorted_dists[threshold_idx]

        # Create eroded mask (keep pixels further from edge)
        eroded = ((dist > threshold) & (mask > 0)).astype(np.uint8) * 255
    else:
        eroded = mask.copy()

    return eroded


def occlude_from_random_side(mask: np.ndarray, occlusion_percent: float = 0.3) -> np.ndarray:
    """
    Remove a chunk from a random side of the mask to simulate real occlusion.
    """
    h, w = mask.shape
    result = mask.copy()

    # Find bounding box of mask
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return result

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    mask_w = x_max - x_min
    mask_h = y_max - y_min

    # Choose random side: 0=left, 1=right, 2=top, 3=bottom
    side = np.random.randint(0, 4)

    if side == 0:  # Remove from left
        cut_x = x_min + int(mask_w * occlusion_percent)
        result[:, :cut_x] = 0
    elif side == 1:  # Remove from right
        cut_x = x_max - int(mask_w * occlusion_percent)
        result[:, cut_x:] = 0
    elif side == 2:  # Remove from top
        cut_y = y_min + int(mask_h * occlusion_percent)
        result[:cut_y, :] = 0
    else:  # Remove from bottom
        cut_y = y_max - int(mask_h * occlusion_percent)
        result[cut_y:, :] = 0

    return result


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute IoU between prediction and ground truth."""
    pred_bin = (pred > 127).astype(np.float32)
    gt_bin = (gt > 127).astype(np.float32)

    intersection = (pred_bin * gt_bin).sum()
    union = ((pred_bin + gt_bin) > 0).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def create_comparison_image(
    original_rgb: np.ndarray,
    gt_mask: np.ndarray,
    visible_mask: np.ndarray,
    predicted_mask: np.ndarray,
    iou: float
) -> np.ndarray:
    """
    Create a nice comparison visualization.
    Shows: Original | Original+Visible(green) | Original+Predicted(magenta) | Original+GT(cyan)
    """
    h, w = original_rgb.shape[:2]

    # Resize masks to match image
    if gt_mask.shape[:2] != (h, w):
        gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if visible_mask.shape[:2] != (h, w):
        visible_mask = cv2.resize(visible_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if predicted_mask.shape[:2] != (h, w):
        predicted_mask = cv2.resize(predicted_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    def overlay_mask(img, mask, color, alpha=0.5):
        out = img.copy()
        m = mask > 127
        overlay = np.zeros_like(out)
        overlay[:] = color
        out[m] = (alpha * overlay[m] + (1 - alpha) * out[m]).astype(np.uint8)
        # Draw contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)
        return out

    # Create panels
    panel1 = original_rgb.copy()  # Original
    panel2 = overlay_mask(original_rgb, visible_mask, (0, 255, 0))  # Visible (green)
    panel3 = overlay_mask(original_rgb, predicted_mask, (255, 0, 255))  # Predicted (magenta)
    panel4 = overlay_mask(original_rgb, gt_mask, (0, 255, 255))  # Ground truth (cyan)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel1, "Original", (5, 20), font, 0.5, (0, 0, 0), 2)
    cv2.putText(panel2, "Visible", (5, 20), font, 0.5, (0, 100, 0), 2)
    cv2.putText(panel3, f"Predicted", (5, 20), font, 0.5, (100, 0, 100), 2)
    cv2.putText(panel4, f"GT (IoU:{iou:.2f})", (5, 20), font, 0.5, (0, 100, 100), 2)

    # Stack horizontally
    row = np.concatenate([panel1, panel2, panel3, panel4], axis=1)

    return row


def predict_with_model(model, rgb, visible_mask, size, device, shape_only=False):
    """Run model prediction."""
    model.eval()

    # Letterbox
    rgb_sq, info = letterbox(rgb, size, pad_value=255)
    vis_sq, _ = letterbox(visible_mask, size, pad_value=0)

    with torch.no_grad():
        if shape_only:
            x = torch.from_numpy(vis_sq[None, None, ...].astype(np.float32) / 255.0)
        else:
            rgb_t = torch.from_numpy(rgb_sq.astype(np.float32) / 255.0).permute(2, 0, 1)
            vis_t = torch.from_numpy(vis_sq[None, ...].astype(np.float32) / 255.0)
            x = torch.cat([rgb_t, vis_t], dim=0).unsqueeze(0)

        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask_sq = (probs >= 0.5).astype(np.uint8) * 255

    # Unletterbox
    mask = unletterbox(mask_sq, info)
    return mask


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test leaf completion with ground truth")
    parser.add_argument("--model", required=True, help="Model checkpoint")
    parser.add_argument("--images", required=True, help="Complete images folder")
    parser.add_argument("--output", required=True, help="Output folder for results")
    parser.add_argument("--num", type=int, default=3, help="Number of images to test")
    parser.add_argument("--occlusion", type=float, default=0.3, help="Occlusion percentage (0.1-0.5)")
    parser.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    shape_only = ckpt.get("shape_only", False)
    size = ckpt.get("size", 256)

    in_ch = 1 if shape_only else 4
    model = UNetCompletion(in_ch=in_ch, base_ch=32)
    model.load_state_dict(ckpt["state_dict"])

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    model.to(device)
    model.eval()

    print(f"Model loaded (shape_only={shape_only}, size={size}, device={device})")

    # Find images
    images_dir = Path(args.images)
    exts = {".png", ".jpg", ".jpeg"}
    image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in exts][:args.num]

    print(f"Testing on {len(image_paths)} images with {args.occlusion*100:.0f}% occlusion")

    # Output dir
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    ious = []
    all_rows = []

    for img_path in image_paths:
        print(f"\nProcessing: {img_path.name}")

        # Load image
        rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Generate ground truth mask
        gt_mask = auto_generate_mask(rgb, method="color")
        print(f"  GT mask pixels: {(gt_mask > 0).sum()}")

        # Create synthetic occlusion (remove from edge)
        visible_mask = occlude_from_random_side(gt_mask, args.occlusion)
        print(f"  Visible mask pixels: {(visible_mask > 0).sum()}")

        # Predict completion
        predicted_mask = predict_with_model(model, rgb, visible_mask, size, device, shape_only)
        print(f"  Predicted mask pixels: {(predicted_mask > 0).sum()}")

        # Compute IoU
        iou = compute_iou(predicted_mask, gt_mask)
        ious.append(iou)
        print(f"  IoU: {iou:.4f}")

        # Create comparison visualization
        comparison = create_comparison_image(rgb, gt_mask, visible_mask, predicted_mask, iou)
        all_rows.append(comparison)

        # Save individual result
        out_path = out_dir / f"{img_path.stem}_comparison.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    # Create combined grid
    if all_rows:
        # Make all rows same width
        max_w = max(r.shape[1] for r in all_rows)
        padded_rows = []
        for r in all_rows:
            if r.shape[1] < max_w:
                pad = np.full((r.shape[0], max_w - r.shape[1], 3), 255, dtype=np.uint8)
                r = np.concatenate([r, pad], axis=1)
            padded_rows.append(r)

        grid = np.concatenate(padded_rows, axis=0)
        grid_path = out_dir / "comparison_grid.png"
        cv2.imwrite(str(grid_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"\nSaved comparison grid: {grid_path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"RESULTS:")
    print(f"  Mean IoU: {np.mean(ious):.4f}")
    print(f"  Min IoU:  {np.min(ious):.4f}")
    print(f"  Max IoU:  {np.max(ious):.4f}")
    print(f"{'='*50}")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
