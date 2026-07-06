#!/usr/bin/env python3
"""
Test on real incomplete leaves - show visible vs predicted completion.
"""
import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from leaf_completion_v2 import UNetCompletion, letterbox, unletterbox, auto_generate_mask
import torch


def predict_with_model(model, rgb, visible_mask, size, device, shape_only=False):
    """Run model prediction."""
    model.eval()
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
    return unletterbox(mask_sq, info)


def create_visualization(rgb, visible_mask, predicted_mask):
    """
    Create visualization showing:
    - Original image
    - Visible mask outline (green)
    - Predicted completion outline (magenta) - shows the ADDED part
    """
    h, w = rgb.shape[:2]

    # Resize masks
    if visible_mask.shape[:2] != (h, w):
        visible_mask = cv2.resize(visible_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if predicted_mask.shape[:2] != (h, w):
        predicted_mask = cv2.resize(predicted_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Create output image
    out = rgb.copy()

    # Find the "completed" region (predicted but not visible)
    visible_bin = visible_mask > 127
    predicted_bin = predicted_mask > 127
    completed_region = predicted_bin & ~visible_bin  # What was ADDED by completion

    # Overlay visible region (light green)
    out[visible_bin] = (0.7 * out[visible_bin] + 0.3 * np.array([0, 255, 0])).astype(np.uint8)

    # Overlay completed/added region (magenta - this is the interesting part!)
    out[completed_region] = (0.5 * out[completed_region] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)

    # Draw contours
    vis_contours, _ = cv2.findContours(visible_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(out, vis_contours, -1, (0, 200, 0), 2)  # Green = visible
    cv2.drawContours(out, pred_contours, -1, (255, 0, 255), 2)  # Magenta = predicted full

    # Add legend
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, "Green=Visible", (5, 15), font, 0.4, (0, 150, 0), 1)
    cv2.putText(out, "Magenta=Completed", (5, 30), font, 0.4, (150, 0, 150), 1)

    # Create side-by-side: Original | Annotated
    panel_orig = rgb.copy()
    cv2.putText(panel_orig, "Original", (5, 15), font, 0.4, (0, 0, 0), 1)

    result = np.concatenate([panel_orig, out], axis=1)

    # Calculate completion stats
    visible_area = visible_bin.sum()
    predicted_area = predicted_bin.sum()
    added_area = completed_region.sum()
    completion_percent = (added_area / max(visible_area, 1)) * 100

    return result, {
        "visible_pixels": int(visible_area),
        "predicted_pixels": int(predicted_area),
        "added_pixels": int(added_area),
        "completion_percent": completion_percent
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", required=True, help="Incomplete images folder")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="mps")
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

    # Find images
    images_dir = Path(args.images)
    exts = {".png", ".jpg", ".jpeg"}
    image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in exts]

    print(f"Processing {len(image_paths)} incomplete images...")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for img_path in image_paths:
        print(f"\n{img_path.name}:")

        # Load image
        rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Auto-generate visible mask (what we can see)
        visible_mask = auto_generate_mask(rgb, method="color")

        # Predict completion
        predicted_mask = predict_with_model(model, rgb, visible_mask, size, device, shape_only)

        # Create visualization
        viz, stats = create_visualization(rgb, visible_mask, predicted_mask)

        print(f"  Visible: {stats['visible_pixels']} px")
        print(f"  Predicted: {stats['predicted_pixels']} px")
        print(f"  Added: {stats['added_pixels']} px ({stats['completion_percent']:.1f}% expansion)")

        # Save
        out_path = out_dir / f"{img_path.stem}_completion.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))

        all_results.append(viz)

    # Create grid
    if all_results:
        max_w = max(r.shape[1] for r in all_results)
        padded = []
        for r in all_results:
            if r.shape[1] < max_w:
                pad = np.full((r.shape[0], max_w - r.shape[1], 3), 255, dtype=np.uint8)
                r = np.concatenate([r, pad], axis=1)
            padded.append(r)

        grid = np.concatenate(padded, axis=0)
        grid_path = out_dir / "incomplete_completion_grid.png"
        cv2.imwrite(str(grid_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"\nSaved grid: {grid_path}")

    print(f"\nDone! Results in: {out_dir}")


if __name__ == "__main__":
    main()
