#!/usr/bin/env python3
"""
Test on real incomplete leaves - WITH PADDING to allow extension.

Key insight: Model can't predict outside image bounds.
Solution: Pad the image with white space, predict, then see if mask extends into padding.
"""
import sys
from typing import Tuple
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from leaf_completion_v2 import UNetCompletion, auto_generate_mask
import torch


def pad_image_for_extension(rgb: np.ndarray, pad_percent: float = 0.3) -> Tuple[np.ndarray, dict]:
    """
    Pad image with white space to allow leaf extension.
    Returns padded image and padding info.
    """
    h, w = rgb.shape[:2]
    pad_h = int(h * pad_percent)
    pad_w = int(w * pad_percent)

    # Create padded canvas
    new_h = h + 2 * pad_h
    new_w = w + 2 * pad_w
    padded = np.full((new_h, new_w, 3), 255, dtype=np.uint8)

    # Place original image in center
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = rgb

    return padded, {'pad_h': pad_h, 'pad_w': pad_w, 'orig_h': h, 'orig_w': w}


def predict_with_padding(model, rgb, visible_mask, size, device, pad_percent=0.3):
    """
    Run prediction with padding to allow extension beyond original bounds.
    """
    from leaf_completion_v2 import letterbox, unletterbox

    h, w = rgb.shape[:2]
    pad_h = int(h * pad_percent)
    pad_w = int(w * pad_percent)

    # Create padded image
    new_h = h + 2 * pad_h
    new_w = w + 2 * pad_w
    padded_rgb = np.full((new_h, new_w, 3), 255, dtype=np.uint8)
    padded_rgb[pad_h:pad_h+h, pad_w:pad_w+w] = rgb

    # Create padded visible mask
    padded_vis = np.zeros((new_h, new_w), dtype=np.uint8)
    padded_vis[pad_h:pad_h+h, pad_w:pad_w+w] = visible_mask

    # Letterbox padded image
    rgb_sq, info = letterbox(padded_rgb, size, pad_value=255)
    vis_sq, _ = letterbox(padded_vis, size, pad_value=0)

    model.eval()
    with torch.no_grad():
        rgb_t = torch.from_numpy(rgb_sq.astype(np.float32) / 255.0).permute(2, 0, 1)
        vis_t = torch.from_numpy(vis_sq[None, ...].astype(np.float32) / 255.0)
        x = torch.cat([rgb_t, vis_t], dim=0).unsqueeze(0).to(device)

        logits = model(x)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    pred_sq = (probs >= 0.5).astype(np.uint8) * 255
    pred_padded = unletterbox(pred_sq, info)

    # Extract predicted mask for original region and extended region
    pred_original = pred_padded[pad_h:pad_h+h, pad_w:pad_w+w]
    pred_full = pred_padded

    return pred_full, pred_original, (pad_h, pad_w, h, w)


def create_visualization(rgb, visible_mask, pred_original, pred_full, pad_info):
    """Create visualization showing original and extension."""
    pad_h, pad_w, h, w = pad_info

    # Original region comparison
    out1 = rgb.copy()
    vis_bin = visible_mask > 127
    pred_bin = pred_original > 127

    # Green = visible, Magenta = predicted extension
    extended = pred_bin & ~vis_bin
    out1[vis_bin] = (0.7 * out1[vis_bin] + 0.3 * np.array([0, 255, 0])).astype(np.uint8)
    out1[extended] = (0.5 * out1[extended] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)

    # Draw contours
    vis_cnt, _ = cv2.findContours(visible_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_cnt, _ = cv2.findContours(pred_original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out1, vis_cnt, -1, (0, 200, 0), 2)
    cv2.drawContours(out1, pred_cnt, -1, (255, 0, 255), 2)

    # Full padded view
    new_h = h + 2 * pad_h
    new_w = w + 2 * pad_w
    out2 = np.full((new_h, new_w, 3), 240, dtype=np.uint8)  # Light gray padding
    out2[pad_h:pad_h+h, pad_w:pad_w+w] = rgb

    # Draw original boundary
    cv2.rectangle(out2, (pad_w, pad_h), (pad_w+w-1, pad_h+h-1), (100, 100, 100), 2)

    # Overlay prediction on padded view
    pred_bin_full = pred_full > 127
    out2[pred_bin_full] = (0.6 * out2[pred_bin_full] + 0.4 * np.array([255, 0, 255])).astype(np.uint8)

    pred_cnt_full, _ = cv2.findContours(pred_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out2, pred_cnt_full, -1, (255, 0, 255), 2)

    # Resize out2 to match out1 height
    scale = h / new_h
    out2_resized = cv2.resize(out2, (int(new_w * scale), h))

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out1, "Original+Pred", (5, 15), font, 0.4, (0, 0, 0), 1)
    cv2.putText(out2_resized, "Padded View", (5, 15), font, 0.4, (0, 0, 0), 1)

    result = np.concatenate([out1, out2_resized], axis=1)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--pad-percent", type=float, default=0.4, help="How much to pad (0.3 = 30%)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    size = ckpt.get("size", 256)

    model = UNetCompletion(in_ch=4, base_ch=32)
    model.load_state_dict(ckpt["state_dict"])

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    model.to(device)
    model.eval()

    images_dir = Path(args.images)
    exts = {".png", ".jpg", ".jpeg"}
    image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in exts]

    print(f"Processing {len(image_paths)} images with {args.pad_percent*100:.0f}% padding...")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for img_path in image_paths:
        print(f"\n{img_path.name}:")

        rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        visible_mask = auto_generate_mask(rgb)

        # Predict with padding
        pred_full, pred_original, pad_info = predict_with_padding(
            model, rgb, visible_mask, size, device, args.pad_percent
        )

        # Stats
        vis_px = (visible_mask > 127).sum()
        pred_orig_px = (pred_original > 127).sum()
        pred_full_px = (pred_full > 127).sum()
        extended_in_orig = ((pred_original > 127) & (visible_mask <= 127)).sum()
        extended_in_pad = pred_full_px - (pred_full[pad_info[0]:pad_info[0]+pad_info[2],
                                                     pad_info[1]:pad_info[1]+pad_info[3]] > 127).sum()

        print(f"  Visible: {vis_px} px")
        print(f"  Predicted (original region): {pred_orig_px} px")
        print(f"  Extended in original: {extended_in_orig} px")
        print(f"  Predicted (with padding): {pred_full_px} px")
        print(f"  Extended into padding: {extended_in_pad} px")

        # Visualization
        viz = create_visualization(rgb, visible_mask, pred_original, pred_full, pad_info)
        all_results.append(viz)

        out_path = out_dir / f"{img_path.stem}_extended.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))

    # Grid
    if all_results:
        max_w = max(r.shape[1] for r in all_results)
        padded = []
        for r in all_results:
            if r.shape[1] < max_w:
                pad = np.full((r.shape[0], max_w - r.shape[1], 3), 255, dtype=np.uint8)
                r = np.concatenate([r, pad], axis=1)
            padded.append(r)
        grid = np.concatenate(padded, axis=0)
        cv2.imwrite(str(out_dir / "extended_grid.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"\nSaved grid: {out_dir}/extended_grid.png")

    print(f"\nDone! Results in: {out_dir}")


if __name__ == "__main__":
    main()
