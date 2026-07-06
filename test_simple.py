#!/usr/bin/env python3
"""
Simple test - just show the predicted complete leaf outline with dotted border.
"""
import sys
from typing import Tuple
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from leaf_completion_v2 import UNetCompletion, letterbox, unletterbox, auto_generate_mask
import torch


def draw_dotted_contour(img, contour, color, thickness=2, gap=10):
    """Draw a dotted/dashed contour."""
    for i in range(0, len(contour), gap):
        pt1 = tuple(contour[i][0])
        pt2 = tuple(contour[min(i + gap//2, len(contour)-1)][0])
        cv2.line(img, pt1, pt2, color, thickness)


def predict_with_padding(model, rgb, visible_mask, size, device, pad_percent=0.5):
    """Predict with padding to allow extension."""
    h, w = rgb.shape[:2]
    pad_h = int(h * pad_percent)
    pad_w = int(w * pad_percent)

    # Padded image
    new_h = h + 2 * pad_h
    new_w = w + 2 * pad_w
    padded_rgb = np.full((new_h, new_w, 3), 255, dtype=np.uint8)
    padded_rgb[pad_h:pad_h+h, pad_w:pad_w+w] = rgb

    # Padded visible mask
    padded_vis = np.zeros((new_h, new_w), dtype=np.uint8)
    padded_vis[pad_h:pad_h+h, pad_w:pad_w+w] = visible_mask

    # Letterbox and predict
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

    return pred_padded, (pad_h, pad_w, h, w)


def create_simple_viz(rgb, visible_mask, pred_full, pad_info):
    """
    Simple visualization:
    - Show original leaf
    - Draw SOLID green line for visible boundary
    - Draw DOTTED magenta line for predicted complete boundary
    """
    pad_h, pad_w, h, w = pad_info

    # Create larger canvas to show extension
    canvas_h = h + 2 * pad_h
    canvas_w = w + 2 * pad_w
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    # Place original image in center
    canvas[pad_h:pad_h+h, pad_w:pad_w+w] = rgb

    # Draw original image boundary (thin gray)
    cv2.rectangle(canvas, (pad_w, pad_h), (pad_w+w-1, pad_h+h-1), (180, 180, 180), 1)

    # Get visible mask contour (solid green)
    padded_vis = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    padded_vis[pad_h:pad_h+h, pad_w:pad_w+w] = visible_mask
    vis_contours, _ = cv2.findContours(padded_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get predicted contour (dotted magenta)
    pred_contours, _ = cv2.findContours(pred_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw predicted boundary as DOTTED magenta line
    for cnt in pred_contours:
        if len(cnt) > 10:
            # Draw dotted line
            pts = cnt.reshape(-1, 2)
            for i in range(0, len(pts) - 1, 2):
                pt1 = tuple(pts[i])
                pt2 = tuple(pts[min(i+1, len(pts)-1)])
                cv2.line(canvas, pt1, pt2, (255, 0, 255), 3)  # Magenta dotted

    # Draw visible boundary as SOLID green line
    cv2.drawContours(canvas, vis_contours, -1, (0, 200, 0), 2)

    # Add legend
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Green = What we can see", (10, 25), font, 0.6, (0, 150, 0), 2)
    cv2.putText(canvas, "Magenta = Predicted complete shape", (10, 50), font, 0.6, (200, 0, 200), 2)
    cv2.putText(canvas, "Gray box = Original image boundary", (10, 75), font, 0.6, (120, 120, 120), 2)

    return canvas


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="mps")
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

    images_dir = Path(args.images)
    exts = {".png", ".jpg", ".jpeg"}
    image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in exts]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for img_path in image_paths:
        print(f"Processing: {img_path.name}")

        rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        visible_mask = auto_generate_mask(rgb)

        pred_full, pad_info = predict_with_padding(model, rgb, visible_mask, size, device, pad_percent=0.5)

        viz = create_simple_viz(rgb, visible_mask, pred_full, pad_info)
        all_results.append(viz)

        out_path = out_dir / f"{img_path.stem}_completed.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))

    # Make grid
    if all_results:
        max_w = max(r.shape[1] for r in all_results)
        max_h = max(r.shape[0] for r in all_results)
        padded = []
        for r in all_results:
            # Pad to same size
            p = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
            p[:r.shape[0], :r.shape[1]] = r
            padded.append(p)

        # Stack vertically
        grid = np.concatenate(padded, axis=0)
        grid_path = out_dir / "results_grid.png"
        cv2.imwrite(str(grid_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"\nSaved: {grid_path}")

    print("Done!")


if __name__ == "__main__":
    main()
