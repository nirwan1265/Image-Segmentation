#!/usr/bin/env python3
"""
Test WITHOUT padding - predict only within original image bounds.
Shows what the model predicts should be filled in.
"""
import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from leaf_completion_v2 import UNetCompletion, letterbox, unletterbox, auto_generate_mask
import torch


def predict_no_padding(model, rgb, visible_mask, size, device):
    """Predict without any padding - just the original image."""
    # Letterbox to model size
    rgb_sq, info = letterbox(rgb, size, pad_value=255)
    vis_sq, _ = letterbox(visible_mask, size, pad_value=0)

    model.eval()
    with torch.no_grad():
        rgb_t = torch.from_numpy(rgb_sq.astype(np.float32) / 255.0).permute(2, 0, 1)
        vis_t = torch.from_numpy(vis_sq[None, ...].astype(np.float32) / 255.0)
        x = torch.cat([rgb_t, vis_t], dim=0).unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    pred_sq = (probs >= 0.5).astype(np.uint8) * 255
    pred = unletterbox(pred_sq, info)

    return pred


def draw_dotted_contour(img, contour, color, thickness=2, gap=5):
    """Draw a dotted contour."""
    pts = contour.reshape(-1, 2)
    for i in range(0, len(pts), 2):
        pt1 = tuple(pts[i])
        pt2 = tuple(pts[min(i + 1, len(pts) - 1)])
        cv2.line(img, pt1, pt2, color, thickness)


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
        h, w = rgb.shape[:2]

        visible_mask = auto_generate_mask(rgb)
        pred_mask = predict_no_padding(model, rgb, visible_mask, size, device)

        # Scale up for better visualization (images are tiny)
        scale = max(4, 200 // max(h, w))
        rgb_big = cv2.resize(rgb, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        vis_big = cv2.resize(visible_mask, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        pred_big = cv2.resize(pred_mask, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

        # Create visualization
        canvas = rgb_big.copy()

        # Get contours
        vis_cnt, _ = cv2.findContours(vis_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pred_cnt, _ = cv2.findContours(pred_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw predicted as dotted magenta
        for cnt in pred_cnt:
            if cv2.contourArea(cnt) > 50:
                draw_dotted_contour(canvas, cnt, (255, 0, 255), thickness=2, gap=8)

        # Draw visible as solid green
        for cnt in vis_cnt:
            if cv2.contourArea(cnt) > 50:
                cv2.drawContours(canvas, [cnt], -1, (0, 200, 0), 2)

        # Add stats
        vis_px = (visible_mask > 127).sum()
        pred_px = (pred_mask > 127).sum()
        diff = pred_px - vis_px
        pct = (diff / max(vis_px, 1)) * 100

        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, f"Visible: {vis_px}px", (5, 20), font, 0.5, (0, 100, 0), 1)
        cv2.putText(canvas, f"Predicted: {pred_px}px (+{diff}, {pct:.0f}%)", (5, 40), font, 0.5, (150, 0, 150), 1)

        all_results.append(canvas)

        out_path = out_dir / f"{img_path.stem}_result.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    # Grid
    if all_results:
        max_h = max(r.shape[0] for r in all_results)
        max_w = max(r.shape[1] for r in all_results)

        padded = []
        for r in all_results:
            p = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
            p[:r.shape[0], :r.shape[1]] = r
            padded.append(p)

        # 2 columns
        rows = []
        for i in range(0, len(padded), 2):
            row = [padded[i]]
            if i + 1 < len(padded):
                row.append(padded[i + 1])
            else:
                row.append(np.full((max_h, max_w, 3), 255, dtype=np.uint8))
            rows.append(np.concatenate(row, axis=1))

        grid = np.concatenate(rows, axis=0)
        grid_path = out_dir / "grid.png"
        cv2.imwrite(str(grid_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"\nSaved: {grid_path}")

    print(f"Done! Results in: {out_dir}")


if __name__ == "__main__":
    main()
