#!/usr/bin/env python3
"""
Clean ellipse completion - pure white background, just leaf + ellipse outline.
"""
import numpy as np
import cv2
from pathlib import Path


def auto_generate_mask(rgb: np.ndarray) -> np.ndarray:
    """Auto-generate mask for leaf on white background."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lower = np.array([20, 25, 25])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest], -1, 255, -1)

    return mask


def fit_ellipse_to_mask(mask: np.ndarray):
    """Fit ellipse to mask contour."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return None
    return cv2.fitEllipse(cnt)


def draw_dotted_ellipse(img, ellipse, color, thickness=2, gap=8):
    """Draw dotted ellipse."""
    (cx, cy), (w, h), angle = ellipse
    n_points = 100
    angles = np.linspace(0, 2 * np.pi, n_points)
    a, b = w / 2, h / 2
    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))

    points = []
    for t in angles:
        x = a * np.cos(t)
        y = b * np.sin(t)
        x_rot = cx + x * cos_a - y * sin_a
        y_rot = cy + x * sin_a + y * cos_a
        points.append((int(x_rot), int(y_rot)))

    for i in range(0, len(points) - 1, 2):
        cv2.line(img, points[i], points[min(i + 1, len(points) - 1)], color, thickness)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=float, default=1.3)
    args = parser.parse_args()

    images_dir = Path(args.images)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg"}
    image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in exts]

    print(f"Processing {len(image_paths)} images...")

    all_results = []

    for img_path in image_paths:
        print(f"{img_path.name}")

        rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        mask = auto_generate_mask(rgb)
        ellipse = fit_ellipse_to_mask(mask)

        if ellipse is None:
            print("  Skipped - no ellipse")
            continue

        (cx, cy), (ew, eh), angle = ellipse

        # Calculate canvas size to fit extended ellipse
        ext_w = ew * args.scale
        ext_h = eh * args.scale

        # Need canvas big enough for the ellipse
        canvas_w = int(max(w, ext_w) + 40)
        canvas_h = int(max(h, ext_h) + 40)

        # Scale for visualization (images are tiny)
        vis_scale = max(3, 150 // max(h, w))
        canvas_w *= vis_scale
        canvas_h *= vis_scale

        # Create pure white canvas
        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

        # Center offset
        off_x = (canvas_w - w * vis_scale) // 2
        off_y = (canvas_h - h * vis_scale) // 2

        # Draw the leaf (filled green shape)
        mask_big = cv2.resize(mask, (w * vis_scale, h * vis_scale), interpolation=cv2.INTER_NEAREST)

        # Place leaf mask as green filled shape on canvas
        leaf_color = (100, 180, 100)  # Light green
        for y in range(mask_big.shape[0]):
            for x in range(mask_big.shape[1]):
                if mask_big[y, x] > 127:
                    cy_pos = off_y + y
                    cx_pos = off_x + x
                    if 0 <= cy_pos < canvas_h and 0 <= cx_pos < canvas_w:
                        canvas[cy_pos, cx_pos] = leaf_color

        # Draw visible boundary (solid dark green)
        cnt_big, _ = cv2.findContours(mask_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnt_big:
            # Offset contours
            for cnt in cnt_big:
                cnt[:, :, 0] += off_x
                cnt[:, :, 1] += off_y
            cv2.drawContours(canvas, cnt_big, -1, (0, 130, 0), 2)

        # Extended ellipse in canvas coordinates
        ellipse_ext = (
            (off_x + cx * vis_scale, off_y + cy * vis_scale),
            (ew * vis_scale * args.scale, eh * vis_scale * args.scale),
            angle
        )

        # Draw extended ellipse (dotted magenta)
        draw_dotted_ellipse(canvas, ellipse_ext, (200, 0, 200), thickness=3, gap=10)

        all_results.append(canvas)

        out_path = out_dir / f"{img_path.stem}_ellipse.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    # Grid
    if all_results:
        max_h = max(r.shape[0] for r in all_results)
        max_w = max(r.shape[1] for r in all_results)

        padded = []
        for r in all_results:
            p = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
            yo = (max_h - r.shape[0]) // 2
            xo = (max_w - r.shape[1]) // 2
            p[yo:yo+r.shape[0], xo:xo+r.shape[1]] = r
            padded.append(p)

        rows = []
        for i in range(0, len(padded), 2):
            row = [padded[i]]
            if i + 1 < len(padded):
                row.append(padded[i + 1])
            else:
                row.append(np.full((max_h, max_w, 3), 255, dtype=np.uint8))
            rows.append(np.concatenate(row, axis=1))

        grid = np.concatenate(rows, axis=0)
        cv2.imwrite(str(out_dir / "grid.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"\nSaved: {out_dir}/grid.png")

    print("Done!")


if __name__ == "__main__":
    main()
