#!/usr/bin/env python3
"""
Ellipse fitting for leaf completion - simple geometric approach.
No ML bullshit, just fit an ellipse to the visible part and extend it.
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
    """Fit an ellipse to the mask contour. Returns ellipse params or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:  # Need at least 5 points for ellipse
        return None

    ellipse = cv2.fitEllipse(cnt)
    return ellipse  # ((cx, cy), (width, height), angle)


def draw_dotted_ellipse(img, ellipse, color, thickness=2, gap=10):
    """Draw a dotted ellipse."""
    (cx, cy), (w, h), angle = ellipse

    # Generate points along ellipse
    n_points = 100
    angles = np.linspace(0, 2 * np.pi, n_points)

    # Ellipse parametric equations
    a, b = w / 2, h / 2
    cos_angle = np.cos(np.radians(angle))
    sin_angle = np.sin(np.radians(angle))

    points = []
    for t in angles:
        x = a * np.cos(t)
        y = b * np.sin(t)
        # Rotate
        x_rot = cx + x * cos_angle - y * sin_angle
        y_rot = cy + x * sin_angle + y * cos_angle
        points.append((int(x_rot), int(y_rot)))

    # Draw dotted
    for i in range(0, len(points) - 1, 2):
        cv2.line(img, points[i], points[min(i + 1, len(points) - 1)], color, thickness)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ellipse fitting for leaf completion")
    parser.add_argument("--images", required=True, help="Input images folder")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--scale", type=float, default=1.2, help="How much to scale ellipse (1.0 = fit exactly, 1.2 = 20% bigger)")
    args = parser.parse_args()

    images_dir = Path(args.images)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg"}
    image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in exts]

    print(f"Processing {len(image_paths)} images with ellipse fitting (scale={args.scale})...")

    all_results = []

    for img_path in image_paths:
        print(f"\n{img_path.name}:")

        rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        mask = auto_generate_mask(rgb)
        ellipse = fit_ellipse_to_mask(mask)

        if ellipse is None:
            print("  Could not fit ellipse")
            continue

        (cx, cy), (ew, eh), angle = ellipse
        print(f"  Ellipse: center=({cx:.0f},{cy:.0f}), size=({ew:.0f}x{eh:.0f}), angle={angle:.0f}°")

        # Scale up for visualization
        vis_scale = max(4, 200 // max(h, w))
        rgb_big = cv2.resize(rgb, (w * vis_scale, h * vis_scale), interpolation=cv2.INTER_NEAREST)
        mask_big = cv2.resize(mask, (w * vis_scale, h * vis_scale), interpolation=cv2.INTER_NEAREST)

        # Scale ellipse params
        ellipse_scaled = (
            (cx * vis_scale, cy * vis_scale),
            (ew * vis_scale, eh * vis_scale),
            angle
        )

        # Extended ellipse (user-defined scale factor)
        ellipse_extended = (
            (cx * vis_scale, cy * vis_scale),
            (ew * vis_scale * args.scale, eh * vis_scale * args.scale),
            angle
        )

        # Draw on canvas
        canvas = rgb_big.copy()

        # Draw visible contour (solid green)
        cnt_big, _ = cv2.findContours(mask_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnt_big:
            cv2.drawContours(canvas, cnt_big, -1, (0, 200, 0), 2)

        # Draw fitted ellipse (dotted cyan) - exact fit
        draw_dotted_ellipse(canvas, ellipse_scaled, (0, 200, 200), thickness=2, gap=8)

        # Draw extended ellipse (dotted magenta) - predicted complete shape
        draw_dotted_ellipse(canvas, ellipse_extended, (255, 0, 255), thickness=3, gap=6)

        # Legend
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_h = 18
        cv2.putText(canvas, "Green: Visible", (5, line_h), font, 0.5, (0, 150, 0), 1)
        cv2.putText(canvas, "Cyan: Fitted ellipse", (5, line_h * 2), font, 0.5, (0, 150, 150), 1)
        cv2.putText(canvas, f"Magenta: Extended ({args.scale}x)", (5, line_h * 3), font, 0.5, (200, 0, 200), 1)

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
        grid_path = out_dir / "ellipse_grid.png"
        cv2.imwrite(str(grid_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"\nSaved grid: {grid_path}")

    print(f"\nDone! Results in: {out_dir}")


if __name__ == "__main__":
    main()
