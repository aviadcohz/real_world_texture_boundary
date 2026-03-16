#!/usr/bin/env python3
"""
Compare 256x256 vs 512x512 upscaling + smooth mask boundaries.

Addresses:
1. Does 256x256 preserve original content better than 512?
2. Smooth mask boundaries using polynomial/spline approximation of contours.
"""

import sys
sys.path.insert(0, "/home/aviad/real_world_texture_boundary")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path
from texture_boundary_Architexture.core.texture_refiner_pipeline import (
    TextureRefinerPipeline, SRBackend, StandardizeMode,
)

CROPS_DIR = Path("/datasets/ade20k/Detecture_dataset/crops")
TEST_CROPS = [
    "training_ADE_train_00005346_t2_crop0",   # 33x33
    "training_ADE_train_00005842_t0_crop4",    # 44x44
    "training_ADE_train_00003145_t1_crop0",    # 72x54
    "training_ADE_train_00009295_t0_crop1",    # 63x84
    "training_ADE_train_00009493_t2_crop4",    # 104x78
]
OUTPUT_DIR = Path("/home/aviad/real_world_texture_boundary/texture_boundary_Architexture/sr_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def smooth_mask_contour(mask: np.ndarray, epsilon_frac: float = 0.005,
                        gaussian_ksize: int = 0) -> np.ndarray:
    """Smooth binary mask boundaries using contour polygon approximation.

    Method:
    1. Find contours of the binary mask.
    2. Approximate each contour with cv2.approxPolyDP (Douglas-Peucker algorithm)
       — this fits a polygon with fewer vertices, removing staircase artifacts.
    3. Optionally apply a small Gaussian blur + re-threshold for sub-pixel smoothing.
    4. Re-draw the smoothed contours as a filled mask.

    Args:
        mask: Binary mask (H, W), uint8, values in {0, 255}.
        epsilon_frac: Fraction of contour perimeter for polygon approximation.
                      Lower = more faithful to original, higher = smoother.
                      0.005 is a good balance.
        gaussian_ksize: If > 0, apply Gaussian blur of this kernel size after
                        polygon fill, then re-threshold. Use odd values (3, 5, 7).
                        0 = skip (polygon-only smoothing).

    Returns:
        Smoothed binary mask, same shape, uint8 {0, 255}.
    """
    # Find contours
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return mask

    # Approximate each contour with Douglas-Peucker
    smoothed = np.zeros_like(mask)
    for i, cnt in enumerate(contours):
        epsilon = epsilon_frac * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # Check if this is a hole (inner contour) via hierarchy
        is_hole = hierarchy[0][i][3] >= 0 if hierarchy is not None else False
        if is_hole:
            cv2.drawContours(smoothed, [approx], 0, 0, -1)
        else:
            cv2.drawContours(smoothed, [approx], 0, 255, -1)

    # Optional: Gaussian blur + re-threshold for sub-pixel smoothing
    if gaussian_ksize > 0:
        blurred = cv2.GaussianBlur(smoothed.astype(np.float32),
                                    (gaussian_ksize, gaussian_ksize), 0)
        smoothed = (blurred > 127).astype(np.uint8) * 255

    return smoothed


def smooth_mask_spline(mask: np.ndarray, n_points: int = 100,
                       smoothing_factor: float = 5.0) -> np.ndarray:
    """Smooth binary mask boundaries using B-spline interpolation.

    This gives smoother, more natural curves than polygon approximation.

    Args:
        mask: Binary mask (H, W), uint8, values in {0, 255}.
        n_points: Number of points to sample along each contour for spline fitting.
        smoothing_factor: Spline smoothing parameter. Higher = smoother.

    Returns:
        Smoothed binary mask, same shape, uint8 {0, 255}.
    """
    from scipy.interpolate import splprep, splev

    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return mask

    smoothed = np.zeros_like(mask)

    for i, cnt in enumerate(contours):
        if len(cnt) < 10:  # too small for spline
            cv2.drawContours(smoothed, [cnt], 0, 255, -1)
            continue

        # Extract x, y from contour
        pts = cnt.squeeze()
        if pts.ndim != 2:
            continue
        x, y = pts[:, 0].astype(float), pts[:, 1].astype(float)

        try:
            # Fit periodic B-spline
            tck, u = splprep([x, y], s=smoothing_factor * len(x), per=True, k=3)
            # Evaluate at more points for smooth curve
            u_new = np.linspace(0, 1, max(n_points, len(x)))
            x_new, y_new = splev(u_new, tck)

            # Create smoothed contour
            smooth_pts = np.column_stack([x_new, y_new]).astype(np.int32)
            smooth_pts = smooth_pts.reshape(-1, 1, 2)

            is_hole = hierarchy[0][i][3] >= 0 if hierarchy is not None else False
            if is_hole:
                cv2.drawContours(smoothed, [smooth_pts], 0, 0, -1)
            else:
                cv2.drawContours(smoothed, [smooth_pts], 0, 255, -1)
        except (ValueError, TypeError):
            # Fallback to original contour
            cv2.drawContours(smoothed, [cnt], 0, 255, -1)

    return smoothed


def create_mask_overlay(image: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray,
                        alpha=0.35) -> np.ndarray:
    """Overlay masks on image: red=A, blue=B."""
    overlay = image.copy()
    overlay[mask_a > 127, 0] = np.clip(overlay[mask_a > 127, 0].astype(int) + 100, 0, 255).astype(np.uint8)
    overlay[mask_a > 127, 2] = (overlay[mask_a > 127, 2] * 0.5).astype(np.uint8)
    overlay[mask_b > 127, 2] = np.clip(overlay[mask_b > 127, 2].astype(int) + 100, 0, 255).astype(np.uint8)
    overlay[mask_b > 127, 0] = (overlay[mask_b > 127, 0] * 0.5).astype(np.uint8)
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)


def create_boundary_zoom(image: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray,
                         zoom_size: int = 200, output_size: int = 512) -> np.ndarray:
    """Extract and zoom into the boundary region between masks."""
    # Find boundary region
    kernel = np.ones((5, 5), np.uint8)
    dilated_a = cv2.dilate(mask_a, kernel, iterations=2)
    dilated_b = cv2.dilate(mask_b, kernel, iterations=2)
    boundary = cv2.bitwise_and(dilated_a, dilated_b)

    # Find centroid of boundary
    moments = cv2.moments(boundary)
    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cy, cx = mask_a.shape[0] // 2, mask_a.shape[1] // 2

    h, w = image.shape[:2]
    half = zoom_size // 2
    y1 = max(0, cy - half)
    x1 = max(0, cx - half)
    y2 = min(h, y1 + zoom_size)
    x2 = min(w, x1 + zoom_size)

    crop_img = image[y1:y2, x1:x2]
    crop_overlay = create_mask_overlay(
        crop_img, mask_a[y1:y2, x1:x2], mask_b[y1:y2, x1:x2], alpha=0.4
    )
    # Draw boundary line
    boundary_crop = boundary[y1:y2, x1:x2]
    crop_overlay[boundary_crop > 0] = [0, 255, 0]  # green boundary

    # Resize to output_size for visibility
    crop_overlay = cv2.resize(crop_overlay, (output_size, output_size),
                               interpolation=cv2.INTER_NEAREST)
    return crop_overlay


def add_label(image: np.ndarray, text: str, font_size: int = 16) -> np.ndarray:
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([0, 0, tw + 10, th + 8], fill=(0, 0, 0))
    draw.text((5, 2), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def main():
    import torch

    # Load test data
    test_data = []
    for name in TEST_CROPS:
        img = Image.open(CROPS_DIR / "images" / f"{name}.jpg").convert("RGB")
        ma = np.array(Image.open(CROPS_DIR / "masks_texture" / f"{name}_mask_a.png").convert("L"))
        mb = np.array(Image.open(CROPS_DIR / "masks_texture" / f"{name}_mask_b.png").convert("L"))
        test_data.append((name, img, ma, mb))
        print(f"Loaded {name}: {img.size}")

    # ========== PART 1: 256 vs 512 comparison ==========
    print("\n=== Part 1: 256x256 vs 512x512 (x2plus) ===")

    results_256 = []
    results_512 = []

    pipe_256 = TextureRefinerPipeline(
        min_resolution=256, model_name="RealESRGAN_x2plus",
        standardize=StandardizeMode.CENTER_CROP, device="cuda",
    )
    pipe_512 = TextureRefinerPipeline(
        min_resolution=512, model_name="RealESRGAN_x2plus",
        standardize=StandardizeMode.CENTER_CROP, device="cuda",
    )

    for name, img, ma, mb in test_data:
        r256 = pipe_256.process_crop(img, ma, mb)
        r512 = pipe_512.process_crop(img, ma, mb)
        results_256.append(r256)
        results_512.append(r512)
        print(f"  {name}: {img.size} -> 256: scale={r256['scale_factor']}x SR={r256['sr_size']} | "
              f"512: scale={r512['scale_factor']}x SR={r512['sr_size']}")

    del pipe_256, pipe_512
    torch.cuda.empty_cache()

    # Build 256 vs 512 comparison
    cell = 512
    pad = 4
    n = len(test_data)
    # Columns: Original(bicubic to 512) | 256->resized to 512 | 512
    grid_w = 3 * (cell + pad) + pad
    grid_h = n * (cell + pad) + pad
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40

    for row, (name, img, ma, mb) in enumerate(test_data):
        y = row * (cell + pad) + pad
        w_orig, h_orig = img.size

        # Col 0: Original bicubic
        orig = np.array(img.resize((cell, cell), Image.BICUBIC))
        x = pad
        grid[y:y+cell, x:x+cell] = add_label(orig, f"Bicubic {w_orig}x{h_orig}")

        # Col 1: 256x256 result (resize to 512 for display)
        r256 = results_256[row]
        img256 = np.array(r256["image"].resize((cell, cell), Image.LANCZOS))
        x = cell + 2 * pad
        grid[y:y+cell, x:x+cell] = add_label(
            img256, f"SR->256 (scale={r256['scale_factor']}x)")

        # Col 2: 512x512 result
        r512 = results_512[row]
        img512 = np.array(r512["image"])
        x = 2 * (cell + pad) + pad
        grid[y:y+cell, x:x+cell] = add_label(
            img512, f"SR->512 (scale={r512['scale_factor']}x)")

    path_256vs512 = OUTPUT_DIR / "comparison_256_vs_512.jpg"
    Image.fromarray(grid).save(str(path_256vs512), quality=95)
    print(f"\nSaved: {path_256vs512}")

    # ========== PART 2: Mask smoothing comparison ==========
    print("\n=== Part 2: Mask boundary smoothing ===")

    # Use 512 results for smoothing demo
    # Columns: Raw NN masks | Polygon smoothed | Spline smoothed | Polygon+Gaussian
    # Two rows per crop: full overlay + boundary zoom

    methods = [
        ("Nearest-Neighbor\n(raw)", lambda m: m),
        ("Polygon\n(eps=0.003)", lambda m: smooth_mask_contour(m, epsilon_frac=0.003)),
        ("Polygon\n(eps=0.008)", lambda m: smooth_mask_contour(m, epsilon_frac=0.008)),
        ("Polygon+Gauss(5)\n(eps=0.005)", lambda m: smooth_mask_contour(m, epsilon_frac=0.005, gaussian_ksize=5)),
        ("B-Spline\n(s=3.0)", lambda m: smooth_mask_spline(m, smoothing_factor=3.0)),
    ]

    n_methods = len(methods)
    grid_w = n_methods * (cell + pad) + pad
    grid_h = n * 2 * (cell + pad) + pad  # 2 rows per crop (overlay + zoom)
    grid_smooth = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40

    for crop_idx in range(n):
        r512 = results_512[crop_idx]
        sr_rgb = np.array(r512["image"])
        raw_ma = r512["mask_a"]
        raw_mb = r512["mask_b"]
        name = TEST_CROPS[crop_idx]
        w_orig, h_orig = test_data[crop_idx][1].size

        for col, (method_name, smooth_fn) in enumerate(methods):
            ma_s = smooth_fn(raw_ma.copy())
            mb_s = smooth_fn(raw_mb.copy())

            x = col * (cell + pad) + pad

            # Row 1: Full overlay
            y1 = crop_idx * 2 * (cell + pad) + pad
            overlay = create_mask_overlay(sr_rgb, ma_s, mb_s, alpha=0.4)
            label = f"{method_name.split(chr(10))[0]} ({w_orig}x{h_orig})"
            grid_smooth[y1:y1+cell, x:x+cell] = add_label(overlay, label)

            # Row 2: Boundary zoom
            y2 = y1 + cell + pad
            zoom = create_boundary_zoom(sr_rgb, ma_s, mb_s, zoom_size=180, output_size=cell)
            label2 = method_name.split('\n')[-1] if '\n' in method_name else "zoom"
            grid_smooth[y2:y2+cell, x:x+cell] = add_label(zoom, f"Zoom: {label2}")

    path_smooth = OUTPUT_DIR / "comparison_mask_smoothing.jpg"
    Image.fromarray(grid_smooth).save(str(path_smooth), quality=95)
    print(f"Saved: {path_smooth}")

    # Also save individual per-crop smoothing strips (easier to view)
    for crop_idx in range(n):
        r512 = results_512[crop_idx]
        sr_rgb = np.array(r512["image"])
        raw_ma = r512["mask_a"]
        raw_mb = r512["mask_b"]
        name = TEST_CROPS[crop_idx]

        strip_w = n_methods * (cell + pad) + pad
        strip_h = 2 * (cell + pad) + pad
        strip = np.ones((strip_h, strip_w, 3), dtype=np.uint8) * 40

        for col, (method_name, smooth_fn) in enumerate(methods):
            ma_s = smooth_fn(raw_ma.copy())
            mb_s = smooth_fn(raw_mb.copy())
            x = col * (cell + pad) + pad

            overlay = create_mask_overlay(sr_rgb, ma_s, mb_s, alpha=0.4)
            strip[pad:pad+cell, x:x+cell] = add_label(overlay, method_name.replace('\n', ' '))

            zoom = create_boundary_zoom(sr_rgb, ma_s, mb_s, zoom_size=180, output_size=cell)
            strip[pad+cell+pad:pad+2*cell+pad, x:x+cell] = add_label(zoom, "Boundary zoom")

        strip_path = OUTPUT_DIR / f"smooth_{name}.jpg"
        Image.fromarray(strip).save(str(strip_path), quality=95)

    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
