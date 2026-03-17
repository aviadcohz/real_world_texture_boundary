#!/usr/bin/env python3
"""Generate pipeline overview figures for the README.

Creates two composite figures:
1. Pipeline overview: source image → segmentation mask → VLM analysis → binary masks → boundary overlay
2. Crop extraction: full overlay → crop extraction → refined crop → SR-enhanced crop

Usage:
    python -m texture_boundary_Architexture.scripts.generate_pipeline_figure \
        --exp-dir /path/to/experiment \
        --output-dir docs/figures
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int = 16):
    """Load a readable font, falling back to default."""
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _add_label(img: Image.Image, text: str, position: str = "top") -> Image.Image:
    """Add a centered label bar above or below an image."""
    w, h = img.size
    font = _load_font(max(14, w // 20))
    bar_h = max(28, h // 12)

    bar = Image.new("RGB", (w, bar_h), (30, 30, 30))
    draw = ImageDraw.Draw(bar)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, (bar_h - th) // 2), text, fill=(240, 240, 240), font=font)

    if position == "top":
        canvas = Image.new("RGB", (w, h + bar_h))
        canvas.paste(bar, (0, 0))
        canvas.paste(img, (0, bar_h))
    else:
        canvas = Image.new("RGB", (w, h + bar_h))
        canvas.paste(img, (0, 0))
        canvas.paste(bar, (0, h))

    return canvas


def _resize_to_height(img: Image.Image, target_h: int) -> Image.Image:
    """Resize image to target height, preserving aspect ratio."""
    w, h = img.size
    new_w = int(w * target_h / h)
    return img.resize((new_w, target_h), Image.Resampling.LANCZOS)


def _draw_arrow(canvas: Image.Image, x: int, y: int, length: int = 40):
    """Draw a right-pointing arrow on the canvas."""
    draw = ImageDraw.Draw(canvas)
    y_mid = y
    # Arrow shaft
    draw.line([(x, y_mid), (x + length, y_mid)], fill=(200, 200, 200), width=3)
    # Arrow head
    draw.polygon([
        (x + length, y_mid),
        (x + length - 10, y_mid - 7),
        (x + length - 10, y_mid + 7),
    ], fill=(200, 200, 200))


def _create_mask_overlay(image_path: str, mask_a_path: str, mask_b_path: str,
                         alpha: float = 0.4) -> Image.Image:
    """Create a red/blue mask overlay on the source image."""
    img = np.array(Image.open(image_path).convert("RGB")).astype(np.float32)
    mask_a = np.array(Image.open(mask_a_path).convert("L")) > 127
    mask_b = np.array(Image.open(mask_b_path).convert("L")) > 127

    overlay = img.copy()
    if mask_a.any():
        overlay[mask_a] = (1 - alpha) * overlay[mask_a] + alpha * np.array([255, 50, 50], dtype=np.float32)
    if mask_b.any():
        overlay[mask_b] = (1 - alpha) * overlay[mask_b] + alpha * np.array([50, 100, 255], dtype=np.float32)

    # Boundary
    edge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask_a.astype(np.uint8), edge_k)
    boundary = cv2.dilate(mask_a.astype(np.uint8) - eroded, edge_k)
    overlay[boundary > 0] = [255, 255, 0]

    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def generate_pipeline_figure(exp_dir: Path, output_dir: Path, sample_idx: int = 0):
    """Generate the main pipeline overview figure.

    Shows: Source Image → Segmentation Mask → Binary Masks (A/B) → Boundary Overlay
    """
    with open(exp_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Find a good sample (one with crops)
    samples_with_crops = [e for e in metadata if len(e.get("crops", [])) >= 2]
    if not samples_with_crops:
        samples_with_crops = metadata
    entry = samples_with_crops[min(sample_idx, len(samples_with_crops) - 1)]

    # Load images
    src_img = Image.open(entry["image_path"]).convert("RGB")
    mask_img = Image.open(entry["mask_path"]).convert("RGB")
    mask_a = Image.open(entry["mask_a_path"]).convert("L")
    mask_b = Image.open(entry["mask_b_path"]).convert("L")

    # Create overlay
    overlay = _create_mask_overlay(entry["image_path"], entry["mask_a_path"], entry["mask_b_path"])

    # Normalize heights
    target_h = 300
    panels = [
        _add_label(_resize_to_height(src_img, target_h), "Source Image"),
        _add_label(_resize_to_height(mask_img, target_h), "Segmentation Mask"),
        _add_label(_resize_to_height(mask_a.convert("RGB"), target_h), f"Mask A: {entry['texture_a'][:25]}"),
        _add_label(_resize_to_height(mask_b.convert("RGB"), target_h), f"Mask B: {entry['texture_b'][:25]}"),
        _add_label(_resize_to_height(overlay, target_h), "Boundary Overlay"),
    ]

    # Compute layout
    arrow_w = 50
    total_w = sum(p.width for p in panels) + arrow_w * (len(panels) - 1)
    total_h = panels[0].height

    canvas = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    x = 0
    for i, panel in enumerate(panels):
        canvas.paste(panel, (x, 0))
        x += panel.width
        if i < len(panels) - 1:
            _draw_arrow(canvas, x + 5, total_h // 2, arrow_w - 10)
            x += arrow_w

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "pipeline_overview.jpg"
    canvas.save(str(out_path), quality=95)
    print(f"Saved pipeline overview: {out_path}")
    return out_path


def generate_crop_figure(exp_dir: Path, output_dir: Path, max_examples: int = 3,
                         sample_indices: list = None):
    """Generate crop extraction examples figure.

    Shows multiple rows, each: Full Image with crop box → Cropped region → Overlay

    Args:
        sample_indices: Specific metadata indices to use (for curated diversity).
            If None, picks the first entries with crops.
    """
    with open(exp_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Find entries with crops
    examples = []
    if sample_indices:
        for idx in sample_indices:
            if idx < len(metadata):
                entry = metadata[idx]
                crops = entry.get("crops", [])
                if crops:
                    examples.append((entry, crops[0]))
    else:
        for entry in metadata:
            crops = entry.get("crops", [])
            if crops:
                examples.append((entry, crops[0]))
            if len(examples) >= max_examples:
                break

    if not examples:
        print("No crops found — skipping crop figure")
        return None

    rows = []
    target_h = 220

    for entry, crop in examples:
        src_img = Image.open(entry["image_path"]).convert("RGB")
        y1, x1, y2, x2 = crop["box"]

        # Draw crop box on source image
        src_with_box = src_img.copy()
        draw = ImageDraw.Draw(src_with_box)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)

        # Load crop
        crop_img = Image.open(crop["crop_image_path"]).convert("RGB")

        # Crop overlay
        crop_overlay = _create_mask_overlay(
            crop["crop_image_path"], crop["crop_mask_a_path"], crop["crop_mask_b_path"]
        )

        # Refined if available
        panels = [
            _add_label(_resize_to_height(src_with_box, target_h), "Source + Crop Box"),
            _add_label(_resize_to_height(crop_img, target_h), f"Crop ({x2-x1}x{y2-y1})"),
            _add_label(_resize_to_height(crop_overlay, target_h), "Boundary Overlay"),
        ]

        if "refined_image_path" in crop and Path(crop["refined_image_path"]).exists():
            ref_img = Image.open(crop["refined_image_path"]).convert("RGB")
            ref_overlay = _create_mask_overlay(
                crop["refined_image_path"],
                crop["refined_mask_a_path"],
                crop["refined_mask_b_path"],
            )
            panels.append(_add_label(_resize_to_height(ref_img, target_h), "SR Enhanced"))
            panels.append(_add_label(_resize_to_height(ref_overlay, target_h), "Refined Boundary"))

        # Compose row
        arrow_w = 40
        row_w = sum(p.width for p in panels) + arrow_w * (len(panels) - 1)
        row_h = panels[0].height
        row = Image.new("RGB", (row_w, row_h), (20, 20, 20))
        x = 0
        for i, panel in enumerate(panels):
            row.paste(panel, (x, 0))
            x += panel.width
            if i < len(panels) - 1:
                _draw_arrow(row, x + 3, row_h // 2, arrow_w - 6)
                x += arrow_w
        rows.append(row)

    # Stack rows vertically
    max_w = max(r.width for r in rows)
    gap = 10
    total_h = sum(r.height for r in rows) + gap * (len(rows) - 1)
    canvas = Image.new("RGB", (max_w, total_h), (20, 20, 20))
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height + gap

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "crop_examples.jpg"
    canvas.save(str(out_path), quality=95)
    print(f"Saved crop examples: {out_path}")
    return out_path


def generate_results_grid(exp_dir: Path, output_dir: Path, max_samples: int = 8,
                          grid_indices: list = None):
    """Generate a grid of overlay visualization results.

    Args:
        grid_indices: Specific metadata indices to use for curated diversity.
            If None, picks the first max_samples visualizations.
    """
    viz_dir = exp_dir / "visualizations"
    if not viz_dir.exists():
        print("No visualizations directory — skipping results grid")
        return None

    if grid_indices:
        # Map metadata indices to visualization filenames
        with open(exp_dir / "metadata.json") as f:
            metadata = json.load(f)
        viz_files = []
        for idx in grid_indices:
            if idx < len(metadata):
                crop_name = metadata[idx]["crop_name"]
                vf = viz_dir / f"{crop_name}.jpg"
                if vf.exists():
                    viz_files.append(vf)
    else:
        viz_files = sorted(viz_dir.glob("*.jpg"))[:max_samples]

    if not viz_files:
        return None

    target_h = 250
    images = [_resize_to_height(Image.open(f).convert("RGB"), target_h) for f in viz_files]

    # Grid layout: 2 rows
    cols = (len(images) + 1) // 2
    gap = 6
    col_w = max(img.width for img in images)
    grid_w = cols * (col_w + gap) - gap
    grid_h = 2 * (target_h + gap) - gap

    canvas = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        x = col * (col_w + gap) + (col_w - img.width) // 2
        y = row * (target_h + gap)
        canvas.paste(img, (x, y))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "results_grid.jpg"
    canvas.save(str(out_path), quality=95)
    print(f"Saved results grid: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate pipeline figures for README")
    parser.add_argument("--exp-dir", required=True, help="Path to experiment output directory")
    parser.add_argument("--output-dir", default="docs/figures", help="Output directory for figures")
    parser.add_argument("--sample-idx", type=int, default=0, help="Index of sample for pipeline overview")
    parser.add_argument("--max-crop-examples", type=int, default=3)
    parser.add_argument("--crop-indices", type=int, nargs="+", default=None,
                        help="Specific metadata indices for crop examples (for curated diversity)")
    parser.add_argument("--max-grid-samples", type=int, default=8)
    parser.add_argument("--grid-indices", type=int, nargs="+", default=None,
                        help="Specific metadata indices for results grid (for curated diversity)")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    output_dir = Path(args.output_dir)

    print(f"Generating figures from: {exp_dir}")
    print(f"Output: {output_dir}\n")

    generate_pipeline_figure(exp_dir, output_dir, args.sample_idx)
    generate_crop_figure(exp_dir, output_dir, args.max_crop_examples, args.crop_indices)
    generate_results_grid(exp_dir, output_dir, args.max_grid_samples, args.grid_indices)

    print(f"\nDone! Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
