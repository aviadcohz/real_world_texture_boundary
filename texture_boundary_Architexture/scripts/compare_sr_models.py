#!/usr/bin/env python3
"""Visual comparison of SR models on texture boundary crops.

Generates a grid: rows = crops, columns = [Original (resized), Model1, Model2, ...]
with mask overlay on each to verify alignment.
"""

import sys
import os
sys.path.insert(0, "/home/aviad/real_world_texture_boundary")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path

# Test crops: (image_file, mask_a_file, mask_b_file)
CROPS_DIR = Path("/datasets/ade20k/Detecture_dataset/crops")
TEST_CROPS = [
    "training_ADE_train_00005346_t2_crop0",   # 33x33  (tiny)
    "training_ADE_train_00005842_t0_crop4",    # 44x44  (small)
    "training_ADE_train_00003145_t1_crop0",    # 72x54  (medium-small)
    "training_ADE_train_00009295_t0_crop1",    # 63x84  (medium)
    "training_ADE_train_00009493_t2_crop4",    # 104x78 (medium-large)
]

MODELS = [
    ("RealESRGAN_x4plus", "x4plus"),
    ("RealESRGAN_x2plus", "x2plus"),
    ("realesr-general-x4v3", "general-v3"),
]

OUTPUT_DIR = Path("/home/aviad/real_world_texture_boundary/texture_boundary_Architexture/sr_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_mask_overlay(image: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray, alpha=0.35) -> np.ndarray:
    """Overlay masks on image: red=texture_a, blue=texture_b, green=boundary."""
    overlay = image.copy()
    # Red for mask A
    overlay[mask_a > 127, 0] = np.clip(overlay[mask_a > 127, 0].astype(int) + 100, 0, 255).astype(np.uint8)
    overlay[mask_a > 127, 2] = (overlay[mask_a > 127, 2] * 0.5).astype(np.uint8)
    # Blue for mask B
    overlay[mask_b > 127, 2] = np.clip(overlay[mask_b > 127, 2].astype(int) + 100, 0, 255).astype(np.uint8)
    overlay[mask_b > 127, 0] = (overlay[mask_b > 127, 0] * 0.5).astype(np.uint8)
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return result


def add_label(image: np.ndarray, text: str, font_size: int = 18) -> np.ndarray:
    """Add text label at top of image."""
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    # Black background for text
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([0, 0, tw + 10, th + 8], fill=(0, 0, 0))
    draw.text((5, 2), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def main():
    from texture_boundary_Architexture.core.texture_refiner_pipeline import (
        TextureRefinerPipeline, SRBackend, StandardizeMode,
    )

    # Load all test data
    test_data = []
    for name in TEST_CROPS:
        img_path = CROPS_DIR / "images" / f"{name}.jpg"
        ma_path = CROPS_DIR / "masks_texture" / f"{name}_mask_a.png"
        mb_path = CROPS_DIR / "masks_texture" / f"{name}_mask_b.png"
        img = Image.open(img_path).convert("RGB")
        ma = np.array(Image.open(ma_path).convert("L"))
        mb = np.array(Image.open(mb_path).convert("L"))
        test_data.append((name, img, ma, mb))
        print(f"Loaded {name}: {img.size}")

    # Run each model
    model_results = {}  # model_label -> [(result_dict, ...)]

    for model_name, label in MODELS:
        print(f"\n{'='*60}")
        print(f"Testing model: {model_name} ({label})")
        print(f"{'='*60}")

        pipeline = TextureRefinerPipeline(
            min_resolution=512,
            backend=SRBackend.REAL_ESRGAN,
            standardize=StandardizeMode.CENTER_CROP,
            model_name=model_name,
            device="cuda",
            half=True,
        )

        results = []
        for name, img, ma, mb in test_data:
            print(f"  Processing {name} ({img.size})...", end=" ")
            result = pipeline.process_crop(img, ma, mb)
            print(f"scale={result['scale_factor']}x, SR={result['sr_size']}, "
                  f"final={result['image'].size}")
            results.append(result)

            # Save individual results
            out_dir = OUTPUT_DIR / label
            out_dir.mkdir(exist_ok=True)
            result["image"].save(out_dir / f"{name}_sr.jpg", quality=95)
            Image.fromarray(result["mask_a"]).save(out_dir / f"{name}_mask_a.png")
            Image.fromarray(result["mask_b"]).save(out_dir / f"{name}_mask_b.png")

        model_results[label] = results

        # Free GPU memory between models
        del pipeline
        import torch
        torch.cuda.empty_cache()

    # Build comparison grid
    print(f"\nBuilding comparison grid...")
    target = 512
    n_crops = len(test_data)
    n_models = len(MODELS)
    # Columns: Original (bicubic 512x512) | Model1 | Model1+mask | Model2 | Model2+mask | ...
    # Actually: Original | Model1 | Model2 | Model3 | (all with mask overlay below)

    # Two grids: one for SR images, one for SR + mask overlay
    cell_size = target
    pad = 4
    header_h = 30

    # Grid 1: SR comparison (no masks)
    cols = 1 + n_models  # original + models
    grid_w = cols * (cell_size + pad) + pad
    grid_h = n_crops * (cell_size + pad + header_h) + pad
    grid_img = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40  # dark gray bg

    # Grid 2: SR + mask overlay
    grid_mask = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40

    for row, (name, img, ma, mb) in enumerate(test_data):
        y = row * (cell_size + pad + header_h) + pad

        # Original (bicubic upscale to 512x512 for comparison)
        orig_rgb = np.array(img.resize((target, target), Image.BICUBIC))
        orig_ma = cv2.resize(ma, (target, target), interpolation=cv2.INTER_NEAREST)
        orig_mb = cv2.resize(mb, (target, target), interpolation=cv2.INTER_NEAREST)

        w_orig, h_orig = img.size
        labeled = add_label(orig_rgb, f"Original {w_orig}x{h_orig} (bicubic)")
        x = pad
        grid_img[y:y+cell_size, x:x+cell_size] = labeled

        overlay = create_mask_overlay(orig_rgb, orig_ma, orig_mb)
        overlay = add_label(overlay, f"Original {w_orig}x{h_orig} + masks")
        grid_mask[y:y+cell_size, x:x+cell_size] = overlay

        # Model columns
        for col, (_, label) in enumerate(MODELS):
            result = model_results[label][row]
            sr_rgb = np.array(result["image"])
            sr_ma = result["mask_a"]
            sr_mb = result["mask_b"]

            x = (col + 1) * (cell_size + pad) + pad
            labeled = add_label(sr_rgb, f"{label} ({result['scale_factor']}x)")
            grid_img[y:y+cell_size, x:x+cell_size] = labeled

            overlay = create_mask_overlay(sr_rgb, sr_ma, sr_mb)
            overlay = add_label(overlay, f"{label} + masks")
            grid_mask[y:y+cell_size, x:x+cell_size] = overlay

    # Save grids
    sr_path = OUTPUT_DIR / "comparison_sr_only.jpg"
    mask_path = OUTPUT_DIR / "comparison_with_masks.jpg"
    Image.fromarray(grid_img).save(str(sr_path), quality=95)
    Image.fromarray(grid_mask).save(str(mask_path), quality=95)
    print(f"\nSaved: {sr_path}")
    print(f"Saved: {mask_path}")

    # Also save per-crop side-by-side strips for easier zooming
    for row, (name, img, ma, mb) in enumerate(test_data):
        strip_w = cols * (cell_size + pad) + pad
        strip_h = 2 * (cell_size + pad) + pad  # top=SR, bottom=mask overlay
        strip = np.ones((strip_h, strip_w, 3), dtype=np.uint8) * 40

        w_orig, h_orig = img.size

        # Top row: SR images
        orig_rgb = np.array(img.resize((target, target), Image.BICUBIC))
        orig_ma = cv2.resize(ma, (target, target), interpolation=cv2.INTER_NEAREST)
        orig_mb = cv2.resize(mb, (target, target), interpolation=cv2.INTER_NEAREST)

        x = pad
        strip[pad:pad+cell_size, x:x+cell_size] = add_label(orig_rgb, f"Bicubic {w_orig}x{h_orig}")
        strip[pad+cell_size+pad:pad+2*cell_size+pad, x:x+cell_size] = add_label(
            create_mask_overlay(orig_rgb, orig_ma, orig_mb), "Bicubic + masks")

        for col, (_, label) in enumerate(MODELS):
            result = model_results[label][row]
            sr_rgb = np.array(result["image"])
            x = (col + 1) * (cell_size + pad) + pad
            strip[pad:pad+cell_size, x:x+cell_size] = add_label(
                sr_rgb, f"{label} ({result['scale_factor']}x)")
            strip[pad+cell_size+pad:pad+2*cell_size+pad, x:x+cell_size] = add_label(
                create_mask_overlay(sr_rgb, result["mask_a"], result["mask_b"]),
                f"{label} + masks")

        strip_path = OUTPUT_DIR / f"strip_{name}.jpg"
        Image.fromarray(strip).save(str(strip_path), quality=95)

    print(f"\nAll outputs in: {OUTPUT_DIR}")
    print("Files:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        if f.is_file():
            print(f"  {f.name}")


if __name__ == "__main__":
    main()
