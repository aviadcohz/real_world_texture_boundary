#!/usr/bin/env python3
"""Batch-process all crops through TextureRefinerPipeline.

Reads crops from /datasets/ade20k/Detecture_dataset/crops/,
runs Real-ESRGAN x2plus upscaling + mask smoothing, and saves
results to an output directory with updated metadata.

Usage:
    python scripts/batch_refine.py                          # defaults
    python scripts/batch_refine.py --output_dir /tmp/out    # custom output
    python scripts/batch_refine.py --limit 100              # first 100 only
    python scripts/batch_refine.py --resume                 # skip already processed
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.texture_refiner_pipeline import TextureRefinerPipeline


CROPS_DIR = Path("/datasets/ade20k/Detecture_dataset/crops")
DEFAULT_OUTPUT = CROPS_DIR / "refined"


def parse_args():
    p = argparse.ArgumentParser(description="Batch texture refinement")
    p.add_argument("--input_dir", type=str, default=str(CROPS_DIR),
                    help="Root dir containing metadata.json, images/, masks_texture/")
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT),
                    help="Output directory for refined crops")
    p.add_argument("--limit", type=int, default=0,
                    help="Process only first N crops (0 = all)")
    p.add_argument("--resume", action="store_true",
                    help="Skip crops whose output already exists")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tile", type=int, default=0,
                    help="Tile size for SR (0 = no tiling)")
    p.add_argument("--no_smooth", action="store_true",
                    help="Disable mask smoothing")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Load metadata
    meta_path = input_dir / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)
    print(f"Loaded {len(metadata)} entries from {meta_path}")

    if args.limit > 0:
        metadata = metadata[:args.limit]
        print(f"  -> Limited to first {args.limit}")

    # Create output dirs
    out_images = output_dir / "images"
    out_masks = output_dir / "masks_texture"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    # Init pipeline (lazy model load on first crop)
    pipeline = TextureRefinerPipeline(
        device=args.device,
        tile=args.tile,
    )

    # Process
    results_meta = []
    skipped = 0
    failed = 0
    t0 = time.time()

    for i, entry in enumerate(metadata):
        crop_name = entry.get("crop_name", f"crop_{i}")

        # Output paths
        out_img_path = out_images / f"{crop_name}.png"
        out_mask_a_path = out_masks / f"{crop_name}_mask_a.png"
        out_mask_b_path = out_masks / f"{crop_name}_mask_b.png"

        # Resume support
        if args.resume and out_img_path.exists() and out_mask_a_path.exists() and out_mask_b_path.exists():
            skipped += 1
            # Still add to metadata (read sizes from existing files)
            results_meta.append({
                **entry,
                "refined_image": str(out_img_path),
                "refined_mask_a": str(out_mask_a_path),
                "refined_mask_b": str(out_mask_b_path),
                "refined": True,
                "skipped_resume": True,
            })
            continue

        # Load inputs
        try:
            img_path = entry.get("image_path", entry.get("image"))
            image = Image.open(img_path).convert("RGB")
            mask_a = cv2.imread(entry["mask_a_path"], cv2.IMREAD_GRAYSCALE)
            mask_b = cv2.imread(entry["mask_b_path"], cv2.IMREAD_GRAYSCALE)

            if mask_a is None or mask_b is None:
                raise FileNotFoundError(f"Missing mask for {crop_name}")

        except Exception as e:
            print(f"  [{i+1}/{len(metadata)}] SKIP {crop_name}: {e}")
            failed += 1
            results_meta.append({**entry, "refined": False, "error": str(e)})
            continue

        # Run pipeline
        result = pipeline.process_crop(image, mask_a, mask_b)

        # Save outputs
        result["image"].save(str(out_img_path))
        cv2.imwrite(str(out_mask_a_path), result["mask_a"])
        cv2.imwrite(str(out_mask_b_path), result["mask_b"])

        # Update metadata
        results_meta.append({
            **entry,
            "refined_image": str(out_img_path),
            "refined_mask_a": str(out_mask_a_path),
            "refined_mask_b": str(out_mask_b_path),
            "refined": True,
            "scale_factor": result["scale_factor"],
            "input_size": list(result["input_size"]),
            "sr_size": list(result["sr_size"]),
            "output_size": [result["image"].size[1], result["image"].size[0]],
        })

        elapsed = time.time() - t0
        rate = (i + 1 - skipped) / elapsed if elapsed > 0 else 0
        w, h = image.size
        out_w, out_h = result["image"].size
        print(f"  [{i+1}/{len(metadata)}] {crop_name}: "
              f"{w}x{h} -> {out_w}x{out_h} "
              f"(scale={result['scale_factor']}x) "
              f"[{rate:.1f} crops/s]")

    # Save metadata
    out_meta_path = output_dir / "metadata.json"
    with open(out_meta_path, "w") as f:
        json.dump(results_meta, f, indent=2)

    elapsed = time.time() - t0
    processed = len(metadata) - skipped - failed
    print(f"\nDone in {elapsed:.1f}s — "
          f"{processed} processed, {skipped} skipped (resume), {failed} failed")
    print(f"Output: {output_dir}")
    print(f"Metadata: {out_meta_path}")


if __name__ == "__main__":
    main()
