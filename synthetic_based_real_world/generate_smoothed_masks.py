#!/usr/bin/env python3
"""
Generate smoothed SA2VA boundary masks for ControlNet training data.

For each image in training_pairs.json:
1. Run SA2VA segmentation for both textures
2. Clean masks (morphological + complementary enforcement)
3. Smooth mask boundaries (periodic B-spline)
4. Extract boundary between smoothed masks
5. Save boundary mask to output directory

Then update training_pairs.json to point to the new masks.

Usage:
    python generate_smoothed_masks.py DATASET_ROOT
    python generate_smoothed_masks.py DATASET_ROOT --output-dir masks_for_train
    python generate_smoothed_masks.py DATASET_ROOT --dry-run
    python generate_smoothed_masks.py DATASET_ROOT --skip-json-update

The script is resumable: if interrupted, just re-run — it skips images
whose output masks already exist.

DATASET_ROOT must contain:
    images/              — training images
    masks/               — original boundary masks
    training_pairs.json  — mapping file with {image, conditioning_image, text}
"""

import argparse
import json
import logging
import shutil
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

# ── Path setup for imports ───────────────────────────────────────────────────
PIPELINE_ROOT = Path("/home/aviad/real_world_texture_boundary/texture_boundary_pipeline")
SCRIPT_DIR = Path("/home/aviad/real_world_texture_boundary/sytatic_based_real_world")
sys.path.insert(0, str(PIPELINE_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from models.sa2va_vlm import Sa2VAModel
from core.sa2va_boundaries import extract_morphological_boundary
from sa2va_smooth_masks import (
    clean_mask,
    _ensure_complementary,
    smooth_mask_boundary,
    parse_description,
)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate smoothed SA2VA boundary masks for training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        nargs="?",
        default=Path("/datasets/google_landmarks_v2_scale/real_world_texture_boundary/results/controlnet_data_for_training"),
        help="Root directory containing images/, masks/, training_pairs.json "
             "(default: current training dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="refinment_masks",
        help="Name of output subdirectory for generated masks (default: masks_for_train)",
    )
    parser.add_argument(
        "--boundary-thickness",
        type=int,
        default=3,
        help="Thickness for morphological boundary extraction (default: 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup and count remaining work without processing",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Log progress every N images (default: 50)",
    )
    parser.add_argument(
        "--skip-json-update",
        action="store_true",
        help="Generate masks but do not update training_pairs.json",
    )
    parser.add_argument(
        "--overlay-dir",
        type=str,
        default="refinment_overlays",
        help="Subdirectory for overlay visualizations (default: refinment_overlays). "
             "Use --no-overlays to disable.",
    )
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="Skip overlay generation after mask generation",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Name of output JSON file (default: metadata_sa2va_masks.json for metadata.json, "
             "in-place update for training_pairs.json)",
    )
    return parser.parse_args()


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(log_dir):
    logger = logging.getLogger("generate_smoothed_masks")
    logger.setLevel(logging.DEBUG)

    # Console: INFO, compact
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(ch)

    # File: DEBUG, full timestamps
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "generation.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(fh)

    return logger


# ── Dataset validation ───────────────────────────────────────────────────────

def validate_dataset(dataset_root):
    """Validate dataset structure and return parsed training pairs.

    Supports two JSON formats:
      - training_pairs.json: [{"image": "<abs>", "conditioning_image": "<abs>", "text": "..."}]
      - metadata.json:       {"samples": [{"image": "images/0000.png", "mask": "masks/0000.png", "prompt": "..."}]}
    """
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"

    for p, name in [(images_dir, "images/"), (masks_dir, "masks/")]:
        if not p.exists():
            raise FileNotFoundError(f"Required path not found: {p}")

    # Try training_pairs.json first, then metadata.json
    json_path = dataset_root / "training_pairs.json"
    if not json_path.exists():
        json_path = dataset_root / "metadata.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"No training_pairs.json or metadata.json found in {dataset_root}"
        )

    with open(json_path) as f:
        data = json.load(f)

    # Normalize to common format: [{"image": abs, "conditioning_image": abs, "text": str}]
    if isinstance(data, dict) and "samples" in data:
        # metadata.json format — paths are relative to dataset_root
        training_pairs = []
        for s in data["samples"]:
            img_path = Path(s["image"])
            if not img_path.is_absolute():
                img_path = dataset_root / img_path
            mask_path = Path(s.get("mask", s.get("conditioning_image", "")))
            if not mask_path.is_absolute():
                mask_path = dataset_root / mask_path
            training_pairs.append({
                "image": str(img_path),
                "conditioning_image": str(mask_path),
                "text": s.get("prompt", s.get("text", "")),
            })
    elif isinstance(data, list):
        training_pairs = data
    else:
        raise ValueError(f"Unrecognized JSON format in {json_path}")

    if len(training_pairs) == 0:
        raise ValueError(f"No entries found in {json_path}")

    required_keys = {"image", "conditioning_image", "text"}
    missing = required_keys - set(training_pairs[0].keys())
    if missing:
        raise ValueError(f"First entry missing required keys: {missing}")

    return training_pairs


# ── Single-image processing ──────────────────────────────────────────────────

def process_single_image(sa2va_model, image_path, text, output_path,
                         logger, boundary_thickness=3):
    """Process one image through the full SA2VA smooth pipeline.

    Returns dict with status, timing, and diagnostics.
    """
    stem = image_path.stem
    t0 = time.time()

    # Resume: skip if already done
    if output_path.exists() and output_path.stat().st_size > 0:
        return {"status": "skipped", "stem": stem, "time_seconds": 0.0}

    try:
        texture_a, texture_b = parse_description(text)

        # SA2VA segmentation (2 forward passes)
        raw_a = sa2va_model.segment_texture(str(image_path), texture_a)
        raw_b = sa2va_model.segment_texture(str(image_path), texture_b)

        # Clean masks
        clean_a = clean_mask(raw_a)
        clean_b = clean_mask(raw_b)

        # Ensure complementary
        clean_a, clean_b = _ensure_complementary(
            clean_a, clean_b, texture_a, texture_b
        )

        # Smooth boundaries
        smooth_a = smooth_mask_boundary(clean_a)
        smooth_b = smooth_mask_boundary(clean_b)

        # Extract boundary
        boundary = extract_morphological_boundary(
            smooth_a, smooth_b, thickness=boundary_thickness
        )

        # Save
        cv2.imwrite(str(output_path), boundary)

        elapsed = time.time() - t0
        bnd_px = int((boundary > 127).sum())
        logger.debug(f"  {stem}: OK ({bnd_px} bnd px, {elapsed:.1f}s)")

        return {
            "status": "success",
            "stem": stem,
            "time_seconds": elapsed,
            "boundary_pixels": bnd_px,
        }

    except Exception as e:
        elapsed = time.time() - t0

        # CUDA OOM: clear cache and continue
        if "CUDA out of memory" in str(e):
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            logger.warning(f"  {stem}: CUDA OOM — cleared cache")

        logger.warning(f"  {stem}: FAILED — {e}")
        logger.debug(f"  {stem}: traceback:\n{traceback.format_exc()}")

        return {
            "status": "error",
            "stem": stem,
            "time_seconds": elapsed,
            "error_message": str(e),
        }


# ── Dataset processing loop ──────────────────────────────────────────────────

def process_dataset(training_pairs, output_dir, sa2va_model,
                    logger, log_every=50, boundary_thickness=3):
    """Process all images with automatic resume and progress logging."""
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(training_pairs)
    success = 0
    skipped = 0
    errors = 0
    error_details = []
    recent_times = []

    t_start = time.time()

    for i, entry in enumerate(training_pairs):
        image_path = Path(entry["image"])
        stem = image_path.stem
        text = entry["text"]
        output_path = output_dir / f"{stem}.png"

        result = process_single_image(
            sa2va_model, image_path, text, output_path,
            logger, boundary_thickness,
        )

        if result["status"] == "success":
            success += 1
            recent_times.append(result["time_seconds"])
        elif result["status"] == "skipped":
            skipped += 1
        else:
            errors += 1
            error_details.append(result)
            recent_times.append(result.get("time_seconds", 2.0))

        # Keep rolling window for ETA
        if len(recent_times) > 100:
            recent_times.pop(0)

        # Progress logging
        processed = i + 1
        if processed % log_every == 0 or processed == total:
            pct = 100.0 * processed / total
            avg_t = sum(recent_times) / len(recent_times) if recent_times else 2.0
            remaining = total - processed
            eta = timedelta(seconds=int(avg_t * remaining))
            rate = 1.0 / avg_t if avg_t > 0 else 0

            logger.info(
                f"[{processed:>5}/{total}] {pct:5.1f}% | "
                f"OK:{success} Skip:{skipped} Err:{errors} | "
                f"ETA: {eta} | {rate:.2f} img/s"
            )

    elapsed = time.time() - t_start

    return {
        "total": total,
        "success": success,
        "skipped": skipped,
        "errors": errors,
        "error_details": error_details,
        "elapsed_seconds": elapsed,
    }


# ── JSON update ──────────────────────────────────────────────────────────────

def update_training_json(dataset_root, output_dir, logger, output_json_name=None):
    """Write updated JSON with new mask paths.

    For training_pairs.json: backup original, update in-place.
    For metadata.json: write a new file (default: metadata_sa2va_masks.json).
    """
    # Detect source format
    orig_json = dataset_root / "training_pairs.json"
    is_metadata = False
    if not orig_json.exists():
        orig_json = dataset_root / "metadata.json"
        is_metadata = True

    with open(orig_json) as f:
        data = json.load(f)

    if is_metadata:
        # metadata.json format — write a NEW json, don't overwrite original
        out_json = dataset_root / (output_json_name or "metadata_sa2va_masks.json")
        samples = data["samples"]
        updated = 0
        for entry in samples:
            img_path = Path(entry["image"])
            if not img_path.is_absolute():
                img_path = dataset_root / img_path
            stem = img_path.stem
            new_mask = output_dir / f"{stem}.png"
            if new_mask.exists() and new_mask.stat().st_size > 0:
                entry["sa2va_mask"] = str(new_mask)
                updated += 1

        with open(out_json, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Updated {updated} entries with sa2va_mask field")
        logger.info(f"Written to: {out_json}")
        return out_json

    else:
        # training_pairs.json format — backup and update in-place
        bak_path = orig_json.with_suffix(".json.bak")
        if bak_path.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            bak_path = orig_json.with_name(f"training_pairs.json.bak.{ts}")
        shutil.copy2(orig_json, bak_path)
        logger.info(f"Backed up original JSON → {bak_path}")

        pairs = data
        updated = 0
        kept = 0
        for entry in pairs:
            stem = Path(entry["image"]).stem
            new_mask = output_dir / f"{stem}.png"
            if new_mask.exists() and new_mask.stat().st_size > 0:
                entry["conditioning_image"] = str(new_mask)
                updated += 1
            else:
                kept += 1

        with open(orig_json, "w") as f:
            json.dump(pairs, f, indent=2)

        with open(orig_json) as f:
            verify = json.load(f)
        assert len(verify) == len(pairs), "JSON verification failed"

        logger.info(f"Updated {updated} entries, kept {kept} original paths")
        logger.info(f"Written to: {orig_json}")
        return bak_path


# ── Summary ──────────────────────────────────────────────────────────────────

def print_summary(results, logger):
    elapsed = timedelta(seconds=int(results["elapsed_seconds"]))
    logger.info("=" * 70)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total entries:   {results['total']}")
    logger.info(f"  Successful:      {results['success']}")
    logger.info(f"  Skipped (exist): {results['skipped']}")
    logger.info(f"  Errors:          {results['errors']}")
    logger.info(f"  Wall time:       {elapsed}")
    if results["success"] > 0:
        rate = results["success"] / results["elapsed_seconds"]
        logger.info(f"  Processing rate: {rate:.2f} img/s")
    if results["error_details"]:
        logger.info(f"\nFailed images ({len(results['error_details'])}):")
        for err in results["error_details"][:20]:
            logger.info(f"  {err['stem']}: {err.get('error_message', '?')}")
        if len(results["error_details"]) > 20:
            logger.info(f"  ... and {len(results['error_details']) - 20} more")


# ── Overlay generation ────────────────────────────────────────────────────────

def generate_overlays(training_pairs, mask_dir, overlay_dir, logger,
                      color=(0, 255, 0), alpha=0.85, dilate_px=6):
    """Generate overlay images: boundary masks drawn on source images.

    The boundary mask is dilated to make lines thick and clearly visible.
    """
    overlay_dir.mkdir(parents=True, exist_ok=True)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
    )

    done = 0
    skipped = 0
    errors = 0

    for i, entry in enumerate(training_pairs):
        image_path = Path(entry["image"])
        stem = image_path.stem
        mask_path = mask_dir / f"{stem}.png"
        out_path = overlay_dir / f"{stem}.png"

        if out_path.exists():
            skipped += 1
            continue
        if not mask_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            errors += 1
            continue

        # Dilate boundary for visibility
        thick_mask = cv2.dilate((mask > 127).astype(np.uint8), kernel)
        boundary = thick_mask > 0
        out = img.copy()
        if boundary.any():
            out[boundary] = (
                (1 - alpha) * out[boundary]
                + alpha * np.array(color, dtype=np.float64)
            ).astype(np.uint8)
        cv2.imwrite(str(out_path), out)
        done += 1

    logger.info(f"Overlays: {done} created, {skipped} skipped, {errors} errors → {overlay_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = dataset_root / args.output_dir

    # Validate
    training_pairs = validate_dataset(dataset_root)

    # Logging
    logger = setup_logging(output_dir)
    logger.info(f"Dataset root:        {dataset_root}")
    logger.info(f"Output dir:          {output_dir}")
    logger.info(f"Total entries:       {len(training_pairs)}")
    logger.info(f"Boundary thickness:  {args.boundary_thickness}")

    # Count already processed (for resume info)
    existing = sum(
        1 for p in training_pairs
        if (output_dir / f"{Path(p['image']).stem}.png").exists()
    )
    logger.info(f"Already processed:   {existing}/{len(training_pairs)}")
    logger.info(f"Remaining:           {len(training_pairs) - existing}")

    if args.dry_run:
        logger.info("DRY RUN — exiting without processing")
        return

    # Check if all masks already exist — skip SA2VA processing entirely
    all_masks_exist = existing == len(training_pairs)

    if all_masks_exist:
        logger.info("All masks already exist — skipping SA2VA processing")
    else:
        # Load model
        logger.info("Loading SA2VA model...")
        t_load = time.time()
        sa2va_model = Sa2VAModel(device="cuda", lazy_load=True)
        logger.info(f"SA2VA model ready ({time.time() - t_load:.1f}s)")

        # Process
        results = process_dataset(
            training_pairs=training_pairs,
            output_dir=output_dir,
            sa2va_model=sa2va_model,
            logger=logger,
            log_every=args.log_every,
            boundary_thickness=args.boundary_thickness,
        )

        # Unload model
        sa2va_model.unload()
        logger.info("SA2VA model unloaded")

        # Summary
        print_summary(results, logger)

        # Save error manifest
        if results["error_details"]:
            error_path = output_dir / "errors.json"
            with open(error_path, "w") as f:
                json.dump(results["error_details"], f, indent=2)
            logger.info(f"Error manifest: {error_path}")

        # Update JSON
        if not args.skip_json_update and results["success"] > 0:
            update_training_json(dataset_root, output_dir, logger, args.output_json)
        elif args.skip_json_update:
            logger.info("JSON update skipped (--skip-json-update)")
        else:
            logger.warning("JSON update skipped: no successful generations")

    # Check if output JSON already exists — skip update if so
    if all_masks_exist and not args.skip_json_update:
        out_json_name = args.output_json or "metadata_sa2va_masks.json"
        out_json_path = dataset_root / out_json_name
        if out_json_path.exists():
            logger.info(f"Output JSON already exists — skipping: {out_json_path}")
        else:
            update_training_json(dataset_root, output_dir, logger, args.output_json)

    # Generate overlays for visual debugging
    if not args.no_overlays:
        overlay_dir = dataset_root / args.overlay_dir
        logger.info(f"Generating overlays → {overlay_dir}")
        generate_overlays(training_pairs, output_dir, overlay_dir, logger)
    else:
        logger.info("Overlay generation skipped (--no-overlays)")


if __name__ == "__main__":
    main()
