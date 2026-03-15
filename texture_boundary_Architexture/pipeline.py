#!/usr/bin/env python3
"""Texture transition extraction pipeline.

Takes pairs of (original image, colored annotation mask) and uses QwenVL to:
1. Identify significant texture transitions
2. Report which RGB mask colors belong to each texture
3. Extract binary masks by color matching
4. Output RWTD/Qwen2SAM-compatible training data with full metadata
"""

import argparse
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from texture_boundary_Architexture.config.prompts import build_texture_transition_prompt
from texture_boundary_Architexture.core.mask_extraction import (
    extract_binary_mask,
    find_dominant_colors,
    format_dominant_colors,
    keep_adjacent_components,
    quantize_mask,
    sample_oracle_points,
    validate_mask_pair,
)
from texture_boundary_Architexture.core.deduplication import is_duplicate_transition
from texture_boundary_Architexture.core.visualization import save_transition_visualizations


def parse_qwen_response(response: str) -> list:
    """Parse Qwen's JSON response with robustness to formatting issues.

    Returns list of valid transition dicts, or empty list on failure.
    """
    text = response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(data, dict) or "transitions" not in data:
        return []

    transitions = data["transitions"]
    if not isinstance(transitions, list):
        return []

    valid = []
    for t in transitions:
        if not isinstance(t, dict):
            continue
        required = ["texture_a", "texture_b", "colors_a", "colors_b"]
        if not all(k in t for k in required):
            continue
        colors_ok = True
        for key in ["colors_a", "colors_b"]:
            if not isinstance(t[key], list) or len(t[key]) == 0:
                colors_ok = False
                break
            for c in t[key]:
                if not isinstance(c, list) or len(c) != 3:
                    colors_ok = False
                    break
                if not all(isinstance(v, (int, float)) and 0 <= v <= 255 for v in c):
                    colors_ok = False
                    break
        if colors_ok:
            valid.append(t)

    return valid


def process_image(
    model,
    image_id: str,
    data_dir: Path,
    exp_dir: Path,
    sample_index: int,
    visualize: bool = True,
) -> list:
    """Process a single image: Qwen analysis + mask extraction + full metadata.

    Args:
        model: QwenVLM model instance.
        image_id: Image filename stem.
        data_dir: Source data directory (images/, masks/).
        exp_dir: Experiment output directory.
        sample_index: Starting index for naming samples.
        visualize: Whether to generate overlay visualizations.

    Returns list of RWTD-format metadata dicts for valid transitions.
    """
    src_image = data_dir / "images" / f"{image_id}.jpg"
    src_mask = data_dir / "masks" / f"{image_id}.jpg"

    if not src_image.exists() or not src_mask.exists():
        print(f"  SKIP {image_id}: missing files")
        return []

    # Load mask and find dominant colors
    mask_array = np.array(Image.open(src_mask).convert("RGB"))
    dominant_colors = find_dominant_colors(mask_array)

    if len(dominant_colors) < 2:
        print(f"  SKIP {image_id}: fewer than 2 dominant colors")
        return []

    # Build prompt with color palette (only show colors > 2% to avoid Qwen using tiny regions)
    prompt_colors = [(c, f) for c, f in dominant_colors if f >= 0.02]
    colors_str = format_dominant_colors(prompt_colors)
    prompt_text = build_texture_transition_prompt(colors_str)

    # Build 2-image message for Qwen
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(src_image)},
                {"type": "image", "image": str(src_mask)},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Call Qwen
    try:
        response = model.custom_generate(messages, max_tokens=1024)
    except Exception as e:
        print(f"  ERROR {image_id}: Qwen failed — {e}")
        return []

    # Parse response
    transitions = parse_qwen_response(response)
    if not transitions:
        print(f"  WARN {image_id}: no valid transitions parsed")
        return []

    # Prepare output directories
    images_dir = exp_dir / "images"
    masks_texture_dir = exp_dir / "masks_texture"
    masks_dir = exp_dir / "masks"
    viz_dir = exp_dir / "visualizations"
    overlays_dir = exp_dir / "overlays"
    for d in [images_dir, masks_texture_dir, masks_dir, viz_dir, overlays_dir]:
        d.mkdir(parents=True, exist_ok=True)

    h, w = mask_array.shape[:2]
    metadata_entries = []
    viz_transitions = []

    # Quantize mask once — each pixel assigned to nearest dominant color
    quantized_labels = quantize_mask(mask_array, dominant_colors)

    # Track masks within this image for dedup
    image_transitions = []

    for i, t in enumerate(transitions):
        mask_a = extract_binary_mask(mask_array, t["colors_a"], dominant_colors,
                                     quantized_labels=quantized_labels)
        mask_b = extract_binary_mask(mask_array, t["colors_b"], dominant_colors,
                                     quantized_labels=quantized_labels)

        # Filter out disconnected components (e.g., sky sharing a color with grass)
        # Keep only components of each mask that are adjacent to the other mask
        mask_a = keep_adjacent_components(mask_a, mask_b)
        mask_b = keep_adjacent_components(mask_b, mask_a)

        valid, reason = validate_mask_pair(mask_a, mask_b)
        if not valid:
            continue

        # Check for duplicates within this image (mask-based)
        if is_duplicate_transition(mask_a, mask_b, image_transitions):
            continue

        # Check for text-based duplicates (same descriptions, different colors)
        desc_pair = (t["texture_a"].lower().strip(), t["texture_b"].lower().strip())
        is_text_dup = False
        for prev_t in metadata_entries:
            prev_pair = (prev_t["texture_a"].lower().strip(), prev_t["texture_b"].lower().strip())
            if desc_pair == prev_pair or desc_pair == (prev_pair[1], prev_pair[0]):
                is_text_dup = True
                break
        if is_text_dup:
            continue

        image_transitions.append((mask_a, mask_b))

        # Sample name
        idx = sample_index + len(metadata_entries)
        crop_name = f"{image_id}_t{i}"

        # Copy source image into experiment
        dst_image = images_dir / f"{crop_name}.jpg"
        if not dst_image.exists():
            shutil.copy2(src_image, dst_image)

        # Copy source colored mask into experiment
        dst_mask = masks_dir / f"{crop_name}.jpg"
        if not dst_mask.exists():
            shutil.copy2(src_mask, dst_mask)

        # Copy overlay if available
        src_overlay = data_dir / "overlays" / f"{image_id}.jpg"
        if src_overlay.exists():
            dst_overlay = overlays_dir / f"{crop_name}.jpg"
            if not dst_overlay.exists():
                shutil.copy2(src_overlay, dst_overlay)

        # Save binary masks as PNG (0/255)
        mask_a_path = masks_texture_dir / f"{crop_name}_mask_a.png"
        mask_b_path = masks_texture_dir / f"{crop_name}_mask_b.png"
        Image.fromarray((mask_a.astype(np.uint8) * 255)).save(str(mask_a_path))
        Image.fromarray((mask_b.astype(np.uint8) * 255)).save(str(mask_b_path))

        # Sample oracle points (4 per mask)
        points_a = sample_oracle_points(mask_a, n_points=4)
        points_b = sample_oracle_points(mask_b, n_points=4)

        # Build RWTD-compatible metadata entry
        entry = {
            "image": str(dst_image.resolve()),
            "image_path": str(dst_image.resolve()),
            "mask_path": str(dst_mask.resolve()),
            "mask_a_path": str(mask_a_path.resolve()),
            "mask_b_path": str(mask_b_path.resolve()),
            "texture_a": t["texture_a"],
            "texture_b": t["texture_b"],
            "description": f"{t['texture_a']} to {t['texture_b']}",
            "vlm_assigned": True,
            "oracle_points": {
                "point_prompt_mask_a": points_a,
                "point_prompt_mask_b": points_b,
            },
            "crop_name": crop_name,
            "coords": [0, 0, w, h],
            "source_image_id": image_id,
        }
        metadata_entries.append(entry)

        viz_transitions.append({
            "mask_a": mask_a,
            "mask_b": mask_b,
            "texture_a": t["texture_a"],
            "texture_b": t["texture_b"],
        })

    # Save visualizations
    if visualize and viz_transitions:
        save_transition_visualizations(str(src_image), viz_transitions, str(viz_dir), image_id)

    return metadata_entries


def run_pipeline(
    data_dir: str = "/home/aviad/detecture_data",
    output_dir: str = None,
    exp_name: str = None,
    max_images: int = None,
    skip_existing: bool = True,
    visualize: bool = True,
    device: str = "cuda",
):
    """Run the full texture transition extraction pipeline.

    Args:
        data_dir: Directory with images/ and masks/ subdirectories.
        output_dir: Base output directory. Each experiment creates a subfolder.
        exp_name: Experiment name. Defaults to timestamp.
        max_images: Process only first N images (for testing).
        skip_existing: Skip images that already have results.
        visualize: Generate overlay visualizations.
        device: CUDA device.
    """
    data_dir = Path(data_dir)

    if output_dir is None:
        output_dir = Path("/home/aviad/detecture_experiments")
    else:
        output_dir = Path(output_dir)

    if exp_name is None:
        exp_name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")

    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {exp_dir}")

    # Discover images
    image_files = sorted((data_dir / "images").glob("*.jpg"))
    image_ids = [f.stem for f in image_files]

    if max_images:
        image_ids = image_ids[:max_images]

    print(f"Found {len(image_ids)} images to process")

    # Check for existing results (resume support)
    metadata_path = exp_dir / "metadata.json"
    existing_metadata = []
    processed_ids = set()

    if skip_existing and metadata_path.exists():
        with open(metadata_path) as f:
            existing_metadata = json.load(f)
        for entry in existing_metadata:
            processed_ids.add(entry.get("source_image_id", ""))
        print(f"Resuming: {len(processed_ids)} images already processed")

    # Load model
    from texture_boundary_Architexture.models.qwen_vlm import create_qwen_model

    print("Loading Qwen model...")
    model = create_qwen_model(model_size="8B", device=device)

    # Process images
    all_metadata = list(existing_metadata)
    stats = {
        "total_images": len(image_ids),
        "processed": 0,
        "skipped_existing": 0,
        "failed": 0,
        "transitions_found": 0,
        "duplicates_removed": 0,
    }

    start_time = time.time()

    for idx, image_id in enumerate(image_ids):
        if image_id in processed_ids:
            stats["skipped_existing"] += 1
            continue

        print(f"[{idx+1}/{len(image_ids)}] Processing {image_id}...")

        entries = process_image(
            model, image_id, data_dir, exp_dir,
            sample_index=len(all_metadata), visualize=visualize,
        )

        if entries:
            all_metadata.extend(entries)
            stats["transitions_found"] += len(entries)
            stats["processed"] += 1
            print(f"  -> {len(entries)} transitions extracted")
        else:
            stats["failed"] += 1

        # Save incrementally every 10 images
        if (idx + 1) % 10 == 0:
            with open(metadata_path, "w") as f:
                json.dump(all_metadata, f, indent=2)

    # Final save
    elapsed = time.time() - start_time
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    stats["total_transitions"] = len(all_metadata)
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats["experiment"] = str(exp_dir)

    stats_path = exp_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Save experiment config for reproducibility
    config = {
        "data_dir": str(data_dir),
        "exp_name": exp_name,
        "max_images": max_images,
        "total_images_available": len(list((data_dir / "images").glob("*.jpg"))),
        "timestamp": datetime.now().isoformat(),
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*50}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*50}")
    print(f"Experiment: {exp_dir}")
    print(f"Images processed: {stats['processed']}")
    print(f"Images skipped (existing): {stats['skipped_existing']}")
    print(f"Images failed: {stats['failed']}")
    print(f"Total transitions: {stats['total_transitions']}")
    print(f"Time: {elapsed:.0f}s")
    print(f"{'='*50}")
    print(f"\nOutput structure:")
    print(f"  {exp_dir}/metadata.json  <- training metadata (RWTD format)")
    print(f"  {exp_dir}/images/               <- source images")
    print(f"  {exp_dir}/masks/                <- colored annotation masks")
    print(f"  {exp_dir}/masks_texture/         <- binary masks (mask_a, mask_b)")
    print(f"  {exp_dir}/visualizations/        <- overlay visualizations")
    print(f"  {exp_dir}/overlays/              <- original overlays from Detecture")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Texture transition extraction pipeline")
    parser.add_argument("--data-dir", default="/home/aviad/detecture_data")
    parser.add_argument("--output-dir", default="/datasets/ade20k/",
                        help="Base output dir (default: /home/aviad/detecture_experiments)")
    parser.add_argument("--exp-name", default="Detecture_dataset",
                        help="Experiment name (default: exp_TIMESTAMP)")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--no-skip", action="store_true", help="Don't skip existing results")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        max_images=args.max_images,
        skip_existing=not args.no_skip,
        visualize=not args.no_viz,
        device=args.device,
    )
