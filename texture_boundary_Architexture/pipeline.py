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
from texture_boundary_Architexture.core.transition_cropper import clean_mask, find_best_crops, refine_crop_masks
from texture_boundary_Architexture.core.texture_refiner_pipeline import TextureRefinerPipeline
from texture_boundary_Architexture.core.visualization import (
    create_transition_overlay,
    save_transition_visualizations,
)


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


def _nms_crop_boxes(boxes, scores, iou_threshold=0.60):
    """Non-maximum suppression across crop boxes. Keep higher-scoring crop when IoU > threshold."""
    if not boxes:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        y1_i, x1_i, y2_i, x2_i = boxes[i]
        area_i = (y2_i - y1_i) * (x2_i - x1_i)
        for j in order:
            if j in suppressed or j == i:
                continue
            y1_j, x1_j, y2_j, x2_j = boxes[j]
            area_j = (y2_j - y1_j) * (x2_j - x1_j)
            iy1, ix1 = max(y1_i, y1_j), max(x1_i, x1_j)
            iy2, ix2 = min(y2_i, y2_j), min(x2_i, x2_j)
            inter = max(0, iy2 - iy1) * max(0, ix2 - ix1)
            union = area_i + area_j - inter
            if union > 0 and inter / union > iou_threshold:
                suppressed.add(j)
    return keep


def process_image(
    model,
    image_id: str,
    data_dir: Path,
    exp_dir: Path,
    sample_index: int,
    visualize: bool = True,
    crop_transitions: bool = True,
    refiner: TextureRefinerPipeline = None,
) -> list:
    """Process a single image: Qwen analysis + mask extraction + full metadata.

    Args:
        model: QwenVLM model instance.
        image_id: Image filename stem.
        data_dir: Source data directory (images/, masks/).
        exp_dir: Experiment output directory.
        sample_index: Starting index for naming samples.
        visualize: Whether to generate overlay visualizations.
        crop_transitions: Whether to extract tight boundary crops.

    Returns list of RWTD-format metadata dicts for valid transitions.
    """
    # Find source files (try common extensions)
    src_image = None
    src_mask = None
    for ext in ["jpg", "jpeg", "png", "bmp", "tiff"]:
        candidate = data_dir / "images" / f"{image_id}.{ext}"
        if candidate.exists():
            src_image = candidate
            break
    for ext in ["jpg", "jpeg", "png", "bmp", "tiff"]:
        candidate = data_dir / "masks" / f"{image_id}.{ext}"
        if candidate.exists():
            src_mask = candidate
            break

    if src_image is None or src_mask is None:
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
    crops_images_dir = exp_dir / "crops" / "images"
    crops_masks_dir = exp_dir / "crops" / "masks_texture"
    crops_viz_dir = exp_dir / "crops" / "visualizations"
    refined_images_dir = exp_dir / "crops" / "refined" / "images"
    refined_masks_dir = exp_dir / "crops" / "refined" / "masks_texture"
    refined_viz_dir = exp_dir / "crops" / "refined" / "visualizations"
    dirs = [images_dir, masks_texture_dir, masks_dir, viz_dir, overlays_dir]
    if crop_transitions:
        dirs.extend([crops_images_dir, crops_masks_dir, crops_viz_dir])
        if refiner is not None:
            dirs.extend([refined_images_dir, refined_masks_dir, refined_viz_dir])
    for d in dirs:
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
        src_overlay = None
        for ext in ["jpg", "jpeg", "png"]:
            candidate = data_dir / "overlays" / f"{image_id}.{ext}"
            if candidate.exists():
                src_overlay = candidate
                break
        if src_overlay is not None:
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
            "crops": [],
        }

        # Extract tight boundary crops with color-based refinement
        if crop_transitions:
            src_img_array = np.array(Image.open(src_image).convert("RGB"))
            crops = find_best_crops(mask_a, mask_b, image=src_img_array)
            if crops:
                crop_details = []
                MIN_CROP_SIDE = 64
                for j, (cy1, cx1, cy2, cx2, score, _, _) in enumerate(crops):
                    # Skip crops smaller than minimum size
                    crop_h, crop_w = cy2 - cy1, cx2 - cx1
                    if min(crop_h, crop_w) < MIN_CROP_SIDE:
                        continue

                    # Crop image and masks, clean before refinement
                    crop_img_arr = src_img_array[cy1:cy2, cx1:cx2]
                    crop_ma = clean_mask(mask_a[cy1:cy2, cx1:cx2])
                    crop_mb = clean_mask(mask_b[cy1:cy2, cx1:cx2])

                    # Refine: assign third-class pixels to nearest texture
                    crop_ma, crop_mb = refine_crop_masks(
                        crop_img_arr, crop_ma, crop_mb, max_distance=40.0
                    )

                    # Clean again after refinement to remove scattered pixels
                    crop_ma = clean_mask(crop_ma)
                    crop_mb = clean_mask(crop_mb)

                    # Save cropped files
                    crop_img_path = crops_images_dir / f"{crop_name}_crop{j}.jpg"
                    crop_ma_path = crops_masks_dir / f"{crop_name}_crop{j}_mask_a.png"
                    crop_mb_path = crops_masks_dir / f"{crop_name}_crop{j}_mask_b.png"
                    Image.fromarray(crop_img_arr).save(str(crop_img_path), quality=95)
                    Image.fromarray((crop_ma.astype(np.uint8) * 255)).save(str(crop_ma_path))
                    Image.fromarray((crop_mb.astype(np.uint8) * 255)).save(str(crop_mb_path))

                    # SR refinement: upscale crop + smooth masks
                    refined_info = {}
                    if refiner is not None:
                        result = refiner.process_crop(
                            crop_img_arr,
                            crop_ma.astype(np.uint8) * 255,
                            crop_mb.astype(np.uint8) * 255,
                        )
                        ref_img_path = refined_images_dir / f"{crop_name}_crop{j}.png"
                        ref_ma_path = refined_masks_dir / f"{crop_name}_crop{j}_mask_a.png"
                        ref_mb_path = refined_masks_dir / f"{crop_name}_crop{j}_mask_b.png"
                        result["image"].save(str(ref_img_path))
                        Image.fromarray(result["mask_a"]).save(str(ref_ma_path))
                        Image.fromarray(result["mask_b"]).save(str(ref_mb_path))
                        refined_info = {
                            "refined_image_path": str(ref_img_path.resolve()),
                            "refined_mask_a_path": str(ref_ma_path.resolve()),
                            "refined_mask_b_path": str(ref_mb_path.resolve()),
                            "scale_factor": result["scale_factor"],
                            "refined_size": list(result["sr_size"]),
                        }
                        # Refined visualization overlay
                        ref_rgb = np.array(result["image"])
                        ref_ma_bool = result["mask_a"] > 127
                        ref_mb_bool = result["mask_b"] > 127
                        ref_overlay = create_transition_overlay(
                            ref_rgb, ref_ma_bool, ref_mb_bool,
                            t["texture_a"], t["texture_b"],
                            show_labels=False,
                        )
                        ref_viz_path = refined_viz_dir / f"{crop_name}_crop{j}.jpg"
                        ref_overlay.save(str(ref_viz_path), quality=92)

                    # Save visualization overlay (no labels — crops are small)
                    overlay = create_transition_overlay(
                        crop_img_arr, crop_ma, crop_mb,
                        t["texture_a"], t["texture_b"],
                        show_labels=False,
                    )
                    crop_viz_path = crops_viz_dir / f"{crop_name}_crop{j}.jpg"
                    overlay.save(str(crop_viz_path), quality=92)

                    # Oracle points for refined cropped masks
                    crop_pts_a = sample_oracle_points(crop_ma, n_points=4)
                    crop_pts_b = sample_oracle_points(crop_mb, n_points=4)

                    # Final balance after refinement
                    area = crop_ma.size
                    final_frac_a = round(float(crop_ma.sum()) / area, 3)
                    final_frac_b = round(float(crop_mb.sum()) / area, 3)

                    crop_details.append({
                        "crop_index": j,
                        "box": [int(cy1), int(cx1), int(cy2), int(cx2)],
                        "score": round(float(score), 6),
                        "crop_image_path": str(crop_img_path.resolve()),
                        "crop_mask_a_path": str(crop_ma_path.resolve()),
                        "crop_mask_b_path": str(crop_mb_path.resolve()),
                        "crop_size": [int(cx2 - cx1), int(cy2 - cy1)],
                        "balance": [final_frac_a, final_frac_b],
                        "oracle_points": {
                            "point_prompt_mask_a": crop_pts_a,
                            "point_prompt_mask_b": crop_pts_b,
                        },
                        **refined_info,
                    })
                entry["crops"] = crop_details

        metadata_entries.append(entry)

        viz_transitions.append({
            "mask_a": mask_a,
            "mask_b": mask_b,
            "texture_a": t["texture_a"],
            "texture_b": t["texture_b"],
        })

    # Deduplicate crops across transitions within this image
    # Crops from different transitions can overlap heavily (e.g., sand/stone and sand/water
    # transitions share the same boundary region). Remove duplicates by box IoU.
    if crop_transitions and len(metadata_entries) > 1:
        all_crops = []  # (transition_idx, crop_idx_in_list, box, score)
        for t_idx, entry in enumerate(metadata_entries):
            for c_idx, crop in enumerate(entry.get("crops", [])):
                box = tuple(crop["box"])  # (y1, x1, y2, x2)
                all_crops.append((t_idx, c_idx, box, crop["score"]))

        if len(all_crops) > 1:
            boxes = [c[2] for c in all_crops]
            scores = [c[3] for c in all_crops]
            keep_indices = _nms_crop_boxes(boxes, scores, iou_threshold=0.60)
            keep_set = set(keep_indices)

            n_before = len(all_crops)
            # Rebuild crop lists, keeping only non-suppressed crops
            for t_idx, entry in enumerate(metadata_entries):
                kept = []
                for c_idx, crop in enumerate(entry.get("crops", [])):
                    # Find this crop's index in all_crops
                    global_idx = next(
                        i for i, (ti, ci, _, _) in enumerate(all_crops)
                        if ti == t_idx and ci == c_idx
                    )
                    if global_idx in keep_set:
                        kept.append(crop)
                entry["crops"] = kept

            n_after = sum(len(e.get("crops", [])) for e in metadata_entries)
            if n_before > n_after:
                print(f"  Deduped crops: {n_before} → {n_after} (removed {n_before - n_after} overlapping)")

    # Save visualizations
    if visualize and viz_transitions:
        save_transition_visualizations(str(src_image), viz_transitions, str(viz_dir), image_id)

    return metadata_entries


def _print_pipeline_summary(stats, crop_metadata, all_metadata, exp_dir, elapsed):
    """Print comprehensive pipeline statistics with crop size distributions."""
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Experiment:  {exp_dir}")
    print(f"  Time:        {elapsed:.0f}s")
    print(f"{'='*60}")

    # --- Image-level stats ---
    print(f"\n--- Images ---")
    print(f"  Processed:       {stats['processed']}")
    print(f"  Skipped (exist): {stats['skipped_existing']}")
    print(f"  Failed:          {stats['failed']}")
    print(f"  Transitions:     {stats['total_transitions']}")

    # --- Original image size distribution ---
    if all_metadata:
        seen_images = {}
        for e in all_metadata:
            sid = e.get("source_image_id", "")
            if sid not in seen_images:
                coords = e.get("coords", [0, 0, 0, 0])
                seen_images[sid] = (coords[2], coords[3])  # w, h

        if seen_images:
            widths = [w for w, h in seen_images.values() if w > 0]
            heights = [h for w, h in seen_images.values() if h > 0]
            if widths:
                print(f"\n--- Original Image Sizes ({len(seen_images)} unique images) ---")
                min_sides = [min(w, h) for w, h in seen_images.values() if w > 0]
                max_sides = [max(w, h) for w, h in seen_images.values() if w > 0]
                # Size buckets
                buckets = [(0, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 9999)]
                labels = ["<256", "256-512", "512-1K", "1K-2K", "2K+"]
                for (lo, hi), label in zip(buckets, labels):
                    count = sum(1 for s in min_sides if lo <= s < hi)
                    if count > 0:
                        pct = count / len(min_sides) * 100
                        bar = "#" * int(pct / 2)
                        print(f"  {label:>8}px: {count:>5} ({pct:5.1f}%) {bar}")
                print(f"  Mean:  {sum(widths)/len(widths):.0f}x{sum(heights)/len(heights):.0f}")
                sorted_w = sorted(widths)
                sorted_h = sorted(heights)
                print(f"  Median: {sorted_w[len(sorted_w)//2]}x{sorted_h[len(sorted_h)//2]}")
                print(f"  Range:  {min(widths)}x{min(heights)} — {max(widths)}x{max(heights)}")

    if not crop_metadata:
        print(f"\n  No crops extracted.")
        return

    # --- Crop-level stats ---
    total_crops = len(crop_metadata)
    refined_crops = [c for c in crop_metadata if "refined_image_path" in c]
    n_refined = len(refined_crops)

    print(f"\n--- Crops ---")
    print(f"  Total crops (>=64px min side): {total_crops}")
    print(f"  Refined (SR applied):          {n_refined}")

    # Count how many transitions have at least one crop
    transitions_with_crops = sum(
        1 for e in all_metadata if len(e.get("crops", [])) > 0
    )
    transitions_no_crops = len(all_metadata) - transitions_with_crops
    print(f"  Transitions with crops:        {transitions_with_crops}")
    print(f"  Transitions without crops:     {transitions_no_crops} (all crops <64px)")

    # --- Input size distribution ---
    input_sizes = []
    for c in crop_metadata:
        w, h = c["coords"][2], c["coords"][3]
        if w > 0 and h > 0:
            input_sizes.append((w, h))

    if input_sizes:
        min_sides = [min(w, h) for w, h in input_sizes]
        max_sides = [max(w, h) for w, h in input_sizes]

        print(f"\n--- Input Crop Size Distribution (min side) ---")
        buckets = [(64, 96), (96, 128), (128, 192), (192, 256), (256, 512)]
        for lo, hi in buckets:
            count = sum(1 for s in min_sides if lo <= s < hi)
            pct = count / len(min_sides) * 100
            bar = "#" * int(pct / 2)
            print(f"  {lo:>4}-{hi:<4}px: {count:>5} ({pct:5.1f}%) {bar}")

        avg_min = sum(min_sides) / len(min_sides)
        print(f"  Mean min side: {avg_min:.0f}px")
        sorted_min = sorted(min_sides)
        print(f"  Median:        {sorted_min[len(sorted_min)//2]}px")
        print(f"  Range:         {min(min_sides)}-{max(min_sides)}px")

    # --- Refined output size distribution ---
    if refined_crops:
        ref_sizes = [c["refined_size"] for c in refined_crops]
        ref_min_sides = [min(h, w) for h, w in ref_sizes]
        scale_factors = [c["scale_factor"] for c in refined_crops]

        print(f"\n--- Refined Output Size Distribution (min side) ---")
        buckets_ref = [(64, 128), (128, 256), (256, 384), (384, 512), (512, 1024)]
        for lo, hi in buckets_ref:
            count = sum(1 for s in ref_min_sides if lo <= s < hi)
            pct = count / len(ref_min_sides) * 100
            bar = "#" * int(pct / 2)
            print(f"  {lo:>4}-{hi:<4}px: {count:>5} ({pct:5.1f}%) {bar}")

        avg_ref = sum(ref_min_sides) / len(ref_min_sides)
        print(f"  Mean min side: {avg_ref:.0f}px")
        sorted_ref = sorted(ref_min_sides)
        print(f"  Median:        {sorted_ref[len(sorted_ref)//2]}px")
        print(f"  Range:         {min(ref_min_sides)}-{max(ref_min_sides)}px")

        # Scale factor breakdown
        print(f"\n--- Scale Factors ---")
        from collections import Counter
        scale_counts = Counter(scale_factors)
        for scale in sorted(scale_counts):
            count = scale_counts[scale]
            pct = count / len(scale_factors) * 100
            print(f"  {scale}x: {count:>5} ({pct:5.1f}%)")

        # Input → Output size flow
        print(f"\n--- Size Flow (input → refined) ---")
        flow_buckets = {}
        for c in refined_crops:
            in_w, in_h = c["coords"][2], c["coords"][3]
            in_min = min(in_w, in_h) if in_w > 0 and in_h > 0 else 0
            out_h, out_w = c["refined_size"]
            out_min = min(out_h, out_w)

            # Bucket input
            for lo, hi in [(64, 96), (96, 128), (128, 192), (192, 256), (256, 512)]:
                if lo <= in_min < hi:
                    key = f"{lo}-{hi}px"
                    if key not in flow_buckets:
                        flow_buckets[key] = []
                    flow_buckets[key].append(out_min)
                    break

        for bucket in sorted(flow_buckets):
            outs = flow_buckets[bucket]
            avg_out = sum(outs) / len(outs)
            print(f"  {bucket:>10} ({len(outs):>4} crops) → avg {avg_out:.0f}px refined")

    # --- Output structure ---
    print(f"\n--- Output Structure ---")
    print(f"  {exp_dir}/metadata.json")
    print(f"  {exp_dir}/crops/metadata.json       ({total_crops} crops)")
    if n_refined:
        print(f"  {exp_dir}/crops/refined/images/     ({n_refined} refined)")
        print(f"  {exp_dir}/crops/refined/visualizations/")
    print(f"{'='*60}\n")

    # Add distribution to stats for JSON export
    if input_sizes:
        stats["crop_input_min_side_mean"] = round(avg_min, 1)
        stats["crop_input_min_side_median"] = sorted_min[len(sorted_min) // 2]
    if refined_crops:
        stats["refined_count"] = n_refined
        stats["refined_output_min_side_mean"] = round(avg_ref, 1)
        stats["refined_output_min_side_median"] = sorted_ref[len(sorted_ref) // 2]
        stats["scale_factor_distribution"] = dict(
            sorted((str(k), v) for k, v in scale_counts.items())
        )


def run_pipeline(
    data_dir: str,
    output_dir: str = None,
    exp_name: str = None,
    max_images: int = None,
    skip_existing: bool = True,
    visualize: bool = True,
    crop_transitions: bool = True,
    refine_crops: bool = True,
    device: str = "cuda",
    model_size: str = "8B",
    image_ext: str = "jpg",
):
    """Run the full texture transition extraction pipeline.

    Args:
        data_dir: Directory with images/ and masks/ subdirectories.
        output_dir: Base output directory. Each experiment creates a subfolder.
            Defaults to <data_dir>/experiments.
        exp_name: Experiment name. Defaults to timestamp.
        max_images: Process only first N images (for testing).
        skip_existing: Skip images that already have results.
        visualize: Generate overlay visualizations.
        crop_transitions: Extract tight boundary crops.
        refine_crops: Apply Real-ESRGAN super-resolution to crops.
        device: CUDA device.
        model_size: Qwen model size — "8B" or "2B".
        image_ext: Image file extension in data_dir (default: "jpg").
    """
    data_dir = Path(data_dir)

    if output_dir is None:
        output_dir = data_dir / "experiments"
    else:
        output_dir = Path(output_dir)

    if exp_name is None:
        exp_name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")

    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {exp_dir}")

    # Discover images (support multiple extensions)
    image_dir = data_dir / "images"
    if not image_dir.exists():
        raise FileNotFoundError(
            f"No 'images/' directory found in {data_dir}. "
            f"Expected structure:\n  {data_dir}/images/  (source images)\n  {data_dir}/masks/   (colored segmentation masks)"
        )
    image_files = sorted(image_dir.glob(f"*.{image_ext}"))
    if not image_files:
        # Try common extensions as fallback
        for ext in ["jpg", "jpeg", "png", "bmp", "tiff"]:
            image_files = sorted(image_dir.glob(f"*.{ext}"))
            if image_files:
                break
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

    print(f"Loading Qwen-{model_size} model...")
    model = create_qwen_model(model_size=model_size, device=device)

    # Init SR refiner (lazy-loads on first crop)
    refiner = None
    if refine_crops and crop_transitions:
        refiner = TextureRefinerPipeline(device=device)
        print("SR refinement enabled (Real-ESRGAN x2plus, max 2x, max 512px)")

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
            crop_transitions=crop_transitions, refiner=refiner,
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

    # Build crop-level metadata (one entry per crop, RWTD-compatible format)
    crop_metadata = []
    for entry in all_metadata:
        for crop in entry.get("crops", []):
            crop_entry = {
                "image": crop["crop_image_path"],
                "image_path": crop["crop_image_path"],
                "mask_a_path": crop["crop_mask_a_path"],
                "mask_b_path": crop["crop_mask_b_path"],
                "texture_a": entry["texture_a"],
                "texture_b": entry["texture_b"],
                "description": entry["description"],
                "vlm_assigned": entry.get("vlm_assigned", True),
                "oracle_points": crop["oracle_points"],
                "crop_name": f"{entry['crop_name']}_crop{crop['crop_index']}",
                "coords": [0, 0, crop["crop_size"][1], crop["crop_size"][0]],
                "source_image_id": entry.get("source_image_id", ""),
                "source_transition": entry["crop_name"],
                "source_box": crop["box"],
                "crop_score": crop["score"],
                "balance": crop["balance"],
            }
            # Add refined paths if SR was applied
            if "refined_image_path" in crop:
                crop_entry["refined_image_path"] = crop["refined_image_path"]
                crop_entry["refined_mask_a_path"] = crop["refined_mask_a_path"]
                crop_entry["refined_mask_b_path"] = crop["refined_mask_b_path"]
                crop_entry["scale_factor"] = crop["scale_factor"]
                crop_entry["refined_size"] = crop["refined_size"]
            crop_metadata.append(crop_entry)

    crops_meta_path = exp_dir / "crops" / "metadata.json"
    crops_meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(crops_meta_path, "w") as f:
        json.dump(crop_metadata, f, indent=2)

    stats["total_transitions"] = len(all_metadata)
    stats["total_crops"] = len(crop_metadata)
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats["experiment"] = str(exp_dir)

    # Save experiment config for reproducibility
    config = {
        "data_dir": str(data_dir),
        "exp_name": exp_name,
        "max_images": max_images,
        "total_images_available": len(image_ids),
        "model_size": model_size,
        "image_ext": image_ext,
        "timestamp": datetime.now().isoformat(),
        "min_crop_side": 64,
        "refine_crops": refine_crops,
        "max_scale": 2,
        "max_output": 512,
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Compute detailed crop statistics
    _print_pipeline_summary(stats, crop_metadata, all_metadata, exp_dir, elapsed)

    # Save stats (including crop size distribution)
    stats_path = exp_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Texture transition extraction pipeline")
    parser.add_argument("--data-dir", required=True,
                        help="Directory with images/ and masks/ subdirectories")
    parser.add_argument("--output-dir", default=None,
                        help="Base output dir (default: <data-dir>/experiments)")
    parser.add_argument("--exp-name", default=None,
                        help="Experiment name (default: exp_TIMESTAMP)")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--no-skip", action="store_true", help="Don't skip existing results")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")
    parser.add_argument("--no-crop", action="store_true", help="Skip boundary crop extraction")
    parser.add_argument("--no-refine", action="store_true", help="Skip SR refinement of crops")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-size", default="8B", choices=["8B", "2B"],
                        help="Qwen model size (default: 8B)")
    parser.add_argument("--image-ext", default="jpg",
                        help="Image file extension in data-dir (default: jpg)")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        max_images=args.max_images,
        skip_existing=not args.no_skip,
        visualize=not args.no_viz,
        crop_transitions=not args.no_crop,
        refine_crops=not args.no_refine,
        device=args.device,
        model_size=args.model_size,
        image_ext=args.image_ext,
    )
