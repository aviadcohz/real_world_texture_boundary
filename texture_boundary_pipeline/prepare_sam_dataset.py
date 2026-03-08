"""
prepare_sam_dataset.py

Collects passed crops from a pipeline run into a flat, training-ready
directory structure for SAM fine-tuning.

Selection criteria (both must hold):
  1. Crop passed the entropy filter  (filter/entropy_filter_results.json)
  2. Crop has oracle_points          (not null in processed_bboxes.json)

Output structure (flat — no category subdirectories):
  <output_dir>/
    images/            original source images  (deduplicated by image name)
    crops/             crop images             (1024×1024 JPGs)
    masks/             boundary masks          (PNGs)
    masks_textures/    _mask_a.png / _mask_b.png per crop
    metadata.json      one record per crop with all fields SAM needs

metadata.json record keys:
  image          source image filename
  image_path     absolute path inside output images/
  description    texture boundary description
  coords         [x1, y1, x2, y2] bbox in original image space
  crop_name      crop filename
  crop_path      absolute path inside output crops/
  crop_category  size category (tiny/small/medium/large/xlarge)
  mask_path      absolute path inside output masks/
  mask_a_path    absolute path inside output masks_textures/
  mask_b_path    absolute path inside output masks_textures/
  oracle_points  {point_prompt_mask_a: [[x,y],[x,y]],
                  point_prompt_mask_b: [[x,y],[x,y]]}
  texture_a      first texture label
  texture_b      second texture label
  entropy_a      Shannon entropy of texture A region
  entropy_b      Shannon entropy of texture B region

Usage:
  python prepare_sam_dataset.py <run_dir> <output_dir>

Example:
  python prepare_sam_dataset.py \\
      /home/aviad/results/debug_for_training/run_20260222_140936 \\
      /datasets/sam_training/run_20260222
"""

import json
import shutil
import sys
from pathlib import Path


def prepare_sam_dataset(run_dir: str | Path, output_dir: str | Path) -> Path:
    """
    Collect passed crops into a flat SAM training dataset.

    Args:
        run_dir:    Path to the pipeline run directory
                    (contains processed_bboxes.json, filter/, crops/, masks/, masks_textures/)
        output_dir: Destination directory (will be created if needed)

    Returns:
        Path to the written metadata.json
    """
    run_dir    = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve()

    # ── validate inputs ───────────────────────────────────────────────────────
    bbox_json   = run_dir / "processed_bboxes.json"
    filter_json = run_dir / "filter" / "entropy_filter_results.json"

    for p in [bbox_json, filter_json]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    # ── create output directories ─────────────────────────────────────────────
    images_dir         = output_dir / "images"
    crops_dir          = output_dir / "crops"
    masks_dir          = output_dir / "masks"
    masks_textures_dir = output_dir / "masks_textures"

    for d in [images_dir, crops_dir, masks_dir, masks_textures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    with open(bbox_json) as f:
        bbox_data = json.load(f)

    with open(filter_json) as f:
        filter_data = json.load(f)

    # Build lookup: crop_name -> filter entry  (passed only)
    filter_lookup: dict[str, dict] = {
        entry["crop_name"]: entry
        for entry in filter_data
        if entry.get("passed")
    }

    # ── collect ───────────────────────────────────────────────────────────────
    metadata      = []
    copied_images : set[str] = set()
    missing_files : list[str] = []

    stats = {
        "total_boxes":        0,
        "skipped_not_passed": 0,
        "skipped_no_oracle":  0,
        "collected":          0,
    }

    for img_entry in bbox_data:
        source_image_name = img_entry["image"]
        source_image_path = Path(img_entry["image_path"])

        for box in img_entry["boxes"]:
            stats["total_boxes"] += 1
            crop_name = box["crop_name"]
            category  = box["crop_category"]
            stem      = Path(crop_name).stem

            # ── filter: must have passed entropy filter ────────────────────────
            filter_entry = filter_lookup.get(crop_name)
            if filter_entry is None:
                stats["skipped_not_passed"] += 1
                continue

            # ── filter: must have oracle points ───────────────────────────────
            oracle_points = box.get("oracle_points")
            if oracle_points is None:
                stats["skipped_no_oracle"] += 1
                continue

            # ── resolve source paths ──────────────────────────────────────────
            crop_src   = run_dir / "crops"          / category / crop_name
            mask_src   = run_dir / "masks"          / category / f"{stem}.png"
            mask_a_src = run_dir / "masks_textures" / category / f"{stem}_mask_a.png"
            mask_b_src = run_dir / "masks_textures" / category / f"{stem}_mask_b.png"

            # ── destination paths (flat) ──────────────────────────────────────
            dest_image  = images_dir         / source_image_name
            dest_crop   = crops_dir          / crop_name
            dest_mask   = masks_dir          / f"{stem}.png"
            dest_mask_a = masks_textures_dir / f"{stem}_mask_a.png"
            dest_mask_b = masks_textures_dir / f"{stem}_mask_b.png"

            # ── copy source image (once per unique image) ─────────────────────
            if source_image_name not in copied_images:
                if source_image_path.exists():
                    shutil.copy2(source_image_path, dest_image)
                    copied_images.add(source_image_name)
                else:
                    missing_files.append(str(source_image_path))
                    print(f"  ⚠️  Source image not found: {source_image_path}")

            # ── copy crop + masks ─────────────────────────────────────────────
            for src, dst in [
                (crop_src,   dest_crop),
                (mask_src,   dest_mask),
                (mask_a_src, dest_mask_a),
                (mask_b_src, dest_mask_b),
            ]:
                if src.exists():
                    shutil.copy2(src, dst)
                else:
                    missing_files.append(str(src))
                    print(f"  ⚠️  Missing: {src}")

            # ── build metadata record ─────────────────────────────────────────
            metadata.append({
                "image":         source_image_name,
                "image_path":    str(dest_image),
                "description":   box.get("description", ""),
                "coords":        box.get("coords"),       # [x1, y1, x2, y2]
                "crop_name":     crop_name,
                "crop_path":     str(dest_crop),
                "crop_category": category,
                "mask_path":     str(dest_mask),
                "mask_a_path":   str(dest_mask_a),
                "mask_b_path":   str(dest_mask_b),
                "oracle_points": oracle_points,
                "texture_a":     filter_entry.get("texture_a", ""),
                "texture_b":     filter_entry.get("texture_b", ""),
                "entropy_a":     filter_entry.get("entropy_a"),
                "entropy_b":     filter_entry.get("entropy_b"),
            })

            stats["collected"] += 1

    # ── write metadata ────────────────────────────────────────────────────────
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SAM DATASET PREPARED")
    print(f"{'='*60}")
    print(f"  Source run:          {run_dir}")
    print(f"  Output:              {output_dir}")
    print(f"  Total boxes:         {stats['total_boxes']}")
    print(f"  Skipped (no pass):   {stats['skipped_not_passed']}")
    print(f"  Skipped (no oracle): {stats['skipped_no_oracle']}")
    print(f"  Collected crops:     {stats['collected']}")
    print(f"  Unique images:       {len(copied_images)}")
    if missing_files:
        print(f"  Missing files:       {len(missing_files)}")
    print(f"\n  Output layout:")
    print(f"    images/            {len(copied_images)} source images")
    print(f"    crops/             {stats['collected']} crop images")
    print(f"    masks/             {stats['collected']} boundary masks")
    print(f"    masks_textures/    {stats['collected'] * 2} texture masks")
    print(f"    metadata.json      {stats['collected']} records")
    print(f"{'='*60}\n")

    return metadata_path


if __name__ == "__main__":
    run_dir = "/home/aviad/RWTD_for_training/run_20260223_125403"
    output_dir = "/datasets/SAM_testset_training"
    prepare_sam_dataset(run_dir=run_dir, output_dir=output_dir)
