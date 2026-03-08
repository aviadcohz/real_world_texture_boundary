"""
convert_to_qwen2sam.py

Converts pipeline output (from prepare_sam_dataset.py or raw pipeline run)
into Qwen2SAM v3 training metadata format.

Qwen2SAM v3 metadata format (metadata_phase1.json):
{
    "image_path": "/path/to/crop.jpg",       # crop is the "image"
    "mask_a_path": "/path/to/mask_a.png",
    "mask_b_path": "/path/to/mask_b.png",
    "texture_a": "smooth flat plaster surface with fine grain",  # 5-10 words
    "texture_b": "rough red brick pattern with deep mortar lines",  # 5-10 words
    "coords": [0, 0, width, height]           # full crop (masks at crop scale)
}

Key design:
- Uses crops directly as images (not original full images)
- coords = [0, 0, crop_w, crop_h] since masks are already at crop resolution
- Qwen2SAM dataset handles resize to 1008x1008 internally
- Descriptions validated: 4-12 words per texture (targets 5-10)

Usage:
    python scripts/convert_to_qwen2sam.py <input_metadata> <output_path> [--min-words 4] [--max-words 12]

Examples:
    # From prepare_sam_dataset output:
    python scripts/convert_to_qwen2sam.py \
        /datasets/SAM_testset_training/metadata.json \
        /datasets/qwen2sam_training/metadata_phase1.json

    # From raw pipeline run:
    python scripts/convert_to_qwen2sam.py \
        /home/aviad/RWTD_for_training/run_20260223_125403/filter/entropy_filter_results.json \
        /datasets/qwen2sam_training/metadata_phase1.json
"""

import json
import sys
from pathlib import Path
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.sa2va_boundaries import validate_texture_description


def convert_prepared_dataset(input_path: str, output_path: str,
                              min_words: int = 4, max_words: int = 12) -> dict:
    """
    Convert prepare_sam_dataset metadata.json → Qwen2SAM metadata_phase1.json.

    Args:
        input_path: Path to metadata.json from prepare_sam_dataset.py
        output_path: Path to write Qwen2SAM metadata_phase1.json
        min_words: Minimum words per texture description
        max_words: Maximum words per texture description

    Returns:
        Stats dict with counts
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        records = json.load(f)

    qwen2sam_records = []
    stats = {
        "total": len(records),
        "converted": 0,
        "skipped_no_masks": 0,
        "skipped_bad_description": 0,
        "skipped_missing_crop": 0,
    }

    for record in records:
        crop_path = record.get("crop_path", "")
        mask_a_path = record.get("mask_a_path", "")
        mask_b_path = record.get("mask_b_path", "")
        texture_a = record.get("texture_a", "")
        texture_b = record.get("texture_b", "")

        # Verify crop exists and get dimensions
        if not Path(crop_path).exists():
            stats["skipped_missing_crop"] += 1
            continue

        # Verify masks exist
        if not Path(mask_a_path).exists() or not Path(mask_b_path).exists():
            stats["skipped_no_masks"] += 1
            continue

        # Validate description length
        is_valid, reason = validate_texture_description(
            texture_a, texture_b, min_words=min_words, max_words=max_words
        )
        if not is_valid:
            stats["skipped_bad_description"] += 1
            print(f"  Skipped: {reason}")
            continue

        # Get crop dimensions for coords
        with Image.open(crop_path) as img:
            w, h = img.size

        qwen2sam_records.append({
            "image_path": crop_path,
            "mask_a_path": mask_a_path,
            "mask_b_path": mask_b_path,
            "texture_a": texture_a,
            "texture_b": texture_b,
            "coords": [0, 0, w, h],
        })
        stats["converted"] += 1

    # Write output
    with open(output_path, "w") as f:
        json.dump(qwen2sam_records, f, indent=2)

    return stats


def convert_filter_results(input_path: str, output_path: str,
                            min_words: int = 4, max_words: int = 12) -> dict:
    """
    Convert raw entropy_filter_results.json → Qwen2SAM metadata_phase1.json.

    This handles the case where the user runs the conversion directly from
    pipeline output without going through prepare_sam_dataset.py first.

    Args:
        input_path: Path to entropy_filter_results.json
        output_path: Path to write Qwen2SAM metadata_phase1.json
        min_words: Minimum words per texture description
        max_words: Maximum words per texture description

    Returns:
        Stats dict with counts
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        records = json.load(f)

    qwen2sam_records = []
    stats = {
        "total": len(records),
        "converted": 0,
        "skipped_not_passed": 0,
        "skipped_no_masks": 0,
        "skipped_bad_description": 0,
        "skipped_missing_crop": 0,
    }

    for record in records:
        # Only use passed samples
        if not record.get("passed", False):
            stats["skipped_not_passed"] += 1
            continue

        crop_path = record.get("crop_path", "")
        mask_a_path = record.get("mask_a_path", "")
        mask_b_path = record.get("mask_b_path", "")
        texture_a = record.get("texture_a", "")
        texture_b = record.get("texture_b", "")

        # Verify crop exists
        if not Path(crop_path).exists():
            stats["skipped_missing_crop"] += 1
            continue

        # Verify masks exist
        if not mask_a_path or not mask_b_path:
            stats["skipped_no_masks"] += 1
            continue
        if not Path(mask_a_path).exists() or not Path(mask_b_path).exists():
            stats["skipped_no_masks"] += 1
            continue

        # Validate description length
        is_valid, reason = validate_texture_description(
            texture_a, texture_b, min_words=min_words, max_words=max_words
        )
        if not is_valid:
            stats["skipped_bad_description"] += 1
            print(f"  Skipped: {reason}")
            continue

        # Get crop dimensions
        with Image.open(crop_path) as img:
            w, h = img.size

        qwen2sam_records.append({
            "image_path": crop_path,
            "mask_a_path": mask_a_path,
            "mask_b_path": mask_b_path,
            "texture_a": texture_a,
            "texture_b": texture_b,
            "coords": [0, 0, w, h],
        })
        stats["converted"] += 1

    # Write output
    with open(output_path, "w") as f:
        json.dump(qwen2sam_records, f, indent=2)

    return stats


def detect_and_convert(input_path: str, output_path: str,
                        min_words: int = 4, max_words: int = 12) -> dict:
    """
    Auto-detect input format and convert to Qwen2SAM metadata.

    Detects whether input is:
    - prepare_sam_dataset metadata.json (has "crop_category" field)
    - entropy_filter_results.json (has "passed" / "entropy_a" fields)
    """
    with open(input_path) as f:
        records = json.load(f)

    if not records:
        print("ERROR: Empty input file")
        return {"total": 0, "converted": 0}

    sample = records[0]

    # Detect format
    if "passed" in sample and "entropy_a" in sample:
        print("Detected: entropy_filter_results.json format")
        return convert_filter_results(input_path, output_path, min_words, max_words)
    elif "crop_category" in sample or "crop_path" in sample:
        print("Detected: prepare_sam_dataset metadata.json format")
        return convert_prepared_dataset(input_path, output_path, min_words, max_words)
    else:
        print("WARNING: Unknown format, trying prepared dataset format...")
        return convert_prepared_dataset(input_path, output_path, min_words, max_words)


def print_stats(stats: dict):
    """Print conversion statistics."""
    print(f"\n{'='*60}")
    print("QWEN2SAM CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total input records:     {stats['total']}")
    if "skipped_not_passed" in stats:
        print(f"  Skipped (not passed):    {stats['skipped_not_passed']}")
    print(f"  Skipped (no masks):      {stats['skipped_no_masks']}")
    print(f"  Skipped (bad desc):      {stats['skipped_bad_description']}")
    print(f"  Skipped (missing crop):  {stats['skipped_missing_crop']}")
    print(f"  Successfully converted:  {stats['converted']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert pipeline output to Qwen2SAM v3 training metadata"
    )
    parser.add_argument("input", help="Path to input metadata/filter results JSON")
    parser.add_argument("output", help="Path to output metadata_phase1.json")
    parser.add_argument("--min-words", type=int, default=4,
                        help="Min words per texture description (default: 4)")
    parser.add_argument("--max-words", type=int, default=12,
                        help="Max words per texture description (default: 12)")
    args = parser.parse_args()

    stats = detect_and_convert(args.input, args.output, args.min_words, args.max_words)
    print_stats(stats)
    print(f"Output: {args.output}")
