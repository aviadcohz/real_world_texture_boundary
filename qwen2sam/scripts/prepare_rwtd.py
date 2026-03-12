"""
Adapt RWTD dataset metadata for Phase1Dataset compatibility.

The RWTD dataset uses a different metadata format than what Phase1Dataset expects:
  - RWTD uses "image" instead of "image_path"
  - RWTD has no "coords" field (whole image = bounding box)
  - RWTD has no "crop_name" field

This script reads RWTD's metadata.json, adds the missing fields, writes an
adapted metadata file, and creates train/test splits.

Usage:
  python -m qwen2sam.scripts.prepare_rwtd \
      --data_root /home/aviad/RWTD \
      --test_frac 0.1 \
      --seed 42
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter

import cv2


def adapt_rwtd_entry(entry: dict) -> dict:
    """Add Phase1Dataset-compatible fields to an RWTD metadata entry."""
    adapted = dict(entry)

    # Rename "image" -> "image_path"
    if "image" in adapted and "image_path" not in adapted:
        adapted["image_path"] = adapted["image"]

    # Add crop_name from image filename (e.g., "8.jpg" -> "8")
    if "crop_name" not in adapted:
        img_path = adapted.get("image_path", adapted.get("image", ""))
        adapted["crop_name"] = Path(img_path).stem

    # Add coords = [0, 0, W, H] (whole image is the bounding box)
    # Read actual image dimensions
    if "coords" not in adapted:
        img_path = adapted.get("image_path", adapted.get("image", ""))
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            adapted["coords"] = [0, 0, w, h]
        else:
            adapted["coords"] = [0, 0, 256, 256]  # fallback

    return adapted


def has_valid_oracle_points(entry: dict, min_points: int = 2) -> bool:
    """Check if an entry has valid oracle points for both masks."""
    oracle = entry.get("oracle_points")
    if oracle is None:
        return False
    pts_a = oracle.get("point_prompt_mask_a")
    pts_b = oracle.get("point_prompt_mask_b")
    if pts_a is None or pts_b is None:
        return False
    if len(pts_a) < min_points or len(pts_b) < min_points:
        return False
    return True


def has_valid_files(entry: dict) -> bool:
    """Check if the required files exist on disk."""
    for key in ["image_path", "mask_a_path", "mask_b_path"]:
        path = entry.get(key, "")
        if not path or not Path(path).exists():
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Adapt RWTD dataset and create splits")
    parser.add_argument("--data_root", type=str, default="/home/aviad/RWTD")
    parser.add_argument("--metadata_file", type=str, default="metadata.json")
    parser.add_argument("--output_metadata", type=str, default="metadata_phase1.json",
                        help="Output adapted metadata filename")
    parser.add_argument("--test_frac", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_points", type=int, default=2)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    metadata_path = data_root / args.metadata_file

    with open(metadata_path) as f:
        metadata = json.load(f)

    print(f"RWTD entries: {len(metadata)}")

    # ---- Adapt entries --------------------------------------------------- #
    print("Adapting RWTD metadata to Phase1Dataset format...")
    adapted = []
    for entry in metadata:
        adapted.append(adapt_rwtd_entry(entry))

    # Save adapted metadata
    adapted_path = data_root / args.output_metadata
    with open(adapted_path, "w") as f:
        json.dump(adapted, f, indent=2)
    print(f"Adapted metadata saved: {adapted_path}")

    # ---- Filter valid entries -------------------------------------------- #
    valid_indices = []
    skipped_indices = []
    skip_reasons = Counter()

    for i, entry in enumerate(adapted):
        if not has_valid_oracle_points(entry, args.min_points):
            skipped_indices.append(i)
            skip_reasons["no_oracle_points"] += 1
            continue
        if not has_valid_files(entry):
            skipped_indices.append(i)
            skip_reasons["missing_files"] += 1
            continue
        valid_indices.append(i)

    print(f"\nValid entries: {len(valid_indices)}")
    print(f"Skipped entries: {len(skipped_indices)}")
    for reason, count in skip_reasons.items():
        print(f"  {reason}: {count}")

    # ---- Shuffle and split ----------------------------------------------- #
    rng = random.Random(args.seed)
    shuffled = valid_indices.copy()
    rng.shuffle(shuffled)

    test_size = max(1, int(len(shuffled) * args.test_frac))
    test_indices = sorted(shuffled[:test_size])
    train_indices = sorted(shuffled[test_size:])

    print(f"\nSplit (seed={args.seed}):")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Test:  {len(test_indices)} samples")

    # ---- Print test set diversity ---------------------------------------- #
    print(f"\nTest set samples:")
    for idx in test_indices:
        e = adapted[idx]
        name = e.get("crop_name", f"idx_{idx}")
        print(f"  [{idx}] {name}: \"{e.get('texture_a', '?')}\" vs \"{e.get('texture_b', '?')}\"")

    # ---- Save splits ----------------------------------------------------- #
    splits = {
        "seed": args.seed,
        "test_frac": args.test_frac,
        "total_entries": len(adapted),
        "valid_entries": len(valid_indices),
        "train_indices": train_indices,
        "test_indices": test_indices,
        "skipped_indices": skipped_indices,
        "metadata_file": args.output_metadata,
    }

    splits_path = data_root / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\nSaved: {splits_path}")
    print(f"\nTo train, use:")
    print(f"  --data_root {data_root}")
    print(f"  --metadata_file {args.output_metadata}")
    print(f"  --splits {splits_path}")


if __name__ == "__main__":
    main()
