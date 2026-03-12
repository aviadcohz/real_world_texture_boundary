"""
Create a deterministic train/test split for Qwen2SAM overfitting validation.

Reads metadata.json, filters out entries without valid oracle points,
shuffles deterministically, and saves a splits.json with fixed indices.

Usage:
  python -m qwen2sam.scripts.prepare_splits \
      --data_root /datasets/SAM_testset_training \
      --test_frac 0.1 \
      --seed 42
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter


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
    parser = argparse.ArgumentParser(description="Create train/test splits")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--metadata_file", type=str, default="metadata.json")
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_points", type=int, default=2)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    metadata_path = data_root / args.metadata_file

    with open(metadata_path) as f:
        metadata = json.load(f)

    print(f"Total entries: {len(metadata)}")

    # ---- Filter valid entries ---------------------------------------- #
    valid_indices = []
    skipped_indices = []
    skip_reasons = Counter()

    for i, entry in enumerate(metadata):
        if not has_valid_oracle_points(entry, args.min_points):
            skipped_indices.append(i)
            skip_reasons["no_oracle_points"] += 1
            continue
        if not has_valid_files(entry):
            skipped_indices.append(i)
            skip_reasons["missing_files"] += 1
            continue
        valid_indices.append(i)

    print(f"Valid entries: {len(valid_indices)}")
    print(f"Skipped entries: {len(skipped_indices)}")
    for reason, count in skip_reasons.items():
        print(f"  {reason}: {count}")

    # ---- Shuffle and split ------------------------------------------- #
    rng = random.Random(args.seed)
    shuffled = valid_indices.copy()
    rng.shuffle(shuffled)

    test_size = max(1, int(len(shuffled) * args.test_frac))
    test_indices = sorted(shuffled[:test_size])
    train_indices = sorted(shuffled[test_size:])

    print(f"\nSplit (seed={args.seed}):")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Test:  {len(test_indices)} samples")

    # ---- Print test set diversity ------------------------------------ #
    print(f"\nTest set samples:")
    for idx in test_indices:
        e = metadata[idx]
        name = e.get("crop_name", Path(e.get("image_path", e.get("image", f"idx_{idx}"))).stem)
        print(f"  [{idx}] {name}: \"{e.get('texture_a', '?')}\" vs \"{e.get('texture_b', '?')}\"")

    # ---- Save -------------------------------------------------------- #
    splits = {
        "seed": args.seed,
        "test_frac": args.test_frac,
        "total_entries": len(metadata),
        "valid_entries": len(valid_indices),
        "train_indices": train_indices,
        "test_indices": test_indices,
        "skipped_indices": skipped_indices,
    }

    splits_path = data_root / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\nSaved: {splits_path}")


if __name__ == "__main__":
    main()
