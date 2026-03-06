#!/usr/bin/env python3
"""
Download datasets needed for Stage 1 training.

Datasets:
  1. COCO 2017 (primary): 118K train images, 5K val images, 80 categories
  2. SAM3 checkpoint from HuggingFace

Usage:
    python scripts/download_datasets.py --data-dir /data --download coco sam3
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def download_coco(data_dir: str):
    """Download COCO 2017 train/val images and annotations."""
    coco_dir = Path(data_dir) / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)

    urls = {
        "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    }

    for fname, url in urls.items():
        target = coco_dir / fname
        extracted_name = fname.replace(".zip", "")

        # Skip if already extracted
        if (coco_dir / extracted_name).exists() or (coco_dir / "annotations").exists():
            print(f"  [SKIP] {extracted_name} already exists")
            continue

        if not target.exists():
            print(f"  Downloading {fname}...")
            subprocess.run(
                ["wget", "-c", url, "-O", str(target)],
                check=True,
            )

        print(f"  Extracting {fname}...")
        subprocess.run(
            ["unzip", "-q", "-o", str(target), "-d", str(coco_dir)],
            check=True,
        )

    # Verify
    expected_paths = [
        coco_dir / "train2017",
        coco_dir / "val2017",
        coco_dir / "annotations" / "instances_train2017.json",
        coco_dir / "annotations" / "instances_val2017.json",
    ]
    all_ok = True
    for p in expected_paths:
        if p.exists():
            print(f"  [OK] {p}")
        else:
            print(f"  [MISSING] {p}")
            all_ok = False

    if all_ok:
        print(f"\nCOCO 2017 ready at {coco_dir}")
        print(f"  annotation_file: {coco_dir / 'annotations' / 'instances_train2017.json'}")
        print(f"  image_dir: {coco_dir / 'train2017'}")
    else:
        print("\nWARNING: Some files missing!")

    return coco_dir


def download_sam3_checkpoint():
    """Download SAM3 checkpoint from HuggingFace."""
    print("Downloading SAM3 checkpoint from HuggingFace...")
    try:
        sys.path.insert(0, "/home/aviad/sam3")
        from sam3.model_builder import download_ckpt_from_hf
        path = download_ckpt_from_hf()
        print(f"  SAM3 checkpoint: {path}")
        return path
    except Exception as e:
        print(f"  Error: {e}")
        print("  You can manually download from: https://huggingface.co/facebook/sam3")
        return None


def print_config_template(data_dir: str):
    """Print the config values to use in stage1.yaml."""
    coco_dir = Path(data_dir) / "coco"
    print("\n" + "=" * 60)
    print("Update your configs/stage1.yaml with these paths:")
    print("=" * 60)
    print(f"""
data:
  annotation_file: "{coco_dir / 'annotations' / 'instances_train2017.json'}"
  image_dir: "{coco_dir / 'train2017'}"
""")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for Qwen2SAM_advance")
    parser.add_argument(
        "--data-dir", type=str, default="/data",
        help="Root directory for datasets",
    )
    parser.add_argument(
        "--download", nargs="+", choices=["coco", "sam3", "all"],
        default=["all"],
        help="What to download",
    )
    args = parser.parse_args()

    targets = set(args.download)
    if "all" in targets:
        targets = {"coco", "sam3"}

    if "coco" in targets:
        print("\n=== Downloading COCO 2017 ===")
        download_coco(args.data_dir)

    if "sam3" in targets:
        print("\n=== Downloading SAM3 Checkpoint ===")
        download_sam3_checkpoint()

    print_config_template(args.data_dir)


if __name__ == "__main__":
    main()
