"""
Phase 1 - Step 1: Prepare ControlNet training data.

This script:
1. Reads the manually filtered overlays from overlays_filtered/ to determine
   which crops passed quality filtering.
2. Copies the corresponding images, masks, and overlays to a clean training
   directory (controlnet_data_for_training/).
3. Builds a JSON mapping file where each entry contains:
   {"image": path, "conditioning_image": mask_path, "text": prompt}
   The prompt (texture transition description) is extracted from processed_bboxes.json.
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict


# ── Paths ──────────────────────────────────────────────────────────────────────
SOURCE_ROOT = Path("/datasets/google_landmarks_v2_scale/real_world_texture_boundary/results/run_20260208_230720")
OVERLAYS_FILTERED = SOURCE_ROOT / "overlays_filtered"
SOURCE_IMAGES = SOURCE_ROOT / "images"
SOURCE_MASKS = SOURCE_ROOT / "masks"
SOURCE_OVERLAYS = SOURCE_ROOT / "overlays"

DEST_ROOT = Path("/datasets/google_landmarks_v2_scale/real_world_texture_boundary/results/controlnet_data_for_training")
DEST_IMAGES = DEST_ROOT / "images"
DEST_MASKS = DEST_ROOT / "masks"
DEST_OVERLAYS = DEST_ROOT / "overlays"

PROCESSED_BBOXES = Path("/home/aviad/real_world_texture_boundary/google_landmarks_v2_scale/run_20260208_230720/processed_bboxes.json")

OUTPUT_JSON = DEST_ROOT / "training_pairs.json"


def get_filtered_stems():
    """Get the set of filename stems that passed manual filtering.

    overlays_filtered/ contains JPEG files like:
        0000059611c7d079_0_633_581_783.jpg
    We extract the stem (without extension) to match across directories.
    """
    stems = set()
    for f in OVERLAYS_FILTERED.iterdir():
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            stems.add(f.stem)
    print(f"Found {len(stems)} filtered overlays")
    return stems


def build_description_lookup(processed_bboxes_path):
    """Build a lookup: crop_stem -> texture transition description.

    processed_bboxes.json structure:
    [
      {
        "image": "0000059611c7d079.jpg",
        "boxes": [
          {
            "description": "snow-covered metal to icy water",
            "crop_name": "0000059611c7d079_0_633_581_783.jpg",
            ...
          }
        ]
      }
    ]

    We create: {"0000059611c7d079_0_633_581_783": "snow-covered metal to icy water", ...}
    """
    with open(processed_bboxes_path, "r") as f:
        data = json.load(f)

    lookup = {}
    for entry in data:
        for box in entry.get("boxes", []):
            crop_name = box.get("crop_name", "")
            description = box.get("description", "")
            if crop_name and description:
                stem = Path(crop_name).stem
                lookup[stem] = description

    print(f"Built description lookup with {len(lookup)} entries")
    return lookup


def copy_filtered_data(filtered_stems):
    """Copy images, masks, and overlays for filtered stems to destination."""
    os.makedirs(DEST_IMAGES, exist_ok=True)
    os.makedirs(DEST_MASKS, exist_ok=True)
    os.makedirs(DEST_OVERLAYS, exist_ok=True)

    copied = 0
    missing_image = 0
    missing_mask = 0

    for stem in sorted(filtered_stems):
        # Image: .jpg
        src_img = SOURCE_IMAGES / f"{stem}.jpg"
        # Mask: .png
        src_mask = SOURCE_MASKS / f"{stem}.png"
        # Overlay: .jpg
        src_overlay = OVERLAYS_FILTERED / f"{stem}.jpg"

        if not src_img.exists():
            missing_image += 1
            continue
        if not src_mask.exists():
            missing_mask += 1
            continue

        shutil.copy2(src_img, DEST_IMAGES / f"{stem}.jpg")
        shutil.copy2(src_mask, DEST_MASKS / f"{stem}.png")
        shutil.copy2(src_overlay, DEST_OVERLAYS / f"{stem}.jpg")
        copied += 1

    print(f"Copied {copied} triplets (image + mask + overlay)")
    if missing_image:
        print(f"  WARNING: {missing_image} stems had no matching image")
    if missing_mask:
        print(f"  WARNING: {missing_mask} stems had no matching mask")

    return copied


def build_training_json(filtered_stems, description_lookup):
    """Build the training JSON mapping file.

    Each entry:
    {
        "image": "images/<stem>.jpg",
        "conditioning_image": "masks/<stem>.png",
        "text": "<texture transition description>"
    }

    Paths are relative to DEST_ROOT so the dataset is portable.
    """
    entries = []
    missing_desc = 0

    for stem in sorted(filtered_stems):
        # Verify both files exist in destination
        if not (DEST_IMAGES / f"{stem}.jpg").exists():
            continue
        if not (DEST_MASKS / f"{stem}.png").exists():
            continue

        description = description_lookup.get(stem, "")
        if not description:
            missing_desc += 1
            continue

        entries.append({
            "image": str(DEST_IMAGES / f"{stem}.jpg"),
            "conditioning_image": str(DEST_MASKS / f"{stem}.png"),
            "text": description,
        })

    with open(OUTPUT_JSON, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"\nTraining JSON saved to: {OUTPUT_JSON}")
    print(f"  Total entries: {len(entries)}")
    if missing_desc:
        print(f"  WARNING: {missing_desc} stems had no description in processed_bboxes.json (skipped)")

    return entries


def print_sample(entries, n=5):
    """Print a few sample entries for verification."""
    print(f"\n{'='*60}")
    print(f"Sample entries (first {n}):")
    print(f"{'='*60}")
    for entry in entries[:n]:
        print(f"  Image: {entry['image']}")
        print(f"  Mask:  {entry['conditioning_image']}")
        print(f"  Text:  {entry['text']}")
        print(f"  ---")


def print_stats(entries):
    """Print dataset statistics."""
    # Count unique source images
    source_images = set()
    for e in entries:
        # Stem format: <source_image_id>_<x1>_<y1>_<x2>_<y2>
        parts = e["image"].replace("images/", "").replace(".jpg", "")
        # Source image ID is the first part (16 hex chars)
        source_id = parts.split("_")[0]
        source_images.add(source_id)

    # Prompt length stats
    prompt_lengths = [len(e["text"].split()) for e in entries]
    avg_len = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0

    print(f"\n{'='*60}")
    print(f"Dataset Statistics:")
    print(f"{'='*60}")
    print(f"  Total training pairs: {len(entries)}")
    print(f"  Unique source images: {len(source_images)}")
    print(f"  Avg crops per source:  {len(entries)/len(source_images):.1f}")
    print(f"  Avg prompt length:     {avg_len:.1f} words")
    print(f"  Shortest prompt:       {min(prompt_lengths)} words")
    print(f"  Longest prompt:        {max(prompt_lengths)} words")


if __name__ == "__main__":
    print("Phase 1 - Step 1: Preparing ControlNet training data\n")

    # 1. Get filtered stems
    filtered_stems = get_filtered_stems()

    # 2. Build description lookup from processed_bboxes.json
    description_lookup = build_description_lookup(PROCESSED_BBOXES)

    # 3. Copy filtered data to training directory
    num_copied = copy_filtered_data(filtered_stems)

    # 4. Build training JSON
    entries = build_training_json(filtered_stems, description_lookup)

    # 5. Show samples and stats
    print_sample(entries)
    print_stats(entries)

    print(f"\nDone! Training data ready at: {DEST_ROOT}")
