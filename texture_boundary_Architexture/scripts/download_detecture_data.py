#!/usr/bin/env python3
"""Download images, colored annotation masks, and overlays from the Detecture review site."""

import argparse
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "https://scientific-computing-user.github.io/ade20k-texture-miner-site/review"


def download_file(url: str, dst: Path, retries: int = 3) -> bool:
    if dst.exists() and dst.stat().st_size > 0:
        return True
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, str(dst))
            return True
        except Exception as e:
            if attempt == retries - 1:
                print(f"  FAILED: {dst.name} — {e}", file=sys.stderr)
                return False
    return False


def download_entry(entry: dict, output_dir: Path, skip_overlays: bool = False) -> dict:
    """Download all assets for one entry. Returns dict with success status."""
    image_id = entry["image_id"]
    results = {"image_id": image_id, "original": False, "mask": False, "overlay": False}

    # Original image
    orig_url = f"{BASE_URL}/originals/{image_id}.jpg"
    orig_dst = output_dir / "originals" / f"{image_id}.jpg"
    results["original"] = download_file(orig_url, orig_dst)

    # Colored annotation mask
    mask_url = f"{BASE_URL}/masks/{image_id}.jpg"
    mask_dst = output_dir / "masks" / f"{image_id}.jpg"
    results["mask"] = download_file(mask_url, mask_dst)

    # Overlay
    if not skip_overlays:
        overlay_url = f"{BASE_URL}/overlays/{image_id}.jpg"
        overlay_dst = output_dir / "overlays" / f"{image_id}.jpg"
        results["overlay"] = download_file(overlay_url, overlay_dst)
    else:
        results["overlay"] = True  # skipped = ok

    return results


def main():
    parser = argparse.ArgumentParser(description="Download Detecture review data")
    parser.add_argument("--output-dir", type=str, default="/home/aviad/detecture_data")
    parser.add_argument("--status", type=str, default="selected",
                        choices=["all", "selected", "borderline", "rejected"],
                        help="Which images to download by status")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--skip-overlays", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Create directories
    for subdir in ["originals", "masks", "overlays"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Download manifest
    manifest_url = f"{BASE_URL}/data.json"
    manifest_path = output_dir / "data.json"
    print(f"Downloading manifest from {manifest_url}...")
    urllib.request.urlretrieve(manifest_url, str(manifest_path))

    with open(manifest_path) as f:
        all_entries = json.load(f)
    print(f"Total entries in manifest: {len(all_entries)}")

    # Filter by status
    if args.status == "all":
        entries = all_entries
    else:
        entries = [e for e in all_entries if e.get("status") == args.status]
    print(f"Entries with status '{args.status}': {len(entries)}")

    # Save filtered manifest
    filtered_path = output_dir / f"manifest_{args.status}.json"
    with open(filtered_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Saved filtered manifest to {filtered_path}")

    # Download assets in parallel
    print(f"\nDownloading assets with {args.workers} workers...")
    success = {"original": 0, "mask": 0, "overlay": 0}
    failed = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_entry, entry, output_dir, args.skip_overlays): entry
            for entry in entries
        }
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            for key in ["original", "mask", "overlay"]:
                if result[key]:
                    success[key] += 1
            if not all(result[v] for v in ["original", "mask"]):
                failed.append(result["image_id"])
            if i % 50 == 0 or i == len(entries):
                print(f"  Progress: {i}/{len(entries)}")

    # Summary
    print(f"\n{'='*50}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*50}")
    print(f"Originals: {success['original']}/{len(entries)}")
    print(f"Masks:     {success['mask']}/{len(entries)}")
    print(f"Overlays:  {success['overlay']}/{len(entries)}")
    if failed:
        print(f"\nFailed ({len(failed)}): {failed[:10]}{'...' if len(failed) > 10 else ''}")
    print(f"\nData saved to: {output_dir}")


if __name__ == "__main__":
    main()
