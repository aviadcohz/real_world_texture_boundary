#!/usr/bin/env python3
"""Visualize refined crops: original vs refined, with mask overlays."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def make_overlay(image_rgb, mask, color=(0, 255, 0), alpha=0.4):
    """Overlay mask boundary on image."""
    overlay = image_rgb.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    # Fill with transparent color
    filled = image_rgb.copy()
    cv2.drawContours(filled, contours, -1, color, -1)
    return cv2.addWeighted(overlay, 1 - alpha * 0.3, filled, alpha * 0.3, 0)


def main():
    refined_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/datasets/ade20k/Detecture_dataset/crops/refined_debug")
    n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    with open(refined_dir / "metadata.json") as f:
        meta = json.load(f)

    # Pick a spread of sizes
    meta_sorted = sorted(meta[:n_show * 2], key=lambda e: e.get("input_size", [0, 0])[0])
    samples = meta_sorted[:n_show]

    rows = []
    for entry in samples:
        crop_name = entry["crop_name"]
        inp_h, inp_w = entry["input_size"]
        out_h, out_w = entry["output_size"]

        # Load original
        orig = cv2.imread(entry["image_path"])
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        orig_mask_a = cv2.imread(entry["mask_a_path"], cv2.IMREAD_GRAYSCALE)
        orig_mask_b = cv2.imread(entry["mask_b_path"], cv2.IMREAD_GRAYSCALE)

        # Load refined
        ref = cv2.imread(entry["refined_image"])
        ref_rgb = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
        ref_mask_a = cv2.imread(entry["refined_mask_a"], cv2.IMREAD_GRAYSCALE)
        ref_mask_b = cv2.imread(entry["refined_mask_b"], cv2.IMREAD_GRAYSCALE)

        # Resize original to match refined for side-by-side (nearest for fair comparison)
        orig_up = cv2.resize(orig_rgb, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        orig_mask_a_up = cv2.resize(orig_mask_a, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        orig_mask_b_up = cv2.resize(orig_mask_b, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        # Create overlays
        orig_overlay = make_overlay(orig_up, orig_mask_a_up, (255, 0, 0))
        orig_overlay = make_overlay(orig_overlay, orig_mask_b_up, (0, 0, 255))

        ref_overlay = make_overlay(ref_rgb, ref_mask_a, (255, 0, 0))
        ref_overlay = make_overlay(ref_overlay, ref_mask_b, (0, 0, 255))

        # Add labels
        label_h = 30
        for img, label in [(orig_overlay, f"Original {inp_w}x{inp_h} (NN upscaled)"),
                           (ref_overlay, f"Refined {out_w}x{out_h} (ESRGAN+smooth)")]:
            cv2.putText(img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Side by side
        pair = np.hstack([orig_overlay, ref_overlay])
        rows.append(pair)

    # Pad all rows to same width, then stack vertically
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.ones((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8) * 200
            r = np.hstack([r, pad])
        padded.append(r)
        padded.append(np.ones((4, max_w, 3), dtype=np.uint8) * 200)
    grid = np.vstack(padded[:-1])

    out_path = refined_dir / "comparison_grid.png"
    Image.fromarray(grid).save(str(out_path))
    print(f"Saved comparison grid ({len(samples)} samples) to {out_path}")


if __name__ == "__main__":
    main()
