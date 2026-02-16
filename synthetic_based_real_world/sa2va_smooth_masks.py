"""
SA2VA mask cleaning and boundary smoothing.

Pipeline per image:
1. Run SA2VA segmentation for both textures → raw noisy masks
2. Clean masks: morphological close, remove small components, fill holes
3. Smooth mask boundaries: extract contour → periodic B-spline → redraw
4. Extract boundary between the two smoothed masks
5. Visualize all stages side by side
"""

import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import splprep, splev
from scipy.ndimage import binary_fill_holes

sys.path.insert(0, str(Path("/home/aviad/real_world_texture_boundary/texture_boundary_pipeline")))

from models.sa2va_vlm import Sa2VAModel
from core.sa2va_boundaries import extract_morphological_boundary

DATA_ROOT = Path("/datasets/google_landmarks_v2_scale/real_world_texture_boundary/results/controlnet_data_for_training")
OUTPUT_DIR = Path("/home/aviad/real_world_texture_boundary/sytatic_based_real_world/sa2va_comparison")
TRAINING_JSON = DATA_ROOT / "training_pairs.json"


def load_descriptions():
    with open(TRAINING_JSON) as f:
        pairs = json.load(f)
    return {Path(p["image"]).stem: p["text"] for p in pairs}


def parse_description(desc):
    if " to " in desc:
        a, b = desc.split(" to ", 1)
        return a.strip(), b.strip()
    return desc, desc


# ── Stage 1: Clean mask ──────────────────────────────────────────────────────

def clean_mask(mask, close_k=25, open_k=11, collapse_ratio=0.50):
    """Turn a noisy SA2VA mask into a solid filled region.

    Primary path (morphological):
        1. Binarize → close → open → keep largest component → fill holes

    Fallback (blur-vote) — activated when morphological cleaning loses
    more than (1 - collapse_ratio) of the original white pixels:
        1. Gaussian-blur the raw mask (large kernel) — acts as a soft vote
        2. Threshold the blurred image
        3. Close → keep largest → fill holes

    This handles checkerboard / scattered masks that collapse under
    standard morphological ops.
    """
    binary = (mask > 127).astype(np.uint8)
    raw_area = int(binary.sum())

    if raw_area == 0:
        return mask  # nothing to clean

    result = _morphological_clean(binary, close_k, open_k)
    clean_area = int((result > 127).astype(np.uint8).sum())

    # If we kept enough area, we're done
    if clean_area >= collapse_ratio * raw_area:
        print(f"    Morph clean OK ({clean_area}/{raw_area} px = {100*clean_area/raw_area:.0f}%)")
        return result

    # Fallback: blur-vote approach for scattered / checkerboard masks
    print(f"    Morph cleaning collapsed mask ({clean_area}/{raw_area} px = "
          f"{100*clean_area/raw_area:.0f}%) — using blur-vote fallback")
    return _blur_vote_clean(mask, binary, close_k)


def _morphological_clean(binary, close_k, open_k):
    """Standard close → open → largest component → fill holes."""
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, k_open)
    return _keep_largest_filled(morphed)


def _blur_vote_clean(mask, binary, close_k):
    """Gaussian blur as a pixel-voting mechanism, then threshold.

    Every pixel gets the average of its neighbourhood.  Where many nearby
    pixels were white the blurred value is high → those regions survive.
    """
    # Large blur kernel — roughly same scale as closing kernel
    blur_k = close_k * 2 + 1  # must be odd
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (blur_k, blur_k), 0)

    # Threshold: keep pixels where the local vote exceeds 15%
    voted = (blurred > 0.15 * 255).astype(np.uint8)

    # Light closing to tidy up
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    voted = cv2.morphologyEx(voted, cv2.MORPH_CLOSE, k)

    return _keep_largest_filled(voted)


def _keep_largest_filled(binary):
    """Keep largest connected component and fill interior holes."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return binary * 255

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    largest_mask = (labels == largest).astype(np.uint8)

    filled = binary_fill_holes(largest_mask).astype(np.uint8)
    return filled * 255


# ── Stage 1b: Ensure masks are complementary ─────────────────────────────────

def _ensure_complementary(mask_a, mask_b, name_a="A", name_b="B"):
    """Make sure the two texture masks partition the image without heavy overlap.

    Cases handled:
      1. One mask collapsed (< 5% of image)  → derive from NOT(other)
      2. Heavy overlap (IoU of foreground > 0.5) → keep the *smaller* mask
         (more likely to be the focused object), derive larger = NOT(smaller)
      3. Both OK → return as-is
    """
    bin_a = (mask_a > 127).astype(np.uint8)
    bin_b = (mask_b > 127).astype(np.uint8)
    area_a = int(bin_a.sum())
    area_b = int(bin_b.sum())
    total = mask_a.shape[0] * mask_a.shape[1]
    min_area = int(0.05 * total)

    # Case 1: collapse
    if area_a < min_area and area_b >= min_area:
        print(f"    '{name_a}' collapsed ({area_a} px) — deriving from NOT('{name_b}')")
        return (255 - mask_b), mask_b
    if area_b < min_area and area_a >= min_area:
        print(f"    '{name_b}' collapsed ({area_b} px) — deriving from NOT('{name_a}')")
        return mask_a, (255 - mask_a)

    # Case 2: heavy overlap — one mask swallowed the other
    intersection = int((bin_a & bin_b).sum())
    union = int((bin_a | bin_b).sum())
    iou = intersection / union if union > 0 else 0
    overlap_on_smaller = intersection / min(area_a, area_b) if min(area_a, area_b) > 0 else 0

    print(f"    Mask overlap: IoU={iou:.2f}, "
          f"overlap/smaller={overlap_on_smaller:.2f}, "
          f"areas: {name_a}={area_a} ({100*area_a/total:.0f}%), "
          f"{name_b}={area_b} ({100*area_b/total:.0f}%)")

    if overlap_on_smaller > 0.70:
        # The smaller mask is more "focused" — keep it, derive the other
        if area_a <= area_b:
            print(f"    High overlap — keeping '{name_a}' (smaller), "
                  f"deriving '{name_b}' = NOT('{name_a}')")
            return mask_a, (255 - mask_a)
        else:
            print(f"    High overlap — keeping '{name_b}' (smaller), "
                  f"deriving '{name_a}' = NOT('{name_b}')")
            return (255 - mask_b), mask_b

    return mask_a, mask_b


# ── Stage 2: Smooth boundary ─────────────────────────────────────────────────

def smooth_mask_boundary(mask, smoothing=0.005, n_pts=600):
    """Smooth the outline of a binary mask with a periodic B-spline.

    1. Extract the largest external contour
    2. Fit a periodic cubic B-spline (controls smoothness via `s`)
    3. Resample the spline at `n_pts` evenly-spaced points
    4. Fill the smoothed contour → new mask
    """
    binary = (mask > 127).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask

    contour = max(contours, key=cv2.contourArea).squeeze()  # (N, 2)
    if contour.ndim != 2 or len(contour) < 20:
        return mask

    # Subsample for numerical stability
    max_input = 2000
    if len(contour) > max_input:
        idx = np.linspace(0, len(contour) - 1, max_input, dtype=int)
        contour = contour[idx]

    x = contour[:, 0].astype(float)
    y = contour[:, 1].astype(float)

    # Smoothing parameter scales with perimeter length
    perimeter = cv2.arcLength(contour.reshape(-1, 1, 2).astype(np.float32), closed=True)
    s = smoothing * perimeter * len(contour)

    try:
        tck, _ = splprep([x, y], s=s, per=True, k=3)
        u_new = np.linspace(0, 1, n_pts)
        x_s, y_s = splev(u_new, tck)
    except Exception as e:
        print(f"    spline fit failed: {e}")
        return mask

    smooth_contour = np.column_stack([x_s, y_s]).astype(np.int32)
    out = np.zeros_like(binary)
    cv2.fillPoly(out, [smooth_contour], 1)
    return out * 255


# ── Pipeline per image ────────────────────────────────────────────────────────

def run_pipeline(sa2va_model, stem, desc):
    img_path = DATA_ROOT / "images" / f"{stem}.jpg"
    mask_path = DATA_ROOT / "masks" / f"{stem}.png"
    if not img_path.exists() or not mask_path.exists():
        print(f"  SKIP: missing files for {stem}")
        return None

    print(f"\n  Processing: {stem}")
    print(f"  Description: {desc}")

    existing_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    texture_a, texture_b = parse_description(desc)
    print(f"  Texture A: '{texture_a}'")
    print(f"  Texture B: '{texture_b}'")

    # 1. SA2VA raw masks
    print("  [1/3] SA2VA segmentation...")
    raw_a = sa2va_model.segment_texture(str(img_path), texture_a)
    raw_b = sa2va_model.segment_texture(str(img_path), texture_b)

    # 2. Clean
    print("  [2/3] Cleaning masks...")
    print(f"    Cleaning mask A ('{texture_a}')...")
    clean_a = clean_mask(raw_a)
    print(f"    Cleaning mask B ('{texture_b}')...")
    clean_b = clean_mask(raw_b)

    # Ensure the two masks are complementary (they partition the image).
    # Problem cases:
    #   - One mask collapsed to near-zero  → derive from NOT(other)
    #   - One mask expanded to cover everything (fill_holes swallowed the
    #     other region) → derive from NOT(other)
    #   - Both masks overlap heavily → keep the smaller (more focused) one,
    #     derive the larger as NOT(smaller)
    clean_a, clean_b = _ensure_complementary(clean_a, clean_b, texture_a, texture_b)

    # 3. Smooth
    print("  [3/3] Smoothing boundaries...")
    smooth_a = smooth_mask_boundary(clean_a)
    smooth_b = smooth_mask_boundary(clean_b)

    # Boundaries at each stage
    raw_bnd = extract_morphological_boundary(raw_a, raw_b, thickness=3)
    clean_bnd = extract_morphological_boundary(clean_a, clean_b, thickness=3)
    smooth_bnd = extract_morphological_boundary(smooth_a, smooth_b, thickness=3)

    return dict(
        stem=stem, desc=desc,
        texture_a=texture_a, texture_b=texture_b,
        raw_a=raw_a, raw_b=raw_b,
        clean_a=clean_a, clean_b=clean_b,
        smooth_a=smooth_a, smooth_b=smooth_b,
        raw_bnd=raw_bnd, clean_bnd=clean_bnd, smooth_bnd=smooth_bnd,
        existing_mask=existing_mask,
    )


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize(result):
    """3-row × 4-col figure showing every stage.

    Row 0  (Raw):      Image | raw mask_a | raw mask_b | raw boundary
    Row 1  (Cleaned):  existing mask | clean_a | clean_b | clean boundary
    Row 2  (Smoothed): smooth_a | smooth_b | smooth boundary | overlay
    """
    stem = result["stem"]
    img = cv2.imread(str(DATA_ROOT / "images" / f"{stem}.jpg"))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    fig.suptitle(
        f"SA2VA Clean → Smooth pipeline: {stem}\n\"{result['desc']}\"",
        fontsize=13,
    )

    # ── Row 0: Raw ──
    axes[0, 0].imshow(img_rgb);              axes[0, 0].set_title("Original Image")
    axes[0, 1].imshow(result["raw_a"], cmap="gray"); axes[0, 1].set_title(f"Raw: '{result['texture_a']}'")
    axes[0, 2].imshow(result["raw_b"], cmap="gray"); axes[0, 2].set_title(f"Raw: '{result['texture_b']}'")
    axes[0, 3].imshow(result["raw_bnd"], cmap="gray"); axes[0, 3].set_title("Raw Boundary")

    # ── Row 1: Cleaned ──
    axes[1, 0].imshow(result["existing_mask"], cmap="gray"); axes[1, 0].set_title("Existing Pipeline Mask")
    axes[1, 1].imshow(result["clean_a"], cmap="gray"); axes[1, 1].set_title(f"Cleaned: '{result['texture_a']}'")
    axes[1, 2].imshow(result["clean_b"], cmap="gray"); axes[1, 2].set_title(f"Cleaned: '{result['texture_b']}'")
    axes[1, 3].imshow(result["clean_bnd"], cmap="gray"); axes[1, 3].set_title("Cleaned Boundary")

    # ── Row 2: Smoothed + overlay ──
    axes[2, 0].imshow(result["smooth_a"], cmap="gray"); axes[2, 0].set_title(f"Smoothed: '{result['texture_a']}'")
    axes[2, 1].imshow(result["smooth_b"], cmap="gray"); axes[2, 1].set_title(f"Smoothed: '{result['texture_b']}'")
    axes[2, 2].imshow(result["smooth_bnd"], cmap="gray"); axes[2, 2].set_title("Smoothed Boundary")

    # Overlay on image: red = existing, green = smoothed SA2VA
    overlay = img.copy()
    ex_bin = result["existing_mask"] > 30
    overlay[ex_bin, 2] = 255;  overlay[ex_bin, 0] //= 2;  overlay[ex_bin, 1] //= 2
    sm_bin = result["smooth_bnd"] > 30
    overlay[sm_bin, 1] = 255;  overlay[sm_bin, 0] //= 2;  overlay[sm_bin, 2] //= 2

    axes[2, 3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2, 3].set_title("Overlay (red=existing, green=smoothed)")

    for row in axes:
        for ax in row:
            ax.axis("off")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{stem}_smoothed.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    default_stems = [
        "0000ae056149919f_0_228_778_578",
        "0000e69998d37a98_168_138_353_278",
        "000d59f6199efc33_238_0_608_455",
        "000db83105d64f5d_0_268_338_328",
    ]
    stems = sys.argv[1:] if len(sys.argv) > 1 else default_stems

    desc_lookup = load_descriptions()

    print("Loading SA2VA model...")
    sa2va_model = Sa2VAModel(device="cuda", lazy_load=True)

    results = []
    for stem in stems:
        desc = desc_lookup.get(stem)
        if desc is None:
            print(f"  SKIP: no description for {stem}")
            continue
        result = run_pipeline(sa2va_model, stem, desc)
        if result is not None:
            out_path = visualize(result)
            results.append((stem, out_path))

    sa2va_model.unload()

    print(f"\n{'='*60}")
    print(f"Done! Generated {len(results)} smoothed comparisons:")
    for stem, path in results:
        print(f"  {stem} → {path}")


if __name__ == "__main__":
    main()
