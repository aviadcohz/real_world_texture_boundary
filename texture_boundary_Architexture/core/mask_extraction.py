"""Extract binary masks from colored annotation masks by RGB color matching."""

from typing import List, Tuple, Optional

import cv2
import numpy as np


def find_dominant_colors(
    mask_array: np.ndarray,
    tolerance: int = 15,
    min_fraction: float = 0.005,
) -> List[Tuple[np.ndarray, float]]:
    """Cluster mask pixels into dominant colors using iterative greedy assignment.

    Args:
        mask_array: RGB image array (H, W, 3), uint8.
        tolerance: Max per-channel distance to consider same cluster.
        min_fraction: Minimum fraction of total pixels to keep a cluster.

    Returns:
        List of (center_rgb_array, fraction) sorted by fraction descending.
        center_rgb is a (3,) int array.
    """
    h, w = mask_array.shape[:2]
    total = h * w
    pixels = mask_array.reshape(-1, 3).astype(np.int16)

    # Subsample for speed if large
    if len(pixels) > 200_000:
        idx = np.random.default_rng(42).choice(len(pixels), 200_000, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    assigned = np.zeros(len(sample), dtype=bool)
    clusters = []

    for _ in range(50):  # max clusters
        remaining = sample[~assigned]
        if len(remaining) == 0:
            break

        # Pick the most common unassigned pixel as seed
        # Use quantization for speed
        quant = (remaining // 8).astype(np.int32)
        keys = quant[:, 0] * 10000 + quant[:, 1] * 100 + quant[:, 2]
        unique_keys, counts = np.unique(keys, return_counts=True)
        best_key = unique_keys[np.argmax(counts)]
        seed_idx = np.where(keys == best_key)[0][0]
        seed = remaining[seed_idx].astype(np.int16)

        # Find all sample pixels within tolerance of seed
        diffs = np.abs(sample.astype(np.int16) - seed)
        match = np.all(diffs <= tolerance, axis=1) & ~assigned
        if match.sum() == 0:
            break

        center = sample[match].mean(axis=0).astype(np.int16)

        # Re-match with computed center
        diffs2 = np.abs(sample.astype(np.int16) - center)
        match2 = np.all(diffs2 <= tolerance, axis=1) & ~assigned

        cluster_center = sample[match2].mean(axis=0).astype(int)
        fraction = match2.sum() / len(sample)
        assigned |= match2

        if fraction >= min_fraction:
            clusters.append((np.array(cluster_center, dtype=int), fraction))

    # Merge clusters whose centers are too close (JPEG artifacts)
    merge_dist = tolerance * 1.5  # ~22 for tolerance=15
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                ci, fi = clusters[i]
                cj, fj = clusters[j]
                if np.linalg.norm(ci.astype(float) - cj.astype(float)) < merge_dist:
                    # Weighted average center
                    new_center = ((ci * fi + cj * fj) / (fi + fj)).astype(int)
                    clusters[i] = (new_center, fi + fj)
                    clusters.pop(j)
                    merged = True
                    break
            if merged:
                break

    # Count actual pixels (not just sample) for each cluster
    result = []
    for center, _ in clusters:
        diffs = np.abs(pixels - center.astype(np.int16))
        count = np.all(diffs <= tolerance, axis=1).sum()
        frac = count / total
        if frac >= min_fraction:
            result.append((center, frac))

    result.sort(key=lambda x: -x[1])
    return result


def format_dominant_colors(colors: List[Tuple[np.ndarray, float]]) -> str:
    """Format dominant colors for inclusion in the prompt.

    Returns string like:
        [148, 103, 89] (32.5%)
        [90, 225, 120] (18.2%)
    """
    lines = []
    for center, frac in colors:
        rgb = center.tolist()
        lines.append(f"[{rgb[0]}, {rgb[1]}, {rgb[2]}] ({frac*100:.1f}%)")
    return "\n".join(lines)


def match_color_to_dominant(
    query_rgb: List[int],
    dominant_colors: List[Tuple[np.ndarray, float]],
    max_distance: float = 40.0,
) -> Optional[np.ndarray]:
    """Find the closest dominant color to a query RGB.

    Args:
        query_rgb: [R, G, B] values from Qwen's response.
        dominant_colors: Output of find_dominant_colors.
        max_distance: Maximum Euclidean distance to accept a match.

    Returns:
        Matched dominant color center, or None if no match.
    """
    query = np.array(query_rgb, dtype=float)
    best_dist = float("inf")
    best_center = None

    for center, _ in dominant_colors:
        dist = np.linalg.norm(query - center.astype(float))
        if dist < best_dist:
            best_dist = dist
            best_center = center

    if best_dist <= max_distance:
        return best_center
    return None


def quantize_mask(
    mask_array: np.ndarray,
    dominant_colors: List[Tuple[np.ndarray, float]],
) -> np.ndarray:
    """Assign each pixel to its nearest dominant color (by Euclidean distance).

    This prevents similar-but-distinct colors from bleeding into each other,
    which happens with per-channel tolerance when adjacent regions have
    similar shades (e.g., two blues for sky vs sea).

    Args:
        mask_array: RGB image array (H, W, 3), uint8.
        dominant_colors: Output of find_dominant_colors.

    Returns:
        Integer label array (H, W) where each pixel = index into dominant_colors.
        Pixels far from all centers get label -1.
    """
    h, w = mask_array.shape[:2]
    pixels = mask_array.reshape(-1, 3).astype(np.float32)
    centers = np.array([c.astype(np.float32) for c, _ in dominant_colors])

    # Compute squared Euclidean distance to each center
    # pixels: (N, 3), centers: (K, 3) → dists: (N, K)
    dists = np.sum((pixels[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1)
    min_dists = np.sqrt(dists[np.arange(len(pixels)), labels])

    # Mark pixels too far from any center as unassigned
    labels[min_dists > 60.0] = -1

    return labels.reshape(h, w)


def extract_binary_mask(
    mask_array: np.ndarray,
    target_colors: List[List[int]],
    dominant_colors: List[Tuple[np.ndarray, float]],
    tolerance: int = 20,
    quantized_labels: np.ndarray = None,
) -> np.ndarray:
    """Extract binary mask for pixels matching any of the target colors.

    Uses quantized label assignment (nearest-neighbor) to prevent merging
    of adjacent regions with similar colors. Each pixel belongs to exactly
    one dominant color.

    Args:
        mask_array: RGB image array (H, W, 3), uint8.
        target_colors: List of [R, G, B] from Qwen's response.
        dominant_colors: Pre-computed dominant colors for snapping.
        tolerance: Unused (kept for backward compatibility).
        quantized_labels: Pre-computed quantized label map from quantize_mask().
            If None, will be computed on the fly.

    Returns:
        Boolean array (H, W). True where pixel matches a target color.
    """
    h, w = mask_array.shape[:2]
    result = np.zeros((h, w), dtype=bool)

    if quantized_labels is None:
        quantized_labels = quantize_mask(mask_array, dominant_colors)

    for color in target_colors:
        # Snap to nearest dominant color
        matched = match_color_to_dominant(color, dominant_colors)
        if matched is None:
            continue

        # Find which dominant color index this matched to
        for idx, (center, _) in enumerate(dominant_colors):
            if np.array_equal(matched, center):
                result |= (quantized_labels == idx)
                break

    # Morphological closing to fill small JPEG gaps
    if result.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        result = closed.astype(bool)

    return result


def compute_shared_boundary(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    dilation_px: int = 5,
) -> Tuple[int, int, float]:
    """Measure the shared boundary between two masks.

    Dilates mask_b and checks how much of mask_a's perimeter is adjacent to it
    (and vice versa). Returns the boundary length and the ratio relative to the
    smaller mask's perimeter.

    Args:
        mask_a: Boolean array (H, W).
        mask_b: Boolean array (H, W).
        dilation_px: Pixels to dilate when checking adjacency (gap tolerance).

    Returns:
        (shared_boundary_pixels, smaller_perimeter_pixels, shared_ratio)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * dilation_px + 1, 2 * dilation_px + 1))

    # Perimeter of mask_a: pixels in mask_a that border non-mask_a
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    perim_a = cv2.dilate(mask_a.astype(np.uint8), edge_kernel) - mask_a.astype(np.uint8)
    perim_a = (perim_a > 0) & mask_a  # inner edge pixels
    # Actually compute inner perimeter: erode and subtract
    eroded_a = cv2.erode(mask_a.astype(np.uint8), edge_kernel)
    perim_a = mask_a.astype(np.uint8) - eroded_a
    perim_a_count = int(perim_a.sum())

    eroded_b = cv2.erode(mask_b.astype(np.uint8), edge_kernel)
    perim_b = mask_b.astype(np.uint8) - eroded_b
    perim_b_count = int(perim_b.sum())

    smaller_perim = min(perim_a_count, perim_b_count)
    if smaller_perim == 0:
        return 0, 0, 0.0

    # Dilate mask_b, then count how many perimeter pixels of mask_a are adjacent
    dilated_b = cv2.dilate(mask_b.astype(np.uint8), kernel)
    shared_from_a = int((perim_a & dilated_b).sum())

    # Dilate mask_a, then count how many perimeter pixels of mask_b are adjacent
    dilated_a = cv2.dilate(mask_a.astype(np.uint8), kernel)
    shared_from_b = int((perim_b & dilated_a).sum())

    # Use the maximum of both directions (asymmetric shapes)
    shared_boundary = max(shared_from_a, shared_from_b)
    shared_ratio = shared_boundary / smaller_perim

    return shared_boundary, smaller_perim, shared_ratio


def validate_mask_pair(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    min_pixels: int = 100,
    max_overlap_ratio: float = 0.15,
    min_area_fraction: float = 0.01,
    max_area_ratio: float = 20.0,
    min_boundary_ratio: float = 0.10,
    boundary_dilation_px: int = 5,
) -> Tuple[bool, str]:
    """Validate a pair of extracted binary masks.

    Args:
        mask_a: Boolean array (H, W).
        mask_b: Boolean array (H, W).
        min_pixels: Minimum pixel count per mask.
        max_overlap_ratio: Maximum allowed overlap as fraction of smaller mask.
        min_area_fraction: Each mask must cover at least this fraction of the image.
        max_area_ratio: Maximum ratio between larger and smaller mask areas.
        min_boundary_ratio: Minimum fraction of the smaller mask's perimeter that
            must be adjacent to the other mask. Rejects pairs that don't share
            a significant boundary (e.g., masks on opposite sides of the image).
        boundary_dilation_px: Pixel gap tolerance when checking adjacency.

    Returns:
        (is_valid, reason_if_invalid)
    """
    total_pixels = mask_a.size
    count_a = int(mask_a.sum())
    count_b = int(mask_b.sum())

    if count_a < min_pixels:
        return False, f"mask_a too small ({count_a} pixels)"
    if count_b < min_pixels:
        return False, f"mask_b too small ({count_b} pixels)"

    # Each mask must cover a minimum fraction of the image
    frac_a = count_a / total_pixels
    frac_b = count_b / total_pixels
    if frac_a < min_area_fraction:
        return False, f"mask_a too small ({frac_a:.1%} of image)"
    if frac_b < min_area_fraction:
        return False, f"mask_b too small ({frac_b:.1%} of image)"

    # Reject extreme size imbalance (e.g., one mask covers 80%, other covers 4%)
    ratio = max(count_a, count_b) / max(min(count_a, count_b), 1)
    if ratio > max_area_ratio:
        return False, f"area ratio too extreme ({ratio:.1f}x)"

    overlap = (mask_a & mask_b).sum()
    min_count = min(count_a, count_b)
    if min_count > 0 and overlap / min_count > max_overlap_ratio:
        return False, f"masks overlap too much ({overlap/min_count:.1%})"

    # Check that the two masks share a significant boundary
    shared_px, smaller_perim, shared_ratio = compute_shared_boundary(
        mask_a, mask_b, dilation_px=boundary_dilation_px
    )
    if shared_ratio < min_boundary_ratio:
        return False, (
            f"masks don't share a significant boundary "
            f"({shared_ratio:.1%} of perimeter, need {min_boundary_ratio:.0%}; "
            f"{shared_px} shared pixels out of {smaller_perim} perimeter)"
        )

    return True, ""


def sample_oracle_points(
    mask: np.ndarray,
    n_points: int = 4,
    min_separation: int = 30,
    seed: int = 42,
) -> List[List[int]]:
    """Sample interior points from a binary mask using distance transform.

    Points are sampled from areas deep inside the mask (far from edges),
    with minimum separation between them.

    Args:
        mask: Boolean array (H, W).
        n_points: Number of points to sample.
        min_separation: Minimum pixel distance between sampled points.
        seed: Random seed for reproducibility.

    Returns:
        List of [x, y] coordinates (column, row). May return fewer than
        n_points if the mask is too small.
    """
    if not mask.any():
        return []

    # Distance transform — higher values = deeper inside
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)

    # Get candidate points sorted by distance (deepest first)
    ys, xs = np.where(dist > 0)
    if len(ys) == 0:
        return []

    dists = dist[ys, xs]
    order = np.argsort(-dists)
    ys = ys[order]
    xs = xs[order]

    rng = np.random.default_rng(seed)
    points = []

    for y, x in zip(ys, xs):
        # Check minimum separation from already selected points
        too_close = False
        for px, py in points:
            if abs(x - px) + abs(y - py) < min_separation:
                too_close = True
                break
        if not too_close:
            points.append([int(x), int(y)])
        if len(points) >= n_points:
            break

    # If we couldn't get enough points with greedy approach, fill randomly
    if len(points) < n_points:
        remaining_mask = dist > 0
        for px, py in points:
            y_lo = max(0, py - min_separation // 2)
            y_hi = min(mask.shape[0], py + min_separation // 2)
            x_lo = max(0, px - min_separation // 2)
            x_hi = min(mask.shape[1], px + min_separation // 2)
            remaining_mask[y_lo:y_hi, x_lo:x_hi] = False

        ry, rx = np.where(remaining_mask)
        if len(ry) > 0:
            need = n_points - len(points)
            idx = rng.choice(len(ry), min(need, len(ry)), replace=False)
            for i in idx:
                points.append([int(rx[i]), int(ry[i])])

    return points
