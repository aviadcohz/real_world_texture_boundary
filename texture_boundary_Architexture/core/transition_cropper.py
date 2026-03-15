"""Automatic texture transition crop extraction.

Uses a top-down binary search approach to find the maximum rectangular crops
around texture boundaries. Third-class pixels (not in mask_a or mask_b) are
assigned to the nearest texture by color distance, or left as-is if far from both.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize


def clean_mask(
    mask: np.ndarray,
    min_component_area: int = 50,
) -> np.ndarray:
    """Clean a binary mask by filling holes, removing tiny components, and smoothing edges.

    Args:
        mask: Boolean array (H, W).
        min_component_area: Remove connected components smaller than this.

    Returns:
        Cleaned boolean array (H, W).
    """
    if not mask.any():
        return mask

    mask_u8 = mask.astype(np.uint8)

    # Step 1: Fill holes — find external contours and fill them solid
    filled = np.zeros_like(mask_u8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, 1, -1)

    # Step 2: Remove small connected components
    num_labels, labels = cv2.connectedComponents(filled)
    cleaned = np.zeros_like(filled)
    for i in range(1, num_labels):
        if (labels == i).sum() >= min_component_area:
            cleaned[labels == i] = 1

    # Step 3: Aggressive smoothing — close then open with 5x5 kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)

    # Step 4: Remove any fragments created by morphological ops
    num_labels2, labels2 = cv2.connectedComponents(cleaned)
    if num_labels2 > 2:
        final = np.zeros_like(cleaned)
        for i in range(1, num_labels2):
            if (labels2 == i).sum() >= min_component_area:
                final[labels2 == i] = 1
        cleaned = final

    return cleaned.astype(bool)


def extract_boundary_skeleton(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    dilation_px: int = 3,
) -> np.ndarray:
    """Extract a 1-pixel-wide skeleton of the boundary between two masks.

    Dilates both masks, finds their intersection within the safe zone,
    then skeletonizes to a thin line.

    Args:
        mask_a: Boolean array (H, W).
        mask_b: Boolean array (H, W).
        dilation_px: Pixels to dilate each mask before finding intersection.

    Returns:
        Boolean array (H, W) with 1-pixel-wide boundary skeleton.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1)
    )
    dilated_a = cv2.dilate(mask_a.astype(np.uint8), kernel).astype(bool)
    dilated_b = cv2.dilate(mask_b.astype(np.uint8), kernel).astype(bool)

    safe_zone = mask_a | mask_b
    boundary_zone = dilated_a & dilated_b & safe_zone

    if not boundary_zone.any():
        return np.zeros_like(mask_a, dtype=bool)

    skeleton = skeletonize(boundary_zone)

    # Fallback: if skeletonize produced nothing, use boundary_zone directly
    if not skeleton.any():
        return boundary_zone

    return skeleton


def sample_anchor_points(
    skeleton: np.ndarray,
    spacing: int = 30,
    max_anchors: int = 50,
) -> List[Tuple[int, int]]:
    """Sample anchor points evenly along a boundary skeleton.

    Args:
        skeleton: Boolean array (H, W) — 1px-wide boundary.
        spacing: Sample every N-th pixel.
        max_anchors: Maximum number of anchors to return.

    Returns:
        List of (y, x) anchor coordinates.
    """
    ys, xs = np.where(skeleton)
    if len(ys) == 0:
        return []

    if len(ys) < spacing:
        # Too few pixels — return centroid
        return [(int(ys.mean()), int(xs.mean()))]

    # Subsample at regular intervals
    indices = np.arange(0, len(ys), spacing)
    if len(indices) > max_anchors:
        indices = np.linspace(0, len(ys) - 1, max_anchors, dtype=int)

    return [(int(ys[i]), int(xs[i])) for i in indices]


def binary_search_max_box(
    anchor_y: int,
    anchor_x: int,
    aspect_ratio: Tuple[int, int],
    safe_zone: np.ndarray,
    min_size: int = 32,
    min_purity: float = 1.0,
) -> Optional[Tuple[int, int, int, int]]:
    """Find the maximum bounding box centered on an anchor point.

    Uses binary search on scale to find the largest box (with given aspect
    ratio) where at least min_purity fraction of pixels belong to the safe zone.

    Args:
        anchor_y: Y coordinate of box center.
        anchor_x: X coordinate of box center.
        aspect_ratio: (width_ratio, height_ratio), e.g. (4, 3).
        safe_zone: Boolean array (H, W) — True where pixels are valid.
        min_size: Minimum dimension for a valid crop.
        min_purity: Minimum fraction of pixels that must be in safe zone (0.0-1.0).

    Returns:
        (y1, x1, y2, x2) of the largest valid box, or None if no valid box.
    """
    h, w = safe_zone.shape
    w_ratio, h_ratio = aspect_ratio

    # Maximum scale that fits in the image centered on anchor
    max_scale_candidates = []
    if h_ratio > 0:
        max_scale_candidates.append(2 * anchor_y // h_ratio)
        max_scale_candidates.append(2 * (h - anchor_y) // h_ratio)
    if w_ratio > 0:
        max_scale_candidates.append(2 * anchor_x // w_ratio)
        max_scale_candidates.append(2 * (w - anchor_x) // w_ratio)

    if not max_scale_candidates:
        return None

    max_scale = min(max_scale_candidates)
    min_scale = max(1, (min_size + max(w_ratio, h_ratio) - 1) // max(w_ratio, h_ratio))

    if max_scale < min_scale:
        return None

    best_box = None
    lo, hi = min_scale, max_scale

    while lo <= hi:
        mid = (lo + hi) // 2
        bw = mid * w_ratio
        bh = mid * h_ratio

        y1 = anchor_y - bh // 2
        x1 = anchor_x - bw // 2
        y2 = y1 + bh
        x2 = x1 + bw

        # Clamp to image bounds
        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(h, y2)
        x2 = min(w, x2)

        if y2 - y1 < min_size or x2 - x1 < min_size:
            lo = mid + 1
            continue

        crop_zone = safe_zone[y1:y2, x1:x2]
        purity = crop_zone.sum() / crop_zone.size

        if purity >= min_purity:
            best_box = (y1, x1, y2, x2)
            lo = mid + 1  # Try larger
        else:
            hi = mid - 1  # Shrink

    return best_box


def refine_crop_masks(
    image_crop: np.ndarray,
    mask_a_crop: np.ndarray,
    mask_b_crop: np.ndarray,
    max_distance: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign third-class pixels to nearest texture by color distance.

    For pixels in the crop that belong to neither mask_a nor mask_b,
    compute Euclidean distance to the mean color of each texture.
    Assign to the closer texture if within max_distance, otherwise leave unassigned.

    Args:
        image_crop: RGB array (H, W, 3), uint8.
        mask_a_crop: Boolean array (H, W).
        mask_b_crop: Boolean array (H, W).
        max_distance: Maximum color distance to accept assignment.

    Returns:
        (refined_mask_a, refined_mask_b) as boolean arrays.
    """
    third_class = ~(mask_a_crop | mask_b_crop)
    if not third_class.any():
        return mask_a_crop, mask_b_crop

    # Compute mean color of each texture region
    pixels = image_crop.astype(np.float32)

    if not mask_a_crop.any() or not mask_b_crop.any():
        return mask_a_crop, mask_b_crop

    mean_a = pixels[mask_a_crop].mean(axis=0)  # (3,)
    mean_b = pixels[mask_b_crop].mean(axis=0)  # (3,)

    # Get third-class pixel colors
    third_pixels = pixels[third_class]  # (N, 3)

    # Euclidean distance to each texture mean
    dist_a = np.sqrt(np.sum((third_pixels - mean_a) ** 2, axis=1))
    dist_b = np.sqrt(np.sum((third_pixels - mean_b) ** 2, axis=1))

    # Assign to closer texture if within threshold
    assign_a = (dist_a <= dist_b) & (dist_a <= max_distance)
    assign_b = (dist_b < dist_a) & (dist_b <= max_distance)

    refined_a = mask_a_crop.copy()
    refined_b = mask_b_crop.copy()

    third_ys, third_xs = np.where(third_class)
    refined_a[third_ys[assign_a], third_xs[assign_a]] = True
    refined_b[third_ys[assign_b], third_xs[assign_b]] = True

    return refined_a, refined_b


def compute_crop_score(
    box: Tuple[int, int, int, int],
    skeleton: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    min_balance: float = 0.20,
) -> Tuple[float, float, float]:
    """Score a crop box by boundary density and balance.

    Args:
        box: (y1, x1, y2, x2) crop coordinates.
        skeleton: Boolean array (H, W) — boundary skeleton.
        mask_a: Boolean array (H, W).
        mask_b: Boolean array (H, W).
        min_balance: Minimum fraction each mask must occupy.

    Returns:
        (score, frac_a, frac_b). Score is -1.0 if balance check fails.
    """
    y1, x1, y2, x2 = box
    area = (y2 - y1) * (x2 - x1)
    if area == 0:
        return (-1.0, 0.0, 0.0)

    crop_a = mask_a[y1:y2, x1:x2]
    crop_b = mask_b[y1:y2, x1:x2]

    frac_a = float(crop_a.sum()) / area
    frac_b = float(crop_b.sum()) / area

    if min(frac_a, frac_b) < min_balance:
        return (-1.0, frac_a, frac_b)

    boundary_count = float(skeleton[y1:y2, x1:x2].sum())
    score = boundary_count / area

    return (score, frac_a, frac_b)


def nms_boxes(
    boxes: List[Tuple[int, int, int, int]],
    scores: List[float],
    iou_threshold: float = 0.3,
) -> List[int]:
    """Non-maximum suppression for axis-aligned boxes.

    Args:
        boxes: List of (y1, x1, y2, x2).
        scores: Corresponding scores.
        iou_threshold: Suppress boxes with IoU above this.

    Returns:
        List of indices to keep.
    """
    if not boxes:
        return []

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    suppressed = set()

    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        y1_i, x1_i, y2_i, x2_i = boxes[i]
        area_i = (y2_i - y1_i) * (x2_i - x1_i)

        for j in order:
            if j in suppressed or j == i:
                continue
            y1_j, x1_j, y2_j, x2_j = boxes[j]
            area_j = (y2_j - y1_j) * (x2_j - x1_j)

            # Intersection
            iy1 = max(y1_i, y1_j)
            ix1 = max(x1_i, x1_j)
            iy2 = min(y2_i, y2_j)
            ix2 = min(x2_i, x2_j)
            inter = max(0, iy2 - iy1) * max(0, ix2 - ix1)

            union = area_i + area_j - inter
            if union > 0 and inter / union > iou_threshold:
                suppressed.add(j)

    return keep


def find_best_crops(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    image: np.ndarray = None,
    aspect_ratios: List[Tuple[int, int]] = None,
    max_crops: int = 5,
    anchor_spacing: int = 30,
    min_balance: float = 0.20,
    min_size: int = 32,
    dilation_px: int = 3,
    iou_threshold: float = 0.3,
    min_purity: float = 0.90,
    max_color_distance: float = 40.0,
) -> List[Tuple[int, int, int, int, float, float, float]]:
    """Find the best texture transition crops from a mask pair.

    Two-stage approach:
    1. Binary search with relaxed purity (default 90%) to find large candidate boxes.
    2. Refine: assign third-class pixels to nearest texture by color distance.
       Pixels far from both textures are left unassigned.

    Args:
        mask_a: Boolean array (H, W) for texture A.
        mask_b: Boolean array (H, W) for texture B.
        image: RGB array (H, W, 3) uint8. Required for color-based refinement.
            If None, no refinement is performed (pure safe_zone only).
        aspect_ratios: List of (w_ratio, h_ratio) tuples.
        max_crops: Maximum number of crops to return.
        anchor_spacing: Pixel spacing between anchor points on skeleton.
        min_balance: Minimum fraction each mask must cover in crop.
        min_size: Minimum crop dimension in pixels.
        dilation_px: Dilation for boundary extraction.
        iou_threshold: NMS IoU threshold.
        min_purity: Minimum fraction of crop in safe zone (mask_a | mask_b).
            1.0 = strict, 0.90 = allow 10% third-class pixels.
        max_color_distance: Max Euclidean RGB distance for assigning
            third-class pixels to a texture. Beyond this, pixel stays unassigned.

    Returns:
        List of (y1, x1, y2, x2, score, frac_a, frac_b) tuples,
        sorted by score descending. frac_a/frac_b reflect refined masks
        when image is provided. Empty list if no valid crops found.
    """
    if aspect_ratios is None:
        aspect_ratios = [(1, 1), (4, 3), (3, 4)]

    # Step 0: Clean masks — fill holes, remove tiny components, smooth edges
    mask_a = clean_mask(mask_a)
    mask_b = clean_mask(mask_b)

    # Step 1: Extract boundary skeleton
    skeleton = extract_boundary_skeleton(mask_a, mask_b, dilation_px)
    if not skeleton.any():
        return []

    # Step 2: Sample anchor points
    anchors = sample_anchor_points(skeleton, anchor_spacing)
    if not anchors:
        return []

    # Step 3: Safe zone (union of both masks)
    safe_zone = mask_a | mask_b

    # Step 4: Binary search for each anchor × aspect ratio
    all_boxes = []
    all_scores = []
    all_balances = []

    for anchor_y, anchor_x in anchors:
        for ratio in aspect_ratios:
            box = binary_search_max_box(
                anchor_y, anchor_x, ratio, safe_zone, min_size,
                min_purity=min_purity,
            )
            if box is None:
                continue

            # If image provided, compute score on refined masks
            if image is not None:
                y1, x1, y2, x2 = box
                crop_img = image[y1:y2, x1:x2]
                crop_a = mask_a[y1:y2, x1:x2]
                crop_b = mask_b[y1:y2, x1:x2]
                ref_a, ref_b = refine_crop_masks(
                    crop_img, crop_a, crop_b, max_color_distance
                )
                # Compute balance on refined masks
                area = (y2 - y1) * (x2 - x1)
                frac_a = float(ref_a.sum()) / area
                frac_b = float(ref_b.sum()) / area
                if min(frac_a, frac_b) < min_balance:
                    continue
                boundary_count = float(skeleton[y1:y2, x1:x2].sum())
                score = boundary_count / area
            else:
                score, frac_a, frac_b = compute_crop_score(
                    box, skeleton, mask_a, mask_b, min_balance
                )
                if score < 0:
                    continue

            all_boxes.append(box)
            all_scores.append(score)
            all_balances.append((frac_a, frac_b))

    if not all_boxes:
        return []

    # Step 5: NMS
    keep_indices = nms_boxes(all_boxes, all_scores, iou_threshold)

    # Step 6: Return top crops
    results = []
    for idx in keep_indices[:max_crops]:
        y1, x1, y2, x2 = all_boxes[idx]
        score = all_scores[idx]
        frac_a, frac_b = all_balances[idx]
        results.append((y1, x1, y2, x2, score, frac_a, frac_b))

    return results
