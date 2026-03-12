"""Detect and filter near-duplicate texture transitions."""

from typing import List, Tuple

import numpy as np


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks."""
    intersection = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def is_duplicate_transition(
    new_mask_a: np.ndarray,
    new_mask_b: np.ndarray,
    existing_transitions: List[Tuple[np.ndarray, np.ndarray]],
    iou_threshold: float = 0.5,
) -> bool:
    """Check if a new transition duplicates any existing one.

    A transition is considered duplicate if BOTH mask_a and mask_b have
    high IoU with an existing transition's masks (in either order).

    Args:
        new_mask_a: Binary mask for new texture A.
        new_mask_b: Binary mask for new texture B.
        existing_transitions: List of (mask_a, mask_b) tuples.
        iou_threshold: IoU threshold above which masks are considered same.

    Returns:
        True if this transition is a duplicate.
    """
    for ex_a, ex_b in existing_transitions:
        # Check A↔A and B↔B
        if (mask_iou(new_mask_a, ex_a) > iou_threshold and
                mask_iou(new_mask_b, ex_b) > iou_threshold):
            return True
        # Check A↔B and B↔A (swapped textures)
        if (mask_iou(new_mask_a, ex_b) > iou_threshold and
                mask_iou(new_mask_b, ex_a) > iou_threshold):
            return True

    return False
