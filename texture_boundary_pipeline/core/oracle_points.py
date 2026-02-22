"""
Oracle point sampling for SAM training.

For each pair of texture masks (mask_a, mask_b) produced by Sa2VA,
samples interior point prompts that allow SAM to distinguish between
the two textures.

Uses Euclidean distance transform to find pixels that are maximally
far from region boundaries — these are the most reliable SAM prompts.

Coordinates are returned as [x, y] in the 1024×1024 mask/crop space.
"""

import numpy as np
from typing import List, Optional, Dict
from scipy.ndimage import distance_transform_edt


def _sample_interior_points(
    mask: np.ndarray,
    n: int = 2,
    min_separation: float = 50.0
) -> Optional[List[List[int]]]:
    """
    Sample n points from the foreground (255) region of a binary mask.

    Picks the most interior pixels using distance transform, ensuring
    each successive point is spatially separated from prior ones.

    Args:
        mask: Binary mask (H×W, values 0 or 255)
        n: Number of points to sample
        min_separation: Minimum pixel distance between sampled points

    Returns:
        List of [x, y] coordinates, or None if region is too small.
    """
    foreground = (mask == 255).astype(np.float32)
    if foreground.sum() < 10:
        return None

    dist = distance_transform_edt(foreground)
    points = []

    for _ in range(n):
        if dist.max() == 0:
            break

        row, col = np.unravel_index(dist.argmax(), dist.shape)
        points.append([int(col), int(row)])  # [x, y] convention

        # Suppress neighbourhood so next point is well-separated
        rows, cols = np.ogrid[:dist.shape[0], :dist.shape[1]]
        exclusion = ((rows - row) ** 2 + (cols - col) ** 2) <= (min_separation ** 2)
        dist[exclusion] = 0

    return points if len(points) == n else None


def sample_oracle_points(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    n_points: int = 2,
    min_separation: float = 50.0
) -> Optional[Dict[str, List[List[int]]]]:
    """
    Sample oracle point prompts for both texture masks.

    Args:
        mask_a: Binary mask for texture A (H×W, values 0 or 255)
        mask_b: Binary mask for texture B (H×W, values 0 or 255)
        n_points: Number of points to sample per mask (default 2)
        min_separation: Minimum pixel distance between points within a mask

    Returns:
        Dict with keys 'point_prompt_mask_a' and 'point_prompt_mask_b',
        each containing a list of [x, y] coordinates.
        Returns None if either mask region is too small to sample from.
    """
    pts_a = _sample_interior_points(mask_a, n=n_points, min_separation=min_separation)
    pts_b = _sample_interior_points(mask_b, n=n_points, min_separation=min_separation)

    if pts_a is None or pts_b is None:
        return None

    return {
        'point_prompt_mask_a': pts_a,
        'point_prompt_mask_b': pts_b,
    }
