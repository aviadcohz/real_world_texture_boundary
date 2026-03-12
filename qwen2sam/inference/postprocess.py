"""
Post-processing suite for Qwen2SAM mask outputs.

Transforms raw predicted masks into a single clean boundary line:

  Mask A + Mask B
       ↓
  1. Binarize (threshold)
  2. Morphological cleanup (open/close to remove noise)
  3. Soft erosion (shrink masks slightly for clean edges)
  4. Boundary extraction (mask - erode(mask))
  5. Interface extraction (boundary_A ∩ boundary_B with dilation)
  6. Skeletonization (thin to 1-pixel line)
  7. Optional: small component removal
       ↓
  Single clean boundary line (H, W) binary

All operations use OpenCV (not PyTorch) since this runs at inference
time on CPU with no gradient requirements.
"""

import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class PostprocessConfig:
    """Configuration for boundary post-processing."""
    # Binarization
    threshold: float = 0.5

    # Morphological cleanup
    open_kernel_size: int = 3
    open_iterations: int = 1
    close_kernel_size: int = 3
    close_iterations: int = 1

    # Erosion for boundary extraction
    erode_kernel_size: int = 3
    erode_iterations: int = 1

    # Interface extraction
    boundary_dilate_kernel_size: int = 3
    boundary_dilate_iterations: int = 3

    # Skeletonization
    thin_method: str = "zhang_suen"   # "zhang_suen" or "morphological"

    # Small component removal
    min_component_area: int = 50      # Remove connected components smaller than this

    # Output
    output_dtype: str = "uint8"       # "uint8" (0/255) or "float32" (0.0/1.0)


# ===================================================================== #
#  Core morphological operations (OpenCV, no gradients)                   #
# ===================================================================== #

def _to_uint8(mask: np.ndarray) -> np.ndarray:
    """Convert mask to uint8 {0, 255} for OpenCV morphology."""
    if mask.dtype == np.uint8:
        return mask
    return (mask > 0.5).astype(np.uint8) * 255


def _to_float(mask: np.ndarray) -> np.ndarray:
    """Convert uint8 mask to float32 {0.0, 1.0}."""
    return (mask > 127).astype(np.float32)


def binarize(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Binarize a soft mask using a threshold.

    If input appears to be logits (values outside [0,1]), applies sigmoid first.

    Args:
        mask: (H, W) soft mask or logits
        threshold: binarization threshold

    Returns:
        (H, W) uint8 {0, 255}
    """
    if mask.min() < -0.5 or mask.max() > 1.5:
        # Likely logits — apply sigmoid
        mask = 1.0 / (1.0 + np.exp(-mask.astype(np.float64)))

    return ((mask > threshold).astype(np.uint8) * 255)


def morph_open(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """
    Morphological opening: remove small foreground noise.

    opening = dilate(erode(mask))
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(
        _to_uint8(mask), cv2.MORPH_OPEN, kernel, iterations=iterations
    )


def morph_close(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """
    Morphological closing: fill small background holes.

    closing = erode(dilate(mask))
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(
        _to_uint8(mask), cv2.MORPH_CLOSE, kernel, iterations=iterations
    )


def erode(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """Morphological erosion."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.erode(_to_uint8(mask), kernel, iterations=iterations)


def dilate(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """Morphological dilation."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.dilate(_to_uint8(mask), kernel, iterations=iterations)


# ===================================================================== #
#  Boundary and interface extraction                                      #
# ===================================================================== #

def extract_boundary(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """
    Extract 1-pixel boundary of a binary mask.

    boundary = mask - erode(mask)

    Args:
        mask: (H, W) binary mask (uint8 or float)

    Returns:
        (H, W) uint8 {0, 255} boundary pixels
    """
    mask_u8 = _to_uint8(mask)
    eroded = erode(mask_u8, kernel_size, iterations)
    return cv2.subtract(mask_u8, eroded)


def extract_interface(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    dilate_kernel_size: int = 3,
    dilate_iterations: int = 3,
) -> np.ndarray:
    """
    Extract the interface where Mask A meets Mask B.

    1. Extract boundary of each mask
    2. Dilate each boundary
    3. Return intersection: boundary_A AND boundary_B

    For complementary masks, this isolates the seam between them.

    Args:
        mask_a: (H, W) binary mask A
        mask_b: (H, W) binary mask B
        dilate_kernel_size: dilation kernel size
        dilate_iterations: how much to dilate boundaries before intersection

    Returns:
        (H, W) uint8 {0, 255} interface region
    """
    boundary_a = extract_boundary(mask_a)
    boundary_b = extract_boundary(mask_b)

    # Dilate so complementary boundaries overlap
    boundary_a = dilate(boundary_a, dilate_kernel_size, dilate_iterations)
    boundary_b = dilate(boundary_b, dilate_kernel_size, dilate_iterations)

    return cv2.bitwise_and(boundary_a, boundary_b)


# ===================================================================== #
#  Skeletonization                                                        #
# ===================================================================== #

def skeletonize_zhang_suen(mask: np.ndarray) -> np.ndarray:
    """
    Zhang-Suen thinning to reduce a region to a 1-pixel skeleton.

    Uses OpenCV's ximgproc.thinning if available, otherwise falls back
    to a morphological approximation.

    Args:
        mask: (H, W) binary mask uint8 {0, 255}

    Returns:
        (H, W) uint8 {0, 255} skeleton
    """
    mask_u8 = _to_uint8(mask)

    try:
        # OpenCV ximgproc provides Zhang-Suen thinning
        skeleton = cv2.ximgproc.thinning(
            mask_u8, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
        )
        return skeleton
    except AttributeError:
        # Fallback: iterative morphological skeletonization
        return _morphological_skeleton(mask_u8)


def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    """
    Morphological skeletonization fallback.

    Iteratively erodes the mask and accumulates points that would
    disappear after opening.

    skeleton = Union over k of (erode^k(A) - open(erode^k(A)))
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skeleton = np.zeros_like(mask)
    current = mask.copy()

    while True:
        eroded = cv2.erode(current, kernel)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
        temp = cv2.subtract(eroded, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        current = eroded.copy()

        if cv2.countNonZero(current) == 0:
            break

    return skeleton


def skeletonize(
    mask: np.ndarray,
    method: str = "zhang_suen",
) -> np.ndarray:
    """
    Skeletonize a binary mask to a 1-pixel-wide line.

    Args:
        mask: (H, W) binary mask
        method: "zhang_suen" (preferred) or "morphological"

    Returns:
        (H, W) uint8 {0, 255} skeleton
    """
    if method == "zhang_suen":
        return skeletonize_zhang_suen(mask)
    else:
        return _morphological_skeleton(_to_uint8(mask))


# ===================================================================== #
#  Small component removal                                                #
# ===================================================================== #

def remove_small_components(
    mask: np.ndarray,
    min_area: int = 50,
) -> np.ndarray:
    """
    Remove connected components smaller than min_area pixels.

    Cleans up spurious fragments from the boundary/skeleton.

    Args:
        mask: (H, W) binary mask uint8 {0, 255}
        min_area: minimum component area in pixels

    Returns:
        (H, W) uint8 {0, 255} cleaned mask
    """
    mask_u8 = _to_uint8(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    cleaned = np.zeros_like(mask_u8)
    for label in range(1, num_labels):  # skip background (label 0)
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


# ===================================================================== #
#  Full pipeline                                                          #
# ===================================================================== #

def extract_boundary_line(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    config: PostprocessConfig | None = None,
) -> np.ndarray:
    """
    Full post-processing pipeline: two masks → single clean boundary line.

    Steps:
      1. Binarize masks
      2. Morphological cleanup (open + close)
      3. Extract interface region (dilated boundary intersection)
      4. Skeletonize to 1-pixel line
      5. Remove small spurious components

    Args:
        mask_a: (H, W) mask for texture A (logits, soft, or binary)
        mask_b: (H, W) mask for texture B (logits, soft, or binary)
        config: post-processing parameters (uses defaults if None)

    Returns:
        (H, W) binary boundary line (uint8 0/255 or float32 0.0/1.0)
    """
    if config is None:
        config = PostprocessConfig()

    # ---- 1. Binarize ------------------------------------------------ #
    bin_a = binarize(mask_a, config.threshold)
    bin_b = binarize(mask_b, config.threshold)

    # ---- 2. Morphological cleanup ----------------------------------- #
    if config.open_iterations > 0:
        bin_a = morph_open(bin_a, config.open_kernel_size, config.open_iterations)
        bin_b = morph_open(bin_b, config.open_kernel_size, config.open_iterations)
    if config.close_iterations > 0:
        bin_a = morph_close(bin_a, config.close_kernel_size, config.close_iterations)
        bin_b = morph_close(bin_b, config.close_kernel_size, config.close_iterations)

    # ---- 3. Soft erosion (shrink slightly for cleaner edges) -------- #
    if config.erode_iterations > 0:
        bin_a = erode(bin_a, config.erode_kernel_size, config.erode_iterations)
        bin_b = erode(bin_b, config.erode_kernel_size, config.erode_iterations)

    # ---- 4. Interface extraction ------------------------------------ #
    interface = extract_interface(
        bin_a, bin_b,
        dilate_kernel_size=config.boundary_dilate_kernel_size,
        dilate_iterations=config.boundary_dilate_iterations,
    )

    # ---- 5. Skeletonize --------------------------------------------- #
    boundary = skeletonize(interface, method=config.thin_method)

    # ---- 6. Remove small components --------------------------------- #
    if config.min_component_area > 0:
        boundary = remove_small_components(boundary, config.min_component_area)

    # ---- Output format ---------------------------------------------- #
    if config.output_dtype == "float32":
        return _to_float(boundary)
    return boundary
