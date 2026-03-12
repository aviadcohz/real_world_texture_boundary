"""
Mask alignment and coordinate transformation utilities.

Handles the core alignment problem:
  - GT masks are crop-sized (same dimensions as bbox region)
  - SAM operates on full images (1024x1024)
  - Need to embed crop-sized masks into full image canvas
  - Need to transform coordinates between crop/image/SAM spaces
"""

import numpy as np
import torch
import cv2
from typing import Sequence


# --------------------------------------------------------------------------- #
#  Mask alignment                                                              #
# --------------------------------------------------------------------------- #

def embed_mask_in_canvas(
    crop_mask: np.ndarray,
    bbox: Sequence[int],
    canvas_h: int,
    canvas_w: int,
) -> np.ndarray:
    """
    Embed a crop-sized binary mask into a full image canvas of zeros.

    The crop mask is resized to match the bbox dimensions and placed at
    the bbox location within the canvas.

    Args:
        crop_mask: (Hm, Wm) binary mask in crop coordinate space
        bbox: (x1, y1, x2, y2) bounding box in the full image
        canvas_h: full image height
        canvas_w: full image width

    Returns:
        (canvas_h, canvas_w) mask with values placed at bbox region
    """
    x1, y1, x2, y2 = [int(c) for c in bbox]

    # Clamp to canvas bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(canvas_w, x2)
    y2 = min(canvas_h, y2)

    bbox_h = y2 - y1
    bbox_w = x2 - x1

    if bbox_h <= 0 or bbox_w <= 0:
        return np.zeros((canvas_h, canvas_w), dtype=np.float32)

    # Resize crop mask to match bbox dimensions
    if crop_mask.shape[0] != bbox_h or crop_mask.shape[1] != bbox_w:
        resized = cv2.resize(
            crop_mask.astype(np.float32),
            (bbox_w, bbox_h),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        resized = crop_mask.astype(np.float32)

    # Binarize (in case of interpolation artifacts)
    resized = (resized > 0.5).astype(np.float32)

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    canvas[y1:y2, x1:x2] = resized
    return canvas


# --------------------------------------------------------------------------- #
#  Coordinate transforms                                                       #
# --------------------------------------------------------------------------- #

def points_crop_to_image(
    points: list[list[float]],
    bbox: Sequence[int],
    mask_shape: tuple[int, int],
) -> list[list[float]]:
    """
    Convert point coordinates from crop/mask space to full image space.

    Oracle points are sampled from the crop-sized mask. To use them as
    SAM prompts on the full image, we must:
      1. Scale from mask dimensions to bbox dimensions
      2. Offset by bbox origin

    Args:
        points: [[x, y], ...] in crop/mask coordinate space
        bbox: (x1, y1, x2, y2) in full image space
        mask_shape: (H_mask, W_mask) dimensions of the mask the points
                    were sampled from

    Returns:
        [[x, y], ...] in full image space
    """
    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    mask_h, mask_w = mask_shape

    scale_x = bbox_w / mask_w if mask_w > 0 else 1.0
    scale_y = bbox_h / mask_h if mask_h > 0 else 1.0

    return [
        [p[0] * scale_x + x1, p[1] * scale_y + y1]
        for p in points
    ]


def scale_coords_to_sam(
    coords: list[list[float]],
    orig_h: int,
    orig_w: int,
    sam_size: int = 1024,
) -> list[list[float]]:
    """
    Scale [x, y] coordinates from original image space to SAM's 1024x1024 space.

    Args:
        coords: [[x, y], ...] in original image pixel space
        orig_h: original image height
        orig_w: original image width
        sam_size: SAM input resolution (default 1024)

    Returns:
        [[x, y], ...] scaled to SAM space
    """
    sx = sam_size / orig_w if orig_w > 0 else 1.0
    sy = sam_size / orig_h if orig_h > 0 else 1.0

    return [[p[0] * sx, p[1] * sy] for p in coords]


def scale_bbox_to_sam(
    bbox: Sequence[float],
    orig_h: int,
    orig_w: int,
    sam_size: int = 1024,
) -> list[float]:
    """
    Scale bbox [x1, y1, x2, y2] from original image space to SAM space.
    """
    x1, y1, x2, y2 = bbox
    sx = sam_size / orig_w if orig_w > 0 else 1.0
    sy = sam_size / orig_h if orig_h > 0 else 1.0
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


# --------------------------------------------------------------------------- #
#  Image preprocessing for SAM                                                 #
# --------------------------------------------------------------------------- #

# ImageNet normalization (used by SAM 2)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image_for_sam(
    image: np.ndarray,
    sam_size: int = 1024,
) -> torch.Tensor:
    """
    Resize and normalize an image for SAM 2 input.

    Args:
        image: (H, W, 3) uint8 RGB image
        sam_size: target resolution (default 1024)

    Returns:
        (3, sam_size, sam_size) float32 tensor, ImageNet-normalized
    """
    # Resize to SAM input resolution
    if image.shape[0] != sam_size or image.shape[1] != sam_size:
        image = cv2.resize(image, (sam_size, sam_size), interpolation=cv2.INTER_LINEAR)

    # uint8 [0,255] → float32 [0,1]
    image = image.astype(np.float32) / 255.0

    # ImageNet normalization
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    # HWC → CHW
    image = np.transpose(image, (2, 0, 1))

    return torch.from_numpy(image)


def resize_mask(
    mask: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Resize a binary mask using nearest-neighbor interpolation."""
    resized = cv2.resize(
        mask.astype(np.float32),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return (resized > 0.5).astype(np.float32)
