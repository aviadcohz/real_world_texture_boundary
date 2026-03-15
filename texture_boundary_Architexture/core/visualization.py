"""Visualization utilities for texture transition mask overlays."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_transition_overlay(
    image: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    texture_a_desc: str,
    texture_b_desc: str,
    alpha: float = 0.4,
    color_a: Tuple[int, int, int] = (255, 50, 50),
    color_b: Tuple[int, int, int] = (50, 100, 255),
    show_labels: bool = True,
) -> Image.Image:
    """Create an overlay showing mask_a (red) and mask_b (blue) on the original image.

    Args:
        image: RGB array (H, W, 3), uint8.
        mask_a: Boolean array (H, W) for texture A.
        mask_b: Boolean array (H, W) for texture B.
        texture_a_desc: Description of texture A.
        texture_b_desc: Description of texture B.
        alpha: Blending strength for mask overlay.
        color_a: RGB color for texture A overlay.
        color_b: RGB color for texture B overlay.
        show_labels: Whether to draw text labels on the overlay.

    Returns:
        PIL Image with overlay and optionally labels.
    """
    overlay = image.astype(np.float32).copy()

    # Apply colored tints
    for mask, color in [(mask_a, color_a), (mask_b, color_b)]:
        if mask.any():
            tint = np.array(color, dtype=np.float32)
            overlay[mask] = (1 - alpha) * overlay[mask] + alpha * tint

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Draw boundary between the two masks (where they are adjacent)
    # Dilate both masks slightly and find intersection = boundary zone
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_a = cv2.dilate(mask_a.astype(np.uint8), kernel, iterations=1).astype(bool)
    dilated_b = cv2.dilate(mask_b.astype(np.uint8), kernel, iterations=1).astype(bool)
    boundary = dilated_a & dilated_b & ~(mask_a & mask_b)
    overlay[boundary] = [255, 255, 0]  # yellow boundary

    # Convert to PIL and optionally add text labels
    pil_img = Image.fromarray(overlay)

    if show_labels:
        draw = ImageDraw.Draw(pil_img)

        # Try to use a readable font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except (IOError, OSError):
            font = ImageFont.load_default()

        # Draw labels with background
        labels = [
            (f"A: {texture_a_desc}", color_a, 5),
            (f"B: {texture_b_desc}", color_b, 22),
        ]
        for text, color, y_pos in labels:
            bbox = draw.textbbox((5, y_pos), text, font=font)
            draw.rectangle([bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1], fill=(0, 0, 0))
            draw.text((5, y_pos), text, fill=color, font=font)

    return pil_img


def save_transition_visualizations(
    image_path: str,
    transitions: List[dict],
    output_dir: str,
    image_id: str,
) -> List[str]:
    """Save overlay visualizations for all transitions of one image.

    Args:
        image_path: Path to original image.
        transitions: List of dicts with keys: mask_a, mask_b, texture_a, texture_b.
        output_dir: Directory to save visualizations.
        image_id: Base name for output files.

    Returns:
        List of saved file paths.
    """
    image = np.array(Image.open(image_path).convert("RGB"))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for i, t in enumerate(transitions):
        overlay = create_transition_overlay(
            image=image,
            mask_a=t["mask_a"],
            mask_b=t["mask_b"],
            texture_a_desc=t["texture_a"],
            texture_b_desc=t["texture_b"],
        )
        out_path = out_dir / f"{image_id}_t{i}.jpg"
        overlay.save(str(out_path), quality=92)
        saved.append(str(out_path))

    return saved
