"""
CLI entry point for Qwen2SAM inference.

Processes single images or directories, produces:
  - Text description of the texture transition
  - Two complementary segmentation masks
  - A clean 1-pixel boundary line
  - Visualization overlays

Usage:
  # Single image:
  python -m qwen2sam.inference.run_inference \
      --config configs/inference.yaml \
      --checkpoint checkpoints/phase3/best.pt \
      --image path/to/image.jpg \
      --output_dir results/

  # Directory of images:
  python -m qwen2sam.inference.run_inference \
      --config configs/inference.yaml \
      --checkpoint checkpoints/phase3/best.pt \
      --image_dir path/to/images/ \
      --output_dir results/
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from qwen2sam.inference.pipeline import Qwen2SAMPipeline, PredictionResult
from qwen2sam.inference.postprocess import (
    PostprocessConfig,
    extract_boundary_line,
)


# ===================================================================== #
#  Visualization                                                          #
# ===================================================================== #

def create_mask_overlay(
    image: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    alpha: float = 0.4,
    color_a: tuple = (255, 50, 50),
    color_b: tuple = (50, 50, 255),
) -> np.ndarray:
    """
    Overlay two masks on the image with distinct colors.

    Args:
        image: (H, W, 3) uint8 RGB
        mask_a: (H, W) binary mask A
        mask_b: (H, W) binary mask B
        alpha: overlay opacity
        color_a/b: RGB tuples for each mask

    Returns:
        (H, W, 3) uint8 overlay image
    """
    overlay = image.copy()

    mask_a_bool = mask_a > 0.5 if mask_a.dtype != np.uint8 else mask_a > 127
    mask_b_bool = mask_b > 0.5 if mask_b.dtype != np.uint8 else mask_b > 127

    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_a_bool,
            np.clip(image[:, :, c] * (1 - alpha) + color_a[c] * alpha, 0, 255),
            overlay[:, :, c],
        )
        overlay[:, :, c] = np.where(
            mask_b_bool,
            np.clip(image[:, :, c] * (1 - alpha) + color_b[c] * alpha, 0, 255),
            overlay[:, :, c],
        )

    return overlay.astype(np.uint8)


def draw_boundary_on_image(
    image: np.ndarray,
    boundary: np.ndarray,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw boundary line on the image.

    Args:
        image: (H, W, 3) uint8 RGB
        boundary: (H, W) binary boundary mask
        color: RGB boundary color
        thickness: line dilation for visibility

    Returns:
        (H, W, 3) uint8 image with boundary overlay
    """
    result = image.copy()

    boundary_u8 = (boundary > 0.5).astype(np.uint8) if boundary.dtype != np.uint8 else boundary
    if boundary_u8.max() == 1:
        boundary_u8 = boundary_u8 * 255

    # Dilate the skeleton slightly for visibility
    if thickness > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (thickness, thickness)
        )
        boundary_u8 = cv2.dilate(boundary_u8, kernel, iterations=1)

    mask_bool = boundary_u8 > 127
    for c in range(3):
        result[:, :, c] = np.where(mask_bool, color[c], result[:, :, c])

    return result


def create_comparison_grid(
    image: np.ndarray,
    mask_overlay: np.ndarray,
    boundary_overlay: np.ndarray,
    boundary_only: np.ndarray,
) -> np.ndarray:
    """
    Create a 2×2 grid: original | mask overlay | boundary overlay | boundary map.

    Args:
        image: original (H, W, 3)
        mask_overlay: masks overlaid (H, W, 3)
        boundary_overlay: boundary on image (H, W, 3)
        boundary_only: boundary on black (H, W) or (H, W, 3)

    Returns:
        (2*H, 2*W, 3) grid image
    """
    H, W = image.shape[:2]

    # Make boundary_only 3-channel
    if boundary_only.ndim == 2:
        boundary_only_u8 = (boundary_only > 0.5).astype(np.uint8) if boundary_only.dtype != np.uint8 else boundary_only
        if boundary_only_u8.max() <= 1:
            boundary_only_u8 = boundary_only_u8 * 255
        boundary_only = cv2.cvtColor(boundary_only_u8, cv2.COLOR_GRAY2RGB)

    # Ensure all panels are same size
    panels = [image, mask_overlay, boundary_overlay, boundary_only]
    panels = [cv2.resize(p, (W, H)) if p.shape[:2] != (H, W) else p for p in panels]

    top = np.concatenate([panels[0], panels[1]], axis=1)
    bottom = np.concatenate([panels[2], panels[3]], axis=1)
    return np.concatenate([top, bottom], axis=0)


# ===================================================================== #
#  Single image processing                                                #
# ===================================================================== #

def process_single_image(
    pipeline: Qwen2SAMPipeline,
    image_path: str,
    output_dir: Path,
    pp_config: PostprocessConfig,
    vis_cfg: dict,
) -> dict:
    """
    Process a single image through the full pipeline.

    Returns:
        dict with metadata (text, IoU, timings, etc.)
    """
    stem = Path(image_path).stem

    # Load image
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)

    # ---- Predict ----------------------------------------------------- #
    t0 = time.time()
    result = pipeline.predict(image_pil)
    t_predict = time.time() - t0

    print(f"  Text: {result.text}")
    print(f"  IoU: A={result.iou_a:.3f}, B={result.iou_b:.3f}")
    print(f"  SEG tokens found: {result.seg_tokens_found}")

    # ---- Post-process ------------------------------------------------ #
    t1 = time.time()
    if result.seg_tokens_found:
        boundary = extract_boundary_line(
            result.mask_a_logits, result.mask_b_logits, pp_config
        )
        result.boundary = boundary
    else:
        boundary = np.zeros(image_np.shape[:2], dtype=np.uint8)
    t_postprocess = time.time() - t1

    print(f"  Timings: predict={t_predict:.2f}s, postprocess={t_postprocess:.2f}s")

    # ---- Save outputs ------------------------------------------------ #
    # Masks
    cv2.imwrite(
        str(output_dir / f"{stem}_mask_a.png"),
        (result.mask_a * 255).astype(np.uint8),
    )
    cv2.imwrite(
        str(output_dir / f"{stem}_mask_b.png"),
        (result.mask_b * 255).astype(np.uint8),
    )

    # Boundary
    boundary_save = boundary if boundary.dtype == np.uint8 else (boundary * 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / f"{stem}_boundary.png"), boundary_save)

    # ---- Visualizations ---------------------------------------------- #
    alpha = vis_cfg.get("mask_alpha", 0.4)
    color_a = tuple(vis_cfg.get("mask_a_color", [255, 50, 50]))
    color_b = tuple(vis_cfg.get("mask_b_color", [50, 50, 255]))
    boundary_color = tuple(vis_cfg.get("boundary_color", [0, 255, 0]))
    boundary_thickness = vis_cfg.get("boundary_thickness", 2)

    mask_overlay = create_mask_overlay(
        image_np, result.mask_a, result.mask_b, alpha, color_a, color_b
    )
    boundary_overlay = draw_boundary_on_image(
        image_np, boundary, boundary_color, boundary_thickness
    )
    grid = create_comparison_grid(
        image_np, mask_overlay, boundary_overlay, boundary_save
    )

    # Save visualizations
    cv2.imwrite(
        str(output_dir / f"{stem}_mask_overlay.png"),
        cv2.cvtColor(mask_overlay, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(output_dir / f"{stem}_boundary_overlay.png"),
        cv2.cvtColor(boundary_overlay, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(output_dir / f"{stem}_grid.png"),
        cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
    )

    return {
        "image": str(image_path),
        "text": result.text,
        "iou_a": result.iou_a,
        "iou_b": result.iou_b,
        "seg_tokens_found": result.seg_tokens_found,
        "predict_time": round(t_predict, 3),
        "postprocess_time": round(t_postprocess, 3),
    }


# ===================================================================== #
#  Main                                                                   #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(description="Qwen2SAM Inference")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to inference config YAML")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Phase 3 checkpoint (overrides config)")
    parser.add_argument("--image", type=str, default=None,
                        help="Single image path")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Directory of images")
    parser.add_argument("--output_dir", type=str, default="results/",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Must specify --image or --image_dir")

    # ---- Load config ------------------------------------------------- #
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ---- Build pipeline ---------------------------------------------- #
    ckpt_path = args.checkpoint or cfg.get("checkpoint", {}).get("phase3_checkpoint")
    if not ckpt_path:
        raise ValueError("Must specify checkpoint path")

    print("Loading Qwen2SAM pipeline...")
    pipeline = Qwen2SAMPipeline.from_checkpoint(
        ckpt_path, cfg, device=args.device
    )

    # ---- Post-process config ----------------------------------------- #
    pp_dict = cfg.get("inference", {}).get("postprocess", {})
    pp_config = PostprocessConfig(**{
        k: v for k, v in pp_dict.items()
        if k in PostprocessConfig.__dataclass_fields__
    })

    vis_cfg = cfg.get("visualization", {})

    # ---- Output directory -------------------------------------------- #
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Collect images ---------------------------------------------- #
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.image_dir:
        img_dir = Path(args.image_dir)
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]:
            image_paths.extend(sorted(img_dir.glob(ext)))

    print(f"Processing {len(image_paths)} image(s)...")
    print(f"Output: {output_dir}")
    print()

    # ---- Process ----------------------------------------------------- #
    all_results = []
    for i, img_path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] {img_path}")
        result = process_single_image(
            pipeline, str(img_path), output_dir, pp_config, vis_cfg
        )
        all_results.append(result)
        print()

    # ---- Save summary ------------------------------------------------ #
    summary_path = output_dir / "results.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ---- Print summary ----------------------------------------------- #
    print(f"{'='*60}")
    print(f"Processed {len(all_results)} images")
    seg_found = sum(1 for r in all_results if r["seg_tokens_found"])
    print(f"SEG tokens found: {seg_found}/{len(all_results)}")
    if all_results:
        avg_iou_a = np.mean([r["iou_a"] for r in all_results if r["seg_tokens_found"]])
        avg_iou_b = np.mean([r["iou_b"] for r in all_results if r["seg_tokens_found"]])
        avg_time = np.mean([r["predict_time"] for r in all_results])
        print(f"Avg IoU: A={avg_iou_a:.3f}, B={avg_iou_b:.3f}")
        print(f"Avg predict time: {avg_time:.2f}s")
    print(f"Results saved: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
