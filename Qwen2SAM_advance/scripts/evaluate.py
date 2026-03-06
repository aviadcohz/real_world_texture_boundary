#!/usr/bin/env python3
"""
Evaluation script for Qwen2SAM_advance checkpoints.

Runs COCO-format evaluation on val/test set, computing:
  - COCO AP (bbox) via pycocotools
  - COCO AP (segm) if masks are enabled
  - Per-image IoU, Dice, confidence distributions
  - Visualization grids for sample images

Usage:
    # Evaluate a checkpoint on COCO val2017
    python scripts/evaluate.py \
        --config configs/stage1.yaml \
        --checkpoint checkpoints/stage1/best.pt \
        --output_dir eval_results/stage1

    # Quick eval on subset
    python scripts/evaluate.py \
        --config configs/stage1.yaml \
        --checkpoint checkpoints/stage1/epoch_20.pt \
        --max_samples 500 \
        --output_dir eval_results/stage1_ep20
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, "/home/aviad/sam3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import yaml

from Qwen2SAM_advance.models.sam3_qwen_builder import build_sam3_qwen_model
from Qwen2SAM_advance.models.checkpoint_utils import load_checkpoint
from Qwen2SAM_advance.data.dataset_stage1 import build_coco_dataset, build_collate_fn
from Qwen2SAM_advance.training.visualize import (
    create_sample_grid, draw_boxes, draw_masks_overlay,
    cxcywh_to_xyxy, INSTANCE_COLORS,
)

from sam3.model.model_misc import SAM3Output


# ===================================================================== #
#  Metrics
# ===================================================================== #

def compute_box_iou(box1, box2):
    """Compute IoU between two boxes in XYXY format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / max(union, 1e-6)


def compute_mask_iou(pred, gt):
    """Compute IoU between two binary masks."""
    pred_b = pred > 0.5
    gt_b = gt > 0.5
    intersection = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_mask_dice(pred, gt):
    """Compute Dice coefficient between two binary masks."""
    pred_b = pred > 0.5
    gt_b = gt > 0.5
    intersection = (pred_b & gt_b).sum()
    total = pred_b.sum() + gt_b.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def hungarian_match_boxes(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """
    Match predicted boxes to GT boxes using greedy matching by IoU.

    Returns list of (pred_idx, gt_idx, iou) tuples for matched pairs.
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return []

    # Compute IoU matrix
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    iou_matrix = np.zeros((n_pred, n_gt))
    for i in range(n_pred):
        for j in range(n_gt):
            iou_matrix[i, j] = compute_box_iou(pred_boxes[i], gt_boxes[j])

    # Greedy matching: highest IoU first
    matches = []
    used_pred = set()
    used_gt = set()

    while True:
        if len(used_pred) == n_pred or len(used_gt) == n_gt:
            break
        # Find best remaining match
        best_iou = -1
        best_i, best_j = -1, -1
        for i in range(n_pred):
            if i in used_pred:
                continue
            for j in range(n_gt):
                if j in used_gt:
                    continue
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_i, best_j = i, j

        if best_iou < iou_threshold:
            break

        matches.append((best_i, best_j, best_iou))
        used_pred.add(best_i)
        used_gt.add(best_j)

    return matches


def compute_ap_at_iou(all_scores, all_matched, n_gt_total, iou_thresh=0.5):
    """
    Compute Average Precision at a given IoU threshold.

    Args:
        all_scores: list of prediction confidence scores
        all_matched: list of booleans (True if matched a GT at >= iou_thresh)
        n_gt_total: total number of GT objects
    """
    if n_gt_total == 0:
        return 0.0

    # Sort by score descending
    sorted_indices = np.argsort(-np.array(all_scores))
    sorted_matched = np.array(all_matched)[sorted_indices]

    tp = np.cumsum(sorted_matched)
    fp = np.cumsum(~sorted_matched)

    precision = tp / (tp + fp)
    recall = tp / n_gt_total

    # AP via 101-point interpolation (COCO style)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = recall >= t
        if mask.any():
            ap += precision[mask].max()
    ap /= 101

    return ap


# ===================================================================== #
#  COCO AP evaluation via pycocotools (optional, more accurate)
# ===================================================================== #

def run_coco_eval(coco_gt_path, predictions, iou_types=["bbox"]):
    """
    Run official COCO evaluation using pycocotools.

    Args:
        coco_gt_path: path to COCO annotations JSON
        predictions: list of dicts with {image_id, category_id, bbox, score}
                     or {image_id, category_id, segmentation, score}
        iou_types: list of "bbox" and/or "segm"

    Returns:
        dict of AP metrics
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("WARNING: pycocotools not installed, skipping COCO AP evaluation")
        return {}

    coco_gt = COCO(coco_gt_path)
    results = {}

    for iou_type in iou_types:
        if not predictions:
            results[iou_type] = {"AP": 0.0}
            continue

        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        results[iou_type] = {
            "AP": coco_eval.stats[0],
            "AP50": coco_eval.stats[1],
            "AP75": coco_eval.stats[2],
            "AP_small": coco_eval.stats[3],
            "AP_medium": coco_eval.stats[4],
            "AP_large": coco_eval.stats[5],
            "AR1": coco_eval.stats[6],
            "AR10": coco_eval.stats[7],
            "AR100": coco_eval.stats[8],
        }

    return results


# ===================================================================== #
#  Prediction
# ===================================================================== #

@torch.no_grad()
def predict_batch(model, batch_dict, device):
    """
    Run model on a batch, extract predictions.

    Returns:
        pred_boxes_xyxy: (B, Q, 4) normalized XYXY
        pred_scores: (B, Q) confidence
        pred_masks: (B, Q, H, W) or None
        gt_boxes_cxcywh: (B, max_boxes, 4) padded
        gt_num_boxes: (B,) int
        gt_masks: list of tensors or None
    """
    key = list(batch_dict.keys())[0]
    batch = batch_dict[key]

    # Move to device
    batch.img_batch = batch.img_batch.to(device, non_blocking=True)
    for stage_input in batch.find_inputs:
        for attr in ["img_ids", "text_ids", "input_boxes", "input_boxes_label",
                     "input_boxes_mask", "input_points", "input_points_mask"]:
            val = getattr(stage_input, attr, None)
            if val is not None and isinstance(val, torch.Tensor):
                setattr(stage_input, attr, val.to(device, non_blocking=True))
    for stage_target in batch.find_targets:
        for attr in ["num_boxes", "boxes", "boxes_padded", "is_exhaustive",
                     "segments", "is_valid_segment", "object_ids", "object_ids_padded",
                     "repeated_boxes", "semantic_segments"]:
            val = getattr(stage_target, attr, None)
            if val is not None and isinstance(val, torch.Tensor):
                setattr(stage_target, attr, val.to(device, non_blocking=True))

    # Set images for Qwen
    qwen_encoder = model.backbone.language_backbone
    if batch.raw_images is not None and len(batch.raw_images) > 0:
        qwen_encoder.set_image(batch.raw_images)
    else:
        qwen_encoder.set_image(None)

    # Forward
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        find_stages = model(batch)

    qwen_encoder.set_image(None)

    # Extract predictions
    with SAM3Output.iteration_mode(
        find_stages, iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE
    ) as stages:
        if len(stages) == 0 or len(stages[0]) == 0:
            return None
        preds = stages[0][-1]

    pred_logits = preds.get("pred_logits")
    pred_boxes_raw = preds.get("pred_boxes")  # CxCyWH
    pred_boxes_xyxy = preds.get("pred_boxes_xyxy")
    pred_masks = preds.get("pred_masks")

    if pred_logits is None:
        return None

    scores = pred_logits.sigmoid().squeeze(-1).cpu()  # (B, Q)

    if pred_boxes_xyxy is not None:
        boxes = pred_boxes_xyxy.cpu()
    elif pred_boxes_raw is not None:
        # Convert CxCyWH to XYXY
        cx, cy, w, h = pred_boxes_raw.unbind(-1)
        boxes = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1).cpu()
    else:
        boxes = None

    masks = pred_masks.cpu() if pred_masks is not None else None

    # Extract GT
    gt_boxes = None
    gt_num = None
    gt_segs = None
    if len(batch.find_targets) > 0:
        gt = batch.find_targets[0]
        if gt.boxes_padded is not None:
            gt_boxes = gt.boxes_padded.cpu()
            gt_num = gt.num_boxes.cpu() if gt.num_boxes is not None else None
        if gt.segments is not None:
            gt_segs = gt.segments.cpu()

    # Image tensor for visualization
    img_batch = batch.img_batch.cpu()

    return {
        "pred_boxes": boxes,
        "pred_scores": scores,
        "pred_masks": masks,
        "gt_boxes": gt_boxes,
        "gt_num_boxes": gt_num,
        "gt_masks": gt_segs,
        "img_batch": img_batch,
        "batch": batch,
    }


# ===================================================================== #
#  Main evaluation
# ===================================================================== #

def evaluate(cfg, checkpoint_path, output_dir, max_samples=None, vis_samples=20):
    """
    Full evaluation of a Qwen2SAM_advance checkpoint.

    Produces:
      - metrics.json: aggregate metrics (AP, mean IoU, etc.)
      - per_image_metrics.csv: per-image IoU and confidence
      - visualizations/: grid images for sample images
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    # ---- Build model ---- #
    print("Building model...")
    model = build_sam3_qwen_model(
        qwen_model_name=cfg["model"]["qwen_model_name"],
        qwen_dtype=cfg["model"].get("qwen_dtype", "bfloat16"),
        freeze_qwen=True,
        use_lora=cfg["model"].get("use_lora", False),
        lora_r=cfg["model"].get("lora_r", 16),
        lora_alpha=cfg["model"].get("lora_alpha", 32),
        adapter_num_tokens=cfg["model"].get("adapter_num_tokens", 32),
        adapter_num_layers=cfg["model"].get("adapter_num_layers", 2),
        adapter_num_heads=cfg["model"].get("adapter_num_heads", 8),
        enable_segmentation=cfg.get("enable_masks", False),
        eval_mode=True,
        load_sam3_from_hf=True,
        device="cpu",
    )

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    info = load_checkpoint(model, checkpoint_path, device="cpu")
    print(f"  Epoch: {info.get('epoch', '?')}, best_metric: {info.get('best_metric', '?')}")

    model = model.to(device)
    model.eval()

    # ---- Build dataset ---- #
    data_cfg = cfg["data"]
    val_ann = data_cfg.get("val_annotation_file", data_cfg["annotation_file"])
    val_img = data_cfg.get("val_image_dir", data_cfg["image_dir"])

    print(f"Loading dataset: {val_ann}")
    dataset = build_coco_dataset(
        annotation_file=val_ann,
        image_dir=val_img,
        resolution=data_cfg.get("resolution", 1008),
        with_masks=cfg.get("enable_masks", False),
    )

    if max_samples and max_samples < len(dataset):
        indices = np.random.RandomState(42).choice(len(dataset), size=max_samples, replace=False)
        dataset = Subset(dataset, indices.tolist())
    print(f"Eval dataset: {len(dataset)} samples")

    collate_fn = build_collate_fn(with_masks=cfg.get("enable_masks", False))
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=collate_fn, pin_memory=True,
    )

    # ---- Run evaluation ---- #
    print("Running evaluation...")
    all_image_metrics = []
    all_pred_scores = []
    all_pred_matched = []
    total_gt_boxes = 0
    vis_count = 0

    t0 = time.time()
    for step, batch_dict in enumerate(dataloader):
        try:
            result = predict_batch(model, batch_dict, device)
        except Exception as e:
            print(f"  Error at step {step}: {e}")
            continue

        if result is None:
            continue

        B = result["pred_scores"].shape[0]
        for b in range(B):
            scores = result["pred_scores"][b].numpy()  # (Q,)
            boxes = result["pred_boxes"][b].numpy() if result["pred_boxes"] is not None else None  # (Q, 4) XYXY

            # GT
            gt_num = 0
            gt_boxes_xyxy = np.zeros((0, 4))
            if result["gt_boxes"] is not None and result["gt_num_boxes"] is not None:
                gt_num = int(result["gt_num_boxes"][b].item())
                gt_cxcywh = result["gt_boxes"][b][:gt_num].numpy()
                if len(gt_cxcywh) > 0:
                    gt_boxes_xyxy = cxcywh_to_xyxy(gt_cxcywh)

            total_gt_boxes += gt_num

            # Filter predictions by confidence
            conf_mask = scores > 0.3
            if conf_mask.sum() == 0:
                topk = min(100, len(scores))
                top_idx = np.argsort(scores)[::-1][:topk]
            else:
                top_idx = np.where(conf_mask)[0]
                top_idx = top_idx[np.argsort(scores[top_idx])[::-1]][:100]

            pred_boxes_sel = boxes[top_idx] if boxes is not None else np.zeros((0, 4))
            pred_scores_sel = scores[top_idx]

            # Match predictions to GT
            matches = hungarian_match_boxes(pred_boxes_sel, pred_scores_sel, gt_boxes_xyxy, iou_threshold=0.5)
            matched_pred = set(m[0] for m in matches)
            matched_ious = [m[2] for m in matches]

            for idx in range(len(pred_scores_sel)):
                all_pred_scores.append(float(pred_scores_sel[idx]))
                all_pred_matched.append(idx in matched_pred)

            # Per-image metrics
            mean_matched_iou = np.mean(matched_ious) if matched_ious else 0.0
            recall = len(matches) / max(gt_num, 1)
            precision = len(matches) / max(len(pred_scores_sel), 1)
            max_conf = float(scores.max()) if len(scores) > 0 else 0.0
            mean_conf = float(pred_scores_sel.mean()) if len(pred_scores_sel) > 0 else 0.0

            image_metrics = {
                "step": step,
                "gt_boxes": gt_num,
                "pred_boxes_above_0.3": int(conf_mask.sum()),
                "matched": len(matches),
                "mean_matched_iou": mean_matched_iou,
                "recall@0.5": recall,
                "precision@0.5": precision,
                "max_confidence": max_conf,
                "mean_confidence": mean_conf,
            }

            # Mask metrics (if available)
            if result["pred_masks"] is not None and result["gt_masks"] is not None:
                pred_masks_all = result["pred_masks"][b].sigmoid().numpy()  # (Q, H, W)
                gt_masks_all = result["gt_masks"][b].numpy() if result["gt_masks"].dim() > 2 else None

                if gt_masks_all is not None and len(matches) > 0:
                    mask_ious = []
                    mask_dices = []
                    for pi, gi, _ in matches:
                        real_pi = top_idx[pi]
                        pm = pred_masks_all[real_pi]
                        gm = gt_masks_all[gi] if gi < len(gt_masks_all) else None
                        if gm is not None:
                            # Resize pred mask to GT size if needed
                            if pm.shape != gm.shape:
                                pm = cv2.resize(pm, (gm.shape[1], gm.shape[0]),
                                                interpolation=cv2.INTER_LINEAR)
                            mask_ious.append(compute_mask_iou(pm, gm))
                            mask_dices.append(compute_mask_dice(pm, gm))

                    if mask_ious:
                        image_metrics["mean_mask_iou"] = np.mean(mask_ious)
                        image_metrics["mean_mask_dice"] = np.mean(mask_dices)

            all_image_metrics.append(image_metrics)

            # ---- Visualization ---- #
            if vis_count < vis_samples:
                try:
                    img_tensor = result["img_batch"][b]
                    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
                    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
                    img_np = ((img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    gt_boxes_padded = result["gt_boxes"][b].numpy() if result["gt_boxes"] is not None else np.zeros((0, 4))

                    grid = create_sample_grid(
                        image_bgr=img_bgr,
                        gt_boxes_cxcywh=gt_boxes_padded,
                        gt_num_boxes=gt_num,
                        pred_boxes_xyxy=pred_boxes_sel,
                        pred_scores=pred_scores_sel,
                        title=f"sample_{step}",
                        cell_size=320,
                    )
                    cv2.imwrite(str(vis_dir / f"sample_{step}_eval.png"), grid)
                    vis_count += 1
                except Exception as e:
                    print(f"  Visualization error at step {step}: {e}")

        if (step + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{step+1}/{len(dataloader)}] {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nEvaluation done: {len(all_image_metrics)} images in {elapsed:.1f}s")

    # ---- Aggregate metrics ---- #
    if not all_image_metrics:
        print("No valid predictions!")
        return

    # AP computation
    ap50 = compute_ap_at_iou(all_pred_scores, all_pred_matched, total_gt_boxes, iou_thresh=0.5)

    summary = {
        "checkpoint": str(checkpoint_path),
        "num_images": len(all_image_metrics),
        "total_gt_boxes": total_gt_boxes,
        "AP@0.5": ap50,
        "mean_matched_iou": float(np.mean([m["mean_matched_iou"] for m in all_image_metrics])),
        "mean_recall@0.5": float(np.mean([m["recall@0.5"] for m in all_image_metrics])),
        "mean_precision@0.5": float(np.mean([m["precision@0.5"] for m in all_image_metrics])),
        "mean_max_confidence": float(np.mean([m["max_confidence"] for m in all_image_metrics])),
        "mean_confidence": float(np.mean([m["mean_confidence"] for m in all_image_metrics])),
    }

    # Mask metrics (if available)
    mask_ious = [m["mean_mask_iou"] for m in all_image_metrics if "mean_mask_iou" in m]
    mask_dices = [m["mean_mask_dice"] for m in all_image_metrics if "mean_mask_dice" in m]
    if mask_ious:
        summary["mean_mask_iou"] = float(np.mean(mask_ious))
        summary["mean_mask_dice"] = float(np.mean(mask_dices))

    # Save results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Per-image CSV
    fieldnames = list(all_image_metrics[0].keys())
    with open(output_dir / "per_image_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_image_metrics:
            writer.writerow(m)

    print(f"\nResults saved to {output_dir}/")
    print(f"  metrics.json, per_image_metrics.csv, visualizations/")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2SAM_advance checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, default="eval_results/stage1", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit evaluation to N images")
    parser.add_argument("--vis_samples", type=int, default=20, help="Number of visualization grids to save")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg, args.checkpoint, args.output_dir,
             max_samples=args.max_samples, vis_samples=args.vis_samples)


if __name__ == "__main__":
    main()
