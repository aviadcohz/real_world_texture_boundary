"""
Evaluation script for Qwen2SAM v2_tracker (DETR + SAM3 Tracker).

3-way comparison:
  1. Zero-shot: Fresh model, DETR masks → bilinear upsample (baseline)
  2. Stage 1 DETR: v2/best.pt, DETR masks → bilinear upsample
  3. Stage 2 Tracker: v2_tracker checkpoint, tracker refined masks

Produces:
  - Per-sample CSVs with IoU, Dice, ARI for all 3 models
  - summary.json with aggregated metrics + improvement stats
  - Per-sample visualization grids: Image | GT | Zero-shot | DETR | Tracker
  - training_curves.png (copied from checkpoint dir)

Usage:
  conda activate texture_boundary
  python -m qwen2sam.scripts.evaluate_v2_tracker \
      --config qwen2sam/configs/v2_tracker.yaml \
      --checkpoint checkpoints/v2_tracker/best.pt \
      --split test --output_dir eval_results/v2_tracker_test
"""

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from qwen2sam.models.qwen2sam_v2 import Qwen2SAMv2
from qwen2sam.models.qwen2sam_v2_tracker import Qwen2SAMv2Tracker
from qwen2sam.data.dataset_v2 import V2Dataset, V2Collator
from qwen2sam.training.train_v2_tracker import V2TrackerCollator
from qwen2sam.training.train_phase1 import load_config, set_seed


# ===================================================================== #
#  Metrics                                                                #
# ===================================================================== #

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred > 0.5
    gt_b = gt > 0.5
    intersection = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred > 0.5
    gt_b = gt > 0.5
    intersection = (pred_b & gt_b).sum()
    total = pred_b.sum() + gt_b.sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2.0 * intersection / total)


def compute_ari(pred_a: np.ndarray, pred_b: np.ndarray,
                gt_a: np.ndarray, gt_b: np.ndarray) -> float:
    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        return float("nan")
    pred_labels = np.zeros(pred_a.shape, dtype=np.int32)
    pred_labels[pred_a > 0.5] = 1
    pred_labels[pred_b > 0.5] = 2
    gt_labels = np.zeros(gt_a.shape, dtype=np.int32)
    gt_labels[gt_a > 0.5] = 1
    gt_labels[gt_b > 0.5] = 2
    return float(adjusted_rand_score(gt_labels.ravel(), pred_labels.ravel()))


def compute_sample_metrics(pred_a, pred_b, gt_a, gt_b, crop_name):
    # Hungarian matching: try both A/B assignments, pick the better one.
    iou_direct = (compute_iou(pred_a, gt_a) + compute_iou(pred_b, gt_b)) / 2.0
    iou_swapped = (compute_iou(pred_a, gt_b) + compute_iou(pred_b, gt_a)) / 2.0
    if iou_swapped > iou_direct:
        pred_a, pred_b = pred_b, pred_a

    iou_a = compute_iou(pred_a, gt_a)
    iou_b = compute_iou(pred_b, gt_b)
    dice_a = compute_dice(pred_a, gt_a)
    dice_b = compute_dice(pred_b, gt_b)
    ari = compute_ari(pred_a, pred_b, gt_a, gt_b)
    return {
        "crop_name": crop_name,
        "iou_a": iou_a, "iou_b": iou_b,
        "mean_iou": (iou_a + iou_b) / 2.0,
        "dice_a": dice_a, "dice_b": dice_b,
        "mean_dice": (dice_a + dice_b) / 2.0,
        "ari": ari,
    }


def aggregate_metrics(all_metrics, tag):
    return {
        "tag": tag,
        "num_samples": len(all_metrics),
        "mean_iou": float(np.mean([m["mean_iou"] for m in all_metrics])),
        "mean_iou_a": float(np.mean([m["iou_a"] for m in all_metrics])),
        "mean_iou_b": float(np.mean([m["iou_b"] for m in all_metrics])),
        "mean_dice": float(np.mean([m["mean_dice"] for m in all_metrics])),
        "mean_ari": float(np.nanmean([m["ari"] for m in all_metrics])),
    }


def save_metrics_csv(all_metrics, path):
    fieldnames = ["crop_name", "iou_a", "iou_b", "mean_iou",
                   "dice_a", "dice_b", "mean_dice", "ari"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow({k: m[k] for k in fieldnames})


# ===================================================================== #
#  Visualization helpers                                                  #
# ===================================================================== #

COLOR_A = (0, 0, 220)       # red in BGR
COLOR_B = (220, 80, 0)      # blue in BGR
COLOR_BOUNDARY = (0, 255, 255)  # yellow


def mask_overlay(image: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray,
                 alpha: float = 0.45) -> np.ndarray:
    vis = image.copy()
    overlay = image.copy()
    overlay[mask_a > 0.5] = COLOR_A
    overlay[mask_b > 0.5] = COLOR_B
    return cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)


def binary_mask_image(mask_a: np.ndarray, mask_b: np.ndarray,
                      h: int, w: int) -> np.ndarray:
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[mask_a > 0.5] = COLOR_A
    canvas[mask_b > 0.5] = COLOR_B
    return canvas


def boundary_image(mask_a: np.ndarray, mask_b: np.ndarray,
                   h: int, w: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ma = (mask_a > 0.5).astype(np.uint8) * 255
    mb = (mask_b > 0.5).astype(np.uint8) * 255
    bd_a = ma - cv2.erode(ma, kernel, iterations=1)
    bd_b = mb - cv2.erode(mb, kernel, iterations=1)
    da = cv2.dilate(bd_a, kernel, iterations=2)
    db = cv2.dilate(bd_b, kernel, iterations=2)
    interface = ((da > 0) & (db > 0)).astype(np.uint8) * 255
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[bd_a > 0] = (0, 0, 150)
    canvas[bd_b > 0] = (150, 60, 0)
    canvas[interface > 0] = COLOR_BOUNDARY
    return canvas


def create_grid_figure(
    image_bgr: np.ndarray,
    gt_a: np.ndarray, gt_b: np.ndarray,
    zs_a: np.ndarray, zs_b: np.ndarray,
    detr_a: np.ndarray, detr_b: np.ndarray,
    trk_a: np.ndarray, trk_b: np.ndarray,
    metrics_zs: dict,
    metrics_detr: dict,
    metrics_trk: dict,
    title: str = "",
    cell_size: int = 256,
) -> np.ndarray:
    """
    Create a 3-row x 5-col grid for one sample.

    Columns: Image | Ground Truth | Zero-shot | DETR | Tracker
    Row 1: Mask overlay (red=A, blue=B on image)
    Row 2: Binary masks (colored, no background)
    Row 3: Boundary visualization
    """
    h, w = image_bgr.shape[:2]
    scale = cell_size / max(h, w)
    ch, cw = int(h * scale), int(w * scale)

    def ri(img):
        return cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)

    def rm(mask):
        return cv2.resize(mask.astype(np.float32), (cw, ch),
                          interpolation=cv2.INTER_NEAREST)

    def make_col(ma, mb):
        return (
            mask_overlay(img, ma, mb),
            binary_mask_image(ma, mb, ch, cw),
            boundary_image(ma, mb, ch, cw),
        )

    img = ri(image_bgr)
    ga, gb = rm(gt_a), rm(gt_b)
    za, zb = rm(zs_a), rm(zs_b)
    da, db = rm(detr_a), rm(detr_b)
    ta, tb = rm(trk_a), rm(trk_b)

    # Col 0: Original image
    col_image = (img.copy(),
                 np.zeros((ch, cw, 3), dtype=np.uint8),
                 np.zeros((ch, cw, 3), dtype=np.uint8))
    col_gt = make_col(ga, gb)
    col_zs = make_col(za, zb)
    col_detr = make_col(da, db)
    col_trk = make_col(ta, tb)

    cols = [col_image, col_gt, col_zs, col_detr, col_trk]
    col_labels = [
        title,
        "Ground Truth",
        f"ZS mIoU={metrics_zs.get('mean_iou', 0):.3f}",
        f"DETR mIoU={metrics_detr.get('mean_iou', 0):.3f}",
        f"Trk mIoU={metrics_trk.get('mean_iou', 0):.3f}",
    ]
    row_labels = ["Overlay", "Masks", "Boundary"]

    # --- Assemble grid ---
    sep = 2
    header_h = 36
    row_label_w = 70

    bar_w = row_label_w + len(cols) * (cw + sep)
    bar = np.zeros((header_h, bar_w, 3), dtype=np.uint8) + 30
    x = row_label_w
    for lbl in col_labels:
        cv2.putText(bar, lbl, (x + 4, header_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        x += cw + sep

    def make_row(row_idx, row_label):
        rl = np.zeros((ch, row_label_w, 3), dtype=np.uint8) + 20
        cv2.putText(rl, row_label, (4, ch // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        sep_col = np.ones((ch, sep, 3), dtype=np.uint8) * 80
        row = rl
        for col in cols:
            row = np.concatenate([row, sep_col, col[row_idx]], axis=1)
        return row

    sep_row = np.ones((sep, bar.shape[1], 3), dtype=np.uint8) * 80
    rows = [bar]
    for ri_idx, rl in enumerate(row_labels):
        rows.append(sep_row)
        rows.append(make_row(ri_idx, rl))

    return np.concatenate(rows, axis=0)


# ===================================================================== #
#  Prediction helpers                                                     #
# ===================================================================== #

@torch.no_grad()
def predict_v2(model, batch, device):
    """Run base Qwen2SAMv2 on a batch, return binarized DETR masks."""
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)

    qwen_inputs = {
        k: batch[k] for k in [
            "input_ids", "attention_mask", "pixel_values",
            "image_grid_thw", "labels",
        ]
        if k in batch and isinstance(batch.get(k), torch.Tensor)
    }

    seg_a_pos = batch["seg_a_positions"]
    seg_b_pos = batch["seg_b_positions"]
    valid = (seg_a_pos >= 0) & (seg_b_pos >= 0)
    if not valid.any():
        return None, None

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(qwen_inputs, batch["sam_images"], seg_a_pos, seg_b_pos)

    # Select best query by confidence
    scores_a = outputs["pred_logits_a"][0].squeeze(-1).sigmoid()
    scores_b = outputs["pred_logits_b"][0].squeeze(-1).sigmoid()
    best_a = scores_a.argmax().item()
    best_b = scores_b.argmax().item()

    gt_h, gt_w = batch["masks_a"].shape[-2:]
    pred_mask_a = outputs["pred_masks_a"][0, best_a]
    pred_mask_b = outputs["pred_masks_b"][0, best_b]

    # Bilinear upsample from 288 to GT resolution
    if pred_mask_a.shape != (gt_h, gt_w):
        pred_mask_a = F.interpolate(
            pred_mask_a[None, None].float(), size=(gt_h, gt_w),
            mode="bilinear", align_corners=False,
        ).squeeze()
        pred_mask_b = F.interpolate(
            pred_mask_b[None, None].float(), size=(gt_h, gt_w),
            mode="bilinear", align_corners=False,
        ).squeeze()

    pred_a = (pred_mask_a.sigmoid().cpu().numpy() > 0.5).astype(np.float32)
    pred_b = (pred_mask_b.sigmoid().cpu().numpy() > 0.5).astype(np.float32)
    return pred_a, pred_b


@torch.no_grad()
def predict_tracker(model, batch, device):
    """Run Qwen2SAMv2Tracker, return both DETR and tracker refined masks."""
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)

    qwen_inputs = {
        k: batch[k] for k in [
            "input_ids", "attention_mask", "pixel_values",
            "image_grid_thw", "labels",
        ]
        if k in batch and isinstance(batch.get(k), torch.Tensor)
    }

    seg_a_pos = batch["seg_a_positions"]
    seg_b_pos = batch["seg_b_positions"]
    point_pos = batch["point_positions"]

    valid = (seg_a_pos >= 0) & (seg_b_pos >= 0) & (point_pos.min(dim=1).values >= 0)
    if not valid.any():
        return None, None, None, None

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(
            qwen_inputs, batch["sam_images"],
            seg_a_pos, seg_b_pos, point_pos,
        )

    gt_h, gt_w = batch["masks_a"].shape[-2:]

    # ---- DETR masks (best query by confidence, bilinear upsample) ---- #
    scores_a = outputs["pred_logits_a"][0].squeeze(-1).sigmoid()
    scores_b = outputs["pred_logits_b"][0].squeeze(-1).sigmoid()
    best_a = scores_a.argmax().item()
    best_b = scores_b.argmax().item()

    detr_a = outputs["pred_masks_a"][0, best_a]
    detr_b = outputs["pred_masks_b"][0, best_b]
    if detr_a.shape != (gt_h, gt_w):
        detr_a = F.interpolate(
            detr_a[None, None].float(), size=(gt_h, gt_w),
            mode="bilinear", align_corners=False,
        ).squeeze()
        detr_b = F.interpolate(
            detr_b[None, None].float(), size=(gt_h, gt_w),
            mode="bilinear", align_corners=False,
        ).squeeze()
    detr_a_np = (detr_a.sigmoid().cpu().numpy() > 0.5).astype(np.float32)
    detr_b_np = (detr_b.sigmoid().cpu().numpy() > 0.5).astype(np.float32)

    # ---- Tracker refined masks (bilinear upsample 288→GT) ------------ #
    refined_a = outputs["refined_masks_a"][0, 0]  # (288, 288)
    refined_b = outputs["refined_masks_b"][0, 0]
    if refined_a.shape != (gt_h, gt_w):
        refined_a = F.interpolate(
            refined_a[None, None].float(), size=(gt_h, gt_w),
            mode="bilinear", align_corners=False,
        ).squeeze()
        refined_b = F.interpolate(
            refined_b[None, None].float(), size=(gt_h, gt_w),
            mode="bilinear", align_corners=False,
        ).squeeze()
    trk_a_np = (refined_a.sigmoid().cpu().numpy() > 0.5).astype(np.float32)
    trk_b_np = (refined_b.sigmoid().cpu().numpy() > 0.5).astype(np.float32)

    return detr_a_np, detr_b_np, trk_a_np, trk_b_np


# ===================================================================== #
#  Main                                                                   #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2SAM v2_tracker (3-way)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="v2_tracker checkpoint (e.g. checkpoints/v2_tracker/best.pt)")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="eval_results/v2_tracker")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--cell_size", type=int, default=256)
    parser.add_argument("--no_vis", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Dataset ----------------------------------------------------- #
    data_root = args.data_root or cfg["data"]["data_root"]
    dataset = V2Dataset(
        data_root=data_root,
        metadata_file=cfg["data"].get("metadata_file", "metadata.json"),
        image_size=cfg["model"].get("image_size", 1008),
    )

    train_n = cfg["data"].get("train_size", 10)
    val_n = cfg["data"].get("val_size", train_n)
    if args.split == "train":
        eval_indices = list(range(train_n))
    elif args.split == "val":
        eval_indices = list(range(train_n, train_n + val_n))
    elif args.split == "test":
        eval_indices = list(range(train_n + val_n, len(dataset)))
    else:
        eval_indices = list(range(len(dataset)))
    print(f"Split '{args.split}': {len(eval_indices)} samples")

    # ---- Output dirs ------------------------------------------------- #
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    if not args.no_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================= #
    #  Pass 1: Zero-shot (fresh v2 model, no checkpoint)                  #
    # ================================================================= #
    print("\n--- Pass 1: Zero-shot (untrained model) ---")
    zs_model = Qwen2SAMv2(cfg, device=str(device))
    zs_model.qwen.eval()
    zs_model.projector.eval()
    zs_model.sam3.eval()

    v2_collator = V2Collator(
        processor=zs_model.processor,
        system_prompt=cfg["data"].get("system_prompt", ""),
        user_prompt=cfg["data"].get("user_prompt", ""),
        seg_a_id=zs_model.seg_a_id,
        seg_b_id=zs_model.seg_b_id,
        inference=True,
    )

    t0 = time.time()
    zs_metrics_all = []
    zs_preds = {}
    for i, idx in enumerate(eval_indices):
        raw_sample = dataset[idx]
        crop_name = f"sample_{idx}"
        gt_a_np = raw_sample["mask_a"].numpy()
        gt_b_np = raw_sample["mask_b"].numpy()

        single_batch = v2_collator([raw_sample])
        pred_a, pred_b = predict_v2(zs_model, single_batch, device)
        if pred_a is None:
            zs_preds[idx] = (None, None)
            continue

        metrics = compute_sample_metrics(pred_a, pred_b, gt_a_np, gt_b_np, crop_name)
        zs_metrics_all.append(metrics)
        zs_preds[idx] = (pred_a, pred_b)

        if (i + 1) % 20 == 0 or (i + 1) == len(eval_indices):
            running_iou = np.mean([m["mean_iou"] for m in zs_metrics_all])
            print(f"  ZS: {i+1}/{len(eval_indices)} | running mIoU: {running_iou:.4f}")

    zs_elapsed = time.time() - t0
    zs_summary = aggregate_metrics(zs_metrics_all, "zero_shot")
    print(f"  Zero-shot done ({zs_elapsed:.1f}s): mIoU={zs_summary['mean_iou']:.4f}, ARI={zs_summary['mean_ari']:.4f}")

    del zs_model
    torch.cuda.empty_cache()

    # ================================================================= #
    #  Pass 2: Stage 1 DETR (v2 checkpoint loaded into base model)        #
    # ================================================================= #
    print("\n--- Pass 2: Stage 1 DETR (v2 checkpoint) ---")
    v2_ckpt_path = cfg.get("tracker", {}).get("v2_checkpoint", "checkpoints/v2/best.pt")
    detr_model = Qwen2SAMv2(cfg, device=str(device))

    print(f"  Loading v2 checkpoint: {v2_ckpt_path}")
    ckpt = torch.load(v2_ckpt_path, map_location="cpu", weights_only=False)
    detr_model.projector.load_state_dict(ckpt["projector_state_dict"])
    if "sam3_trainable_state_dict" in ckpt:
        detr_model.sam3.load_state_dict(ckpt["sam3_trainable_state_dict"], strict=False)
    if "qwen_lora_state_dict" in ckpt:
        detr_model.qwen.load_state_dict(ckpt["qwen_lora_state_dict"], strict=False)
    print(f"  Loaded epoch {ckpt.get('epoch', '?')}")

    detr_model.qwen.eval()
    detr_model.projector.eval()
    detr_model.sam3.eval()

    t1 = time.time()
    detr_metrics_all = []
    detr_preds = {}
    for i, idx in enumerate(eval_indices):
        raw_sample = dataset[idx]
        crop_name = f"sample_{idx}"
        gt_a_np = raw_sample["mask_a"].numpy()
        gt_b_np = raw_sample["mask_b"].numpy()

        single_batch = v2_collator([raw_sample])
        pred_a, pred_b = predict_v2(detr_model, single_batch, device)
        if pred_a is None:
            detr_preds[idx] = (None, None)
            continue

        metrics = compute_sample_metrics(pred_a, pred_b, gt_a_np, gt_b_np, crop_name)
        detr_metrics_all.append(metrics)
        detr_preds[idx] = (pred_a, pred_b)

        if (i + 1) % 20 == 0 or (i + 1) == len(eval_indices):
            running_iou = np.mean([m["mean_iou"] for m in detr_metrics_all])
            print(f"  DETR: {i+1}/{len(eval_indices)} | running mIoU: {running_iou:.4f}")

    detr_elapsed = time.time() - t1
    detr_summary = aggregate_metrics(detr_metrics_all, "detr_stage1")
    print(f"  DETR done ({detr_elapsed:.1f}s): mIoU={detr_summary['mean_iou']:.4f}, ARI={detr_summary['mean_ari']:.4f}")

    del detr_model
    torch.cuda.empty_cache()

    # ================================================================= #
    #  Pass 3: Stage 2 Tracker (v2_tracker checkpoint)                    #
    # ================================================================= #
    print("\n--- Pass 3: Stage 2 Tracker (v2_tracker checkpoint) ---")
    tracker_model = Qwen2SAMv2Tracker(cfg, device=str(device))

    print(f"  Loading tracker checkpoint: {args.checkpoint}")
    trk_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Load all tracker components
    tracker_model.base.projector.load_state_dict(trk_ckpt["projector_state_dict"])
    if "sam3_trainable_state_dict" in trk_ckpt:
        tracker_model.base.sam3.load_state_dict(
            trk_ckpt["sam3_trainable_state_dict"], strict=False
        )
    if "qwen_lora_state_dict" in trk_ckpt:
        tracker_model.base.qwen.load_state_dict(
            trk_ckpt["qwen_lora_state_dict"], strict=False
        )
    if "coord_head_state_dict" in trk_ckpt:
        tracker_model.coord_head.load_state_dict(trk_ckpt["coord_head_state_dict"])
    if "point_projector_state_dict" in trk_ckpt:
        tracker_model.point_projector.load_state_dict(trk_ckpt["point_projector_state_dict"])
    if "sam_prompt_encoder_state_dict" in trk_ckpt:
        tracker_model.sam_prompt_encoder.load_state_dict(
            trk_ckpt["sam_prompt_encoder_state_dict"]
        )
    if "sam_mask_decoder_state_dict" in trk_ckpt:
        tracker_model.sam_mask_decoder.load_state_dict(
            trk_ckpt["sam_mask_decoder_state_dict"]
        )
    print(f"  Loaded epoch {trk_ckpt.get('epoch', '?')}")

    tracker_model.base.qwen.eval()
    tracker_model.base.projector.eval()
    tracker_model.base.sam3.eval()
    tracker_model.coord_head.eval()
    tracker_model.point_projector.eval()
    tracker_model.sam_prompt_encoder.eval()
    tracker_model.sam_mask_decoder.eval()

    # Tracker collator (with POINT tokens)
    trk_collator = V2TrackerCollator(
        processor=tracker_model.base.processor,
        system_prompt=cfg["data"].get("system_prompt", ""),
        user_prompt=cfg["data"].get("user_prompt", ""),
        seg_a_id=tracker_model.base.seg_a_id,
        seg_b_id=tracker_model.base.seg_b_id,
        point_token_ids=tracker_model.point_token_ids,
        num_points_per_texture=tracker_model.num_points,
        inference=True,
    )

    t2 = time.time()
    trk_metrics_all = []
    detr_in_trk_metrics_all = []  # DETR performance within tracker model

    for i, idx in enumerate(eval_indices):
        raw_sample = dataset[idx]
        crop_name = f"sample_{idx}"
        gt_a_np = raw_sample["mask_a"].numpy()
        gt_b_np = raw_sample["mask_b"].numpy()
        image_bgr = cv2.cvtColor(np.array(raw_sample["image"]), cv2.COLOR_RGB2BGR)

        single_batch = trk_collator([raw_sample])
        detr_a, detr_b, trk_a, trk_b = predict_tracker(tracker_model, single_batch, device)
        if trk_a is None:
            continue

        # Tracker refined mask metrics
        trk_met = compute_sample_metrics(trk_a, trk_b, gt_a_np, gt_b_np, crop_name)
        trk_metrics_all.append(trk_met)

        # DETR metrics within tracker model (to check DETR didn't degrade)
        detr_trk_met = compute_sample_metrics(detr_a, detr_b, gt_a_np, gt_b_np, crop_name)
        detr_in_trk_metrics_all.append(detr_trk_met)

        # Generate 5-column visualization
        if not args.no_vis:
            zs_a, zs_b = zs_preds.get(idx, (None, None))
            detr_s1_a, detr_s1_b = detr_preds.get(idx, (None, None))
            if zs_a is not None and detr_s1_a is not None:
                zs_met = next((m for m in zs_metrics_all if m["crop_name"] == crop_name), {})
                detr_met = next((m for m in detr_metrics_all if m["crop_name"] == crop_name), {})
                grid = create_grid_figure(
                    image_bgr, gt_a_np, gt_b_np,
                    zs_a, zs_b, detr_s1_a, detr_s1_b, trk_a, trk_b,
                    zs_met, detr_met, trk_met,
                    title=crop_name, cell_size=args.cell_size,
                )
                cv2.imwrite(str(vis_dir / f"{crop_name}_eval.png"), grid)

        if (i + 1) % 20 == 0 or (i + 1) == len(eval_indices):
            running_iou = np.mean([m["mean_iou"] for m in trk_metrics_all])
            running_ari = np.nanmean([m["ari"] for m in trk_metrics_all])
            print(f"  Tracker: {i+1}/{len(eval_indices)} | running mIoU: {running_iou:.4f}, ARI: {running_ari:.4f}")

    trk_elapsed = time.time() - t2
    trk_summary = aggregate_metrics(trk_metrics_all, "tracker_stage2")
    detr_in_trk_summary = aggregate_metrics(detr_in_trk_metrics_all, "detr_in_tracker")
    print(f"  Tracker done ({trk_elapsed:.1f}s): mIoU={trk_summary['mean_iou']:.4f}, ARI={trk_summary['mean_ari']:.4f}")

    # ---- Save CSVs --------------------------------------------------- #
    save_metrics_csv(zs_metrics_all, output_dir / "metrics_zero_shot.csv")
    save_metrics_csv(detr_metrics_all, output_dir / "metrics_detr_stage1.csv")
    save_metrics_csv(trk_metrics_all, output_dir / "metrics_tracker.csv")
    save_metrics_csv(detr_in_trk_metrics_all, output_dir / "metrics_detr_in_tracker.csv")

    # ---- Compute improvements ---------------------------------------- #
    def compute_improvement(a_summary, b_summary):
        imp = {}
        imp_pct = {}
        for k in ["mean_iou", "mean_dice", "mean_ari"]:
            delta = b_summary[k] - a_summary[k]
            imp[k] = delta
            base = a_summary[k]
            imp_pct[k] = round(100 * delta / base, 2) if abs(base) > 1e-8 else 0.0
        return imp, imp_pct

    trk_vs_zs_imp, trk_vs_zs_pct = compute_improvement(zs_summary, trk_summary)
    trk_vs_detr_imp, trk_vs_detr_pct = compute_improvement(detr_summary, trk_summary)

    summary = {
        "zero_shot": zs_summary,
        "detr_stage1": detr_summary,
        "tracker_stage2": trk_summary,
        "detr_in_tracker": detr_in_trk_summary,
        "improvement_tracker_vs_zero_shot": trk_vs_zs_imp,
        "improvement_tracker_vs_zero_shot_pct": trk_vs_zs_pct,
        "improvement_tracker_vs_detr": trk_vs_detr_imp,
        "improvement_tracker_vs_detr_pct": trk_vs_detr_pct,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Copy training_curves.png
    ckpt_dir = Path(args.checkpoint).parent
    curves_src = ckpt_dir / "training_curves.png"
    if curves_src.exists():
        shutil.copy2(str(curves_src), str(output_dir / "training_curves.png"))
        print(f"Copied training_curves.png from {ckpt_dir}")

    # ---- Analysis ---------------------------------------------------- #
    sorted_by_iou = sorted(trk_metrics_all, key=lambda m: m["mean_iou"])
    n_show = min(5, len(sorted_by_iou))
    worst = sorted_by_iou[:n_show]
    best = sorted_by_iou[-n_show:][::-1]
    n_good = sum(1 for m in trk_metrics_all if m["mean_iou"] >= 0.7)
    n_medium = sum(1 for m in trk_metrics_all if 0.4 <= m["mean_iou"] < 0.7)
    n_bad = sum(1 for m in trk_metrics_all if m["mean_iou"] < 0.4)

    # ---- Print 3-way summary ----------------------------------------- #
    total_elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  Qwen2SAM v2_tracker Evaluation — {args.split} set ({total_elapsed:.1f}s)")
    print(f"{'='*80}")
    print(f"  {'':20s} {'Zero-shot':>12s}  {'DETR (S1)':>12s}  {'Tracker (S2)':>12s}  {'Trk vs DETR':>12s}")
    print(f"  {'─'*74}")
    print(f"  {'Mean IoU':20s} {zs_summary['mean_iou']:12.4f}  {detr_summary['mean_iou']:12.4f}  {trk_summary['mean_iou']:12.4f}  {trk_vs_detr_imp['mean_iou']:+12.4f} ({trk_vs_detr_pct['mean_iou']:+.1f}%)")
    print(f"  {'Mean IoU (A)':20s} {zs_summary['mean_iou_a']:12.4f}  {detr_summary['mean_iou_a']:12.4f}  {trk_summary['mean_iou_a']:12.4f}")
    print(f"  {'Mean IoU (B)':20s} {zs_summary['mean_iou_b']:12.4f}  {detr_summary['mean_iou_b']:12.4f}  {trk_summary['mean_iou_b']:12.4f}")
    print(f"  {'Mean Dice':20s} {zs_summary['mean_dice']:12.4f}  {detr_summary['mean_dice']:12.4f}  {trk_summary['mean_dice']:12.4f}  {trk_vs_detr_imp['mean_dice']:+12.4f} ({trk_vs_detr_pct['mean_dice']:+.1f}%)")
    print(f"  {'Mean ARI':20s} {zs_summary['mean_ari']:12.4f}  {detr_summary['mean_ari']:12.4f}  {trk_summary['mean_ari']:12.4f}  {trk_vs_detr_imp['mean_ari']:+12.4f} ({trk_vs_detr_pct['mean_ari']:+.1f}%)")
    print(f"  {'Samples':20s} {zs_summary['num_samples']:12d}  {detr_summary['num_samples']:12d}  {trk_summary['num_samples']:12d}")

    # Also show DETR within tracker model (to check degradation)
    print(f"\n  DETR within tracker model:")
    print(f"  {'Mean IoU':20s} {detr_in_trk_summary['mean_iou']:12.4f}  (vs Stage 1: {detr_in_trk_summary['mean_iou'] - detr_summary['mean_iou']:+.4f})")
    print(f"  {'Mean ARI':20s} {detr_in_trk_summary['mean_ari']:12.4f}  (vs Stage 1: {detr_in_trk_summary['mean_ari'] - detr_summary['mean_ari']:+.4f})")

    print(f"\n{'='*80}")
    print(f"\n  Quality distribution (tracker refined):")
    print(f"    Good  (IoU >= 0.7): {n_good:3d} ({100*n_good/max(len(trk_metrics_all),1):.1f}%)")
    print(f"    Medium (0.4-0.7):  {n_medium:3d} ({100*n_medium/max(len(trk_metrics_all),1):.1f}%)")
    print(f"    Bad   (IoU < 0.4): {n_bad:3d} ({100*n_bad/max(len(trk_metrics_all),1):.1f}%)")
    print(f"\n  Top {n_show} best samples (tracker):")
    for m in best:
        print(f"    {m['crop_name']:>15s}  mIoU={m['mean_iou']:.4f}  dice={m['mean_dice']:.4f}  ari={m['ari']:.4f}")
    print(f"\n  Top {n_show} worst samples (tracker):")
    for m in worst:
        print(f"    {m['crop_name']:>15s}  mIoU={m['mean_iou']:.4f}  dice={m['mean_dice']:.4f}  ari={m['ari']:.4f}")
    print(f"\n{'='*80}")
    print(f"  Output: {output_dir}/")
    print(f"    summary.json                — 3-way comparison")
    print(f"    metrics_zero_shot.csv       — per-sample zero-shot metrics")
    print(f"    metrics_detr_stage1.csv     — per-sample DETR (Stage 1) metrics")
    print(f"    metrics_tracker.csv         — per-sample tracker refined metrics")
    print(f"    metrics_detr_in_tracker.csv — DETR quality within tracker model")
    if not args.no_vis:
        print(f"    visualizations/             — {len(trk_metrics_all)} sample grids (5-col)")
    if (output_dir / "training_curves.png").exists():
        print(f"    training_curves.png         — loss & IoU curves")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
