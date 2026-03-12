"""
Evaluation script for Qwen2SAM v3_tracker (Multi-Token DETR + SAM3 Tracker).

3-way comparison:
  1. Zero-shot: Fresh v3 model, DETR masks (baseline)
  2. Stage 1 v3: v3/best.pt, multi-token DETR masks
  3. Stage 2 Tracker: v3_tracker checkpoint, tracker refined masks

Usage:
  conda activate texture_boundary
  python -m qwen2sam.scripts.evaluate_v3_tracker \
      --config qwen2sam/configs/v3_tracker.yaml \
      --checkpoint checkpoints/v3_tracker/best.pt \
      --split test --output_dir eval_results/v3_tracker_test
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from qwen2sam.models.qwen2sam_v3 import Qwen2SAMv3
from qwen2sam.models.qwen2sam_v3_tracker import Qwen2SAMv3Tracker
from qwen2sam.data.dataset_v3 import V3Dataset, V3Collator
from qwen2sam.training.train_v3_tracker import V3TrackerCollator
from qwen2sam.training.train_phase1 import load_config, set_seed

from qwen2sam.scripts.evaluate_v2 import (
    compute_sample_metrics, aggregate_metrics, save_metrics_csv,
)


# ===================================================================== #
#  Prediction helpers                                                     #
# ===================================================================== #

@torch.no_grad()
def predict_v3(model, batch, device):
    """Run Qwen2SAMv3 on a batch, return binarized DETR masks."""
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

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(qwen_inputs, batch["sam_images"])

    scores_a = outputs["pred_logits_a"][0].squeeze(-1).sigmoid()
    scores_b = outputs["pred_logits_b"][0].squeeze(-1).sigmoid()
    best_a = scores_a.argmax().item()
    best_b = scores_b.argmax().item()

    gt_h, gt_w = batch["masks_a"].shape[-2:]
    pred_mask_a = outputs["pred_masks_a"][0, best_a]
    pred_mask_b = outputs["pred_masks_b"][0, best_b]

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
    """Run Qwen2SAMv3Tracker, return both DETR and tracker refined masks."""
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

    point_pos = batch["point_positions"]
    valid = point_pos.min(dim=1).values >= 0
    if not valid.any():
        return None, None, None, None

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(qwen_inputs, batch["sam_images"], point_pos)

    gt_h, gt_w = batch["masks_a"].shape[-2:]

    # DETR masks
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

    # Tracker refined masks
    refined_a = outputs["refined_masks_a"][0, 0]
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
    parser = argparse.ArgumentParser(description="Evaluate Qwen2SAM v3_tracker (3-way)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="eval_results/v3_tracker")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--no_vis", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = args.data_root or cfg["data"]["data_root"]
    dataset = V3Dataset(
        data_root=data_root,
        metadata_file=cfg["data"].get("metadata_file", "metadata_phase1.json"),
        image_size=cfg["model"].get("image_size", 1008),
    )

    train_n = cfg["data"].get("train_size", 26)
    val_n = cfg["data"].get("val_size", 0)
    if args.split == "train":
        eval_indices = list(range(train_n))
    elif args.split == "val":
        eval_indices = list(range(train_n, train_n + val_n))
    elif args.split == "test":
        eval_indices = list(range(train_n + val_n, len(dataset)))
    else:
        eval_indices = list(range(len(dataset)))
    print(f"Split '{args.split}': {len(eval_indices)} samples")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================= #
    #  Pass 1: Zero-shot (fresh v3 model)                                 #
    # ================================================================= #
    print("\n--- Pass 1: Zero-shot (untrained v3 model) ---")
    zs_model = Qwen2SAMv3(cfg, device=str(device))
    zs_model.qwen.eval()
    zs_model.projector.eval()
    zs_model.sam3.eval()

    v3_collator = V3Collator(
        processor=zs_model.processor,
        system_prompt=cfg["data"].get("system_prompt", ""),
        user_prompt=cfg["data"].get("user_prompt", ""),
        token_ids=zs_model.token_ids,
        inference=True,
    )

    t0 = time.time()
    zs_metrics_all = []
    for i, idx in enumerate(eval_indices):
        raw_sample = dataset[idx]
        crop_name = f"sample_{idx}"
        gt_a_np = raw_sample["mask_a"].numpy()
        gt_b_np = raw_sample["mask_b"].numpy()

        single_batch = v3_collator([raw_sample])
        pred_a, pred_b = predict_v3(zs_model, single_batch, device)

        metrics = compute_sample_metrics(pred_a, pred_b, gt_a_np, gt_b_np, crop_name)
        zs_metrics_all.append(metrics)

        if (i + 1) % 50 == 0 or (i + 1) == len(eval_indices):
            running_iou = np.mean([m["mean_iou"] for m in zs_metrics_all])
            print(f"  ZS: {i+1}/{len(eval_indices)} | mIoU: {running_iou:.4f}")

    zs_summary = aggregate_metrics(zs_metrics_all, "zero_shot")
    print(f"  Zero-shot: mIoU={zs_summary['mean_iou']:.4f}")

    del zs_model
    torch.cuda.empty_cache()

    # ================================================================= #
    #  Pass 2: Stage 1 v3 (v3/best.pt)                                    #
    # ================================================================= #
    print("\n--- Pass 2: Stage 1 v3 (DETR) ---")
    v3_ckpt_path = cfg.get("tracker", {}).get("v3_checkpoint", "checkpoints/v3/best.pt")
    v3_model = Qwen2SAMv3(cfg, device=str(device))

    print(f"  Loading v3 checkpoint: {v3_ckpt_path}")
    ckpt = torch.load(v3_ckpt_path, map_location="cpu", weights_only=False)
    v3_model.projector.load_state_dict(ckpt["projector_state_dict"])
    if "sam3_trainable_state_dict" in ckpt:
        v3_model.sam3.load_state_dict(ckpt["sam3_trainable_state_dict"], strict=False)
    if "qwen_lora_state_dict" in ckpt:
        v3_model.qwen.load_state_dict(ckpt["qwen_lora_state_dict"], strict=False)
    if "align_projector_state_dict" in ckpt and v3_model.align_projector is not None:
        v3_model.align_projector.load_state_dict(ckpt["align_projector_state_dict"])

    v3_model.qwen.eval()
    v3_model.projector.eval()
    v3_model.sam3.eval()

    detr_metrics_all = []
    for i, idx in enumerate(eval_indices):
        raw_sample = dataset[idx]
        crop_name = f"sample_{idx}"
        gt_a_np = raw_sample["mask_a"].numpy()
        gt_b_np = raw_sample["mask_b"].numpy()

        single_batch = v3_collator([raw_sample])
        pred_a, pred_b = predict_v3(v3_model, single_batch, device)

        metrics = compute_sample_metrics(pred_a, pred_b, gt_a_np, gt_b_np, crop_name)
        detr_metrics_all.append(metrics)

        if (i + 1) % 50 == 0 or (i + 1) == len(eval_indices):
            running_iou = np.mean([m["mean_iou"] for m in detr_metrics_all])
            print(f"  DETR: {i+1}/{len(eval_indices)} | mIoU: {running_iou:.4f}")

    detr_summary = aggregate_metrics(detr_metrics_all, "v3_detr_stage1")
    print(f"  v3 DETR: mIoU={detr_summary['mean_iou']:.4f}")

    del v3_model
    torch.cuda.empty_cache()

    # ================================================================= #
    #  Pass 3: Stage 2 Tracker                                            #
    # ================================================================= #
    print("\n--- Pass 3: Stage 2 Tracker ---")
    tracker_model = Qwen2SAMv3Tracker(cfg, device=str(device))

    print(f"  Loading tracker checkpoint: {args.checkpoint}")
    trk_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

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
    if "align_projector_state_dict" in trk_ckpt and tracker_model.base.align_projector is not None:
        tracker_model.base.align_projector.load_state_dict(trk_ckpt["align_projector_state_dict"])
    if "sam_prompt_encoder_state_dict" in trk_ckpt:
        tracker_model.sam_prompt_encoder.load_state_dict(trk_ckpt["sam_prompt_encoder_state_dict"])
    if "sam_mask_decoder_state_dict" in trk_ckpt:
        tracker_model.sam_mask_decoder.load_state_dict(trk_ckpt["sam_mask_decoder_state_dict"])
    print(f"  Loaded epoch {trk_ckpt.get('epoch', '?')}")

    tracker_model.base.qwen.eval()
    tracker_model.base.projector.eval()
    tracker_model.base.sam3.eval()
    tracker_model.coord_head.eval()
    tracker_model.sam_prompt_encoder.eval()
    tracker_model.sam_mask_decoder.eval()

    trk_collator = V3TrackerCollator(
        processor=tracker_model.base.processor,
        system_prompt=cfg["data"].get("system_prompt", ""),
        user_prompt=cfg["data"].get("user_prompt", ""),
        token_ids=tracker_model.base.token_ids,
        point_token_ids=tracker_model.point_token_ids,
        num_points_per_texture=tracker_model.num_points,
        inference=True,
    )

    trk_metrics_all = []
    for i, idx in enumerate(eval_indices):
        raw_sample = dataset[idx]
        crop_name = f"sample_{idx}"
        gt_a_np = raw_sample["mask_a"].numpy()
        gt_b_np = raw_sample["mask_b"].numpy()

        single_batch = trk_collator([raw_sample])
        detr_a, detr_b, trk_a, trk_b = predict_tracker(tracker_model, single_batch, device)
        if trk_a is None:
            continue

        trk_met = compute_sample_metrics(trk_a, trk_b, gt_a_np, gt_b_np, crop_name)
        trk_metrics_all.append(trk_met)

        if (i + 1) % 50 == 0 or (i + 1) == len(eval_indices):
            running_iou = np.mean([m["mean_iou"] for m in trk_metrics_all])
            print(f"  Tracker: {i+1}/{len(eval_indices)} | mIoU: {running_iou:.4f}")

    trk_summary = aggregate_metrics(trk_metrics_all, "v3_tracker_stage2")
    print(f"  Tracker: mIoU={trk_summary['mean_iou']:.4f}")

    # ---- Save CSVs --------------------------------------------------- #
    save_metrics_csv(zs_metrics_all, output_dir / "metrics_zero_shot.csv")
    save_metrics_csv(detr_metrics_all, output_dir / "metrics_v3_detr.csv")
    save_metrics_csv(trk_metrics_all, output_dir / "metrics_tracker.csv")

    # ---- Compute improvements ---------------------------------------- #
    def compute_improvement(a, b):
        imp, pct = {}, {}
        for k in ["mean_iou", "mean_dice", "mean_ari"]:
            delta = b[k] - a[k]
            imp[k] = delta
            pct[k] = round(100 * delta / a[k], 2) if abs(a[k]) > 1e-8 else 0.0
        return imp, pct

    trk_vs_zs_imp, trk_vs_zs_pct = compute_improvement(zs_summary, trk_summary)
    trk_vs_detr_imp, trk_vs_detr_pct = compute_improvement(detr_summary, trk_summary)

    summary = {
        "zero_shot": zs_summary,
        "v3_detr_stage1": detr_summary,
        "v3_tracker_stage2": trk_summary,
        "improvement_tracker_vs_zero_shot": trk_vs_zs_imp,
        "improvement_tracker_vs_zero_shot_pct": trk_vs_zs_pct,
        "improvement_tracker_vs_detr": trk_vs_detr_imp,
        "improvement_tracker_vs_detr_pct": trk_vs_detr_pct,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Copy training curves
    ckpt_dir = Path(args.checkpoint).parent
    curves_src = ckpt_dir / "training_curves.png"
    if curves_src.exists():
        shutil.copy2(str(curves_src), str(output_dir / "training_curves.png"))

    # ---- Analysis ---------------------------------------------------- #
    sorted_by_iou = sorted(trk_metrics_all, key=lambda m: m["mean_iou"])
    n_show = min(5, len(sorted_by_iou))
    worst = sorted_by_iou[:n_show]
    best = sorted_by_iou[-n_show:][::-1]
    n_good = sum(1 for m in trk_metrics_all if m["mean_iou"] >= 0.7)
    n_medium = sum(1 for m in trk_metrics_all if 0.4 <= m["mean_iou"] < 0.7)
    n_bad = sum(1 for m in trk_metrics_all if m["mean_iou"] < 0.4)

    # ---- Print summary ----------------------------------------------- #
    total_elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  Qwen2SAM v3_tracker Evaluation — {args.split} set ({total_elapsed:.1f}s)")
    print(f"{'='*80}")
    print(f"  {'':20s} {'Zero-shot':>12s}  {'v3 DETR':>12s}  {'Tracker':>12s}  {'Trk vs DETR':>12s}")
    print(f"  {'─'*74}")
    print(f"  {'Mean IoU':20s} {zs_summary['mean_iou']:12.4f}  {detr_summary['mean_iou']:12.4f}  {trk_summary['mean_iou']:12.4f}  {trk_vs_detr_imp['mean_iou']:+12.4f} ({trk_vs_detr_pct['mean_iou']:+.1f}%)")
    print(f"  {'Mean Dice':20s} {zs_summary['mean_dice']:12.4f}  {detr_summary['mean_dice']:12.4f}  {trk_summary['mean_dice']:12.4f}  {trk_vs_detr_imp['mean_dice']:+12.4f} ({trk_vs_detr_pct['mean_dice']:+.1f}%)")
    print(f"  {'Mean ARI':20s} {zs_summary['mean_ari']:12.4f}  {detr_summary['mean_ari']:12.4f}  {trk_summary['mean_ari']:12.4f}  {trk_vs_detr_imp['mean_ari']:+12.4f} ({trk_vs_detr_pct['mean_ari']:+.1f}%)")
    print(f"  {'Samples':20s} {zs_summary['num_samples']:12d}  {detr_summary['num_samples']:12d}  {trk_summary['num_samples']:12d}")
    print(f"{'='*80}")
    print(f"\n  Quality distribution (tracker refined):")
    print(f"    Good  (IoU >= 0.7): {n_good:3d} ({100*n_good/max(len(trk_metrics_all),1):.1f}%)")
    print(f"    Medium (0.4-0.7):  {n_medium:3d} ({100*n_medium/max(len(trk_metrics_all),1):.1f}%)")
    print(f"    Bad   (IoU < 0.4): {n_bad:3d} ({100*n_bad/max(len(trk_metrics_all),1):.1f}%)")
    print(f"\n  Top {n_show} best:")
    for m in best:
        print(f"    {m['crop_name']:>15s}  mIoU={m['mean_iou']:.4f}  dice={m['mean_dice']:.4f}  ari={m['ari']:.4f}")
    print(f"\n  Top {n_show} worst:")
    for m in worst:
        print(f"    {m['crop_name']:>15s}  mIoU={m['mean_iou']:.4f}  dice={m['mean_dice']:.4f}  ari={m['ari']:.4f}")
    print(f"\n{'='*80}")
    print(f"  Output: {output_dir}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
