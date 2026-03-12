"""
Evaluation script for Qwen2SAM v3 (multi-token description architecture).

Compares zero-shot (untrained) vs trained model, producing:
  - Per-sample CSV with IoU, Dice, ARI (for both zero-shot and trained)
  - summary.json with aggregated metrics + improvement stats
  - Per-sample visualization grids: Image | GT | Zero-shot | Trained
  - training_curves.png (copied from checkpoint dir)

Usage:
  conda activate texture_boundary
  python -m qwen2sam.scripts.evaluate_v3 \
      --config qwen2sam/configs/v3.yaml \
      --checkpoint checkpoints/v3/best.pt \
      --split test --output_dir eval_results/v3_test
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
from torch.utils.data import Subset

from qwen2sam.models.qwen2sam_v3 import Qwen2SAMv3
from qwen2sam.data.dataset_v3 import V3Dataset, V3Collator
from qwen2sam.training.train_phase1 import load_config, set_seed

# Reuse metrics and visualization from v2 eval
from qwen2sam.scripts.evaluate_v2 import (
    compute_iou, compute_dice, compute_ari,
    compute_sample_metrics, aggregate_metrics,
    save_metrics_csv, create_grid_figure,
)


# ===================================================================== #
#  Prediction                                                             #
# ===================================================================== #

@torch.no_grad()
def predict_single(model, batch, device):
    """Run v3 model on a single batch, return binarized masks."""
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

    # Select best query by confidence
    scores_a = outputs["pred_logits_a"][0].squeeze(-1).sigmoid()
    scores_b = outputs["pred_logits_b"][0].squeeze(-1).sigmoid()
    best_a = scores_a.argmax().item()
    best_b = scores_b.argmax().item()

    gt_h, gt_w = batch["masks_a"].shape[-2:]

    # Use hires path if available, otherwise fallback to bilinear
    if "hires_pixel_a" in outputs and outputs["hires_pixel_a"] is not None:
        hq_a = outputs["hires_queries_a"][0, best_a].float()
        hp_a = outputs["hires_pixel_a"]
        if hp_a.ndim == 4:
            hp_a = hp_a[0]
        pred_mask_a = torch.einsum("c,chw->hw", hq_a, hp_a.float())

        hq_b = outputs["hires_queries_b"][0, best_b].float()
        hp_b = outputs["hires_pixel_b"]
        if hp_b.ndim == 4:
            hp_b = hp_b[0]
        pred_mask_b = torch.einsum("c,chw->hw", hq_b, hp_b.float())

        if pred_mask_a.shape != (gt_h, gt_w):
            pred_mask_a = F.interpolate(
                pred_mask_a[None, None].float(), size=(gt_h, gt_w),
                mode="bilinear", align_corners=False,
            ).squeeze()
            pred_mask_b = F.interpolate(
                pred_mask_b[None, None].float(), size=(gt_h, gt_w),
                mode="bilinear", align_corners=False,
            ).squeeze()
    else:
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

    # Also return description lengths for analysis
    desc_info = {
        "desc_len_a": outputs["desc_lengths_a"][0].item(),
        "desc_len_b": outputs["desc_lengths_b"][0].item(),
    }

    return pred_a, pred_b, desc_info


# ===================================================================== #
#  Main                                                                   #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2SAM v3")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="eval_results/v3")
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

    # ---- Output dirs ------------------------------------------------- #
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    if not args.no_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================= #
    #  Pass 1: Zero-shot (fresh model, no checkpoint)                     #
    # ================================================================= #
    print("\n--- Pass 1: Zero-shot (untrained model) ---")
    print("Building zero-shot model...")
    zs_model = Qwen2SAMv3(cfg, device=str(device))
    zs_model.qwen.eval()
    zs_model.projector.eval()
    zs_model.sam3.eval()

    collator = V3Collator(
        processor=zs_model.processor,
        system_prompt=cfg["data"].get("system_prompt", ""),
        user_prompt=cfg["data"].get("user_prompt", ""),
        token_ids=zs_model.token_ids,
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

        single_batch = collator([raw_sample])
        pred_a, pred_b, _ = predict_single(zs_model, single_batch, device)

        metrics = compute_sample_metrics(pred_a, pred_b, gt_a_np, gt_b_np, crop_name)
        zs_metrics_all.append(metrics)
        zs_preds[idx] = (pred_a, pred_b)

        if (i + 1) % 20 == 0 or (i + 1) == len(eval_indices):
            running_iou = np.mean([m["mean_iou"] for m in zs_metrics_all])
            print(f"  ZS: {i+1}/{len(eval_indices)} | running mIoU: {running_iou:.4f}")

    zs_elapsed = time.time() - t0
    zs_summary = aggregate_metrics(zs_metrics_all, "zero_shot")
    print(f"  Zero-shot done ({zs_elapsed:.1f}s): mIoU={zs_summary['mean_iou']:.4f}")

    del zs_model
    torch.cuda.empty_cache()

    # ================================================================= #
    #  Pass 2: Trained model                                              #
    # ================================================================= #
    print("\n--- Pass 2: Trained model ---")
    print("Building trained model...")
    model = Qwen2SAMv3(cfg, device=str(device))

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.projector.load_state_dict(ckpt["projector_state_dict"])
    if "sam3_trainable_state_dict" in ckpt:
        model.sam3.load_state_dict(ckpt["sam3_trainable_state_dict"], strict=False)
    if "qwen_lora_state_dict" in ckpt:
        model.qwen.load_state_dict(ckpt["qwen_lora_state_dict"], strict=False)
    if "hires_head_state_dict" in ckpt and hasattr(model, "hires_head") and model.hires_head is not None:
        model.hires_head.load_state_dict(ckpt["hires_head_state_dict"])
        print("  HiRes head loaded")
    if "align_projector_state_dict" in ckpt and hasattr(model, "align_projector") and model.align_projector is not None:
        model.align_projector.load_state_dict(ckpt["align_projector_state_dict"])
        print("  Align projector loaded")
    print(f"  Loaded epoch {ckpt.get('epoch', '?')}")

    model.qwen.eval()
    model.projector.eval()
    model.sam3.eval()

    t1 = time.time()
    trained_metrics_all = []
    desc_lengths = {"a": [], "b": []}
    for i, idx in enumerate(eval_indices):
        raw_sample = dataset[idx]
        crop_name = f"sample_{idx}"
        gt_a_np = raw_sample["mask_a"].numpy()
        gt_b_np = raw_sample["mask_b"].numpy()
        image_bgr = cv2.cvtColor(np.array(raw_sample["image"]), cv2.COLOR_RGB2BGR)

        single_batch = collator([raw_sample])
        pred_a, pred_b, desc_info = predict_single(model, single_batch, device)

        t_metrics = compute_sample_metrics(pred_a, pred_b, gt_a_np, gt_b_np, crop_name)
        trained_metrics_all.append(t_metrics)
        desc_lengths["a"].append(desc_info["desc_len_a"])
        desc_lengths["b"].append(desc_info["desc_len_b"])

        if not args.no_vis:
            zs_a, zs_b = zs_preds.get(idx, (None, None))
            if zs_a is not None:
                zs_met = next((m for m in zs_metrics_all if m["crop_name"] == crop_name), {})
                grid = create_grid_figure(
                    image_bgr, gt_a_np, gt_b_np,
                    zs_a, zs_b, pred_a, pred_b,
                    t_metrics, zs_met,
                    title=crop_name, cell_size=args.cell_size,
                )
                cv2.imwrite(str(vis_dir / f"{crop_name}_eval.png"), grid)

        if (i + 1) % 20 == 0 or (i + 1) == len(eval_indices):
            running_iou = np.mean([m["mean_iou"] for m in trained_metrics_all])
            print(f"  Trained: {i+1}/{len(eval_indices)} | running mIoU: {running_iou:.4f}")

    trained_elapsed = time.time() - t1
    trained_summary = aggregate_metrics(trained_metrics_all, "trained")
    print(f"  Trained done ({trained_elapsed:.1f}s): mIoU={trained_summary['mean_iou']:.4f}")

    # ---- Save CSVs --------------------------------------------------- #
    save_metrics_csv(zs_metrics_all, output_dir / "metrics_zero_shot.csv")
    save_metrics_csv(trained_metrics_all, output_dir / "metrics_trained.csv")

    # ---- Compute improvement ----------------------------------------- #
    improvement = {
        "mean_iou": trained_summary["mean_iou"] - zs_summary["mean_iou"],
        "mean_dice": trained_summary["mean_dice"] - zs_summary["mean_dice"],
        "mean_ari": trained_summary["mean_ari"] - zs_summary["mean_ari"],
    }
    improvement_pct = {}
    for k in improvement:
        base = zs_summary[k]
        improvement_pct[k] = round(100 * improvement[k] / base, 2) if abs(base) > 1e-8 else 0.0

    summary = {
        "trained": trained_summary,
        "zero_shot": zs_summary,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "desc_lengths": {
            "mean_a": float(np.mean(desc_lengths["a"])),
            "mean_b": float(np.mean(desc_lengths["b"])),
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Copy training_curves.png
    ckpt_dir = Path(args.checkpoint).parent
    curves_src = ckpt_dir / "training_curves.png"
    if curves_src.exists():
        shutil.copy2(str(curves_src), str(output_dir / "training_curves.png"))

    # ---- Analysis ---------------------------------------------------- #
    sorted_by_iou = sorted(trained_metrics_all, key=lambda m: m["mean_iou"])
    n_show = min(5, len(sorted_by_iou))
    worst = sorted_by_iou[:n_show]
    best = sorted_by_iou[-n_show:][::-1]
    n_good = sum(1 for m in trained_metrics_all if m["mean_iou"] >= 0.7)
    n_medium = sum(1 for m in trained_metrics_all if 0.4 <= m["mean_iou"] < 0.7)
    n_bad = sum(1 for m in trained_metrics_all if m["mean_iou"] < 0.4)

    # ---- Print summary ------------------------------------------------ #
    total_elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  Qwen2SAM v3 Evaluation — {args.split} set ({total_elapsed:.1f}s)")
    print(f"{'='*65}")
    print(f"  {'':20s} {'Zero-shot':>12s}  {'Trained':>12s}  {'Improvement':>12s}")
    print(f"  {'─'*58}")
    print(f"  {'Mean IoU':20s} {zs_summary['mean_iou']:12.4f}  {trained_summary['mean_iou']:12.4f}  {improvement['mean_iou']:+12.4f} ({improvement_pct['mean_iou']:+.1f}%)")
    print(f"  {'Mean IoU (A)':20s} {zs_summary['mean_iou_a']:12.4f}  {trained_summary['mean_iou_a']:12.4f}")
    print(f"  {'Mean IoU (B)':20s} {zs_summary['mean_iou_b']:12.4f}  {trained_summary['mean_iou_b']:12.4f}")
    print(f"  {'Mean Dice':20s} {zs_summary['mean_dice']:12.4f}  {trained_summary['mean_dice']:12.4f}  {improvement['mean_dice']:+12.4f} ({improvement_pct['mean_dice']:+.1f}%)")
    print(f"  {'Mean ARI':20s} {zs_summary['mean_ari']:12.4f}  {trained_summary['mean_ari']:12.4f}  {improvement['mean_ari']:+12.4f} ({improvement_pct['mean_ari']:+.1f}%)")
    print(f"  {'Samples':20s} {zs_summary['num_samples']:12d}  {trained_summary['num_samples']:12d}")
    print(f"  {'Avg desc len (A)':20s} {'—':>12s}  {np.mean(desc_lengths['a']):12.1f}")
    print(f"  {'Avg desc len (B)':20s} {'—':>12s}  {np.mean(desc_lengths['b']):12.1f}")
    print(f"{'='*65}")
    print(f"\n  Quality distribution (trained):")
    print(f"    Good  (IoU >= 0.7): {n_good:3d} ({100*n_good/max(len(trained_metrics_all),1):.1f}%)")
    print(f"    Medium (0.4-0.7):  {n_medium:3d} ({100*n_medium/max(len(trained_metrics_all),1):.1f}%)")
    print(f"    Bad   (IoU < 0.4): {n_bad:3d} ({100*n_bad/max(len(trained_metrics_all),1):.1f}%)")
    print(f"\n  Top {n_show} best samples:")
    for m in best:
        print(f"    {m['crop_name']:>15s}  mIoU={m['mean_iou']:.4f}  dice={m['mean_dice']:.4f}  ari={m['ari']:.4f}")
    print(f"\n  Top {n_show} worst samples:")
    for m in worst:
        print(f"    {m['crop_name']:>15s}  mIoU={m['mean_iou']:.4f}  dice={m['mean_dice']:.4f}  ari={m['ari']:.4f}")
    print(f"\n{'='*65}")
    print(f"  Output: {output_dir}/")
    print(f"    summary.json             — zero-shot vs trained comparison")
    print(f"    metrics_zero_shot.csv    — per-sample zero-shot metrics")
    print(f"    metrics_trained.csv      — per-sample trained metrics")
    if not args.no_vis:
        print(f"    visualizations/          — {len(trained_metrics_all)} sample grids")
    if (output_dir / "training_curves.png").exists():
        print(f"    training_curves.png      — loss & IoU curves")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
