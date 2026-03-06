"""
Stage 1 Training: General segmentation recovery with Qwen replacing CLIP.

Trains the cross-attention adapter + SAM3 encoder/decoder to work with
Qwen2.5-VL embeddings instead of CLIP. Uses COCO-format datasets.

Features:
  - Weights & Biases integration for monitoring
  - Periodic validation with AP metrics
  - Visual logging (predicted boxes/masks on images)
  - Per-component loss tracking
  - Gradient norm monitoring
  - Learning rate scheduling visualization

Usage:
    python training/train_stage1.py --config configs/stage1.yaml
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/aviad/sam3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

import yaml

from Qwen2SAM_advance.models.sam3_qwen_builder import (
    build_sam3_qwen_model,
    get_trainable_params,
)
from Qwen2SAM_advance.models.checkpoint_utils import save_checkpoint, load_checkpoint
from Qwen2SAM_advance.data.dataset_stage1 import (
    build_coco_dataset,
    build_collate_fn,
)
from Qwen2SAM_advance.training.visualize import FixedSampleVisualizer

from sam3.model.data_misc import BatchedDatapoint
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Masks
from sam3.train.matcher import BinaryHungarianMatcherV2


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def init_wandb(cfg):
    """Initialize Weights & Biases run."""
    import wandb

    wandb_cfg = cfg.get("wandb", {})
    run = wandb.init(
        project=wandb_cfg.get("project", "Qwen2SAM_advance"),
        name=wandb_cfg.get("run_name", None),
        group=wandb_cfg.get("group", "stage1"),
        tags=wandb_cfg.get("tags", ["stage1", "segmentation-recovery"]),
        config=cfg,
        resume="allow",
    )

    # Define custom charts for the dashboard
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("lr/*", step_metric="train/step")

    return run


def log_predictions_to_wandb(
    model, batch, find_stages, epoch, step, max_images=4,
):
    """
    Log prediction visualizations to W&B.

    Shows predicted boxes overlaid on input images with confidence scores.
    """
    import wandb

    try:
        # Denormalize images for visualization
        img_batch = batch.img_batch.cpu()  # (B, 3, H, W)
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        imgs = (img_batch * std + mean).clamp(0, 1)

        # Get predictions from the last stage, last step
        with torch.no_grad():
            from sam3.model.model_misc import SAM3Output
            with SAM3Output.iteration_mode(
                find_stages, iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE
            ) as stages:
                if len(stages) == 0 or len(stages[0]) == 0:
                    return
                preds = stages[0][-1]  # Last step of first stage

        pred_logits = preds.get("pred_logits")  # (B, Q, 1)
        pred_boxes = preds.get("pred_boxes_xyxy")  # (B, Q, 4)

        if pred_logits is None or pred_boxes is None:
            return

        pred_scores = pred_logits.sigmoid().squeeze(-1).cpu()  # (B, Q)
        pred_boxes = pred_boxes.cpu()  # (B, Q, 4)

        wandb_images = []
        n = min(max_images, imgs.shape[0])

        for i in range(n):
            img_np = (imgs[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            H, W = img_np.shape[:2]

            # Get top-k predictions above threshold
            scores_i = pred_scores[i]
            boxes_i = pred_boxes[i]  # (Q, 4) XYXY normalized [0,1]

            # Filter by confidence
            mask = scores_i > 0.3
            if mask.sum() == 0:
                # Show top-5 even if low confidence
                topk = min(5, len(scores_i))
                _, indices = scores_i.topk(topk)
                mask = torch.zeros_like(scores_i, dtype=torch.bool)
                mask[indices] = True

            scores_sel = scores_i[mask].numpy()
            boxes_sel = boxes_i[mask].numpy()

            # Convert normalized XYXY to pixel coords
            boxes_pixel = boxes_sel.copy()
            boxes_pixel[:, [0, 2]] *= W
            boxes_pixel[:, [1, 3]] *= H

            # Build W&B box data
            box_data = []
            for j in range(len(scores_sel)):
                x1, y1, x2, y2 = boxes_pixel[j]
                box_data.append({
                    "position": {
                        "minX": float(x1) / W,
                        "minY": float(y1) / H,
                        "maxX": float(x2) / W,
                        "maxY": float(y2) / H,
                    },
                    "class_id": 0,
                    "scores": {"confidence": float(scores_sel[j])},
                })

            # Get GT boxes
            gt_boxes_data = []
            if len(batch.find_targets) > 0:
                gt = batch.find_targets[0]
                if gt.boxes_padded is not None:
                    gt_boxes_padded = gt.boxes_padded[i].cpu()  # (max_boxes, 4) CxCyWH
                    num_gt = int(gt.num_boxes[i].item()) if gt.num_boxes is not None else 0
                    for j in range(num_gt):
                        cx, cy, w, h = gt_boxes_padded[j].numpy()
                        if w > 0 and h > 0:
                            gt_boxes_data.append({
                                "position": {
                                    "minX": float(cx - w / 2),
                                    "minY": float(cy - h / 2),
                                    "maxX": float(cx + w / 2),
                                    "maxY": float(cy + h / 2),
                                },
                                "class_id": 1,
                            })

            boxes_dict = {
                "predictions": {"box_data": box_data, "class_labels": {0: "pred", 1: "gt"}},
            }
            if gt_boxes_data:
                boxes_dict["ground_truth"] = {
                    "box_data": gt_boxes_data,
                    "class_labels": {0: "pred", 1: "gt"},
                }

            wandb_images.append(wandb.Image(img_np, boxes=boxes_dict))

        wandb.log({
            "predictions": wandb_images,
            "epoch": epoch,
        })

    except Exception as e:
        print(f"Warning: Failed to log predictions to W&B: {e}")


def log_gradient_stats(model):
    """Compute gradient statistics for monitoring."""
    stats = {}
    total_norm = 0.0

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            total_norm += grad_norm ** 2

            # Track per-component gradient norms
            if "adapter" in name:
                key = "adapter"
            elif "encoder" in name and "transformer" in name:
                key = "encoder"
            elif "decoder" in name and "transformer" in name:
                key = "decoder"
            elif "dot_prod_scoring" in name:
                key = "scoring"
            elif "segmentation_head" in name:
                key = "seg_head"
            else:
                continue

            grad_key = f"gradients/{key}_norm"
            stats[grad_key] = stats.get(grad_key, 0.0) + grad_norm ** 2

    stats["gradients/total_norm"] = total_norm ** 0.5
    for k in list(stats.keys()):
        if k != "gradients/total_norm":
            stats[k] = stats[k] ** 0.5

    return stats


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_dataloader, loss_fn, device, epoch, max_batches=100):
    """
    Run validation and return metrics.

    Computes:
      - Average validation loss (and components)
      - Mean confidence of positive predictions
      - Detection quality: recall, precision, mean IoU of matched boxes
    """
    model.eval()
    total_loss = 0.0
    component_losses = {}
    num_batches = 0
    all_scores = []

    # Detection quality tracking
    total_gt = 0
    total_matched = 0
    total_pred = 0
    matched_ious = []

    for step, batch_dict in enumerate(val_dataloader):
        if step >= max_batches:
            break

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

        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                find_stages = model(batch)
                find_targets = [model.back_convert(t) for t in batch.find_targets]
                loss_dict = loss_fn(find_stages, find_targets)

            core_loss = loss_dict["core_loss"]
            if math.isfinite(core_loss.item()):
                total_loss += core_loss.item()
                num_batches += 1

                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor) and k != "core_loss":
                        component_losses[k] = component_losses.get(k, 0.0) + v.item()

            # Collect prediction quality stats
            from sam3.model.model_misc import SAM3Output
            with SAM3Output.iteration_mode(
                find_stages, iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE
            ) as stages:
                if len(stages) > 0 and len(stages[0]) > 0:
                    preds = stages[0][-1]
                    if "pred_logits" in preds:
                        scores = preds["pred_logits"].sigmoid().squeeze(-1)
                        all_scores.append(scores.cpu())

                    # Box matching quality
                    pred_boxes_xyxy = preds.get("pred_boxes_xyxy")
                    if pred_boxes_xyxy is not None and len(batch.find_targets) > 0:
                        gt_target = batch.find_targets[0]
                        if gt_target.boxes_padded is not None and gt_target.num_boxes is not None:
                            for b in range(pred_boxes_xyxy.shape[0]):
                                p_scores = scores[b].cpu().numpy()
                                p_boxes = pred_boxes_xyxy[b].cpu().numpy()
                                n_gt = int(gt_target.num_boxes[b].item())
                                gt_cxcywh = gt_target.boxes_padded[b, :n_gt].cpu().numpy()

                                if n_gt == 0:
                                    continue

                                # Convert GT CxCyWH to XYXY
                                cx, cy, w, h = gt_cxcywh[:, 0], gt_cxcywh[:, 1], gt_cxcywh[:, 2], gt_cxcywh[:, 3]
                                gt_xyxy = np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=1)

                                # Filter preds by confidence > 0.3
                                mask = p_scores > 0.3
                                if mask.sum() == 0:
                                    top_idx = np.argsort(p_scores)[::-1][:10]
                                else:
                                    top_idx = np.where(mask)[0]

                                sel_boxes = p_boxes[top_idx]
                                total_gt += n_gt
                                total_pred += len(sel_boxes)

                                # Greedy IoU matching
                                used_gt = set()
                                for pi in range(len(sel_boxes)):
                                    best_iou = 0
                                    best_gi = -1
                                    for gi in range(n_gt):
                                        if gi in used_gt:
                                            continue
                                        pb = sel_boxes[pi]
                                        gb = gt_xyxy[gi]
                                        inter_x1 = max(pb[0], gb[0])
                                        inter_y1 = max(pb[1], gb[1])
                                        inter_x2 = min(pb[2], gb[2])
                                        inter_y2 = min(pb[3], gb[3])
                                        inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                                        a1 = (pb[2]-pb[0]) * (pb[3]-pb[1])
                                        a2 = (gb[2]-gb[0]) * (gb[3]-gb[1])
                                        iou = inter / max(a1 + a2 - inter, 1e-6)
                                        if iou > best_iou:
                                            best_iou = iou
                                            best_gi = gi
                                    if best_iou >= 0.5:
                                        total_matched += 1
                                        matched_ious.append(best_iou)
                                        used_gt.add(best_gi)

        except Exception as e:
            print(f"Validation error at step {step}: {e}")
            continue

        qwen_encoder.set_image(None)

    model.train()

    if num_batches == 0:
        return {}

    # Aggregate metrics
    metrics = {
        "val/loss": total_loss / num_batches,
    }

    for k, v in component_losses.items():
        metrics[f"val/{k}"] = v / num_batches

    if all_scores:
        scores_cat = torch.cat(all_scores, dim=0).flatten()
        metrics["val/mean_confidence"] = scores_cat.mean().item()
        metrics["val/max_confidence"] = scores_cat.max().item()
        metrics["val/pct_above_0.5"] = (scores_cat > 0.5).float().mean().item()
        metrics["val/pct_above_0.3"] = (scores_cat > 0.3).float().mean().item()

    # Detection quality
    if total_gt > 0:
        metrics["val/recall@0.5"] = total_matched / total_gt
    if total_pred > 0:
        metrics["val/precision@0.5"] = total_matched / total_pred
    if matched_ious:
        metrics["val/mean_matched_iou"] = float(np.mean(matched_ious))

    return metrics


# ---------------------------------------------------------------------------
# Loss & optimizer builders
# ---------------------------------------------------------------------------

def build_loss_fn(cfg, device="cuda"):
    """Build SAM3's loss functions and matcher."""
    loss_cfg = cfg["loss"]

    matcher = BinaryHungarianMatcherV2(
        focal=True,
        cost_class=loss_cfg.get("cost_class", 2.0),
        cost_bbox=loss_cfg.get("cost_bbox", 5.0),
        cost_giou=loss_cfg.get("cost_giou", 2.0),
        alpha=0.25,
        gamma=2,
        stable=False,
    )

    loss_fns = nn.ModuleList()

    # Classification loss
    ce_weight_dict = {"loss_ce": loss_cfg.get("weight_ce", 20.0)}
    if loss_cfg.get("use_presence", True):
        ce_weight_dict["presence_loss"] = loss_cfg.get("presence_weight", 20.0)
    loss_fns.append(IABCEMdetr(
        weight_dict=ce_weight_dict,
        pos_weight=loss_cfg.get("pos_weight", 5.0),
        alpha=0.25,
        gamma=2,
        use_presence=loss_cfg.get("use_presence", True),
    ))

    # Box regression loss
    loss_fns.append(Boxes(
        weight_dict={
            "loss_bbox": loss_cfg.get("weight_bbox", 5.0),
            "loss_giou": loss_cfg.get("weight_giou", 2.0),
        },
    ))

    # Mask loss (optional)
    if cfg.get("enable_masks", False):
        loss_fns.append(Masks(
            weight_dict={
                "loss_mask": loss_cfg.get("weight_mask", 5.0),
                "loss_dice": loss_cfg.get("weight_dice", 1.0),
            },
        ))

    loss_wrapper = Sam3LossWrapper(
        loss_fns_find=loss_fns,
        normalization="local",
        matcher=matcher,
    )

    return loss_wrapper


def build_optimizer(model, cfg):
    """Build optimizer with separate param groups."""
    optim_cfg = cfg["optimizer"]

    param_groups = []
    groups = get_trainable_params(model, print_summary=False)

    lr_map = {
        "adapter": optim_cfg.get("lr_adapter", 1e-4),
        "encoder": optim_cfg.get("lr_encoder", 8e-5),
        "decoder": optim_cfg.get("lr_decoder", 8e-5),
        "scoring": optim_cfg.get("lr_scoring", 8e-5),
        "seg_head": optim_cfg.get("lr_seg_head", 8e-5),
        "queries": optim_cfg.get("lr_queries", 8e-5),
        "lora": optim_cfg.get("lr_lora", 5e-6),
        "other": optim_cfg.get("lr_other", 8e-5),
    }

    for group_name, params in groups.items():
        if len(params) == 0:
            continue
        lr = lr_map.get(group_name, optim_cfg.get("lr", 1e-4))
        param_groups.append({
            "params": [p for _, p in params],
            "lr": lr,
            "name": group_name,
        })

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=optim_cfg.get("weight_decay", 0.05),
    )

    return optimizer


def build_scheduler(optimizer, cfg, steps_per_epoch):
    """Build learning rate scheduler."""
    sched_cfg = cfg.get("scheduler", {})
    warmup_epochs = sched_cfg.get("warmup_epochs", 5)
    total_epochs = cfg["training"]["epochs"]

    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Qwen2SAMTrainer:
    """
    Stage 1 trainer for SAM3 with Qwen text encoder.

    Injects raw PIL images into Qwen before each forward pass,
    so Qwen can see the actual image alongside text prompts.
    """

    def __init__(self, model, loss_fn, optimizer, scheduler, scaler, cfg, device):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.cfg = cfg
        self.device = device
        self.grad_clip = cfg["training"].get("grad_clip", 0.1)
        self.global_step = 0

    def _get_qwen_encoder(self):
        return self.model.backbone.language_backbone

    def _move_batch_to_device(self, batch):
        """Move all tensor fields of a BatchedDatapoint to device."""
        batch.img_batch = batch.img_batch.to(self.device, non_blocking=True)
        for stage_input in batch.find_inputs:
            for attr in ["img_ids", "text_ids", "input_boxes", "input_boxes_label",
                         "input_boxes_mask", "input_points", "input_points_mask"]:
                val = getattr(stage_input, attr, None)
                if val is not None and isinstance(val, torch.Tensor):
                    setattr(stage_input, attr, val.to(self.device, non_blocking=True))
        for stage_target in batch.find_targets:
            for attr in ["num_boxes", "boxes", "boxes_padded", "is_exhaustive",
                         "segments", "is_valid_segment", "object_ids", "object_ids_padded",
                         "repeated_boxes", "semantic_segments"]:
                val = getattr(stage_target, attr, None)
                if val is not None and isinstance(val, torch.Tensor):
                    setattr(stage_target, attr, val.to(self.device, non_blocking=True))

    def train_step(self, batch_dict):
        """Single training step. Returns loss dict and find_stages for logging."""
        self.optimizer.zero_grad(set_to_none=True)

        key = list(batch_dict.keys())[0]
        batch = batch_dict[key]
        self._move_batch_to_device(batch)

        # Set raw PIL images for Qwen
        qwen_encoder = self._get_qwen_encoder()
        if batch.raw_images is not None and len(batch.raw_images) > 0:
            qwen_encoder.set_image(batch.raw_images)
        else:
            qwen_encoder.set_image(None)

        # Forward pass
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            find_stages = self.model(batch)
            find_targets = [
                self.model.back_convert(t) for t in batch.find_targets
            ]
            loss_dict = self.loss_fn(find_stages, find_targets)

        core_loss = loss_dict["core_loss"]

        if not math.isfinite(core_loss.item()):
            print(f"WARNING: Loss is {core_loss.item()}, skipping step")
            qwen_encoder.set_image(None)
            return None, None, None

        # Backward
        self.scaler.scale(core_loss).backward()

        # Gradient stats (before clipping)
        grad_stats = log_gradient_stats(self.model)

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.grad_clip,
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.global_step += 1

        qwen_encoder.set_image(None)

        loss_scalars = {k: v.item() if isinstance(v, torch.Tensor) else v
                        for k, v in loss_dict.items()}

        return loss_scalars, grad_stats, (find_stages, batch)


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train(cfg):
    """Main training function with W&B integration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- W&B init ---- #
    use_wandb = cfg.get("wandb", {}).get("enabled", True)
    if use_wandb:
        import wandb
        run = init_wandb(cfg)
        print(f"W&B run: {run.url}")
    else:
        wandb = None

    print(f"Device: {device}")

    # ---- Build model ---- #
    print("Building model...")
    model = build_sam3_qwen_model(
        qwen_model_name=cfg["model"]["qwen_model_name"],
        qwen_dtype=cfg["model"].get("qwen_dtype", "bfloat16"),
        freeze_qwen=cfg["model"].get("freeze_qwen", True),
        use_lora=cfg["model"].get("use_lora", False),
        lora_r=cfg["model"].get("lora_r", 16),
        lora_alpha=cfg["model"].get("lora_alpha", 32),
        adapter_num_tokens=cfg["model"].get("adapter_num_tokens", 32),
        adapter_num_layers=cfg["model"].get("adapter_num_layers", 2),
        adapter_num_heads=cfg["model"].get("adapter_num_heads", 8),
        enable_segmentation=cfg.get("enable_masks", False),
        eval_mode=False,
        load_sam3_from_hf=True,
        device="cpu",
    )
    model = model.to(device)

    print("\nTrainable parameters:")
    param_groups = get_trainable_params(model, print_summary=True)

    if use_wandb:
        total_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in model.parameters())
        wandb.config.update({
            "trainable_params_M": total_trainable / 1e6,
            "total_params_M": total_params / 1e6,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "gpu_memory_GB": torch.cuda.get_device_properties(0).total_mem / 1e9 if torch.cuda.is_available() else 0,
        })

    # ---- Build datasets ---- #
    print("\nBuilding datasets...")
    data_cfg = cfg["data"]

    train_dataset = build_coco_dataset(
        annotation_file=data_cfg["annotation_file"],
        image_dir=data_cfg["image_dir"],
        resolution=data_cfg.get("resolution", 1008),
        with_masks=cfg.get("enable_masks", False),
    )
    print(f"Train dataset: {len(train_dataset)} samples")

    collate_fn = build_collate_fn(with_masks=cfg.get("enable_masks", False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"].get("batch_size", 1),
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Validation dataset (COCO val2017)
    val_loader = None
    val_dataset = None
    val_cfg = data_cfg.get("val_annotation_file")
    val_img_dir = data_cfg.get("val_image_dir")
    if val_cfg and val_img_dir:
        val_dataset = build_coco_dataset(
            annotation_file=val_cfg,
            image_dir=val_img_dir,
            resolution=data_cfg.get("resolution", 1008),
            with_masks=cfg.get("enable_masks", False),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        print(f"Val dataset: {len(val_dataset)} samples")

    # ---- Fixed sample visualizer ---- #
    visualizer = None
    if use_wandb:
        vis_cfg = cfg.get("visualization", {})
        try:
            visualizer = FixedSampleVisualizer(
                train_dataset=train_dataset,
                collate_fn=collate_fn,
                val_dataset=val_dataset if val_loader is not None else None,
                n_train=vis_cfg.get("n_train_samples", 4),
                n_val=vis_cfg.get("n_val_samples", 4),
                cell_size=vis_cfg.get("cell_size", 256),
            )
        except Exception as e:
            print(f"Warning: Failed to init visualizer: {e}")

    # ---- Build loss, optimizer, scheduler ---- #
    loss_fn = build_loss_fn(cfg, device=device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = GradScaler()

    # ---- Resume ---- #
    start_epoch = 0
    best_val_loss = float("inf")
    ckpt_dir = Path(cfg["training"].get("checkpoint_dir", "checkpoints/stage1"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resume_path = cfg["training"].get("resume_from", None)
    if resume_path and Path(resume_path).exists():
        print(f"Resuming from {resume_path}")
        info = load_checkpoint(model, resume_path, optimizer, device="cpu")
        start_epoch = info["epoch"] + 1
        best_val_loss = info.get("best_metric", float("inf")) or float("inf")

    # ---- Trainer ---- #
    trainer = Qwen2SAMTrainer(
        model=model, loss_fn=loss_fn, optimizer=optimizer,
        scheduler=scheduler, scaler=scaler, cfg=cfg, device=device,
    )
    trainer.global_step = start_epoch * len(train_loader)

    epochs = cfg["training"]["epochs"]
    log_interval = cfg["training"].get("log_interval", 50)
    save_interval = cfg["training"].get("save_interval", 5)
    val_interval = cfg["training"].get("val_interval", 5)
    visual_log_interval = cfg["training"].get("visual_log_interval", 500)

    print(f"\nStarting training: {epochs} epochs, {len(train_loader)} steps/epoch")
    print(f"Checkpoints: {ckpt_dir}")
    if use_wandb:
        print(f"W&B dashboard: {run.url}")

    # ---- Training loop ---- #
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_losses = {}
        epoch_steps = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            try:
                loss_dict, grad_stats, extra = trainer.train_step(batch)
            except Exception as e:
                print(f"Error at epoch {epoch} step {step}: {e}")
                import traceback
                traceback.print_exc()
                continue

            if loss_dict is None:
                continue

            epoch_steps += 1
            for k, v in loss_dict.items():
                if isinstance(v, (int, float)):
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v

            # ---- Log to W&B every step ---- #
            if use_wandb:
                log_data = {
                    "train/step": trainer.global_step,
                    "train/core_loss": loss_dict.get("core_loss", 0),
                }

                # Component losses
                for k, v in loss_dict.items():
                    if k != "core_loss" and isinstance(v, (int, float)):
                        log_data[f"train/{k}"] = v

                # Learning rates per group
                for pg in optimizer.param_groups:
                    if "name" in pg:
                        log_data[f"lr/{pg['name']}"] = pg["lr"]

                # Gradient norms
                if grad_stats:
                    log_data.update(grad_stats)

                # GPU memory
                if torch.cuda.is_available():
                    log_data["system/gpu_memory_GB"] = torch.cuda.max_memory_allocated() / 1e9
                    log_data["system/gpu_memory_reserved_GB"] = torch.cuda.max_memory_reserved() / 1e9

                wandb.log(log_data)

            # ---- Console log ---- #
            if (step + 1) % log_interval == 0:
                avg_loss = epoch_losses.get("core_loss", 0) / max(epoch_steps, 1)
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                steps_per_sec = (step + 1) / elapsed
                print(
                    f"  [Epoch {epoch+1}/{epochs}] "
                    f"Step {step+1}/{len(train_loader)} | "
                    f"Loss: {loss_dict.get('core_loss', 0):.4f} (avg: {avg_loss:.4f}) | "
                    f"LR: {lr:.2e} | "
                    f"{steps_per_sec:.2f} steps/s"
                )

            # ---- Visual predictions log (fixed samples) ---- #
            if use_wandb and visualizer is not None and (trainer.global_step % visual_log_interval == 0):
                visualizer.log_to_wandb(model, device, epoch, trainer.global_step)

        # ---- Epoch summary ---- #
        avg_epoch_loss = epoch_losses.get("core_loss", 0) / max(epoch_steps, 1)
        elapsed = time.time() - t0
        print(
            f"\nEpoch {epoch+1}/{epochs} complete: "
            f"avg_loss={avg_epoch_loss:.4f}, time={elapsed:.1f}s"
        )

        if use_wandb:
            epoch_log = {
                "epoch": epoch + 1,
                "train/epoch_loss": avg_epoch_loss,
                "train/epoch_time_s": elapsed,
                "train/steps_per_sec": epoch_steps / max(elapsed, 1),
            }
            for k, v in epoch_losses.items():
                if k != "core_loss":
                    epoch_log[f"train/epoch_{k}"] = v / max(epoch_steps, 1)
            wandb.log(epoch_log)

        # ---- Validation ---- #
        if val_loader is not None and (epoch + 1) % val_interval == 0:
            print(f"  Running validation...")
            val_metrics = validate(
                model, val_loader, loss_fn, device, epoch,
                max_batches=cfg["training"].get("val_max_batches", 200),
            )
            if val_metrics:
                for k, v in val_metrics.items():
                    print(f"    {k}: {v:.4f}")
                if use_wandb:
                    val_metrics["epoch"] = epoch + 1
                    wandb.log(val_metrics)

                val_loss = val_metrics.get("val/loss", float("inf"))
            else:
                val_loss = avg_epoch_loss
        else:
            val_loss = avg_epoch_loss

        # ---- Save checkpoint ---- #
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, epoch_steps,
                str(ckpt_dir / f"epoch_{epoch+1}.pt"),
                skip_qwen_base=True,
                best_metric=val_loss,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, epoch_steps,
                str(ckpt_dir / "best.pt"),
                skip_qwen_base=True,
                best_metric=best_val_loss,
            )
            print(f"  New best: {best_val_loss:.4f}")
            if use_wandb:
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_epoch"] = epoch + 1

    # ---- Finish ---- #
    print(f"\nTraining complete. Best loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {ckpt_dir}")

    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Training: Qwen2SAM_advance")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
