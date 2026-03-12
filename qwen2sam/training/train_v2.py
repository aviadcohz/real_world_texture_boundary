"""
Training loop for Qwen2SAM v2 (SAM3 DETR-based architecture).

Jointly trains:
  - Qwen2.5-VL LoRA adapters (via contrastive alignment loss)
  - MLP Projector (2048 → 256)
  - SAM3 Fusion Encoder, Object Queries, Seg Head, Scoring Head

Loss = λ_seg · DETR(focal+dice+cls+box) + λ_align · Contrastive

Usage:
  conda activate texture_boundary
  python -m qwen2sam.training.train_v2 --config qwen2sam/configs/v2.yaml
"""

import argparse
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset, random_split

from qwen2sam.models.qwen2sam_v2 import Qwen2SAMv2
from qwen2sam.models.losses_v2 import v2_seg_loss, v2_total_loss
from qwen2sam.models.losses import alignment_loss, alignment_loss_with_bank, AlignmentMemoryBank
from qwen2sam.data.dataset_v2 import V2Dataset, V2Collator
from qwen2sam.training.train_phase1 import (
    set_seed,
    load_config,
    load_splits,
    get_amp_dtype,
    get_lr,
    AverageMeter,
    WarmupCosineScheduler,
)


# -------------------------------------------------------------------- #
#  Metrics                                                               #
# -------------------------------------------------------------------- #

V2_METRICS = [
    "total", "seg_loss", "lm_loss",
    "alignment_loss", "alignment_acc",
    "mask_iou", "focal", "dice", "cls", "box_l1", "box_giou", "exclusivity",
]


# -------------------------------------------------------------------- #
#  Checkpointing                                                         #
# -------------------------------------------------------------------- #

def save_v2_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, path):
    """Save v2 checkpoint with all trainable components."""
    state = {
        "epoch": epoch,
        "metrics": metrics,
        "projector_state_dict": model.projector.state_dict(),
    }

    # SAM3 trainable params
    sam3_trainable = {
        k: v for k, v in model.sam3.state_dict().items()
        if any(
            k.startswith(prefix) for prefix in [
                "transformer.encoder.",
                "transformer.decoder.query_embed",
                "transformer.decoder.bbox_embed",
                "segmentation_head.",
                "dot_prod_scoring.",
                "class_embed",
            ]
        )
    }
    state["sam3_trainable_state_dict"] = sam3_trainable

    # LoRA weights
    lora_state = {
        k: v for k, v in model.qwen.state_dict().items()
        if "lora" in k.lower() or "seg_a" in k.lower() or "seg_b" in k.lower()
    }
    state["qwen_lora_state_dict"] = lora_state

    # HiRes head (if it exists)
    if hasattr(model, "hires_head") and model.hires_head is not None:
        state["hires_head_state_dict"] = model.hires_head.state_dict()

    # Align projector (if it exists)
    if hasattr(model, "align_projector") and model.align_projector is not None:
        state["align_projector_state_dict"] = model.align_projector.state_dict()

    # Optimizer / scheduler / scaler
    state["optimizer_state_dict"] = optimizer.state_dict()
    state["scheduler_state_dict"] = scheduler.state_dict()
    state["scaler_state_dict"] = scaler.state_dict()

    torch.save(state, path)
    print(f"  Checkpoint saved: {path}")


def load_v2_checkpoint(model, optimizer, scheduler, scaler, path):
    """Load v2 checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    model.projector.load_state_dict(ckpt["projector_state_dict"])

    if "sam3_trainable_state_dict" in ckpt:
        missing, unexpected = model.sam3.load_state_dict(
            ckpt["sam3_trainable_state_dict"], strict=False
        )
        print(f"  SAM3 loaded ({len(missing)} missing, {len(unexpected)} unexpected)")

    if "qwen_lora_state_dict" in ckpt:
        missing, unexpected = model.qwen.load_state_dict(
            ckpt["qwen_lora_state_dict"], strict=False
        )
        print(f"  LoRA loaded ({len(missing)} missing, {len(unexpected)} unexpected)")

    if "hires_head_state_dict" in ckpt and hasattr(model, "hires_head") and model.hires_head is not None:
        model.hires_head.load_state_dict(ckpt["hires_head_state_dict"])
        print("  HiRes head loaded")

    if "align_projector_state_dict" in ckpt and hasattr(model, "align_projector") and model.align_projector is not None:
        model.align_projector.load_state_dict(ckpt["align_projector_state_dict"])
        print("  Align projector loaded")

    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])

    epoch = ckpt["epoch"]
    print(f"  Resumed from epoch {epoch + 1}")
    return epoch


# -------------------------------------------------------------------- #
#  Training step                                                         #
# -------------------------------------------------------------------- #

def train_one_epoch(model, loader, optimizer, scheduler, scaler, cfg, device, epoch,
                    align_bank=None):
    """Run one training epoch."""
    model.qwen.train()
    model.projector.train()
    if model.hires_head is not None:
        model.hires_head.train()
    # SAM3 stays in eval mode to disable activation checkpointing
    # (which conflicts with AMP autocast). Gradients still flow for
    # trainable params since requires_grad is set independently.
    model.sam3.eval()

    amp_enabled = cfg.get("amp", {}).get("enabled", True)
    amp_dtype = get_amp_dtype(cfg)
    grad_accum = cfg["training"].get("gradient_accumulation_steps", 1)
    max_grad_norm = cfg["training"].get("max_grad_norm", 1.0)
    log_every = cfg.get("logging", {}).get("log_every_n_steps", 10)
    loss_cfg = cfg.get("loss", {})

    # seg_grad warmup: disable seg gradients to LoRA for first N epochs
    seg_grad_to_lm_base = cfg["training"].get("seg_grad_to_lm", False)
    seg_grad_warmup = cfg["training"].get("seg_grad_warmup_epochs", 0)
    if seg_grad_to_lm_base and seg_grad_warmup > 0 and epoch < seg_grad_warmup:
        seg_grad_to_lm = False
        if epoch == 0:
            print(f"  seg_grad_to_lm: DISABLED (warmup until epoch {seg_grad_warmup})")
    else:
        seg_grad_to_lm = seg_grad_to_lm_base
        if seg_grad_to_lm_base and seg_grad_warmup > 0 and epoch == seg_grad_warmup:
            print(f"  seg_grad_to_lm: ENABLED (warmup complete)")

    meters = {k: AverageMeter() for k in V2_METRICS}
    t0 = time.time()

    for step, batch in enumerate(loader):
        # ---- Move to device ------------------------------------------ #
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

        # Skip samples with missing SEG positions
        valid = (seg_a_pos >= 0) & (seg_b_pos >= 0)
        if not valid.any():
            continue
        if not valid.all():
            vi = valid.nonzero(as_tuple=True)[0]
            qwen_inputs = {k: v[vi] for k, v in qwen_inputs.items()}
            seg_a_pos = seg_a_pos[vi]
            seg_b_pos = seg_b_pos[vi]
            for key in ["sam_images", "masks_a", "masks_b",
                        "gt_boxes_cxcywh", "gt_boxes_xyxy",
                        "align_target_a", "align_target_b"]:
                if key in batch:
                    batch[key] = batch[key][vi]

        sam_images = batch["sam_images"]
        gt_a = batch["masks_a"]
        gt_b = batch["masks_b"]
        gt_boxes = batch["gt_boxes_cxcywh"]
        gt_boxes_xyxy = batch["gt_boxes_xyxy"]
        B = sam_images.shape[0]

        # ---- Forward ------------------------------------------------- #
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(
                qwen_inputs, sam_images, seg_a_pos, seg_b_pos,
                seg_grad_to_lm=seg_grad_to_lm,
            )

            # Segmentation loss (DETR-style) — same box for both textures
            seg_loss, seg_metrics = v2_seg_loss(
                outputs, gt_a, gt_b,
                gt_boxes, gt_boxes, gt_boxes_xyxy, gt_boxes_xyxy,
                focal_weight=loss_cfg.get("focal_weight", 5.0),
                dice_weight=loss_cfg.get("dice_weight", 1.0),
                cls_weight=loss_cfg.get("cls_weight", 2.0),
                box_l1_weight=loss_cfg.get("box_l1_weight", 5.0),
                box_giou_weight=loss_cfg.get("box_giou_weight", 2.0),
                exclusivity_weight=loss_cfg.get("exclusivity_weight", 0.5),
            )

            # Alignment loss (with optional memory bank for richer negatives)
            align_loss_val = None
            align_metrics = {}
            if "align_target_a" in batch and outputs.get("align_a") is not None:
                if align_bank is not None:
                    align_loss_val, align_metrics = alignment_loss_with_bank(
                        outputs["align_a"], outputs["align_b"],
                        batch["align_target_a"], batch["align_target_b"],
                        bank=align_bank,
                        temperature=loss_cfg.get("alignment_temperature", 0.07),
                    )
                else:
                    align_loss_val, align_metrics = alignment_loss(
                        outputs["align_a"], outputs["align_b"],
                        batch["align_target_a"], batch["align_target_b"],
                        temperature=loss_cfg.get("alignment_temperature", 0.07),
                    )

            # Combined loss
            total = v2_total_loss(
                outputs["lm_loss"], seg_loss,
                alignment_loss_val=align_loss_val,
                lm_weight=loss_cfg.get("lm_weight", 0.0),
                seg_weight=loss_cfg.get("seg_weight", 1.0),
                alignment_weight=loss_cfg.get("alignment_weight", 1.0),
            )
            total = total / grad_accum

        # ---- Backward ------------------------------------------------ #
        scaler.scale(total).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            all_params = []
            for group in optimizer.param_groups:
                all_params.extend(group["params"])
            torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        # ---- Metrics ------------------------------------------------- #
        meters["total"].update(total.item() * grad_accum, B)
        meters["seg_loss"].update(seg_metrics["seg_total"], B)
        meters["lm_loss"].update(outputs["lm_loss"].item(), B)
        meters["mask_iou"].update(seg_metrics["mask_iou"], B)
        for k in ["focal", "dice", "cls", "box_l1", "box_giou", "exclusivity"]:
            meters[k].update(seg_metrics.get(k, 0), B)

        if align_metrics:
            for k in ["alignment_loss", "alignment_acc"]:
                if k in align_metrics:
                    meters[k].update(align_metrics[k], B)

        # ---- Log ----------------------------------------------------- #
        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            align_str = ""
            if meters["alignment_loss"].count > 0:
                align_str = f"align={meters['alignment_loss'].avg:.4f} "
            print(
                f"  [E{epoch+1}] {step+1}/{len(loader)} | "
                f"loss={meters['total'].avg:.4f} "
                f"{align_str}"
                f"seg={meters['seg_loss'].avg:.4f} "
                f"iou={meters['mask_iou'].avg:.4f} "
                f"lr={get_lr(optimizer):.2e} ({elapsed:.1f}s)"
            )

    return {k: m.avg for k, m in meters.items()}


# -------------------------------------------------------------------- #
#  Validation                                                            #
# -------------------------------------------------------------------- #

@torch.no_grad()
def validate(model, loader, cfg, device):
    """Run validation."""
    model.qwen.eval()
    model.projector.eval()
    model.sam3.eval()
    if model.hires_head is not None:
        model.hires_head.eval()

    amp_enabled = cfg.get("amp", {}).get("enabled", True)
    amp_dtype = get_amp_dtype(cfg)
    loss_cfg = cfg.get("loss", {})

    meters = {k: AverageMeter() for k in V2_METRICS}

    for batch in loader:
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
            continue
        if not valid.all():
            vi = valid.nonzero(as_tuple=True)[0]
            qwen_inputs = {k: v[vi] for k, v in qwen_inputs.items()}
            seg_a_pos, seg_b_pos = seg_a_pos[vi], seg_b_pos[vi]
            for key in ["sam_images", "masks_a", "masks_b",
                        "gt_boxes_cxcywh", "gt_boxes_xyxy",
                        "align_target_a", "align_target_b"]:
                if key in batch:
                    batch[key] = batch[key][vi]

        sam_images = batch["sam_images"]
        gt_a, gt_b = batch["masks_a"], batch["masks_b"]
        gt_boxes = batch["gt_boxes_cxcywh"]
        gt_boxes_xyxy = batch["gt_boxes_xyxy"]
        B = sam_images.shape[0]

        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(qwen_inputs, sam_images, seg_a_pos, seg_b_pos)

            seg_loss, seg_metrics = v2_seg_loss(
                outputs, gt_a, gt_b,
                gt_boxes, gt_boxes, gt_boxes_xyxy, gt_boxes_xyxy,
                focal_weight=loss_cfg.get("focal_weight", 5.0),
                dice_weight=loss_cfg.get("dice_weight", 1.0),
                cls_weight=loss_cfg.get("cls_weight", 2.0),
                box_l1_weight=loss_cfg.get("box_l1_weight", 5.0),
                box_giou_weight=loss_cfg.get("box_giou_weight", 2.0),
                exclusivity_weight=loss_cfg.get("exclusivity_weight", 0.5),
            )

            align_loss_val = None
            align_metrics = {}
            if "align_target_a" in batch and outputs.get("align_a") is not None:
                align_loss_val, align_metrics = alignment_loss(
                    outputs["align_a"], outputs["align_b"],
                    batch["align_target_a"], batch["align_target_b"],
                )

            total = v2_total_loss(
                outputs["lm_loss"], seg_loss,
                alignment_loss_val=align_loss_val,
                lm_weight=loss_cfg.get("lm_weight", 0.0),
                seg_weight=loss_cfg.get("seg_weight", 1.0),
                alignment_weight=loss_cfg.get("alignment_weight", 1.0),
            )

        meters["total"].update(total.item(), B)
        meters["seg_loss"].update(seg_metrics["seg_total"], B)
        meters["lm_loss"].update(outputs["lm_loss"].item(), B)
        meters["mask_iou"].update(seg_metrics["mask_iou"], B)
        for k in ["focal", "dice", "cls", "box_l1", "box_giou", "exclusivity"]:
            meters[k].update(seg_metrics.get(k, 0), B)

        if align_metrics:
            for k in ["alignment_loss", "alignment_acc"]:
                if k in align_metrics:
                    meters[k].update(align_metrics[k], B)

    return {k: m.avg for k, m in meters.items()}


# -------------------------------------------------------------------- #
#  Main                                                                  #
# -------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Qwen2SAM v2 Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--splits", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Build model ------------------------------------------------- #
    print("Building Qwen2SAM v2 model...")
    model = Qwen2SAMv2(cfg, device=str(device))

    param_counts = model.num_trainable_params()
    print("Trainable parameters:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")

    # ---- Data -------------------------------------------------------- #
    data_root = args.data_root or cfg["data"]["data_root"]
    use_augmentation = cfg["data"].get("augmentation", False)
    dataset = V2Dataset(
        data_root=data_root,
        metadata_file=cfg["data"].get("metadata_file", "metadata.json"),
        image_size=cfg["model"].get("image_size", 1008),
        augment=use_augmentation,
    )

    # Alignment embedder
    loss_cfg = cfg.get("loss", {})
    text_embedder = None
    if loss_cfg.get("alignment_weight", 0.0) > 0:
        all_textures = []
        for entry in dataset.metadata:
            all_textures.append(entry["texture_a"])
            all_textures.append(entry["texture_b"])

        align_embedder_type = cfg.get("model", {}).get("align_embedder", "qwen")
        if align_embedder_type == "sentence":
            from qwen2sam.models.alignment import SentenceTextEmbedder
            align_model_name = cfg.get("model", {}).get("align_model_name", "all-mpnet-base-v2")
            text_embedder = SentenceTextEmbedder(model_name=align_model_name)
            text_embedder.precompute(texture_labels=all_textures)
        else:
            from qwen2sam.models.alignment import QwenTextEmbedder
            text_embedder = QwenTextEmbedder()
            text_embedder.precompute(texture_labels=all_textures, qwen_model=model.qwen,
                                     processor=model.processor, device=device)

    # Train/val split
    if args.splits:
        splits = load_splits(args.splits)
        train_set = Subset(dataset, splits["train_indices"])
        val_set = Subset(dataset, splits["test_indices"])
    elif "train_size" in cfg["data"]:
        # Fixed split: train / val / test
        train_n = cfg["data"]["train_size"]
        val_n = cfg["data"].get("val_size", train_n)  # default val_size = train_size
        train_set = Subset(dataset, list(range(train_n)))
        if val_n > 0:
            val_set = Subset(dataset, list(range(train_n, train_n + val_n)))
        else:
            # Use 10 test samples for validation when val_size=0
            val_start = train_n
            val_end = min(train_n + 10, len(dataset))
            val_set = Subset(dataset, list(range(val_start, val_end)))
        test_start = train_n + max(val_n, 0)
        test_indices = list(range(test_start, len(dataset)))
        print(f"Test set: {len(test_indices)} samples (indices {test_start}..{len(dataset)-1})")
    else:
        val_frac = cfg["data"].get("val_split", 0.1)
        val_size = int(len(dataset) * val_frac)
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.get("seed", 42)),
        )
    print(f"Train: {len(train_set)} | Val: {len(val_set)}")

    # Collators: train uses GT text, val uses generic inference template
    data_cfg = cfg["data"]
    sys_prompt = data_cfg.get(
        "system_prompt",
        "You are a texture boundary segmentation assistant. "
        "Identify textures and mark with <SEG_A> and <SEG_B> tokens.",
    )
    usr_prompt = data_cfg.get(
        "user_prompt",
        "Describe and segment the texture transition in this image.",
    )
    train_collator = V2Collator(
        processor=model.processor,
        system_prompt=sys_prompt, user_prompt=usr_prompt,
        seg_a_id=model.seg_a_id, seg_b_id=model.seg_b_id,
        text_embedder=text_embedder,
        inference=False,  # GT texture names for training
    )
    val_collator = V2Collator(
        processor=model.processor,
        system_prompt=sys_prompt, user_prompt=usr_prompt,
        seg_a_id=model.seg_a_id, seg_b_id=model.seg_b_id,
        inference=True,  # generic template — no GT leakage
    )

    batch_size = cfg["training"].get("batch_size", 1)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=cfg["data"].get("num_workers", 2),
        pin_memory=True, collate_fn=train_collator, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=cfg["data"].get("num_workers", 2),
        pin_memory=True, collate_fn=val_collator,
    )

    # ---- Optimizer --------------------------------------------------- #
    train_cfg = cfg["training"]
    base_lr = train_cfg.get("learning_rate", 1e-4)
    param_groups = model.get_parameter_groups(base_lr)
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    steps_per_epoch = max(
        len(train_loader) // train_cfg.get("gradient_accumulation_steps", 1), 1
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=train_cfg.get("warmup_epochs", 2),
        total_epochs=train_cfg.get("num_epochs", 50),
        min_lr=train_cfg.get("min_lr", 1e-6),
        steps_per_epoch=steps_per_epoch,
    )

    amp_enabled = cfg.get("amp", {}).get("enabled", True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # ---- Resume ------------------------------------------------------ #
    start_epoch = 0
    if args.resume:
        start_epoch = load_v2_checkpoint(
            model, optimizer, scheduler, scaler, args.resume
        ) + 1

    # ---- Checkpoint dir ---------------------------------------------- #
    ckpt_cfg = cfg.get("checkpoint", {})
    ckpt_dir = Path(ckpt_cfg.get("dir", "checkpoints/v2"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training loop ----------------------------------------------- #
    num_epochs = train_cfg.get("num_epochs", 50)
    best_val_iou = 0.0
    training_history = []

    # Alignment memory bank (optional)
    bank_size = loss_cfg.get("alignment_bank_size", 0)
    align_bank = None
    if bank_size > 0 and loss_cfg.get("alignment_weight", 0) > 0:
        align_dim = cfg["model"].get("align_embed_dim", 0)
        if align_dim == 0:
            # Use Qwen hidden dim
            qwen_cfg = getattr(model.qwen.config, "text_config", model.qwen.config)
            align_dim = qwen_cfg.hidden_size
        align_bank = AlignmentMemoryBank(bank_size=bank_size, embed_dim=align_dim)
        print(f"  Alignment memory bank: size={bank_size}, dim={align_dim}")

    print(f"\n{'='*60}")
    print("Qwen2SAM v2 — Training Start")
    print(f"  Architecture: Qwen + LoRA → Projector → SAM3 Fusion+DETR")
    print(f"  Loss: Focal({loss_cfg.get('focal_weight', 5)}) + "
          f"Dice({loss_cfg.get('dice_weight', 1)}) + "
          f"Cls + Box + Alignment")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, num_epochs):
        t_epoch = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, cfg, device, epoch,
            align_bank=align_bank,
        )

        val_metrics = {}
        val_every = cfg.get("validation", {}).get("every_n_epochs", 1)
        if (epoch + 1) % val_every == 0:
            val_metrics = validate(model, val_loader, cfg, device)

        elapsed = time.time() - t_epoch

        # Summary
        train_str = (
            f"Train: loss={train_metrics['total']:.4f} "
            f"seg={train_metrics['seg_loss']:.4f} "
            f"iou={train_metrics['mask_iou']:.4f}"
        )
        val_str = ""
        if val_metrics:
            val_str = (
                f" | Val: loss={val_metrics['total']:.4f} "
                f"iou={val_metrics['mask_iou']:.4f}"
            )

        print(f"\nEpoch {epoch+1}/{num_epochs} ({elapsed:.1f}s) | {train_str}{val_str}")

        # Record history
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["total"],
            "train_iou": train_metrics["mask_iou"],
        }
        if val_metrics:
            epoch_record["val_loss"] = val_metrics["total"]
            epoch_record["val_iou"] = val_metrics["mask_iou"]
        training_history.append(epoch_record)

        # Plot training curves
        from qwen2sam.training.train_phase1 import plot_training_curves
        plot_training_curves(training_history, str(ckpt_dir), phase_name="V2 SAM3")

        # Save
        save_every = ckpt_cfg.get("save_every_n_epochs", 5)
        if (epoch + 1) % save_every == 0:
            save_v2_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
                str(ckpt_dir / f"epoch_{epoch+1:04d}.pt"),
            )

        if val_metrics and val_metrics.get("mask_iou", 0) > best_val_iou:
            best_val_iou = val_metrics["mask_iou"]
            save_v2_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                val_metrics,
                str(ckpt_dir / "best.pt"),
            )
            print(f"  New best model (val_iou={best_val_iou:.4f})")

    print(f"\n{'='*60}")
    print(f"Training complete. Best val IoU: {best_val_iou:.4f}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
