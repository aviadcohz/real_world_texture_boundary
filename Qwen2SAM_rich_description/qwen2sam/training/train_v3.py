"""
Training loop for Qwen2SAM v3 (multi-token description architecture).

Jointly trains:
  - Qwen2.5-VL LoRA adapters (via LM loss + alignment + optional seg grad)
  - Description Projector (per-token 2048 → 256)
  - SAM3 Fusion Encoder, Object Queries, Seg Head, Scoring Head

Loss = λ_seg · DETR(focal+dice+cls+box) + λ_lm · LM + λ_align · Contrastive

Usage:
  conda activate texture_boundary
  python -m qwen2sam.training.train_v3 --config qwen2sam/configs/v3.yaml
"""

import argparse
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from qwen2sam.models.qwen2sam_v3 import Qwen2SAMv3
from qwen2sam.models.losses_v2 import v2_seg_loss
from qwen2sam.models.losses_v3 import v3_total_loss
from qwen2sam.models.losses import (
    alignment_loss, alignment_loss_with_bank, AlignmentMemoryBank,
)
from qwen2sam.data.dataset_v3 import V3Dataset, V3Collator
from qwen2sam.training.train_phase1 import (
    set_seed,
    load_config,
    get_amp_dtype,
    get_lr,
    AverageMeter,
    WarmupCosineScheduler,
)
from qwen2sam.training.wandb_utils import (
    init_wandb, wandb_active, log_step, log_epoch,
    compute_gradient_norms, get_system_metrics,
    TextureBoundaryVisualizer, finish_wandb,
    log_query_diagnostics, log_mask_diagnostics,
    log_weight_norms, log_loss_ratios, log_learning_health,
)


V3_METRICS = [
    "total", "seg_loss", "lm_loss",
    "alignment_loss", "alignment_acc",
    "mask_iou", "focal", "dice", "cls", "box_l1", "box_giou", "exclusivity",
    "desc_len_a", "desc_len_b",
]


# -------------------------------------------------------------------- #
#  Checkpointing                                                         #
# -------------------------------------------------------------------- #

def save_v3_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, path):
    state = {
        "epoch": epoch,
        "metrics": metrics,
        "projector_state_dict": model.projector.state_dict(),
    }

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

    lora_state = {
        k: v for k, v in model.qwen.state_dict().items()
        if "lora" in k.lower() or "start_seg" in k.lower() or "end_seg" in k.lower()
    }
    state["qwen_lora_state_dict"] = lora_state

    if hasattr(model, "hires_head") and model.hires_head is not None:
        state["hires_head_state_dict"] = model.hires_head.state_dict()
    if hasattr(model, "align_projector") and model.align_projector is not None:
        state["align_projector_state_dict"] = model.align_projector.state_dict()

    state["optimizer_state_dict"] = optimizer.state_dict()
    state["scheduler_state_dict"] = scheduler.state_dict()
    state["scaler_state_dict"] = scaler.state_dict()

    torch.save(state, path)
    print(f"  Checkpoint saved: {path}")


def load_v3_checkpoint(model, optimizer, scheduler, scaler, path):
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

    if "align_projector_state_dict" in ckpt and model.align_projector is not None:
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
                    align_bank=None, global_step=0, visualizer=None, val_collator=None):
    model.qwen.train()
    model.projector.train()
    if model.hires_head is not None:
        model.hires_head.train()
    model.sam3.eval()

    amp_enabled = cfg.get("amp", {}).get("enabled", True)
    amp_dtype = get_amp_dtype(cfg)
    grad_accum = cfg["training"].get("gradient_accumulation_steps", 1)
    max_grad_norm = cfg["training"].get("max_grad_norm", 1.0)
    log_every = cfg.get("logging", {}).get("log_every_n_steps", 10)
    vis_every = cfg.get("logging", {}).get("vis_every_n_steps", 200)
    loss_cfg = cfg.get("loss", {})

    # seg_grad warmup
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

    meters = {k: AverageMeter() for k in V3_METRICS}
    t0 = time.time()

    for step, batch in enumerate(loader):
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

        sam_images = batch["sam_images"]
        gt_a = batch["masks_a"]
        gt_b = batch["masks_b"]
        gt_boxes = batch["gt_boxes_cxcywh"]
        gt_boxes_xyxy = batch["gt_boxes_xyxy"]
        B = sam_images.shape[0]

        # ---- Forward ------------------------------------------------- #
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(
                qwen_inputs, sam_images,
                seg_grad_to_lm=seg_grad_to_lm,
            )

            # Segmentation loss (DETR-style)
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

            # Alignment loss
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
            total = v3_total_loss(
                outputs["lm_loss"], seg_loss,
                alignment_loss_val=align_loss_val,
                lm_weight=loss_cfg.get("lm_weight", 0.5),
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
        # Track description lengths
        meters["desc_len_a"].update(outputs["desc_lengths_a"].float().mean().item(), B)
        meters["desc_len_b"].update(outputs["desc_lengths_b"].float().mean().item(), B)

        # ---- W&B step logging ---------------------------------------- #
        global_step += 1
        is_grad_step = (step + 1) % grad_accum == 0

        if wandb_active() and is_grad_step:
            step_metrics = {k: m.val for k, m in meters.items() if m.count > 0}
            log_step(
                global_step, step_metrics, optimizer=optimizer,
                model=model if is_grad_step else None, stage="v3",
            )
            # Advanced diagnostics every 50 grad steps
            if global_step % 50 == 0:
                log_query_diagnostics(outputs, global_step)
                log_mask_diagnostics(outputs, gt_a, gt_b, global_step)
                log_loss_ratios(step_metrics, global_step)
            # Weight norms every 200 grad steps
            if global_step % 200 == 0:
                log_weight_norms(model, global_step, stage="v3")

        # ---- Visualization ------------------------------------------- #
        if (visualizer is not None and val_collator is not None
                and global_step % vis_every == 0):
            model.qwen.eval()
            model.projector.eval()
            visualizer.log_to_wandb(
                model, val_collator, device, epoch, global_step,
                stage="v3", amp_dtype=amp_dtype,
            )
            model.qwen.train()
            model.projector.train()

        # ---- Log ----------------------------------------------------- #
        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            align_str = ""
            if meters["alignment_loss"].count > 0:
                align_str = f"align={meters['alignment_loss'].avg:.4f} "
            desc_str = f"desc_len={meters['desc_len_a'].avg:.1f}/{meters['desc_len_b'].avg:.1f}"
            print(
                f"  [E{epoch+1}] {step+1}/{len(loader)} | "
                f"loss={meters['total'].avg:.4f} "
                f"{align_str}"
                f"seg={meters['seg_loss'].avg:.4f} "
                f"lm={meters['lm_loss'].avg:.4f} "
                f"iou={meters['mask_iou'].avg:.4f} "
                f"{desc_str} "
                f"lr={get_lr(optimizer):.2e} ({elapsed:.1f}s)"
            )

    return {k: m.avg for k, m in meters.items()}, global_step


# -------------------------------------------------------------------- #
#  Validation                                                            #
# -------------------------------------------------------------------- #

@torch.no_grad()
def validate(model, loader, cfg, device):
    model.qwen.eval()
    model.projector.eval()
    model.sam3.eval()
    if model.hires_head is not None:
        model.hires_head.eval()

    amp_enabled = cfg.get("amp", {}).get("enabled", True)
    amp_dtype = get_amp_dtype(cfg)
    loss_cfg = cfg.get("loss", {})

    meters = {k: AverageMeter() for k in V3_METRICS}

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

        sam_images = batch["sam_images"]
        gt_a, gt_b = batch["masks_a"], batch["masks_b"]
        gt_boxes = batch["gt_boxes_cxcywh"]
        gt_boxes_xyxy = batch["gt_boxes_xyxy"]
        B = sam_images.shape[0]

        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(qwen_inputs, sam_images)

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

            total = v3_total_loss(
                outputs["lm_loss"], seg_loss,
                alignment_loss_val=align_loss_val,
                lm_weight=loss_cfg.get("lm_weight", 0.5),
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
    parser = argparse.ArgumentParser(description="Qwen2SAM v3 Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Build model ------------------------------------------------- #
    print("Building Qwen2SAM v3 model...")
    model = Qwen2SAMv3(cfg, device=str(device))

    param_counts = model.num_trainable_params()
    print("Trainable parameters:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")

    # ---- Data -------------------------------------------------------- #
    data_root = args.data_root or cfg["data"]["data_root"]
    use_augmentation = cfg["data"].get("augmentation", False)
    dataset = V3Dataset(
        data_root=data_root,
        metadata_file=cfg["data"].get("metadata_file", "metadata_phase1.json"),
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

        align_embedder_type = cfg.get("model", {}).get("align_embedder", "sentence")
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
    train_n = cfg["data"].get("train_size", 26)
    val_n = cfg["data"].get("val_size", 0)
    train_set = Subset(dataset, list(range(train_n)))
    if val_n > 0:
        val_set = Subset(dataset, list(range(train_n, train_n + val_n)))
    else:
        val_start = train_n
        val_end = min(train_n + 10, len(dataset))
        val_set = Subset(dataset, list(range(val_start, val_end)))
    print(f"Train: {len(train_set)} | Val: {len(val_set)}")

    # Collators
    data_cfg = cfg["data"]
    sys_prompt = data_cfg.get("system_prompt", "You are a texture boundary segmentation assistant.")
    usr_prompt = data_cfg.get("user_prompt", "Describe and segment the texture transition in this image.")

    train_collator = V3Collator(
        processor=model.processor,
        system_prompt=sys_prompt, user_prompt=usr_prompt,
        token_ids=model.token_ids,
        text_embedder=text_embedder,
        inference=False,
    )
    val_collator = V3Collator(
        processor=model.processor,
        system_prompt=sys_prompt, user_prompt=usr_prompt,
        token_ids=model.token_ids,
        inference=True,
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
        total_epochs=train_cfg.get("num_epochs", 100),
        min_lr=train_cfg.get("min_lr", 1e-6),
        steps_per_epoch=steps_per_epoch,
    )

    amp_enabled = cfg.get("amp", {}).get("enabled", True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # ---- Resume ------------------------------------------------------ #
    start_epoch = 0
    if args.resume:
        start_epoch = load_v3_checkpoint(
            model, optimizer, scheduler, scaler, args.resume
        ) + 1

    # ---- Checkpoint dir ---------------------------------------------- #
    ckpt_cfg = cfg.get("checkpoint", {})
    ckpt_dir = Path(ckpt_cfg.get("dir", "checkpoints/v3"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Alignment memory bank --------------------------------------- #
    bank_size = loss_cfg.get("alignment_bank_size", 0)
    align_bank = None
    if bank_size > 0 and loss_cfg.get("alignment_weight", 0) > 0:
        align_dim = cfg["model"].get("align_embed_dim", 0)
        if align_dim == 0:
            qwen_cfg = getattr(model.qwen.config, "text_config", model.qwen.config)
            align_dim = qwen_cfg.hidden_size
        align_bank = AlignmentMemoryBank(bank_size=bank_size, embed_dim=align_dim)
        print(f"  Alignment memory bank: size={bank_size}, dim={align_dim}")

    # ---- W&B init --------------------------------------------------- #
    use_wandb = init_wandb(cfg, stage="v3")
    if use_wandb:
        print("  W&B logging enabled")

    # ---- Visualizer -------------------------------------------------- #
    visualizer = None
    if use_wandb:
        vis_cfg = cfg.get("visualization", {})
        visualizer = TextureBoundaryVisualizer(
            train_set, val_set,
            n_train=vis_cfg.get("n_train_samples", 4),
            n_val=vis_cfg.get("n_val_samples", 4),
            cell_size=vis_cfg.get("cell_size", 320),
        )

    # ---- Training loop ----------------------------------------------- #
    num_epochs = train_cfg.get("num_epochs", 100)
    best_val_iou = 0.0
    training_history = []
    global_step = 0

    print(f"\n{'='*60}")
    print("Qwen2SAM v3 — Multi-Token Description Training")
    print(f"  Max description tokens: {model.max_desc_len}")
    print(f"  Loss: Focal({loss_cfg.get('focal_weight', 5)}) + "
          f"Dice({loss_cfg.get('dice_weight', 1)}) + "
          f"LM({loss_cfg.get('lm_weight', 0.5)}) + "
          f"Align({loss_cfg.get('alignment_weight', 1)})")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, num_epochs):
        t_epoch = time.time()

        train_metrics, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, cfg, device, epoch,
            align_bank=align_bank,
            global_step=global_step,
            visualizer=visualizer,
            val_collator=val_collator,
        )

        val_metrics = {}
        val_every = cfg.get("validation", {}).get("every_n_epochs", 1)
        if (epoch + 1) % val_every == 0:
            val_metrics = validate(model, val_loader, cfg, device)

        elapsed = time.time() - t_epoch

        train_str = (
            f"Train: loss={train_metrics['total']:.4f} "
            f"seg={train_metrics['seg_loss']:.4f} "
            f"lm={train_metrics['lm_loss']:.4f} "
            f"iou={train_metrics['mask_iou']:.4f}"
        )
        val_str = ""
        if val_metrics:
            val_str = (
                f" | Val: loss={val_metrics['total']:.4f} "
                f"iou={val_metrics['mask_iou']:.4f}"
            )

        print(f"\nEpoch {epoch+1}/{num_epochs} ({elapsed:.1f}s) | {train_str}{val_str}")

        # W&B epoch logging
        log_epoch(epoch, train_metrics, val_metrics, best_val_iou, elapsed)
        log_learning_health(train_metrics, val_metrics, epoch)

        # Epoch-level visualization (every 10 epochs)
        vis_epoch_every = cfg.get("visualization", {}).get("every_n_epochs", 10)
        if (visualizer is not None and val_collator is not None
                and (epoch + 1) % vis_epoch_every == 0):
            model.qwen.eval()
            model.projector.eval()
            amp_dtype = get_amp_dtype(cfg)
            visualizer.log_to_wandb(
                model, val_collator, device, epoch, global_step,
                stage="v3", amp_dtype=amp_dtype,
            )
            model.qwen.train()
            model.projector.train()

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["total"],
            "train_iou": train_metrics["mask_iou"],
        }
        if val_metrics:
            epoch_record["val_loss"] = val_metrics["total"]
            epoch_record["val_iou"] = val_metrics["mask_iou"]
        training_history.append(epoch_record)

        from qwen2sam.training.train_phase1 import plot_training_curves
        plot_training_curves(training_history, str(ckpt_dir), phase_name="V3 Multi-Token")

        save_every = ckpt_cfg.get("save_every_n_epochs", 5)
        if (epoch + 1) % save_every == 0:
            save_v3_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
                str(ckpt_dir / f"epoch_{epoch+1:04d}.pt"),
            )

        if val_metrics and val_metrics.get("mask_iou", 0) > best_val_iou:
            best_val_iou = val_metrics["mask_iou"]
            save_v3_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                val_metrics,
                str(ckpt_dir / "best.pt"),
            )
            print(f"  New best model (val_iou={best_val_iou:.4f})")

    print(f"\n{'='*60}")
    print(f"Training complete. Best val IoU: {best_val_iou:.4f}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"{'='*60}")

    finish_wandb()


if __name__ == "__main__":
    main()
