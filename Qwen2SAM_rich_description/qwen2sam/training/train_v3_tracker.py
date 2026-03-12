"""
Training loop for Qwen2SAM v3_tracker (Stage 2: Multi-Token DETR + SAM3 Tracker).

Jointly trains:
  - Qwen2.5-VL LoRA adapters (via alignment + tracker gradient)
  - CoordHead MLP (via tracker mask loss gradient)
  - SAM3 Tracker heads: sam_prompt_encoder + sam_mask_decoder
  - (Optional) DETR components at lower LR

Loss = detr_weight · DETR_loss + tracker_weight · Tracker_loss + align_weight · Alignment

Usage:
  conda activate texture_boundary
  python -m qwen2sam.training.train_v3_tracker --config qwen2sam/configs/v3_tracker.yaml
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from qwen2sam.models.qwen2sam_v3_tracker import Qwen2SAMv3Tracker
from qwen2sam.models.qwen2sam_v2_tracker import POINT_TOKENS_A, POINT_TOKENS_B
from qwen2sam.models.losses_v2_tracker import v2_tracker_total_loss
from qwen2sam.data.dataset_v3 import V3Dataset, V3Collator
from qwen2sam.data.dataset_phase3 import create_labels
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
    TextureBoundaryVisualizer, finish_wandb,
    log_query_diagnostics, log_mask_diagnostics,
    log_weight_norms, log_loss_ratios, log_learning_health,
)


# -------------------------------------------------------------------- #
#  V3 Tracker Collator (V3 bracket tokens + POINT tokens)                #
# -------------------------------------------------------------------- #

class V3TrackerCollator:
    """
    Collate function for v3_tracker training.

    Extends V3Collator pattern with POINT tokens appended to assistant text.
    """

    INFERENCE_TEMPLATE = (
        "The transition is from "
        "<START_SEG_A> first texture region <END_SEG_A> "
        "to "
        "<START_SEG_B> second texture region <END_SEG_B>."
    )

    def __init__(
        self,
        processor,
        system_prompt: str,
        user_prompt: str,
        token_ids: dict,
        point_token_ids: list[int],
        num_points_per_texture: int = 4,
        text_embedder=None,
        inference: bool = False,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.token_ids = token_ids
        self.point_token_ids = point_token_ids
        self.num_points = num_points_per_texture
        self.text_embedder = text_embedder
        self.inference = inference

        self.point_str_a = " ".join(POINT_TOKENS_A[:num_points_per_texture])
        self.point_str_b = " ".join(POINT_TOKENS_B[:num_points_per_texture])

    def _make_assistant_text(self, sample: dict) -> str:
        if self.inference:
            base = self.INFERENCE_TEMPLATE
        else:
            base = sample["assistant_text"]
        return f"{base} Points: {self.point_str_a} {self.point_str_b}"

    def __call__(self, samples: list[dict]) -> dict:
        texts = []
        images = []
        for s in samples:
            assistant_text = self._make_assistant_text(s)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.user_prompt},
                    ],
                },
                {"role": "assistant", "content": assistant_text},
            ]
            texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            )
            images.append(s["image"])

        qwen_inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True,
        )

        labels = create_labels(
            qwen_inputs["input_ids"], qwen_inputs["attention_mask"],
            self.tokenizer,
        )

        # Find POINT token positions
        point_positions = []
        for i in range(len(samples)):
            ids = qwen_inputs["input_ids"][i]
            positions = []
            for tok_id in self.point_token_ids:
                pos = (ids == tok_id).nonzero(as_tuple=True)[0]
                positions.append(pos[-1].item() if len(pos) > 0 else -1)
            point_positions.append(positions)

        sam_images = torch.stack([s["sam_image"] for s in samples])
        masks_a = torch.stack([s["mask_a"] for s in samples])
        masks_b = torch.stack([s["mask_b"] for s in samples])
        gt_boxes_cxcywh = torch.stack([s["gt_box_cxcywh"] for s in samples])
        gt_boxes_xyxy = torch.stack([s["gt_box_xyxy"] for s in samples])

        batch_dict = {
            **{k: v for k, v in qwen_inputs.items()},
            "labels": labels,
            "point_positions": torch.tensor(point_positions, dtype=torch.long),
            "sam_images": sam_images,
            "masks_a": masks_a,
            "masks_b": masks_b,
            "gt_boxes_cxcywh": gt_boxes_cxcywh,
            "gt_boxes_xyxy": gt_boxes_xyxy,
        }

        if self.text_embedder is not None:
            target_a = torch.stack([self.text_embedder[s["texture_a"]] for s in samples])
            target_b = torch.stack([self.text_embedder[s["texture_b"]] for s in samples])
            batch_dict["align_target_a"] = target_a
            batch_dict["align_target_b"] = target_b

        return batch_dict


# -------------------------------------------------------------------- #
#  Metrics                                                               #
# -------------------------------------------------------------------- #

TRACKER_METRICS = [
    "total", "detr_loss", "tracker_loss", "align_loss", "lm_loss",
    "detr_mask_iou", "tracker_iou",
    "alignment_loss", "alignment_acc",
]


# -------------------------------------------------------------------- #
#  Checkpointing                                                         #
# -------------------------------------------------------------------- #

def save_v3_tracker_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, path):
    state = {
        "epoch": epoch,
        "metrics": metrics,
        "projector_state_dict": model.base.projector.state_dict(),
    }

    sam3_trainable = {
        k: v for k, v in model.base.sam3.state_dict().items()
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
        k: v for k, v in model.base.qwen.state_dict().items()
        if "lora" in k.lower() or "start_seg" in k.lower() or "end_seg" in k.lower()
        or "point_" in k.lower() or "embed_tokens" in k.lower()
    }
    state["qwen_lora_state_dict"] = lora_state

    state["coord_head_state_dict"] = model.coord_head.state_dict()
    if model.base.align_projector is not None:
        state["align_projector_state_dict"] = model.base.align_projector.state_dict()
    state["sam_prompt_encoder_state_dict"] = model.sam_prompt_encoder.state_dict()
    state["sam_mask_decoder_state_dict"] = model.sam_mask_decoder.state_dict()

    state["optimizer_state_dict"] = optimizer.state_dict()
    state["scheduler_state_dict"] = scheduler.state_dict()
    state["scaler_state_dict"] = scaler.state_dict()

    torch.save(state, path)
    print(f"  Checkpoint saved: {path}")


def load_v3_tracker_checkpoint(model, optimizer, scheduler, scaler, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    model.base.projector.load_state_dict(ckpt["projector_state_dict"])
    if "sam3_trainable_state_dict" in ckpt:
        model.base.sam3.load_state_dict(ckpt["sam3_trainable_state_dict"], strict=False)
    if "qwen_lora_state_dict" in ckpt:
        model.base.qwen.load_state_dict(ckpt["qwen_lora_state_dict"], strict=False)
    if "coord_head_state_dict" in ckpt:
        model.coord_head.load_state_dict(ckpt["coord_head_state_dict"])
    if "align_projector_state_dict" in ckpt and model.base.align_projector is not None:
        model.base.align_projector.load_state_dict(ckpt["align_projector_state_dict"])
    if "sam_prompt_encoder_state_dict" in ckpt:
        model.sam_prompt_encoder.load_state_dict(ckpt["sam_prompt_encoder_state_dict"])
    if "sam_mask_decoder_state_dict" in ckpt:
        model.sam_mask_decoder.load_state_dict(ckpt["sam_mask_decoder_state_dict"])

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
                    global_step=0, visualizer=None, val_collator=None):
    model.base.qwen.train()
    model.base.projector.train()
    model.coord_head.train()
    model.base.sam3.eval()
    model.sam_prompt_encoder.eval()
    model.sam_mask_decoder.eval()

    amp_enabled = cfg.get("amp", {}).get("enabled", True)
    amp_dtype = get_amp_dtype(cfg)
    grad_accum = cfg["training"].get("gradient_accumulation_steps", 1)
    max_grad_norm = cfg["training"].get("max_grad_norm", 1.0)
    log_every = cfg.get("logging", {}).get("log_every_n_steps", 10)
    vis_every = cfg.get("logging", {}).get("vis_every_n_steps", 200)
    loss_cfg = cfg.get("loss", {})
    seg_grad_to_lm = cfg["training"].get("seg_grad_to_lm", False)

    meters = {k: AverageMeter() for k in TRACKER_METRICS}
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

        point_pos = batch["point_positions"]
        valid = point_pos.min(dim=1).values >= 0
        if not valid.any():
            continue
        if not valid.all():
            vi = valid.nonzero(as_tuple=True)[0]
            qwen_inputs = {k: v[vi] for k, v in qwen_inputs.items()}
            point_pos = point_pos[vi]
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

        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(
                qwen_inputs, sam_images, point_pos,
                seg_grad_to_lm=seg_grad_to_lm,
            )

            total, loss_metrics = v2_tracker_total_loss(
                outputs, gt_a, gt_b,
                gt_boxes, gt_boxes, gt_boxes_xyxy, gt_boxes_xyxy,
                align_target_a=batch.get("align_target_a"),
                align_target_b=batch.get("align_target_b"),
                loss_cfg=loss_cfg,
            )
            total = total / grad_accum

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

        meters["total"].update(total.item() * grad_accum, B)
        meters["detr_loss"].update(loss_metrics.get("detr_loss", 0), B)
        meters["tracker_loss"].update(loss_metrics.get("tracker_loss", 0), B)
        meters["align_loss"].update(loss_metrics.get("align_loss", 0), B)
        meters["detr_mask_iou"].update(loss_metrics.get("detr_mask_iou", 0), B)
        meters["tracker_iou"].update(loss_metrics.get("tracker_iou", 0), B)
        if "alignment_loss" in loss_metrics:
            meters["alignment_loss"].update(loss_metrics["alignment_loss"], B)
        if "alignment_acc" in loss_metrics:
            meters["alignment_acc"].update(loss_metrics["alignment_acc"], B)
        if "lm_loss" in loss_metrics:
            meters["lm_loss"].update(loss_metrics["lm_loss"], B)

        # ---- W&B step logging ---------------------------------------- #
        global_step += 1
        is_grad_step = (step + 1) % grad_accum == 0

        if wandb_active() and is_grad_step:
            step_metrics = {k: m.val for k, m in meters.items() if m.count > 0}
            log_step(
                global_step, step_metrics, optimizer=optimizer,
                model=model if is_grad_step else None, stage="tracker",
            )
            # Advanced diagnostics every 50 grad steps
            if global_step % 50 == 0:
                log_query_diagnostics(outputs, global_step)
                log_mask_diagnostics(outputs, gt_a, gt_b, global_step)
                log_loss_ratios(step_metrics, global_step)
            # Weight norms every 200 grad steps
            if global_step % 200 == 0:
                log_weight_norms(model, global_step, stage="tracker")

        # ---- Visualization ------------------------------------------- #
        if (visualizer is not None and val_collator is not None
                and global_step % vis_every == 0):
            model.base.qwen.eval()
            model.base.projector.eval()
            model.coord_head.eval()
            visualizer.log_to_wandb(
                model, val_collator, device, epoch, global_step,
                stage="tracker", amp_dtype=amp_dtype,
            )
            model.base.qwen.train()
            model.base.projector.train()
            model.coord_head.train()

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(
                f"  [E{epoch+1}] {step+1}/{len(loader)} | "
                f"loss={meters['total'].avg:.4f} "
                f"detr={meters['detr_loss'].avg:.4f} "
                f"trk={meters['tracker_loss'].avg:.4f} "
                f"detr_iou={meters['detr_mask_iou'].avg:.4f} "
                f"trk_iou={meters['tracker_iou'].avg:.4f} "
                f"lr={get_lr(optimizer):.2e} ({elapsed:.1f}s)"
            )

    return {k: m.avg for k, m in meters.items()}, global_step


# -------------------------------------------------------------------- #
#  Validation                                                            #
# -------------------------------------------------------------------- #

@torch.no_grad()
def validate(model, loader, cfg, device):
    model.base.qwen.eval()
    model.base.projector.eval()
    model.base.sam3.eval()
    model.coord_head.eval()
    model.sam_prompt_encoder.eval()
    model.sam_mask_decoder.eval()

    amp_enabled = cfg.get("amp", {}).get("enabled", True)
    amp_dtype = get_amp_dtype(cfg)
    loss_cfg = cfg.get("loss", {})

    meters = {k: AverageMeter() for k in TRACKER_METRICS}

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

        point_pos = batch["point_positions"]
        valid = point_pos.min(dim=1).values >= 0
        if not valid.any():
            continue

        sam_images = batch["sam_images"]
        gt_a, gt_b = batch["masks_a"], batch["masks_b"]
        gt_boxes = batch["gt_boxes_cxcywh"]
        gt_boxes_xyxy = batch["gt_boxes_xyxy"]
        B = sam_images.shape[0]

        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(qwen_inputs, sam_images, point_pos)

            total, loss_metrics = v2_tracker_total_loss(
                outputs, gt_a, gt_b,
                gt_boxes, gt_boxes, gt_boxes_xyxy, gt_boxes_xyxy,
                align_target_a=batch.get("align_target_a"),
                align_target_b=batch.get("align_target_b"),
                loss_cfg=loss_cfg,
            )

        meters["total"].update(total.item(), B)
        meters["tracker_loss"].update(loss_metrics.get("tracker_loss", 0), B)
        meters["tracker_iou"].update(loss_metrics.get("tracker_iou", 0), B)
        meters["detr_mask_iou"].update(loss_metrics.get("detr_mask_iou", 0), B)
        if "lm_loss" in loss_metrics:
            meters["lm_loss"].update(loss_metrics["lm_loss"], B)

    return {k: m.avg for k, m in meters.items()}


# -------------------------------------------------------------------- #
#  Main                                                                  #
# -------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Qwen2SAM v3_tracker Training (Stage 2)")
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

    # ---- Build model --------------------------------------------------- #
    print("Building Qwen2SAM v3_tracker model...")
    model = Qwen2SAMv3Tracker(cfg, device=str(device))

    # ---- Data ---------------------------------------------------------- #
    data_root = args.data_root or cfg["data"]["data_root"]
    dataset = V3Dataset(
        data_root=data_root,
        metadata_file=cfg["data"].get("metadata_file", "metadata_phase1.json"),
        image_size=cfg["model"].get("image_size", 1008),
        augment=cfg["data"].get("augmentation", False),
    )

    # Alignment embedder
    loss_cfg = cfg.get("loss", {})
    text_embedder = None
    if loss_cfg.get("alignment_weight", 0.0) > 0:
        all_textures = []
        for entry in dataset.metadata:
            all_textures.append(entry["texture_a"])
            all_textures.append(entry["texture_b"])

        embedder_type = cfg.get("tracker", {}).get("align_embedder", "sentence")
        if embedder_type == "sentence":
            from qwen2sam.models.alignment import SentenceTextEmbedder
            model_name = cfg.get("tracker", {}).get("align_model_name", "all-mpnet-base-v2")
            text_embedder = SentenceTextEmbedder(model_name=model_name)
            text_embedder.precompute(texture_labels=all_textures)
        else:
            from qwen2sam.models.alignment import QwenTextEmbedder
            text_embedder = QwenTextEmbedder()
            text_embedder.precompute(
                texture_labels=all_textures,
                qwen_model=model.base.qwen,
                processor=model.base.processor,
                device=device,
            )

    # Train/val split
    train_n = cfg["data"]["train_size"]
    val_n = cfg["data"].get("val_size", 0)
    train_set = Subset(dataset, list(range(train_n)))
    if val_n > 0:
        val_set = Subset(dataset, list(range(train_n, train_n + val_n)))
    else:
        val_start = train_n
        val_end = min(train_n + 10, len(dataset))
        val_set = Subset(dataset, list(range(val_start, val_end)))
    print(f"Train: {len(train_set)} | Val: {len(val_set)}")

    data_cfg = cfg["data"]
    sys_prompt = data_cfg.get("system_prompt", "")
    usr_prompt = data_cfg.get("user_prompt", "")

    train_collator = V3TrackerCollator(
        processor=model.base.processor,
        system_prompt=sys_prompt, user_prompt=usr_prompt,
        token_ids=model.base.token_ids,
        point_token_ids=model.point_token_ids,
        num_points_per_texture=model.num_points,
        text_embedder=text_embedder,
        inference=False,
    )
    val_collator = V3TrackerCollator(
        processor=model.base.processor,
        system_prompt=sys_prompt, user_prompt=usr_prompt,
        token_ids=model.base.token_ids,
        point_token_ids=model.point_token_ids,
        num_points_per_texture=model.num_points,
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

    # ---- Optimizer ----------------------------------------------------- #
    train_cfg = cfg["training"]
    base_lr = train_cfg.get("learning_rate", 5e-5)
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
        warmup_epochs=train_cfg.get("warmup_epochs", 3),
        total_epochs=train_cfg.get("num_epochs", 200),
        min_lr=train_cfg.get("min_lr", 1e-6),
        steps_per_epoch=steps_per_epoch,
    )

    amp_enabled = cfg.get("amp", {}).get("enabled", True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # ---- Resume -------------------------------------------------------- #
    start_epoch = 0
    if args.resume:
        start_epoch = load_v3_tracker_checkpoint(
            model, optimizer, scheduler, scaler, args.resume
        ) + 1

    # ---- Checkpoint dir ------------------------------------------------ #
    ckpt_cfg = cfg.get("checkpoint", {})
    ckpt_dir = Path(ckpt_cfg.get("dir", "checkpoints/v3_tracker"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- W&B init --------------------------------------------------- #
    use_wandb = init_wandb(cfg, stage="v3_tracker")
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

    # ---- Training loop ------------------------------------------------- #
    num_epochs = train_cfg.get("num_epochs", 200)
    best_val_iou = 0.0
    training_history = []
    global_step = 0

    print(f"\n{'='*60}")
    print("Qwen2SAM v3_tracker — Stage 2 Training Start")
    print(f"  Architecture: Multi-Token DETR + Tracker (Qwen point coords)")
    print(f"  DETR weight: {loss_cfg.get('detr_weight', 0.0)}")
    print(f"  Tracker weight: {loss_cfg.get('tracker_weight', 1.0)}")
    print(f"  Alignment weight: {loss_cfg.get('alignment_weight', 1.0)}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, num_epochs):
        t_epoch = time.time()

        train_metrics, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, cfg, device, epoch,
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
            f"detr_iou={train_metrics['detr_mask_iou']:.4f} "
            f"trk_iou={train_metrics['tracker_iou']:.4f}"
        )
        val_str = ""
        if val_metrics:
            val_str = (
                f" | Val: loss={val_metrics['total']:.4f} "
                f"trk_iou={val_metrics['tracker_iou']:.4f}"
            )

        print(f"\nEpoch {epoch+1}/{num_epochs} ({elapsed:.1f}s) | {train_str}{val_str}")

        # W&B epoch logging
        log_epoch(epoch, train_metrics, val_metrics, best_val_iou, elapsed)
        log_learning_health(train_metrics, val_metrics, epoch)

        # Epoch-level visualization (every 10 epochs)
        vis_epoch_every = cfg.get("visualization", {}).get("every_n_epochs", 10)
        if (visualizer is not None and val_collator is not None
                and (epoch + 1) % vis_epoch_every == 0):
            model.base.qwen.eval()
            model.base.projector.eval()
            model.coord_head.eval()
            amp_dtype = get_amp_dtype(cfg)
            visualizer.log_to_wandb(
                model, val_collator, device, epoch, global_step,
                stage="tracker", amp_dtype=amp_dtype,
            )
            model.base.qwen.train()
            model.base.projector.train()
            model.coord_head.train()

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["total"],
            "train_iou": train_metrics["tracker_iou"],
        }
        if val_metrics:
            epoch_record["val_loss"] = val_metrics["total"]
            epoch_record["val_iou"] = val_metrics["tracker_iou"]
        training_history.append(epoch_record)

        from qwen2sam.training.train_phase1 import plot_training_curves
        plot_training_curves(training_history, str(ckpt_dir), phase_name="V3 Tracker")

        save_every = ckpt_cfg.get("save_every_n_epochs", 10)
        if (epoch + 1) % save_every == 0:
            save_v3_tracker_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
                str(ckpt_dir / f"epoch_{epoch+1:04d}.pt"),
            )

        val_iou = val_metrics.get("tracker_iou", 0)
        if val_metrics and val_iou > best_val_iou:
            best_val_iou = val_iou
            save_v3_tracker_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                val_metrics,
                str(ckpt_dir / "best.pt"),
            )
            print(f"  New best model (val_tracker_iou={best_val_iou:.4f})")

    print(f"\n{'='*60}")
    print(f"Training complete. Best val tracker IoU: {best_val_iou:.4f}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"{'='*60}")

    finish_wandb()


if __name__ == "__main__":
    main()
