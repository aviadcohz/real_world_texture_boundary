"""
Weights & Biases integration for Qwen2SAM v3 training.

Provides:
  - W&B initialization with config logging
  - Per-step and per-epoch metric logging
  - Gradient norm tracking per component
  - GPU memory monitoring
  - Fixed-sample mask visualizations (GT vs predicted overlays)
  - Learning rate tracking per param group
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# -------------------------------------------------------------------- #
#  Initialization                                                        #
# -------------------------------------------------------------------- #

def init_wandb(cfg, stage="v3"):
    """Initialize W&B run from config. Returns True if wandb is active."""
    logging_cfg = cfg.get("logging", {})
    if not logging_cfg.get("use_wandb", False):
        return False

    import wandb

    project = logging_cfg.get("wandb_project", "qwen2sam-v3")
    run_name = logging_cfg.get("wandb_run_name", None)
    group = logging_cfg.get("wandb_group", stage)
    tags = logging_cfg.get("wandb_tags", [stage])

    wandb.init(
        project=project,
        name=run_name,
        group=group,
        tags=tags,
        config=cfg,
        resume="allow",
    )

    # Define metric axes
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("lr/*", step_metric="train/step")
    wandb.define_metric("gradients/*", step_metric="train/step")
    wandb.define_metric("system/*", step_metric="train/step")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("epoch")

    return True


def wandb_active():
    """Check if wandb is currently active."""
    try:
        import wandb
        return wandb.run is not None
    except ImportError:
        return False


# -------------------------------------------------------------------- #
#  Gradient statistics                                                    #
# -------------------------------------------------------------------- #

def compute_gradient_norms(model, stage="v3"):
    """
    Compute per-component gradient norms.

    Returns dict like:
        {"gradients/lora_norm": 0.12, "gradients/projector_norm": 0.45, ...}
    """
    component_grads = {}

    if stage == "v3":
        components = {
            "lora": lambda k: "lora" in k.lower(),
            "projector": lambda k: k.startswith("projector."),
            "encoder": lambda k: "transformer.encoder" in k,
            "queries": lambda k: "query_embed" in k,
            "seg_head": lambda k: "segmentation_head" in k,
            "scoring": lambda k: "dot_prod_scoring" in k or "class_embed" in k,
            "align_proj": lambda k: k.startswith("align_projector"),
        }
        named_params = list(model.named_parameters())
    else:  # tracker
        components = {
            "lora": lambda k: "lora" in k.lower(),
            "coord_head": lambda k: k.startswith("coord_head."),
            "prompt_enc": lambda k: "sam_prompt_encoder" in k,
            "mask_dec": lambda k: "sam_mask_decoder" in k,
            "projector": lambda k: "projector" in k and "align" not in k,
            "align_proj": lambda k: "align_projector" in k,
        }
        named_params = list(model.named_parameters())

    total_norm_sq = 0.0
    for comp_name, match_fn in components.items():
        norm_sq = 0.0
        for name, param in named_params:
            if param.grad is not None and match_fn(name):
                norm_sq += param.grad.data.float().norm().item() ** 2
        component_grads[f"gradients/{comp_name}_norm"] = norm_sq ** 0.5
        total_norm_sq += norm_sq

    component_grads["gradients/total_norm"] = total_norm_sq ** 0.5
    return component_grads


# -------------------------------------------------------------------- #
#  Learning rate tracking                                                 #
# -------------------------------------------------------------------- #

def get_lr_per_group(optimizer):
    """Extract learning rate for each named param group."""
    lr_dict = {}
    for group in optimizer.param_groups:
        name = group.get("name", f"group_{id(group)}")
        lr_dict[f"lr/{name}"] = group["lr"]
    return lr_dict


# -------------------------------------------------------------------- #
#  System metrics                                                         #
# -------------------------------------------------------------------- #

def get_system_metrics():
    """Get GPU memory usage."""
    if not torch.cuda.is_available():
        return {}
    return {
        "system/gpu_memory_GB": torch.cuda.max_memory_allocated() / 1e9,
        "system/gpu_memory_reserved_GB": torch.cuda.max_memory_reserved() / 1e9,
    }


# -------------------------------------------------------------------- #
#  Step & epoch logging                                                   #
# -------------------------------------------------------------------- #

def log_step(global_step, metrics_dict, optimizer=None, model=None, stage="v3"):
    """Log per-step metrics to wandb."""
    if not wandb_active():
        return

    import wandb

    log_dict = {"train/step": global_step}

    # Training metrics
    for k, v in metrics_dict.items():
        log_dict[f"train/{k}"] = v

    # Learning rates
    if optimizer is not None:
        log_dict.update(get_lr_per_group(optimizer))

    # Gradient norms (only on gradient update steps)
    if model is not None:
        log_dict.update(compute_gradient_norms(model, stage=stage))

    # System metrics
    log_dict.update(get_system_metrics())

    wandb.log(log_dict)


def log_epoch(epoch, train_metrics, val_metrics=None, best_val_iou=None,
              epoch_time=None):
    """Log per-epoch summary metrics to wandb."""
    if not wandb_active():
        return

    import wandb

    log_dict = {"epoch": epoch + 1}

    # Train epoch averages
    for k, v in train_metrics.items():
        log_dict[f"train/epoch_{k}"] = v

    # Validation metrics
    if val_metrics:
        for k, v in val_metrics.items():
            log_dict[f"val/{k}"] = v

    if epoch_time is not None:
        log_dict["train/epoch_time_s"] = epoch_time

    if best_val_iou is not None:
        log_dict["val/best_iou"] = best_val_iou

    wandb.log(log_dict)

    # Update summary
    if best_val_iou is not None:
        wandb.run.summary["best_val_iou"] = best_val_iou
        wandb.run.summary["best_epoch"] = epoch + 1


# -------------------------------------------------------------------- #
#  Mask visualization helpers                                             #
# -------------------------------------------------------------------- #

# Fixed colors for A/B textures
COLOR_A = (220, 60, 60)    # Red (RGB)
COLOR_B = (60, 60, 220)    # Blue (RGB)
COLOR_A_GT = (180, 40, 40)
COLOR_B_GT = (40, 40, 180)


def _denorm_image(sam_image_tensor):
    """Denormalize SAM image tensor (mean/std=0.5) to uint8 RGB numpy."""
    img = sam_image_tensor.cpu().float()
    img = img * 0.5 + 0.5  # denorm
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def _mask_overlay(image_rgb, mask, color, alpha=0.45):
    """Overlay a binary mask on an RGB image."""
    vis = image_rgb.copy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().float().numpy()
    if mask.shape != image_rgb.shape[:2]:
        mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    binary = mask > 0.5
    overlay = vis.copy()
    overlay[binary] = color
    return cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)


def _compute_iou(pred, gt):
    """Compute IoU between two binary masks (numpy arrays)."""
    p = (pred > 0.5).astype(np.float32)
    g = (gt > 0.5).astype(np.float32)
    inter = (p * g).sum()
    union = p.sum() + g.sum() - inter
    return inter / max(union, 1)


def _dual_mask_overlay(image, mask_a, mask_b, alpha=0.45):
    """Overlay both A (red) and B (blue) masks on image."""
    vis = image.copy()
    overlay = image.copy()
    overlay[mask_a > 0.5] = COLOR_A
    overlay[mask_b > 0.5] = COLOR_B
    return cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)


def _boundary_image(mask_a, mask_b, h, w):
    """Visualize texture boundary between A and B masks."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ma = (mask_a > 0.5).astype(np.uint8) * 255
    mb = (mask_b > 0.5).astype(np.uint8) * 255
    bd_a = ma - cv2.erode(ma, kernel, iterations=1)
    bd_b = mb - cv2.erode(mb, kernel, iterations=1)
    da = cv2.dilate(bd_a, kernel, iterations=2)
    db = cv2.dilate(bd_b, kernel, iterations=2)
    interface = ((da > 0) & (db > 0)).astype(np.uint8) * 255
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[bd_a > 0] = (150, 40, 40)
    canvas[bd_b > 0] = (40, 40, 150)
    canvas[interface > 0] = (0, 220, 220)  # yellow boundary
    return canvas


def _create_4col_grid(image_rgb, gt_a, gt_b, zs_a, zs_b, pred_a, pred_b,
                      title="", cell_size=280):
    """
    Create a 3-row x 4-col comparison grid (like evaluate_v3).

    Columns: Image | Ground Truth | Zero-shot | Trained
    Row 1: Mask overlay on image (A=red, B=blue)
    Row 2: Binary mask panels
    Row 3: Boundary visualization

    Returns RGB numpy array.
    """
    h, w = image_rgb.shape[:2]
    scale = cell_size / max(h, w)
    ch, cw = int(h * scale), int(w * scale)

    def ri(img):
        return cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)

    def rm(m):
        if isinstance(m, torch.Tensor):
            m = m.cpu().float().numpy()
        return cv2.resize(m, (cw, ch), interpolation=cv2.INTER_NEAREST)

    def binary_panel(ma, mb):
        canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
        canvas[ma > 0.5] = COLOR_A
        canvas[mb > 0.5] = COLOR_B
        return canvas

    img = ri(image_rgb)
    ga, gb = rm(gt_a), rm(gt_b)
    za, zb = rm(zs_a), rm(zs_b)
    pa, pb = rm(pred_a), rm(pred_b)

    # Compute IoUs
    zs_iou_a, zs_iou_b = _compute_iou(za, ga), _compute_iou(zb, gb)
    tr_iou_a, tr_iou_b = _compute_iou(pa, ga), _compute_iou(pb, gb)
    zs_miou = (zs_iou_a + zs_iou_b) / 2
    tr_miou = (tr_iou_a + tr_iou_b) / 2

    # Build columns: (overlay, binary, boundary)
    def make_col(ma, mb):
        return (
            _dual_mask_overlay(img, ma, mb),
            binary_panel(ma, mb),
            _boundary_image(ma, mb, ch, cw),
        )

    col_image = (img.copy(), np.zeros((ch, cw, 3), dtype=np.uint8) + 20,
                 np.zeros((ch, cw, 3), dtype=np.uint8) + 20)
    col_gt = make_col(ga, gb)
    col_zs = make_col(za, zb)
    col_pred = make_col(pa, pb)
    cols = [col_image, col_gt, col_zs, col_pred]

    col_labels = [
        title[:20] if len(title) > 20 else title,
        "Ground Truth",
        f"Zero-shot mIoU={zs_miou:.3f}",
        f"Trained mIoU={tr_miou:.3f}",
    ]
    row_labels = ["Overlay", "Masks", "Boundary"]

    # Assemble grid
    sep = 2
    header_h = 32
    row_label_w = 65
    n_cols = len(cols)

    bar_w = row_label_w + n_cols * (cw + sep)
    bar = np.zeros((header_h, bar_w, 3), dtype=np.uint8) + 25
    x = row_label_w
    for lbl in col_labels:
        cv2.putText(bar, lbl, (x + 4, header_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        x += cw + sep

    def make_row(row_idx, row_label):
        rl = np.zeros((ch, row_label_w, 3), dtype=np.uint8) + 15
        cv2.putText(rl, row_label, (4, ch // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
        sep_col = np.ones((ch, sep, 3), dtype=np.uint8) * 80
        row = rl
        for col in cols:
            row = np.concatenate([row, sep_col, col[row_idx]], axis=1)
        return row

    sep_row = np.ones((sep, bar_w, 3), dtype=np.uint8) * 80
    rows = [bar]
    for ri_idx, rl in enumerate(row_labels):
        rows.append(sep_row)
        rows.append(make_row(ri_idx, rl))

    # Score bar at bottom
    score_h = 24
    score_bar = np.zeros((score_h, bar_w, 3), dtype=np.uint8) + 12
    delta = tr_miou - zs_miou
    delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
    color = (100, 255, 100) if delta >= 0 else (100, 100, 255)
    score_text = (f"Zero-shot: A={zs_iou_a:.3f} B={zs_iou_b:.3f} mIoU={zs_miou:.3f}  |  "
                  f"Trained: A={tr_iou_a:.3f} B={tr_iou_b:.3f} mIoU={tr_miou:.3f}  ({delta_str})")
    cv2.putText(score_bar, score_text, (5, score_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    rows.append(score_bar)
    return np.concatenate(rows, axis=0)


# -------------------------------------------------------------------- #
#  Fixed Sample Visualizer for texture boundary                          #
# -------------------------------------------------------------------- #

class TextureBoundaryVisualizer:
    """
    Picks fixed train+val samples and periodically logs 4-column
    comparison grids (Image | GT | Zero-shot | Trained) to W&B.

    Zero-shot predictions are cached once at initialization.
    """

    def __init__(self, train_dataset, val_dataset=None,
                 n_train=4, n_val=4, cell_size=280, seed=42):
        rng = np.random.RandomState(seed)

        self.cell_size = cell_size
        self.train_samples = []
        self.val_samples = []

        # Zero-shot cache: {sample_name: (zs_a, zs_b)} — filled on first call
        self._zs_cache = {}
        self._zs_initialized = False

        train_indices = rng.choice(
            len(train_dataset),
            size=min(n_train, len(train_dataset)),
            replace=False,
        )
        for idx in train_indices:
            sample = train_dataset[int(idx)]
            if sample is not None:
                self.train_samples.append((f"train_{idx}", sample))

        if val_dataset is not None:
            val_indices = rng.choice(
                len(val_dataset),
                size=min(n_val, len(val_dataset)),
                replace=False,
            )
            for idx in val_indices:
                sample = val_dataset[int(idx)]
                if sample is not None:
                    self.val_samples.append((f"val_{idx}", sample))

        print(f"TextureBoundaryVisualizer: {len(self.train_samples)} train, "
              f"{len(self.val_samples)} val fixed samples")

    @torch.no_grad()
    def _extract_masks(self, model, collator, sample, device,
                       stage="v3", amp_dtype=torch.bfloat16):
        """Run inference on a single sample, return (pred_a, pred_b) as numpy."""
        batch = collator([sample])
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

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            if stage == "v3":
                outputs = model(qwen_inputs, sam_images)
            else:
                point_pos = batch.get("point_positions")
                if point_pos is not None and point_pos.min() < 0:
                    return None, None
                outputs = model(qwen_inputs, sam_images, point_pos)

        # Extract masks
        if stage != "v3" and "refined_masks_a" in outputs:
            pred_a = outputs["refined_masks_a"][0, 0].sigmoid().cpu().numpy()
            pred_b = outputs["refined_masks_b"][0, 0].sigmoid().cpu().numpy()
        else:
            scores_a = outputs["pred_logits_a"].sigmoid().squeeze(-1)[0]
            scores_b = outputs["pred_logits_b"].sigmoid().squeeze(-1)[0]
            best_a = scores_a.argmax().item()
            best_b = scores_b.argmax().item()
            pred_a = outputs["pred_masks_a"][0, best_a].sigmoid().cpu().numpy()
            pred_b = outputs["pred_masks_b"][0, best_b].sigmoid().cpu().numpy()

        return pred_a, pred_b

    @torch.no_grad()
    def _cache_zero_shot(self, model, collator, device, stage, amp_dtype):
        """Run the model at its current state to get zero-shot predictions.
        Called once before any training updates the model."""
        if self._zs_initialized:
            return

        print("  Caching zero-shot predictions for visualizer...")
        all_samples = self.train_samples + self.val_samples

        for name, sample in all_samples:
            try:
                zs_a, zs_b = self._extract_masks(
                    model, collator, sample, device,
                    stage=stage, amp_dtype=amp_dtype,
                )
                if zs_a is not None:
                    self._zs_cache[name] = (zs_a, zs_b)
            except Exception as e:
                print(f"    Warning: zero-shot failed for {name}: {e}")

        self._zs_initialized = True
        print(f"  Cached {len(self._zs_cache)} zero-shot predictions")

    def log_to_wandb(self, model, collator, device, epoch, global_step,
                     stage="v3", amp_dtype=torch.bfloat16):
        """Generate 4-column comparison grids and log to W&B."""
        if not wandb_active():
            return

        import wandb

        # Cache zero-shot on first call (before model has trained much)
        if not self._zs_initialized:
            self._cache_zero_shot(model, collator, device, stage, amp_dtype)

        images = []
        all_samples = (
            [(n, s, "train") for n, s in self.train_samples] +
            [(n, s, "val") for n, s in self.val_samples]
        )

        for name, sample, split in all_samples:
            try:
                # Get current predictions
                pred_a, pred_b = self._extract_masks(
                    model, collator, sample, device,
                    stage=stage, amp_dtype=amp_dtype,
                )
                if pred_a is None:
                    continue

                # Get zero-shot (cached)
                zs_a, zs_b = self._zs_cache.get(name, (
                    np.zeros_like(pred_a), np.zeros_like(pred_b),
                ))

                # GT masks
                gt_a = sample["mask_a"].cpu().float().numpy() if isinstance(
                    sample["mask_a"], torch.Tensor) else sample["mask_a"]
                gt_b = sample["mask_b"].cpu().float().numpy() if isinstance(
                    sample["mask_b"], torch.Tensor) else sample["mask_b"]

                # Image
                image_rgb = _denorm_image(sample["sam_image"])

                grid = _create_4col_grid(
                    image_rgb, gt_a, gt_b,
                    zs_a, zs_b, pred_a, pred_b,
                    title=name, cell_size=self.cell_size,
                )
                caption = f"{name} (ep{epoch+1})"
                images.append(wandb.Image(grid, caption=caption))

            except Exception as e:
                print(f"  Warning: viz failed for {name}: {e}")
                continue

        if images:
            wandb.log({
                "visualizations/predictions": images,
                "train/step": global_step,
            })


# -------------------------------------------------------------------- #
#  IoU distribution histogram                                            #
# -------------------------------------------------------------------- #

def log_iou_histogram(ious_a, ious_b, epoch):
    """Log IoU distribution as a histogram to wandb."""
    if not wandb_active():
        return

    import wandb

    all_ious = ious_a + ious_b
    mean_ious = [(a + b) / 2 for a, b in zip(ious_a, ious_b)]

    wandb.log({
        "val/iou_hist_a": wandb.Histogram(ious_a),
        "val/iou_hist_b": wandb.Histogram(ious_b),
        "val/iou_hist_mean": wandb.Histogram(mean_ious),
        "val/pct_good_iou": sum(1 for x in mean_ious if x >= 0.7) / max(len(mean_ious), 1) * 100,
        "val/pct_bad_iou": sum(1 for x in mean_ious if x < 0.4) / max(len(mean_ious), 1) * 100,
        "epoch": epoch + 1,
    })


# -------------------------------------------------------------------- #
#  Finish                                                                #
# -------------------------------------------------------------------- #

def finish_wandb():
    """Finish wandb run if active."""
    if wandb_active():
        import wandb
        wandb.finish()


# -------------------------------------------------------------------- #
#  Advanced diagnostics for convergence debugging                        #
# -------------------------------------------------------------------- #

def log_query_diagnostics(outputs, global_step):
    """
    Log diagnostics about DETR query utilization.
    Helps detect dead queries and score distribution issues.
    """
    if not wandb_active():
        return

    import wandb

    diag = {}

    for side in ["a", "b"]:
        logits = outputs.get(f"pred_logits_{side}")
        if logits is None:
            continue

        scores = logits.sigmoid().squeeze(-1)  # (B, Q)
        B, Q = scores.shape

        # Per-batch stats
        max_scores = scores.max(dim=1).values  # (B,)
        mean_scores = scores.mean(dim=1)  # (B,)

        diag[f"queries/{side}_max_score"] = max_scores.mean().item()
        diag[f"queries/{side}_mean_score"] = mean_scores.mean().item()
        diag[f"queries/{side}_std_score"] = scores.std().item()

        # Dead query ratio: queries that never score above 0.1 across batch
        alive = (scores > 0.1).any(dim=0).float()  # (Q,)
        diag[f"queries/{side}_alive_pct"] = alive.mean().item() * 100

        # Score entropy (higher = more uniform = less confident)
        probs = scores / scores.sum(dim=1, keepdim=True).clamp(min=1e-8)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=1).mean()
        diag[f"queries/{side}_entropy"] = entropy.item()

        # Confidence percentiles
        flat = scores.flatten()
        diag[f"queries/{side}_p50"] = flat.median().item()
        diag[f"queries/{side}_p90"] = flat.quantile(0.9).item()
        diag[f"queries/{side}_p99"] = flat.quantile(0.99).item()

    diag["train/step"] = global_step
    wandb.log(diag)


def log_mask_diagnostics(outputs, gt_a, gt_b, global_step):
    """
    Log mask quality diagnostics per step.
    Tracks mask coverage, overlap, and prediction sharpness.
    """
    if not wandb_active():
        return

    import wandb

    diag = {}

    for side, gt in [("a", gt_a), ("b", gt_b)]:
        logits = outputs.get(f"pred_logits_{side}")
        masks = outputs.get(f"pred_masks_{side}")
        if logits is None or masks is None:
            continue

        scores = logits.sigmoid().squeeze(-1)  # (B, Q)
        best_idx = scores.argmax(dim=1)  # (B,)
        B = masks.shape[0]

        # Get best predicted mask per sample
        best_masks = []
        for i in range(B):
            best_masks.append(masks[i, best_idx[i]].sigmoid())
        best_masks = torch.stack(best_masks)  # (B, H, W)

        # Mask sharpness: how binary are predictions? (closer to 0/1 = sharper)
        sharpness = (best_masks - 0.5).abs().mean()
        diag[f"masks/{side}_sharpness"] = sharpness.item()

        # Mask coverage: fraction of image covered
        binary = (best_masks > 0.5).float()
        coverage = binary.mean()
        diag[f"masks/{side}_coverage"] = coverage.item()

        # GT coverage for comparison
        if gt is not None:
            gt_coverage = (gt > 0.5).float().mean()
            diag[f"masks/{side}_gt_coverage"] = gt_coverage.item()

    # A/B overlap (should be low)
    masks_a = outputs.get("pred_masks_a")
    masks_b = outputs.get("pred_masks_b")
    if masks_a is not None and masks_b is not None:
        logits_a = outputs["pred_logits_a"].sigmoid().squeeze(-1)
        logits_b = outputs["pred_logits_b"].sigmoid().squeeze(-1)
        B = masks_a.shape[0]
        overlap = 0.0
        for i in range(B):
            ma = masks_a[i, logits_a[i].argmax()].sigmoid()
            mb = masks_b[i, logits_b[i].argmax()].sigmoid()
            binary_a = (ma > 0.5).float()
            binary_b = (mb > 0.5).float()
            overlap += (binary_a * binary_b).sum() / max((binary_a + binary_b).clamp(max=1).sum(), 1)
        diag["masks/ab_overlap"] = (overlap / B).item()

    diag["train/step"] = global_step
    wandb.log(diag)


def log_weight_norms(model, global_step, stage="v3"):
    """
    Log L2 norms of key parameter groups.
    Sudden changes indicate instability; flat lines indicate frozen/dead params.
    """
    if not wandb_active():
        return

    import wandb

    norms = {}

    if stage == "v3":
        groups = {
            "lora": lambda k: "lora" in k.lower(),
            "projector": lambda k: k.startswith("projector."),
            "encoder": lambda k: "transformer.encoder" in k,
            "seg_head": lambda k: "segmentation_head" in k,
        }
        named_params = list(model.named_parameters())
    else:
        groups = {
            "lora": lambda k: "lora" in k.lower(),
            "coord_head": lambda k: k.startswith("coord_head."),
            "prompt_enc": lambda k: "sam_prompt_encoder" in k,
            "mask_dec": lambda k: "sam_mask_decoder" in k,
        }
        named_params = list(model.named_parameters())

    for group_name, match_fn in groups.items():
        norm_sq = 0.0
        count = 0
        for name, param in named_params:
            if match_fn(name):
                norm_sq += param.data.float().norm().item() ** 2
                count += 1
        norms[f"weights/{group_name}_norm"] = norm_sq ** 0.5
        norms[f"weights/{group_name}_n_params"] = count

    norms["train/step"] = global_step
    wandb.log(norms)


def log_loss_ratios(metrics_dict, global_step):
    """
    Log ratios between loss components.
    Helps detect when one loss dominates and others can't learn.
    """
    if not wandb_active():
        return

    import wandb

    total = metrics_dict.get("total", 1.0)
    if total <= 0:
        return

    ratios = {"train/step": global_step}

    for key in ["seg_loss", "lm_loss", "alignment_loss", "focal", "dice",
                "tracker_loss", "detr_loss"]:
        if key in metrics_dict and metrics_dict[key] != 0:
            ratios[f"loss_ratio/{key}"] = metrics_dict[key] / total

    wandb.log(ratios)


def log_learning_health(train_metrics, val_metrics, epoch):
    """
    Log high-level training health indicators per epoch.
    Detects overfitting, underfitting, and stagnation.
    """
    if not wandb_active():
        return

    import wandb

    health = {"epoch": epoch + 1}

    # Train/val gap (overfitting indicator)
    if val_metrics:
        train_loss = train_metrics.get("total", 0)
        val_loss = val_metrics.get("total", 0)
        if train_loss > 0:
            health["health/overfit_ratio"] = val_loss / train_loss

        # IoU gap
        train_iou_key = "mask_iou" if "mask_iou" in train_metrics else "tracker_iou"
        val_iou_key = "mask_iou" if "mask_iou" in val_metrics else "tracker_iou"
        train_iou = train_metrics.get(train_iou_key, 0)
        val_iou = val_metrics.get(val_iou_key, 0)
        if train_iou > 0:
            health["health/iou_gap"] = train_iou - val_iou
            health["health/generalization"] = val_iou / max(train_iou, 1e-6)

    wandb.log(health)
