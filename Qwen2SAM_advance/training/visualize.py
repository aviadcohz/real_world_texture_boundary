"""
Training-time visualization for W&B.

Generates grid images showing GT vs predicted boxes/masks on fixed sample images.
Logged periodically during training to monitor learning progress.

Grid layout (per sample):
  Columns: Image | Ground Truth | Predictions
  Row 1: Box overlay (colored boxes on image)
  Row 2: Mask overlay (if masks enabled)
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import math

# Colors for different instances (BGR)
INSTANCE_COLORS = [
    (0, 0, 220),     # red
    (220, 80, 0),     # blue
    (0, 200, 0),      # green
    (0, 200, 220),    # yellow
    (220, 0, 220),    # magenta
    (220, 150, 0),    # cyan
    (0, 120, 255),    # orange
    (180, 0, 120),    # purple
    (100, 200, 100),  # light green
    (100, 100, 220),  # salmon
]


def draw_boxes(image, boxes, scores=None, color_by_index=True, linewidth=2):
    """
    Draw boxes on image.

    Args:
        image: (H, W, 3) BGR uint8
        boxes: (N, 4) normalized XYXY [0,1] or CxCyWH [0,1]
        scores: (N,) confidence scores, optional
        color_by_index: use different color per box
        linewidth: box border width
    Returns:
        image with boxes drawn
    """
    vis = image.copy()
    H, W = vis.shape[:2]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)

        color = INSTANCE_COLORS[i % len(INSTANCE_COLORS)] if color_by_index else (0, 255, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, linewidth)

        if scores is not None:
            label = f"{scores[i]:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
            cv2.putText(vis, label, (x1 + 1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


def draw_masks_overlay(image, masks, alpha=0.45):
    """
    Draw instance masks as colored overlay.

    Args:
        image: (H, W, 3) BGR uint8
        masks: list of (H, W) binary masks
        alpha: blend factor
    """
    vis = image.copy()
    overlay = image.copy()
    for i, mask in enumerate(masks):
        color = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
        overlay[mask > 0.5] = color
    return cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)


def cxcywh_to_xyxy(boxes):
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2), all normalized."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def create_sample_grid(
    image_bgr,
    gt_boxes_cxcywh, gt_num_boxes,
    pred_boxes_xyxy, pred_scores,
    gt_masks=None, pred_masks=None,
    title="", cell_size=256,
):
    """
    Create a visualization grid for one sample.

    Columns: Image | Ground Truth | Predictions
    Row 1: Box overlay
    Row 2: Mask overlay (only if masks provided)

    Args:
        image_bgr: (H, W, 3) BGR uint8
        gt_boxes_cxcywh: (max_boxes, 4) normalized CxCyWH, padded
        gt_num_boxes: int, number of valid GT boxes
        pred_boxes_xyxy: (Q, 4) normalized XYXY
        pred_scores: (Q,) confidence scores
        gt_masks: optional list of (H, W) numpy masks
        pred_masks: optional list of (H, W) numpy masks
        title: text label
        cell_size: target cell size
    Returns:
        grid image as numpy array (BGR uint8)
    """
    h, w = image_bgr.shape[:2]
    scale = cell_size / max(h, w)
    ch, cw = int(h * scale), int(w * scale)

    def ri(img):
        return cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)

    def rm(mask):
        return cv2.resize(mask.astype(np.float32), (cw, ch),
                          interpolation=cv2.INTER_NEAREST)

    img = ri(image_bgr)

    # --- GT boxes ---
    gt_valid = gt_boxes_cxcywh[:gt_num_boxes]
    gt_xyxy = cxcywh_to_xyxy(gt_valid) if len(gt_valid) > 0 else np.zeros((0, 4))
    gt_box_img = draw_boxes(img, gt_xyxy, scores=None, color_by_index=True)

    # --- Pred boxes (top by confidence, threshold 0.3 or top-10) ---
    mask = pred_scores > 0.3
    if mask.sum() == 0:
        topk = min(10, len(pred_scores))
        indices = np.argsort(pred_scores)[::-1][:topk]
    else:
        indices = np.where(mask)[0]
        # Sort by score descending
        indices = indices[np.argsort(pred_scores[indices])[::-1]][:20]

    pred_sel_boxes = pred_boxes_xyxy[indices]
    pred_sel_scores = pred_scores[indices]
    pred_box_img = draw_boxes(img, pred_sel_boxes, scores=pred_sel_scores, color_by_index=True)

    # --- Build columns ---
    has_masks = gt_masks is not None and pred_masks is not None

    col_image = [img.copy()]
    col_gt = [gt_box_img]
    col_pred = [pred_box_img]

    if has_masks:
        gt_masks_resized = [rm(m) for m in gt_masks]
        pred_masks_resized = [rm(m) for m in pred_masks]
        col_image.append(img.copy())
        col_gt.append(draw_masks_overlay(img, gt_masks_resized))
        col_pred.append(draw_masks_overlay(img, pred_masks_resized))

    cols = [col_image, col_gt, col_pred]

    n_pred = len(pred_sel_scores)
    max_score = float(pred_sel_scores[0]) if n_pred > 0 else 0
    col_labels = [
        title,
        f"GT ({gt_num_boxes} obj)",
        f"Pred ({n_pred} obj, max={max_score:.2f})",
    ]
    row_labels = ["Boxes"]
    if has_masks:
        row_labels.append("Masks")

    # --- Assemble grid ---
    sep = 2
    header_h = 36
    row_label_w = 60

    bar_w = row_label_w + len(cols) * (cw + sep)
    bar = np.zeros((header_h, bar_w, 3), dtype=np.uint8) + 30
    x = row_label_w
    for lbl in col_labels:
        cv2.putText(bar, lbl, (x + 4, header_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        x += cw + sep

    def make_row(row_idx, row_label):
        rl = np.zeros((ch, row_label_w, 3), dtype=np.uint8) + 20
        cv2.putText(rl, row_label, (4, ch // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
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


# ---------------------------------------------------------------------------
# Fixed sample manager for periodic W&B logging
# ---------------------------------------------------------------------------

class FixedSampleVisualizer:
    """
    Picks a fixed set of samples at init and generates visualizations
    periodically during training. Logs grids to W&B.

    Usage:
        visualizer = FixedSampleVisualizer(dataset, collate_fn, n_train=4, n_val=4)
        # In training loop:
        if step % vis_interval == 0:
            visualizer.log_to_wandb(model, device, epoch, step)
    """

    def __init__(self, train_dataset, collate_fn, val_dataset=None,
                 n_train=4, n_val=4, cell_size=256, seed=42):
        """
        Args:
            train_dataset: training dataset
            collate_fn: SAM3 collate function
            val_dataset: optional validation dataset
            n_train: number of fixed train samples
            n_val: number of fixed val samples
            cell_size: visualization cell size
            seed: random seed for sample selection
        """
        self.collate_fn = collate_fn
        self.cell_size = cell_size

        rng = np.random.RandomState(seed)

        # Pick fixed train samples
        train_indices = rng.choice(len(train_dataset), size=min(n_train, len(train_dataset)), replace=False)
        self.train_samples = []
        for idx in train_indices:
            sample = train_dataset[idx]
            if sample is not None:
                self.train_samples.append((f"train_{idx}", sample))

        # Pick fixed val samples
        self.val_samples = []
        if val_dataset is not None:
            val_indices = rng.choice(len(val_dataset), size=min(n_val, len(val_dataset)), replace=False)
            for idx in val_indices:
                sample = val_dataset[idx]
                if sample is not None:
                    self.val_samples.append((f"val_{idx}", sample))

        print(f"FixedSampleVisualizer: {len(self.train_samples)} train, {len(self.val_samples)} val samples")

    @torch.no_grad()
    def _predict_and_visualize(self, model, sample_name, sample, device):
        """Run model on a single sample and create grid visualization."""
        from sam3.model.model_misc import SAM3Output

        # Collate single sample into batch
        batch_dict = self.collate_fn([sample])
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

        # Extract predictions from last stage, last step
        with SAM3Output.iteration_mode(
            find_stages, iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE
        ) as stages:
            if len(stages) == 0 or len(stages[0]) == 0:
                return None
            preds = stages[0][-1]

        pred_logits = preds.get("pred_logits")
        pred_boxes = preds.get("pred_boxes_xyxy")
        if pred_logits is None or pred_boxes is None:
            return None

        pred_scores = pred_logits.sigmoid().squeeze(-1)[0].cpu().numpy()  # (Q,)
        pred_boxes_np = pred_boxes[0].cpu().numpy()  # (Q, 4) XYXY normalized

        # Denormalize image for visualization
        img_tensor = batch.img_batch[0].cpu()  # (3, H, W)
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        img_np = ((img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Extract GT boxes
        gt_boxes_cxcywh = np.zeros((0, 4))
        gt_num_boxes = 0
        if len(batch.find_targets) > 0:
            gt = batch.find_targets[0]
            if gt.boxes_padded is not None:
                gt_boxes_cxcywh = gt.boxes_padded[0].cpu().numpy()  # (max_boxes, 4) CxCyWH
                gt_num_boxes = int(gt.num_boxes[0].item()) if gt.num_boxes is not None else 0

        # Extract masks if available
        gt_masks = None
        pred_masks_list = None
        pred_seg = preds.get("pred_masks")
        if pred_seg is not None and len(batch.find_targets) > 0:
            gt_target = batch.find_targets[0]
            if gt_target.segments is not None:
                # GT masks
                gt_segs = gt_target.segments[0].cpu().numpy()  # (N, H, W)
                gt_masks = [gt_segs[j] for j in range(min(gt_num_boxes, len(gt_segs)))]

                # Pred masks — select top predictions
                mask = pred_scores > 0.3
                if mask.sum() == 0:
                    indices = np.argsort(pred_scores)[::-1][:5]
                else:
                    indices = np.where(mask)[0]
                    indices = indices[np.argsort(pred_scores[indices])[::-1]][:10]

                pred_masks_np = pred_seg[0].sigmoid().cpu().numpy()  # (Q, H, W)
                H, W = img_bgr.shape[:2]
                pred_masks_list = []
                for j in indices:
                    m = pred_masks_np[j]
                    if m.shape != (H, W):
                        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
                    pred_masks_list.append((m > 0.5).astype(np.float32))

        grid = create_sample_grid(
            image_bgr=img_bgr,
            gt_boxes_cxcywh=gt_boxes_cxcywh,
            gt_num_boxes=gt_num_boxes,
            pred_boxes_xyxy=pred_boxes_np,
            pred_scores=pred_scores,
            gt_masks=gt_masks,
            pred_masks=pred_masks_list,
            title=sample_name,
            cell_size=self.cell_size,
        )
        return grid

    def log_to_wandb(self, model, device, epoch, step):
        """Generate visualizations for all fixed samples and log to W&B."""
        import wandb

        model.eval()
        images = []

        for name, sample in self.train_samples + self.val_samples:
            try:
                grid = self._predict_and_visualize(model, name, sample, device)
                if grid is not None:
                    # Convert BGR to RGB for W&B
                    grid_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
                    images.append(wandb.Image(grid_rgb, caption=f"{name} (ep{epoch+1} step{step})"))
            except Exception as e:
                print(f"Warning: visualization failed for {name}: {e}")
                continue

        if images:
            wandb.log({
                "visualizations": images,
                "train/step": step,
            })

        model.train()
