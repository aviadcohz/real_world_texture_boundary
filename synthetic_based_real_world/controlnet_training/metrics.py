"""
Edge detection metrics: ODS, OIS, AP with distance-tolerant matching.

Implements the BSDS-style evaluation protocol used in the ControlNet paper
(Section 4.3, "Condition Reconstruction").

Matching rule:
    A predicted edge pixel is a "true positive" if there exists a GT edge
    pixel within `max_dist` pixels (Euclidean). This uses the distance
    transform for efficient O(H×W) computation instead of brute-force
    pairwise matching.

Metrics:
    ODS — One threshold for the entire dataset, maximizing F-score
    OIS — Best threshold per image, averaged F-score
    AP  — Area under the dataset-wide Precision-Recall curve
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


# ── Distance-tolerant matching ───────────────────────────────────────────────

def match_edges(pred_binary: np.ndarray, gt_binary: np.ndarray, max_dist: float):
    """
    Distance-tolerant edge matching.

    Args:
        pred_binary: (H, W) bool — predicted edge map (thresholded)
        gt_binary:   (H, W) bool — ground truth edge map
        max_dist:    Maximum matching distance in pixels

    Returns:
        tp:  true positives (matched predicted pixels)
        fp:  false positives (unmatched predicted pixels)
        fn:  false negatives (unmatched GT pixels)
    """
    # Distance transform of the COMPLEMENT gives distance to nearest edge
    # For each predicted pixel, check if nearest GT pixel is within max_dist
    if gt_binary.sum() == 0:
        return 0, int(pred_binary.sum()), 0
    if pred_binary.sum() == 0:
        return 0, 0, int(gt_binary.sum())

    # Distance from each pixel to nearest GT edge
    gt_dist = distance_transform_edt(~gt_binary)
    # Distance from each pixel to nearest predicted edge
    pred_dist = distance_transform_edt(~pred_binary)

    # True positives: predicted pixels close to GT
    tp = int((gt_dist[pred_binary] <= max_dist).sum())
    fp = int(pred_binary.sum()) - tp
    # False negatives: GT pixels not close to any prediction
    fn = int((pred_dist[gt_binary] > max_dist).sum())

    return tp, fp, fn


# ── Per-image precision/recall at multiple thresholds ────────────────────────

def compute_pr_per_image(
    soft_pred: np.ndarray,
    gt_binary: np.ndarray,
    max_dist: float,
    thresholds: np.ndarray,
):
    """
    Compute precision, recall, F-score at each threshold for one image.

    Args:
        soft_pred:  (H, W) float in [0, 1] — soft edge probability map
        gt_binary:  (H, W) bool — ground truth edges
        max_dist:   Matching distance tolerance
        thresholds: Array of thresholds to evaluate

    Returns:
        precisions: (N,) array
        recalls:    (N,) array
        fscores:    (N,) array
        counts:     (N, 3) array of [tp, fp, fn] per threshold
    """
    n = len(thresholds)
    counts = np.zeros((n, 3), dtype=np.int64)  # tp, fp, fn

    for i, t in enumerate(thresholds):
        pred_bin = soft_pred >= t
        tp, fp, fn = match_edges(pred_bin, gt_binary, max_dist)
        counts[i] = [tp, fp, fn]

    tp = counts[:, 0].astype(float)
    fp = counts[:, 1].astype(float)
    fn = counts[:, 2].astype(float)

    precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    fscore = np.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )

    return precision, recall, fscore, counts


# ── Dataset-level metrics ────────────────────────────────────────────────────

def compute_ods_ois_ap(
    soft_preds: list,
    gt_masks: list,
    max_dist: float = None,
    n_thresholds: int = 99,
):
    """
    Compute ODS, OIS, and AP over a dataset.

    Args:
        soft_preds: List of (H, W) float arrays in [0, 1] — soft edge maps
        gt_masks:   List of (H, W) bool arrays — ground truth edges
        max_dist:   Matching distance (default: BSDS standard 0.0075 × diagonal)
        n_thresholds: Number of thresholds to sweep (default: 99)

    Returns:
        dict with keys: ods_f, ods_p, ods_r, ods_threshold,
                        ois_f, ois_p, ois_r,
                        ap,
                        per_threshold (detailed P/R/F at each threshold)
    """
    thresholds = np.linspace(1.0 / (n_thresholds + 1), 1.0 - 1.0 / (n_thresholds + 1), n_thresholds)
    n_images = len(soft_preds)

    # Collect per-image results
    all_counts = np.zeros((n_images, n_thresholds, 3), dtype=np.int64)
    all_fscores = np.zeros((n_images, n_thresholds))

    for img_idx in range(n_images):
        soft = soft_preds[img_idx]
        gt = gt_masks[img_idx]

        # Default max_dist: BSDS standard
        if max_dist is None:
            h, w = gt.shape
            diag = np.sqrt(h**2 + w**2)
            md = 0.0075 * diag
        else:
            md = max_dist

        _, _, fscores, counts = compute_pr_per_image(soft, gt, md, thresholds)
        all_counts[img_idx] = counts
        all_fscores[img_idx] = fscores

    # ── ODS: one threshold for all images ────────────────────────────
    # Sum tp/fp/fn across all images at each threshold
    dataset_counts = all_counts.sum(axis=0)  # (n_thresholds, 3)
    tp_d = dataset_counts[:, 0].astype(float)
    fp_d = dataset_counts[:, 1].astype(float)
    fn_d = dataset_counts[:, 2].astype(float)

    precision_d = np.where(tp_d + fp_d > 0, tp_d / (tp_d + fp_d), 0.0)
    recall_d = np.where(tp_d + fn_d > 0, tp_d / (tp_d + fn_d), 0.0)
    fscore_d = np.where(
        precision_d + recall_d > 0,
        2 * precision_d * recall_d / (precision_d + recall_d),
        0.0,
    )

    ods_idx = np.argmax(fscore_d)
    ods_f = fscore_d[ods_idx]
    ods_p = precision_d[ods_idx]
    ods_r = recall_d[ods_idx]
    ods_threshold = thresholds[ods_idx]

    # ── OIS: best threshold per image, average ───────────────────────
    best_per_image = all_fscores.max(axis=1)  # (n_images,)
    ois_f = best_per_image.mean()

    # Also get per-image best P and R
    best_idx_per_image = all_fscores.argmax(axis=1)
    ois_p_list = []
    ois_r_list = []
    for img_idx in range(n_images):
        bi = best_idx_per_image[img_idx]
        tp_i = float(all_counts[img_idx, bi, 0])
        fp_i = float(all_counts[img_idx, bi, 1])
        fn_i = float(all_counts[img_idx, bi, 2])
        p = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0.0
        r = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0.0
        ois_p_list.append(p)
        ois_r_list.append(r)
    ois_p = np.mean(ois_p_list)
    ois_r = np.mean(ois_r_list)

    # ── AP: area under the dataset-wide P-R curve ────────────────────
    # Sort by recall (ascending) for proper AUC
    sorted_idx = np.argsort(recall_d)
    recall_sorted = recall_d[sorted_idx]
    precision_sorted = precision_d[sorted_idx]

    # Add boundary point at recall=0 for proper integration
    if len(recall_sorted) > 0 and recall_sorted[0] > 0:
        recall_sorted = np.concatenate([[0.0], recall_sorted])
        precision_sorted = np.concatenate([[precision_sorted[0]], precision_sorted])

    # Trapezoidal integration
    ap = np.trapz(precision_sorted, recall_sorted)
    # Clip to [0, 1]
    ap = float(np.clip(ap, 0.0, 1.0))

    return {
        "ods_f": float(ods_f),
        "ods_p": float(ods_p),
        "ods_r": float(ods_r),
        "ods_threshold": float(ods_threshold),
        "ois_f": float(ois_f),
        "ois_p": float(ois_p),
        "ois_r": float(ois_r),
        "ap": float(ap),
        "per_threshold": {
            "thresholds": thresholds.tolist(),
            "precision": precision_d.tolist(),
            "recall": recall_d.tolist(),
            "fscore": fscore_d.tolist(),
        },
    }


# ── Convenience ──────────────────────────────────────────────────────────────

def format_results(results: dict) -> str:
    """Format results as a readable string."""
    return (
        f"ODS: F={results['ods_f']:.4f}  P={results['ods_p']:.4f}  "
        f"R={results['ods_r']:.4f}  (t={results['ods_threshold']:.3f})\n"
        f"OIS: F={results['ois_f']:.4f}  P={results['ois_p']:.4f}  "
        f"R={results['ois_r']:.4f}\n"
        f"AP:  {results['ap']:.4f}"
    )
