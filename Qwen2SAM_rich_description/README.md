# Qwen2SAM v3 — Rich Description (Multi-Token) Architecture

## Results (227 test samples, 26 train)

| Stage | mIoU | Dice | ARI | Good (>=0.7) |
|-------|------|------|-----|--------------|
| Zero-shot SAM3 | 0.448 | 0.483 | 0.734 | — |
| Stage 1: v3 DETR (100ep) | 0.827 | 0.889 | 0.694 | 80.6% |
| **Stage 2: v3_tracker (200ep)** | **0.865** | **0.909** | **0.763** | **81.5%** |

Tracker adds +4.5% mIoU and +10.1% ARI over DETR alone.

## Architecture

- **v3 (Stage 1)**: Qwen generates variable-length description tokens between `<START_SEG_A>...<END_SEG_A>` brackets (up to 16 tokens). These are projected per-token via DescriptionProjector (2048→1024→256) and fed as multi-token prompts to SAM3's fusion encoder.
- **v3_tracker (Stage 2)**: Wraps v3 with SAM3 tracker heads (PromptEncoder + MaskDecoder). Qwen also predicts 4 positive + 4 negative point coordinates per texture via CoordHead. Coarse DETR masks + points → refined masks.

## Checkpoints

- `checkpoints/v3/best.pt` — Stage 1 DETR (epoch 19, val_iou=0.817)
- `checkpoints/v3_tracker/best.pt` — Stage 2 Tracker (epoch 48, val_tracker_iou=0.842)

## How to Restore / Evaluate

```bash
cd /home/aviad/real_world_texture_boundary

# Evaluate v3 DETR (Stage 1)
python -m qwen2sam.scripts.evaluate_v3 \
  --config qwen2sam/configs/v3.yaml \
  --checkpoint checkpoints/v3/best.pt \
  --split test --output_dir eval_results/v3_test --no_vis

# Evaluate v3_tracker (Stage 2) — 3-way comparison
python -m qwen2sam.scripts.evaluate_v3_tracker \
  --config qwen2sam/configs/v3_tracker.yaml \
  --checkpoint checkpoints/v3_tracker/best.pt \
  --split test --output_dir eval_results/v3_tracker_test --no_vis
```

## How to Retrain

```bash
# Stage 1: DETR (100 epochs)
python -u -m qwen2sam.training.train_v3 --config qwen2sam/configs/v3.yaml

# Stage 2: Tracker (200 epochs, requires Stage 1 checkpoint)
python -u -m qwen2sam.training.train_v3_tracker --config qwen2sam/configs/v3_tracker.yaml
```

## File Structure

```
qwen2sam/
  models/
    qwen2sam_v3.py          # Stage 1 model (multi-token description)
    qwen2sam_v3_tracker.py   # Stage 2 model (tracker refinement)
    losses_v3.py             # v3 losses (focal+dice+alignment+LM)
    qwen2sam_v2.py           # Base model (dependency)
    qwen2sam_v2_tracker.py   # v2 tracker (CoordHead, POINT_TOKENS)
    losses.py                # Shared losses
    losses_v2.py             # v2 seg loss
    losses_v2_tracker.py     # Tracker losses
    alignment.py             # Text embedders
  data/
    dataset_v3.py            # V3Dataset + V3Collator
    dataset_v2.py            # Base dataset (dependency)
    dataset_phase3.py        # Label creation (dependency)
  training/
    train_v3.py              # Stage 1 training loop
    train_v3_tracker.py      # Stage 2 training loop
    train_phase1.py          # Shared utilities (dependency)
  scripts/
    evaluate_v3.py           # Stage 1 evaluation
    evaluate_v3_tracker.py   # Stage 2 evaluation (3-way comparison)
    evaluate_v2.py           # Shared eval utilities (dependency)
  configs/
    v3.yaml                  # Stage 1 config
    v3_tracker.yaml          # Stage 2 config
checkpoints/
  v3/best.pt                 # Stage 1 checkpoint
  v3_tracker/best.pt         # Stage 2 checkpoint
eval_results/
  v3_tracker_test/           # Full test set evaluation
```

## Dependencies

- Conda env: `texture_boundary` (Python 3.11)
- SAM3: `/home/aviad/sam3/` (pip install -e "[dev]")
- Dataset: `/home/aviad/RWTD/` (253 samples, splits.json)
- Qwen2-VL-2B-Instruct (HuggingFace, auto-downloaded)
