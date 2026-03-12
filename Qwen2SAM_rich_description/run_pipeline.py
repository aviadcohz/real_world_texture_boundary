#!/usr/bin/env python3
"""
Qwen2SAM v3 — Full Training & Evaluation Pipeline Runner

Run all steps sequentially with a single command:
    python run_pipeline.py

Or run individual steps:
    python run_pipeline.py --step train_v3
    python run_pipeline.py --step eval_v3
    python run_pipeline.py --step train_tracker
    python run_pipeline.py --step eval_tracker

Steps:
  1. train_v3        — Stage 1: DETR + multi-token description training
  2. eval_v3         — Evaluate Stage 1 checkpoint on RWTD test set
  3. train_tracker   — Stage 2: Tracker refinement training
  4. eval_tracker    — Evaluate Stage 2 checkpoint on RWTD test set (3-way)

Options:
  --smoke           Run a quick smoke test (10 samples, 5 epochs) instead
  --resume          Resume training from last checkpoint
  --skip_eval       Skip evaluation steps
  --wandb_off       Disable W&B logging
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ---- Configuration -------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).parent
CONDA_ENV = "texture_boundary"

# Detecture dataset configs
V3_CONFIG = "qwen2sam/configs/v3_detecture.yaml"
TRACKER_CONFIG = "qwen2sam/configs/v3_tracker_detecture.yaml"

# Smoke test config (created dynamically if needed)
SMOKE_CONFIG = "qwen2sam/configs/v3_smoke_test.yaml"

# Checkpoints
V3_CHECKPOINT_DIR = "checkpoints/v3_detecture"
TRACKER_CHECKPOINT_DIR = "checkpoints/v3_tracker_detecture"

# Evaluation
EVAL_DATA_ROOT = "/home/aviad/RWTD"
EVAL_V3_OUTPUT = "eval_results/v3_detecture"
EVAL_TRACKER_OUTPUT = "eval_results/v3_tracker_detecture"


# ---- Helpers -------------------------------------------------------------- #

def run_cmd(cmd, description, cwd=None):
    """Run a command and stream output in real-time."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"  CMD: {cmd}")
    print(f"{'='*70}\n")

    start = time.time()
    result = subprocess.run(
        cmd, shell=True, cwd=cwd or str(PROJECT_ROOT),
        # Stream stdout/stderr directly to terminal
        stdout=sys.stdout, stderr=sys.stderr,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False

    print(f"\n  DONE in {elapsed:.1f}s")
    return True


def find_latest_checkpoint(ckpt_dir):
    """Find latest epoch checkpoint for resuming."""
    ckpt_path = PROJECT_ROOT / ckpt_dir
    if not ckpt_path.exists():
        return None

    epoch_files = sorted(ckpt_path.glob("epoch_*.pt"))
    if epoch_files:
        return str(epoch_files[-1])

    best = ckpt_path / "best.pt"
    if best.exists():
        return str(best)

    return None


def check_checkpoint_exists(ckpt_dir, name="best.pt"):
    """Check if a checkpoint file exists."""
    path = PROJECT_ROOT / ckpt_dir / name
    return path.exists()


def create_smoke_config():
    """Create a minimal smoke test config if it doesn't exist."""
    smoke_path = PROJECT_ROOT / SMOKE_CONFIG
    if smoke_path.exists():
        return

    smoke_content = """# Smoke test config — 10 samples, 5 epochs
model:
  qwen_model: "Qwen/Qwen2.5-VL-3B-Instruct"
  qwen_dtype: "bfloat16"
  gradient_checkpointing: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.0
  lora_target_modules: ["q_proj", "v_proj"]
  desc_projector_hidden: 1024
  max_desc_tokens: 16
  sam3_checkpoint: null
  freeze_decoder_layers: true
  sam3_lr_scale: 1.0
  image_size: 1008
  align_embedder: "sentence"
  align_embed_dim: 768
  align_model_name: "all-mpnet-base-v2"

hires:
  enabled: false

data:
  data_root: "/datasets/ade20k/Detecture_dataset"
  metadata_file: "metadata.json"
  train_size: 10
  val_size: 5
  num_workers: 2
  augmentation: false
  system_prompt: "You are a texture boundary segmentation assistant."
  user_prompt: "Identify the two textures in this image and segment them."

training:
  batch_size: 2
  gradient_accumulation_steps: 2
  num_epochs: 5
  learning_rate: 1.0e-4
  weight_decay: 0.01
  warmup_epochs: 1
  min_lr: 1.0e-6
  max_grad_norm: 1.0
  seg_grad_to_lm: false

loss:
  seg_weight: 1.0
  focal_weight: 5.0
  dice_weight: 1.0
  cls_weight: 2.0
  box_l1_weight: 5.0
  box_giou_weight: 2.0
  exclusivity_weight: 0.5
  alignment_weight: 1.0
  alignment_temperature: 0.07
  lm_weight: 0.5

amp:
  enabled: true
  dtype: "bfloat16"

checkpoint:
  dir: "checkpoints/v3_smoke"
  save_every_n_epochs: 5

logging:
  log_every_n_steps: 2
  use_wandb: false

validation:
  every_n_epochs: 1

seed: 42
"""
    smoke_path.write_text(smoke_content)
    print(f"  Created smoke config: {smoke_path}")


# ---- Pipeline Steps ------------------------------------------------------ #

def step_train_v3(args):
    """Stage 1: Train DETR with multi-token descriptions."""
    if args.smoke:
        create_smoke_config()
        config = SMOKE_CONFIG
        ckpt_dir = "checkpoints/v3_smoke"
    else:
        config = V3_CONFIG
        ckpt_dir = V3_CHECKPOINT_DIR

    cmd = f"python -m qwen2sam.training.train_v3 --config {config}"

    if args.resume:
        ckpt = find_latest_checkpoint(ckpt_dir)
        if ckpt:
            cmd += f" --resume {ckpt}"
            print(f"  Resuming from: {ckpt}")
        else:
            print("  No checkpoint found, starting fresh")

    if args.wandb_off:
        # Override wandb via env var
        os.environ["WANDB_MODE"] = "disabled"

    return run_cmd(cmd, "Stage 1: Training Qwen2SAM v3 (DETR + Descriptions)")


def step_eval_v3(args):
    """Evaluate Stage 1 on RWTD test set."""
    if args.smoke:
        config = SMOKE_CONFIG
        ckpt = "checkpoints/v3_smoke/best.pt"
        output = "eval_results/v3_smoke"
    else:
        config = V3_CONFIG
        ckpt = f"{V3_CHECKPOINT_DIR}/best.pt"
        output = EVAL_V3_OUTPUT

    if not check_checkpoint_exists(os.path.dirname(ckpt)):
        print(f"  Checkpoint not found: {ckpt}")
        print("  Run train_v3 first!")
        return False

    cmd = (
        f"python -m qwen2sam.scripts.evaluate_v3 "
        f"--config {config} "
        f"--checkpoint {ckpt} "
        f"--data_root {EVAL_DATA_ROOT} "
        f"--output_dir {output} "
        f"--split test"
    )
    return run_cmd(cmd, "Stage 1: Evaluating v3 on RWTD test set")


def step_train_tracker(args):
    """Stage 2: Train tracker refinement."""
    if args.smoke:
        print("  Smoke test for tracker not configured — skipping")
        return True

    config = TRACKER_CONFIG
    ckpt_dir = TRACKER_CHECKPOINT_DIR

    # Verify Stage 1 checkpoint exists
    v3_best = PROJECT_ROOT / V3_CHECKPOINT_DIR / "best.pt"
    if not v3_best.exists():
        print(f"  Stage 1 checkpoint not found: {v3_best}")
        print("  Run train_v3 first!")
        return False

    cmd = f"python -m qwen2sam.training.train_v3_tracker --config {config}"

    if args.resume:
        ckpt = find_latest_checkpoint(ckpt_dir)
        if ckpt:
            cmd += f" --resume {ckpt}"
            print(f"  Resuming from: {ckpt}")

    if args.wandb_off:
        os.environ["WANDB_MODE"] = "disabled"

    return run_cmd(cmd, "Stage 2: Training Qwen2SAM v3_tracker (Refinement)")


def step_eval_tracker(args):
    """Evaluate Stage 2 on RWTD test set (3-way comparison)."""
    if args.smoke:
        print("  Smoke test for tracker eval not configured — skipping")
        return True

    config = TRACKER_CONFIG
    ckpt = f"{TRACKER_CHECKPOINT_DIR}/best.pt"

    if not check_checkpoint_exists(TRACKER_CHECKPOINT_DIR):
        print(f"  Tracker checkpoint not found: {ckpt}")
        print("  Run train_tracker first!")
        return False

    cmd = (
        f"python -m qwen2sam.scripts.evaluate_v3_tracker "
        f"--config {config} "
        f"--checkpoint {ckpt} "
        f"--data_root {EVAL_DATA_ROOT} "
        f"--output_dir {EVAL_TRACKER_OUTPUT} "
        f"--split test"
    )
    return run_cmd(cmd, "Stage 2: Evaluating v3_tracker on RWTD test (3-way)")


# ---- Main ---------------------------------------------------------------- #

STEPS = {
    "train_v3": step_train_v3,
    "eval_v3": step_eval_v3,
    "train_tracker": step_train_tracker,
    "eval_tracker": step_eval_tracker,
}

FULL_PIPELINE = ["train_v3", "eval_v3", "train_tracker", "eval_tracker"]


def main():
    parser = argparse.ArgumentParser(
        description="Qwen2SAM v3 Full Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--step", type=str, default=None,
        choices=list(STEPS.keys()),
        help="Run a single step (default: run all steps)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run smoke test (10 samples, 5 epochs)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint",
    )
    parser.add_argument(
        "--skip_eval", action="store_true",
        help="Skip evaluation steps",
    )
    parser.add_argument(
        "--wandb_off", action="store_true",
        help="Disable W&B logging",
    )
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"#  Qwen2SAM v3 Pipeline Runner")
    print(f"#  Working dir: {PROJECT_ROOT}")
    print(f"#  Mode: {'smoke test' if args.smoke else 'full training'}")
    print(f"#  W&B: {'disabled' if args.wandb_off else 'enabled'}")
    print(f"{'#'*70}")

    if args.step:
        steps = [args.step]
    else:
        steps = FULL_PIPELINE.copy()
        if args.skip_eval:
            steps = [s for s in steps if not s.startswith("eval")]

    results = {}
    for step_name in steps:
        success = STEPS[step_name](args)
        results[step_name] = success
        if not success:
            print(f"\n  Step '{step_name}' failed! Stopping pipeline.")
            break

    # Summary
    print(f"\n{'='*70}")
    print("Pipeline Summary:")
    for step_name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {step_name:20s} [{status}]")
    print(f"{'='*70}")

    # Print eval results location
    if "eval_v3" in results and results.get("eval_v3"):
        output = "eval_results/v3_smoke" if args.smoke else EVAL_V3_OUTPUT
        print(f"\n  Stage 1 results: {output}/summary.json")
    if "eval_tracker" in results and results.get("eval_tracker"):
        print(f"  Stage 2 results: {EVAL_TRACKER_OUTPUT}/summary.json")

    all_ok = all(results.values())
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
