"""
Checkpoint utilities for Qwen2SAM_advance.

Handles loading SAM3 pretrained weights while skipping CLIP text encoder keys,
and saving/loading checkpoints without Qwen's 3B base weights.
"""

from pathlib import Path
from typing import Optional, Set

import torch
from iopath.common.file_io import g_pathmgr


# Keys that belong to CLIP text encoder — skip when loading SAM3 checkpoint
CLIP_KEY_PREFIX = "backbone.language_backbone."


def load_sam3_checkpoint_skip_clip(model, checkpoint_path: str):
    """
    Load SAM3 pretrained weights, skipping all CLIP text encoder keys.

    The CLIP keys (backbone.language_backbone.*) are incompatible with
    QwenTextEncoder and will be initialized from scratch (adapter) or
    from HuggingFace (Qwen base weights).

    Args:
        model: Sam3Image model with QwenTextEncoder
        checkpoint_path: Path to SAM3 pretrained .pt file
    """
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)

    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    # Extract detector weights, strip "detector." prefix
    sam3_state = {
        k.replace("detector.", ""): v
        for k, v in ckpt.items()
        if "detector" in k
    }

    # Filter out CLIP text encoder keys
    filtered_state = {
        k: v for k, v in sam3_state.items()
        if not k.startswith(CLIP_KEY_PREFIX)
    }

    skipped = len(sam3_state) - len(filtered_state)
    print(f"Loading SAM3 checkpoint: {len(filtered_state)} keys loaded, "
          f"{skipped} CLIP text encoder keys skipped")

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)

    # Expected missing: all language_backbone keys (Qwen + adapter)
    expected_missing = [k for k in missing_keys if k.startswith(CLIP_KEY_PREFIX)]
    truly_missing = [k for k in missing_keys if not k.startswith(CLIP_KEY_PREFIX)]

    if truly_missing:
        print(f"WARNING: Unexpected missing keys: {truly_missing}")
    if unexpected_keys:
        print(f"WARNING: Unexpected keys in checkpoint: {unexpected_keys}")

    print(f"SAM3 checkpoint loaded successfully "
          f"({len(expected_missing)} language_backbone keys initialized fresh)")


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    step: int,
    save_path: str,
    skip_qwen_base: bool = True,
    best_metric: Optional[float] = None,
):
    """
    Save training checkpoint, optionally skipping Qwen's base weights.

    Skipping Qwen base weights saves ~6GB per checkpoint. The base weights
    are loaded from HuggingFace cache on resume.

    Args:
        model: The full model
        optimizer: Optimizer state
        epoch: Current epoch
        step: Current step
        save_path: Path to save checkpoint
        skip_qwen_base: If True, skip Qwen's base (non-LoRA) weights
        best_metric: Best metric value (for tracking)
    """
    state_dict = model.state_dict()

    if skip_qwen_base:
        # Keep only: adapter weights, LoRA weights, all non-Qwen weights
        keys_to_save = {}
        skipped = 0
        for k, v in state_dict.items():
            is_qwen_base = (
                CLIP_KEY_PREFIX + "qwen." in k
                and "lora" not in k.lower()
            )
            if is_qwen_base:
                skipped += 1
            else:
                keys_to_save[k] = v

        print(f"Saving checkpoint: {len(keys_to_save)} keys "
              f"({skipped} Qwen base keys skipped)")
        state_dict = keys_to_save

    checkpoint = {
        "model": state_dict,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
        "skip_qwen_base": skip_qwen_base,
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    model,
    checkpoint_path: str,
    optimizer=None,
    device: str = "cpu",
):
    """
    Load a training checkpoint.

    If skip_qwen_base was True when saving, Qwen base weights won't be in
    the checkpoint — they're already loaded from HuggingFace in the model.

    Returns:
        dict with 'epoch', 'step', 'best_metric'
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model_state = ckpt["model"]
    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)

    # If Qwen base was skipped, missing keys are expected
    if ckpt.get("skip_qwen_base", False):
        qwen_missing = [k for k in missing_keys if "qwen." in k and "lora" not in k.lower()]
        other_missing = [k for k in missing_keys if k not in qwen_missing]
        if other_missing:
            print(f"WARNING: Unexpected missing keys: {other_missing}")
        print(f"Loaded checkpoint (epoch {ckpt['epoch']}), "
              f"{len(qwen_missing)} Qwen base keys from HuggingFace")
    else:
        if missing_keys:
            print(f"WARNING: Missing keys: {missing_keys}")

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    return {
        "epoch": ckpt.get("epoch", 0),
        "step": ckpt.get("step", 0),
        "best_metric": ckpt.get("best_metric"),
    }
