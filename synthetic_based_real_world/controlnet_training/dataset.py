"""
ControlNet training dataset for texture boundary generation.

Loads training_pairs.json and provides (image, conditioning_mask, text) tuples
with proper preprocessing for Stable Diffusion ControlNet training.

Preprocessing contract:
    Target image      → resize 512×512, normalize to [-1, 1]  (VAE input range)
    Conditioning mask → resize 512×512, normalize to [0, 1]   (hint encoder input)
    Text prompt       → CLIP tokenized, padded to 77 tokens

Dataset splits (deterministic, seeded shuffle):
    70% train      — used for ControlNet training
    20% validation — used for validation loss tracking
    10% inference  — held out for synthetic data generation
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "inference": 0.10}
SPLIT_SEED = 42  # deterministic split across runs


def load_and_split_pairs(
    json_path: str,
    min_source_size: int = 128,
    split_ratios: dict = None,
    seed: int = SPLIT_SEED,
) -> dict:
    """
    Load training_pairs.json, filter small images, split into train/val/inference.

    Returns dict with keys "train", "val", "inference", each a list of entries.
    The split is deterministic (same seed → same split every run).
    """
    if split_ratios is None:
        split_ratios = SPLIT_RATIOS

    with open(json_path) as f:
        all_pairs = json.load(f)

    # Filter out images below minimum size
    valid_pairs = []
    skipped = 0
    for entry in all_pairs:
        img_path = Path(entry["image"])
        if not img_path.exists():
            skipped += 1
            continue
        with Image.open(img_path) as img:
            w, h = img.size
        if w < min_source_size or h < min_source_size:
            skipped += 1
            continue
        valid_pairs.append(entry)

    # Deterministic shuffle
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(valid_pairs))

    n_train = int(len(valid_pairs) * split_ratios["train"])
    n_val = int(len(valid_pairs) * split_ratios["val"])

    splits = {
        "train": [valid_pairs[i] for i in indices[:n_train]],
        "val": [valid_pairs[i] for i in indices[n_train:n_train + n_val]],
        "inference": [valid_pairs[i] for i in indices[n_train + n_val:]],
    }

    print(
        f"Dataset: {len(valid_pairs)} valid pairs ({skipped} skipped), "
        f"split → train:{len(splits['train'])} "
        f"val:{len(splits['val'])} "
        f"inference:{len(splits['inference'])}"
    )

    return splits


class TextureBoundaryDataset(Dataset):
    """
    Dataset for ControlNet texture boundary training.

    Each sample returns:
        pixel_values:      (3, 512, 512) float32 in [-1, 1]
        conditioning_image: (1, 512, 512) float32 in [0, 1]
        input_ids:         (77,) long — CLIP token IDs
    """

    def __init__(
        self,
        pairs: list,
        tokenizer,
        resolution: int = 512,
    ):
        """
        Args:
            pairs:      List of {image, conditioning_image, text} dicts
            tokenizer:  CLIPTokenizer instance
            resolution: Target resolution (default 512)
        """
        self.pairs = pairs
        self.resolution = resolution
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        entry = self.pairs[idx]

        # ── Load image ───────────────────────────────────────────────
        image = Image.open(entry["image"]).convert("RGB")
        image = image.resize(
            (self.resolution, self.resolution), Image.LANCZOS
        )
        # [0, 255] uint8 → [-1, 1] float32  (VAE pretrained range)
        image = np.array(image).astype(np.float32)
        image = (image / 127.5) - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC → CHW

        # ── Load conditioning mask ───────────────────────────────────
        mask = Image.open(entry["conditioning_image"]).convert("L")
        # NEAREST preserves binary edges — no interpolation blur
        mask = mask.resize(
            (self.resolution, self.resolution), Image.NEAREST
        )
        # [0, 255] uint8 → [0, 1] float32  (hint encoder input range)
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)  # H,W → 1,H,W

        # ── Tokenize text ────────────────────────────────────────────
        tokens = self.tokenizer(
            entry["text"],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,  # 77
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.squeeze(0)  # (1, 77) → (77,)

        return {
            "pixel_values": image,
            "conditioning_image": mask,
            "input_ids": input_ids,
        }
