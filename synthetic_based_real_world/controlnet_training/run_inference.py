#!/usr/bin/env python3
"""
Generate synthetic texture-boundary images using a trained ControlNet.

Loads boundary masks from the inference split, runs ControlNet + SD 1.5,
and saves generated images alongside their conditioning masks.

Just press "Run Python File" in VS Code — no CLI args needed.
All configuration is in the section below.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — edit these before running                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

EPOCH = 100
CHECKPOINT_PATH = Path(
    "/home/aviad/real_world_texture_boundary"
    f"/synthetic_based_real_world/controlnet_training/checkpoints/controlnet_epoch_{EPOCH}"
)
PRETRAINED_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"

DATA_JSON = Path(
    "/home/aviad/real_world_texture_boundary"
    "/synthetic_based_real_world/controlnet_training/checkpoints/inference_split.json"
)

OUTPUT_DIR = Path(f"/datasets/synthatic_dataset/run_{datetime.now().strftime('%d_%m')}")

NUM_IMAGES = 1000         # how many images to generate
INFERENCE_STEPS = 30      # diffusion denoising steps
GUIDANCE_SCALE = 7.5      # classifier-free guidance scale
SEED = 42                 # random seed for reproducibility
DEVICE = "auto"           # "auto" (tries GPU, falls back CPU), "cuda", or "cpu"
SHUFFLE_PROMPTS = False   # True = randomly pair masks with prompts to enrich data

# ═══════════════════════════════════════════════════════════════════════════════


def get_device():
    if DEVICE == "auto":
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            print(f"GPU free memory: {free_gb:.1f} GB")
            return "cuda" if free_gb > 6 else "cpu"
        return "cpu"
    return DEVICE


def mask_to_tensor(mask_pil, device):
    """Convert grayscale PIL mask → 1-channel tensor [0,1] shape (1,1,H,W)."""
    mask_np = np.array(mask_pil).astype(np.float32) / 255.0
    return torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)


def load_samples():
    with open(DATA_JSON) as f:
        all_samples = json.load(f)

    rng = np.random.RandomState(SEED)

    # Filter to samples whose masks exist on disk
    valid = []
    for entry in all_samples:
        if Path(entry["conditioning_image"]).exists():
            valid.append(entry)
        else:
            print(f"  WARNING: mask not found, skipping: {entry['conditioning_image']}")

    samples = []
    round_num = 0

    while len(samples) < NUM_IMAGES:
        # --- matched pairs ---
        need = min(NUM_IMAGES - len(samples), len(valid))
        indices = rng.choice(len(valid), size=need, replace=False)
        matched = [dict(valid[i]) for i in sorted(indices)]
        for s in matched:
            s["pair_type"] = "matched"
        samples.extend(matched)
        round_num += 1
        print(f"  Round {round_num}: +{len(matched)} matched pairs  (total {len(samples)})")

        if len(samples) >= NUM_IMAGES:
            break

        # --- shuffled cross-pairs ---
        if SHUFFLE_PROMPTS:
            need = min(NUM_IMAGES - len(samples), len(valid))
            masks = [v["conditioning_image"] for v in valid]
            prompts = [v["text"] for v in valid]
            rng.shuffle(prompts)
            extra_idx = rng.choice(len(valid), size=need, replace=False)
            shuffled = []
            for i in sorted(extra_idx):
                shuffled.append({
                    "conditioning_image": masks[i],
                    "text": prompts[i],
                    "image": valid[i]["image"],
                    "pair_type": "shuffled",
                })
            samples.extend(shuffled)
            print(f"         +{len(shuffled)} shuffled cross-pairs  (total {len(samples)})")
        else:
            # without shuffle we can't generate more than len(valid), stop
            break

    samples = samples[:NUM_IMAGES]
    print(f"Final: {len(samples)} samples "
          f"({sum(1 for s in samples if s['pair_type'] == 'matched')} matched, "
          f"{sum(1 for s in samples if s['pair_type'] == 'shuffled')} shuffled)")
    return samples


def build_pipeline(device, dtype):
    import diffusers
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )
    diffusers.utils.logging.set_verbosity_error()

    print(f"Loading ControlNet from {CHECKPOINT_PATH} ...")
    controlnet = ControlNetModel.from_pretrained(
        str(CHECKPOINT_PATH), torch_dtype=dtype
    ).to(device)

    print(f"Loading SD pipeline ({PRETRAINED_MODEL}) ...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        PRETRAINED_MODEL,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    return pipe


def main():
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device} ({dtype})")

    # Load data
    samples = load_samples()
    if not samples:
        print("ERROR: no valid samples found")
        return

    # Create output dirs
    images_dir = OUTPUT_DIR / "images"
    masks_dir = OUTPUT_DIR / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Build pipeline
    pipe = build_pipeline(device, dtype)
    print(f"\nGenerating {len(samples)} images → {OUTPUT_DIR}\n")

    metadata = []
    t0 = time.time()

    for idx, entry in enumerate(samples):
        # Load and preprocess mask
        mask_pil = (
            Image.open(entry["conditioning_image"])
            .convert("L")
            .resize((512, 512), Image.NEAREST)
        )
        mask_tensor = mask_to_tensor(mask_pil, device)
        prompt = entry["text"]

        # Run inference
        generator = torch.Generator(device=device).manual_seed(SEED + idx)
        result = pipe(
            prompt=prompt,
            image=mask_tensor,
            num_inference_steps=INFERENCE_STEPS,
            generator=generator,
            guidance_scale=GUIDANCE_SCALE,
        )
        gen_image = result.images[0]

        # Save outputs
        img_name = f"{idx:04d}.png"
        gen_image.save(images_dir / img_name)
        mask_pil.save(masks_dir / img_name)

        metadata.append({
            "index": idx,
            "image": f"images/{img_name}",
            "mask": f"masks/{img_name}",
            "prompt": prompt,
            "source_image": entry["image"],
            "source_mask": entry["conditioning_image"],
        })

        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - t0
            per_img = elapsed / (idx + 1)
            remaining = per_img * (len(samples) - idx - 1)
            print(
                f"  [{idx+1:4d}/{len(samples)}]  "
                f"{per_img:.1f}s/img  "
                f"~{remaining/60:.0f}min remaining  "
                f'"{prompt[:50]}"'
            )

    # Save metadata
    meta_path = OUTPUT_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "checkpoint": str(CHECKPOINT_PATH),
                "pretrained_model": PRETRAINED_MODEL,
                "num_images": len(metadata),
                "inference_steps": INFERENCE_STEPS,
                "guidance_scale": GUIDANCE_SCALE,
                "seed": SEED,
                "created": datetime.now().isoformat(),
                "samples": metadata,
            },
            f,
            indent=2,
        )

    elapsed = time.time() - t0
    print(f"\nDone! Generated {len(metadata)} images in {elapsed/60:.1f} minutes")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Metadata: {meta_path}")

    # Cleanup
    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
