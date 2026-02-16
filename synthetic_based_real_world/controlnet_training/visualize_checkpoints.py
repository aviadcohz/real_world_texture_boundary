#!/usr/bin/env python3
"""
Visualize ControlNet checkpoints: run inference and log triplet views to W&B.

For each checkpoint, runs inference on 10 fixed samples and logs to W&B:
    - Boundary mask (conditioning input)
    - Generated image (ControlNet output)
    - Original image (ground truth that the mask was derived from)
    - Text prompt

Just press "Run Python File" in VS Code — no CLI args needed.
All configuration is in the section below.
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — edit these before running                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

WANDB_RUN_ID = "0n0jca8u"                              # your current training run ID
WANDB_PROJECT = "controlnet-texture-boundary"
WANDB_ENTITY = "aviadcohz-tel-aviv-university"

CHECKPOINT_DIR = Path(
    "/home/aviad/real_world_texture_boundary"
    "/synthetic_based_real_world/controlnet_training/checkpoints"
)
INFERENCE_JSON = CHECKPOINT_DIR / "inference_split.json"
PRETRAINED_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"

NUM_SAMPLES = 10          # number of inference samples per checkpoint
INFERENCE_STEPS = 30      # diffusion steps for generation
SEED = 42                 # fixed seed for reproducible results
DEVICE = "auto"           # "auto" (tries GPU, falls back CPU), "cuda", or "cpu"
WATCH_MODE = False        # True = keep polling for new checkpoints every 5 min
EPOCHS = [5, 40, 60]             # None = all checkpoints, or e.g. [5, 10, 25, 50] for specific ones

# ═══════════════════════════════════════════════════════════════════════════════


def get_device():
    if DEVICE == "auto":
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            print(f"GPU free memory: {free_gb:.1f} GB")
            return "cuda" if free_gb > 6 else "cpu"
        return "cpu"
    return DEVICE


def discover_checkpoints():
    found = {}
    for d in sorted(CHECKPOINT_DIR.glob("controlnet_epoch_*")):
        if (d / "diffusion_pytorch_model.safetensors").exists():
            m = re.search(r"controlnet_epoch_(\d+)", d.name)
            if m:
                ep = int(m.group(1))
                found[ep] = d
    best = CHECKPOINT_DIR / "best_model"
    if (best / "diffusion_pytorch_model.safetensors").exists():
        found["best"] = best
    if EPOCHS is not None:
        found = {k: v for k, v in found.items() if k in EPOCHS or k == "best"}
    return found


def load_samples():
    with open(INFERENCE_JSON) as f:
        all_samples = json.load(f)
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(all_samples), size=min(NUM_SAMPLES, len(all_samples)), replace=False)
    samples = []
    for i in sorted(idx):
        entry = all_samples[i]
        ip, mp = Path(entry["image"]), Path(entry["conditioning_image"])
        if ip.exists() and mp.exists():
            samples.append({
                "image_pil": Image.open(ip).convert("RGB").resize((512, 512), Image.LANCZOS),
                "mask_pil": Image.open(mp).convert("L").resize((512, 512), Image.NEAREST),
                "text": entry["text"],
            })
    print(f"Loaded {len(samples)} inference samples")
    return samples


def mask_to_tensor(mask_pil, device):
    """Convert grayscale PIL mask → 1-channel tensor [0,1] shape (1,1,H,W)."""
    mask_np = np.array(mask_pil).astype(np.float32) / 255.0
    return torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)


def run_inference(pipeline, samples, device):
    generated = []
    for i, sample in enumerate(samples):
        print(f"  Generating sample {i+1}/{len(samples)}: \"{sample['text'][:50]}\"")
        gen = torch.Generator(device=device).manual_seed(SEED)
        mask_tensor = mask_to_tensor(sample["mask_pil"], device)
        out = pipeline(
            prompt=sample["text"],
            image=mask_tensor,
            num_inference_steps=INFERENCE_STEPS,
            generator=gen,
            guidance_scale=7.5,
        ).images[0]
        generated.append(out)
    return generated


def make_triplet(mask_pil, generated_pil, original_pil, prompt, cell=256):
    """Create [Mask | Generated | Original] with prompt label."""
    w = 3 * cell + 8
    h = cell + 22
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(Image.merge("RGB", [mask_pil.resize((cell, cell))] * 3), (2, 0))
    canvas.paste(generated_pil.resize((cell, cell), Image.LANCZOS), (cell + 4, 0))
    canvas.paste(original_pil.resize((cell, cell), Image.LANCZOS), (2 * cell + 6, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
    draw.text((4, cell + 4), f"Mask | Generated | GT  --  \"{prompt[:80]}\"", fill="gray", font=font)
    return canvas


def main():
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device} ({dtype})")

    samples = load_samples()
    if not samples:
        print("ERROR: no valid inference samples found")
        return

    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    import wandb

    # Connect to existing W&B run
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        id=WANDB_RUN_ID,
        resume="must",
    )
    print(f"Connected to W&B: {run.url}")

    visualized = set()

    while True:
        checkpoints = discover_checkpoints()
        new_ckpts = {k: v for k, v in checkpoints.items() if k not in visualized}

        if not new_ckpts:
            if WATCH_MODE:
                print("\nNo new checkpoints, waiting 5 min...")
                time.sleep(300)
                continue
            elif not visualized:
                print("No checkpoints found!")
            break

        for ep_key, ckpt_path in sorted(new_ckpts.items(), key=lambda x: (isinstance(x[0], str), x[0])):
            label = f"epoch_{ep_key}" if isinstance(ep_key, int) else ep_key
            print(f"\n{'='*50}")
            print(f"Inference: {label}")
            print(f"{'='*50}")

            try:
                # Load ControlNet from this checkpoint
                controlnet = ControlNetModel.from_pretrained(
                    str(ckpt_path), torch_dtype=dtype,
                ).to(device)

                # Build full pipeline
                pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    PRETRAINED_MODEL,
                    controlnet=controlnet,
                    torch_dtype=dtype,
                    safety_checker=None,
                ).to(device)
                pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

                # Run inference on all samples
                generated = run_inference(pipe, samples, device)

                # ── Log to W&B as a Table (scrollable grid) ──────────
                columns = ["sample_id", "prompt", "mask", "generated", "ground_truth", "triplet"]
                table = wandb.Table(columns=columns)

                for i, (sample, gen_img) in enumerate(zip(samples, generated)):
                    triplet = make_triplet(
                        sample["mask_pil"], gen_img, sample["image_pil"], sample["text"]
                    )
                    table.add_data(
                        i,
                        sample["text"],
                        wandb.Image(sample["mask_pil"], caption="Boundary mask"),
                        wandb.Image(gen_img, caption="ControlNet output"),
                        wandb.Image(sample["image_pil"], caption="Ground truth"),
                        wandb.Image(triplet, caption=f"{label}: {sample['text'][:50]}"),
                    )

                run.log({f"inference/{label}": table})
                print(f"Logged {len(samples)} samples to W&B: inference/{label}")

                # Also log individual triplets for the Media panel
                for i, (sample, gen_img) in enumerate(zip(samples, generated)):
                    triplet = make_triplet(
                        sample["mask_pil"], gen_img, sample["image_pil"], sample["text"]
                    )
                    run.log({
                        f"triplets/{label}/sample_{i}": wandb.Image(
                            triplet, caption=f"{sample['text'][:60]}"
                        ),
                    })

                visualized.add(ep_key)
                print(f"Done with {label}!")

            except Exception as e:
                print(f"ERROR on {label}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                try:
                    del controlnet, pipe
                except NameError:
                    pass
                if device == "cuda":
                    torch.cuda.empty_cache()

        if not WATCH_MODE:
            break

    print(f"\nAll done! Visualized: {sorted(e for e in visualized if isinstance(e, int))}")
    print(f"View results at: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
