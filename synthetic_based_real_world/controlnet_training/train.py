#!/usr/bin/env python3
"""
ControlNet training script for texture boundary conditioning.

Training loop:
    1. Encode target image → latent z via frozen VAE
    2. Sample random timestep t, add noise → z_t
    3. Encode text prompt → CLIP embeddings
    4. ControlNet(z_t, t, text, boundary_mask) → residuals
    5. UNet(z_t, t, text, controlnet_residuals) → predicted noise ε_θ
    6. Loss = MSE(ε_θ, ε)
    7. Backprop through ControlNet only (UNet/VAE/CLIP frozen)

Monitoring via Weights & Biases:
    - Training loss (per step + epoch average)
    - Validation loss (per epoch)
    - Learning rate, GPU memory
    - Sample generated images (periodically)

Usage:
    python train.py
    python train.py --batch-size 4 --lr 1e-5 --epochs 50
    python train.py --resume checkpoint_epoch_10.pt
    python train.py --wandb-project my_project --wandb-run my_run
"""

import argparse
import json
import logging
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont

import wandb

from controlnet_model import create_controlnet, load_frozen_components
from dataset import TextureBoundaryDataset, load_and_split_pairs

# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_JSON = (
    "/datasets/google_landmarks_v2_scale/real_world_texture_boundary"
    "/results/controlnet_data_for_training/training_pairs.json"
)
DEFAULT_PRETRAINED = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_OUTPUT = (
    "/home/aviad/real_world_texture_boundary"
    "/synthetic_based_real_world/controlnet_training/checkpoints"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ControlNet for texture boundary conditioning."
    )
    # Data
    parser.add_argument(
        "--json-path", type=str, default=DEFAULT_JSON,
        help="Path to training_pairs.json",
    )
    parser.add_argument(
        "--min-source-size", type=int, default=128,
        help="Filter images smaller than this (default: 128, drops 64×64)",
    )

    # Model
    parser.add_argument(
        "--pretrained-model", type=str, default=DEFAULT_PRETRAINED,
        help="HuggingFace SD 1.5 model ID",
    )

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1,
        help="Accumulate gradients over N steps (effective batch = batch_size × N)",
    )
    parser.add_argument(
        "--mixed-precision", action="store_true",
        help="Use fp16 mixed precision for frozen components",
    )

    # Checkpointing
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT,
        help="Directory for checkpoints and logs",
    )
    parser.add_argument(
        "--save-every", type=int, default=5,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--log-every", type=int, default=50,
        help="Log training stats every N steps",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )

    # Weights & Biases
    parser.add_argument(
        "--wandb-project", type=str, default="controlnet-texture-boundary",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-run", type=str, default=None,
        help="W&B run name (auto-generated if not set)",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B logging (use only console + file logs)",
    )
    parser.add_argument(
        "--viz-samples", type=int, default=10,
        help="Number of inference samples to visualize at each checkpoint",
    )
    parser.add_argument(
        "--viz-steps", type=int, default=30,
        help="Number of diffusion steps for visualization inference",
    )

    return parser.parse_args()


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(output_dir: Path):
    logger = logging.getLogger("controlnet_train")
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(ch)

    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "train.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(fh)

    return logger


# ── Training / Validation step ───────────────────────────────────────────────

def forward_step(
    batch,
    controlnet,
    unet,
    vae,
    text_encoder,
    noise_scheduler,
    weight_dtype,
    device,
):
    """
    Single forward pass. Returns the MSE loss.
    Used for both training and validation.

    The forward pass follows the diffusion training procedure:
        1. VAE encode image → latent z₀
        2. Sample noise ε ~ N(0,1) and timestep t ~ U(1, T)
        3. Create noisy latent: z_t = √ᾱ_t · z₀ + √(1-ᾱ_t) · ε
        4. ControlNet(z_t, t, text_emb, boundary_mask) → residuals
        5. UNet(z_t, t, text_emb, residuals) → ε_θ
        6. Loss = ‖ε - ε_θ‖²
    """
    pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
    conditioning = batch["conditioning_image"].to(device, dtype=weight_dtype)
    input_ids = batch["input_ids"].to(device)

    # 1. Encode image → latent space (frozen VAE, no grad)
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    # 2. Sample noise and timestep
    noise = torch.randn_like(latents)
    batch_size = latents.shape[0]
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps,
        (batch_size,), device=device, dtype=torch.long,
    )

    # 3. Add noise to latents: z_t = √ᾱ_t · z₀ + √(1-ᾱ_t) · ε
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # 4. Get text embeddings (frozen CLIP, no grad)
    with torch.no_grad():
        encoder_hidden_states = text_encoder(input_ids)[0]

    noisy_latents = noisy_latents.to(dtype=weight_dtype)
    encoder_hidden_states = encoder_hidden_states.to(dtype=weight_dtype)

    # 5. ControlNet forward → residuals  (THIS is what we're training)
    down_block_res_samples, mid_block_res_sample = controlnet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        controlnet_cond=conditioning,
        return_dict=False,
    )

    # 6. Frozen UNet forward with ControlNet residuals → noise prediction
    #    NOTE: no torch.no_grad() here — UNet params are frozen via
    #    requires_grad_(False), but the computation graph must stay intact
    #    so gradients flow back through the residuals to ControlNet.
    noise_pred = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        down_block_additional_residuals=[
            s.to(dtype=weight_dtype) for s in down_block_res_samples
        ],
        mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
    ).sample

    # 7. MSE loss: L = ‖ε - ε_θ‖²
    loss = F.mse_loss(noise_pred.float(), noise.float())

    return loss


# ── Validation loop ──────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    val_dataloader,
    controlnet,
    unet,
    vae,
    text_encoder,
    noise_scheduler,
    weight_dtype,
    device,
):
    """Run validation and return average loss."""
    controlnet.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_dataloader:
        with torch.amp.autocast("cuda", enabled=(weight_dtype == torch.float16)):
            loss = forward_step(
                batch, controlnet, unet, vae, text_encoder,
                noise_scheduler, weight_dtype, device,
            )
        total_loss += loss.item()
        n_batches += 1

    controlnet.train()
    return total_loss / max(n_batches, 1)


# ── Visualization ─────────────────────────────────────────────────────────────

def select_viz_samples(inference_pairs, num_samples, seed=42):
    """Pick a fixed set of samples from the inference split for visualization."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(inference_pairs), size=min(num_samples, len(inference_pairs)), replace=False)
    samples = []
    for i in sorted(idx):
        entry = inference_pairs[i]
        img_path, mask_path = Path(entry["image"]), Path(entry["conditioning_image"])
        if img_path.exists() and mask_path.exists():
            samples.append({
                "image_pil": Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS),
                "mask_pil": Image.open(mask_path).convert("L").resize((512, 512), Image.NEAREST),
                "text": entry["text"],
            })
    return samples


def make_triplet_image(mask_pil, generated_pil, original_pil, prompt, cell=256):
    """Create a single image: [Mask | Generated | Original] with prompt label."""
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
    draw.text((4, cell + 4), f"Mask | Generated | GT  —  \"{prompt[:80]}\"", fill="gray", font=font)
    return canvas


@torch.no_grad()
def visualize_checkpoint(
    controlnet,
    unet,
    vae,
    text_encoder,
    tokenizer,
    noise_scheduler,
    viz_samples,
    num_steps,
    epoch,
    device,
    logger,
):
    """Run inference on viz_samples and log triplet views to W&B."""
    from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler

    controlnet.eval()
    logger.info(f"Generating {len(viz_samples)} visualization samples...")

    pipe = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    columns = ["sample", "prompt", "mask", "generated", "ground_truth", "triplet"]
    table = wandb.Table(columns=columns)

    generator = torch.Generator(device=device).manual_seed(42)

    for i, sample in enumerate(viz_samples):
        # Convert grayscale mask to 1-channel tensor — ControlNet has conditioning_channels=1
        mask_np = np.array(sample["mask_pil"]).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)

        gen_img = pipe(
            prompt=sample["text"],
            image=mask_tensor,
            num_inference_steps=num_steps,
            generator=generator,
            guidance_scale=7.5,
        ).images[0]

        triplet = make_triplet_image(
            sample["mask_pil"], gen_img, sample["image_pil"], sample["text"],
        )

        table.add_data(
            i,
            sample["text"],
            wandb.Image(sample["mask_pil"], caption="Boundary mask"),
            wandb.Image(gen_img, caption="ControlNet output"),
            wandb.Image(sample["image_pil"], caption="Ground truth"),
            wandb.Image(triplet, caption=f"epoch {epoch+1}: {sample['text'][:50]}"),
        )

    wandb.log({f"inference/epoch_{epoch+1}": table})
    logger.info(f"Logged {len(viz_samples)} triplet visualizations to W&B")

    # Restore ControlNet to train mode
    controlnet.train()
    # Clean up pipeline reference (does not delete the shared model objects)
    del pipe
    torch.cuda.empty_cache()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)
    device = "cuda"

    weight_dtype = torch.float16 if args.mixed_precision else torch.float32

    # ── Load frozen SD components ────────────────────────────────────
    logger.info(f"Loading pretrained SD components from {args.pretrained_model}...")
    components = load_frozen_components(
        pretrained_model=args.pretrained_model,
        device=device,
        dtype=weight_dtype,
    )
    unet = components["unet"]
    vae = components["vae"]
    text_encoder = components["text_encoder"]
    tokenizer = components["tokenizer"]
    noise_scheduler = components["noise_scheduler"]
    logger.info("Frozen components loaded (UNet, VAE, CLIP, scheduler)")

    # ── Create ControlNet (trainable) ────────────────────────────────
    logger.info("Creating ControlNet with 1-channel hint encoder...")
    controlnet = create_controlnet(
        pretrained_model=args.pretrained_model,
        conditioning_channels=1,
    ).to(device, dtype=torch.float32)
    controlnet.train()

    trainable_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
    logger.info(f"ControlNet: {trainable_params/1e6:.1f}M trainable parameters")

    # ── Gradient checkpointing ───────────────────────────────────────
    controlnet.enable_gradient_checkpointing()
    unet.enable_gradient_checkpointing()
    logger.info("Gradient checkpointing enabled (ControlNet + UNet)")

    # ── Dataset split & DataLoaders ──────────────────────────────────
    logger.info(f"Loading dataset from {args.json_path}...")
    splits = load_and_split_pairs(
        json_path=args.json_path,
        min_source_size=args.min_source_size,
    )

    train_dataset = TextureBoundaryDataset(splits["train"], tokenizer)
    val_dataset = TextureBoundaryDataset(splits["val"], tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    logger.info(
        f"Train: {len(train_dataset)} samples, {len(train_dataloader)} batches | "
        f"Val: {len(val_dataset)} samples, {len(val_dataloader)} batches | "
        f"Inference (held out): {len(splits['inference'])} samples"
    )

    # Save inference split for later use
    inference_json = output_dir / "inference_split.json"
    with open(inference_json, "w") as f:
        json.dump(splits["inference"], f, indent=2)
    logger.info(f"Inference split saved → {inference_json}")

    # ── Select fixed visualization samples ──────────────────────────
    viz_samples = select_viz_samples(splits["inference"], args.viz_samples)
    logger.info(f"Selected {len(viz_samples)} fixed samples for visualization")

    # ── Optimizer ────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=args.lr,
        weight_decay=1e-2,
    )

    # ── Resume from checkpoint ───────────────────────────────────────
    start_epoch = 0
    global_step = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        controlnet.load_state_dict(ckpt["controlnet_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        logger.info(f"Resumed at epoch {start_epoch}, step {global_step}")

    # ── Weights & Biases ─────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            entity="aviadcohz-tel-aviv-university",
            project=args.wandb_project,
            name=args.wandb_run,
            config={
                **vars(args),
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "inference_size": len(splits["inference"]),
                "trainable_params": trainable_params,
                "architecture": "ControlNet-SD1.5",
                "dataset": "texture-boundary-15k",
            },
            resume="allow" if args.resume else None,
        )
        wandb.watch(controlnet, log="gradients", log_freq=100)
        logger.info(f"W&B initialized: {wandb.run.url}")
    else:
        logger.info("W&B disabled (--no-wandb)")

    # ── Save training config ─────────────────────────────────────────
    config = vars(args)
    config["train_size"] = len(train_dataset)
    config["val_size"] = len(val_dataset)
    config["inference_size"] = len(splits["inference"])
    config["trainable_params"] = trainable_params
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # ── Training loop ────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("TRAINING START")
    logger.info("=" * 70)
    logger.info(f"  Epochs:          {args.epochs}")
    logger.info(f"  Batch size:      {args.batch_size}")
    logger.info(f"  Grad accum:      {args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch:  {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  Learning rate:   {args.lr}")
    logger.info(f"  Mixed precision: {args.mixed_precision}")
    logger.info(f"  Output dir:      {output_dir}")
    logger.info("=" * 70)

    scaler = torch.amp.GradScaler("cuda") if args.mixed_precision else None
    best_val_loss = float("inf")
    t_train_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        controlnet.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t_epoch_start = time.time()

        for step, batch in enumerate(train_dataloader):
            with torch.amp.autocast("cuda", enabled=args.mixed_precision):
                loss = forward_step(
                    batch, controlnet, unet, vae, text_encoder,
                    noise_scheduler, weight_dtype, device,
                )
                loss = loss / args.gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # W&B step-level logging
                if use_wandb:
                    wandb.log({
                        "train/loss_step": loss.item() * args.gradient_accumulation_steps,
                        "train/global_step": global_step,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }, step=global_step)

            epoch_loss += loss.item() * args.gradient_accumulation_steps
            epoch_steps += 1

            # Console logging
            if (step + 1) % args.log_every == 0:
                avg_loss = epoch_loss / epoch_steps
                logger.info(
                    f"Epoch {epoch+1}/{args.epochs} | "
                    f"Step {step+1}/{len(train_dataloader)} | "
                    f"Loss: {loss.item() * args.gradient_accumulation_steps:.6f} | "
                    f"Avg: {avg_loss:.6f} | "
                    f"GStep: {global_step}"
                )

        # ── Epoch summary ────────────────────────────────────────────
        train_avg_loss = epoch_loss / max(epoch_steps, 1)
        epoch_time = time.time() - t_epoch_start

        # ── Validation ───────────────────────────────────────────────
        val_loss = validate(
            val_dataloader, controlnet, unet, vae, text_encoder,
            noise_scheduler, weight_dtype, device,
        )

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} DONE | "
            f"Train loss: {train_avg_loss:.6f} | "
            f"Val loss: {val_loss:.6f} | "
            f"Time: {timedelta(seconds=int(epoch_time))}"
        )

        # W&B epoch-level logging
        if use_wandb:
            log_dict = {
                "train/loss_epoch": train_avg_loss,
                "val/loss_epoch": val_loss,
                "epoch": epoch + 1,
                "epoch_time_seconds": epoch_time,
            }
            # GPU memory stats
            if torch.cuda.is_available():
                log_dict["gpu/memory_allocated_gb"] = (
                    torch.cuda.memory_allocated() / 1e9
                )
                log_dict["gpu/memory_reserved_gb"] = (
                    torch.cuda.memory_reserved() / 1e9
                )
            wandb.log(log_dict, step=global_step)

        # ── Save best model ──────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / "best_model"
            controlnet.save_pretrained(best_path)
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "controlnet_state_dict": controlnet.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_avg_loss,
                "val_loss": val_loss,
            }, output_dir / "best_checkpoint.pt")
            logger.info(
                f"New best val loss: {val_loss:.6f} → saved to {best_path}"
            )
            if use_wandb:
                wandb.run.summary["best_val_loss"] = val_loss
                wandb.run.summary["best_epoch"] = epoch + 1

        # ── Periodic checkpoint ──────────────────────────────────────
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "controlnet_state_dict": controlnet.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_avg_loss,
                "val_loss": val_loss,
            }, ckpt_path)
            logger.info(f"Saved checkpoint → {ckpt_path}")

            diffusers_dir = output_dir / f"controlnet_epoch_{epoch+1}"
            controlnet.save_pretrained(diffusers_dir)
            logger.info(f"Saved diffusers format → {diffusers_dir}")

            # ── Visualize checkpoint on W&B ─────────────────────────
            if use_wandb and viz_samples:
                try:
                    visualize_checkpoint(
                        controlnet=controlnet,
                        unet=unet,
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        noise_scheduler=noise_scheduler,
                        viz_samples=viz_samples,
                        num_steps=args.viz_steps,
                        epoch=epoch,
                        device=device,
                        logger=logger,
                    )
                except Exception as e:
                    logger.warning(f"Visualization failed (non-fatal): {e}")

    # ── Finish ───────────────────────────────────────────────────────
    total_time = time.time() - t_train_start
    logger.info("=" * 70)
    logger.info(f"TRAINING COMPLETE — {timedelta(seconds=int(total_time))}")
    logger.info(f"Best val loss: {best_val_loss:.6f}")
    logger.info("=" * 70)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
