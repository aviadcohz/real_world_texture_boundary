#!/usr/bin/env python3
"""
Evaluation pipeline for ControlNet texture boundary conditioning.

Implements the "Condition Reconstruction" protocol from the ControlNet paper:
    1. Generate images from RWTD test masks using trained ControlNet
    2. Extract edges from generated images (HED / Canny)
    3. Compare extracted edges vs. GT masks → ODS, OIS, AP
    4. Optionally compute FID (generated vs. real image quality)

RWTD prompts:
    Expects a JSON file mapping each RWTD image to a text prompt.
    Generate this with a VLM captioner (e.g., QwenVL) beforehand:
        [{"id": "1", "text": "shell on sandy ocean floor"}, ...]
    If no prompt file exists, a default prompt is used.

Usage:
    python evaluate.py --controlnet-path checkpoints/best_model
    python evaluate.py --controlnet-path checkpoints/best_model --edge-method hed
    python evaluate.py --controlnet-path checkpoints/best_model --no-wandb
"""

import argparse
import json
import logging
import time
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_RWTD = "/home/aviad/RWTD"
DEFAULT_PRETRAINED = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_OUTPUT = (
    "/home/aviad/real_world_texture_boundary"
    "/synthetic_based_real_world/controlnet_training/eval_results"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ControlNet condition fidelity on RWTD test set."
    )
    # Model
    parser.add_argument(
        "--controlnet-path", type=str, required=True,
        help="Path to trained ControlNet (diffusers format directory)",
    )
    parser.add_argument(
        "--pretrained-model", type=str, default=DEFAULT_PRETRAINED,
        help="HuggingFace SD 1.5 model ID",
    )

    # Data
    parser.add_argument(
        "--rwtd-path", type=str, default=DEFAULT_RWTD,
        help="Path to RWTD test dataset (images/ + masks/)",
    )
    parser.add_argument(
        "--prompts-json", type=str, default=None,
        help="JSON file with RWTD prompts [{id, text}, ...]. "
             "If not provided, uses default prompt.",
    )
    parser.add_argument(
        "--default-prompt", type=str,
        default="texture boundary between two natural surfaces",
        help="Fallback prompt when no prompts JSON is provided",
    )

    # Generation
    parser.add_argument(
        "--num-inference-steps", type=int, default=30,
        help="Diffusion sampling steps (default: 30)",
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=7.5,
        help="Classifier-free guidance scale (default: 7.5)",
    )
    parser.add_argument(
        "--controlnet-scale", type=float, default=1.0,
        help="ControlNet conditioning scale (default: 1.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible generation",
    )

    # Edge extraction
    parser.add_argument(
        "--edge-method", type=str, default="canny",
        choices=["canny", "hed"],
        help="Edge detection method for reconstructed edges (default: canny)",
    )
    parser.add_argument(
        "--canny-low", type=int, default=50,
        help="Canny low threshold (default: 50)",
    )
    parser.add_argument(
        "--canny-high", type=int, default=150,
        help="Canny high threshold (default: 150)",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT,
        help="Directory for evaluation results and generated images",
    )
    parser.add_argument(
        "--save-images", action="store_true",
        help="Save generated images and extracted edges for inspection",
    )

    # FID
    parser.add_argument(
        "--compute-fid", action="store_true",
        help="Compute FID between generated and real RWTD images",
    )

    # Subset
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Evaluate only the first N images (for quick signal). Default: all.",
    )

    # W&B
    parser.add_argument(
        "--wandb-project", type=str, default="controlnet-texture-boundary",
    )
    parser.add_argument("--no-wandb", action="store_true")

    return parser.parse_args()


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(output_dir: Path):
    logger = logging.getLogger("controlnet_eval")
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(ch)

    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "eval.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(fh)

    return logger


# ── RWTD loader ──────────────────────────────────────────────────────────────

def load_rwtd(rwtd_path: str, prompts_json: str = None, default_prompt: str = ""):
    """
    Load RWTD test set: images, masks, and prompts.

    Supports two prompt formats:
        1. training_pairs.json (from prepare_rwtd_pairs.py):
           [{"image": path, "conditioning_image": mask_path, "text": desc}, ...]
        2. Separate prompts JSON:
           [{"id": "1", "text": "shell on sand"}, ...]
        3. No prompts → uses default_prompt for all images.

    Returns list of dicts: [{id, image_path, mask_path, text}, ...]
    """
    rwtd = Path(rwtd_path)

    # Try training_pairs.json first (output of prepare_rwtd_pairs.py)
    training_pairs_path = rwtd / "training_pairs.json"
    if training_pairs_path.exists():
        with open(training_pairs_path) as f:
            pairs = json.load(f)
        entries = []
        for p in pairs:
            entries.append({
                "id": Path(p["image"]).stem,
                "image_path": p["image"],
                "mask_path": p["conditioning_image"],
                "text": p["text"],
            })
        return entries

    # Fallback: scan images/ + masks/ directories
    images_dir = rwtd / "images"
    masks_dir = rwtd / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"RWTD must have images/ and masks/ at {rwtd}")

    image_files = sorted(images_dir.glob("*.jpg"))
    ids = [f.stem for f in image_files]

    # Load prompts if available
    prompts = {}
    if prompts_json and Path(prompts_json).exists():
        with open(prompts_json) as f:
            prompt_list = json.load(f)
        prompts = {str(p["id"]): p["text"] for p in prompt_list}

    entries = []
    for img_id in ids:
        mask_path = masks_dir / f"{img_id}.jpg"
        if not mask_path.exists():
            continue
        entries.append({
            "id": img_id,
            "image_path": str(images_dir / f"{img_id}.jpg"),
            "mask_path": str(mask_path),
            "text": prompts.get(img_id, default_prompt),
        })

    return entries


# ── Edge extraction ──────────────────────────────────────────────────────────

def extract_edges_canny(image: np.ndarray, low: int = 50, high: int = 150):
    """
    Canny edge detection → soft edge map [0, 1].

    Returns binary edges as float to be used as soft predictions
    (Canny is inherently binary, but we return float for API consistency).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    edges = cv2.Canny(gray, low, high)
    return edges.astype(np.float32) / 255.0


def extract_edges_hed(image: np.ndarray, hed_detector=None):
    """
    HED (Holistically-Nested Edge Detection) → soft edge map [0, 1].

    Returns a probability map (continuous values).
    """
    from controlnet_aux import HEDdetector

    if hed_detector is None:
        hed_detector = HEDdetector.from_pretrained("lllyasviel/Annotators")

    pil_image = Image.fromarray(image)
    hed_result = hed_detector(pil_image)
    # HED returns a PIL image — convert to float array
    soft_edges = np.array(hed_result).astype(np.float32) / 255.0
    if len(soft_edges.shape) == 3:
        soft_edges = soft_edges.mean(axis=2)
    return soft_edges


# ── Generation pipeline ─────────────────────────────────────────────────────

def build_pipeline(controlnet_path: str, pretrained_model: str, device: str = "cuda"):
    """Build the StableDiffusionControlNetPipeline for generation."""
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    ).to(device)

    # Faster scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    return pipe


def generate_image(
    pipe,
    mask_path: str,
    prompt: str,
    num_steps: int = 30,
    guidance_scale: float = 7.5,
    controlnet_scale: float = 1.0,
    seed: int = 42,
):
    """
    Generate a single image using the ControlNet pipeline.

    Args:
        pipe:             StableDiffusionControlNetPipeline
        mask_path:        Path to boundary mask image
        prompt:           Text prompt
        num_steps:        Number of diffusion sampling steps
        guidance_scale:   CFG scale
        controlnet_scale: ControlNet conditioning strength
        seed:             Random seed

    Returns:
        generated_image: (H, W, 3) uint8 RGB numpy array
    """
    # Load mask as 1-channel, resize to 512×512
    mask = Image.open(mask_path).convert("L").resize((512, 512), Image.NEAREST)
    # Convert to 1-channel tensor [0,1] shape (1,1,H,W) — ControlNet has conditioning_channels=1
    mask_np = np.array(mask).astype(np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(pipe.device)

    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        image=mask_tensor,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_scale,
        generator=generator,
    )

    generated = np.array(result.images[0])
    return generated


# ── FID computation ──────────────────────────────────────────────────────────

def compute_fid_score(real_dir: str, gen_dir: str):
    """Compute FID between real RWTD images and generated images."""
    from cleanfid import fid

    score = fid.compute_fid(real_dir, gen_dir)
    return score


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)
    device = "cuda"

    # ── W&B ──────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(
            entity="aviadcohz-tel-aviv-university",
            project=args.wandb_project,
            name=f"eval_{Path(args.controlnet_path).name}",
            config=vars(args),
            job_type="evaluation",
        )

    # ── Load RWTD test set ───────────────────────────────────────────
    logger.info(f"Loading RWTD from {args.rwtd_path}...")
    entries = load_rwtd(
        args.rwtd_path, args.prompts_json, args.default_prompt
    )
    logger.info(f"RWTD: {len(entries)} test images")

    if args.max_images is not None:
        entries = entries[:args.max_images]
        logger.info(f"Subset: using first {len(entries)} images (--max-images)")

    has_custom_prompts = args.prompts_json and Path(args.prompts_json).exists()
    logger.info(
        f"Prompts: {'custom from ' + args.prompts_json if has_custom_prompts else 'default: ' + repr(args.default_prompt)}"
    )

    # ── Build generation pipeline ────────────────────────────────────
    logger.info(f"Loading ControlNet from {args.controlnet_path}...")
    pipe = build_pipeline(args.controlnet_path, args.pretrained_model, device)
    logger.info("Pipeline ready")

    # ── HED detector (if needed) ─────────────────────────────────────
    hed_detector = None
    if args.edge_method == "hed":
        from controlnet_aux import HEDdetector
        hed_detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
        logger.info("HED detector loaded")

    # ── Generate + extract edges ─────────────────────────────────────
    logger.info("=" * 70)
    logger.info(f"GENERATION + EDGE EXTRACTION ({args.edge_method})")
    logger.info("=" * 70)

    gen_dir = output_dir / "generated"
    edge_dir = output_dir / "edges"
    if args.save_images:
        gen_dir.mkdir(parents=True, exist_ok=True)
        edge_dir.mkdir(parents=True, exist_ok=True)

    soft_preds = []
    gt_masks = []
    t_start = time.time()

    for i, entry in enumerate(entries):
        img_id = entry["id"]

        # Generate
        generated = generate_image(
            pipe,
            entry["mask_path"],
            entry["text"],
            num_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            controlnet_scale=args.controlnet_scale,
            seed=args.seed,
        )

        # Extract edges from generated image
        if args.edge_method == "canny":
            soft_edges = extract_edges_canny(
                generated, args.canny_low, args.canny_high
            )
        else:
            soft_edges = extract_edges_hed(generated, hed_detector)

        # Load GT mask and binarize
        gt_raw = cv2.imread(entry["mask_path"], cv2.IMREAD_GRAYSCALE)
        # Resize GT to match generated (512×512)
        gt_resized = cv2.resize(gt_raw, (512, 512), interpolation=cv2.INTER_NEAREST)
        gt_binary = gt_resized > 127

        # Resize soft edges to match GT if needed
        if soft_edges.shape != gt_binary.shape:
            soft_edges = cv2.resize(
                soft_edges, (gt_binary.shape[1], gt_binary.shape[0])
            )

        soft_preds.append(soft_edges)
        gt_masks.append(gt_binary)

        # Save if requested
        if args.save_images:
            Image.fromarray(generated).save(gen_dir / f"{img_id}.png")
            cv2.imwrite(
                str(edge_dir / f"{img_id}.png"),
                (soft_edges * 255).astype(np.uint8),
            )

        if (i + 1) % 10 == 0 or (i + 1) == len(entries):
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(entries) - i - 1) / rate if rate > 0 else 0
            logger.info(
                f"[{i+1:>4}/{len(entries)}] "
                f"{rate:.2f} img/s | ETA: {timedelta(seconds=int(eta))}"
            )

    # Free GPU memory
    del pipe
    if hed_detector is not None:
        del hed_detector
    torch.cuda.empty_cache()

    # ── Compute ODS / OIS / AP ───────────────────────────────────────
    logger.info("=" * 70)
    logger.info("COMPUTING METRICS")
    logger.info("=" * 70)

    from metrics import compute_ods_ois_ap, format_results

    results = compute_ods_ois_ap(soft_preds, gt_masks)
    logger.info(f"\n{format_results(results)}")

    # ── Compute FID ──────────────────────────────────────────────────
    fid_score = None
    if args.compute_fid and args.save_images:
        logger.info("Computing FID...")
        real_dir = str(Path(args.rwtd_path) / "images")
        fid_score = compute_fid_score(real_dir, str(gen_dir))
        logger.info(f"FID: {fid_score:.2f}")
        results["fid"] = fid_score

    # ── Save results ─────────────────────────────────────────────────
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {results_path}")

    # ── W&B logging ──────────────────────────────────────────────────
    if use_wandb:
        import wandb

        log_dict = {
            "eval/ods_f": results["ods_f"],
            "eval/ods_p": results["ods_p"],
            "eval/ods_r": results["ods_r"],
            "eval/ois_f": results["ois_f"],
            "eval/ois_p": results["ois_p"],
            "eval/ois_r": results["ois_r"],
            "eval/ap": results["ap"],
        }
        if fid_score is not None:
            log_dict["eval/fid"] = fid_score

        wandb.log(log_dict)
        wandb.run.summary.update(log_dict)

        # Log P-R curve
        pr_data = results["per_threshold"]
        pr_table = wandb.Table(
            columns=["threshold", "precision", "recall", "fscore"],
            data=[
                [t, p, r, f]
                for t, p, r, f in zip(
                    pr_data["thresholds"],
                    pr_data["precision"],
                    pr_data["recall"],
                    pr_data["fscore"],
                )
            ],
        )
        wandb.log({"eval/pr_curve": pr_table})

        # Log sample images if saved
        if args.save_images:
            sample_entries = entries[:8]
            images_wandb = []
            for entry in sample_entries:
                img_id = entry["id"]
                gen_path = gen_dir / f"{img_id}.png"
                edge_path = edge_dir / f"{img_id}.png"
                gt_path = entry["mask_path"]
                if gen_path.exists():
                    images_wandb.append(wandb.Image(
                        str(gen_path),
                        caption=f"ID:{img_id} | {entry['text'][:50]}",
                    ))
            if images_wandb:
                wandb.log({"eval/generated_samples": images_wandb})

        wandb.finish()

    # ── Summary ──────────────────────────────────────────────────────
    total_time = time.time() - t_start
    logger.info("=" * 70)
    logger.info(f"EVALUATION COMPLETE — {timedelta(seconds=int(total_time))}")
    logger.info("=" * 70)
    logger.info(format_results(results))
    if fid_score is not None:
        logger.info(f"FID: {fid_score:.2f}")


if __name__ == "__main__":
    main()
