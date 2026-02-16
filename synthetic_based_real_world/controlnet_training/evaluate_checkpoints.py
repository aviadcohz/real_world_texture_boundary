#!/usr/bin/env python3
"""
Evaluate ODS/OIS/AP across all checkpoints and plot convergence graph.

Runs the condition-reconstruction protocol on each saved checkpoint
against the RWTD test set, then plots ODS and OIS vs. epoch.

Just press "Run Python File" in VS Code — no CLI args needed.
All configuration is in the section below.
"""

import json
import re
import time
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

CHECKPOINT_DIR = Path(
    "/home/aviad/real_world_texture_boundary"
    "/synthetic_based_real_world/controlnet_training/checkpoints"
)
RWTD_PATH = Path("/home/aviad/RWTD")
PRETRAINED_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
OUTPUT_DIR = CHECKPOINT_DIR / "eval_across_epochs"

DEVICE = "auto"            # "auto", "cuda", or "cpu"
MAX_IMAGES = 30          # None = all 256, or e.g. 30 for a quick estimate
INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
SEED = 42
EDGE_METHOD = "canny"      # "canny" or "hed"
CANNY_LOW = 50
CANNY_HIGH = 150

EPOCHS = None              # None = all available, or e.g. [5, 25, 50] for specific ones

# W&B — set to your run ID to log results there, or None to skip
WANDB_RUN_ID = "0n0jca8u"
WANDB_PROJECT = "controlnet-texture-boundary"
WANDB_ENTITY = "aviadcohz-tel-aviv-university"

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
                found[int(m.group(1))] = d
    best = CHECKPOINT_DIR / "best_model"
    if (best / "diffusion_pytorch_model.safetensors").exists():
        found["best"] = best
    if EPOCHS is not None:
        found = {k: v for k, v in found.items() if k in EPOCHS or k == "best"}
    return found


def load_rwtd():
    """Load RWTD test images and masks."""
    # Check for training_pairs.json first
    pairs_json = RWTD_PATH / "training_pairs.json"
    if pairs_json.exists():
        with open(pairs_json) as f:
            pairs = json.load(f)
        entries = []
        for p in pairs:
            if Path(p["image"]).exists() and Path(p["conditioning_image"]).exists():
                entries.append({
                    "mask_path": p["conditioning_image"],
                    "text": p["text"],
                })
        return entries

    # Fallback: scan images/ + masks/
    images_dir = RWTD_PATH / "images"
    masks_dir = RWTD_PATH / "masks"
    entries = []
    for img_file in sorted(images_dir.glob("*.jpg")):
        mask_file = masks_dir / f"{img_file.stem}.jpg"
        if mask_file.exists():
            entries.append({
                "mask_path": str(mask_file),
                "text": "texture boundary between two natural surfaces",
            })
    return entries


def generate_and_extract_edges(pipe, entries, device):
    """Generate images from masks, extract edges, collect GT masks."""
    soft_preds = []
    gt_masks = []

    for i, entry in enumerate(entries):
        # Load mask
        mask = Image.open(entry["mask_path"]).convert("L").resize((512, 512), Image.NEAREST)
        mask_np = np.array(mask).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)

        # Generate
        generator = torch.Generator(device=device).manual_seed(SEED)
        result = pipe(
            prompt=entry["text"],
            image=mask_tensor,
            num_inference_steps=INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        )
        generated = np.array(result.images[0])

        # Extract edges from generated image
        if EDGE_METHOD == "canny":
            gray = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
            soft_edges = edges.astype(np.float32) / 255.0
        else:
            from controlnet_aux import HEDdetector
            hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
            hed_result = hed(Image.fromarray(generated))
            soft_edges = np.array(hed_result).astype(np.float32) / 255.0
            if len(soft_edges.shape) == 3:
                soft_edges = soft_edges.mean(axis=2)

        # GT mask
        gt_raw = cv2.imread(entry["mask_path"], cv2.IMREAD_GRAYSCALE)
        gt_resized = cv2.resize(gt_raw, (512, 512), interpolation=cv2.INTER_NEAREST)
        gt_binary = gt_resized > 127

        if soft_edges.shape != gt_binary.shape:
            soft_edges = cv2.resize(soft_edges, (gt_binary.shape[1], gt_binary.shape[0]))

        soft_preds.append(soft_edges)
        gt_masks.append(gt_binary)

        if (i + 1) % 10 == 0 or (i + 1) == len(entries):
            print(f"    [{i+1:>3}/{len(entries)}] generated + extracted edges")

    return soft_preds, gt_masks


def plot_convergence(all_results, output_path):
    """Plot ODS and OIS vs. epoch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Separate numeric epochs from "best"
    numeric = {k: v for k, v in all_results.items() if isinstance(k, int)}
    epochs_sorted = sorted(numeric.keys())
    ods_vals = [numeric[e]["ods_f"] for e in epochs_sorted]
    ois_vals = [numeric[e]["ois_f"] for e in epochs_sorted]
    ap_vals = [numeric[e]["ap"] for e in epochs_sorted]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(epochs_sorted, ods_vals, "o-", color="#2196F3", linewidth=2, markersize=6, label="ODS (F-score)")
    ax.plot(epochs_sorted, ois_vals, "s-", color="#FF9800", linewidth=2, markersize=6, label="OIS (F-score)")
    ax.plot(epochs_sorted, ap_vals, "^-", color="#4CAF50", linewidth=2, markersize=6, label="AP")

    # Mark best checkpoint if available
    if "best" in all_results:
        best = all_results["best"]
        ax.axhline(y=best["ods_f"], color="#2196F3", linestyle="--", alpha=0.5, label=f"Best ckpt ODS={best['ods_f']:.4f}")
        ax.axhline(y=best["ois_f"], color="#FF9800", linestyle="--", alpha=0.5, label=f"Best ckpt OIS={best['ois_f']:.4f}")

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("ControlNet Condition Reconstruction: ODS / OIS / AP vs. Epoch", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Annotate max values
    if ods_vals:
        best_ods_idx = np.argmax(ods_vals)
        ax.annotate(
            f"{ods_vals[best_ods_idx]:.4f}",
            xy=(epochs_sorted[best_ods_idx], ods_vals[best_ods_idx]),
            textcoords="offset points", xytext=(0, 12), fontsize=9, color="#2196F3",
            ha="center", fontweight="bold",
        )
        best_ois_idx = np.argmax(ois_vals)
        ax.annotate(
            f"{ois_vals[best_ois_idx]:.4f}",
            xy=(epochs_sorted[best_ois_idx], ois_vals[best_ois_idx]),
            textcoords="offset points", xytext=(0, 12), fontsize=9, color="#FF9800",
            ha="center", fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {output_path}")
    return fig


def main():
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device} ({dtype})")

    # Load RWTD
    entries = load_rwtd()
    if MAX_IMAGES is not None:
        entries = entries[:MAX_IMAGES]
    print(f"RWTD: {len(entries)} images (edge method: {EDGE_METHOD})")

    # Discover checkpoints
    checkpoints = discover_checkpoints()
    if not checkpoints:
        print("No checkpoints found!")
        return
    print(f"Checkpoints: {sorted(k for k in checkpoints if isinstance(k, int))} + {'best' if 'best' in checkpoints else 'no best'}")

    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
    from metrics import compute_ods_ois_ap

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # Check for cached results to avoid recomputing
    cache_path = OUTPUT_DIR / "results_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        # Convert string keys back to int where possible
        for k, v in cached.items():
            try:
                all_results[int(k)] = v
            except ValueError:
                all_results[k] = v
        print(f"Loaded {len(all_results)} cached results")

    for ep_key, ckpt_path in sorted(checkpoints.items(), key=lambda x: (isinstance(x[0], str), x[0])):
        if ep_key in all_results:
            label = f"epoch_{ep_key}" if isinstance(ep_key, int) else ep_key
            print(f"\n{label}: using cached results (ODS={all_results[ep_key]['ods_f']:.4f})")
            continue

        label = f"epoch_{ep_key}" if isinstance(ep_key, int) else ep_key
        print(f"\n{'='*50}")
        print(f"Evaluating: {label}")
        print(f"{'='*50}")
        t0 = time.time()

        try:
            # Load checkpoint
            controlnet = ControlNetModel.from_pretrained(
                str(ckpt_path), torch_dtype=dtype,
            ).to(device)

            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                PRETRAINED_MODEL,
                controlnet=controlnet,
                torch_dtype=dtype,
                safety_checker=None,
            ).to(device)
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.set_progress_bar_config(disable=True)

            # Generate + extract edges
            soft_preds, gt_masks = generate_and_extract_edges(pipe, entries, device)

            # Compute metrics
            results = compute_ods_ois_ap(soft_preds, gt_masks)
            elapsed = time.time() - t0

            all_results[ep_key] = {
                "ods_f": results["ods_f"],
                "ods_p": results["ods_p"],
                "ods_r": results["ods_r"],
                "ois_f": results["ois_f"],
                "ois_p": results["ois_p"],
                "ois_r": results["ois_r"],
                "ap": results["ap"],
                "ods_threshold": results["ods_threshold"],
            }

            print(f"  ODS: F={results['ods_f']:.4f}  P={results['ods_p']:.4f}  R={results['ods_r']:.4f}")
            print(f"  OIS: F={results['ois_f']:.4f}  P={results['ois_p']:.4f}  R={results['ois_r']:.4f}")
            print(f"  AP:  {results['ap']:.4f}")
            print(f"  Time: {timedelta(seconds=int(elapsed))}")

            # Save cache after each checkpoint (in case of interruption)
            cache_data = {str(k): v for k, v in all_results.items()}
            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)

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

    # ── Print summary table ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY: ODS / OIS / AP per checkpoint")
    print(f"{'='*60}")
    print(f"{'Epoch':<10} {'ODS-F':>8} {'OIS-F':>8} {'AP':>8}")
    print(f"{'-'*10} {'-'*8} {'-'*8} {'-'*8}")

    for k in sorted((k for k in all_results if isinstance(k, int))):
        r = all_results[k]
        print(f"{'ep_'+str(k):<10} {r['ods_f']:>8.4f} {r['ois_f']:>8.4f} {r['ap']:>8.4f}")
    if "best" in all_results:
        r = all_results["best"]
        print(f"{'best':<10} {r['ods_f']:>8.4f} {r['ois_f']:>8.4f} {r['ap']:>8.4f}")

    # ── Plot convergence graph ───────────────────────────────────────
    plot_path = OUTPUT_DIR / "ods_ois_convergence.png"
    plot_convergence(all_results, plot_path)

    # ── Save full results JSON ───────────────────────────────────────
    results_path = OUTPUT_DIR / "all_results.json"
    with open(results_path, "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"Results saved: {results_path}")

    # ── Log to W&B ───────────────────────────────────────────────────
    if WANDB_RUN_ID:
        import wandb

        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            id=WANDB_RUN_ID,
            resume="must",
        )

        # Log the convergence plot
        run.log({"eval/ods_ois_convergence": wandb.Image(str(plot_path))})

        # Log per-epoch metrics
        for k in sorted((k for k in all_results if isinstance(k, int))):
            r = all_results[k]
            run.log({
                "eval/ods_f": r["ods_f"],
                "eval/ois_f": r["ois_f"],
                "eval/ap": r["ap"],
                "eval/epoch": k,
            })

        if "best" in all_results:
            run.summary["eval/best_ods_f"] = all_results["best"]["ods_f"]
            run.summary["eval/best_ois_f"] = all_results["best"]["ois_f"]
            run.summary["eval/best_ap"] = all_results["best"]["ap"]

        run.finish()
        print(f"Logged to W&B: {run.url}")

    print("\nDone!")


if __name__ == "__main__":
    main()
