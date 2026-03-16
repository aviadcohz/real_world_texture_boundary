#!/usr/bin/env python3
"""
V2 Comparison: max 4x upscale, max 512 output, SUPIR fidelity sweep.

Constraints from user:
1. Max 4x upscale from original crop
2. Max output size 512x512
3. SUPIR: sweep CFG/fidelity scale to reduce hallucination
"""

import sys
import os
sys.path.insert(0, "/home/aviad/real_world_texture_boundary")
sys.path.insert(0, "/home/aviad/SUPIR")

import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path
import torch

CROPS_DIR = Path("/datasets/ade20k/Detecture_dataset/crops")
OUTPUT_DIR = Path("/home/aviad/real_world_texture_boundary/texture_boundary_Architexture/sr_comparison/v2_constrained")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SCALE = 4
MAX_OUTPUT = 512

TEST_CROPS = [
    {"name": "training_ADE_train_00005346_t2_crop0",  # 33x33
     "texture_a": "dry yellowish grass with sparse blades",
     "texture_b": "light beige sandy soil with scattered pebbles"},
    {"name": "training_ADE_train_00005842_t0_crop4",  # 44x44
     "texture_a": "dense green coniferous forest with vertical trunks",
     "texture_b": "rough weathered rock with jagged edges"},
    {"name": "training_ADE_train_00003145_t1_crop0",  # 72x54
     "texture_a": "smooth green ocean water with gentle waves",
     "texture_b": "dark wet sand with fine granular texture"},
    {"name": "training_ADE_train_00009295_t0_crop1",  # 63x84
     "texture_a": "dense green grass with scattered shadows",
     "texture_b": "smooth wet sandy ground with ripples"},
    {"name": "training_ADE_train_00009493_t2_crop4",  # 104x78
     "texture_a": "wet gray gravel with scattered stones",
     "texture_b": "rough weathered stone with visible cracks"},
]


def smooth_mask(mask, epsilon_frac=0.005, gaussian_ksize=5):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask
    smoothed = np.zeros_like(mask)
    for i, cnt in enumerate(contours):
        epsilon = epsilon_frac * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        is_hole = hierarchy[0][i][3] >= 0 if hierarchy is not None else False
        cv2.drawContours(smoothed, [approx], 0, 0 if is_hole else 255, -1)
    if gaussian_ksize > 0:
        blurred = cv2.GaussianBlur(smoothed.astype(np.float32), (gaussian_ksize, gaussian_ksize), 0)
        smoothed = (blurred > 127).astype(np.uint8) * 255
    return smoothed


def create_overlay(image, mask_a, mask_b, alpha=0.35):
    overlay = image.astype(np.float32).copy()
    ma = (mask_a > 127).astype(np.float32)[:, :, None]
    mb = (mask_b > 127).astype(np.float32)[:, :, None]
    r_boost = np.zeros_like(overlay); r_boost[:,:,0] = 80; r_boost[:,:,1] = -30; r_boost[:,:,2] = -30
    b_boost = np.zeros_like(overlay); b_boost[:,:,0] = -30; b_boost[:,:,1] = -30; b_boost[:,:,2] = 80
    overlay = overlay + r_boost * ma + b_boost * mb
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)


def draw_boundary(image, mask_a, mask_b):
    kernel = np.ones((3, 3), np.uint8)
    boundary = cv2.bitwise_and(cv2.dilate(mask_a, kernel, 1), cv2.dilate(mask_b, kernel, 1))
    result = image.copy()
    result[boundary > 0] = [0, 255, 0]
    return result


def add_label(image, text, font_size=14):
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([0, 0, tw + 10, th + 6], fill=(0, 0, 0, 200))
    draw.text((5, 1), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def compute_constrained_scale(h, w, max_scale=MAX_SCALE, max_output=MAX_OUTPUT):
    """Compute scale: min(max_scale, max_output/min_dim)."""
    min_dim = min(h, w)
    needed = max_output / min_dim
    scale = min(max_scale, needed)
    return scale


class SUPIRUpscaler:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = None

    def load(self):
        if self.model is not None:
            return
        import open_clip.transformer as oct
        _orig_attn = oct.ResidualAttentionBlock.attention
        def _patched_attn(self_block, q_x, k_x, v_x, attn_mask=None):
            return _orig_attn(self_block, q_x, k_x, v_x, attn_mask=None)
        oct.ResidualAttentionBlock.attention = _patched_attn

        from SUPIR.util import create_SUPIR_model, convert_dtype
        print("[SUPIR] Loading model...")
        self.model = create_SUPIR_model('/home/aviad/SUPIR/options/SUPIR_v0_local.yaml', SUPIR_sign='Q')
        self.model = self.model.half()
        self.model.init_tile_vae(encoder_tile_size=512, decoder_tile_size=64)
        self.model.ae_dtype = convert_dtype('bf16')
        self.model.model.dtype = convert_dtype('fp16')
        self.model = self.model.to(self.device)
        print("[SUPIR] Model loaded.")

    def upscale(self, image_pil, caption="", upscale=1, min_size=256,
                edm_steps=30, s_cfg=7.5, s_stage2=1.0, seed=42):
        self.load()
        from SUPIR.util import PIL2Tensor, Tensor2PIL

        LQ_img, h0, w0 = PIL2Tensor(image_pil, upsacle=upscale, min_size=min_size)
        LQ_img = LQ_img.unsqueeze(0).to(self.device)[:, :3, :, :]
        captions = [caption]

        # a_prompt: texture-focused positive prompt
        a_prompt = ("high quality, extremely detailed, sharp natural texture, "
                    "photorealistic, 8k resolution, precise material detail")
        n_prompt = ("blurry, low quality, painting, cartoon, over-smooth, "
                    "artifacts, deformed, watermark")

        samples = self.model.batchify_sample(
            LQ_img, captions, num_steps=edm_steps,
            restoration_scale=-1, s_churn=5, s_noise=1.003,
            cfg_scale=s_cfg, control_scale=s_stage2, seed=seed,
            num_samples=1, p_p=a_prompt, n_p=n_prompt,
            color_fix_type='Wavelet', use_linear_CFG=True,
            use_linear_control_scale=False,
            cfg_scale_start=1.0, control_scale_start=0.0,
        )
        return Tensor2PIL(samples[0], h0, w0)


def main():
    from texture_boundary_Architexture.core.texture_refiner_pipeline import (
        TextureRefinerPipeline, SRBackend, StandardizeMode,
    )

    print("Loading test crops...")
    test_data = []
    for c in TEST_CROPS:
        name = c["name"]
        img = Image.open(CROPS_DIR / "images" / f"{name}.jpg").convert("RGB")
        ma = np.array(Image.open(CROPS_DIR / "masks_texture" / f"{name}_mask_a.png").convert("L"))
        mb = np.array(Image.open(CROPS_DIR / "masks_texture" / f"{name}_mask_b.png").convert("L"))
        w, h = img.size
        scale = compute_constrained_scale(h, w)
        out_h, out_w = min(int(h * scale), MAX_OUTPUT), min(int(w * scale), MAX_OUTPUT)
        print(f"  {name}: {w}x{h} -> {out_w}x{out_h} (scale={scale:.1f}x)")
        test_data.append({**c, "image": img, "mask_a": ma, "mask_b": mb,
                          "scale": scale, "out_h": out_h, "out_w": out_w})

    # ===== ESRGAN x2plus (constrained) =====
    print("\n=== Real-ESRGAN x2plus (max 4x) ===")
    esrgan_results = []
    for d in test_data:
        w, h = d["image"].size
        # Use pipeline with constrained min_resolution
        target = min(min(w, h) * MAX_SCALE, MAX_OUTPUT)
        pipe = TextureRefinerPipeline(
            min_resolution=target, model_name="RealESRGAN_x2plus",
            standardize=StandardizeMode.NONE, device="cuda",
        )
        r = pipe.process_crop(d["image"], d["mask_a"], d["mask_b"])
        # Crop to max output
        img_r = r["image"]
        if img_r.size[0] > MAX_OUTPUT or img_r.size[1] > MAX_OUTPUT:
            # Center crop
            iw, ih = img_r.size
            left = max(0, (iw - MAX_OUTPUT) // 2)
            top = max(0, (ih - MAX_OUTPUT) // 2)
            img_r = img_r.crop((left, top, left + min(iw, MAX_OUTPUT), top + min(ih, MAX_OUTPUT)))
            ma_r = r["mask_a"][top:top+min(ih,MAX_OUTPUT), left:left+min(iw,MAX_OUTPUT)]
            mb_r = r["mask_b"][top:top+min(ih,MAX_OUTPUT), left:left+min(iw,MAX_OUTPUT)]
        else:
            ma_r, mb_r = r["mask_a"], r["mask_b"]
        esrgan_results.append({"image": img_r, "mask_a": ma_r, "mask_b": mb_r,
                                "scale": r["scale_factor"]})
        print(f"  {d['name']}: scale={r['scale_factor']}x -> {img_r.size}")
        del pipe
    torch.cuda.empty_cache()

    # ===== SUPIR with fidelity sweep =====
    print("\n=== SUPIR fidelity sweep ===")
    supir = SUPIRUpscaler(device="cuda")

    # Sweep: s_cfg controls fidelity (higher = closer to original)
    # s_stage2 controls how much the control (input image) influences output
    fidelity_configs = [
        {"label": "SUPIR low-fid\n(cfg=4, ctrl=0.8)", "s_cfg": 4.0, "s_stage2": 0.8},
        {"label": "SUPIR mid-fid\n(cfg=7.5, ctrl=1.0)", "s_cfg": 7.5, "s_stage2": 1.0},
        {"label": "SUPIR high-fid\n(cfg=12, ctrl=1.2)", "s_cfg": 12.0, "s_stage2": 1.2},
    ]

    supir_results = {cfg["label"]: [] for cfg in fidelity_configs}

    for d in test_data:
        w, h = d["image"].size
        upscale = min(MAX_SCALE, MAX_OUTPUT // min(w, h))
        upscale = max(1, upscale)
        min_size = min(min(w, h) * upscale, MAX_OUTPUT)

        caption = (f"a close-up photograph showing the boundary between "
                   f"{d['texture_a']} and {d['texture_b']}, "
                   f"sharp texture detail, natural materials")

        for cfg in fidelity_configs:
            print(f"  {d['name']}: {cfg['label'].split(chr(10))[0]}...", end=" ", flush=True)
            result = supir.upscale(
                d["image"], caption=caption,
                upscale=upscale, min_size=min_size,
                edm_steps=30, s_cfg=cfg["s_cfg"], s_stage2=cfg["s_stage2"],
            )
            # Crop to max
            if result.size[0] > MAX_OUTPUT or result.size[1] > MAX_OUTPUT:
                iw, ih = result.size
                left = max(0, (iw - MAX_OUTPUT) // 2)
                top = max(0, (ih - MAX_OUTPUT) // 2)
                result = result.crop((left, top, left + min(iw, MAX_OUTPUT), top + min(ih, MAX_OUTPUT)))
            supir_results[cfg["label"]].append(result)
            print(f"-> {result.size}")

    del supir
    torch.cuda.empty_cache()

    # ===== Build visualization =====
    print("\n=== Building visualization ===")

    display_size = 512  # display cell size
    pad = 3
    n = len(test_data)
    # Columns: Original | ESRGAN | SUPIR low | SUPIR mid | SUPIR high
    col_labels = ["Original (bicubic)", "ESRGAN x2plus", *[c["label"].split('\n')[0] for c in fidelity_configs]]
    n_cols = 2 + len(fidelity_configs)

    # Two rows per crop: image + mask overlay
    grid_w = n_cols * (display_size + pad) + pad
    grid_h = n * 2 * (display_size + pad) + pad
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 30

    for ci, d in enumerate(test_data):
        w_orig, h_orig = d["image"].size
        y_top = ci * 2 * (display_size + pad) + pad
        y_bot = y_top + display_size + pad

        # Prepare all images at display_size for comparison
        columns = []

        # Col 0: original bicubic
        orig = np.array(d["image"].resize((display_size, display_size), Image.BICUBIC))
        ma_d = cv2.resize(d["mask_a"], (display_size, display_size), interpolation=cv2.INTER_NEAREST)
        mb_d = cv2.resize(d["mask_b"], (display_size, display_size), interpolation=cv2.INTER_NEAREST)
        columns.append((orig, ma_d, mb_d, f"Bicubic {w_orig}x{h_orig}"))

        # Col 1: ESRGAN
        er = esrgan_results[ci]
        eimg = np.array(er["image"].resize((display_size, display_size), Image.LANCZOS))
        ema = cv2.resize(er["mask_a"], (display_size, display_size), interpolation=cv2.INTER_NEAREST)
        emb = cv2.resize(er["mask_b"], (display_size, display_size), interpolation=cv2.INTER_NEAREST)
        columns.append((eimg, ema, emb, f"ESRGAN ({er['scale']}x)"))

        # SUPIR columns
        for cfg in fidelity_configs:
            sr = supir_results[cfg["label"]][ci]
            simg = np.array(sr.resize((display_size, display_size), Image.LANCZOS))
            sma = cv2.resize(d["mask_a"], (display_size, display_size), interpolation=cv2.INTER_NEAREST)
            smb = cv2.resize(d["mask_b"], (display_size, display_size), interpolation=cv2.INTER_NEAREST)
            short_label = cfg["label"].replace('\n', ' ')
            columns.append((simg, sma, smb, short_label))

        for col_idx, (rgb, ma, mb, label) in enumerate(columns):
            x = col_idx * (display_size + pad) + pad

            # Top: SR image
            grid[y_top:y_top+display_size, x:x+display_size] = add_label(rgb, label)

            # Bottom: smoothed mask overlay + boundary
            ma_s = smooth_mask(ma)
            mb_s = smooth_mask(mb)
            ov = create_overlay(rgb, ma_s, mb_s, alpha=0.4)
            ov = draw_boundary(ov, ma_s, mb_s)
            if col_idx == 0:
                # Show texture descriptions
                ov = add_label(ov, f"A: {d['texture_a'][:40]}")
                img_pil = Image.fromarray(ov)
                draw = ImageDraw.Draw(img_pil)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                except: font = ImageFont.load_default()
                draw.rectangle([0, 20, display_size, 38], fill=(0,0,0,200))
                draw.text((5, 21), f"B: {d['texture_b'][:40]}", fill=(255,255,255), font=font)
                ov = np.array(img_pil)
            else:
                ov = add_label(ov, "Smoothed masks + boundary")
            grid[y_bot:y_bot+display_size, x:x+display_size] = ov

    grid_path = OUTPUT_DIR / "comparison_constrained.jpg"
    Image.fromarray(grid).save(str(grid_path), quality=95)
    print(f"Saved: {grid_path}")

    # Per-crop detail strips
    for ci, d in enumerate(test_data):
        w_orig, h_orig = d["image"].size
        strip_w = n_cols * (display_size + pad) + pad
        strip_h = 2 * (display_size + pad) + pad
        strip = np.ones((strip_h, strip_w, 3), dtype=np.uint8) * 30

        # Same columns as above
        imgs = [
            np.array(d["image"].resize((display_size, display_size), Image.BICUBIC)),
            np.array(esrgan_results[ci]["image"].resize((display_size, display_size), Image.LANCZOS)),
        ]
        labels_top = [
            f"Bicubic {w_orig}x{h_orig}",
            f"ESRGAN ({esrgan_results[ci]['scale']}x)",
        ]
        for cfg in fidelity_configs:
            sr = supir_results[cfg["label"]][ci]
            imgs.append(np.array(sr.resize((display_size, display_size), Image.LANCZOS)))
            labels_top.append(cfg["label"].replace('\n', ' '))

        for col_idx in range(n_cols):
            x = col_idx * (display_size + pad) + pad
            rgb = imgs[col_idx]
            strip[pad:pad+display_size, x:x+display_size] = add_label(rgb, labels_top[col_idx], font_size=16)

            ma_d = cv2.resize(d["mask_a"], (display_size, display_size), interpolation=cv2.INTER_NEAREST)
            mb_d = cv2.resize(d["mask_b"], (display_size, display_size), interpolation=cv2.INTER_NEAREST)
            ma_s, mb_s = smooth_mask(ma_d), smooth_mask(mb_d)
            ov = draw_boundary(create_overlay(rgb, ma_s, mb_s, 0.4), ma_s, mb_s)
            ov = add_label(ov, "Smoothed masks", font_size=16)
            strip[pad+display_size+pad:pad+2*display_size+pad, x:x+display_size] = ov

        strip_path = OUTPUT_DIR / f"detail_{d['name']}.jpg"
        Image.fromarray(strip).save(str(strip_path), quality=95)

    print(f"\nAll outputs in: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.glob("*.jpg")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
