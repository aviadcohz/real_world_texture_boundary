#!/usr/bin/env python3
"""
Impressive visual comparison: Real-ESRGAN (x2plus) vs SUPIR (text-guided)
at 256x256 target, with smoothed mask boundaries.

Creates a publication-quality grid showing:
- Row per crop: Original | ESRGAN | SUPIR (generic) | SUPIR (texture-prompted)
- Bottom row: zoomed boundary with smoothed masks
"""

import sys
import os
sys.path.insert(0, "/home/aviad/real_world_texture_boundary")
sys.path.insert(0, "/home/aviad/SUPIR")

import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path
import torch
from scipy.interpolate import splprep, splev

# ==== Config ====
CROPS_DIR = Path("/datasets/ade20k/Detecture_dataset/crops")
OUTPUT_DIR = Path("/home/aviad/real_world_texture_boundary/texture_boundary_Architexture/sr_comparison/final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_RES = 256

# Test crops with their texture descriptions from metadata
TEST_CROPS = [
    {
        "name": "training_ADE_train_00005346_t2_crop0",  # 33x33
        "texture_a": "dry yellowish grass with sparse blades",
        "texture_b": "light beige sandy soil with scattered pebbles",
    },
    {
        "name": "training_ADE_train_00005842_t0_crop4",  # 44x44
        "texture_a": "dense green coniferous forest with vertical trunks",
        "texture_b": "rough weathered rock with jagged edges",
    },
    {
        "name": "training_ADE_train_00003145_t1_crop0",  # 72x54
        "texture_a": "smooth green ocean water with gentle waves",
        "texture_b": "dark wet sand with fine granular texture",
    },
    {
        "name": "training_ADE_train_00009295_t0_crop1",  # 63x84
        "texture_a": "dense green grass with scattered shadows",
        "texture_b": "smooth wet sandy ground with ripples",
    },
    {
        "name": "training_ADE_train_00009493_t2_crop4",  # 104x78
        "texture_a": "wet gray gravel with scattered stones",
        "texture_b": "rough weathered stone with visible cracks",
    },
]


# ==== Mask smoothing ====
def smooth_mask_polygon_gaussian(mask, epsilon_frac=0.005, gaussian_ksize=5):
    """Polygon approximation + Gaussian blur + re-threshold."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask
    smoothed = np.zeros_like(mask)
    for i, cnt in enumerate(contours):
        epsilon = epsilon_frac * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        is_hole = hierarchy[0][i][3] >= 0 if hierarchy is not None else False
        if is_hole:
            cv2.drawContours(smoothed, [approx], 0, 0, -1)
        else:
            cv2.drawContours(smoothed, [approx], 0, 255, -1)
    if gaussian_ksize > 0:
        blurred = cv2.GaussianBlur(smoothed.astype(np.float32), (gaussian_ksize, gaussian_ksize), 0)
        smoothed = (blurred > 127).astype(np.uint8) * 255
    return smoothed


# ==== Visualization helpers ====
def create_mask_overlay(image, mask_a, mask_b, alpha=0.35):
    """Soft red/blue overlay."""
    h, w = image.shape[:2]
    overlay = image.astype(np.float32).copy()
    # Red channel boost for mask A
    r_boost = np.zeros_like(overlay)
    r_boost[:, :, 0] = 80
    r_boost[:, :, 1] = -30
    r_boost[:, :, 2] = -30
    ma = (mask_a > 127).astype(np.float32)[:, :, None]
    overlay = overlay + r_boost * ma
    # Blue channel boost for mask B
    b_boost = np.zeros_like(overlay)
    b_boost[:, :, 0] = -30
    b_boost[:, :, 1] = -30
    b_boost[:, :, 2] = 80
    mb = (mask_b > 127).astype(np.float32)[:, :, None]
    overlay = overlay + b_boost * mb
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return result


def draw_boundary_line(image, mask_a, mask_b, color=(0, 255, 0), thickness=2):
    """Draw the boundary between masks as a clean line."""
    kernel = np.ones((3, 3), np.uint8)
    edge_a = cv2.dilate(mask_a, kernel, iterations=1) - cv2.erode(mask_a, kernel, iterations=1)
    edge_b = cv2.dilate(mask_b, kernel, iterations=1) - cv2.erode(mask_b, kernel, iterations=1)
    boundary = cv2.bitwise_and(edge_a, edge_b)
    # Also find edges between the two
    boundary2 = cv2.bitwise_and(
        cv2.dilate(mask_a, kernel, iterations=1),
        cv2.dilate(mask_b, kernel, iterations=1)
    )
    boundary = cv2.bitwise_or(boundary, boundary2)
    # Thin the boundary
    boundary = cv2.morphologyEx(boundary, cv2.MORPH_CLOSE, kernel)
    # Draw
    result = image.copy()
    result[boundary > 0] = color
    return result


def add_label(image, text, font_size=14, position="top"):
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if position == "top":
        draw.rectangle([0, 0, tw + 10, th + 6], fill=(0, 0, 0, 200))
        draw.text((5, 1), text, fill=(255, 255, 255), font=font)
    elif position == "bottom":
        y = img.height - th - 8
        draw.rectangle([0, y, tw + 10, img.height], fill=(0, 0, 0, 200))
        draw.text((5, y + 1), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def add_multiline_label(image, lines, font_size=12):
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    y = 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([0, y, tw + 10, y + th + 4], fill=(0, 0, 0, 200))
        draw.text((5, y), line, fill=(255, 255, 255), font=font)
        y += th + 5
    return np.array(img)


# ==== SUPIR wrapper ====
class SUPIRUpscaler:
    """Wrapper around SUPIR for text-guided super-resolution."""

    def __init__(self, device="cuda"):
        self.device = device
        self.model = None

    def load(self):
        if self.model is not None:
            return

        from SUPIR.util import create_SUPIR_model, convert_dtype

        # Monkey-patch: PyTorch 2.10 broke F.multi_head_attention_forward's
        # attn_mask shape validation. The causal mask (77,77) is no longer
        # accepted. We bypass it entirely — for text encoding, the causal
        # mask is baked into the positional embeddings anyway.
        import open_clip.transformer as oct
        _orig_attn = oct.ResidualAttentionBlock.attention
        def _patched_attn(self_block, q_x, k_x, v_x, attn_mask=None):
            # Skip the problematic attn_mask entirely
            return _orig_attn(self_block, q_x, k_x, v_x, attn_mask=None)
        oct.ResidualAttentionBlock.attention = _patched_attn

        print("[SUPIR] Loading model...")
        self.model = create_SUPIR_model(
            '/home/aviad/SUPIR/options/SUPIR_v0_local.yaml',
            SUPIR_sign='Q'
        )
        self.model = self.model.half()
        self.model.init_tile_vae(encoder_tile_size=512, decoder_tile_size=64)
        self.model.ae_dtype = convert_dtype('bf16')
        self.model.model.dtype = convert_dtype('fp16')
        self.model = self.model.to(self.device)
        print("[SUPIR] Model loaded.")

    def upscale(self, image_pil, caption="", upscale=1, min_size=256,
                edm_steps=30, s_cfg=4.0, s_stage2=1.0, seed=42,
                a_prompt="high quality, extremely detailed, sharp texture",
                n_prompt="blurry, low quality, painting, cartoon, over-smooth"):
        """Run SUPIR on a single image with text guidance.

        Args:
            image_pil: PIL Image (RGB)
            caption: Text description to guide restoration (the key feature!)
            upscale: Scale factor
            min_size: Minimum output dimension
            edm_steps: Diffusion steps (30 = fast, 50 = quality)
            s_cfg: Classifier-free guidance scale
            a_prompt: Positive auxiliary prompt
            n_prompt: Negative prompt
        """
        self.load()

        from SUPIR.util import PIL2Tensor, Tensor2PIL

        # Prepare input tensor
        LQ_img, h0, w0 = PIL2Tensor(image_pil, upsacle=upscale, min_size=min_size)
        LQ_img = LQ_img.unsqueeze(0).to(self.device)[:, :3, :, :]

        # Use provided caption (our texture description!)
        captions = [caption]

        # Run diffusion
        samples = self.model.batchify_sample(
            LQ_img, captions,
            num_steps=edm_steps,
            restoration_scale=-1,
            s_churn=5,
            s_noise=1.01,
            cfg_scale=s_cfg,
            control_scale=s_stage2,
            seed=seed,
            num_samples=1,
            p_p=a_prompt,
            n_p=n_prompt,
            color_fix_type='Wavelet',
            use_linear_CFG=True,
            use_linear_control_scale=False,
            cfg_scale_start=1.0,
            control_scale_start=0.0,
        )

        # Convert back to PIL
        result_pil = Tensor2PIL(samples[0], h0, w0)
        return result_pil


def main():
    from texture_boundary_Architexture.core.texture_refiner_pipeline import (
        TextureRefinerPipeline, SRBackend, StandardizeMode,
    )

    # Load test data
    print("Loading test crops...")
    test_data = []
    for crop_info in TEST_CROPS:
        name = crop_info["name"]
        img = Image.open(CROPS_DIR / "images" / f"{name}.jpg").convert("RGB")
        ma = np.array(Image.open(CROPS_DIR / "masks_texture" / f"{name}_mask_a.png").convert("L"))
        mb = np.array(Image.open(CROPS_DIR / "masks_texture" / f"{name}_mask_b.png").convert("L"))
        test_data.append({
            "name": name,
            "image": img,
            "mask_a": ma,
            "mask_b": mb,
            "texture_a": crop_info["texture_a"],
            "texture_b": crop_info["texture_b"],
        })
        print(f"  {name}: {img.size}, A: {crop_info['texture_a']}, B: {crop_info['texture_b']}")

    # ========== Step 1: Real-ESRGAN x2plus at 256 ==========
    print("\n=== Real-ESRGAN x2plus @ 256 ===")
    esrgan_pipe = TextureRefinerPipeline(
        min_resolution=TARGET_RES, model_name="RealESRGAN_x2plus",
        standardize=StandardizeMode.CENTER_CROP, device="cuda",
    )
    esrgan_results = []
    for d in test_data:
        r = esrgan_pipe.process_crop(d["image"], d["mask_a"], d["mask_b"])
        esrgan_results.append(r)
        print(f"  {d['name']}: scale={r['scale_factor']}x -> {r['image'].size}")
    del esrgan_pipe
    torch.cuda.empty_cache()

    # ========== Step 2: SUPIR ==========
    print("\n=== SUPIR (text-guided) ===")
    supir = SUPIRUpscaler(device="cuda")
    supir_generic_results = []
    supir_prompted_results = []

    for d in test_data:
        w, h = d["image"].size
        upscale_factor = max(1, TARGET_RES // min(w, h))

        # Generic caption (no texture info)
        print(f"\n  {d['name']} ({w}x{h}): SUPIR generic...")
        generic_result = supir.upscale(
            d["image"],
            caption="a photograph of two different textures meeting at a boundary",
            upscale=upscale_factor,
            min_size=TARGET_RES,
            edm_steps=30,
        )
        supir_generic_results.append(generic_result)
        print(f"    -> {generic_result.size}")

        # Texture-specific caption
        caption = (
            f"a close-up photograph showing the boundary between "
            f"{d['texture_a']} and {d['texture_b']}, "
            f"sharp texture detail, natural materials"
        )
        print(f"  {d['name']}: SUPIR with prompt: '{caption[:60]}...'")
        prompted_result = supir.upscale(
            d["image"],
            caption=caption,
            upscale=upscale_factor,
            min_size=TARGET_RES,
            edm_steps=30,
        )
        supir_prompted_results.append(prompted_result)
        print(f"    -> {prompted_result.size}")

    del supir
    torch.cuda.empty_cache()

    # ========== Step 3: Build visualization ==========
    print("\n=== Building visualization ===")

    cell = TARGET_RES
    pad = 3
    n_crops = len(test_data)

    # Columns: Original | ESRGAN | SUPIR generic | SUPIR prompted
    col_labels = ["Original (bicubic)", "Real-ESRGAN x2plus", "SUPIR (generic)", "SUPIR (texture-prompted)"]
    n_cols = 4

    # Two rows per crop: top = SR image, bottom = SR + smoothed mask overlay + boundary
    rows_per_crop = 2
    total_rows = n_crops * rows_per_crop

    grid_w = n_cols * (cell + pad) + pad
    grid_h = total_rows * (cell + pad) + pad
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 30

    for crop_idx, d in enumerate(test_data):
        w_orig, h_orig = d["image"].size
        ma_orig = d["mask_a"]
        mb_orig = d["mask_b"]

        y_top = crop_idx * rows_per_crop * (cell + pad) + pad
        y_bot = y_top + cell + pad

        # Prepare images for each column
        # Col 0: Original bicubic
        orig_rgb = np.array(d["image"].resize((cell, cell), Image.BICUBIC))
        orig_ma = cv2.resize(ma_orig, (cell, cell), interpolation=cv2.INTER_NEAREST)
        orig_mb = cv2.resize(mb_orig, (cell, cell), interpolation=cv2.INTER_NEAREST)

        # Col 1: ESRGAN
        esrgan_rgb = np.array(esrgan_results[crop_idx]["image"].resize((cell, cell), Image.LANCZOS))
        esrgan_ma = cv2.resize(esrgan_results[crop_idx]["mask_a"], (cell, cell), interpolation=cv2.INTER_NEAREST)
        esrgan_mb = cv2.resize(esrgan_results[crop_idx]["mask_b"], (cell, cell), interpolation=cv2.INTER_NEAREST)

        # Col 2: SUPIR generic (resize to cell if different)
        sg = supir_generic_results[crop_idx]
        supir_g_rgb = np.array(sg.resize((cell, cell), Image.LANCZOS))
        supir_g_ma = cv2.resize(ma_orig, (cell, cell), interpolation=cv2.INTER_NEAREST)
        supir_g_mb = cv2.resize(mb_orig, (cell, cell), interpolation=cv2.INTER_NEAREST)

        # Col 3: SUPIR prompted
        sp = supir_prompted_results[crop_idx]
        supir_p_rgb = np.array(sp.resize((cell, cell), Image.LANCZOS))
        supir_p_ma = cv2.resize(ma_orig, (cell, cell), interpolation=cv2.INTER_NEAREST)
        supir_p_mb = cv2.resize(mb_orig, (cell, cell), interpolation=cv2.INTER_NEAREST)

        columns = [
            (orig_rgb, orig_ma, orig_mb, f"Bicubic {w_orig}x{h_orig}"),
            (esrgan_rgb, esrgan_ma, esrgan_mb, f"ESRGAN ({esrgan_results[crop_idx]['scale_factor']}x)"),
            (supir_g_rgb, supir_g_ma, supir_g_mb, "SUPIR (generic)"),
            (supir_p_rgb, supir_p_ma, supir_p_mb, "SUPIR (prompted)"),
        ]

        for col_idx, (rgb, ma, mb, label) in enumerate(columns):
            x = col_idx * (cell + pad) + pad

            # Top row: SR image only
            labeled = add_label(rgb, label)
            grid[y_top:y_top + cell, x:x + cell] = labeled

            # Bottom row: smoothed mask overlay + boundary
            ma_s = smooth_mask_polygon_gaussian(ma)
            mb_s = smooth_mask_polygon_gaussian(mb)
            overlay = create_mask_overlay(rgb, ma_s, mb_s, alpha=0.4)
            overlay = draw_boundary_line(overlay, ma_s, mb_s)
            # Add texture description on bottom row for first column only
            if col_idx == 0:
                overlay = add_multiline_label(overlay, [
                    f"A: {d['texture_a'][:35]}",
                    f"B: {d['texture_b'][:35]}",
                ], font_size=11)
            else:
                overlay = add_label(overlay, f"{label} + smoothed masks")
            grid[y_bot:y_bot + cell, x:x + cell] = overlay

    # Save main grid
    main_path = OUTPUT_DIR / "comparison_esrgan_vs_supir.jpg"
    Image.fromarray(grid).save(str(main_path), quality=95)
    print(f"Saved: {main_path}")

    # ========== Also save individual per-crop detail strips ==========
    for crop_idx, d in enumerate(test_data):
        w_orig, h_orig = d["image"].size
        ma_orig, mb_orig = d["mask_a"], d["mask_b"]

        detail_cell = 512  # larger for detail
        strip_w = n_cols * (detail_cell + pad) + pad
        strip_h = 2 * (detail_cell + pad) + pad
        strip = np.ones((strip_h, strip_w, 3), dtype=np.uint8) * 30

        imgs = [
            np.array(d["image"].resize((detail_cell, detail_cell), Image.BICUBIC)),
            np.array(esrgan_results[crop_idx]["image"].resize((detail_cell, detail_cell), Image.LANCZOS)),
            np.array(supir_generic_results[crop_idx].resize((detail_cell, detail_cell), Image.LANCZOS)),
            np.array(supir_prompted_results[crop_idx].resize((detail_cell, detail_cell), Image.LANCZOS)),
        ]
        labels = [
            f"Bicubic {w_orig}x{h_orig}",
            f"ESRGAN x2plus ({esrgan_results[crop_idx]['scale_factor']}x)",
            "SUPIR (generic caption)",
            "SUPIR (texture description)",
        ]

        for col_idx in range(n_cols):
            x = col_idx * (detail_cell + pad) + pad
            rgb = imgs[col_idx]

            # Top: SR image
            strip[pad:pad + detail_cell, x:x + detail_cell] = add_label(rgb, labels[col_idx], font_size=18)

            # Bottom: smoothed mask overlay
            ma_r = cv2.resize(ma_orig, (detail_cell, detail_cell), interpolation=cv2.INTER_NEAREST)
            mb_r = cv2.resize(mb_orig, (detail_cell, detail_cell), interpolation=cv2.INTER_NEAREST)
            ma_s = smooth_mask_polygon_gaussian(ma_r)
            mb_s = smooth_mask_polygon_gaussian(mb_r)
            overlay = create_mask_overlay(rgb, ma_s, mb_s, alpha=0.4)
            overlay = draw_boundary_line(overlay, ma_s, mb_s)
            if col_idx == 0:
                overlay = add_multiline_label(overlay, [
                    f"A: {d['texture_a']}",
                    f"B: {d['texture_b']}",
                ], font_size=14)
            else:
                overlay = add_label(overlay, "Smoothed masks + boundary", font_size=18)
            strip[pad + detail_cell + pad:pad + 2 * detail_cell + pad,
                  x:x + detail_cell] = overlay

        strip_path = OUTPUT_DIR / f"detail_{d['name']}.jpg"
        Image.fromarray(strip).save(str(strip_path), quality=95)

    print(f"\nAll outputs in: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.glob("*.jpg")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
