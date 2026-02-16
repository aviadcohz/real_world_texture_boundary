"""
ControlNet for 1-channel boundary mask conditioning.

Architecture:
    1. Hint encoder: 1ch boundary mask (512×512) → 320ch feature map (64×64)
    2. Trainable copy of UNet encoder blocks + middle block
    3. Zero convolutions gating each connection to the frozen UNet decoder

The frozen SD UNet + this ControlNet together predict noise ε_θ.
"""

import torch
import torch.nn as nn
from diffusers import ControlNetModel, UNet2DConditionModel


def zero_module(module: nn.Module) -> nn.Module:
    """Zero-initialize all parameters of a module."""
    for p in module.parameters():
        p.data.zero_()
    return module


def create_controlnet(
    pretrained_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    conditioning_channels: int = 1,
) -> ControlNetModel:
    """
    Create a ControlNet initialized from the pretrained SD UNet encoder.

    The encoder blocks are copied from the UNet (good init), while the
    zero convolutions and hint encoder are initialized to zero / from scratch.

    Args:
        pretrained_model:      HuggingFace model ID for SD 1.5
        conditioning_channels: Number of input channels for the hint image.
                               1 for our binary boundary mask (Option B).
    Returns:
        ControlNetModel with 1-channel hint encoder
    """
    # diffusers ControlNetModel.from_unet() handles:
    #   - Copying encoder blocks from the pretrained UNet
    #   - Creating zero convolutions (all 13 of them)
    #   - Building the hint encoder (controlnet_cond_embedding)
    #
    # We just need to tell it our conditioning has 1 channel.

    controlnet = ControlNetModel.from_unet(
        unet=UNet2DConditionModel.from_pretrained(
            pretrained_model, subfolder="unet"
        ),
        conditioning_channels=conditioning_channels,
    )

    return controlnet


def load_frozen_components(
    pretrained_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
):
    """
    Load the frozen SD components needed for training.

    Returns:
        unet:      Frozen UNet2DConditionModel
        vae:       Frozen AutoencoderKL
        tokenizer: CLIPTokenizer
        text_encoder: Frozen CLIPTextModel
        scheduler: DDPMScheduler (for training noise schedule)
    """
    from diffusers import AutoencoderKL, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer

    # ── Frozen components (no gradients) ─────────────────────────────
    vae = AutoencoderKL.from_pretrained(
        pretrained_model, subfolder="vae"
    ).to(device, dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model, subfolder="unet"
    ).to(device, dtype=dtype)
    unet.requires_grad_(False)
    unet.eval()

    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model, subfolder="text_encoder"
    ).to(device, dtype=dtype)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model, subfolder="tokenizer"
    )

    # ── Noise scheduler ──────────────────────────────────────────────
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model, subfolder="scheduler"
    )

    return {
        "unet": unet,
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "noise_scheduler": noise_scheduler,
    }
