"""
Build SAM3 with Qwen2.5-VL replacing the CLIP text encoder.

Imports SAM3 components from the sam3 package (NOT modified) and swaps
the text encoder for QwenTextEncoder with cross-attention adapter.
"""

import sys
from typing import Optional

import torch

# Ensure sam3 is importable
sys.path.insert(0, "/home/aviad/sam3")

from sam3.model_builder import (
    _create_dot_product_scoring,
    _create_geometry_encoder,
    _create_sam3_model,
    _create_sam3_transformer,
    _create_segmentation_head,
    _create_vision_backbone,
    download_ckpt_from_hf,
)
from sam3.model.vl_combiner import SAM3VLBackbone

from .qwen_text_encoder import QwenTextEncoder
from .checkpoint_utils import load_sam3_checkpoint_skip_clip


class SAM3VLBackboneQwen(SAM3VLBackbone):
    """
    Extended VL backbone that passes images to Qwen before text encoding.

    Overrides forward() to call language_backbone.set_image() with the
    input samples before forward_text(), so Qwen sees both image and text.
    """

    def forward(self, samples, captions, input_boxes=None, additional_text=None):
        # First, process image through Qwen's processor and cache
        # (set_image is called externally by the training loop with
        #  Qwen-processed image data)
        # Then proceed with standard VLBackbone forward
        output = self.forward_image(samples)
        device = output["vision_features"].device
        output.update(self.forward_text(captions, input_boxes, additional_text, device))
        return output


def build_sam3_qwen_model(
    # Qwen config
    qwen_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    qwen_dtype: str = "bfloat16",
    freeze_qwen: bool = True,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    # Adapter config
    adapter_num_tokens: int = 32,
    adapter_num_layers: int = 2,
    adapter_num_heads: int = 8,
    adapter_dropout: float = 0.1,
    # SAM3 config
    enable_segmentation: bool = True,
    eval_mode: bool = True,
    sam3_checkpoint_path: Optional[str] = None,
    load_sam3_from_hf: bool = True,
    device: str = "cuda",
):
    """
    Build SAM3 image model with Qwen2.5-VL replacing CLIP text encoder.

    Args:
        qwen_model_name: HuggingFace model ID for Qwen2.5-VL
        qwen_dtype: Data type for Qwen ("bfloat16" or "float16")
        freeze_qwen: Whether to freeze Qwen base weights
        use_lora: Whether to apply LoRA to Qwen
        adapter_num_tokens: Number of output tokens from cross-attention adapter
        adapter_num_layers: Number of cross-attention layers in adapter
        enable_segmentation: Whether to enable SAM3 segmentation head
        eval_mode: Whether to set model to eval mode
        sam3_checkpoint_path: Path to SAM3 pretrained checkpoint
        load_sam3_from_hf: Download SAM3 checkpoint from HuggingFace if no path given
        device: Target device

    Returns:
        Sam3Image model with Qwen text encoder
    """
    dtype = getattr(torch, qwen_dtype)

    # ---- Create Qwen text encoder ---- #
    text_encoder = QwenTextEncoder(
        d_model=256,
        qwen_model_name=qwen_model_name,
        qwen_dtype=dtype,
        adapter_num_tokens=adapter_num_tokens,
        adapter_num_layers=adapter_num_layers,
        adapter_num_heads=adapter_num_heads,
        adapter_dropout=adapter_dropout,
        freeze_qwen=freeze_qwen,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )

    # ---- Create SAM3 vision components (from sam3 package) ---- #
    vision_encoder = _create_vision_backbone(
        compile_mode=None, enable_inst_interactivity=False
    )

    # ---- Create VL backbone with Qwen ---- #
    backbone = SAM3VLBackboneQwen(visual=vision_encoder, text=text_encoder, scalp=1)

    # ---- Create transformer, scoring, segmentation ---- #
    transformer = _create_sam3_transformer()
    dot_prod_scoring = _create_dot_product_scoring()
    segmentation_head = (
        _create_segmentation_head(compile_mode=None)
        if enable_segmentation
        else None
    )
    input_geometry_encoder = _create_geometry_encoder()

    # ---- Assemble model ---- #
    model = _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
        inst_interactive_predictor=None,
        eval_mode=eval_mode,
    )

    # ---- Load SAM3 pretrained weights (skip CLIP text encoder keys) ---- #
    if sam3_checkpoint_path is None and load_sam3_from_hf:
        sam3_checkpoint_path = download_ckpt_from_hf()

    if sam3_checkpoint_path is not None:
        load_sam3_checkpoint_skip_clip(model, sam3_checkpoint_path)

    # ---- Freeze vision backbone (pretrained, not retrained) ---- #
    for name, param in model.named_parameters():
        if name.startswith("backbone.vision_backbone."):
            param.requires_grad_(False)

    # ---- Setup device ---- #
    if device == "cuda":
        model = model.cuda()
    if eval_mode:
        model.eval()

    return model


def get_trainable_params(model, print_summary: bool = True):
    """Get trainable parameters grouped by component."""
    groups = {
        "adapter": [],
        "lora": [],
        "encoder": [],
        "decoder": [],
        "scoring": [],
        "seg_head": [],
        "queries": [],
        "other": [],
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "adapter" in name:
            groups["adapter"].append((name, param))
        elif "lora" in name:
            groups["lora"].append((name, param))
        elif "encoder" in name and "transformer" in name:
            groups["encoder"].append((name, param))
        elif "decoder" in name and "transformer" in name:
            groups["decoder"].append((name, param))
        elif "dot_prod_scoring" in name:
            groups["scoring"].append((name, param))
        elif "segmentation_head" in name:
            groups["seg_head"].append((name, param))
        elif "query_embed" in name or "reference_points" in name:
            groups["queries"].append((name, param))
        else:
            groups["other"].append((name, param))

    if print_summary:
        total = 0
        for group_name, params in groups.items():
            n = sum(p.numel() for _, p in params)
            if n > 0:
                print(f"  {group_name}: {n/1e6:.2f}M params ({len(params)} tensors)")
                total += n
        print(f"  TOTAL trainable: {total/1e6:.2f}M params")

    return groups
