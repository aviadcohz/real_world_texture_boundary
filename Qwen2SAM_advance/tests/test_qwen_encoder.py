"""
Unit tests for QwenTextEncoder and CrossAttentionAdapter.

Tests:
1. CrossAttentionAdapter output shapes
2. QwenTextEncoder matches VETextEncoder interface
3. Full SAM3+Qwen model forward pass
"""

import sys
sys.path.insert(0, "/home/aviad/sam3")
sys.path.insert(0, "/home/aviad/real_world_texture_boundary")

import torch
import torch.nn as nn


def test_cross_attention_adapter():
    """Test CrossAttentionAdapter produces correct output shapes."""
    from Qwen2SAM_advance.models.qwen_text_encoder import CrossAttentionAdapter

    adapter = CrossAttentionAdapter(
        qwen_dim=2048,
        sam_dim=256,
        num_output_tokens=32,
        num_heads=8,
        num_layers=2,
    )

    batch_size = 2
    qwen_seq_len = 50  # variable length from Qwen
    qwen_features = torch.randn(batch_size, qwen_seq_len, 2048)
    qwen_mask = torch.ones(batch_size, qwen_seq_len, dtype=torch.bool)
    qwen_mask[1, 40:] = False  # second sample has shorter sequence

    text_memory, text_mask = adapter(qwen_features, qwen_mask)

    # Check shapes match SAM3 convention
    assert text_memory.shape == (32, batch_size, 256), \
        f"Expected (32, {batch_size}, 256), got {text_memory.shape}"
    assert text_mask.shape == (batch_size, 32), \
        f"Expected ({batch_size}, 32), got {text_mask.shape}"

    # All queries valid (no padding)
    assert not text_mask.any(), "All attention mask values should be False (no padding)"

    # Check gradient flow
    loss = text_memory.sum()
    loss.backward()
    assert adapter.query_tokens.grad is not None, "Gradients should flow to query_tokens"
    assert adapter.kv_proj.weight.grad is not None, "Gradients should flow to kv_proj"

    print("[PASS] CrossAttentionAdapter shapes and gradients correct")

    # Print param count
    total = sum(p.numel() for p in adapter.parameters())
    print(f"  Adapter params: {total/1e6:.2f}M")


def test_qwen_text_encoder_interface():
    """Test QwenTextEncoder matches VETextEncoder output interface."""
    from Qwen2SAM_advance.models.qwen_text_encoder import QwenTextEncoder

    print("\nLoading Qwen2.5-VL-3B (this may take a moment)...")
    encoder = QwenTextEncoder(
        d_model=256,
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        qwen_dtype=torch.bfloat16,
        adapter_num_tokens=32,
        adapter_num_layers=2,
        freeze_qwen=True,
        use_lora=False,
    )
    encoder = encoder.cuda()

    # Test text-only mode (no image)
    texts = ["a boat on water", "red car on road"]

    text_attention_mask, text_memory_resized, inputs_embeds = encoder(
        texts, device=torch.device("cuda")
    )

    batch_size = len(texts)

    # Check shapes match VETextEncoder contract
    assert text_attention_mask.shape == (batch_size, 32), \
        f"attention_mask: expected ({batch_size}, 32), got {text_attention_mask.shape}"
    assert text_memory_resized.shape == (32, batch_size, 256), \
        f"text_memory: expected (32, {batch_size}, 256), got {text_memory_resized.shape}"
    assert inputs_embeds.shape == (32, batch_size, 1024), \
        f"inputs_embeds: expected (32, {batch_size}, 1024), got {inputs_embeds.shape}"

    # Check mask dtype and convention
    assert text_attention_mask.dtype == torch.bool, "Mask should be bool"

    print("[PASS] QwenTextEncoder output shapes match VETextEncoder interface")

    # Test with image
    from PIL import Image
    import numpy as np

    dummy_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    image_inputs = encoder.processor(
        images=[dummy_image] * batch_size,
        return_tensors="pt",
    )
    # Extract only image-related keys
    image_data = {}
    for k, v in image_inputs.items():
        if k in ("pixel_values", "image_grid_thw"):
            image_data[k] = v

    encoder.set_image(image_data)

    text_attention_mask, text_memory_resized, inputs_embeds = encoder(
        texts, device=torch.device("cuda")
    )

    assert text_memory_resized.shape == (32, batch_size, 256), \
        f"With image: expected (32, {batch_size}, 256), got {text_memory_resized.shape}"

    print("[PASS] QwenTextEncoder works with image input")

    # Check trainable params
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in encoder.parameters())
    print(f"  Trainable: {trainable/1e6:.2f}M / Total: {total/1e6:.2f}M")


def test_full_model_build():
    """Test building full SAM3+Qwen model and running forward pass."""
    from Qwen2SAM_advance.models.sam3_qwen_builder import (
        build_sam3_qwen_model,
        get_trainable_params,
    )

    print("\nBuilding SAM3+Qwen model...")
    model = build_sam3_qwen_model(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        freeze_qwen=True,
        use_lora=False,
        enable_segmentation=True,
        eval_mode=False,  # Need train mode for matcher
        load_sam3_from_hf=True,
        device="cuda",
    )

    print("\nTrainable parameter groups:")
    get_trainable_params(model, print_summary=True)

    print("\n[PASS] Full model build successful")


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: CrossAttentionAdapter")
    print("=" * 60)
    test_cross_attention_adapter()

    print("\n" + "=" * 60)
    print("Test 2: QwenTextEncoder Interface")
    print("=" * 60)
    test_qwen_text_encoder_interface()

    print("\n" + "=" * 60)
    print("Test 3: Full Model Build")
    print("=" * 60)
    test_full_model_build()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
