"""
QwenTextEncoder: Drop-in replacement for SAM3's VETextEncoder (CLIP-based).

Replaces the CLIP text encoder with Qwen2.5-VL-3B, using a cross-attention
adapter (Q-Former style) to project from Qwen's 2048d to SAM3's 256d.

Key difference from CLIP: Qwen processes the IMAGE + text together, producing
visually-grounded embeddings that carry far richer texture/semantic information.

Interface contract (must match VETextEncoder.forward):
    forward(text, input_boxes=None, device=None)
      → (attention_mask[batch, seq], features[seq, batch, 256], embeds[seq, batch, 1024])

    attention_mask: bool, True = padding (inverted PyTorch convention)
    features: seq-first, 256d — consumed by SAM3 encoder/decoder
    embeds: seq-first, 1024d — carried as language_embeds (not used downstream)
"""

from contextlib import nullcontext
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer: LayerNorm → CrossAttn → Residual → FFN → Residual."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            queries: (batch, num_queries, d_model)
            kv: (batch, src_seq, d_model) — projected Qwen features
            key_padding_mask: (batch, src_seq) bool, True = padding
        """
        # Cross-attention with pre-norm
        q_norm = self.norm1(queries)
        attn_out, _ = self.cross_attn(
            q_norm, kv, kv, key_padding_mask=key_padding_mask
        )
        queries = queries + self.dropout(attn_out)

        # FFN with pre-norm
        ffn_out = self.ffn(self.norm2(queries))
        queries = queries + self.dropout(ffn_out)

        return queries


class CrossAttentionAdapter(nn.Module):
    """
    Q-Former style adapter: projects Qwen's 2048d features to SAM3's 256d.

    Uses N learnable query tokens that cross-attend to Qwen's output sequence,
    preserving more information than a linear projection.

    Input:  (batch, qwen_seq, 2048) from Qwen
    Output: (num_output_tokens, batch, 256) in seq-first format for SAM3
    """

    def __init__(
        self,
        qwen_dim: int = 2048,
        sam_dim: int = 256,
        num_output_tokens: int = 32,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_output_tokens = num_output_tokens
        self.sam_dim = sam_dim

        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(num_output_tokens, sam_dim) * 0.02
        )

        # Project Qwen dim to SAM dim for K, V
        self.kv_proj = nn.Linear(qwen_dim, sam_dim)

        # Stack of cross-attention layers
        self.layers = nn.ModuleList([
            CrossAttentionLayer(sam_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(sam_dim)

    def forward(
        self,
        qwen_features: torch.Tensor,
        qwen_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            qwen_features: (batch, qwen_seq, 2048)
            qwen_mask: (batch, qwen_seq) bool, True = valid token (standard convention)

        Returns:
            text_memory_resized: (num_output_tokens, batch, 256) — seq-first for SAM3
            text_attention_mask: (batch, num_output_tokens) bool, True = padding
                                 All False since queries always produce output.
        """
        B = qwen_features.shape[0]

        # Project Qwen features to SAM dim
        kv = self.kv_proj(qwen_features)  # (B, qwen_seq, 256)

        # Expand queries for batch
        queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, N, 256)

        # Invert mask for PyTorch MHA: True = ignore (padding)
        key_padding_mask = None
        if qwen_mask is not None:
            key_padding_mask = ~qwen_mask  # True = padding

        # Cross-attention layers
        for layer in self.layers:
            queries = layer(queries, kv, key_padding_mask)

        queries = self.final_norm(queries)  # (B, N, 256)

        # Convert to SAM3 convention: seq-first
        text_memory_resized = queries.transpose(0, 1)  # (N, B, 256)

        # All queries are valid (no padding)
        text_attention_mask = torch.zeros(
            B, self.num_output_tokens, dtype=torch.bool, device=qwen_features.device
        )

        return text_memory_resized, text_attention_mask


class QwenTextEncoder(nn.Module):
    """
    Drop-in replacement for VETextEncoder using Qwen2.5-VL as backbone.

    Unlike CLIP, Qwen processes the IMAGE alongside text, producing visually-
    grounded embeddings. Call set_image() before forward() to provide the image.

    Matches VETextEncoder.forward() signature and output shapes exactly.
    """

    DUMMY_EMBED_DIM = 1024  # Match CLIP's internal dim for compatibility

    def __init__(
        self,
        d_model: int = 256,
        qwen_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        qwen_dtype: torch.dtype = torch.bfloat16,
        adapter_num_tokens: int = 32,
        adapter_num_layers: int = 2,
        adapter_num_heads: int = 8,
        adapter_dropout: float = 0.1,
        freeze_qwen: bool = True,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.freeze_qwen = freeze_qwen
        self.qwen_dtype = qwen_dtype

        # ---- Load Qwen2.5-VL ---- #
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        # Use sdpa (PyTorch native) — works without flash_attn package
        self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_model_name,
            dtype=qwen_dtype,
            attn_implementation="sdpa",
        )
        self.processor = AutoProcessor.from_pretrained(qwen_model_name)
        self.qwen_hidden_size = self.qwen.config.hidden_size  # 2048 for 3B

        # Freeze Qwen base weights
        if freeze_qwen:
            for p in self.qwen.parameters():
                p.requires_grad_(False)

        # Optional LoRA
        if use_lora:
            self._apply_lora(lora_r, lora_alpha, lora_target_modules)

        # ---- Cross-attention adapter (TRAINABLE) ---- #
        self.adapter = CrossAttentionAdapter(
            qwen_dim=self.qwen_hidden_size,
            sam_dim=d_model,
            num_output_tokens=adapter_num_tokens,
            num_heads=adapter_num_heads,
            num_layers=adapter_num_layers,
            dropout=adapter_dropout,
        )

        # State: PIL images cached before forward_text is called
        self._cached_images = None  # List[PIL.Image] or None

    def _apply_lora(self, r, alpha, target_modules):
        """Apply LoRA to Qwen's attention layers."""
        from peft import LoraConfig, get_peft_model

        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.qwen = get_peft_model(self.qwen, lora_config)

    def set_image(self, images):
        """
        Cache PIL images for the next forward() call.

        Args:
            images: List[PIL.Image] — one per caption in the batch.
                    Or a single PIL.Image (broadcast to all captions).
                    Or None for text-only mode.
        """
        from PIL import Image
        if images is None:
            self._cached_images = None
        elif isinstance(images, Image.Image):
            self._cached_images = [images]  # will be broadcast
        else:
            self._cached_images = list(images)

    def _encode_with_qwen(
        self,
        texts: List[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run Qwen on image+text, return hidden states.

        Returns:
            hidden_states: (batch, seq, 2048)
            attention_mask: (batch, seq) bool, True = valid
        """
        has_images = self._cached_images is not None and len(self._cached_images) > 0

        # Build messages for Qwen processor
        messages_batch = []
        for text in texts:
            content = []
            if has_images:
                content.append({"type": "image"})
            content.append({"type": "text", "text": text})
            messages_batch.append([{"role": "user", "content": content}])

        # Apply chat template to get prompt strings with image placeholders
        prompt_texts = [
            self.processor.apply_chat_template(msgs, add_generation_prompt=False)
            for msgs in messages_batch
        ]

        # Process through Qwen's full processor (handles image + text together)
        if has_images:
            # Broadcast single image to all captions if needed
            images = self._cached_images
            if len(images) == 1 and len(texts) > 1:
                images = images * len(texts)

            model_inputs = self.processor(
                text=prompt_texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
        else:
            # Text-only: use tokenizer directly
            model_inputs = self.processor.tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

        # Move to device
        model_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in model_inputs.items()
        }

        # Forward through Qwen (no generation, just encoder)
        ctx = torch.no_grad() if self.freeze_qwen and not any(
            p.requires_grad for p in self.qwen.parameters()
        ) else nullcontext()

        with ctx:
            outputs = self.qwen.model(
                **model_inputs,
                output_hidden_states=True,
            )

        hidden_states = outputs.last_hidden_state  # (batch, seq, 2048)
        attn_mask = model_inputs["attention_mask"].bool()  # (batch, seq)

        return hidden_states, attn_mask

    def forward(
        self,
        text: Union[List[str], Tuple[torch.Tensor, torch.Tensor, dict]],
        input_boxes: Optional[List] = None,
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Drop-in replacement for VETextEncoder.forward().

        Returns:
            text_attention_mask: (batch, 32) bool, True = padding (inverted convention)
            text_memory_resized: (32, batch, 256) seq-first features
            inputs_embeds: (32, batch, 1024) dummy for compatibility
        """
        if isinstance(text[0], str):
            assert input_boxes is None or len(input_boxes) == 0, "not supported"

            # Encode with Qwen (image + text)
            hidden_states, qwen_mask = self._encode_with_qwen(text, device)

            # Cast to float32 for adapter (adapter runs in fp32)
            hidden_states = hidden_states.float()

            # Project through cross-attention adapter
            text_memory_resized, text_attention_mask = self.adapter(
                hidden_states, qwen_mask
            )

            # Dummy inputs_embeds for compatibility (not used downstream)
            inputs_embeds = torch.zeros(
                text_memory_resized.shape[0],  # seq
                text_memory_resized.shape[1],  # batch
                self.DUMMY_EMBED_DIM,
                device=device,
                dtype=text_memory_resized.dtype,
            )

            return text_attention_mask, text_memory_resized, inputs_embeds
        else:
            # Pre-encoded path (pass-through)
            text_attention_mask, text_memory_resized, tokenized = text
            inputs_embeds = tokenized["inputs_embeds"]
            assert input_boxes is None or len(input_boxes) == 0, (
                "Can't replace boxes in text if it's already encoded"
            )
            return text_attention_mask, text_memory_resized, inputs_embeds.transpose(0, 1)
