"""
Qwen2SAM Inference Pipeline.

End-to-end inference:
  1. Load trained model (Qwen + LoRA + Projector + SAM)
  2. Accept image + optional text query
  3. Run Qwen generation (autoregressive)
  4. Find <SEG_A>/<SEG_B> tokens in generated output
  5. Extract hidden states at those positions
  6. Project through MLP → SAM prompt embeddings
  7. SAM decode → two complementary masks
  8. Return generated text + masks + IoU predictions

Usage:
    pipeline = Qwen2SAMPipeline.from_checkpoint("checkpoints/phase3/best.pt", cfg)
    result = pipeline.predict(image_pil)
    # result.text, result.mask_a, result.mask_b, result.boundary
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image

from qwen2sam.models.qwen2sam_v2 import (
    load_qwen_processor,
    SEG_A_TOKEN,
    SEG_B_TOKEN,
)
from qwen2sam.data.mask_utils import preprocess_image_for_sam


@dataclass
class PredictionResult:
    """Container for inference outputs."""
    text: str                              # Generated description
    mask_a: np.ndarray                     # (H, W) binary mask for texture A
    mask_b: np.ndarray                     # (H, W) binary mask for texture B
    mask_a_logits: np.ndarray              # (H, W) raw logits for mask A
    mask_b_logits: np.ndarray              # (H, W) raw logits for mask B
    iou_a: float                           # Predicted IoU for mask A
    iou_b: float                           # Predicted IoU for mask B
    boundary: np.ndarray | None = None     # (H, W) cleaned boundary line
    image_size: tuple[int, int] = (0, 0)   # Original (H, W) of input image
    seg_tokens_found: bool = True          # Whether both <SEG> tokens were generated


class Qwen2SAMPipeline:
    """
    End-to-end inference pipeline for Qwen2SAM.

    Loads the full model (Qwen + LoRA + Projector + SAM) and provides
    a simple .predict() interface that takes a PIL Image and returns
    segmentation masks with textual descriptions.
    """

    def __init__(
        self,
        model: Qwen2SAM,
        device: torch.device,
        system_prompt: str = (
            "You are a texture boundary segmentation assistant. "
            "When you identify a texture transition, describe each texture "
            "and mark it with <SEG_A> and <SEG_B> tokens."
        ),
        user_prompt: str = (
            "Describe and segment the texture transition in this image."
        ),
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        threshold: float = 0.5,
        image_size: int = 1024,
    ):
        self.model = model
        self.device = device
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.threshold = threshold
        self.image_size = image_size

        # Cache token IDs
        self.seg_a_id = model.seg_a_id
        self.seg_b_id = model.seg_b_id
        self.processor = model.processor
        self.tokenizer = model.processor.tokenizer

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        cfg: dict,
        device: str = "cuda",
    ) -> "Qwen2SAMPipeline":
        """
        Load a trained Qwen2SAM pipeline from a Phase 3 checkpoint.

        Args:
            checkpoint_path: path to Phase 3 .pt checkpoint
            cfg: config dict (or loaded from YAML)
            device: "cuda" or "cpu"

        Returns:
            Ready-to-use pipeline
        """
        torch_device = torch.device(device)

        # Build model
        model = Qwen2SAM(cfg, device=device)

        # Load checkpoint weights
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        model.projector.load_state_dict(ckpt["projector_state_dict"])
        model.sam_model.sam_mask_decoder.load_state_dict(
            ckpt["sam_decoder_state_dict"]
        )
        if "qwen_lora_state_dict" in ckpt:
            model.qwen.load_state_dict(
                ckpt["qwen_lora_state_dict"], strict=False
            )

        model.projector.to(torch_device)
        model.eval()
        print(f"Loaded Qwen2SAM from {checkpoint_path}")

        # Extract config values
        data_cfg = cfg.get("data", {})
        infer_cfg = cfg.get("inference", {})

        _default_sys = (
            "You are a texture boundary segmentation assistant. "
            "When you identify a texture transition, describe each texture "
            "and mark it with <SEG_A> and <SEG_B> tokens."
        )
        _default_user = (
            "Describe and segment the texture transition in this image."
        )

        return cls(
            model=model,
            device=torch_device,
            system_prompt=data_cfg.get("system_prompt", _default_sys),
            user_prompt=data_cfg.get("user_prompt", _default_user),
            max_new_tokens=infer_cfg.get("max_new_tokens", 128),
            temperature=infer_cfg.get("temperature", 0.1),
            threshold=infer_cfg.get("threshold", 0.5),
            image_size=cfg.get("model", {}).get("image_size", 1024),
        )

    # ------------------------------------------------------------------ #
    #  Chat message construction                                           #
    # ------------------------------------------------------------------ #

    def _build_messages(self, user_prompt: str | None = None) -> list[dict]:
        """Build Qwen chat messages for inference (no assistant message)."""
        prompt = user_prompt or self.user_prompt
        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    # ------------------------------------------------------------------ #
    #  Generation with hidden state extraction                             #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _generate_with_hidden_states(
        self,
        image: Image.Image,
        user_prompt: str | None = None,
    ) -> tuple[str, torch.Tensor | None, torch.Tensor | None]:
        """
        Run Qwen generation and extract hidden states at <SEG> positions.

        Strategy:
          1. Generate text autoregressively
          2. Re-run a forward pass on the full generated sequence with
             output_hidden_states=True to extract embeddings at <SEG> positions

        Returns:
            (generated_text, h_a, h_b) — h_a/h_b are (1, hidden_dim) or None
        """
        messages = self._build_messages(user_prompt)

        # Prepare input
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]

        # ---- Step 1: Generate ---------------------------------------- #
        gen_ids = self.model.qwen.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )

        # Decode generated text (new tokens only)
        new_ids = gen_ids[0, input_len:]
        generated_text = self.tokenizer.decode(
            new_ids, skip_special_tokens=False
        )

        # Clean up for display (remove chat control tokens)
        display_text = generated_text.replace("<|im_end|>", "").strip()

        # ---- Step 2: Find <SEG> positions in the full sequence ------- #
        full_ids = gen_ids[0]
        seg_a_positions = (full_ids == self.seg_a_id).nonzero(as_tuple=True)[0]
        seg_b_positions = (full_ids == self.seg_b_id).nonzero(as_tuple=True)[0]

        if len(seg_a_positions) == 0 or len(seg_b_positions) == 0:
            return display_text, None, None

        seg_a_pos = seg_a_positions[0].item()
        seg_b_pos = seg_b_positions[0].item()

        # ---- Step 3: Re-run forward pass to extract hidden states ---- #
        # Build inputs for the full generated sequence
        full_inputs = {
            "input_ids": gen_ids,
            "attention_mask": torch.ones_like(gen_ids),
        }
        # Re-include image tensors
        for key in ["pixel_values", "image_grid_thw"]:
            if key in inputs:
                full_inputs[key] = inputs[key]

        outputs = self.model.qwen(
            **full_inputs,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
        h_a = hidden_states[0, seg_a_pos].unsqueeze(0)  # (1, hidden_dim)
        h_b = hidden_states[0, seg_b_pos].unsqueeze(0)  # (1, hidden_dim)

        return display_text, h_a, h_b

    # ------------------------------------------------------------------ #
    #  SAM mask prediction from embeddings                                 #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _predict_masks(
        self,
        sam_image: torch.Tensor,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project embeddings and decode masks through SAM.

        Args:
            sam_image: (1, 3, 1024, 1024) preprocessed for SAM
            h_a: (1, hidden_dim) hidden state at <SEG_A>
            h_b: (1, hidden_dim) hidden state at <SEG_B>

        Returns:
            mask_a_logits: (1, 1024, 1024)
            mask_b_logits: (1, 1024, 1024)
            iou_a: (1,)
            iou_b: (1,)
        """
        # Project to SAM prompt space
        prompt_a = self.model.projector(h_a)  # (1, 256)
        prompt_b = self.model.projector(h_b)  # (1, 256)

        # SAM image encoding
        sam_features = self.model.encode_sam_image(sam_image)

        # Decode masks
        mask_a, iou_a = self.model.decode_mask(sam_features, prompt_a)
        mask_b, iou_b = self.model.decode_mask(sam_features, prompt_b)

        return (
            mask_a.squeeze(1),   # (1, 1024, 1024)
            mask_b.squeeze(1),
            iou_a.squeeze(1),    # (1,)
            iou_b.squeeze(1),
        )

    # ------------------------------------------------------------------ #
    #  Main prediction interface                                           #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image | np.ndarray | str,
        user_prompt: str | None = None,
        return_logits: bool = True,
    ) -> PredictionResult:
        """
        Run full inference on a single image.

        Args:
            image: PIL Image, numpy array (H,W,3 uint8 RGB), or file path
            user_prompt: optional custom query (overrides default)
            return_logits: whether to include raw logits in result

        Returns:
            PredictionResult with text, masks, IoU scores, and optionally boundary
        """
        # ---- Load image ---------------------------------------------- #
        if isinstance(image, (str, Path)):
            image_pil = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image.convert("RGB")

        orig_w, orig_h = image_pil.size
        image_np = np.array(image_pil)

        # ---- Qwen generation + hidden state extraction --------------- #
        text, h_a, h_b = self._generate_with_hidden_states(image_pil, user_prompt)

        if h_a is None or h_b is None:
            # <SEG> tokens not generated — return empty masks
            empty = np.zeros((orig_h, orig_w), dtype=np.float32)
            return PredictionResult(
                text=text,
                mask_a=empty,
                mask_b=empty,
                mask_a_logits=empty,
                mask_b_logits=empty,
                iou_a=0.0,
                iou_b=0.0,
                boundary=None,
                image_size=(orig_h, orig_w),
                seg_tokens_found=False,
            )

        # ---- SAM image preprocessing -------------------------------- #
        sam_image = preprocess_image_for_sam(image_np, self.image_size)
        sam_image = sam_image.unsqueeze(0).to(self.device)  # (1, 3, 1024, 1024)

        # ---- Mask prediction ----------------------------------------- #
        mask_a_logits, mask_b_logits, iou_a, iou_b = self._predict_masks(
            sam_image, h_a, h_b
        )

        # ---- Convert to numpy at original resolution ----------------- #
        mask_a_logits_np = mask_a_logits[0].cpu().float().numpy()
        mask_b_logits_np = mask_b_logits[0].cpu().float().numpy()

        # Binarize at SAM resolution
        mask_a_bin = (1.0 / (1.0 + np.exp(-mask_a_logits_np)) > self.threshold).astype(np.float32)
        mask_b_bin = (1.0 / (1.0 + np.exp(-mask_b_logits_np)) > self.threshold).astype(np.float32)

        # Resize to original image resolution
        import cv2
        if (orig_h, orig_w) != (self.image_size, self.image_size):
            mask_a_bin = cv2.resize(
                mask_a_bin, (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_b_bin = cv2.resize(
                mask_b_bin, (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )
            if return_logits:
                mask_a_logits_np = cv2.resize(
                    mask_a_logits_np, (orig_w, orig_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                mask_b_logits_np = cv2.resize(
                    mask_b_logits_np, (orig_w, orig_h),
                    interpolation=cv2.INTER_LINEAR,
                )

        return PredictionResult(
            text=text,
            mask_a=mask_a_bin,
            mask_b=mask_b_bin,
            mask_a_logits=mask_a_logits_np if return_logits else np.array([]),
            mask_b_logits=mask_b_logits_np if return_logits else np.array([]),
            iou_a=iou_a[0].item(),
            iou_b=iou_b[0].item(),
            boundary=None,
            image_size=(orig_h, orig_w),
            seg_tokens_found=True,
        )

    @torch.no_grad()
    def predict_batch(
        self,
        images: list[Image.Image | str],
        user_prompt: str | None = None,
    ) -> list[PredictionResult]:
        """
        Run inference on multiple images sequentially.

        For batch processing. Each image is processed independently
        (Qwen generation is autoregressive, so no true batching).
        """
        results = []
        for img in images:
            results.append(self.predict(img, user_prompt))
        return results
