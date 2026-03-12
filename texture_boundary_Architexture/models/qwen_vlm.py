import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from pathlib import Path
from typing import Union, List, Any, Dict, Optional

from .base_vlm import BaseVLM


class QwenVLM(BaseVLM):
    """Qwen3-VL Vision-Language Model implementation with batch support."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        torch_dtype = torch.bfloat16,
        enable_batching: bool = True
    ):
        """
        Initialize Qwen VLM.

        Args:
            model_name: Hugging Face model name (default: Qwen3-VL-8B-Instruct)
            device: Device to run on
            torch_dtype: PyTorch data type
            enable_batching: Enable true batched inference
        """
        super().__init__(model_name, device)
        self.torch_dtype = torch_dtype
        self._supports_true_batching = enable_batching
        self._max_batch_size = 4  # Recommended max batch for 24GB VRAM
        self.load_model()

    def load_model(self) -> None:
        """Load Qwen model and processor."""
        print(f"Loading {self.model_name} on {self.device}...")

        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto"
        )

        self._processor = AutoProcessor.from_pretrained(self.model_name)

        # Enable padding for batching
        if self._processor.tokenizer.pad_token is None:
            self._processor.tokenizer.pad_token = self._processor.tokenizer.eos_token

        # Set left padding for decoder-only models (required for correct generation)
        self._processor.tokenizer.padding_side = 'left'

        print(f"Model loaded successfully!")

    def generate(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        max_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        Generate text from image and prompt.

        Args:
            image: Image path, Path object, or PIL Image
            prompt: Text prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation arguments

        Returns:
            Generated text response
        """
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Process inputs
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate response
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                **kwargs
            )

            # Trim input tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            # Decode
            output_text = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

        return output_text


    def custom_generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        **kwargs
    ) -> str:

        # Process messages
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate response
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                **kwargs
            )

            # Trim input tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            # Decode
            output_text = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

        return output_text


    def batch_generate(
        self,
        images: List[Union[str, Path, Image.Image]],
        prompts: List[str],
        max_tokens: int = 512,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for multiple images with true batching.

        Processes multiple images in a single forward pass for efficiency.

        Args:
            images: List of images
            prompts: List of prompts (one per image)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation arguments

        Returns:
            List of generated responses
        """
        if len(images) != len(prompts):
            raise ValueError(f"Number of images ({len(images)}) must match prompts ({len(prompts)})")

        if not images:
            return []

        # For single image, use optimized single path
        if len(images) == 1:
            return [self.generate(images[0], prompts[0], max_tokens, **kwargs)]

        # Check if true batching is enabled
        if not self._supports_true_batching:
            # Fallback to sequential processing
            return [
                self.generate(img, prompt, max_tokens, **kwargs)
                for img, prompt in zip(images, prompts)
            ]

        # Prepare batch messages
        all_messages = []
        for image, prompt in zip(images, prompts):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            all_messages.append(messages)

        # Process each message to get text representations
        texts = []
        all_image_inputs = []

        for messages in all_messages:
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)

            image_inputs, _ = process_vision_info(messages)
            all_image_inputs.extend(image_inputs)

        # Batch process inputs
        try:
            inputs = self._processor(
                text=texts,
                images=all_image_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # Generate responses for batch
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    **kwargs
                )

                # Trim input tokens for each sequence
                generated_ids_trimmed = []
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids):
                    generated_ids_trimmed.append(out_ids[len(in_ids):])

                # Decode all outputs
                output_texts = self._processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )

            return output_texts

        except RuntimeError as e:
            # If batching fails (e.g., OOM), fall back to sequential
            if "out of memory" in str(e).lower():
                print(f"  Batch OOM, falling back to sequential processing...")
                torch.cuda.empty_cache()
                return [
                    self.generate(img, prompt, max_tokens, **kwargs)
                    for img, prompt in zip(images, prompts)
                ]
            raise

    def batch_generate_chunked(
        self,
        images: List[Union[str, Path, Image.Image]],
        prompts: List[str],
        batch_size: int = None,
        max_tokens: int = 512,
        show_progress: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate responses with automatic chunking for large batches.

        Args:
            images: List of images
            prompts: List of prompts (one per image)
            batch_size: Batch size (default: model's max_batch_size)
            max_tokens: Maximum tokens to generate
            show_progress: Show progress indicator
            **kwargs: Additional generation arguments

        Returns:
            List of generated responses
        """
        if batch_size is None:
            batch_size = self._max_batch_size

        results = []
        total = len(images)

        for i in range(0, total, batch_size):
            batch_images = images[i:i + batch_size]
            batch_prompts = prompts[i:i + batch_size]

            if show_progress:
                batch_num = i // batch_size + 1
                total_batches = (total + batch_size - 1) // batch_size
                print(f"  Processing batch {batch_num}/{total_batches} ({len(batch_images)} images)...")

            batch_results = self.batch_generate(
                batch_images, batch_prompts, max_tokens, **kwargs
            )
            results.extend(batch_results)

        return results

    @property
    def info(self) -> dict:
        """Get model information."""
        base_info = super().info
        base_info.update({
            'torch_dtype': str(self.torch_dtype),
            'processor': self._processor.__class__.__name__ if self._processor else None
        })
        return base_info


# Convenience function for quick model creation
def create_qwen_model(
    model_size: str = "8B",
    device: str = "cuda",
    enable_batching: bool = True
) -> QwenVLM:
    """
    Create Qwen model with common configurations.

    Args:
        model_size: Model size - "8B" or "2B" (default: 8B)
        device: Device to use
        enable_batching: Enable batched inference

    Returns:
        Initialized QwenVLM
    """
    model_names = {
        "8B": "Qwen/Qwen3-VL-8B-Instruct",
        "2B": "Qwen/Qwen3-VL-2B-Instruct"
    }

    if model_size not in model_names:
        raise ValueError(f"Invalid model_size. Choose from: {list(model_names.keys())}")

    return QwenVLM(
        model_name=model_names[model_size],
        device=device,
        enable_batching=enable_batching
    )
