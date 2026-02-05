import torch
import gc
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, List, Dict, Optional
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from skimage.morphology import skeletonize, binary_erosion


class Sa2VAModel:
    """
    Sa2VA model for texture segmentation and boundary extraction.
    Used specifically for generating ground truth boundaries from descriptions.

    Supports batch processing for improved throughput.
    """

    def __init__(self, device: str = None, lazy_load: bool = False):
        """
        Initialize Sa2VA model.

        Args:
            device: 'cuda' or 'cpu' (auto-detect if None)
            lazy_load: If True, don't load model until first use
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._is_loaded = False

        if not lazy_load:
            self.load_model()

    def load_model(self):
        """Load the model and processor."""
        if self._is_loaded:
            return

        print(f"\n{'='*70}")
        print("Loading Sa2VA model for boundary extraction...")
        print(f"Device: {self.device}")
        print(f"{'='*70}")

        self.model = AutoModel.from_pretrained(
            "ByteDance/Sa2VA-Qwen2_5-VL-7B",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            "ByteDance/Sa2VA-Qwen2_5-VL-7B",
            trust_remote_code=True
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "ByteDance/Sa2VA-Qwen2_5-VL-7B",
                trust_remote_code=True
            )
        except:
            self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else None

        self._is_loaded = True
        print("Sa2VA loaded successfully")
        print(f"{'='*70}\n")

    def unload(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self._is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def to(self, device: str):
        """
        Move model to device.

        Args:
            device: Target device ('cuda' or 'cpu')
        """
        if self.model is not None and hasattr(self.model, 'to'):
            self.model.to(device)
            self.device = device

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def segment_texture(
        self,
        image: Union[str, Path, Image.Image],
        texture_description: str
    ) -> np.ndarray:
        """
        Segment a texture region based on description.

        Args:
            image: Image path or PIL Image
            texture_description: Text description of texture

        Returns:
            Binary mask (H×W uint8)
        """
        if not self._is_loaded:
            self.load_model()

        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        orig_size = image.size

        # Run segmentation
        result = self.model.predict_forward(
            image=image,
            text=f"segment all regions with {texture_description}",
            tokenizer=self.tokenizer,
            processor=self.processor
        )

        # Convert to mask
        mask = self._convert_to_mask(result['prediction_masks'][0], orig_size)

        return mask

    def batch_segment_texture(
        self,
        images: List[Union[str, Path, Image.Image]],
        descriptions: List[str],
        show_progress: bool = False
    ) -> List[np.ndarray]:
        """
        Segment texture regions for multiple images.

        Note: Sa2VA processes images sequentially but this method
        provides a consistent interface and can be optimized later.

        Args:
            images: List of images (paths or PIL Images)
            descriptions: List of texture descriptions
            show_progress: Show progress indicator

        Returns:
            List of binary masks
        """
        if not self._is_loaded:
            self.load_model()

        if len(images) != len(descriptions):
            raise ValueError(f"Number of images ({len(images)}) must match descriptions ({len(descriptions)})")

        masks = []
        total = len(images)

        for idx, (image, desc) in enumerate(zip(images, descriptions)):
            if show_progress:
                print(f"  Segmenting {idx + 1}/{total}...")

            try:
                mask = self.segment_texture(image, desc)
                masks.append(mask)
            except Exception as e:
                print(f"  Warning: Failed to segment image {idx}: {e}")
                # Return empty mask on failure
                if isinstance(image, (str, Path)):
                    img = Image.open(image)
                    h, w = img.size[1], img.size[0]
                else:
                    h, w = image.size[1], image.size[0]
                masks.append(np.zeros((h, w), dtype=np.uint8))

        return masks

    def segment_dual_textures(
        self,
        image: Union[str, Path, Image.Image],
        texture_a_desc: str,
        texture_b_desc: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment two texture regions from an image.

        Args:
            image: Image path or PIL Image
            texture_a_desc: Description of first texture
            texture_b_desc: Description of second texture

        Returns:
            Tuple of (mask_a, mask_b)
        """
        if not self._is_loaded:
            self.load_model()

        # Load image once if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        mask_a = self.segment_texture(image, texture_a_desc)
        mask_b = self.segment_texture(image, texture_b_desc)

        return mask_a, mask_b

    def batch_segment_dual_textures(
        self,
        images: List[Union[str, Path, Image.Image]],
        texture_pairs: List[Tuple[str, str]],
        show_progress: bool = False
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Segment dual texture regions for multiple images.

        Args:
            images: List of images
            texture_pairs: List of (texture_a_desc, texture_b_desc) tuples
            show_progress: Show progress indicator

        Returns:
            List of (mask_a, mask_b) tuples
        """
        if not self._is_loaded:
            self.load_model()

        if len(images) != len(texture_pairs):
            raise ValueError(f"Number of images ({len(images)}) must match texture pairs ({len(texture_pairs)})")

        results = []
        total = len(images)

        for idx, (image, (desc_a, desc_b)) in enumerate(zip(images, texture_pairs)):
            if show_progress:
                print(f"  Segmenting {idx + 1}/{total}...")

            try:
                mask_a, mask_b = self.segment_dual_textures(image, desc_a, desc_b)
                results.append((mask_a, mask_b))
            except Exception as e:
                print(f"  Warning: Failed to segment image {idx}: {e}")
                # Return empty masks on failure
                if isinstance(image, (str, Path)):
                    img = Image.open(image)
                    h, w = img.size[1], img.size[0]
                else:
                    h, w = image.size[1], image.size[0]
                results.append((
                    np.zeros((h, w), dtype=np.uint8),
                    np.zeros((h, w), dtype=np.uint8)
                ))

        return results

    def _convert_to_mask(self, mask, orig_size):
        """Convert model output to binary mask"""
        if len(mask.shape) == 3:
            mask = mask[0]

        if mask.dtype == bool or mask.dtype == np.bool_:
            mask = mask.astype(np.uint8)

        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)

        width, height = orig_size
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        return mask


def create_sa2va_model(device: str = None, lazy_load: bool = False) -> Sa2VAModel:
    """
    Factory function to create Sa2VA model.

    Args:
        device: Device to use
        lazy_load: Defer model loading until first use

    Returns:
        Sa2VAModel instance
    """
    return Sa2VAModel(device=device, lazy_load=lazy_load)
