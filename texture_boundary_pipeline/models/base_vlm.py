from abc import ABC, abstractmethod
from typing import Union, Dict, Any, List, Optional
from pathlib import Path
from PIL import Image


class BaseVLM(ABC):
    """Abstract base class for Vision-Language Models."""

    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Initialize VLM.

        Args:
            model_name: Name/path of the model
            device: Device to run on (cuda/cpu)
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

        # Batch processing settings
        self._supports_true_batching = False
        self._max_batch_size = 1
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model and processor.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def generate(
        self, 
        image: Union[str, Path, Image.Image],
        prompt: str,
        max_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        Generate text response from image and prompt.
        
        Args:
            image: Image path, Path object, or PIL Image
            prompt: Text prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Model-specific arguments
        
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def batch_generate(
        self,
        images: list,
        prompts: list,
        max_tokens: int = 512,
        **kwargs
    ) -> list:
        """
        Generate responses for multiple images.

        Args:
            images: List of images
            prompts: List of prompts (one per image)
            max_tokens: Maximum tokens to generate
            **kwargs: Model-specific arguments

        Returns:
            List of generated responses
        """
        pass

    def batch_generate_optimized(
        self,
        images: list,
        prompts: list,
        batch_size: int = 4,
        max_tokens: int = 512,
        show_progress: bool = False,
        **kwargs
    ) -> list:
        """
        Generate responses with automatic batching optimization.

        Uses true batching if model supports it, otherwise falls back
        to sequential processing with proper chunking.

        Args:
            images: List of images
            prompts: List of prompts (one per image)
            batch_size: Batch size for processing
            max_tokens: Maximum tokens to generate
            show_progress: Show progress indicator
            **kwargs: Model-specific arguments

        Returns:
            List of generated responses
        """
        if len(images) != len(prompts):
            raise ValueError(f"Number of images ({len(images)}) must match prompts ({len(prompts)})")

        if not images:
            return []

        results = []
        total = len(images)

        for i in range(0, total, batch_size):
            batch_images = images[i:i + batch_size]
            batch_prompts = prompts[i:i + batch_size]

            if show_progress:
                print(f"  Processing batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}")

            batch_results = self.batch_generate(
                batch_images, batch_prompts, max_tokens, **kwargs
            )
            results.extend(batch_results)

        return results

    def to(self, device: str):
        """
        Move model to device.

        Args:
            device: Target device ('cuda' or 'cpu')
        """
        if self._model is not None and hasattr(self._model, 'to'):
            self._model.to(device)
            self.device = device

    def unload(self):
        """Unload model from memory."""
        import gc
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def supports_batching(self) -> bool:
        """Check if model supports true batched inference."""
        return self._supports_true_batching

    @property
    def max_batch_size(self) -> int:
        """Get recommended maximum batch size."""
        return self._max_batch_size

    @property
    def info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'is_loaded': self.is_loaded,
            'supports_batching': self._supports_true_batching,
            'max_batch_size': self._max_batch_size
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model_name}', device='{self.device}')"