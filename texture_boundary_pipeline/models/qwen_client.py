"""
Remote client for Qwen VLM server.

Implements the same interface as QwenVLM but sends requests to a remote server.
Can be used as a drop-in replacement in the pipeline.

Usage:
    from models.qwen_client import QwenVLMClient

    # Connect to remote server
    client = QwenVLMClient(server_url="http://132.66.150.69:8000")

    # Use like regular model
    response = client.generate(image_path, prompt)
"""

import base64
import io
import requests
from pathlib import Path
from typing import List, Union, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from .base_vlm import BaseVLM


class QwenVLMClient(BaseVLM):
    """
    Remote client for Qwen VLM server.

    Implements BaseVLM interface but sends requests to remote server.
    Supports batched requests over network.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        timeout: int = 120,
        max_retries: int = 3,
        batch_size: int = 8
    ):
        """
        Initialize remote client.

        Args:
            server_url: URL of the Qwen VLM server
            timeout: Request timeout in seconds
            max_retries: Number of retries on failure
            batch_size: Batch size for remote requests
        """
        super().__init__(model_name=f"remote:{server_url}", device="remote")
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self._max_batch_size = batch_size
        self._session = requests.Session()

        # Verify connection
        self._check_connection()

    def _check_connection(self):
        """Check if server is reachable and model is loaded."""
        try:
            response = self._session.get(
                f"{self.server_url}/health",
                timeout=10
            )
            response.raise_for_status()
            health = response.json()

            if not health.get('model_loaded', False):
                print(f"Warning: Server at {self.server_url} is running but model not loaded")
            else:
                print(f"Connected to remote VLM server: {self.server_url}")
                print(f"  Model: {health.get('model_name', 'unknown')}")
                print(f"  Device: {health.get('device', 'unknown')}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Cannot connect to server at {self.server_url}: {e}")

    def load_model(self) -> None:
        """No-op for remote client - model is on server."""
        pass

    def _encode_image(self, image: Union[str, Path, Image.Image]) -> str:
        """Encode image to base64 string."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')

        # Convert to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode('utf-8')

    def generate(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        max_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        Generate response for single image via remote server.

        Args:
            image: Image path or PIL Image
            prompt: Text prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        image_b64 = self._encode_image(image)

        for attempt in range(self.max_retries):
            try:
                response = self._session.post(
                    f"{self.server_url}/generate",
                    json={
                        "image_base64": image_b64,
                        "prompt": prompt,
                        "max_tokens": max_tokens
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()

                if not result.get('success', True):
                    raise RuntimeError(result.get('error', 'Unknown error'))

                return result['response']

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    print(f"  Request failed, retrying ({attempt + 1}/{self.max_retries})...")
                    continue
                raise RuntimeError(f"Remote generation failed after {self.max_retries} attempts: {e}")

    def batch_generate(
        self,
        images: List[Union[str, Path, Image.Image]],
        prompts: List[str],
        max_tokens: int = 512,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for multiple images via remote server.

        Uses server-side batching for efficiency.

        Args:
            images: List of images
            prompts: List of prompts
            max_tokens: Maximum tokens to generate

        Returns:
            List of generated responses
        """
        if len(images) != len(prompts):
            raise ValueError(f"Number of images ({len(images)}) must match prompts ({len(prompts)})")

        if not images:
            return []

        # Encode all images (can be parallelized)
        with ThreadPoolExecutor(max_workers=8) as executor:
            images_b64 = list(executor.map(self._encode_image, images))

        for attempt in range(self.max_retries):
            try:
                response = self._session.post(
                    f"{self.server_url}/batch_generate",
                    json={
                        "images_base64": images_b64,
                        "prompts": prompts,
                        "max_tokens": max_tokens
                    },
                    timeout=self.timeout * 2  # Longer timeout for batches
                )
                response.raise_for_status()
                result = response.json()

                if not result.get('success', True):
                    raise RuntimeError(result.get('error', 'Unknown error'))

                return result['responses']

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    print(f"  Batch request failed, retrying ({attempt + 1}/{self.max_retries})...")
                    continue
                raise RuntimeError(f"Remote batch generation failed: {e}")

    def custom_generate(self, messages, max_tokens: int = 512, **kwargs) -> str:
        """
        Custom generation not supported over remote.
        Falls back to standard generate with extracted content.
        """
        # Extract image and text from messages
        image = None
        text = ""

        for msg in messages:
            if msg.get('role') == 'user':
                for content in msg.get('content', []):
                    if content.get('type') == 'image':
                        image = content.get('image')
                    elif content.get('type') == 'text':
                        text = content.get('text', '')

        if image is None:
            raise ValueError("No image found in messages")

        return self.generate(image, text, max_tokens, **kwargs)

    @property
    def info(self) -> dict:
        """Get client information."""
        return {
            'type': 'remote_client',
            'server_url': self.server_url,
            'timeout': self.timeout,
            'max_batch_size': self._max_batch_size
        }

    def __repr__(self):
        return f"QwenVLMClient(server_url='{self.server_url}')"


def create_remote_model(
    server_url: str = "http://localhost:8000",
    timeout: int = 120,
    batch_size: int = 8
) -> QwenVLMClient:
    """
    Create a remote VLM client.

    Args:
        server_url: URL of the Qwen VLM server
        timeout: Request timeout
        batch_size: Batch size for requests

    Returns:
        QwenVLMClient instance
    """
    return QwenVLMClient(
        server_url=server_url,
        timeout=timeout,
        batch_size=batch_size
    )
