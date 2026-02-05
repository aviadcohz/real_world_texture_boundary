"""
DINOv2 Feature Extractor for Texture Curator.

DINOv2 (Self-DIstillation with NO labels v2) is a self-supervised vision model
that produces high-quality semantic embeddings without any labels.

WHY DINOv2?
- Captures semantic meaning of textures (not just pixels)
- Pre-trained on massive dataset (LVD-142M images)
- Works well for texture/material recognition
- Produces consistent embeddings for similar textures

USAGE:
    extractor = DINOv2Extractor(device="cuda")
    embeddings = extractor.extract_batch(image_paths)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import List, Optional, Union, Tuple
import numpy as np
from tqdm import tqdm
import logging

# Setup logging
logger = logging.getLogger(__name__)


# ============================================
# Image Dataset for Batch Processing
# ============================================

class ImageDataset(Dataset):
    """
    Simple dataset for loading images from paths.
    
    Handles:
    - Loading images from disk
    - Converting to RGB (in case of grayscale/RGBA)
    - Applying transforms (resize, normalize)
    - Graceful error handling for corrupted images
    """
    
    def __init__(
        self,
        image_paths: List[Path],
        transform: transforms.Compose,
    ):
        """
        Args:
            image_paths: List of paths to images
            transform: Torchvision transforms to apply
        """
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, bool]:
        """
        Load and transform an image.
        
        Returns:
            Tuple of (image_tensor, image_id, success_flag)
        """
        path = self.image_paths[idx]
        image_id = path.stem  # filename without extension
        
        try:
            # Load image
            image = Image.open(path)
            
            # Convert to RGB (handles grayscale, RGBA, etc.)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Apply transforms
            image_tensor = self.transform(image)
            
            return image_tensor, image_id, True
            
        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}")
            # Return a dummy tensor on failure
            dummy = torch.zeros(3, 518, 518)
            return dummy, image_id, False


# ============================================
# DINOv2 Extractor
# ============================================

class DINOv2Extractor:
    """
    Extract semantic embeddings using DINOv2.
    
    This class handles:
    - Model loading (from torch hub)
    - Image preprocessing
    - Batch extraction with GPU acceleration
    - Memory-efficient processing for large datasets
    
    MODEL VARIANTS:
    - dinov2_vits14: Small (21M params, 384-dim embeddings)
    - dinov2_vitb14: Base (86M params, 768-dim embeddings) ← RECOMMENDED
    - dinov2_vitl14: Large (307M params, 1024-dim embeddings)
    - dinov2_vitg14: Giant (1.1B params, 1536-dim embeddings)
    """
    
    # Embedding dimensions for each model variant
    EMBEDDING_DIMS = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }
    
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: str = "cuda",
        image_size: int = 518,  # DINOv2 default (14*37)
    ):
        """
        Initialize the DINOv2 extractor.
        
        Args:
            model_name: Which DINOv2 variant to use
            device: "cuda" or "cpu"
            image_size: Size to resize images to (must be divisible by 14)
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.image_size = image_size
        
        # Get embedding dimension
        if model_name not in self.EMBEDDING_DIMS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(self.EMBEDDING_DIMS.keys())}")
        self.embedding_dim = self.EMBEDDING_DIMS[model_name]
        
        # Load model
        logger.info(f"Loading DINOv2 model: {model_name}")
        self.model = self._load_model()
        
        # Create transforms
        self.transform = self._create_transforms()
        
        logger.info(f"DINOv2 extractor ready on {self.device}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def _load_model(self) -> nn.Module:
        """Load DINOv2 model from torch hub."""
        model = torch.hub.load(
            "facebookresearch/dinov2",
            self.model_name,
            pretrained=True,
        )
        model = model.to(self.device)
        model.eval()
        return model
    
    def _create_transforms(self) -> transforms.Compose:
        """
        Create preprocessing transforms for DINOv2.
        
        DINOv2 expects:
        - Images resized to multiple of patch size (14)
        - Normalized with ImageNet stats
        """
        return transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225],    # ImageNet std
            ),
        ])
    
    @torch.no_grad()
    def extract_single(self, image: Union[Image.Image, Path, str]) -> np.ndarray:
        """
        Extract embedding for a single image.
        
        Args:
            image: PIL Image, or path to image
        
        Returns:
            Embedding array of shape (embedding_dim,)
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Transform and add batch dimension
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        features = self.model(image_tensor)
        
        # Convert to numpy
        embedding = features.cpu().numpy().squeeze()
        
        return embedding
    
    @torch.no_grad()
    def extract_batch(
        self,
        image_paths: List[Union[Path, str]],
        batch_size: int = 16,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Extract embeddings for multiple images efficiently.
        
        Args:
            image_paths: List of paths to images
            batch_size: Batch size for GPU processing
            show_progress: Whether to show progress bar
        
        Returns:
            Tuple of:
            - embeddings: Array of shape (N, embedding_dim)
            - ids: List of image IDs (filenames without extension)
            - failed: List of IDs that failed to process
        """
        # Convert to Path objects
        image_paths = [Path(p) for p in image_paths]
        
        # Create dataset and dataloader
        dataset = ImageDataset(image_paths, self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        # Extract embeddings
        all_embeddings = []
        all_ids = []
        failed_ids = []
        
        iterator = tqdm(dataloader, desc="Extracting DINOv2") if show_progress else dataloader
        
        for batch_images, batch_ids, batch_success in iterator:
            # Move to device
            batch_images = batch_images.to(self.device)
            
            # Extract features
            features = self.model(batch_images)
            
            # Process each item in batch
            for i, (embedding, img_id, success) in enumerate(
                zip(features.cpu().numpy(), batch_ids, batch_success)
            ):
                if success:
                    all_embeddings.append(embedding)
                    all_ids.append(img_id)
                else:
                    failed_ids.append(img_id)
        
        # Stack embeddings
        if all_embeddings:
            embeddings = np.stack(all_embeddings, axis=0)
        else:
            embeddings = np.zeros((0, self.embedding_dim))
        
        logger.info(f"Extracted {len(all_ids)} embeddings, {len(failed_ids)} failed")
        
        return embeddings, all_ids, failed_ids
    
    def compute_centroid(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute the centroid (mean) of a set of embeddings.
        
        Args:
            embeddings: Array of shape (N, embedding_dim)
        
        Returns:
            Centroid array of shape (embedding_dim,)
        """
        return np.mean(embeddings, axis=0)
    
    def cosine_similarity(
        self,
        embeddings: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between embeddings and a reference.
        
        Args:
            embeddings: Array of shape (N, embedding_dim)
            reference: Array of shape (embedding_dim,) - e.g., centroid
        
        Returns:
            Similarity scores of shape (N,) in range [-1, 1]
        """
        # Normalize embeddings
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        reference_norm = reference / (np.linalg.norm(reference) + 1e-8)
        
        # Compute cosine similarity
        similarities = embeddings_norm @ reference_norm
        
        return similarities


# ============================================
# Quick Test
# ============================================

if __name__ == "__main__":
    import tempfile
    
    print("=" * 60)
    print("DINOv2 Extractor Test")
    print("=" * 60)
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Create extractor
    print("\nLoading DINOv2...")
    extractor = DINOv2Extractor(
        model_name="dinov2_vitb14",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Create a test image
    print("\nCreating test image...")
    test_image = Image.new("RGB", (256, 256), color=(128, 100, 80))
    
    # Extract embedding
    print("Extracting embedding...")
    embedding = extractor.extract_single(test_image)
    
    print(f"\n✓ Embedding shape: {embedding.shape}")
    print(f"✓ Embedding dtype: {embedding.dtype}")
    print(f"✓ Embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")
    print(f"✓ Embedding norm: {np.linalg.norm(embedding):.3f}")
    
    # Test cosine similarity
    embedding2 = extractor.extract_single(test_image)  # Same image
    test_image3 = Image.new("RGB", (256, 256), color=(200, 50, 50))  # Different image
    embedding3 = extractor.extract_single(test_image3)
    
    # Stack embeddings
    embeddings = np.stack([embedding2, embedding3])
    
    # Compute similarities to first embedding
    similarities = extractor.cosine_similarity(embeddings, embedding)
    
    print(f"\n✓ Similarity (same image): {similarities[0]:.4f}")
    print(f"✓ Similarity (different image): {similarities[1]:.4f}")
    
    print("\n🎉 DINOv2 Extractor test passed!")