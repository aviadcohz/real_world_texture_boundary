"""
Texture Statistics Extractor for Texture Curator.

This module computes low-level texture features that complement the semantic
embeddings from DINOv2. These features capture the statistical properties
of textures that are important for texture boundary detection.

FEATURES COMPUTED:
1. Local Entropy: Measures texture complexity/randomness
2. GLCM (Gray-Level Co-occurrence Matrix):
   - Contrast: Local intensity variation
   - Homogeneity: Closeness of element distribution to diagonal
   - Energy: Sum of squared elements (uniformity)
   - Correlation: How correlated a pixel is to its neighbor

WHY THESE FEATURES?
- Entropy helps distinguish between uniform and complex textures
- GLCM captures spatial relationships between pixels
- These are classic texture features that work well alongside deep learning

USAGE:
    extractor = TextureStatsExtractor()
    stats = extractor.compute_stats(image_path)
    batch_stats = extractor.compute_batch(image_paths)
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional
from dataclasses import dataclass
from skimage.feature import graycomatrix, graycoprops
from skimage.filters.rank import entropy as local_entropy
from skimage.morphology import disk
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from tqdm import tqdm
import logging

# Setup logging
logger = logging.getLogger(__name__)


# ============================================
# Data Structure for Texture Stats
# ============================================

@dataclass
class TextureStats:
    """
    Container for texture statistics of a single image.
    """
    # Image ID
    image_id: str
    
    # Local entropy (average across image)
    entropy_mean: float
    entropy_std: float
    
    # GLCM features
    glcm_contrast: float
    glcm_homogeneity: float
    glcm_energy: float
    glcm_correlation: float
    
    # Processing flag
    success: bool = True
    error_message: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "image_id": self.image_id,
            "entropy_mean": self.entropy_mean,
            "entropy_std": self.entropy_std,
            "glcm_contrast": self.glcm_contrast,
            "glcm_homogeneity": self.glcm_homogeneity,
            "glcm_energy": self.glcm_energy,
            "glcm_correlation": self.glcm_correlation,
            "success": self.success,
            "error_message": self.error_message,
        }
    
    @classmethod
    def empty(cls, image_id: str, error: str = "") -> "TextureStats":
        """Create an empty stats object for failed processing."""
        return cls(
            image_id=image_id,
            entropy_mean=0.0,
            entropy_std=0.0,
            glcm_contrast=0.0,
            glcm_homogeneity=0.0,
            glcm_energy=0.0,
            glcm_correlation=0.0,
            success=False,
            error_message=error,
        )


# ============================================
# Texture Statistics Extractor
# ============================================

class TextureStatsExtractor:
    """
    Extract low-level texture statistics from images.
    
    This complements DINOv2's semantic features with classical
    texture analysis methods.
    """
    
    def __init__(
        self,
        entropy_disk_size: int = 5,
        glcm_distances: List[int] = None,
        glcm_angles: List[float] = None,
        resize_for_stats: Optional[int] = 256,
    ):
        """
        Initialize the texture stats extractor.
        
        Args:
            entropy_disk_size: Radius of disk for local entropy computation
            glcm_distances: Pixel distances for GLCM (default: [1, 2, 3])
            glcm_angles: Angles for GLCM in radians (default: [0, π/4, π/2, 3π/4])
            resize_for_stats: Resize images to this size for faster processing (None to keep original)
        """
        self.entropy_disk_size = entropy_disk_size
        self.glcm_distances = glcm_distances or [1, 2, 3]
        self.glcm_angles = glcm_angles or [0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.resize_for_stats = resize_for_stats
        
        logger.info(f"TextureStatsExtractor initialized")
        logger.info(f"  Entropy disk size: {entropy_disk_size}")
        logger.info(f"  GLCM distances: {self.glcm_distances}")
        logger.info(f"  Resize for stats: {resize_for_stats}")
    
    def _load_and_preprocess(self, image: Union[Image.Image, Path, str]) -> np.ndarray:
        """
        Load and preprocess image for texture analysis.
        
        Args:
            image: PIL Image or path to image
        
        Returns:
            Grayscale image as uint8 array
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize if specified
        if self.resize_for_stats:
            image = image.resize(
                (self.resize_for_stats, self.resize_for_stats),
                Image.Resampling.BILINEAR
            )
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Convert to grayscale
        gray = rgb2gray(image_array)
        
        # Convert to uint8 (required for GLCM)
        gray_uint8 = img_as_ubyte(gray)
        
        return gray_uint8
    
    def compute_entropy(self, gray_image: np.ndarray) -> Tuple[float, float]:
        """
        Compute local entropy statistics.
        
        Local entropy measures the complexity/randomness of the texture
        in local neighborhoods.
        
        Args:
            gray_image: Grayscale image as uint8 array
        
        Returns:
            Tuple of (mean_entropy, std_entropy)
        """
        # Create structuring element (disk)
        selem = disk(self.entropy_disk_size)
        
        # Compute local entropy
        entropy_map = local_entropy(gray_image, selem)
        
        # Compute statistics
        mean_entropy = float(np.mean(entropy_map))
        std_entropy = float(np.std(entropy_map))
        
        return mean_entropy, std_entropy
    
    def compute_glcm_features(self, gray_image: np.ndarray) -> Dict[str, float]:
        """
        Compute GLCM-based texture features.
        
        GLCM (Gray-Level Co-occurrence Matrix) captures spatial relationships
        between pixel intensities.
        
        Features:
        - Contrast: Measures local intensity variation
        - Homogeneity: Measures closeness of distribution to diagonal
        - Energy: Measures uniformity (sum of squared elements)
        - Correlation: Measures linear dependency between neighboring pixels
        
        Args:
            gray_image: Grayscale image as uint8 array
        
        Returns:
            Dictionary with GLCM feature values
        """
        # Compute GLCM
        # We use multiple distances and angles, then average
        glcm = graycomatrix(
            gray_image,
            distances=self.glcm_distances,
            angles=self.glcm_angles,
            levels=256,
            symmetric=True,
            normed=True,
        )
        
        # Compute properties (averaged over all distances/angles)
        contrast = float(np.mean(graycoprops(glcm, "contrast")))
        homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
        energy = float(np.mean(graycoprops(glcm, "energy")))
        correlation = float(np.mean(graycoprops(glcm, "correlation")))
        
        return {
            "contrast": contrast,
            "homogeneity": homogeneity,
            "energy": energy,
            "correlation": correlation,
        }
    
    def compute_stats(self, image: Union[Image.Image, Path, str]) -> TextureStats:
        """
        Compute all texture statistics for a single image.
        
        Args:
            image: PIL Image or path to image
        
        Returns:
            TextureStats object
        """
        # Get image ID
        if isinstance(image, (str, Path)):
            image_id = Path(image).stem
        else:
            image_id = "unknown"
        
        try:
            # Preprocess
            gray = self._load_and_preprocess(image)
            
            # Compute entropy
            entropy_mean, entropy_std = self.compute_entropy(gray)
            
            # Compute GLCM features
            glcm_features = self.compute_glcm_features(gray)
            
            return TextureStats(
                image_id=image_id,
                entropy_mean=entropy_mean,
                entropy_std=entropy_std,
                glcm_contrast=glcm_features["contrast"],
                glcm_homogeneity=glcm_features["homogeneity"],
                glcm_energy=glcm_features["energy"],
                glcm_correlation=glcm_features["correlation"],
                success=True,
            )
            
        except Exception as e:
            logger.warning(f"Failed to compute texture stats for {image_id}: {e}")
            return TextureStats.empty(image_id, str(e))
    
    def compute_batch(
        self,
        image_paths: List[Union[Path, str]],
        show_progress: bool = True,
    ) -> List[TextureStats]:
        """
        Compute texture statistics for multiple images.
        
        Args:
            image_paths: List of paths to images
            show_progress: Whether to show progress bar
        
        Returns:
            List of TextureStats objects
        """
        results = []
        
        iterator = tqdm(image_paths, desc="Computing texture stats") if show_progress else image_paths
        
        for path in iterator:
            stats = self.compute_stats(path)
            results.append(stats)
        
        # Log summary
        success_count = sum(1 for s in results if s.success)
        logger.info(f"Computed texture stats: {success_count}/{len(results)} successful")
        
        return results
    
    def stats_to_array(self, stats_list: List[TextureStats]) -> Tuple[np.ndarray, List[str]]:
        """
        Convert list of TextureStats to numpy array.
        
        Args:
            stats_list: List of TextureStats objects
        
        Returns:
            Tuple of (feature_array, image_ids)
            feature_array shape: (N, 6) - entropy_mean, entropy_std, 4 GLCM features
        """
        successful = [s for s in stats_list if s.success]
        
        if not successful:
            return np.zeros((0, 6)), []
        
        features = np.array([
            [
                s.entropy_mean,
                s.entropy_std,
                s.glcm_contrast,
                s.glcm_homogeneity,
                s.glcm_energy,
                s.glcm_correlation,
            ]
            for s in successful
        ])
        
        ids = [s.image_id for s in successful]
        
        return features, ids


# ============================================
# Quick Test
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Texture Statistics Extractor Test")
    print("=" * 60)
    
    # Create extractor
    extractor = TextureStatsExtractor()
    
    # Create test images with different textures
    print("\nCreating test images...")
    
    # Image 1: Uniform (low entropy, high energy)
    uniform_image = Image.new("RGB", (256, 256), color=(128, 128, 128))
    
    # Image 2: Noisy (high entropy, low energy)
    noise = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    noisy_image = Image.fromarray(noise)
    
    # Image 3: Striped (high contrast)
    stripes = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        if (i // 16) % 2 == 0:
            stripes[i, :] = [200, 200, 200]
        else:
            stripes[i, :] = [50, 50, 50]
    striped_image = Image.fromarray(stripes)
    
    # Compute stats for each
    print("\n--- Uniform Image ---")
    stats1 = extractor.compute_stats(uniform_image)
    print(f"Entropy: {stats1.entropy_mean:.2f} (±{stats1.entropy_std:.2f})")
    print(f"GLCM Contrast: {stats1.glcm_contrast:.4f}")
    print(f"GLCM Homogeneity: {stats1.glcm_homogeneity:.4f}")
    print(f"GLCM Energy: {stats1.glcm_energy:.4f}")
    
    print("\n--- Noisy Image ---")
    stats2 = extractor.compute_stats(noisy_image)
    print(f"Entropy: {stats2.entropy_mean:.2f} (±{stats2.entropy_std:.2f})")
    print(f"GLCM Contrast: {stats2.glcm_contrast:.4f}")
    print(f"GLCM Homogeneity: {stats2.glcm_homogeneity:.4f}")
    print(f"GLCM Energy: {stats2.glcm_energy:.4f}")
    
    print("\n--- Striped Image ---")
    stats3 = extractor.compute_stats(striped_image)
    print(f"Entropy: {stats3.entropy_mean:.2f} (±{stats3.entropy_std:.2f})")
    print(f"GLCM Contrast: {stats3.glcm_contrast:.4f}")
    print(f"GLCM Homogeneity: {stats3.glcm_homogeneity:.4f}")
    print(f"GLCM Energy: {stats3.glcm_energy:.4f}")
    
    # Verify expected patterns
    print("\n--- Verification ---")
    assert stats2.entropy_mean > stats1.entropy_mean, "Noisy should have higher entropy than uniform"
    print("✓ Noisy image has higher entropy than uniform")
    
    assert stats1.glcm_energy > stats2.glcm_energy, "Uniform should have higher energy than noisy"
    print("✓ Uniform image has higher energy (uniformity) than noisy")
    
    assert stats3.glcm_contrast > stats1.glcm_contrast, "Striped should have higher contrast than uniform"
    print("✓ Striped image has higher contrast than uniform")
    
    print("\n🎉 Texture Statistics Extractor test passed!")