"""
Boundary Metrics Extractor for Texture Curator.

This module analyzes the quality of texture boundaries by examining
where the mask sits relative to actual edges in the image.

KEY INSIGHT:
A good texture transition mask should sit on a real physical boundary
where the image gradient is strong. If the mask boundary is in a uniform
region (no gradient), it's likely not a real texture transition.

METRICS COMPUTED:
1. Variance of Laplacian (VoL): Measures sharpness/blur along boundary
2. Gradient Magnitude: Measures edge strength along boundary
3. Edge Density: What percentage of boundary pixels are on image edges

WHY THESE METRICS?
- VoL: High VoL means sharp features (good boundary placement)
- Gradient: Strong gradient = real edge, weak gradient = no edge
- Edge Density: Ensures mask follows actual edges, not arbitrary regions

USAGE:
    extractor = BoundaryMetricsExtractor()
    metrics = extractor.compute_metrics(image_path, mask_path)
    batch_metrics = extractor.compute_batch(image_paths, mask_paths)
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional
from dataclasses import dataclass
import cv2
from tqdm import tqdm
import logging

# Setup logging
logger = logging.getLogger(__name__)


# ============================================
# Data Structure for Boundary Metrics
# ============================================

@dataclass
class BoundaryMetrics:
    """
    Container for boundary quality metrics of a single image/mask pair.
    """
    # Image ID
    image_id: str
    
    # Variance of Laplacian along boundary
    # Higher = sharper features at boundary
    variance_of_laplacian: float
    
    # Gradient magnitude along boundary
    gradient_magnitude_mean: float
    gradient_magnitude_std: float
    gradient_magnitude_max: float
    
    # Edge density: ratio of boundary pixels that are on image edges
    edge_density: float
    
    # Boundary statistics
    boundary_length: int  # Number of pixels in boundary
    
    # Processing flag
    success: bool = True
    error_message: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "image_id": self.image_id,
            "variance_of_laplacian": self.variance_of_laplacian,
            "gradient_magnitude_mean": self.gradient_magnitude_mean,
            "gradient_magnitude_std": self.gradient_magnitude_std,
            "gradient_magnitude_max": self.gradient_magnitude_max,
            "edge_density": self.edge_density,
            "boundary_length": self.boundary_length,
            "success": self.success,
            "error_message": self.error_message,
        }
    
    @classmethod
    def empty(cls, image_id: str, error: str = "") -> "BoundaryMetrics":
        """Create an empty metrics object for failed processing."""
        return cls(
            image_id=image_id,
            variance_of_laplacian=0.0,
            gradient_magnitude_mean=0.0,
            gradient_magnitude_std=0.0,
            gradient_magnitude_max=0.0,
            edge_density=0.0,
            boundary_length=0,
            success=False,
            error_message=error,
        )


# ============================================
# Boundary Metrics Extractor
# ============================================

class BoundaryMetricsExtractor:
    """
    Extract boundary quality metrics from image/mask pairs.
    
    This class analyzes whether a mask boundary sits on a real
    texture edge in the image.
    """
    
    def __init__(
        self,
        boundary_dilation: int = 3,
        canny_low: int = 50,
        canny_high: int = 150,
        resize_for_metrics: Optional[int] = None,
    ):
        """
        Initialize the boundary metrics extractor.
        
        Args:
            boundary_dilation: Pixels to dilate mask boundary for analysis
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            resize_for_metrics: Resize images to this size (None to keep original)
        """
        self.boundary_dilation = boundary_dilation
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.resize_for_metrics = resize_for_metrics
        
        logger.info(f"BoundaryMetricsExtractor initialized")
        logger.info(f"  Boundary dilation: {boundary_dilation}")
        logger.info(f"  Canny thresholds: ({canny_low}, {canny_high})")
    
    def _load_image(self, image: Union[Image.Image, Path, str, np.ndarray]) -> np.ndarray:
        """
        Load and convert image to grayscale.
        
        Args:
            image: PIL Image, path, or numpy array
        
        Returns:
            Grayscale image as uint8 array
        """
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
        elif isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:  # PIL Image
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Resize if specified
        if self.resize_for_metrics:
            gray = cv2.resize(gray, (self.resize_for_metrics, self.resize_for_metrics))
        
        return gray
    
    def _load_mask(self, mask: Union[Image.Image, Path, str, np.ndarray]) -> np.ndarray:
        """
        Load and binarize mask.
        
        Args:
            mask: PIL Image, path, or numpy array
        
        Returns:
            Binary mask as uint8 array (0 or 255)
        """
        if isinstance(mask, np.ndarray):
            mask_array = mask
        elif isinstance(mask, (str, Path)):
            mask_array = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        else:  # PIL Image
            mask_array = np.array(mask.convert("L"))
        
        # Resize if specified
        if self.resize_for_metrics:
            mask_array = cv2.resize(mask_array, (self.resize_for_metrics, self.resize_for_metrics))
        
        # Binarize (threshold at 127)
        _, binary_mask = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)
        
        return binary_mask
    
    def extract_boundary(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract the boundary pixels from a binary mask.
        
        The boundary is defined as pixels where the mask transitions
        from foreground to background.
        
        Args:
            mask: Binary mask (0 or 255)
        
        Returns:
            Binary boundary mask
        """
        # Create kernel for morphological operations
        kernel = np.ones((3, 3), np.uint8)
        
        # Erode mask
        eroded = cv2.erode(mask, kernel, iterations=1)
        
        # Boundary = original - eroded
        boundary = cv2.subtract(mask, eroded)
        
        return boundary
    
    def extract_boundary_region(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract a dilated region around the mask boundary.
        
        This creates a "band" around the boundary for more robust
        metric computation.
        
        Args:
            mask: Binary mask (0 or 255)
        
        Returns:
            Binary mask of the boundary region
        """
        # Get thin boundary
        boundary = self.extract_boundary(mask)
        
        # Dilate to create region
        kernel = np.ones((self.boundary_dilation * 2 + 1, self.boundary_dilation * 2 + 1), np.uint8)
        boundary_region = cv2.dilate(boundary, kernel, iterations=1)
        
        return boundary_region
    
    def compute_variance_of_laplacian(
        self,
        gray_image: np.ndarray,
        boundary_region: np.ndarray,
    ) -> float:
        """
        Compute Variance of Laplacian (VoL) along the boundary region.
        
        VoL is a measure of image sharpness. Higher values indicate
        sharper features (edges, textures) at the boundary.
        
        Args:
            gray_image: Grayscale image
            boundary_region: Binary mask of boundary region
        
        Returns:
            Variance of Laplacian value
        """
        # Compute Laplacian of entire image
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        
        # Extract values only in boundary region
        boundary_pixels = laplacian[boundary_region > 0]
        
        if len(boundary_pixels) == 0:
            return 0.0
        
        # Compute variance
        vol = float(np.var(boundary_pixels))
        
        return vol
    
    def compute_gradient_stats(
        self,
        gray_image: np.ndarray,
        boundary_region: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Compute gradient magnitude statistics along the boundary region.
        
        Strong gradients indicate real edges; weak gradients suggest
        the boundary is not on a real edge.
        
        Args:
            gray_image: Grayscale image
            boundary_region: Binary mask of boundary region
        
        Returns:
            Tuple of (mean, std, max) gradient magnitude
        """
        # Compute gradients using Sobel operators
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Extract values only in boundary region
        boundary_gradients = gradient_magnitude[boundary_region > 0]
        
        if len(boundary_gradients) == 0:
            return 0.0, 0.0, 0.0
        
        mean_grad = float(np.mean(boundary_gradients))
        std_grad = float(np.std(boundary_gradients))
        max_grad = float(np.max(boundary_gradients))
        
        return mean_grad, std_grad, max_grad
    
    def compute_edge_density(
        self,
        gray_image: np.ndarray,
        boundary: np.ndarray,
    ) -> float:
        """
        Compute what fraction of boundary pixels sit on detected edges.
        
        Uses Canny edge detection to find image edges, then computes
        the overlap with the mask boundary.
        
        Args:
            gray_image: Grayscale image
            boundary: Binary boundary mask (thin)
        
        Returns:
            Edge density (0-1), where 1 means all boundary pixels are on edges
        """
        # Detect edges using Canny
        edges = cv2.Canny(gray_image, self.canny_low, self.canny_high)
        
        # Dilate edges slightly for more robust matching
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Count boundary pixels
        boundary_pixels = np.sum(boundary > 0)
        
        if boundary_pixels == 0:
            return 0.0
        
        # Count boundary pixels that overlap with edges
        overlap = np.sum((boundary > 0) & (edges_dilated > 0))
        
        # Compute density
        edge_density = float(overlap / boundary_pixels)
        
        return edge_density
    
    def compute_metrics(
        self,
        image: Union[Image.Image, Path, str, np.ndarray],
        mask: Union[Image.Image, Path, str, np.ndarray],
    ) -> BoundaryMetrics:
        """
        Compute all boundary metrics for an image/mask pair.
        
        Args:
            image: Input image
            mask: Binary mask
        
        Returns:
            BoundaryMetrics object
        """
        # Get image ID
        if isinstance(image, (str, Path)):
            image_id = Path(image).stem
        elif isinstance(mask, (str, Path)):
            image_id = Path(mask).stem
        else:
            image_id = "unknown"
        
        try:
            # Load and preprocess
            gray = self._load_image(image)
            mask_binary = self._load_mask(mask)
            
            # Extract boundary
            boundary = self.extract_boundary(mask_binary)
            boundary_region = self.extract_boundary_region(mask_binary)
            
            # Count boundary pixels
            boundary_length = int(np.sum(boundary > 0))
            
            if boundary_length == 0:
                logger.warning(f"No boundary pixels found for {image_id}")
                return BoundaryMetrics.empty(image_id, "No boundary pixels found")
            
            # Compute metrics
            vol = self.compute_variance_of_laplacian(gray, boundary_region)
            grad_mean, grad_std, grad_max = self.compute_gradient_stats(gray, boundary_region)
            edge_density = self.compute_edge_density(gray, boundary)
            
            return BoundaryMetrics(
                image_id=image_id,
                variance_of_laplacian=vol,
                gradient_magnitude_mean=grad_mean,
                gradient_magnitude_std=grad_std,
                gradient_magnitude_max=grad_max,
                edge_density=edge_density,
                boundary_length=boundary_length,
                success=True,
            )
            
        except Exception as e:
            logger.warning(f"Failed to compute boundary metrics for {image_id}: {e}")
            return BoundaryMetrics.empty(image_id, str(e))
    
    def compute_batch(
        self,
        image_paths: List[Union[Path, str]],
        mask_paths: List[Union[Path, str]],
        show_progress: bool = True,
    ) -> List[BoundaryMetrics]:
        """
        Compute boundary metrics for multiple image/mask pairs.
        
        Args:
            image_paths: List of paths to images
            mask_paths: List of paths to masks (same order as images)
            show_progress: Whether to show progress bar
        
        Returns:
            List of BoundaryMetrics objects
        """
        assert len(image_paths) == len(mask_paths), \
            f"Number of images ({len(image_paths)}) != number of masks ({len(mask_paths)})"
        
        results = []
        
        pairs = list(zip(image_paths, mask_paths))
        iterator = tqdm(pairs, desc="Computing boundary metrics") if show_progress else pairs
        
        for image_path, mask_path in iterator:
            metrics = self.compute_metrics(image_path, mask_path)
            results.append(metrics)
        
        # Log summary
        success_count = sum(1 for m in results if m.success)
        logger.info(f"Computed boundary metrics: {success_count}/{len(results)} successful")
        
        return results
    
    def metrics_to_array(self, metrics_list: List[BoundaryMetrics]) -> Tuple[np.ndarray, List[str]]:
        """
        Convert list of BoundaryMetrics to numpy array.
        
        Args:
            metrics_list: List of BoundaryMetrics objects
        
        Returns:
            Tuple of (feature_array, image_ids)
            feature_array shape: (N, 5) - VoL, grad_mean, grad_std, grad_max, edge_density
        """
        successful = [m for m in metrics_list if m.success]
        
        if not successful:
            return np.zeros((0, 5)), []
        
        features = np.array([
            [
                m.variance_of_laplacian,
                m.gradient_magnitude_mean,
                m.gradient_magnitude_std,
                m.gradient_magnitude_max,
                m.edge_density,
            ]
            for m in successful
        ])
        
        ids = [m.image_id for m in successful]
        
        return features, ids


# ============================================
# Quick Test
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Boundary Metrics Extractor Test")
    print("=" * 60)
    
    # Create extractor
    extractor = BoundaryMetricsExtractor()
    
    # Create test images
    print("\nCreating test images...")
    
    # Test 1: Sharp edge (good boundary)
    # Create an image with a clear edge
    sharp_image = np.zeros((256, 256), dtype=np.uint8)
    sharp_image[:, :128] = 50   # Left half dark
    sharp_image[:, 128:] = 200  # Right half bright
    
    # Mask that follows the edge
    sharp_mask = np.zeros((256, 256), dtype=np.uint8)
    sharp_mask[:, 128:] = 255
    
    # Test 2: No edge (bad boundary)
    # Create a uniform image
    uniform_image = np.ones((256, 256), dtype=np.uint8) * 128
    
    # Same mask (but no edge in image)
    uniform_mask = sharp_mask.copy()
    
    # Test 3: Noisy edge
    noisy_image = sharp_image.copy()
    # Add noise
    noise = np.random.randint(-30, 30, (256, 256), dtype=np.int16)
    noisy_image = np.clip(noisy_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Compute metrics
    print("\n--- Sharp Edge (Good Boundary) ---")
    metrics1 = extractor.compute_metrics(sharp_image, sharp_mask)
    print(f"Variance of Laplacian: {metrics1.variance_of_laplacian:.2f}")
    print(f"Gradient Mean: {metrics1.gradient_magnitude_mean:.2f}")
    print(f"Edge Density: {metrics1.edge_density:.2%}")
    print(f"Boundary Length: {metrics1.boundary_length} pixels")
    
    print("\n--- No Edge (Bad Boundary) ---")
    metrics2 = extractor.compute_metrics(uniform_image, uniform_mask)
    print(f"Variance of Laplacian: {metrics2.variance_of_laplacian:.2f}")
    print(f"Gradient Mean: {metrics2.gradient_magnitude_mean:.2f}")
    print(f"Edge Density: {metrics2.edge_density:.2%}")
    print(f"Boundary Length: {metrics2.boundary_length} pixels")
    
    print("\n--- Noisy Edge ---")
    metrics3 = extractor.compute_metrics(noisy_image, sharp_mask)
    print(f"Variance of Laplacian: {metrics3.variance_of_laplacian:.2f}")
    print(f"Gradient Mean: {metrics3.gradient_magnitude_mean:.2f}")
    print(f"Edge Density: {metrics3.edge_density:.2%}")
    print(f"Boundary Length: {metrics3.boundary_length} pixels")
    
    # Verify expected patterns
    print("\n--- Verification ---")
    
    assert metrics1.gradient_magnitude_mean > metrics2.gradient_magnitude_mean, \
        "Sharp edge should have higher gradient than uniform"
    print("✓ Sharp edge has higher gradient than uniform image")
    
    assert metrics1.edge_density > metrics2.edge_density, \
        "Sharp edge should have higher edge density than uniform"
    print("✓ Sharp edge has higher edge density than uniform image")
    
    assert metrics3.variance_of_laplacian > metrics2.variance_of_laplacian, \
        "Noisy edge should have higher VoL than uniform"
    print("✓ Noisy edge has higher VoL than uniform image")
    
    print("\n🎉 Boundary Metrics Extractor test passed!")