"""
Multi-Stage Mask Refinement Pipeline

Refines segmentation masks by stabilizing semantic mass before boundary extraction.

Pipeline stages:
0. Pre-cleaning (keep only largest component)
1. Initial Structural Cleaning (fill holes + remove noise)
2. Morphological Consolidation (closing)
3. Boundary Liquefaction (Gaussian blur + re-threshold)
4. Semantic Merge (bitwise XOR for boundary extraction)
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology
from typing import Tuple


class MaskRefinementPipeline:
    """
    Multi-stage pipeline for refining noisy segmentation masks.
    """
    
    def __init__(self, 
                 min_object_size: int = 100,
                 closing_kernel_size: int = 7,
                 gaussian_sigma: float = 3.0,
                 rethreshold_value: int = 127):
        """
        Initialize the refinement pipeline.
        
        Args:
            min_object_size: Minimum area (pixels) to keep an object (default: 100)
            closing_kernel_size: Size of elliptical kernel for morphological closing (default: 7)
            gaussian_sigma: Sigma for Gaussian blur (default: 3.0)
            rethreshold_value: Threshold value after Gaussian blur (default: 127 = 0.5)
        """
        self.min_object_size = min_object_size
        self.closing_kernel_size = closing_kernel_size
        self.gaussian_sigma = gaussian_sigma
        self.rethreshold_value = rethreshold_value
        
    def refine_single_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply the complete refinement pipeline to a single mask.
        
        Args:
            mask: Binary segmentation mask (0s and 255s)
            
        Returns:
            Refined binary mask (0s and 255s)
        """
        # Ensure binary mask (0s and 1s for processing)
        original = (mask > 127).astype(bool)
        
        # Stage 0: Pre-cleaning - Keep only largest connected component
        precleaned = self._stage0_keep_largest_component(original)
        
        # Stage 1: Initial Structural Cleaning
        cleaned = self._stage1_structural_cleaning(precleaned)
        
        # Stage 2: Morphological Consolidation
        consolidated = self._stage2_morphological_consolidation(cleaned)
        
        # Stage 3: Boundary Liquefaction
        liquefied = self._stage3_boundary_liquefaction(consolidated)
        
        return (liquefied * 255).astype(np.uint8)
    
    def _stage0_keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """
        Stage 0: Keep only the largest connected component (major structure).
        
        This removes all disconnected noise blobs and keeps only the main object.
        Critical to do BEFORE filling holes - otherwise we'd fill holes in noise too!
        
        Args:
            mask: Binary mask (boolean array)
            
        Returns:
            Binary mask containing only the largest connected component
        """
        # Label all connected components
        labeled = morphology.label(mask, connectivity=2)
        
        # If no components found, return empty mask
        if labeled.max() == 0:
            return np.zeros_like(mask, dtype=bool)
        
        # Count pixels in each component
        component_sizes = np.bincount(labeled.ravel())
        
        # Background is label 0, so start from 1
        component_sizes[0] = 0
        
        # Find the largest component
        largest_component_label = component_sizes.argmax()
        
        # Keep only the largest component
        largest_component = labeled == largest_component_label
        
        return largest_component
    
    def _stage1_structural_cleaning(self, mask: np.ndarray) -> np.ndarray:
        """
        Stage 1: Fill internal holes and remove small noise objects.
        
        This prevents holes from becoming artifacts during smoothing and removes
        isolated pixels that don't represent true semantic regions.
        
        Args:
            mask: Binary mask (boolean array)
            
        Returns:
            Cleaned binary mask
        """
        # Fill internal holes
        filled = ndimage.binary_fill_holes(mask)
        
        # Remove small isolated objects (noise islands)
        # Using connectivity=2 (8-connectivity) for consistent object detection
        # max_size parameter: removes objects with this many pixels or fewer
        cleaned = morphology.remove_small_objects(
            filled, 
            max_size=self.min_object_size,
            connectivity=2
        )
        
        return cleaned
    
    def _stage2_morphological_consolidation(self, mask: np.ndarray) -> np.ndarray:
        """
        Stage 2: Reconnect fragmented components using morphological closing.
        
        Uses elliptical kernel to avoid introducing blocky corners while
        connecting nearby mask regions that belong to the same semantic object.
        
        Args:
            mask: Cleaned binary mask
            
        Returns:
            Consolidated binary mask
        """
        # Create elliptical structuring element
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.closing_kernel_size, self.closing_kernel_size)
        )
        
        # Convert to uint8 for OpenCV
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Morphological closing: dilation followed by erosion
        closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        return (closed > 127).astype(bool)
    
    def _stage3_boundary_liquefaction(self, mask: np.ndarray) -> np.ndarray:
        """
        Stage 3: Smooth jagged boundaries using Gaussian blur + re-thresholding.
        
        The Gaussian blur creates a probability map where noisy edge pixels
        are averaged out. Re-thresholding at 0.5 finds the geometric mean
        of the noisy boundary, producing a single smooth curve.
        
        Args:
            mask: Consolidated binary mask
            
        Returns:
            Mask with smooth boundaries
        """
        # Convert to uint8 for Gaussian blur
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Apply Gaussian blur
        kernel_size = int(6 * self.gaussian_sigma) | 1  # Ensure odd kernel size
        blurred = cv2.GaussianBlur(
            mask_uint8, 
            (kernel_size, kernel_size), 
            self.gaussian_sigma
        )
        
        # Re-threshold at 0.5 (127 for uint8)
        _, smoothed = cv2.threshold(blurred, self.rethreshold_value, 255, cv2.THRESH_BINARY)
        
        return (smoothed > 127).astype(bool)
    
    def refine_mask_pair_and_xor(
        self, 
        mask_a: np.ndarray, 
        mask_b: np.ndarray,
        extract_thin_boundary: bool = True,
        boundary_method: str = 'morphological_gradient'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Refine a pair of masks and compute XOR boundary.
        
        Args:
            mask_a: First binary mask
            mask_b: Second binary mask
            extract_thin_boundary: Whether to extract thin boundary from XOR region
            boundary_method: Method for thin boundary extraction:
                - 'morphological_gradient': dilation - erosion (default)
                - 'skeleton': skeletonize the XOR region
                - 'none': return raw XOR region
            
        Returns:
            Tuple of (refined_mask_a, refined_mask_b, boundary)
            All as uint8 arrays (0s and 255s)
        """
        # Refine each mask individually
        refined_a = self.refine_single_mask(mask_a)
        refined_b = self.refine_single_mask(mask_b)
        
        # Compute XOR boundary region between refined masks
        xor_region = cv2.bitwise_xor(refined_a, refined_b)
        
        if not extract_thin_boundary or boundary_method == 'none':
            return refined_a, refined_b, xor_region
        
        # Extract thin boundary from XOR region
        boundary = self._extract_thin_boundary(xor_region, method=boundary_method)
        
        return refined_a, refined_b, boundary
    
    def _extract_thin_boundary(
        self, 
        region_mask: np.ndarray, 
        method: str = 'morphological_gradient'
    ) -> np.ndarray:
        """
        Extract thin boundary line from a region mask.
        
        Args:
            region_mask: Binary region mask (0s and 255s)
            method: Extraction method:
                - 'morphological_gradient': dilation - erosion
                - 'skeleton': skeletonize
        
        Returns:
            Thin boundary mask (0s and 255s)
        """
        if method == 'morphological_gradient':
            # Morphological gradient: dilation - erosion
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated = cv2.dilate(region_mask, kernel, iterations=1)
            eroded = cv2.erode(region_mask, kernel, iterations=1)
            gradient = cv2.subtract(dilated, eroded)
            return gradient
        
        elif method == 'skeleton':
            # Skeletonize the region to get thin centerline
            from skimage.morphology import skeletonize
            binary = region_mask > 127
            skeleton = skeletonize(binary)
            return (skeleton * 255).astype(np.uint8)
        
        else:
            # Return as-is
            return region_mask


def refine_masks_and_extract_boundary(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    min_object_size: int = 100,
    closing_kernel_size: int = 7,
    gaussian_sigma: float = 3.0,
    extract_thin_boundary: bool = True,
    boundary_method: str = 'morphological_gradient'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to refine mask pair and extract boundary.
    
    Args:
        mask_a: First texture mask (H×W uint8)
        mask_b: Second texture mask (H×W uint8)
        min_object_size: Minimum object size to keep
        closing_kernel_size: Kernel size for morphological closing
        gaussian_sigma: Sigma for Gaussian blur smoothing
        extract_thin_boundary: Whether to extract thin boundary from XOR region
        boundary_method: Method for thin boundary extraction:
            - 'morphological_gradient': dilation - erosion (default)
            - 'skeleton': skeletonize the XOR region
            - 'none': return raw XOR region
    
    Returns:
        Tuple of (refined_mask_a, refined_mask_b, boundary)
        All as uint8 arrays (0s and 255s)
    """
    pipeline = MaskRefinementPipeline(
        min_object_size=min_object_size,
        closing_kernel_size=closing_kernel_size,
        gaussian_sigma=gaussian_sigma
    )
    
    return pipeline.refine_mask_pair_and_xor(
        mask_a, mask_b,
        extract_thin_boundary=extract_thin_boundary,
        boundary_method=boundary_method
    )
