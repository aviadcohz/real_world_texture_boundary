import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, List, Dict
from skimage.morphology import skeletonize, binary_erosion


def parse_texture_description(description: str) -> Tuple[str, str]:
    """
    Parse texture description into two textures.

    Args:
        description: e.g., "smooth grass to rough stone wall"

    Returns:
        (texture_a, texture_b)

    Examples:
        "smooth grass to rough stone wall" → ("smooth grass", "rough stone wall")
        "wood to metal" → ("wood", "metal")
    """
    # Primary: split by " <TO> " (unambiguous separator from grounding prompt)
    if ' <TO> ' in description:
        parts = description.split(' <TO> ', 1)
        return parts[0].strip(), parts[1].strip()

    # Fallback: split by " to " (legacy format)
    parts = description.split(' to ')
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    elif len(parts) > 2:
        # Multiple " to " — pick the most balanced split
        best_split = 1
        best_balance = float('inf')
        for i in range(1, len(parts)):
            left = ' to '.join(parts[:i])
            right = ' to '.join(parts[i:])
            balance = abs(len(left.split()) - len(right.split()))
            if balance < best_balance:
                best_balance = balance
                best_split = i
        left = ' to '.join(parts[:best_split])
        right = ' to '.join(parts[best_split:])
        return left.strip(), right.strip()

    # Try other separators
    for sep in [' and ', ' vs ', ' | ']:
        if sep in description:
            parts = description.split(sep, 1)
            return parts[0].strip(), parts[1].strip()

    # Fallback: split in half
    words = description.split()
    mid = len(words) // 2
    return ' '.join(words[:mid]), ' '.join(words[mid:])


def validate_texture_description(texture_a: str, texture_b: str,
                                  min_words: int = 4, max_words: int = 12) -> Tuple[bool, str]:
    """
    Validate that texture descriptions meet length requirements for Qwen2SAM training.

    Qwen2SAM v3 expects 5-10 word descriptions (8-13 BPE tokens, max 16).
    We use a slightly wider range (4-12) to avoid discarding borderline cases.

    Args:
        texture_a: First texture description
        texture_b: Second texture description
        min_words: Minimum words per description (default 4)
        max_words: Maximum words per description (default 12)

    Returns:
        (is_valid, reason) - True if both descriptions pass, reason if not
    """
    words_a = len(texture_a.split())
    words_b = len(texture_b.split())

    if words_a < min_words:
        return False, f"texture_a too short ({words_a} words): '{texture_a}'"
    if words_b < min_words:
        return False, f"texture_b too short ({words_b} words): '{texture_b}'"
    if words_a > max_words:
        return False, f"texture_a too long ({words_a} words): '{texture_a}'"
    if words_b > max_words:
        return False, f"texture_b too long ({words_b} words): '{texture_b}'"

    return True, "OK"


def extract_morphological_boundary(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    thickness: int = 2
) -> np.ndarray:
    """
    Extract morphological boundary between two texture masks.
    
    Args:
        mask_a: First texture mask (H×W uint8)
        mask_b: Second texture mask (H×W uint8)
        thickness: Erosion kernel size
    
    Returns:
        Boundary mask (H×W uint8)
    """
    # Binarize
    binary_a = mask_a > 127
    binary_b = mask_b > 127
    
    # Erode both masks
    kernel = np.ones((thickness, thickness), np.uint8)
    eroded_a = binary_erosion(binary_a, footprint=kernel)
    eroded_b = binary_erosion(binary_b, footprint=kernel)
    
    # Find boundaries (original - eroded)
    boundary_a = np.logical_and(binary_a, ~eroded_a)
    boundary_b = np.logical_and(binary_b, ~eroded_b)
    
    # Union of boundaries
    boundary = np.logical_or(boundary_a, boundary_b)
    
    # Thin to skeleton
    boundary_thin = skeletonize(boundary)
    
    return (boundary_thin * 255).astype(np.uint8)


def extract_sa2va_boundary(
    sa2va_model,
    crop_path: Union[str, Path],
    description: str,
    boundary_thickness: int = 2
) -> np.ndarray:
    """
    Extract texture boundary using Sa2VA model.
    
    Args:
        sa2va_model: Sa2VAModel instance
        crop_path: Path to crop image
        description: Texture description (e.g., "smooth grass to rough stone")
        boundary_thickness: Thickness for morphological operations
    
    Returns:
        Boundary mask (H×W uint8)
    """
    # Parse description
    texture_a, texture_b = parse_texture_description(description)
    
    # Segment both textures
    mask_a = sa2va_model.segment_texture(crop_path, texture_a)
    mask_b = sa2va_model.segment_texture(crop_path, texture_b)
    
    # Extract boundary
    boundary = extract_morphological_boundary(
        mask_a,
        mask_b,
        thickness=boundary_thickness
    )
    
    return boundary


def extract_sa2va_boundaries_batch(
    sa2va_model,
    batch_data: List[Dict],
    boundary_thickness: int = 2,
    verbose: bool = True
) -> List[Dict]:
    """
    Extract boundaries for multiple crops.
    
    Args:
        sa2va_model: Sa2VAModel instance
        batch_data: List of dicts with:
            - 'crop_path': path to crop image
            - 'description': texture description
            - 'crop_name': crop filename (for logging)
        boundary_thickness: Thickness for morphological operations
        verbose: Print progress
    
    Returns:
        List of dicts with:
            - 'boundary': numpy array (H×W uint8)
            - 'error': error message if failed (None if success)
    """
    results = []
    
    for idx, item in enumerate(batch_data):
        crop_name = item.get('crop_name', f'crop_{idx+1}')
        
        if verbose:
            print(f"      [{idx+1}/{len(batch_data)}] {crop_name}...", end=" ")
        
        try:
            boundary = extract_sa2va_boundary(
                sa2va_model=sa2va_model,
                crop_path=item['crop_path'],
                description=item['description'],
                boundary_thickness=boundary_thickness
            )
            
            results.append({
                'boundary': boundary,
                'error': None
            })
            
            if verbose:
                boundary_pixels = (boundary > 127).sum()
                print(f"✓ ({boundary_pixels} px)")
        
        except Exception as e:
            results.append({
                'boundary': None,
                'error': str(e)
            })
            
            if verbose:
                print(f"✗ Failed: {e}")
    
    return results