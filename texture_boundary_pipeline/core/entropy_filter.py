"""
Entropy-based filtering for texture boundary crops.

Filters crops based on texture entropy - high entropy regions indicate
actual textures, low entropy indicates uniform/semantic regions.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Dict, List


def compute_grayscale_entropy(pixels: np.ndarray) -> float:
    """
    Compute Shannon entropy of grayscale pixel values.

    Args:
        pixels: 1D array of grayscale pixel values (0-255)

    Returns:
        Entropy value (0-8 range, higher = more texture)
    """
    if len(pixels) == 0:
        return 0.0

    # Compute histogram (256 bins for 0-255)
    hist, _ = np.histogram(pixels, bins=256, range=(0, 256))

    # Normalize to probability distribution
    hist = hist / hist.sum()

    # Remove zeros to avoid log(0)
    hist = hist[hist > 0]

    # Shannon entropy: H = -sum(p * log2(p))
    entropy = -np.sum(hist * np.log2(hist))

    return float(entropy)  # Convert to Python float for JSON serialization


def compute_region_entropy(
    image: Union[str, Path, Image.Image, np.ndarray],
    mask: np.ndarray
) -> float:
    """
    Compute entropy of image pixels within a mask region.

    Args:
        image: RGB image (path, PIL Image, or numpy array)
        mask: Binary mask (H×W, values 0 or 255)

    Returns:
        Entropy of the masked region (Python float)
    """
    # Load image if needed
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert('RGB')
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2).astype(np.uint8)
    else:
        gray = image

    # Ensure mask is binary
    binary_mask = mask > 127

    # Get pixels within mask
    pixels = gray[binary_mask]

    if len(pixels) == 0:
        return 0.0

    return float(compute_grayscale_entropy(pixels))


def filter_by_entropy(
    crop_path: Union[str, Path],
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    threshold: float = 4.5
) -> Tuple[bool, Dict]:
    """
    Filter a crop based on entropy of both texture regions.

    Args:
        crop_path: Path to crop image
        mask_a: Mask for texture A
        mask_b: Mask for texture B
        threshold: Minimum entropy for both regions (default 4.5)

    Returns:
        Tuple of (passed, info_dict)
        - passed: True if both regions have entropy >= threshold
        - info_dict: Contains entropy values and pass/fail status
    """
    # Load crop
    crop = Image.open(crop_path).convert('RGB')
    crop_array = np.array(crop)

    # Compute entropy for each region
    entropy_a = compute_region_entropy(crop_array, mask_a)
    entropy_b = compute_region_entropy(crop_array, mask_b)

    # Both must exceed threshold
    passed = (entropy_a >= threshold) and (entropy_b >= threshold)

    info = {
        'crop_name': Path(crop_path).name,
        'crop_path': str(crop_path),
        'entropy_a': round(entropy_a, 3),
        'entropy_b': round(entropy_b, 3),
        'min_entropy': round(min(entropy_a, entropy_b), 3),
        'threshold': threshold,
        'passed': passed
    }

    return passed, info


def batch_filter_by_entropy(
    filter_data: List[Dict],
    threshold: float = 4.5,
    verbose: bool = True
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Filter multiple crops by entropy.

    Args:
        filter_data: List of dicts with:
            - 'crop_path': path to crop
            - 'mask_a': numpy array for texture A
            - 'mask_b': numpy array for texture B
        threshold: Minimum entropy threshold
        verbose: Print progress

    Returns:
        Tuple of (passed_list, failed_list, summary)
    """
    passed_list = []
    failed_list = []

    for idx, item in enumerate(filter_data, 1):
        crop_name = Path(item['crop_path']).name

        if verbose:
            print(f"      [{idx}/{len(filter_data)}] {crop_name}...", end=" ")

        passed, info = filter_by_entropy(
            crop_path=item['crop_path'],
            mask_a=item['mask_a'],
            mask_b=item['mask_b'],
            threshold=threshold
        )

        if passed:
            passed_list.append(info)
            if verbose:
                print(f"✓ PASS (H={info['min_entropy']:.2f})")
        else:
            failed_list.append(info)
            if verbose:
                print(f"✗ FAIL (H={info['min_entropy']:.2f} < {threshold})")

    summary = {
        'total': len(filter_data),
        'passed': len(passed_list),
        'failed': len(failed_list),
        'threshold': threshold,
        'pass_rate': round(len(passed_list) / len(filter_data) * 100, 1) if filter_data else 0
    }

    return passed_list, failed_list, summary
