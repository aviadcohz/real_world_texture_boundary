from PIL import Image
import numpy as np
from pathlib import Path
from typing import Union, Tuple


def pad_image_to_standard_size(
    image_path: Union[str, Path],
    target_size: int = 1024
) -> Tuple[Image.Image, int, int, Tuple[int, int]]:
    """
    Pad image to standard size WITHOUT saving temp file.
    Returns PIL Image object directly.
    
    Args:
        image_path: Path to original image (str or Path)
        target_size: Target width and height (default 1024)
    
    Returns:
        padded_image: PIL Image of size (target_size, target_size)
        offset_x: Padding added on left
        offset_y: Padding added on top
        original_size: (width, height) of original image
    """
    # Load original image
    image = Image.open(image_path).convert('RGB')
    original_width, original_height = image.size
    
    # If image is already larger than target, resize it first
    if original_width > target_size or original_height > target_size:
        # Calculate scaling factor to fit within target_size
        scale = min(target_size / original_width, target_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        original_width, original_height = new_width, new_height
    
    # Calculate padding needed
    pad_width = target_size - original_width
    pad_height = target_size - original_height
    
    # Center the image (padding equally on both sides)
    offset_x = pad_width // 2
    offset_y = pad_height // 2
    
    # Create black background of target size
    padded_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    
    # Paste original image in center
    padded_image.paste(image, (offset_x, offset_y))
    
    return padded_image, offset_x, offset_y, (original_width, original_height)


def calculate_optimal_target_size(
    image_paths: list,
    percentile: int = 95
) -> int:
    """
    Calculate optimal target size based on dataset statistics.
    
    Args:
        image_paths: List of image paths
        percentile: Use this percentile of max dimension (default 95)
    
    Returns:
        target_size: Recommended target size (rounded to nearest 64)
    """
    max_dims = []
    
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            max_dim = max(img.size)
            max_dims.append(max_dim)
            img.close()
        except Exception as e:
            print(f"⚠️  Warning: Could not open {img_path}: {e}")
            continue
    
    if not max_dims:
        print("⚠️  Warning: No valid images found, using default size 1024")
        return 1024  # Default
    
    # Use percentile to avoid outliers
    target = np.percentile(max_dims, percentile)
    
    # Round up to nearest multiple of 64 (good for CNNs)
    target_size = int(np.ceil(target / 64) * 64)
    
    return target_size


def resize_image(
    image: Union[str, Path, Image.Image],
    target_size: Tuple[int, int],
    method: str = 'lanczos'
) -> Image.Image:
    """
    Resize image to target size.
    
    Args:
        image: Image path or PIL Image
        target_size: (width, height) tuple
        method: Resampling method ('lanczos', 'bilinear', 'nearest')
    
    Returns:
        Resized PIL Image
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert('RGB')
    
    # Map method names to PIL constants
    methods = {
        'lanczos': Image.Resampling.LANCZOS,
        'bilinear': Image.Resampling.BILINEAR,
        'nearest': Image.Resampling.NEAREST,
        'bicubic': Image.Resampling.BICUBIC
    }
    
    resample_method = methods.get(method.lower(), Image.Resampling.LANCZOS)
    
    return image.resize(target_size, resample_method)


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Load image and convert to RGB.
    
    Args:
        image_path: Path to image
    
    Returns:
        PIL Image in RGB mode
    """
    return Image.open(image_path).convert('RGB')


def get_image_size(image_path: Union[str, Path]) -> Tuple[int, int]:
    """
    Get image dimensions without loading full image.
    
    Args:
        image_path: Path to image
    
    Returns:
        (width, height) tuple
    """
    with Image.open(image_path) as img:
        return img.size


def save_image(
    image: Image.Image,
    output_path: Union[str, Path],
    quality: int = 95
) -> None:
    """
    Save PIL Image to file.
    
    Args:
        image: PIL Image
        output_path: Output file path
        quality: JPEG quality (1-100)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine format from extension
    ext = output_path.suffix.lower()
    if ext in ['.jpg', '.jpeg']:
        image.save(output_path, 'JPEG', quality=quality)
    elif ext == '.png':
        image.save(output_path, 'PNG')
    else:
        image.save(output_path, quality=quality)


def crop_image(
    image: Union[str, Path, Image.Image],
    bbox: Tuple[int, int, int, int]
) -> Image.Image:
    """
    Crop image using bounding box.
    
    Args:
        image: Image path or PIL Image
        bbox: (x1, y1, x2, y2) coordinates
    
    Returns:
        Cropped PIL Image
    """
    if isinstance(image, (str, Path)):
        image = load_image(image)
    
    return image.crop(bbox)