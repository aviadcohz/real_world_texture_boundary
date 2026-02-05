from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from PIL import Image
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from core.sa2va_boundaries import extract_sa2va_boundary
from utils.image_utils import load_image, resize_image, save_image


# Try to import parallel utilities
try:
    from utils.parallel import parallel_save_crops, ParallelProcessor
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False


@dataclass
class CropTask:
    """Container for a crop extraction task."""
    crop: Image.Image
    path: Path
    quality: int
    bbox_id: int
    category: str
    crop_name: str
    description: str = None
    was_fixed: bool = False


def get_size_category(width: int, height: int) -> Tuple[str, Tuple[int, int]]:
    """
    Categorize bbox by size.

    Args:
        width: Bbox width
        height: Bbox height

    Returns:
        category: Category name ('tiny', 'small', 'medium', 'large', 'xlarge')
        target_size: Target resize dimensions
    """
    area = width * height
    max_dim = max(width, height)

    # Define size categories
    if max_dim < 64 or area < 2000:
        return "tiny", (64, 64)
    elif max_dim < 128 or area < 8000:
        return "small", (128, 128)
    elif max_dim < 256 or area < 32000:
        return "medium", (256, 256)
    elif max_dim < 512 or area < 128000:
        return "large", (512, 512)
    else:
        return "xlarge", (1024, 1024)


def extract_crop(
    image: Union[str, Path, Image.Image],
    bbox: Tuple[int, int, int, int],
    target_size: Tuple[int, int] = None,
    resize_method: str = 'lanczos'
) -> Image.Image:
    """
    Extract and optionally resize a crop from image.

    Args:
        image: Image path or PIL Image
        bbox: (x1, y1, x2, y2) coordinates
        target_size: Optional (width, height) to resize to
        resize_method: Resize method

    Returns:
        Cropped (and resized) PIL Image
    """
    # Load image if path
    if isinstance(image, (str, Path)):
        image = load_image(image)

    # Extract crop
    x1, y1, x2, y2 = bbox
    crop = image.crop((x1, y1, x2, y2))

    # Resize if requested
    if target_size is not None:
        crop = resize_image(crop, target_size, method=resize_method)

    return crop


def _save_single_crop(task: CropTask) -> Dict:
    """Save a single crop and return info dict."""
    task.path.parent.mkdir(parents=True, exist_ok=True)
    task.crop.save(task.path, quality=task.quality)

    return {
        'bbox_id': task.bbox_id,
        'crop_name': task.crop_name,
        'crop_path': str(task.path),
        'category': task.category,
    }


def extract_crops_from_image(
    image_path: Union[str, Path],
    bboxes: List[Dict],
    output_dir: Union[str, Path],
    sa2va_model=None,
    resize_by_category: bool = True,
    extract_boundaries: bool = True,
    boundary_thickness: int = 2,
    parallel_save: bool = True,
    save_workers: int = 8,
    verbose: bool = True
) -> Tuple[Dict[str, int], List[Dict]]:
    """
    Extract all crops from a single image.
    Optionally also extract boundaries masks.

    Args:
        image_path: Path to image
        bboxes: List of bbox dicts with 'coords'
        output_dir: Output directory for crops
        sa2va_model: Sa2VA model for boundary extraction
        resize_by_category: Resize crops by size category
        extract_boundaries: Extract boundaries masks
        boundary_thickness: Thickness of boundary lines
        parallel_save: Use parallel saving (faster for many crops)
        save_workers: Number of parallel workers for saving
        verbose: Print progress

    Returns:
        Tuple of (category_counts, crops_info)
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    # Load image once
    image = load_image(image_path)
    image_basename = image_path.stem

    # Track counts and crop info
    category_counts = {}
    crops_info = []
    crop_tasks = []

    # First pass: extract all crops to memory
    for bbox_dict in bboxes:
        box_id = bbox_dict['id']
        bbox = bbox_dict['coords']
        description = bbox_dict.get('description', f'box_{box_id}')
        was_fixed = bbox_dict.get('was_fixed', False)

        # Get size category
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        category, target_size = get_size_category(w, h)

        # Create category directories
        crop_category_dir = output_dir / category
        crop_category_dir.mkdir(parents=True, exist_ok=True)

        # Create safe filename
        x1, y1, x2, y2 = bbox
        crop_filename = f"{image_basename}_{x1}_{y1}_{x2}_{y2}.jpg"
        if was_fixed:
            crop_filename = f"{image_basename}_{x1}_{y1}_{x2}_{y2}_FIXED.jpg"

        crop_path = crop_category_dir / crop_filename

        # Extract crop
        if resize_by_category:
            crop = extract_crop(image, bbox, target_size=target_size)
        else:
            crop = extract_crop(image, bbox, target_size=None)
            target_size = (w, h)

        # Create task
        task = CropTask(
            crop=crop,
            path=crop_path,
            quality=95,
            bbox_id=box_id,
            category=category,
            crop_name=crop_filename,
            description=description,
            was_fixed=was_fixed
        )
        crop_tasks.append(task)

        # Update counts
        category_counts[category] = category_counts.get(category, 0) + 1

        if verbose:
            status = f"-> {category}/{crop_filename}"
            if was_fixed:
                status += " (fixed)"
            print(f"      Box {box_id}: {w:3}x{h:3}px -> {target_size[0]}x{target_size[1]}px {status}")

    # Second pass: save all crops (parallel or sequential)
    if parallel_save and PARALLEL_AVAILABLE and len(crop_tasks) > 1:
        # Parallel saving
        with ThreadPoolExecutor(max_workers=save_workers) as executor:
            save_results = list(executor.map(_save_single_crop, crop_tasks))
    else:
        # Sequential saving
        save_results = [_save_single_crop(task) for task in crop_tasks]

    # Build crops_info with saved paths
    task_lookup = {task.bbox_id: task for task in crop_tasks}

    for result in save_results:
        task = task_lookup[result['bbox_id']]
        crop_info = {
            'bbox_id': result['bbox_id'],
            'crop_name': result['crop_name'],
            'crop_path': result['crop_path'],
            'category': result['category'],
            'crop_size': task.crop.size
        }

        # Extract boundaries if requested (must be sequential due to GPU)
        boundaries_path = None
        if extract_boundaries and sa2va_model is not None and task.description:
            masks_dir = output_dir.parent / "masks"
            mask_category_dir = masks_dir / task.category
            mask_category_dir.mkdir(parents=True, exist_ok=True)

            boundaries_filename = task.crop_name.replace('.jpg', '.png')
            boundaries_path = mask_category_dir / boundaries_filename

            try:
                boundaries = extract_sa2va_boundary(
                    sa2va_model=sa2va_model,
                    crop_path=task.path,
                    description=task.description,
                    boundary_thickness=boundary_thickness
                )
                Image.fromarray(boundaries).save(boundaries_path)

                if verbose:
                    boundary_pixels = (boundaries > 127).sum()
                    print(f"         + boundaries: {boundary_pixels} px")

            except Exception as e:
                if verbose:
                    print(f"         Warning: Boundary extraction failed: {e}")
                boundaries_path = None

        if boundaries_path is not None:
            crop_info['boundaries_mask_path'] = str(boundaries_path)
            crop_info['boundaries_mask_name'] = boundaries_filename

        crops_info.append(crop_info)

    return category_counts, crops_info


def extract_all_crops(
    images_data: List[Dict],
    output_dir: Union[str, Path],
    sa2va_model=None,
    resize_by_category: bool = True,
    extract_boundaries: bool = True,
    boundary_thickness: int = 2,
    parallel_save: bool = True,
    save_workers: int = 8,
    verbose: bool = True
) -> Tuple[Dict, List[Dict]]:
    """
    Extract crops from all images.
    Optionally also extract boundaries masks.

    Args:
        images_data: List of dicts with 'image_path' and 'boxes'
        output_dir: Output directory for crops
        sa2va_model: Sa2VAModel instance for boundary extraction
        resize_by_category: Resize crops by size category
        extract_boundaries: Extract boundaries masks
        boundary_thickness: Thickness of boundary lines
        parallel_save: Use parallel crop saving
        save_workers: Number of parallel workers
        verbose: Print progress

    Returns:
        Tuple of (summary, crops_mapping)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*70}")
        print(f"EXTRACTING CROPS")
        if parallel_save and PARALLEL_AVAILABLE:
            print(f"  (parallel saving enabled, {save_workers} workers)")
        if extract_boundaries and sa2va_model:
            print(f"+ Boundaries masks")
        print(f"Output: {output_dir}")
        print(f"{'='*70}\n")

    total_crops = 0
    total_fixed = 0
    total_boundaries = 0
    global_category_counts = {}
    crops_mapping = []

    for img_data in images_data:
        image_name = img_data['image']
        image_path = Path(img_data['image_path'])
        bboxes = img_data['boxes']

        if verbose:
            print(f"  {image_name}")
            print(f"     {len(bboxes)} boxes")

        if not image_path.exists():
            if verbose:
                print(f"     Warning: Image not found: {image_path}")
                print()
            continue

        # Extract crops (and boundaries if requested)
        category_counts, crops_info = extract_crops_from_image(
            image_path=image_path,
            bboxes=bboxes,
            output_dir=output_dir,
            sa2va_model=sa2va_model,
            resize_by_category=resize_by_category,
            extract_boundaries=extract_boundaries,
            boundary_thickness=boundary_thickness,
            parallel_save=parallel_save,
            save_workers=save_workers,
            verbose=verbose
        )

        # Save mapping
        crops_mapping.append({
            'image': image_name,
            'crops_info': crops_info
        })

        # Update totals
        total_crops += len(bboxes)
        total_fixed += sum(1 for b in bboxes if b.get('was_fixed', False))

        # Count boundaries
        total_boundaries += sum(1 for c in crops_info if 'boundaries_mask_path' in c)

        for cat, count in category_counts.items():
            global_category_counts[cat] = global_category_counts.get(cat, 0) + count

        if verbose:
            print()

    # Summary
    summary = {
        'total_crops': total_crops,
        'total_fixed': total_fixed,
        'total_boundaries': total_boundaries,
        'category_counts': global_category_counts,
        'output_dir': str(output_dir)
    }

    if verbose:
        print("="*70)
        print("COMPLETE!")
        print("="*70)
        print(f"  Total crops: {total_crops}")
        if total_fixed > 0:
            print(f"  Fixed boxes: {total_fixed}")
        if total_boundaries > 0:
            print(f"  Boundaries masks: {total_boundaries}")
        print(f"\n  Crops by size:")
        for cat, count in sorted(global_category_counts.items()):
            print(f"    {cat:8}: {count:4} crops")
        print(f"\n  Output dirs:")
        print(f"    Crops: {output_dir.absolute()}")
        if total_boundaries > 0:
            masks_dir = output_dir.parent / "masks"
            print(f"    Masks: {masks_dir.absolute()}")
        print("="*70 + "\n")

    return summary, crops_mapping


def extract_all_crops_parallel(
    images_data: List[Dict],
    output_dir: Union[str, Path],
    sa2va_model=None,
    resize_by_category: bool = True,
    extract_boundaries: bool = True,
    boundary_thickness: int = 2,
    num_workers: int = 4,
    verbose: bool = True
) -> Tuple[Dict, List[Dict]]:
    """
    Extract crops from all images with parallel image processing.

    Uses parallel workers to process multiple images simultaneously.
    Note: Boundary extraction still runs sequentially on GPU.

    Args:
        images_data: List of dicts with 'image_path' and 'boxes'
        output_dir: Output directory for crops
        sa2va_model: Sa2VAModel instance for boundary extraction
        resize_by_category: Resize crops by size category
        extract_boundaries: Extract boundaries masks
        boundary_thickness: Thickness of boundary lines
        num_workers: Number of parallel workers for image processing
        verbose: Print progress

    Returns:
        Tuple of (summary, crops_mapping)
    """
    # For now, delegate to the standard function with parallel save enabled
    # True parallel image processing would require more refactoring
    return extract_all_crops(
        images_data=images_data,
        output_dir=output_dir,
        sa2va_model=sa2va_model,
        resize_by_category=resize_by_category,
        extract_boundaries=extract_boundaries,
        boundary_thickness=boundary_thickness,
        parallel_save=True,
        save_workers=num_workers * 2,
        verbose=verbose
    )


def update_processed_bboxes_with_crops(
    processed_bboxes_path: Union[str, Path],
    crops_mapping: List[Dict],
    output_path: Union[str, Path] = None
) -> None:
    """
    Update processed_bboxes.json with crop_name and crop_path for each bbox.

    Args:
        processed_bboxes_path: Path to processed_bboxes.json
        crops_mapping: List from extract_all_crops with crop info
        output_path: Output path (None = overwrite original)
    """
    import json

    processed_bboxes_path = Path(processed_bboxes_path)

    # Load original JSON
    with open(processed_bboxes_path, 'r') as f:
        data = json.load(f)

    print(f"\n{'='*70}")
    print(f"UPDATING processed_bboxes.json WITH CROP INFO")
    print(f"{'='*70}\n")

    # Create mapping dict for quick lookup
    # Format: {image_name: {bbox_id: crop_info}}
    crop_lookup = {}
    for item in crops_mapping:
        image_name = item['image']
        crop_lookup[image_name] = {}
        for crop_info in item['crops_info']:
            bbox_id = crop_info['bbox_id']
            crop_lookup[image_name][bbox_id] = crop_info

    # Update each bbox in data
    updated_count = 0
    for img_data in data:
        image_name = img_data['image']

        if image_name not in crop_lookup:
            print(f"Warning: No crops found for: {image_name}")
            continue

        for bbox in img_data['boxes']:
            bbox_id = bbox['id']

            if bbox_id in crop_lookup[image_name]:
                crop_info = crop_lookup[image_name][bbox_id]

                # Add crop info to bbox
                bbox['crop_name'] = crop_info['crop_name']
                bbox['crop_path'] = crop_info['crop_path']
                bbox['crop_category'] = crop_info['category']

                updated_count += 1
                print(f"  {image_name} bbox#{bbox_id}: {crop_info['crop_name']}")
            else:
                print(f"Warning: {image_name} bbox#{bbox_id}: crop not found")

    # Save updated JSON
    if output_path is None:
        output_path = processed_bboxes_path
    else:
        output_path = Path(output_path)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"UPDATED {updated_count} bboxes")
    print(f"Saved to: {output_path}")
    print(f"{'='*70}\n")



def get_crops_by_category(
    crops_dir: Union[str, Path],
    categories: List[str] = None
) -> Dict[str, List[Path]]:
    """
    Get all crop files organized by category.

    Args:
        crops_dir: Root crops directory
        categories: List of categories to include (None = all)

    Returns:
        Dict mapping category -> list of crop paths
    """
    crops_dir = Path(crops_dir)

    if categories is None:
        categories = ['tiny', 'small', 'medium', 'large', 'xlarge']

    crops_by_category = {}

    for category in categories:
        category_dir = crops_dir / category
        if category_dir.exists():
            crops = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
            crops_by_category[category] = sorted(crops)
        else:
            crops_by_category[category] = []

    return crops_by_category
