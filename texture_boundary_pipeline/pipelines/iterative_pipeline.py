from pathlib import Path
from typing import List, Dict, Union
from datetime import datetime
import torch
import json
import shutil
from PIL import Image

from models import BaseVLM
from core import (
    parse_texture_description,
    extract_morphological_boundary,
    compute_region_entropy,
    refine_masks_and_extract_boundary,
)
from utils import (
    list_images,
    get_prompt,
    save_json,
)
from .basic_pipeline import BasicPipeline


class IterativePipeline:
    """
    Texture boundary detection pipeline with entropy-based filtering.

    Workflow:
    1. Run basic pipeline (grounding + crop extraction)
    2. Unload Qwen model
    3. Load Sa2VA, extract masks, and filter by entropy
    """

    def __init__(
        self,
        model: BaseVLM,
        output_dir: Union[str, Path] = "results",
        extract_masks: bool = True,
        entropy_threshold: float = 4.5,
        boundary_thickness: int = 2,
        collect_dataset: bool = True,
        dataset_base_dir: Union[str, Path] = "/datasets/ade20k",
        # Mask refinement parameters
        refine_masks: bool = True,
        refinement_min_object_size: int = 100,
        refinement_closing_kernel_size: int = 7,
        refinement_gaussian_sigma: float = 3.0,
        **basic_pipeline_kwargs
    ):
        """
        Initialize pipeline.

        Args:
            model: VLM model instance
            output_dir: Base output directory
            extract_masks: Whether to extract masks using Sa2VA
            entropy_threshold: Minimum entropy for both textures (default 4.5)
            boundary_thickness: Thickness for boundary extraction
            collect_dataset: Whether to collect passed crops into a dataset
            dataset_base_dir: Base directory for dataset collection
            refine_masks: Whether to apply mask refinement before boundary extraction
            refinement_min_object_size: Min object size for refinement (default 100)
            refinement_closing_kernel_size: Kernel size for morphological closing (default 7)
            refinement_gaussian_sigma: Sigma for Gaussian blur smoothing (default 3.0)
            **basic_pipeline_kwargs: Arguments for BasicPipeline
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.extract_masks = extract_masks
        self.entropy_threshold = entropy_threshold
        self.sa2va_model = None
        self.boundary_thickness = boundary_thickness
        self.collect_dataset = collect_dataset
        self.dataset_base_dir = Path(dataset_base_dir)
        
        # Mask refinement settings
        self.refine_masks = refine_masks
        self.refinement_min_object_size = refinement_min_object_size
        self.refinement_closing_kernel_size = refinement_closing_kernel_size
        self.refinement_gaussian_sigma = refinement_gaussian_sigma

        # Create basic pipeline - WITHOUT mask extraction
        self.basic_pipeline = BasicPipeline(
            model=model,
            output_dir=output_dir,
            sa2va_model=None,
            extract_boundaries=False,
            **basic_pipeline_kwargs
        )

        self.verbose = basic_pipeline_kwargs.get('verbose', True)

    def unload_vlm_model(self):
        """Unload Qwen model from GPU to free VRAM."""
        if self.verbose:
            print("\n   🔄 Unloading VLM model from GPU...")

        try:
            del self.model._model
            del self.model._processor
            self.model._model = None
            self.model._processor = None

            import gc
            gc.collect()
            torch.cuda.empty_cache()

            if self.verbose:
                print("   ✅ VLM model deleted, CUDA cache cleared")
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  Could not unload model: {e}")

    def load_sa2va_model(self):
        """Load Sa2VA model after Qwen is unloaded."""
        if self.verbose:
            print("\n   🔄 Loading Sa2VA model...")

        try:
            from models import Sa2VAModel
            self.sa2va_model = Sa2VAModel(device='cuda')
            if self.verbose:
                print("   ✅ Sa2VA model loaded successfully")
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  Could not load Sa2VA model: {e}")
            self.sa2va_model = None

    def extract_masks_and_filter(
        self,
        crops_dir: Path,
        processed_data: List[Dict],
        masks_dir: Path,
        filter_dir: Path
    ) -> Dict:
        """
        Extract masks for all crops and filter by entropy.

        Args:
            crops_dir: Directory containing crops
            processed_data: Processed bbox data with descriptions
            masks_dir: Output directory for masks
            filter_dir: Output directory for filter results

        Returns:
            Dict with extraction and filtering stats
        """
        if self.sa2va_model is None:
            if self.verbose:
                print("   ⚠️  No Sa2VA model provided, skipping")
            return {'total': 0, 'passed': 0, 'failed': 0}

        # Build lookup: crop_name -> description
        description_lookup = {}
        for img_data in processed_data:
            for box in img_data['boxes']:
                if 'crop_name' in box and 'description' in box:
                    description_lookup[box['crop_name']] = box['description']

        # Get all crops
        all_crops = []
        for category in ['tiny', 'small', 'medium', 'large', 'xlarge']:
            category_dir = crops_dir / category
            if category_dir.exists():
                crops = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
                all_crops.extend(crops)

        if self.verbose:
            print(f"\n   📦 Found {len(all_crops)} crops to process")
            print(f"   🎯 Entropy threshold: {self.entropy_threshold}")

        # Results tracking
        filter_results = []
        passed_crops = []
        failed_crops = []

        for idx, crop_path in enumerate(all_crops, 1):
            crop_name = crop_path.name
            category = crop_path.parent.name

            description = description_lookup.get(crop_name)
            if not description:
                if self.verbose:
                    print(f"      [{idx}/{len(all_crops)}] {crop_name} - ⚠️  No description, skipping")
                continue

            if self.verbose:
                print(f"      [{idx}/{len(all_crops)}] {crop_name}...", end=" ")

            try:
                # Parse description into two textures
                texture_a, texture_b = parse_texture_description(description)

                # Load crop image
                crop_image = Image.open(crop_path).convert('RGB')

                # Segment both textures
                mask_a = self.sa2va_model.segment_texture(crop_path, texture_a)
                mask_b = self.sa2va_model.segment_texture(crop_path, texture_b)

                # Compute entropy for each region
                entropy_a = compute_region_entropy(crop_image, mask_a)
                entropy_b = compute_region_entropy(crop_image, mask_b)
                min_entropy = min(entropy_a, entropy_b)

                # Check if both pass threshold
                passed = bool((entropy_a >= self.entropy_threshold) and (entropy_b >= self.entropy_threshold))

                # Extract boundary mask - with optional mask refinement
                if self.refine_masks:
                    # Refine both masks and extract XOR boundary
                    refined_a, refined_b, boundary = refine_masks_and_extract_boundary(
                        mask_a, mask_b,
                        min_object_size=self.refinement_min_object_size,
                        closing_kernel_size=self.refinement_closing_kernel_size,
                        gaussian_sigma=self.refinement_gaussian_sigma
                    )
                    # Use refined masks for saving
                    mask_a_to_save = refined_a
                    mask_b_to_save = refined_b
                else:
                    # Fallback to original morphological boundary extraction
                    boundary = extract_morphological_boundary(
                        mask_a, mask_b,
                        thickness=self.boundary_thickness
                    )
                    # Use original masks for saving
                    mask_a_to_save = mask_a
                    mask_b_to_save = mask_b

                # Create result entry
                result = {
                    'crop_name': crop_name,
                    'crop_path': str(crop_path),
                    'category': category,
                    'description': description,
                    'texture_a': texture_a,
                    'texture_b': texture_b,
                    'entropy_a': round(entropy_a, 3),
                    'entropy_b': round(entropy_b, 3),
                    'min_entropy': round(min_entropy, 3),
                    'threshold': self.entropy_threshold,
                    'passed': passed,
                    'mask_refined': self.refine_masks,
                    'boundary_method': 'xor_refined' if self.refine_masks else 'morphological'
                }
                filter_results.append(result)

                # Save mask
                mask_category_dir = masks_dir / category
                mask_category_dir.mkdir(parents=True, exist_ok=True)
                mask_filename = crop_name.replace('.jpg', '.png').replace('.jpeg', '.png')
                mask_path = mask_category_dir / mask_filename
                Image.fromarray(boundary).save(mask_path)

                result['mask_path'] = str(mask_path)

                if passed:
                    passed_crops.append(result)
                    # Copy crop to filter/passed folder
                    filter_passed_dir = filter_dir / "passed" / category
                    filter_passed_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(crop_path, filter_passed_dir / crop_name)

                    if self.verbose:
                        print(f"✓ PASS (H_a={entropy_a:.2f}, H_b={entropy_b:.2f})")
                else:
                    failed_crops.append(result)
                    # Copy crop to filter/failed folder
                    filter_failed_dir = filter_dir / "failed" / category
                    filter_failed_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(crop_path, filter_failed_dir / crop_name)

                    if self.verbose:
                        print(f"✗ FAIL (H_a={entropy_a:.2f}, H_b={entropy_b:.2f})")

            except Exception as e:
                if self.verbose:
                    print(f"✗ Error: {e}")
                filter_results.append({
                    'crop_name': crop_name,
                    'crop_path': str(crop_path),
                    'error': str(e),
                    'passed': False
                })
                failed_crops.append({'crop_name': crop_name, 'error': str(e)})

        # Save filter results JSON
        filter_json_path = filter_dir / "entropy_filter_results.json"
        save_json(filter_results, filter_json_path)

        summary = {
            'total': len(all_crops),
            'processed': len(filter_results),
            'passed': len(passed_crops),
            'failed': len(failed_crops),
            'threshold': self.entropy_threshold,
            'pass_rate': round(len(passed_crops) / len(filter_results) * 100, 1) if filter_results else 0,
            'masks_dir': str(masks_dir),
            'filter_dir': str(filter_dir),
            'filter_json': str(filter_json_path),
            'mask_refined': self.refine_masks,
            'boundary_method': 'xor_refined' if self.refine_masks else 'morphological',
            'refinement_params': {
                'min_object_size': self.refinement_min_object_size,
                'closing_kernel_size': self.refinement_closing_kernel_size,
                'gaussian_sigma': self.refinement_gaussian_sigma
            } if self.refine_masks else None
        }

        return summary

    def collect_dataset_from_results(
        self,
        filter_dir: Path,
        masks_dir: Path
    ) -> Dict:
        """
        Collect all passed crops and masks into a flat dataset structure.

        Creates:
            /datasets/real_texture_boundaries_<date>/images/  - all passed crops
            /datasets/real_texture_boundaries_<date>/masks/   - matching masks

        Args:
            filter_dir: Directory containing filter results (with passed/ subfolder)
            masks_dir: Directory containing masks (organized by size category)

        Returns:
            Dict with collection statistics
        """
        # Generate dataset name with date
        date_str = datetime.now().strftime("%Y%m%d")
        dataset_name = f"real_texture_boundaries_{date_str}"
        dataset_dir = self.dataset_base_dir / dataset_name

        # Create output directories
        images_dir = dataset_dir / "images"
        masks_out_dir = dataset_dir / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_out_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"\n   📁 Dataset directory: {dataset_dir}")

        # Collect all passed crops from all size categories
        passed_dir = filter_dir / "passed"
        collected_images = 0
        collected_masks = 0
        skipped_masks = 0

        for category in ['tiny', 'small', 'medium', 'large', 'xlarge']:
            category_dir = passed_dir / category
            if not category_dir.exists():
                continue

            crops = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))

            for crop_path in crops:
                crop_name = crop_path.name

                # Copy crop to images/ (flat)
                dest_image = images_dir / crop_name
                shutil.copy2(crop_path, dest_image)
                collected_images += 1

                # Find and copy matching mask
                mask_name = crop_name.replace('.jpg', '.png').replace('.jpeg', '.png')
                mask_path = masks_dir / category / mask_name

                if mask_path.exists():
                    dest_mask = masks_out_dir / mask_name
                    shutil.copy2(mask_path, dest_mask)
                    collected_masks += 1
                else:
                    skipped_masks += 1
                    if self.verbose:
                        print(f"   ⚠️  Mask not found for: {crop_name}")

        # Save dataset info
        info = {
            'dataset_name': dataset_name,
            'created': datetime.now().isoformat(),
            'source_filter_dir': str(filter_dir),
            'source_masks_dir': str(masks_dir),
            'total_images': collected_images,
            'total_masks': collected_masks,
            'skipped_masks': skipped_masks,
            'entropy_threshold': self.entropy_threshold
        }
        info_path = dataset_dir / "dataset_info.json"
        save_json(info, info_path)

        summary = {
            'dataset_name': dataset_name,
            'dataset_dir': str(dataset_dir),
            'images_dir': str(images_dir),
            'masks_dir': str(masks_out_dir),
            'collected_images': collected_images,
            'collected_masks': collected_masks,
            'skipped_masks': skipped_masks
        }

        return summary

    def run(
        self,
        image_paths: List[Union[str, Path]],
        grounding_prompt: str = None
    ) -> Dict:
        """
        Run the complete pipeline.

        Args:
            image_paths: List of image paths
            grounding_prompt: Grounding prompt (for basic pipeline)

        Returns:
            Dict with results and statistics
        """
        if self.verbose:
            print("\n" + "="*70)
            print("PIPELINE - STARTING")
            print("="*70)
            print(f"Images: {len(image_paths)}")
            print(f"Entropy threshold: {self.entropy_threshold}")
            print("="*70)

        # Load prompt
        grounding_prompt = get_prompt('grounding')

        # STEP 1: Run basic pipeline (grounding + crop extraction)
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 1: BASIC PIPELINE (GROUNDING + EXTRACTION)")
            print("="*70)

        basic_results = self.basic_pipeline.run(image_paths, prompt=grounding_prompt)
        run_dir = Path(basic_results['output_dir'])

        # Track steps
        all_steps = []

        # STEP 2: Extract masks and filter by entropy
        if self.extract_masks:
            if self.verbose:
                print("\n" + "="*70)
                print("STEP 2: EXTRACT MASKS + ENTROPY FILTER (Sa2VA)")
                print("="*70)

            # Unload Qwen
            self.unload_vlm_model()

            # Load Sa2VA
            self.load_sa2va_model()

            if self.sa2va_model is None:
                if self.verbose:
                    print("   ⚠️  Sa2VA model not available, skipping")
            else:
                # Load processed_bboxes.json
                processed_file = run_dir / "processed_bboxes.json"
                with open(processed_file, 'r') as f:
                    processed_data = json.load(f)

                # Directories
                crops_dir = run_dir / "crops"
                masks_dir = run_dir / "masks"
                filter_dir = run_dir / "filter"

                # Extract masks and filter
                mask_filter_stats = self.extract_masks_and_filter(
                    crops_dir=crops_dir,
                    processed_data=processed_data,
                    masks_dir=masks_dir,
                    filter_dir=filter_dir
                )

                all_steps.append({
                    'step': 'mask_extraction_and_filter',
                    **mask_filter_stats
                })

                if self.verbose:
                    print(f"\n   ✅ Mask extraction + filtering complete")
                    print(f"   Total: {mask_filter_stats['total']}")
                    print(f"   Passed: {mask_filter_stats['passed']}")
                    print(f"   Failed: {mask_filter_stats['failed']}")
                    print(f"   Pass rate: {mask_filter_stats['pass_rate']}%")

                # STEP 3: Collect passed crops into dataset
                if self.collect_dataset and mask_filter_stats['passed'] > 0:
                    if self.verbose:
                        print("\n" + "="*70)
                        print("STEP 3: COLLECT DATASET")
                        print("="*70)

                    dataset_stats = self.collect_dataset_from_results(
                        filter_dir=filter_dir,
                        masks_dir=masks_dir
                    )

                    all_steps.append({
                        'step': 'collect_dataset',
                        **dataset_stats
                    })

                    if self.verbose:
                        print(f"\n   ✅ Dataset collection complete")
                        print(f"   Images collected: {dataset_stats['collected_images']}")
                        print(f"   Masks collected: {dataset_stats['collected_masks']}")
                        print(f"   Dataset: {dataset_stats['dataset_dir']}")

        # Final summary
        summary = {
            'pipeline': 'iterative_with_entropy_filter',
            'entropy_threshold': self.entropy_threshold,
            'steps_completed': len(all_steps),
            'basic_results': basic_results,
            'steps': all_steps,
            'output_dir': str(run_dir)
        }

        # Add dataset info if collected
        for step in all_steps:
            if step.get('step') == 'collect_dataset':
                summary['dataset_dir'] = step['dataset_dir']
                break

        # Save summary
        summary_file = run_dir / "pipeline_summary.json"
        save_json(summary, summary_file)

        if self.verbose:
            print("\n" + "="*70)
            print("✅ PIPELINE COMPLETE!")
            print("="*70)
            print(f"  📁 Output directory: {run_dir}")
            print(f"  📊 Summary: {summary_file}")
            print(f"\n  Output folders:")
            print(f"    crops/          - All extracted crops")
            print(f"    masks/          - Boundary masks")
            print(f"    filter/         - Entropy filter results")
            print(f"    visualizations/ - Bbox visualizations")
            if 'dataset_dir' in summary:
                print(f"\n  📦 Dataset collected to:")
                print(f"    {summary['dataset_dir']}")
            print("="*70 + "\n")

        return summary


def run_iterative_pipeline(
    model: BaseVLM,
    image_dir: Union[str, Path],
    output_dir: Union[str, Path] = "results",
    extract_masks: bool = True,
    entropy_threshold: float = 4.5,
    collect_dataset: bool = True,
    dataset_base_dir: Union[str, Path] = "/datasets/ade20k",
    num_images: int = None,
    # Mask refinement parameters
    refine_masks: bool = True,
    refinement_min_object_size: int = 100,
    refinement_closing_kernel_size: int = 7,
    refinement_gaussian_sigma: float = 3.0,
    **kwargs
) -> Dict:
    """
    Convenience function to run pipeline on a directory of images.

    Args:
        model: VLM model instance
        image_dir: Directory containing images
        output_dir: Output directory
        extract_masks: Whether to extract masks using Sa2VA
        entropy_threshold: Minimum entropy for both textures
        collect_dataset: Whether to collect passed crops into a dataset
        dataset_base_dir: Base directory for dataset (default: /datasets)
        num_images: Number of images to process (None = all)
        refine_masks: Whether to apply mask refinement before boundary extraction
        refinement_min_object_size: Min object size for refinement (default 100)
        refinement_closing_kernel_size: Kernel size for morphological closing (default 7)
        refinement_gaussian_sigma: Sigma for Gaussian blur smoothing (default 3.0)
        **kwargs: Additional pipeline arguments

    Returns:
        Summary dict
    """
    images = list_images(image_dir)

    if num_images is not None:
        images = images[:num_images]

    if not images:
        raise ValueError(f"No images found in {image_dir}")

    pipeline = IterativePipeline(
        model,
        output_dir=output_dir,
        extract_masks=extract_masks,
        entropy_threshold=entropy_threshold,
        collect_dataset=collect_dataset,
        dataset_base_dir=dataset_base_dir,
        refine_masks=refine_masks,
        refinement_min_object_size=refinement_min_object_size,
        refinement_closing_kernel_size=refinement_closing_kernel_size,
        refinement_gaussian_sigma=refinement_gaussian_sigma,
        **kwargs
    )
    return pipeline.run(images)
