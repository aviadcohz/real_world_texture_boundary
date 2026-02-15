"""
Iterative Dual-GPU Pipeline: Scaled grounding with entropy-based filtering.

Workflow:
1. Run dual-GPU pipeline for grounding (local GPU + remote H100)
2. Unload Qwen model
3. Load Sa2VA, extract masks, and filter by entropy
"""

from pathlib import Path
from typing import List, Dict, Union
import torch
import json
import shutil
from PIL import Image
import numpy as np
import cv2
import time

from models import BaseVLM, QwenVLMClient
from core import (
    parse_texture_description,
    extract_morphological_boundary,
    compute_region_entropy,
    refine_masks_and_extract_boundary,
    process_bboxes,
    visualize_all_images,
    extract_all_crops,
    update_processed_bboxes_with_crops,
    validate_bboxes
)
from utils import (
    list_images,
    get_prompt,
    save_json,
    get_timestamp,
    get_image_size
)
from utils.bbox_utils import parse_json_format, adjust_bboxes_after_padding
from utils.image_utils import pad_image_to_standard_size


class IterativeDualGPUPipeline:
    """
    Combines dual-GPU grounding with entropy-based filtering.
    
    Stages:
    1. Dual-GPU grounding (local + remote in parallel)
    2. Bbox processing (validation, IoU filter, boundary fix)
    3. Visualization
    4. Crop extraction
    5. Mask extraction + entropy filtering
    """

    def __init__(
        self,
        local_model: BaseVLM,
        remote_url: str,
        output_dir: Union[str, Path] = "results",
        local_ratio: float = 0.3,
        local_batch_size: int = 4,
        remote_batch_size: int = 12,
        iou_threshold: float = 0.6,
        fix_boundaries: bool = False,
        extract_masks: bool = True,
        entropy_threshold: float = 4.5,
        boundary_thickness: int = 2,
        target_size: int = 1024,
        # Mask refinement parameters
        refine_masks: bool = True,
        refinement_min_object_size: int = 100,
        refinement_closing_kernel_size: int = 7,
        refinement_gaussian_sigma: float = 3.0,
        verbose: bool = True
    ):
        """
        Initialize iterative dual-GPU pipeline.

        Args:
            local_model: Local VLM model instance
            remote_url: URL of remote VLM server
            output_dir: Output directory
            local_ratio: Fraction of images for local GPU (0.0-1.0)
            local_batch_size: Batch size for local GPU
            remote_batch_size: Batch size for remote GPU
            iou_threshold: IoU threshold for bbox filtering
            fix_boundaries: Enable boundary fixing
            extract_masks: Whether to extract masks using Sa2VA
            entropy_threshold: Minimum entropy for both textures
            boundary_thickness: Thickness for boundary extraction
            target_size: Target size for image padding (default 1024)
            refine_masks: Whether to apply mask refinement before boundary extraction
            refinement_min_object_size: Min object size for refinement (default 100)
            refinement_closing_kernel_size: Kernel size for morphological closing (default 7)
            refinement_gaussian_sigma: Sigma for Gaussian blur smoothing (default 3.0)
            verbose: Print progress
        """
        self.local_model = local_model
        self.remote_client = QwenVLMClient(
            server_url=remote_url,
            batch_size=remote_batch_size
        )
        self.output_dir = Path(output_dir)
        self.local_ratio = local_ratio
        self.local_batch_size = local_batch_size
        self.remote_batch_size = remote_batch_size
        self.iou_threshold = iou_threshold
        self.fix_boundaries = fix_boundaries
        self.extract_masks = extract_masks
        self.entropy_threshold = entropy_threshold
        self.sa2va_model = None
        self.boundary_thickness = boundary_thickness
        self.target_size = target_size
        self.verbose = verbose
        
        # Mask refinement settings
        self.refine_masks = refine_masks
        self.refinement_min_object_size = refinement_min_object_size
        self.refinement_closing_kernel_size = refinement_closing_kernel_size
        self.refinement_gaussian_sigma = refinement_gaussian_sigma

        # Create output directory
        self.timestamp = get_timestamp()
        self.run_dir = self.output_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        image_paths: List[Union[str, Path]],
        prompt: str = None
    ) -> Dict:
        """
        Run the complete iterative dual-GPU pipeline.

        Args:
            image_paths: List of image paths
            prompt: Grounding prompt

        Returns:
            Summary dict
        """
        if self.verbose:
            print("\n" + "="*70)
            print("ITERATIVE DUAL-GPU PIPELINE - STARTING")
            print("="*70)
            print(f"Total images: {len(image_paths)}")
            print(f"Local GPU: {int(self.local_ratio * 100)}% of work")
            print(f"Remote GPU: {int((1 - self.local_ratio) * 100)}% of work")
            print(f"Output: {self.run_dir}")
            print("="*70)

        # Load prompt
        if prompt is None:
            prompt = get_prompt('grounding')

        # STEP 1: Dual-GPU Grounding
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 1: DUAL-GPU GROUNDING")
            print("="*70)

        start_time = time.time()
        grounding_results = self._run_dual_gpu_grounding(image_paths, prompt)
        grounding_time = time.time() - start_time

        if self.verbose:
            print(f"\nGrounding completed in {grounding_time:.1f}s")

        # Save grounding results
        grounding_file = self.run_dir / "grounding_results.json"
        save_json(
            [{'image': r['image_name'], 'raw_response': r['raw_response']} 
             for r in grounding_results],
            grounding_file
        )

        # STEP 2: Process bboxes
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 2: PROCESSING BBOXES")
            print("="*70)

        processed_data, stats = self._process_all_bboxes(
            grounding_results, image_paths
        )
        
        processed_file = self.run_dir / "processed_bboxes.json"
        save_json(processed_data, processed_file)

        if self.verbose:
            print(f"  📊 Processing Summary:")
            print(f"     Input boxes:    {stats['total_input']}")
            print(f"     Filtered (IoU): {stats['total_filtered']}")
            print(f"     Fixed (edges):  {stats['total_fixed']}")
            print(f"     Final boxes:    {stats['total_output']}")

        # STEP 3: Visualize
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 3: VISUALIZING")
            print("="*70)

        viz_dir = self.run_dir / "visualizations"
        viz_stats = visualize_all_images(
            processed_data,
            output_dir=viz_dir,
            verbose=self.verbose
        )

        # STEP 4: Extract crops
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 4: EXTRACTING CROPS")
            print("="*70)

        crops_dir = self.run_dir / "crops"
        crop_stats, crops_mapping = extract_all_crops(
            processed_data,
            output_dir=crops_dir,
            resize_by_category=True,
            parallel_save=True,
            save_workers=8,
            verbose=self.verbose
        )

        update_processed_bboxes_with_crops(
            processed_bboxes_path=processed_file,
            crops_mapping=crops_mapping
        )

        # Reload processed data with crop info
        with open(processed_file, 'r') as f:
            processed_data = json.load(f)

        # STEP 5: Extract masks and filter by entropy (optional)
        all_steps = []
        
        if self.extract_masks:
            if self.verbose:
                print("\n" + "="*70)
                print("STEP 5: EXTRACT MASKS + ENTROPY FILTER (Sa2VA)")
                print("="*70)

            # Unload Qwen
            self.unload_vlm_model()

            # Load Sa2VA
            self.load_sa2va_model()

            if self.sa2va_model is not None:
                masks_dir = self.run_dir / "masks"
                filter_dir = self.run_dir / "filter"

                mask_filter_stats = self._extract_masks_and_filter(
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

                # STEP 6: Generate layouts (crop with mask boundary in green)
                if self.verbose:
                    print("\n" + "="*70)
                    print("STEP 6: GENERATING LAYOUTS")
                    print("="*70)

                layouts_dir = self.run_dir / "layouts"
                layout_stats = self._generate_layouts(
                    crops_dir=crops_dir,
                    masks_dir=masks_dir,
                    layouts_dir=layouts_dir
                )
                all_steps.append({
                    'step': 'layout_generation',
                    **layout_stats
                })

        # Final summary
        total_time = time.time() - start_time
        summary = {
            'pipeline': 'iterative_dual_gpu',
            'timestamp': self.timestamp,
            'config': {
                'local_ratio': self.local_ratio,
                'local_batch_size': self.local_batch_size,
                'remote_batch_size': self.remote_batch_size,
                'remote_url': self.remote_client.server_url,
                'iou_threshold': self.iou_threshold,
                'fix_boundaries': self.fix_boundaries,
                'entropy_threshold': self.entropy_threshold,
                'extract_masks': self.extract_masks,
                'target_size': self.target_size,
            },
            'input': {
                'num_images': len(image_paths),
            },
            'timing': {
                'grounding_time': grounding_time,
                'total_time': total_time,
                'images_per_second': len(image_paths) / total_time if total_time > 0 else 0
            },
            'processing': stats,
            'visualization': viz_stats,
            'crops': crop_stats,
            'steps': all_steps,
            'output_dir': str(self.run_dir)
        }

        summary_file = self.run_dir / "pipeline_summary.json"
        save_json(summary, summary_file)

        if self.verbose:
            self._print_summary(summary)

        return summary

    def _run_dual_gpu_grounding(
        self,
        image_paths: List[Path],
        prompt: str
    ) -> List[Dict]:
        """Run grounding on both local and remote GPUs in parallel."""
        from concurrent.futures import ThreadPoolExecutor

        # Split images
        split_idx = int(len(image_paths) * self.local_ratio)
        local_images = image_paths[:split_idx]
        remote_images = image_paths[split_idx:]

        if self.verbose:
            print(f"\nLocal GPU:  {len(local_images)} images (batch_size={self.local_batch_size})")
            print(f"Remote GPU: {len(remote_images)} images (batch_size={self.remote_batch_size})")

        all_results = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            local_future = executor.submit(
                self._process_images_local, local_images, prompt
            )
            remote_future = executor.submit(
                self._process_images_remote, remote_images, prompt
            )

            local_results = local_future.result()
            remote_results = remote_future.result()

        all_results = local_results + remote_results
        return all_results

    def _process_images_local(
        self,
        image_paths: List[Path],
        prompt: str
    ) -> List[Dict]:
        """Process images on local GPU with proper padding."""
        if not image_paths:
            return []

        results = []

        if self.verbose:
            print(f"\n[LOCAL] Starting processing of {len(image_paths)} images...")

        for i in range(0, len(image_paths), self.local_batch_size):
            batch_paths = image_paths[i:i + self.local_batch_size]

            if self.verbose:
                batch_num = i // self.local_batch_size + 1
                total_batches = (len(image_paths) + self.local_batch_size - 1) // self.local_batch_size
                print(f"  [LOCAL] Batch {batch_num}/{total_batches}")

            # Pad images and collect metadata
            padded_images = []
            padding_metadata = []
            for path in batch_paths:
                padded_image, offset_x, offset_y, orig_size = pad_image_to_standard_size(
                    path, target_size=self.target_size
                )
                padded_images.append(padded_image)
                padding_metadata.append({
                    'offset_x': offset_x,
                    'offset_y': offset_y,
                    'original_size': orig_size
                })

            batch_prompts = [prompt] * len(padded_images)

            if hasattr(self.local_model, 'batch_generate'):
                responses = self.local_model.batch_generate(padded_images, batch_prompts)
            else:
                responses = [
                    self.local_model.generate(img, prompt)
                    for img in padded_images
                ]

            for path, response, metadata in zip(batch_paths, responses, padding_metadata):
                path = Path(path)
                # Parse bboxes from padded image coordinates
                bboxes_padded = parse_json_format(response)
                # Adjust bboxes back to original image coordinates
                bboxes = adjust_bboxes_after_padding(
                    bboxes_padded,
                    metadata['offset_x'],
                    metadata['offset_y']
                )
                results.append({
                    'image_name': path.name,
                    'image_path': str(path),
                    'raw_response': response,
                    'bboxes': bboxes,
                    'source': 'local'
                })

        if self.verbose:
            print(f"  [LOCAL] Completed {len(results)} images")

        return results

    def _process_images_remote(
        self,
        image_paths: List[Path],
        prompt: str
    ) -> List[Dict]:
        """Process images on remote GPU with proper padding."""
        if not image_paths:
            return []

        results = []

        if self.verbose:
            print(f"\n[REMOTE] Starting processing of {len(image_paths)} images...")

        for i in range(0, len(image_paths), self.remote_batch_size):
            batch_paths = image_paths[i:i + self.remote_batch_size]

            if self.verbose:
                batch_num = i // self.remote_batch_size + 1
                total_batches = (len(image_paths) + self.remote_batch_size - 1) // self.remote_batch_size
                print(f"  [REMOTE] Batch {batch_num}/{total_batches}")

            # Pad images and collect metadata
            padded_images = []
            padding_metadata = []
            for path in batch_paths:
                padded_image, offset_x, offset_y, orig_size = pad_image_to_standard_size(
                    path, target_size=self.target_size
                )
                padded_images.append(padded_image)
                padding_metadata.append({
                    'offset_x': offset_x,
                    'offset_y': offset_y,
                    'original_size': orig_size
                })

            batch_prompts = [prompt] * len(padded_images)
            responses = self.remote_client.batch_generate(padded_images, batch_prompts)

            for path, response, metadata in zip(batch_paths, responses, padding_metadata):
                path = Path(path)
                # Parse bboxes from padded image coordinates
                bboxes_padded = parse_json_format(response)
                # Adjust bboxes back to original image coordinates
                bboxes = adjust_bboxes_after_padding(
                    bboxes_padded,
                    metadata['offset_x'],
                    metadata['offset_y']
                )
                results.append({
                    'image_name': path.name,
                    'image_path': str(path),
                    'raw_response': response,
                    'bboxes': bboxes,
                    'source': 'remote'
                })

        if self.verbose:
            print(f"  [REMOTE] Completed {len(results)} images")

        return results

    def _process_all_bboxes(
        self,
        grounding_results: List[Dict],
        image_paths: List[Path]
    ) -> tuple:
        """Process and validate all bboxes."""
        processed_data = []
        total_input = 0
        total_filtered = 0
        total_fixed = 0
        total_output = 0

        path_lookup = {Path(p).name: Path(p) for p in image_paths}

        for result in grounding_results:
            image_name = result['image_name']
            image_path = path_lookup.get(image_name)

            if image_path is None:
                continue

            img_width, img_height = get_image_size(image_path)

            valid_boxes, _ = validate_bboxes(
                result['bboxes'],
                img_width,
                img_height,
                verbose=False
            )

            processed_boxes, stats = process_bboxes(
                valid_boxes,
                img_width,
                img_height,
                filter_iou=True,
                iou_threshold=self.iou_threshold,
                fix_boundaries=self.fix_boundaries,
                verbose=False
            )

            processed_data.append({
                'image': image_name,
                'image_path': str(image_path),
                'boxes': processed_boxes
            })

            total_input += stats['input_count']
            total_filtered += stats['filtered_count']
            total_fixed += stats['fixed_count']
            total_output += stats['output_count']

        stats = {
            'total_input': total_input,
            'total_filtered': total_filtered,
            'total_fixed': total_fixed,
            'total_output': total_output,
        }

        return processed_data, stats

    def unload_vlm_model(self):
        """Unload Qwen model from GPU to free VRAM."""
        if self.verbose:
            print("\n   🔄 Unloading VLM model from GPU...")

        try:
            del self.local_model._model
            del self.local_model._processor
            self.local_model._model = None
            self.local_model._processor = None

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

    def _extract_masks_and_filter(
        self,
        crops_dir: Path,
        processed_data: List[Dict],
        masks_dir: Path,
        filter_dir: Path
    ) -> Dict:
        """Extract masks for all crops and filter by entropy."""
        if self.sa2va_model is None:
            if self.verbose:
                print("   ⚠️  No Sa2VA model provided, skipping")
            return {'total': 0, 'passed': 0, 'failed': 0}

        # Build lookup: crop_name -> description
        description_lookup = {}
        for img_data in processed_data:
            for box in img_data.get('boxes', []):
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
                from core import parse_texture_description, compute_region_entropy, extract_morphological_boundary, refine_masks_and_extract_boundary
                
                # Parse description
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
            'filter_json': str(filter_json_path)
        }

        return summary

    def _generate_layouts(self, crops_dir: Path, masks_dir: Path, layouts_dir: Path, alpha: float = 0.5) -> Dict:
        """Generate layout images: crop with mask boundary emphasized in green.

        Produces a flat directory of layout images (no size subdirectories).
        Uses the same overlay style as test_e2e.py create_overlay.
        """
        layouts_dir.mkdir(parents=True, exist_ok=True)

        generated = 0
        errors = 0

        for category in ['tiny', 'small', 'medium', 'large', 'xlarge']:
            category_crop_dir = crops_dir / category
            category_mask_dir = masks_dir / category
            if not category_crop_dir.exists() or not category_mask_dir.exists():
                continue

            for crop_path in sorted(category_crop_dir.glob("*.jpg")):
                mask_filename = crop_path.stem + ".png"
                mask_path = category_mask_dir / mask_filename
                if not mask_path.exists():
                    continue

                try:
                    crop = np.array(Image.open(crop_path).convert("RGB"))
                    mask = np.array(Image.open(mask_path).convert("L"))

                    # Dilate mask to emphasize boundary
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    mask = cv2.dilate(mask, kernel, iterations=2)

                    if mask.shape[:2] != crop.shape[:2]:
                        mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)

                    boundary = mask > 127
                    overlay = crop.copy()
                    overlay[boundary] = [0, 255, 0]

                    result = crop.copy()
                    result[boundary] = (
                        (1 - alpha) * crop[boundary].astype(float) +
                        alpha * overlay[boundary].astype(float)
                    ).astype(np.uint8)

                    # Save flat (no size subdirectory)
                    out_path = layouts_dir / crop_path.name
                    Image.fromarray(result).save(str(out_path), quality=90)
                    generated += 1
                except Exception as e:
                    errors += 1
                    if self.verbose:
                        print(f"      Layout error for {crop_path.name}: {e}")

        if self.verbose:
            print(f"\n   ✅ Layouts generated: {generated} (errors: {errors})")
            print(f"   📁 {layouts_dir}")

        return {'generated': generated, 'errors': errors, 'layouts_dir': str(layouts_dir)}

    def _print_summary(self, summary: Dict):
        """Print final summary."""
        print("\n" + "="*70)
        print("ITERATIVE DUAL-GPU PIPELINE COMPLETE!")
        print("="*70)
        print(f"  Output directory: {self.run_dir}")
        print(f"  Total images: {summary['input']['num_images']}")
        print(f"  Grounding time: {summary['timing']['grounding_time']:.1f}s")
        print(f"  Total time: {summary['timing']['total_time']:.1f}s")
        print(f"  Throughput: {summary['timing']['images_per_second']:.2f} images/sec")
        print(f"  Total crops: {summary['crops'].get('total_crops', 0)}")
        if self.extract_masks and summary['steps']:
            filter_step = summary['steps'][0]
            print(f"  Masks passed entropy filter: {filter_step.get('passed', 0)}")
        print("="*70 + "\n")


def run_iterative_dual_gpu_pipeline(
    local_model: BaseVLM,
    remote_url: str,
    image_dir: Union[str, Path],
    output_dir: Union[str, Path] = "results",
    local_ratio: float = 0.3,
    local_batch_size: int = 4,
    remote_batch_size: int = 12,
    num_images: int = None,
    extract_masks: bool = True,
    entropy_threshold: float = 4.5,
    boundary_thickness: int = 2,
    target_size: int = 1024,
    # Mask refinement parameters
    refine_masks: bool = True,
    refinement_min_object_size: int = 100,
    refinement_closing_kernel_size: int = 7,
    refinement_gaussian_sigma: float = 3.0,
    **kwargs
) -> Dict:
    """
    Run iterative dual-GPU pipeline on a directory of images.

    Args:
        local_model: Local VLM model instance
        remote_url: URL of remote VLM server
        image_dir: Directory containing images
        output_dir: Output directory
        local_ratio: Fraction for local GPU (0.3 = 30% local, 70% remote)
        local_batch_size: Batch size for local GPU
        remote_batch_size: Batch size for remote GPU
        num_images: Number of images to process (None = all)
        extract_masks: Whether to extract masks using Sa2VA
        entropy_threshold: Minimum entropy for both textures
        boundary_thickness: Thickness for boundary extraction
        target_size: Target size for image padding (default 1024)
        refine_masks: Whether to apply mask refinement before boundary extraction
        refinement_min_object_size: Min object size for refinement (default 100)
        refinement_closing_kernel_size: Kernel size for morphological closing (default 7)
        refinement_gaussian_sigma: Sigma for Gaussian blur smoothing (default 3.0)
        **kwargs: Additional pipeline arguments

    Returns:
        Summary dict
    """
    images = list_images(image_dir)

    if not images:
        raise ValueError(f"No images found in {image_dir}")

    if num_images is not None:
        images = images[:num_images]

    pipeline = IterativeDualGPUPipeline(
        local_model=local_model,
        remote_url=remote_url,
        output_dir=output_dir,
        local_ratio=local_ratio,
        local_batch_size=local_batch_size,
        remote_batch_size=remote_batch_size,
        extract_masks=extract_masks,
        entropy_threshold=entropy_threshold,
        boundary_thickness=boundary_thickness,
        target_size=target_size,
        refine_masks=refine_masks,
        refinement_min_object_size=refinement_min_object_size,
        refinement_closing_kernel_size=refinement_closing_kernel_size,
        refinement_gaussian_sigma=refinement_gaussian_sigma,
        **kwargs
    )

    return pipeline.run(images)
