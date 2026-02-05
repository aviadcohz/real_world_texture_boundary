"""
Dual-GPU pipeline for parallel processing on local and remote GPUs.

Splits work between local GPU and remote server (e.g., H100) for ~2x throughput.

Usage:
    from pipelines import run_dual_gpu_pipeline

    results = run_dual_gpu_pipeline(
        local_model=local_model,
        remote_url="http://132.66.150.69:8000",
        image_dir="/path/to/images",
        output_dir="results",
        local_ratio=0.3,  # 30% local, 70% remote (H100 is faster)
    )
"""

from pathlib import Path
from typing import List, Dict, Union, Optional
from concurrent.futures import ThreadPoolExecutor, Future
import threading
import json
import time

from models import BaseVLM
from models.qwen_client import QwenVLMClient
from core import (
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
from utils.bbox_utils import parse_json_format

try:
    from config.scale_config import ScaleConfig
    SCALE_CONFIG_AVAILABLE = True
except ImportError:
    SCALE_CONFIG_AVAILABLE = False


class DualGPUPipeline:
    """
    Pipeline that processes images on both local and remote GPUs in parallel.

    Automatically splits workload and merges results.
    """

    def __init__(
        self,
        local_model: BaseVLM,
        remote_url: str,
        output_dir: Union[str, Path] = "results",
        config: 'ScaleConfig' = None,
        local_ratio: float = 0.3,
        local_batch_size: int = 4,
        remote_batch_size: int = 12,
        iou_threshold: float = 0.6,
        fix_boundaries: bool = True,
        verbose: bool = True
    ):
        """
        Initialize dual-GPU pipeline.

        Args:
            local_model: Local VLM model instance
            remote_url: URL of remote VLM server
            output_dir: Output directory
            config: ScaleConfig instance
            local_ratio: Fraction of images for local GPU (0.0-1.0)
                        Set lower if remote GPU is faster (e.g., 0.3 for H100)
            local_batch_size: Batch size for local GPU
            remote_batch_size: Batch size for remote GPU
            iou_threshold: IoU threshold for filtering
            fix_boundaries: Enable boundary fixing
            verbose: Print progress
        """
        self.local_model = local_model
        self.remote_client = QwenVLMClient(
            server_url=remote_url,
            batch_size=remote_batch_size
        )
        self.output_dir = Path(output_dir)
        self.config = config
        self.local_ratio = local_ratio
        self.local_batch_size = local_batch_size
        self.remote_batch_size = remote_batch_size
        self.iou_threshold = iou_threshold
        self.fix_boundaries = fix_boundaries
        self.verbose = verbose

        # Create output structure
        self.timestamp = get_timestamp()
        self.run_dir = self.output_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Thread-safe result storage
        self._results_lock = threading.Lock()
        self._local_results = []
        self._remote_results = []

    def run(
        self,
        image_paths: List[Union[str, Path]],
        prompt: str = None
    ) -> Dict:
        """
        Run pipeline on both GPUs in parallel.

        Args:
            image_paths: List of image paths
            prompt: Grounding prompt

        Returns:
            Summary dict with results
        """
        if self.verbose:
            print("\n" + "="*70)
            print("DUAL-GPU PIPELINE - STARTING")
            print("="*70)
            print(f"Total images: {len(image_paths)}")
            print(f"Local GPU: {int(self.local_ratio * 100)}% of work")
            print(f"Remote GPU: {int((1 - self.local_ratio) * 100)}% of work")
            print(f"Output: {self.run_dir}")
            print("="*70)

        # Load prompt
        if prompt is None:
            prompt = get_prompt('grounding')

        # Split images between local and remote
        split_idx = int(len(image_paths) * self.local_ratio)
        local_images = image_paths[:split_idx]
        remote_images = image_paths[split_idx:]

        if self.verbose:
            print(f"\nLocal GPU:  {len(local_images)} images (batch_size={self.local_batch_size})")
            print(f"Remote GPU: {len(remote_images)} images (batch_size={self.remote_batch_size})")

        # Process both in parallel
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=2) as executor:
            local_future = executor.submit(
                self._process_local, local_images, prompt
            )
            remote_future = executor.submit(
                self._process_remote, remote_images, prompt
            )

            # Wait for both to complete
            local_results = local_future.result()
            remote_results = remote_future.result()

        grounding_time = time.time() - start_time

        if self.verbose:
            print(f"\nGrounding completed in {grounding_time:.1f}s")

        # Merge results
        all_grounding_results = local_results + remote_results

        # Save grounding results
        grounding_file = self.run_dir / "grounding_results.json"
        save_json(all_grounding_results, grounding_file)

        # Process bboxes
        if self.verbose:
            print("\n PROCESSING BBOXES")

        processed_data, stats = self._process_all_bboxes(
            all_grounding_results, image_paths
        )

        processed_file = self.run_dir / "processed_bboxes.json"
        save_json(processed_data, processed_file)

        # Visualize
        if self.verbose:
            print("\n VISUALIZING")

        viz_dir = self.run_dir / "visualizations"
        viz_stats = visualize_all_images(
            processed_data,
            output_dir=viz_dir,
            verbose=self.verbose
        )

        # Extract crops
        if self.verbose:
            print("\n EXTRACTING CROPS")

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

        # Build summary
        total_time = time.time() - start_time
        summary = {
            'pipeline': 'dual_gpu',
            'timestamp': self.timestamp,
            'config': {
                'local_ratio': self.local_ratio,
                'local_batch_size': self.local_batch_size,
                'remote_batch_size': self.remote_batch_size,
                'remote_url': self.remote_client.server_url,
                'iou_threshold': self.iou_threshold,
                'fix_boundaries': self.fix_boundaries,
            },
            'input': {
                'num_images': len(image_paths),
                'local_images': len(local_images),
                'remote_images': len(remote_images),
            },
            'timing': {
                'grounding_time': grounding_time,
                'total_time': total_time,
                'images_per_second': len(image_paths) / total_time
            },
            'processing': stats,
            'visualization': viz_stats,
            'crops': crop_stats,
            'output_dir': str(self.run_dir)
        }

        summary_file = self.run_dir / "pipeline_summary.json"
        save_json(summary, summary_file)

        if self.verbose:
            self._print_summary(summary)

        return summary

    def _process_local(
        self,
        image_paths: List[Path],
        prompt: str
    ) -> List[Dict]:
        """Process images on local GPU."""
        if not image_paths:
            return []

        results = []

        if self.verbose:
            print(f"\n[LOCAL] Starting processing of {len(image_paths)} images...")

        # Process in batches
        for i in range(0, len(image_paths), self.local_batch_size):
            batch_paths = image_paths[i:i + self.local_batch_size]
            batch_prompts = [prompt] * len(batch_paths)

            if self.verbose:
                batch_num = i // self.local_batch_size + 1
                total_batches = (len(image_paths) + self.local_batch_size - 1) // self.local_batch_size
                print(f"  [LOCAL] Batch {batch_num}/{total_batches}")

            # Use batched generation if available
            if hasattr(self.local_model, 'batch_generate'):
                responses = self.local_model.batch_generate(
                    batch_paths, batch_prompts
                )
            else:
                responses = [
                    self.local_model.generate(p, prompt)
                    for p in batch_paths
                ]

            # Parse results
            for path, response in zip(batch_paths, responses):
                path = Path(path)
                bboxes = parse_json_format(response)
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

    def _process_remote(
        self,
        image_paths: List[Path],
        prompt: str
    ) -> List[Dict]:
        """Process images on remote GPU."""
        if not image_paths:
            return []

        results = []

        if self.verbose:
            print(f"\n[REMOTE] Starting processing of {len(image_paths)} images...")

        # Process in batches
        for i in range(0, len(image_paths), self.remote_batch_size):
            batch_paths = image_paths[i:i + self.remote_batch_size]
            batch_prompts = [prompt] * len(batch_paths)

            if self.verbose:
                batch_num = i // self.remote_batch_size + 1
                total_batches = (len(image_paths) + self.remote_batch_size - 1) // self.remote_batch_size
                print(f"  [REMOTE] Batch {batch_num}/{total_batches}")

            # Use batched remote generation
            responses = self.remote_client.batch_generate(
                batch_paths, batch_prompts
            )

            # Parse results
            for path, response in zip(batch_paths, responses):
                path = Path(path)
                bboxes = parse_json_format(response)
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
        """Process all bboxes with validation and filtering."""
        processed_data = []
        total_stats = {
            'input_count': 0,
            'filtered_count': 0,
            'fixed_count': 0,
            'output_count': 0
        }

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

            for key in total_stats:
                total_stats[key] += stats.get(key, 0)

        return processed_data, total_stats

    def _print_summary(self, summary: Dict):
        """Print final summary."""
        print("\n" + "="*70)
        print("DUAL-GPU PIPELINE COMPLETE!")
        print("="*70)
        print(f"  Output directory: {self.run_dir}")
        print(f"  Total images: {summary['input']['num_images']}")
        print(f"    - Local GPU:  {summary['input']['local_images']}")
        print(f"    - Remote GPU: {summary['input']['remote_images']}")
        print(f"  Grounding time: {summary['timing']['grounding_time']:.1f}s")
        print(f"  Total time: {summary['timing']['total_time']:.1f}s")
        print(f"  Throughput: {summary['timing']['images_per_second']:.2f} images/sec")
        print(f"  Total crops: {summary['crops'].get('total_crops', 0)}")
        print("="*70 + "\n")


def run_dual_gpu_pipeline(
    local_model: BaseVLM,
    remote_url: str,
    image_dir: Union[str, Path],
    output_dir: Union[str, Path] = "results",
    config: 'ScaleConfig' = None,
    local_ratio: float = 0.3,
    local_batch_size: int = 4,
    remote_batch_size: int = 12,
    num_images: int = None,
    **kwargs
) -> Dict:
    """
    Run dual-GPU pipeline on a directory of images.

    Args:
        local_model: Local VLM model instance
        remote_url: URL of remote VLM server (e.g., "http://132.66.150.69:8000")
        image_dir: Directory containing images
        output_dir: Output directory
        config: ScaleConfig instance
        local_ratio: Fraction for local GPU (0.3 = 30% local, 70% remote)
        local_batch_size: Batch size for local GPU
        remote_batch_size: Batch size for remote GPU
        num_images: Number of images to process (None = all)
        **kwargs: Additional pipeline arguments

    Returns:
        Summary dict
    """
    images = list_images(image_dir)

    if not images:
        raise ValueError(f"No images found in {image_dir}")

    if num_images is not None:
        images = images[:num_images]

    pipeline = DualGPUPipeline(
        local_model=local_model,
        remote_url=remote_url,
        output_dir=output_dir,
        config=config,
        local_ratio=local_ratio,
        local_batch_size=local_batch_size,
        remote_batch_size=remote_batch_size,
        **kwargs
    )

    return pipeline.run(images)
