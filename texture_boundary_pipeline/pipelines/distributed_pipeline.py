"""
Distributed pipeline for multi-GPU and multi-node processing.

Supports:
- Multi-GPU processing on a single node
- Multi-node distributed processing
- Data parallel processing with automatic sharding
"""

from pathlib import Path
from typing import List, Dict, Union, Optional
import os
import json

try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from models import BaseVLM
from utils import list_images, get_prompt, save_json, get_timestamp

try:
    from config.scale_config import ScaleConfig, DistributedConfig
    SCALE_CONFIG_AVAILABLE = True
except ImportError:
    SCALE_CONFIG_AVAILABLE = False


class DistributedPipeline:
    """
    Distributed pipeline for processing across multiple GPUs or nodes.

    Usage:
        # Single node, multiple GPUs
        torchrun --nproc_per_node=4 main.py distributed --images /path/to/images

        # Multi-node
        torchrun --nnodes=2 --node_rank=0 --master_addr=... main.py distributed ...
    """

    def __init__(
        self,
        model_factory,
        output_dir: Union[str, Path] = "results",
        config: DistributedConfig = None,
        **pipeline_kwargs
    ):
        """
        Initialize distributed pipeline.

        Args:
            model_factory: Function that creates model instance
            output_dir: Base output directory
            config: DistributedConfig instance
            **pipeline_kwargs: Additional pipeline arguments
        """
        self.model_factory = model_factory
        self.output_dir = Path(output_dir)
        self.config = config or DistributedConfig()
        self.pipeline_kwargs = pipeline_kwargs

        # Distributed state
        self.is_initialized = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.is_main_process = True

        # Model (created per-process)
        self.model = None

    def initialize(self):
        """Initialize distributed environment."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for distributed processing")

        if self.config.enabled:
            # Initialize from environment or config
            if 'RANK' in os.environ:
                # Launched via torchrun
                self.rank = int(os.environ['RANK'])
                self.world_size = int(os.environ['WORLD_SIZE'])
                self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            else:
                self.rank = self.config.rank
                self.world_size = self.config.world_size
                self.local_rank = self.config.local_rank

            # Initialize process group
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.config.backend,
                    init_method=self.config.init_method
                )

            self.is_main_process = (self.rank == 0)

            if self.is_main_process:
                print(f"Distributed initialized: {self.world_size} processes")
        else:
            self.is_main_process = True

        self.is_initialized = True

        # Create model on local GPU
        device = f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        self.model = self.model_factory(device=device)

    def partition_data(self, items: List) -> List:
        """
        Partition data for this process.

        Args:
            items: Full list of items

        Returns:
            Subset of items for this process
        """
        if self.world_size == 1:
            return items

        # Simple round-robin partitioning
        return items[self.rank::self.world_size]

    def gather_results(self, local_results: List[Dict]) -> List[Dict]:
        """
        Gather results from all processes.

        Args:
            local_results: Results from this process

        Returns:
            Combined results from all processes (only on main process)
        """
        if self.world_size == 1:
            return local_results

        if not TORCH_AVAILABLE or not dist.is_initialized():
            return local_results

        # Serialize results
        local_data = json.dumps(local_results).encode()

        # Gather sizes
        local_size = torch.tensor([len(local_data)], dtype=torch.long, device='cuda')
        all_sizes = [torch.zeros(1, dtype=torch.long, device='cuda')
                     for _ in range(self.world_size)]
        dist.all_gather(all_sizes, local_size)

        # Gather data
        max_size = max(s.item() for s in all_sizes)
        local_tensor = torch.zeros(max_size, dtype=torch.uint8, device='cuda')
        local_tensor[:len(local_data)] = torch.tensor(
            list(local_data), dtype=torch.uint8, device='cuda'
        )

        all_tensors = [torch.zeros(max_size, dtype=torch.uint8, device='cuda')
                       for _ in range(self.world_size)]
        dist.all_gather(all_tensors, local_tensor)

        # Deserialize on main process
        if self.is_main_process:
            all_results = []
            for tensor, size in zip(all_tensors, all_sizes):
                data = bytes(tensor[:size.item()].cpu().numpy())
                results = json.loads(data.decode())
                all_results.extend(results)
            return all_results

        return []

    def run(
        self,
        image_paths: List[Union[str, Path]],
        prompt: str = None
    ) -> Dict:
        """
        Run distributed pipeline.

        Args:
            image_paths: List of all image paths
            prompt: Grounding prompt

        Returns:
            Combined results (on main process only)
        """
        if not self.is_initialized:
            self.initialize()

        # Partition data
        local_paths = self.partition_data(image_paths)

        if self.is_main_process:
            print(f"\nDistributed Pipeline")
            print(f"  Total images: {len(image_paths)}")
            print(f"  World size: {self.world_size}")
            print(f"  Images per process: ~{len(image_paths) // self.world_size}")

        if self.rank < self.world_size:
            print(f"  [Rank {self.rank}] Processing {len(local_paths)} images")

        # Load prompt
        if prompt is None:
            prompt = get_prompt('grounding')

        # Process local partition
        local_results = self._process_local(local_paths, prompt)

        # Synchronize
        if self.config.enabled and dist.is_initialized():
            dist.barrier()

        # Gather results
        all_results = self.gather_results(local_results)

        # Build and save summary (main process only)
        summary = {}
        if self.is_main_process:
            summary = self._build_summary(image_paths, all_results)

            # Save combined results
            timestamp = get_timestamp()
            run_dir = self.output_dir / f"distributed_run_{timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)

            save_json(all_results, run_dir / "all_results.json")
            save_json(summary, run_dir / "summary.json")

            print(f"\nResults saved to: {run_dir}")

        return summary

    def _process_local(
        self,
        image_paths: List[Path],
        prompt: str
    ) -> List[Dict]:
        """Process local partition of images."""
        results = []

        for idx, path in enumerate(image_paths):
            try:
                response = self.model.generate(path, prompt)

                results.append({
                    'image_path': str(path),
                    'image_name': Path(path).name,
                    'response': response,
                    'rank': self.rank
                })

                if (idx + 1) % 10 == 0:
                    print(f"  [Rank {self.rank}] {idx + 1}/{len(image_paths)}")

            except Exception as e:
                print(f"  [Rank {self.rank}] Error on {path}: {e}")
                results.append({
                    'image_path': str(path),
                    'error': str(e),
                    'rank': self.rank
                })

        return results

    def _build_summary(
        self,
        all_paths: List[Path],
        all_results: List[Dict]
    ) -> Dict:
        """Build summary from gathered results."""
        successful = [r for r in all_results if 'error' not in r]
        failed = [r for r in all_results if 'error' in r]

        return {
            'pipeline': 'distributed',
            'world_size': self.world_size,
            'total_images': len(all_paths),
            'successful': len(successful),
            'failed': len(failed),
            'results_per_rank': {
                rank: len([r for r in all_results if r.get('rank') == rank])
                for rank in range(self.world_size)
            }
        }

    def cleanup(self):
        """Cleanup distributed resources."""
        if self.config.enabled and dist.is_initialized():
            dist.destroy_process_group()


def run_distributed_pipeline(
    model_factory,
    image_dir: Union[str, Path],
    output_dir: Union[str, Path] = "results",
    config: DistributedConfig = None,
    **kwargs
) -> Dict:
    """
    Convenience function to run distributed pipeline.

    Args:
        model_factory: Function that creates model instance
        image_dir: Directory containing images
        output_dir: Output directory
        config: DistributedConfig instance
        **kwargs: Additional pipeline arguments

    Returns:
        Summary dict (on main process)
    """
    images = list_images(image_dir)

    if not images:
        raise ValueError(f"No images found in {image_dir}")

    pipeline = DistributedPipeline(
        model_factory=model_factory,
        output_dir=output_dir,
        config=config,
        **kwargs
    )

    try:
        return pipeline.run(images)
    finally:
        pipeline.cleanup()


def launch_distributed(
    num_gpus: int,
    script_args: List[str],
    master_port: int = 29500
):
    """
    Launch distributed training using torchrun.

    Args:
        num_gpus: Number of GPUs to use
        script_args: Arguments to pass to the script
        master_port: Port for distributed communication
    """
    import subprocess
    import sys

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        f"--master_port={master_port}",
    ] + script_args

    print(f"Launching: {' '.join(cmd)}")
    subprocess.run(cmd)
