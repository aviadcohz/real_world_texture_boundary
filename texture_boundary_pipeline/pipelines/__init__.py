"""
Pipelines Package

Pipeline orchestrators for texture boundary detection.

Available pipelines:
- BasicPipeline: Simple sequential processing
- IterativePipeline: Multi-stage with entropy filtering
- ScalablePipeline: Optimized for large datasets with batching and prefetching
- DistributedPipeline: Multi-GPU and multi-node processing
"""

from .basic_pipeline import BasicPipeline, run_basic_pipeline
from .iterative_pipeline import IterativePipeline, run_iterative_pipeline

# Scalable pipeline (requires scale utilities)
try:
    from .scalable_pipeline import ScalablePipeline, run_scalable_pipeline
    SCALABLE_AVAILABLE = True
except ImportError:
    SCALABLE_AVAILABLE = False

# Distributed pipeline (requires torch.distributed)
try:
    from .distributed_pipeline import (
        DistributedPipeline,
        run_distributed_pipeline,
        launch_distributed
    )
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

# Dual-GPU pipeline (local + remote server)
try:
    from .dual_gpu_pipeline import DualGPUPipeline, run_dual_gpu_pipeline
    from .iterative_dual_gpu_pipeline import IterativeDualGPUPipeline, run_iterative_dual_gpu_pipeline
    DUAL_GPU_AVAILABLE = True
except ImportError:
    DUAL_GPU_AVAILABLE = False

__all__ = [
    # Basic pipelines
    'BasicPipeline',
    'run_basic_pipeline',
    'IterativePipeline',
    'run_iterative_pipeline',

    # Scalable pipeline
    'ScalablePipeline',
    'run_scalable_pipeline',

    # Distributed pipeline
    'DistributedPipeline',
    'run_distributed_pipeline',
    'launch_distributed',

    # Dual-GPU pipeline
    'DualGPUPipeline',
    'run_dual_gpu_pipeline',
    'IterativeDualGPUPipeline',
    'run_iterative_dual_gpu_pipeline',

    # Availability flags
    'SCALABLE_AVAILABLE',
    'DISTRIBUTED_AVAILABLE',
    'DUAL_GPU_AVAILABLE',
]
