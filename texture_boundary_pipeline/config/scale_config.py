"""
Scalability configuration for the texture boundary pipeline.

Provides centralized configuration for parallel processing, batching,
memory management, and distributed execution.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import json


@dataclass
class ProcessingConfig:
    """Configuration for parallel processing."""

    # VLM batch processing
    vlm_batch_size: int = 4
    """Number of images to process in a single VLM batch."""

    # Worker pools
    io_workers: int = field(default_factory=lambda: min(32, (os.cpu_count() or 1) * 4))
    """Number of I/O worker threads for file operations."""

    cpu_workers: int = field(default_factory=lambda: max(1, (os.cpu_count() or 1) - 1))
    """Number of CPU workers for compute-bound tasks."""

    # Prefetching
    prefetch_size: int = 8
    """Number of images to prefetch while GPU processes."""

    prefetch_batches: int = 2
    """Number of batches to keep prefetched."""

    # Crop saving
    parallel_save: bool = True
    """Enable parallel crop saving."""

    save_workers: int = 8
    """Number of workers for parallel saving."""

    jpeg_quality: int = 95
    """JPEG quality for saved crops."""


@dataclass
class MemoryConfig:
    """Configuration for memory management."""

    # Model management
    offload_to_cpu: bool = True
    """Offload models to CPU instead of unloading."""

    max_gpu_models: int = 1
    """Maximum models to keep on GPU simultaneously."""

    max_cpu_models: int = 2
    """Maximum models to keep on CPU."""

    # Memory optimization
    gradient_checkpointing: bool = False
    """Enable gradient checkpointing (reduces memory, slower)."""

    clear_cache_interval: int = 10
    """Clear CUDA cache every N batches."""

    # Batch size auto-tuning
    auto_batch_size: bool = False
    """Automatically determine optimal batch size."""

    model_memory_gb: float = 16.0
    """Estimated model memory usage for batch size calculation."""

    sample_memory_mb: float = 100.0
    """Estimated memory per sample in MB."""


@dataclass
class DistributedConfig:
    """Configuration for distributed processing."""

    enabled: bool = False
    """Enable distributed processing."""

    backend: str = "nccl"
    """PyTorch distributed backend (nccl, gloo, mpi)."""

    world_size: int = 1
    """Total number of processes."""

    rank: int = 0
    """Current process rank."""

    local_rank: int = 0
    """Local GPU rank."""

    master_addr: str = "localhost"
    """Master node address."""

    master_port: int = 29500
    """Master node port."""

    init_method: str = "env://"
    """Initialization method for distributed."""


@dataclass
class OutputConfig:
    """Configuration for output and streaming."""

    streaming: bool = True
    """Use streaming JSON Lines output."""

    checkpoint_interval: int = 50
    """Save checkpoint every N records."""

    compress_output: bool = False
    """Compress output files with gzip."""

    buffer_size: int = 100
    """Buffer size for streaming writes."""

    async_io: bool = True
    """Use asynchronous I/O for writing."""


@dataclass
class ScaleConfig:
    """Main configuration container for scalability settings."""

    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    """Parallel processing configuration."""

    memory: MemoryConfig = field(default_factory=MemoryConfig)
    """Memory management configuration."""

    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    """Distributed processing configuration."""

    output: OutputConfig = field(default_factory=OutputConfig)
    """Output and streaming configuration."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScaleConfig':
        """Create config from dictionary."""
        return cls(
            processing=ProcessingConfig(**data.get('processing', {})),
            memory=MemoryConfig(**data.get('memory', {})),
            distributed=DistributedConfig(**data.get('distributed', {})),
            output=OutputConfig(**data.get('output', {}))
        )

    @classmethod
    def from_json(cls, path: str) -> 'ScaleConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, path: str) -> 'ScaleConfig':
        """Load config from YAML file."""
        try:
            import yaml
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data)
        except ImportError:
            raise ImportError("PyYAML required for YAML config files")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def save_json(self, path: str):
        """Save config to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_yaml(self, path: str):
        """Save config to YAML file."""
        try:
            import yaml
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        except ImportError:
            raise ImportError("PyYAML required for YAML config files")


# Preset configurations
PRESETS: Dict[str, ScaleConfig] = {}


def _create_presets():
    """Create preset configurations."""
    global PRESETS

    # Single GPU, small dataset
    PRESETS['small'] = ScaleConfig(
        processing=ProcessingConfig(
            vlm_batch_size=2,
            prefetch_size=4,
            io_workers=8
        ),
        memory=MemoryConfig(
            offload_to_cpu=False,
            clear_cache_interval=20
        ),
        output=OutputConfig(
            streaming=False,
            checkpoint_interval=100
        )
    )

    # Single GPU, large dataset
    PRESETS['large'] = ScaleConfig(
        processing=ProcessingConfig(
            vlm_batch_size=4,
            prefetch_size=16,
            io_workers=16,
            parallel_save=True,
            save_workers=12
        ),
        memory=MemoryConfig(
            offload_to_cpu=True,
            clear_cache_interval=5
        ),
        output=OutputConfig(
            streaming=True,
            checkpoint_interval=50,
            async_io=True
        )
    )

    # Multi-GPU
    PRESETS['multi_gpu'] = ScaleConfig(
        processing=ProcessingConfig(
            vlm_batch_size=8,
            prefetch_size=32,
            io_workers=32
        ),
        memory=MemoryConfig(
            offload_to_cpu=True,
            max_gpu_models=1
        ),
        distributed=DistributedConfig(
            enabled=True,
            backend="nccl"
        ),
        output=OutputConfig(
            streaming=True,
            checkpoint_interval=25
        )
    )

    # Memory constrained (e.g., 8GB VRAM)
    PRESETS['low_memory'] = ScaleConfig(
        processing=ProcessingConfig(
            vlm_batch_size=1,
            prefetch_size=4
        ),
        memory=MemoryConfig(
            offload_to_cpu=True,
            clear_cache_interval=1,
            gradient_checkpointing=True
        ),
        output=OutputConfig(
            streaming=True
        )
    )

    # Maximum throughput (high-end GPU)
    PRESETS['max_throughput'] = ScaleConfig(
        processing=ProcessingConfig(
            vlm_batch_size=8,
            prefetch_size=32,
            io_workers=32,
            cpu_workers=16,
            parallel_save=True,
            save_workers=16
        ),
        memory=MemoryConfig(
            offload_to_cpu=False,
            clear_cache_interval=10,
            auto_batch_size=True
        ),
        output=OutputConfig(
            streaming=True,
            async_io=True,
            buffer_size=200
        )
    )


# Initialize presets
_create_presets()


def get_preset(name: str) -> ScaleConfig:
    """
    Get a preset configuration.

    Args:
        name: Preset name ('small', 'large', 'multi_gpu', 'low_memory', 'max_throughput')

    Returns:
        ScaleConfig instance
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]


def get_default_config() -> ScaleConfig:
    """Get default configuration."""
    return ScaleConfig()


# Global config instance
_global_config: Optional[ScaleConfig] = None


def get_config() -> ScaleConfig:
    """Get global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = ScaleConfig()
    return _global_config


def set_config(config: ScaleConfig):
    """Set global configuration instance."""
    global _global_config
    _global_config = config


def configure_from_preset(name: str):
    """Configure globally from preset."""
    set_config(get_preset(name))


def configure_from_file(path: str):
    """Configure globally from file."""
    path = Path(path)
    if path.suffix in ('.yaml', '.yml'):
        set_config(ScaleConfig.from_yaml(str(path)))
    else:
        set_config(ScaleConfig.from_json(str(path)))
