"""
Model caching and memory management utilities.

Provides intelligent model loading, offloading, and caching
to optimize GPU memory usage when switching between models.
"""

import gc
import threading
from typing import Dict, Optional, Any, Callable, Type
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import weakref

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelState(Enum):
    """Model loading states."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    ON_GPU = "on_gpu"
    ON_CPU = "on_cpu"
    ERROR = "error"


@dataclass
class CachedModel:
    """Container for cached model information."""
    name: str
    model: Any
    processor: Any = None
    state: ModelState = ModelState.UNLOADED
    device: str = "cuda"
    last_used: datetime = field(default_factory=datetime.now)
    load_count: int = 0

    def update_last_used(self):
        """Update last used timestamp."""
        self.last_used = datetime.now()
        self.load_count += 1


class ModelCache:
    """
    Global model cache for managing VLM model lifecycle.

    Features:
    - Lazy loading of models
    - CPU offloading instead of full unload
    - LRU eviction when memory constrained
    - Thread-safe operations

    Usage:
        cache = ModelCache.get_instance()

        # Get or load model
        model = cache.get_model("qwen", loader_func)

        # Offload to CPU when not needed
        cache.offload_to_cpu("qwen")

        # Move back to GPU when needed
        cache.move_to_gpu("qwen")
    """

    _instance: Optional['ModelCache'] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> 'ModelCache':
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.clear_all()
            cls._instance = None

    def __init__(self):
        """Initialize model cache."""
        self._models: Dict[str, CachedModel] = {}
        self._lock = threading.RLock()
        self._max_gpu_models = 1  # Default: only one model on GPU at a time
        self._max_cpu_models = 2  # Keep up to 2 models on CPU

    def register_model(
        self,
        name: str,
        model: Any,
        processor: Any = None,
        device: str = "cuda"
    ):
        """
        Register a model in the cache.

        Args:
            name: Model identifier
            model: Model instance
            processor: Optional processor/tokenizer
            device: Current device
        """
        with self._lock:
            state = ModelState.ON_GPU if device == "cuda" else ModelState.ON_CPU

            self._models[name] = CachedModel(
                name=name,
                model=model,
                processor=processor,
                state=state,
                device=device
            )

    def get_model(
        self,
        name: str,
        loader_func: Callable[[], tuple] = None,
        move_to_gpu: bool = True
    ) -> tuple:
        """
        Get a model from cache, loading if necessary.

        Args:
            name: Model identifier
            loader_func: Function that returns (model, processor) tuple
            move_to_gpu: Move to GPU if on CPU

        Returns:
            Tuple of (model, processor)
        """
        with self._lock:
            if name in self._models:
                cached = self._models[name]
                cached.update_last_used()

                if move_to_gpu and cached.state == ModelState.ON_CPU:
                    self._ensure_gpu_space(exclude=name)
                    self._move_to_device(name, "cuda")

                return cached.model, cached.processor

            # Load new model
            if loader_func is None:
                raise ValueError(f"Model '{name}' not cached and no loader provided")

            # Ensure GPU space before loading
            if move_to_gpu:
                self._ensure_gpu_space()

            model, processor = loader_func()

            self.register_model(
                name=name,
                model=model,
                processor=processor,
                device="cuda" if move_to_gpu else "cpu"
            )

            return model, processor

    def offload_to_cpu(self, name: str):
        """
        Move model from GPU to CPU.

        Args:
            name: Model identifier
        """
        with self._lock:
            if name not in self._models:
                return

            self._move_to_device(name, "cpu")

    def move_to_gpu(self, name: str):
        """
        Move model from CPU to GPU.

        Args:
            name: Model identifier
        """
        with self._lock:
            if name not in self._models:
                raise ValueError(f"Model '{name}' not in cache")

            self._ensure_gpu_space(exclude=name)
            self._move_to_device(name, "cuda")

    def _move_to_device(self, name: str, device: str):
        """Internal: move model to device."""
        if not TORCH_AVAILABLE:
            return

        cached = self._models[name]

        if cached.model is None:
            return

        try:
            if hasattr(cached.model, 'to'):
                cached.model.to(device)

            cached.device = device
            cached.state = ModelState.ON_GPU if device == "cuda" else ModelState.ON_CPU

            # Clear CUDA cache after offloading
            if device == "cpu":
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            cached.state = ModelState.ERROR
            raise RuntimeError(f"Failed to move model to {device}: {e}")

    def _ensure_gpu_space(self, exclude: str = None):
        """Ensure there's GPU space by offloading old models."""
        gpu_models = [
            name for name, m in self._models.items()
            if m.state == ModelState.ON_GPU and name != exclude
        ]

        if len(gpu_models) >= self._max_gpu_models:
            # Offload least recently used
            gpu_models.sort(key=lambda n: self._models[n].last_used)
            for name in gpu_models[:len(gpu_models) - self._max_gpu_models + 1]:
                self.offload_to_cpu(name)

    def unload_model(self, name: str):
        """
        Completely unload a model from memory.

        Args:
            name: Model identifier
        """
        with self._lock:
            if name not in self._models:
                return

            cached = self._models[name]

            # Delete model and processor
            if cached.model is not None:
                del cached.model
            if cached.processor is not None:
                del cached.processor

            del self._models[name]

            # Force garbage collection
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def clear_all(self):
        """Unload all cached models."""
        with self._lock:
            names = list(self._models.keys())
            for name in names:
                self.unload_model(name)

    def get_status(self) -> Dict[str, Dict]:
        """Get status of all cached models."""
        with self._lock:
            return {
                name: {
                    'state': cached.state.value,
                    'device': cached.device,
                    'last_used': cached.last_used.isoformat(),
                    'load_count': cached.load_count
                }
                for name, cached in self._models.items()
            }

    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory usage info."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {'available': False}

        return {
            'available': True,
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
        }


def clear_gpu_memory():
    """Force clear GPU memory."""
    gc.collect()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_optimal_batch_size(
    model_memory_gb: float,
    available_memory_gb: float = None,
    sample_memory_mb: float = 100,
    safety_factor: float = 0.8
) -> int:
    """
    Calculate optimal batch size based on available memory.

    Args:
        model_memory_gb: Estimated model memory usage in GB
        available_memory_gb: Available GPU memory (auto-detect if None)
        sample_memory_mb: Estimated memory per sample in MB
        safety_factor: Safety margin (0.8 = use 80% of available)

    Returns:
        Recommended batch size
    """
    if available_memory_gb is None:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated = torch.cuda.memory_allocated() / 1e9
            available_memory_gb = (total - allocated) * safety_factor
        else:
            return 1

    remaining = available_memory_gb - model_memory_gb
    if remaining <= 0:
        return 1

    batch_size = int(remaining * 1024 / sample_memory_mb)
    return max(1, batch_size)


class ModelContext:
    """
    Context manager for temporary model usage.

    Usage:
        with ModelContext("sa2va", loader_func) as (model, processor):
            result = model.generate(...)
        # Model automatically offloaded after context
    """

    def __init__(
        self,
        name: str,
        loader_func: Callable[[], tuple] = None,
        offload_after: bool = True
    ):
        """
        Initialize model context.

        Args:
            name: Model identifier
            loader_func: Function to load model if not cached
            offload_after: Offload to CPU after context exits
        """
        self.name = name
        self.loader_func = loader_func
        self.offload_after = offload_after
        self._cache = ModelCache.get_instance()

    def __enter__(self) -> tuple:
        """Enter context, loading/moving model to GPU."""
        return self._cache.get_model(self.name, self.loader_func, move_to_gpu=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context, optionally offloading model."""
        if self.offload_after:
            self._cache.offload_to_cpu(self.name)
