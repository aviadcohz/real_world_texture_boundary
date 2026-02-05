"""
Parallel processing utilities for scalable pipeline execution.

Provides thread and process pools for CPU-bound and I/O-bound operations.
"""

import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Callable, Any, Dict, Tuple, Optional, Iterator
from functools import partial
from pathlib import Path
import multiprocessing as mp
from PIL import Image
import numpy as np


# Default worker counts
DEFAULT_IO_WORKERS = min(32, (os.cpu_count() or 1) * 4)  # I/O bound tasks
DEFAULT_CPU_WORKERS = max(1, (os.cpu_count() or 1) - 1)  # CPU bound tasks


class ParallelProcessor:
    """
    Manages parallel processing with configurable worker pools.

    Usage:
        processor = ParallelProcessor(io_workers=8, cpu_workers=4)
        results = processor.map_io(load_image, image_paths)
        processor.shutdown()
    """

    def __init__(
        self,
        io_workers: int = None,
        cpu_workers: int = None,
        use_processes: bool = False
    ):
        """
        Initialize parallel processor.

        Args:
            io_workers: Number of I/O workers (threads for file operations)
            cpu_workers: Number of CPU workers (for compute-bound tasks)
            use_processes: Use ProcessPoolExecutor instead of ThreadPoolExecutor
        """
        self.io_workers = io_workers or DEFAULT_IO_WORKERS
        self.cpu_workers = cpu_workers or DEFAULT_CPU_WORKERS
        self.use_processes = use_processes

        self._io_pool = None
        self._cpu_pool = None

    @property
    def io_pool(self) -> ThreadPoolExecutor:
        """Lazy initialization of I/O thread pool."""
        if self._io_pool is None:
            self._io_pool = ThreadPoolExecutor(max_workers=self.io_workers)
        return self._io_pool

    @property
    def cpu_pool(self):
        """Lazy initialization of CPU worker pool."""
        if self._cpu_pool is None:
            if self.use_processes:
                self._cpu_pool = ProcessPoolExecutor(max_workers=self.cpu_workers)
            else:
                self._cpu_pool = ThreadPoolExecutor(max_workers=self.cpu_workers)
        return self._cpu_pool

    def map_io(
        self,
        func: Callable,
        items: List[Any],
        ordered: bool = True
    ) -> List[Any]:
        """
        Map function over items using I/O thread pool.

        Args:
            func: Function to apply
            items: List of items to process
            ordered: Return results in input order (default True)

        Returns:
            List of results
        """
        if not items:
            return []

        futures = [self.io_pool.submit(func, item) for item in items]

        if ordered:
            return [f.result() for f in futures]
        else:
            return [f.result() for f in as_completed(futures)]

    def map_cpu(
        self,
        func: Callable,
        items: List[Any],
        ordered: bool = True
    ) -> List[Any]:
        """
        Map function over items using CPU worker pool.

        Args:
            func: Function to apply
            items: List of items to process
            ordered: Return results in input order (default True)

        Returns:
            List of results
        """
        if not items:
            return []

        futures = [self.cpu_pool.submit(func, item) for item in items]

        if ordered:
            return [f.result() for f in futures]
        else:
            return [f.result() for f in as_completed(futures)]

    def map_with_progress(
        self,
        func: Callable,
        items: List[Any],
        pool_type: str = "io",
        desc: str = "Processing"
    ) -> Iterator[Tuple[int, Any]]:
        """
        Map function with progress yielding.

        Args:
            func: Function to apply
            items: List of items to process
            pool_type: "io" or "cpu"
            desc: Description for progress

        Yields:
            Tuple of (completed_count, result)
        """
        pool = self.io_pool if pool_type == "io" else self.cpu_pool
        futures = {pool.submit(func, item): i for i, item in enumerate(items)}
        results = [None] * len(items)
        completed = 0

        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            completed += 1
            yield completed, results[idx]

    def shutdown(self, wait: bool = True):
        """Shutdown all pools."""
        if self._io_pool is not None:
            self._io_pool.shutdown(wait=wait)
            self._io_pool = None
        if self._cpu_pool is not None:
            self._cpu_pool.shutdown(wait=wait)
            self._cpu_pool = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


def parallel_save_crops(
    crops_data: List[Tuple[Image.Image, Path, int]],
    max_workers: int = None
) -> List[Path]:
    """
    Save multiple crop images in parallel.

    Args:
        crops_data: List of (crop_image, output_path, quality) tuples
        max_workers: Maximum worker threads

    Returns:
        List of saved paths
    """
    max_workers = max_workers or DEFAULT_IO_WORKERS

    def save_single(data: Tuple[Image.Image, Path, int]) -> Path:
        image, path, quality = data
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, quality=quality)
        return path

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(save_single, crops_data))

    return results


def parallel_load_images(
    image_paths: List[Path],
    max_workers: int = None,
    convert_mode: str = "RGB"
) -> List[Tuple[Path, Image.Image]]:
    """
    Load multiple images in parallel.

    Args:
        image_paths: List of image paths to load
        max_workers: Maximum worker threads
        convert_mode: PIL image mode to convert to

    Returns:
        List of (path, image) tuples
    """
    max_workers = max_workers or DEFAULT_IO_WORKERS

    def load_single(path: Path) -> Tuple[Path, Image.Image]:
        img = Image.open(path).convert(convert_mode)
        return (path, img)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(load_single, image_paths))

    return results


def parallel_copy_files(
    file_pairs: List[Tuple[Path, Path]],
    max_workers: int = None
) -> List[Path]:
    """
    Copy multiple files in parallel.

    Args:
        file_pairs: List of (source, destination) path tuples
        max_workers: Maximum worker threads

    Returns:
        List of destination paths
    """
    import shutil
    max_workers = max_workers or DEFAULT_IO_WORKERS

    def copy_single(pair: Tuple[Path, Path]) -> Path:
        src, dst = pair
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(copy_single, file_pairs))

    return results


def batch_process(
    items: List[Any],
    batch_size: int,
    process_func: Callable[[List[Any]], List[Any]]
) -> List[Any]:
    """
    Process items in batches.

    Args:
        items: List of items to process
        batch_size: Size of each batch
        process_func: Function that processes a batch and returns results

    Returns:
        Flattened list of all results
    """
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_func(batch)
        results.extend(batch_results)
    return results


def chunked_iterator(items: List[Any], chunk_size: int) -> Iterator[List[Any]]:
    """
    Yield items in chunks.

    Args:
        items: List of items
        chunk_size: Size of each chunk

    Yields:
        Chunks of items
    """
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]
