"""
Image prefetching utilities for pipeline optimization.

Prefetches images in background threads while GPU processes current batch,
hiding I/O latency behind computation.
"""

import threading
from queue import Queue, Empty
from typing import List, Tuple, Optional, Iterator, Union, Callable, Any
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
import time


@dataclass
class PrefetchedItem:
    """Container for prefetched data."""
    path: Path
    image: Image.Image
    index: int
    metadata: dict = None


class ImagePrefetcher:
    """
    Prefetches images in a background thread.

    Usage:
        prefetcher = ImagePrefetcher(image_paths, prefetch_size=8)
        for path, image in prefetcher:
            # Process image while next ones are being loaded
            process(image)
    """

    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        prefetch_size: int = 8,
        convert_mode: str = "RGB",
        transform: Callable[[Image.Image], Image.Image] = None
    ):
        """
        Initialize prefetcher.

        Args:
            image_paths: List of image paths to prefetch
            prefetch_size: Number of images to keep prefetched
            convert_mode: PIL image mode (default: RGB)
            transform: Optional transform to apply to images
        """
        self.paths = [Path(p) for p in image_paths]
        self.prefetch_size = prefetch_size
        self.convert_mode = convert_mode
        self.transform = transform

        self._queue: Queue[Optional[PrefetchedItem]] = Queue(maxsize=prefetch_size)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._error: Optional[Exception] = None

    def _worker(self):
        """Background worker that loads images."""
        try:
            for idx, path in enumerate(self.paths):
                if self._stop_event.is_set():
                    break

                try:
                    img = Image.open(path).convert(self.convert_mode)

                    if self.transform is not None:
                        img = self.transform(img)

                    item = PrefetchedItem(
                        path=path,
                        image=img,
                        index=idx
                    )
                    self._queue.put(item)

                except Exception as e:
                    # Put error item
                    item = PrefetchedItem(
                        path=path,
                        image=None,
                        index=idx,
                        metadata={'error': str(e)}
                    )
                    self._queue.put(item)

            # Signal end of iteration
            self._queue.put(None)

        except Exception as e:
            self._error = e
            self._queue.put(None)

    def start(self):
        """Start the prefetch worker thread."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._worker, daemon=True)
            self._worker_thread.start()
        return self

    def stop(self):
        """Stop the prefetch worker."""
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)

    def __iter__(self) -> Iterator[Tuple[Path, Image.Image]]:
        """Iterate over prefetched images."""
        self.start()

        while True:
            try:
                item = self._queue.get(timeout=30.0)
            except Empty:
                raise TimeoutError("Prefetch queue timeout")

            if item is None:
                break

            if item.image is None:
                # Skip failed loads
                continue

            yield item.path, item.image

        if self._error is not None:
            raise self._error

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __len__(self) -> int:
        return len(self.paths)


class BatchPrefetcher:
    """
    Prefetches images in batches for batch processing.

    Usage:
        prefetcher = BatchPrefetcher(image_paths, batch_size=4, prefetch_batches=2)
        for batch in prefetcher:
            # Process batch while next ones are being loaded
            model.batch_generate([img for _, img in batch])
    """

    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 4,
        prefetch_batches: int = 2,
        convert_mode: str = "RGB"
    ):
        """
        Initialize batch prefetcher.

        Args:
            image_paths: List of image paths
            batch_size: Number of images per batch
            prefetch_batches: Number of batches to keep prefetched
            convert_mode: PIL image mode
        """
        self.paths = [Path(p) for p in image_paths]
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        self.convert_mode = convert_mode

        self._queue: Queue[Optional[List[Tuple[Path, Image.Image]]]] = Queue(
            maxsize=prefetch_batches
        )
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._error: Optional[Exception] = None

    def _worker(self):
        """Background worker that loads image batches."""
        try:
            batch = []

            for path in self.paths:
                if self._stop_event.is_set():
                    break

                try:
                    img = Image.open(path).convert(self.convert_mode)
                    batch.append((path, img))
                except Exception as e:
                    # Skip failed loads
                    continue

                if len(batch) >= self.batch_size:
                    self._queue.put(batch)
                    batch = []

            # Put remaining items as final batch
            if batch:
                self._queue.put(batch)

            # Signal end
            self._queue.put(None)

        except Exception as e:
            self._error = e
            self._queue.put(None)

    def start(self):
        """Start the prefetch worker thread."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._worker, daemon=True)
            self._worker_thread.start()
        return self

    def stop(self):
        """Stop the prefetch worker."""
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)

    def __iter__(self) -> Iterator[List[Tuple[Path, Image.Image]]]:
        """Iterate over prefetched batches."""
        self.start()

        while True:
            try:
                batch = self._queue.get(timeout=60.0)
            except Empty:
                raise TimeoutError("Batch prefetch queue timeout")

            if batch is None:
                break

            yield batch

        if self._error is not None:
            raise self._error

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def num_batches(self) -> int:
        """Estimate number of batches."""
        return (len(self.paths) + self.batch_size - 1) // self.batch_size


class AsyncDataLoader:
    """
    Asynchronous data loader with double buffering.

    Loads the next batch while current batch is being processed.
    """

    def __init__(
        self,
        data_items: List[Any],
        load_func: Callable[[Any], Any],
        buffer_size: int = 2
    ):
        """
        Initialize async loader.

        Args:
            data_items: List of items to load
            load_func: Function to load each item
            buffer_size: Number of items to buffer
        """
        self.data_items = data_items
        self.load_func = load_func
        self.buffer_size = buffer_size

        self._queue: Queue[Optional[Tuple[int, Any]]] = Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    def _worker(self):
        """Background loader thread."""
        for idx, item in enumerate(self.data_items):
            if self._stop_event.is_set():
                break

            try:
                loaded = self.load_func(item)
                self._queue.put((idx, loaded))
            except Exception as e:
                self._queue.put((idx, None))

        self._queue.put(None)

    def __iter__(self) -> Iterator[Tuple[int, Any]]:
        """Iterate over loaded items."""
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

        while True:
            item = self._queue.get()
            if item is None:
                break
            yield item

    def __len__(self) -> int:
        return len(self.data_items)
