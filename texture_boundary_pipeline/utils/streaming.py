"""
Streaming I/O utilities for large-scale data processing.

Provides incremental writing of results to avoid memory buildup
and enable progress recovery.
"""

import json
from pathlib import Path
from typing import Union, Dict, List, Any, Optional, Iterator
from datetime import datetime
import threading
from queue import Queue, Empty
import gzip
import os


class StreamingJsonWriter:
    """
    Writes JSON records incrementally using JSON Lines format.

    Each record is written as a separate line, allowing for:
    - Incremental writing without loading all data
    - Easy recovery if processing is interrupted
    - Streaming reads of large result files

    Usage:
        with StreamingJsonWriter("results.jsonl") as writer:
            for item in process_items():
                writer.write(item)
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        compress: bool = False,
        buffer_size: int = 100
    ):
        """
        Initialize streaming writer.

        Args:
            output_path: Path to output file (.jsonl or .jsonl.gz)
            compress: Use gzip compression
            buffer_size: Number of records to buffer before flushing
        """
        self.output_path = Path(output_path)
        self.compress = compress
        self.buffer_size = buffer_size

        self._buffer: List[str] = []
        self._file = None
        self._count = 0
        self._lock = threading.Lock()

    def open(self):
        """Open the output file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.compress:
            self._file = gzip.open(self.output_path, 'wt', encoding='utf-8')
        else:
            self._file = open(self.output_path, 'w', encoding='utf-8')

        return self

    def write(self, record: Dict[str, Any]):
        """
        Write a single record.

        Args:
            record: Dictionary to write as JSON line
        """
        with self._lock:
            line = json.dumps(record, ensure_ascii=False)
            self._buffer.append(line)
            self._count += 1

            if len(self._buffer) >= self.buffer_size:
                self._flush()

    def write_batch(self, records: List[Dict[str, Any]]):
        """
        Write multiple records at once.

        Args:
            records: List of dictionaries to write
        """
        with self._lock:
            for record in records:
                line = json.dumps(record, ensure_ascii=False)
                self._buffer.append(line)
                self._count += 1

            if len(self._buffer) >= self.buffer_size:
                self._flush()

    def _flush(self):
        """Flush buffer to file."""
        if self._file is not None and self._buffer:
            self._file.write('\n'.join(self._buffer) + '\n')
            self._file.flush()
            self._buffer.clear()

    def close(self):
        """Close the file, flushing any remaining buffer."""
        with self._lock:
            self._flush()
            if self._file is not None:
                self._file.close()
                self._file = None

    @property
    def count(self) -> int:
        """Number of records written."""
        return self._count

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class StreamingJsonReader:
    """
    Reads JSON Lines files incrementally.

    Usage:
        for record in StreamingJsonReader("results.jsonl"):
            process(record)
    """

    def __init__(
        self,
        input_path: Union[str, Path],
        skip_errors: bool = True
    ):
        """
        Initialize streaming reader.

        Args:
            input_path: Path to input file
            skip_errors: Skip malformed lines instead of raising
        """
        self.input_path = Path(input_path)
        self.skip_errors = skip_errors

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over records."""
        is_gzipped = self.input_path.suffix == '.gz'

        opener = gzip.open if is_gzipped else open

        with opener(self.input_path, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    if not self.skip_errors:
                        raise ValueError(f"Line {line_num}: {e}")

    def count(self) -> int:
        """Count total records without loading all into memory."""
        return sum(1 for _ in self)


class CheckpointWriter:
    """
    Writes results with checkpointing for recovery.

    Periodically saves progress so processing can resume if interrupted.

    Usage:
        with CheckpointWriter("results", checkpoint_interval=100) as writer:
            for i, item in enumerate(items):
                if writer.should_skip(i):
                    continue  # Already processed
                result = process(item)
                writer.write(i, result)
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        checkpoint_interval: int = 100,
        name: str = "results"
    ):
        """
        Initialize checkpoint writer.

        Args:
            output_dir: Directory for output files
            checkpoint_interval: Save checkpoint every N records
            name: Base name for output files
        """
        self.output_dir = Path(output_dir)
        self.checkpoint_interval = checkpoint_interval
        self.name = name

        self._results_path = self.output_dir / f"{name}.jsonl"
        self._checkpoint_path = self.output_dir / f"{name}.checkpoint.json"

        self._writer: Optional[StreamingJsonWriter] = None
        self._processed_indices: set = set()
        self._count = 0

    def open(self):
        """Open writer and load checkpoint if exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load existing checkpoint
        if self._checkpoint_path.exists():
            with open(self._checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
                self._processed_indices = set(checkpoint.get('processed', []))
                self._count = checkpoint.get('count', 0)

        # Open results file in append mode if resuming
        mode = 'a' if self._processed_indices else 'w'
        self._results_path.parent.mkdir(parents=True, exist_ok=True)

        # Use raw file for append support
        self._file = open(self._results_path, mode, encoding='utf-8')

        return self

    def should_skip(self, index: int) -> bool:
        """Check if index was already processed."""
        return index in self._processed_indices

    def write(self, index: int, record: Dict[str, Any]):
        """
        Write a record with its index.

        Args:
            index: Processing index
            record: Result record
        """
        record_with_index = {'_index': index, **record}
        line = json.dumps(record_with_index, ensure_ascii=False)
        self._file.write(line + '\n')

        self._processed_indices.add(index)
        self._count += 1

        # Checkpoint periodically
        if self._count % self.checkpoint_interval == 0:
            self._save_checkpoint()

    def _save_checkpoint(self):
        """Save checkpoint file."""
        self._file.flush()

        checkpoint = {
            'processed': list(self._processed_indices),
            'count': self._count,
            'timestamp': datetime.now().isoformat()
        }

        # Write atomically
        temp_path = self._checkpoint_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(checkpoint, f)
        temp_path.rename(self._checkpoint_path)

    def close(self):
        """Close writer and save final checkpoint."""
        if self._file is not None:
            self._save_checkpoint()
            self._file.close()
            self._file = None

    @property
    def count(self) -> int:
        """Number of records written."""
        return self._count

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AsyncStreamingWriter:
    """
    Asynchronous streaming writer with background I/O thread.

    Writes happen in background thread to avoid blocking main processing.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        queue_size: int = 1000
    ):
        """
        Initialize async writer.

        Args:
            output_path: Path to output file
            queue_size: Maximum queue size before blocking
        """
        self.output_path = Path(output_path)
        self.queue_size = queue_size

        self._queue: Queue[Optional[Dict]] = Queue(maxsize=queue_size)
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._count = 0
        self._error: Optional[Exception] = None

    def _worker(self):
        """Background writer thread."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                while not self._stop_event.is_set():
                    try:
                        record = self._queue.get(timeout=0.1)
                    except Empty:
                        continue

                    if record is None:
                        break

                    line = json.dumps(record, ensure_ascii=False)
                    f.write(line + '\n')

                # Drain remaining queue
                while not self._queue.empty():
                    record = self._queue.get_nowait()
                    if record is not None:
                        line = json.dumps(record, ensure_ascii=False)
                        f.write(line + '\n')

        except Exception as e:
            self._error = e

    def start(self):
        """Start the background writer thread."""
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        return self

    def write(self, record: Dict[str, Any]):
        """
        Queue a record for writing.

        Args:
            record: Dictionary to write
        """
        self._queue.put(record)
        self._count += 1

    def stop(self, wait: bool = True):
        """Stop the writer thread."""
        self._queue.put(None)  # Signal stop
        self._stop_event.set()

        if wait and self._worker_thread is not None:
            self._worker_thread.join(timeout=30.0)

        if self._error is not None:
            raise self._error

    @property
    def count(self) -> int:
        return self._count

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def convert_jsonl_to_json(
    jsonl_path: Union[str, Path],
    json_path: Union[str, Path] = None
) -> Path:
    """
    Convert JSON Lines file to standard JSON array.

    Args:
        jsonl_path: Path to input JSONL file
        json_path: Path to output JSON file (default: same name with .json)

    Returns:
        Path to output file
    """
    jsonl_path = Path(jsonl_path)

    if json_path is None:
        json_path = jsonl_path.with_suffix('.json')
    else:
        json_path = Path(json_path)

    records = list(StreamingJsonReader(jsonl_path))

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    return json_path
