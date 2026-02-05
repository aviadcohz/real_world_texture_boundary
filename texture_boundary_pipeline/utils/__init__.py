# Image utilities
from .image_utils import (
    pad_image_to_standard_size,
    calculate_optimal_target_size,
    resize_image,
    load_image,
    get_image_size,
    save_image,
    crop_image
)

# Bbox utilities
from .bbox_utils import (
    parse_json_format,
    adjust_bboxes_after_padding,
    clamp_bbox,
    bbox_area,
    bbox_dimensions,
    is_valid_bbox,
    calculate_iou,
    convert_coords_format,
    format_bbox_for_json,
    get_bbox_center
)

# I/O utilities
from .io_utils import (
    load_json,
    save_json,
    get_prompt,
    create_output_directory,
    get_timestamp,
    create_results_filename,
    list_images,
    extract_test_name_from_path,
    ensure_directory,
    file_exists,
    read_text_file,
    write_text_file,
    get_relative_path,
    copy_file
)

# Dataset utilities
from .dataset_collector import (
    collect_texture_crops,
    create_dataset_from_multiple_runs
)

# Parallel processing utilities
try:
    from .parallel import (
        ParallelProcessor,
        parallel_save_crops,
        parallel_load_images,
        parallel_copy_files,
        batch_process,
        chunked_iterator
    )
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False

# Prefetching utilities
try:
    from .prefetch import (
        ImagePrefetcher,
        BatchPrefetcher,
        AsyncDataLoader
    )
    PREFETCH_AVAILABLE = True
except ImportError:
    PREFETCH_AVAILABLE = False

# Streaming I/O utilities
try:
    from .streaming import (
        StreamingJsonWriter,
        StreamingJsonReader,
        CheckpointWriter,
        AsyncStreamingWriter,
        convert_jsonl_to_json
    )
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False

# Model cache utilities
try:
    from .model_cache import (
        ModelCache,
        ModelContext,
        clear_gpu_memory,
        get_optimal_batch_size
    )
    MODEL_CACHE_AVAILABLE = True
except ImportError:
    MODEL_CACHE_AVAILABLE = False

__all__ = [
    # Image utilities
    'pad_image_to_standard_size',
    'calculate_optimal_target_size',
    'resize_image',
    'load_image',
    'get_image_size',
    'save_image',
    'crop_image',

    # Bbox utilities
    'parse_json_format',
    'adjust_bboxes_after_padding',
    'clamp_bbox',
    'bbox_area',
    'bbox_dimensions',
    'is_valid_bbox',
    'calculate_iou',
    'convert_coords_format',
    'format_bbox_for_json',
    'get_bbox_center',

    # I/O utilities
    'load_json',
    'save_json',
    'get_prompt',
    'create_output_directory',
    'get_timestamp',
    'create_results_filename',
    'list_images',
    'extract_test_name_from_path',
    'ensure_directory',
    'file_exists',
    'read_text_file',
    'write_text_file',
    'get_relative_path',
    'copy_file',

    # Dataset utilities
    'collect_texture_crops',
    'create_dataset_from_multiple_runs',

    # Parallel processing (conditional)
    'ParallelProcessor',
    'parallel_save_crops',
    'parallel_load_images',
    'parallel_copy_files',
    'batch_process',
    'chunked_iterator',

    # Prefetching (conditional)
    'ImagePrefetcher',
    'BatchPrefetcher',
    'AsyncDataLoader',

    # Streaming (conditional)
    'StreamingJsonWriter',
    'StreamingJsonReader',
    'CheckpointWriter',
    'AsyncStreamingWriter',
    'convert_jsonl_to_json',

    # Model cache (conditional)
    'ModelCache',
    'ModelContext',
    'clear_gpu_memory',
    'get_optimal_batch_size',

    # Availability flags
    'PARALLEL_AVAILABLE',
    'PREFETCH_AVAILABLE',
    'STREAMING_AVAILABLE',
    'MODEL_CACHE_AVAILABLE',
]
