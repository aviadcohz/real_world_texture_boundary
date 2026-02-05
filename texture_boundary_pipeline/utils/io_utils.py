import json
from pathlib import Path
from typing import Union, Dict, List, Any
from datetime import datetime
import shutil
import sys
from pathlib import Path

from config.prompts import get_prompt as _get_prompt


def load_json(file_path: Union[str, Path]) -> Dict:
    """
    Load JSON file.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Parsed JSON data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(
    data: Union[Dict, List],
    file_path: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False
) -> None:

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def get_prompt(prompt_name: str) -> str:
    return _get_prompt(prompt_name)
    # Add config to path if needed
    # config_path = Path(__file__).parent.parent / 'config'
    # if str(config_path) not in sys.path:
    #     sys.path.insert(0, str(config_path))
    #     return _get_prompt(prompt_name)

def create_output_directory(
    base_dir: Union[str, Path],
    test_name: str,
    subdirs: List[str] = None
) -> Path:

    output_dir = Path(base_dir) / test_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if subdirs:
        for subdir in subdirs:
            (output_dir / subdir).mkdir(exist_ok=True)
    
    return output_dir


def get_timestamp(format: str = "%Y%m%d_%H%M%S") -> str:

    return datetime.now().strftime(format)


def create_results_filename(
    prefix: str,
    test_name: str,
    num_images: int = None,
    extension: str = 'json'
) -> str:

    timestamp = get_timestamp()
    
    if num_images is not None:
        return f"{prefix}_{test_name}_{num_images}_imgs_{timestamp}.{extension}"
    else:
        return f"{prefix}_{test_name}_{timestamp}.{extension}"


def list_images(
    directory: Union[str, Path],
    extensions: List[str] = None
) -> List[Path]:

    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    images = []
    for ext in extensions:
        images.extend(directory.glob(f"*{ext}"))
        images.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(images)


def extract_test_name_from_path(json_path: Union[str, Path], parent_dir: str) -> str:

    parts = Path(json_path).parts
    
    try:
        index = parts.index(parent_dir)
        if index + 1 < len(parts):
            return parts[index + 1]
    except ValueError:
        pass
    
    # Fallback: use parent directory name
    return Path(json_path).parent.name


def ensure_directory(path: Union[str, Path]) -> Path:

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_exists(path: Union[str, Path]) -> bool:

    return Path(path).exists()


def read_text_file(path: Union[str, Path]) -> str:

    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def write_text_file(content: str, path: Union[str, Path]) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def get_relative_path(path: Union[str, Path], base: Union[str, Path]) -> Path:

    return Path(path).relative_to(base)


def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> None:

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(src, dst)