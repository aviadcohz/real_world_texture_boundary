from typing import Dict, Type

from .base_vlm import BaseVLM

try:
    from .qwen_vlm import QwenVLM, create_qwen_model
    _QWEN_AVAILABLE = True
except ImportError:
    QwenVLM = None
    create_qwen_model = None
    _QWEN_AVAILABLE = False

MODEL_REGISTRY: Dict[str, Type[BaseVLM]] = {}

if _QWEN_AVAILABLE and QwenVLM is not None:
    MODEL_REGISTRY['qwen'] = QwenVLM
    MODEL_REGISTRY['qwen-8b'] = QwenVLM
    MODEL_REGISTRY['qwen-2b'] = QwenVLM


def create_model(model_name: str, device: str = "cuda", **kwargs) -> BaseVLM:
    model_name_lower = model_name.lower()
    if model_name_lower not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys()) if MODEL_REGISTRY else 'None'
        raise ValueError(f"Unknown model: '{model_name}'. Available: {available}")

    if model_name_lower in ['qwen-8b', 'qwen-2b', 'qwen']:
        size = "2B" if model_name_lower == 'qwen-2b' else "8B"
        return create_qwen_model(model_size=size, device=device)

    model_class = MODEL_REGISTRY[model_name_lower]
    return model_class(device=device, **kwargs)


def list_available_models() -> list:
    return list(MODEL_REGISTRY.keys())
