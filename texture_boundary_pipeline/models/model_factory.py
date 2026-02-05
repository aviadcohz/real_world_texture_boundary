
from typing import Dict, Type

from .base_vlm import BaseVLM

# Try to import implementations (may fail if dependencies missing)
try:
    from .qwen_vlm import QwenVLM, create_qwen_model
    _QWEN_AVAILABLE = True
except ImportError:
    QwenVLM = None
    create_qwen_model = None
    _QWEN_AVAILABLE = False

try:
    from .llava_vlm import LLaVAVLM, LLaVANextVLM
    _LLAVA_AVAILABLE = True
except ImportError:
    LLaVAVLM = None
    LLaVANextVLM = None
    _LLAVA_AVAILABLE = False


# Registry of available models
MODEL_REGISTRY: Dict[str, Type[BaseVLM]] = {}

# Add models to registry if available
if _QWEN_AVAILABLE and QwenVLM is not None:
    MODEL_REGISTRY['qwen'] = QwenVLM
    MODEL_REGISTRY['qwen-8b'] = QwenVLM
    MODEL_REGISTRY['qwen-2b'] = QwenVLM

if _LLAVA_AVAILABLE and LLaVAVLM is not None:
    MODEL_REGISTRY['llava'] = LLaVAVLM
    MODEL_REGISTRY['llava-next'] = LLaVANextVLM


def create_model(
    model_name: str,
    device: str = "cuda",
    **kwargs
) -> BaseVLM:
    """
    Create a VLM instance by name.
    
    Args:
        model_name: Name of the model to create
            Options: 'qwen', 'qwen-8b', 'qwen-2b', 'llava', 'llava-next'
        device: Device to run on (cuda/cpu)
        **kwargs: Additional model-specific arguments
    
    Returns:
        Initialized VLM instance
    
    Examples:
        >>> model = create_model('qwen', device='cuda')
        >>> model = create_model('qwen-8b')
        >>> model = create_model('llava')
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys()) if MODEL_REGISTRY else 'None (install dependencies)'
        raise ValueError(
            f"Unknown or unavailable model: '{model_name}'. "
            f"Available models: {available}"
        )
    
    # Special handling for Qwen variants
    if model_name_lower in ['qwen-8b', 'qwen-2b', 'qwen']:
        if not _QWEN_AVAILABLE or create_qwen_model is None:
            raise ImportError(
                "Qwen models require 'torch' and 'transformers'. "
                "Install with: pip install torch transformers qwen-vl-utils"
            )
        
        if model_name_lower == 'qwen-8b':
            return create_qwen_model(model_size="8B", device=device)
        elif model_name_lower == 'qwen-2b':
            return create_qwen_model(model_size="2B", device=device)
        elif model_name_lower == 'qwen':
            return create_qwen_model(model_size="8B", device=device)
    
    # Generic creation for other models
    model_class = MODEL_REGISTRY[model_name_lower]
    return model_class(device=device, **kwargs)


def list_available_models() -> list:
    """
    List all available model names.
    
    Returns:
        List of model names
    """
    return list(MODEL_REGISTRY.keys())


def register_model(name: str, model_class: Type[BaseVLM]) -> None:
    """
    Register a new model class.
    
    Useful for adding custom models without modifying this file.
    
    Args:
        name: Name to register the model under
        model_class: Model class (must inherit from BaseVLM)
    
    Example:
        >>> class MyCustomVLM(BaseVLM):
        ...     # Implementation
        >>> register_model('my-model', MyCustomVLM)
        >>> model = create_model('my-model')
    """
    if not issubclass(model_class, BaseVLM):
        raise TypeError(f"{model_class} must inherit from BaseVLM")
    
    MODEL_REGISTRY[name.lower()] = model_class
    print(f"✅ Registered model: '{name}'")


def get_model_info(model_name: str) -> dict:
    """
    Get information about a model without loading it.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Dict with model information
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower not in MODEL_REGISTRY:
        return {'error': f'Unknown model: {model_name}'}
    
    model_class = MODEL_REGISTRY[model_name_lower]
    
    return {
        'name': model_name,
        'class': model_class.__name__,
        'module': model_class.__module__,
        'implemented': model_class != LLaVAVLM and model_class != LLaVANextVLM if LLaVAVLM and LLaVANextVLM else True
    }


def check_dependencies() -> dict:
    """
    Check which model dependencies are available.
    
    Returns:
        Dict with availability status for each model type
    """
    return {
        'qwen': _QWEN_AVAILABLE,
        'llava': _LLAVA_AVAILABLE,
        'available_models': list(MODEL_REGISTRY.keys())
    }


# Convenience: Show available models on import
if __name__ == "__main__":
    print("\n" + "="*70)
    print("AVAILABLE VLM MODELS")
    print("="*70)
    
    for model_name in list_available_models():
        info = get_model_info(model_name)
        status = "✅" if info['implemented'] else "⚠️ (placeholder)"
        print(f"  {model_name:15} → {info['class']:20} {status}")
    
    print("="*70)
    print("\nUsage:")
    print("  from models.model_factory import create_model")
    print("  model = create_model('qwen')")
    print("="*70 + "\n")