
# Always import base class (no dependencies)
from .base_vlm import BaseVLM

# Try to import implementations (may fail if dependencies missing)
try:
    from .qwen_vlm import QwenVLM, create_qwen_model
    _QWEN_AVAILABLE = True
except ImportError as e:
    _QWEN_AVAILABLE = False
    QwenVLM = None
    create_qwen_model = None

try:
    from .llava_vlm import LLaVAVLM, LLaVANextVLM
    _LLAVA_AVAILABLE = True
except ImportError:
    _LLAVA_AVAILABLE = False
    LLaVAVLM = None
    LLaVANextVLM = None

# Factory always available
from .model_factory import (
    create_model,
    list_available_models,
    register_model,
    get_model_info,
    check_dependencies,
)

from .sa2va_vlm import Sa2VAModel

# Remote client (no GPU dependencies)
try:
    from .qwen_client import QwenVLMClient, create_remote_model
    _CLIENT_AVAILABLE = True
except ImportError:
    _CLIENT_AVAILABLE = False
    QwenVLMClient = None
    create_remote_model = None

__all__ = [
    # Base class
    'BaseVLM',
    
    # Implementations
    'QwenVLM',
    'LLaVAVLM',
    'LLaVANextVLM',
    'Sa2VAModel',
    
    # Factory functions
    'create_model',
    'create_qwen_model',
    'create_remote_model',
    'list_available_models',
    'register_model',
    'get_model_info',
    "check_dependencies",

    # Remote client
    'QwenVLMClient',
]