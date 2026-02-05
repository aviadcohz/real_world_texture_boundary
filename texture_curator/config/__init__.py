"""
Configuration module for Texture Curator.
"""

# Handle both package and direct imports
try:
    from config.settings import (
        Config,
        ThresholdConfig,
        LLMConfig,
        VisionConfig,
        Phase,
        MaskStatus,
        create_config,
    )
except ImportError:
    from .settings import (
        Config,
        ThresholdConfig,
        LLMConfig,
        VisionConfig,
        Phase,
        MaskStatus,
        create_config,
    )

__all__ = [
    "Config",
    "ThresholdConfig",
    "LLMConfig",
    "VisionConfig",
    "Phase",
    "MaskStatus",
    "create_config",
]