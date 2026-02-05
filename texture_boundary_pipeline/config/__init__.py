"""
Configuration Package

Contains prompt templates and scalability configuration.
"""

from .prompts import get_prompt, PROMPTS

# Import scale configuration
try:
    from .scale_config import (
        ScaleConfig,
        ProcessingConfig,
        MemoryConfig,
        DistributedConfig,
        OutputConfig,
        get_config,
        set_config,
        get_preset,
        get_default_config,
        configure_from_preset,
        configure_from_file,
        PRESETS
    )
    SCALE_CONFIG_AVAILABLE = True
except ImportError:
    SCALE_CONFIG_AVAILABLE = False

__all__ = [
    # Prompts
    'get_prompt',
    'PROMPTS',

    # Scale config (conditional)
    'ScaleConfig',
    'ProcessingConfig',
    'MemoryConfig',
    'DistributedConfig',
    'OutputConfig',
    'get_config',
    'set_config',
    'get_preset',
    'get_default_config',
    'configure_from_preset',
    'configure_from_file',
    'PRESETS',

    # Availability flag
    'SCALE_CONFIG_AVAILABLE',
]
