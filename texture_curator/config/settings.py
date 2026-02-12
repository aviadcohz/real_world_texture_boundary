"""
Configuration for Texture Curator Multi-Agent System.

This file defines all the settings, paths, and thresholds used by the system.
We use dataclasses for type safety and easy serialization.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from enum import Enum


# ============================================
# Enums for Type Safety
# ============================================

class Phase(str, Enum):
    """Current phase in the agent workflow."""
    INIT = "init"
    PROFILING = "profiling"
    MASK_FILTERING = "mask_filtering"
    SCORING = "scoring"
    OPTIMIZING = "optimizing"
    DONE = "done"


class MaskStatus(str, Enum):
    """Status of a candidate's mask."""
    PENDING = "pending"       # Not yet validated
    VALID = "valid"           # Good quality, use as-is
    FIXABLE = "fixable"       # Can be cleaned
    FIXED = "fixed"           # Was cleaned successfully
    REJECTED = "rejected"     # Cannot be used


# ============================================
# Threshold Configuration
# ============================================

@dataclass
class ThresholdConfig:
    """
    Thresholds used for filtering and selection.
    """

    # Diversity weight in final selection (0 = only quality, 1 = only diversity)
    diversity_weight: float = 0.3

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "diversity_weight": self.diversity_weight,
        }


# ============================================
# LLM Configuration
# ============================================

@dataclass
class LLMConfig:
    """
    Configuration for the local LLM (via Ollama).
    """
    
    # Ollama server URL (default is local)
    base_url: str = "http://localhost:11434"
    
    # Model to use for agent reasoning
    model_name: str = "qwen2.5:7b"
    
    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 2048
    
    # Timeout for API calls (seconds)
    timeout: int = 120


# ============================================
# Vision Model Configuration
# ============================================

@dataclass
class VisionConfig:
    """
    Configuration for vision models (DINOv2, Qwen2.5-VL).
    """
    
    # DINOv2 model variant
    # Options: "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"
    dino_model: str = "dinov2_vitb14"  # Base model, good balance
    
    # DINOv2 embedding dimension (depends on model)
    # vits14=384, vitb14=768, vitl14=1024, vitg14=1536
    dino_embedding_dim: int = 768
    
    # Qwen2.5-VL model for VLM audit
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    # Batch size for processing (adjust based on GPU memory)
    batch_size: int = 16
    
    # Image size for processing
    image_size: int = 518  # DINOv2 default


# ============================================
# Mask Filter Configuration
# ============================================

@dataclass
class MaskFilterConfig:
    """Configuration for VLM-based mask quality filter."""

    # VLM model for mask assessment (via Ollama)
    vlm_model: str = "qwen2.5vl:7b"

    # Timeout per image (seconds)
    vlm_timeout: int = 60

    # Use fast math pre-filter to skip obvious failures before VLM
    enable_prefilter: bool = True

    # Height (px) for the side-by-side composite image sent to VLM
    composite_height: int = 384

    # Skip VLM filtering entirely (use only prefilter or no filtering)
    skip_vlm: bool = False


# ============================================
# Main Configuration
# ============================================

@dataclass
class Config:
    """
    Main configuration container for the entire system.
    """
    
    # ─────────────────────────────────────────
    # Dataset Paths
    # ─────────────────────────────────────────
    
    # Reference dataset (Gold Standard)
    rwtd_path: Path = Path("/home/aviad/RWTD")
    
    # Source pool to filter
    source_pool_path: Path = Path("/datasets/ade20k/real_texture_boundaries_20260201")
    
    # Output directory for results
    output_path: Path = Path("./outputs")
    
    # Checkpoint directory
    checkpoint_path: Path = Path("./checkpoints")
    
    # ─────────────────────────────────────────
    # Task Parameters
    # ─────────────────────────────────────────
    
    # Number of samples to select in final dataset
    target_n: int = 10

    # Max candidates to discover (0 = all). Random sample for quick testing.
    max_candidates: int = 0
    
    # ─────────────────────────────────────────
    # Sub-configurations
    # ─────────────────────────────────────────
    
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    mask_filter: MaskFilterConfig = field(default_factory=MaskFilterConfig)
    
    # ─────────────────────────────────────────
    # Runtime flags
    # ─────────────────────────────────────────
    
    # Enable detailed logging
    verbose: bool = True
    
    # Save checkpoints after each agent
    save_checkpoints: bool = True
    
    # Device for PyTorch
    device: str = "cuda"  # "cuda" or "cpu"

    # Only use crops that passed the entropy filter (from filter/passed/)
    filter_passed: bool = False
    
    def __post_init__(self):
        """Ensure paths exist and are Path objects."""
        self.rwtd_path = Path(self.rwtd_path)
        self.source_pool_path = Path(self.source_pool_path)
        self.output_path = Path(self.output_path)
        self.checkpoint_path = Path(self.checkpoint_path)
        
        # Create output directories if they don't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> bool:
        """
        Validate that all required paths exist.
        
        Returns:
            True if valid, raises ValueError otherwise.
        """
        if not self.rwtd_path.exists():
            raise ValueError(f"RWTD path does not exist: {self.rwtd_path}")
        
        if not self.source_pool_path.exists():
            raise ValueError(f"Source pool path does not exist: {self.source_pool_path}")
        
        return True
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "rwtd_path": str(self.rwtd_path),
            "source_pool_path": str(self.source_pool_path),
            "output_path": str(self.output_path),
            "checkpoint_path": str(self.checkpoint_path),
            "target_n": self.target_n,
            "thresholds": self.thresholds.to_dict(),
            "device": self.device,
        }


# ============================================
# Factory function for easy creation
# ============================================

def create_config(
    rwtd_path: Optional[str] = None,
    source_pool_path: Optional[str] = None,
    target_n: int = 10,
    **kwargs
) -> Config:
    """
    Create a Config instance with custom paths.
    
    Args:
        rwtd_path: Path to RWTD dataset
        source_pool_path: Path to source pool
        target_n: Number of samples to select
        **kwargs: Additional config overrides
    
    Returns:
        Configured Config instance
    """
    config = Config(
        target_n=target_n,
        **kwargs
    )
    
    if rwtd_path:
        config.rwtd_path = Path(rwtd_path)
    
    if source_pool_path:
        config.source_pool_path = Path(source_pool_path)
    
    return config


# ============================================
# Quick test
# ============================================

if __name__ == "__main__":
    # Test configuration creation
    config = Config()
    
    print("=" * 50)
    print("Texture Curator Configuration")
    print("=" * 50)
    print(f"RWTD Path: {config.rwtd_path}")
    print(f"Source Pool: {config.source_pool_path}")
    print(f"Target N: {config.target_n}")
    print(f"Device: {config.device}")
    print()
    print("Thresholds:")
    print(f"  Diversity Weight: {config.thresholds.diversity_weight}")
    print()
    print("LLM Config:")
    print(f"  Model: {config.llm.model_name}")
    print(f"  URL: {config.llm.base_url}")
    print()
    print("Vision Config:")
    print(f"  DINOv2: {config.vision.dino_model}")
    print(f"  VLM: {config.vision.vlm_model}")