"""
Data Models for Texture Curator.

These dataclasses define the structure of data flowing through the system.
They are used by all agents and stored in checkpoints.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import numpy as np
import json

from config.settings import MaskStatus


# ============================================
# Distribution Model (for statistical profiles)
# ============================================

@dataclass
class Distribution:
    """
    Represents a statistical distribution of a metric.
    
    Used to capture the "shape" of RWTD characteristics.
    """
    mean: float
    std: float
    min_val: float
    max_val: float
    percentile_25: float = 0.0
    percentile_75: float = 0.0
    
    @classmethod
    def from_values(cls, values: np.ndarray) -> "Distribution":
        """Create Distribution from array of values."""
        return cls(
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            min_val=float(np.min(values)),
            max_val=float(np.max(values)),
            percentile_25=float(np.percentile(values, 25)),
            percentile_75=float(np.percentile(values, 75)),
        )
    
    def score_value(self, value: float) -> float:
        """
        Score how well a value fits this distribution.
        
        Returns a value between 0 and 1, where 1 means perfect fit.
        Uses a simple approach: how many std devs from mean.
        """
        if self.std == 0:
            return 1.0 if value == self.mean else 0.0
        
        z_score = abs(value - self.mean) / self.std
        # Convert to 0-1 score (e.g., 0 std = 1.0, 2 std = 0.5, 4 std = 0.0)
        score = max(0.0, 1.0 - (z_score / 4.0))
        return score
    
    def to_dict(self) -> dict:
        return {
            "mean": float(self.mean),
            "std": float(self.std),
            "min": float(self.min_val),
            "max": float(self.max_val),
            "p25": float(self.percentile_25),
            "p75": float(self.percentile_75),
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Distribution":
        return cls(
            mean=d["mean"],
            std=d["std"],
            min_val=d["min"],
            max_val=d["max"],
            percentile_25=d.get("p25", 0.0),
            percentile_75=d.get("p75", 0.0),
        )


# ============================================
# RWTD Profile (Gold Standard Signature)
# ============================================

@dataclass
class RWTDProfile:
    """
    Statistical profile of the RWTD reference dataset.
    
    This is the "Gold Standard" signature that candidates are compared against.
    Built by the Profiler agent in Phase 1.
    """
    
    # Number of samples in RWTD
    num_samples: int = 0
    
    # Semantic centroid (mean of all DINOv2 embeddings)
    # Shape: (embedding_dim,) e.g., (768,) for vitb14
    centroid_embedding: Optional[np.ndarray] = None
    
    # Covariance matrix for Mahalanobis distance (optional, for advanced scoring)
    # Shape: (embedding_dim, embedding_dim)
    embedding_covariance: Optional[np.ndarray] = None
    
    # Statistical distributions of various metrics
    entropy_distribution: Optional[Distribution] = None
    
    glcm_distributions: Dict[str, Distribution] = field(default_factory=dict)
    # Keys: "contrast", "homogeneity", "energy", "correlation"
    
    boundary_sharpness_distribution: Optional[Distribution] = None
    edge_density_distribution: Optional[Distribution] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    
    def is_complete(self) -> bool:
        """Check if profile has all required components."""
        return (
            self.centroid_embedding is not None and
            self.entropy_distribution is not None and
            self.boundary_sharpness_distribution is not None
        )
    
    def to_dict(self) -> dict:
        """Serialize to dictionary (embeddings saved separately)."""
        return {
            "num_samples": self.num_samples,
            "has_centroid": self.centroid_embedding is not None,
            "entropy": self.entropy_distribution.to_dict() if self.entropy_distribution else None,
            "glcm": {k: v.to_dict() for k, v in self.glcm_distributions.items()},
            "boundary_sharpness": self.boundary_sharpness_distribution.to_dict() if self.boundary_sharpness_distribution else None,
            "edge_density": self.edge_density_distribution.to_dict() if self.edge_density_distribution else None,
            "created_at": self.created_at.isoformat(),
        }


# ============================================
# Candidate Features & Scores
# ============================================

@dataclass
class CandidateFeatures:
    """
    Extracted features for a single candidate image.
    
    Computed by the Analyst agent.
    """
    
    # DINOv2 embedding
    dino_embedding: Optional[np.ndarray] = None  # Shape: (768,)
    
    # Texture statistics
    entropy: float = 0.0
    glcm_contrast: float = 0.0
    glcm_homogeneity: float = 0.0
    glcm_energy: float = 0.0
    glcm_correlation: float = 0.0
    
    # Boundary statistics
    boundary_sharpness: float = 0.0  # Variance of Laplacian
    edge_density: float = 0.0        # % of boundary on image edge
    gradient_magnitude_mean: float = 0.0
    gradient_magnitude_std: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "has_embedding": self.dino_embedding is not None,
            "entropy": float(self.entropy),
            "glcm_contrast": float(self.glcm_contrast),
            "glcm_homogeneity": float(self.glcm_homogeneity),
            "glcm_energy": float(self.glcm_energy),
            "glcm_correlation": float(self.glcm_correlation),
            "boundary_sharpness": float(self.boundary_sharpness),
            "edge_density": float(self.edge_density),
            "gradient_magnitude_mean": float(self.gradient_magnitude_mean),
            "gradient_magnitude_std": float(self.gradient_magnitude_std),
        }


@dataclass
class ScoreBreakdown:
    """
    Detailed scoring breakdown for a candidate.
    
    Shows how the total score is composed.
    """
    
    # Component scores (0-1 range)
    semantic_score: float = 0.0   # Cosine similarity to centroid
    texture_score: float = 0.0    # How well texture stats fit RWTD
    boundary_score: float = 0.0   # Boundary quality score
    
    # Final weighted score
    total_score: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "semantic": float(self.semantic_score),
            "texture": float(self.texture_score),
            "boundary": float(self.boundary_score),
            "total": float(self.total_score),
        }


# ============================================
# Critic Verdict
# ============================================

@dataclass
class CriticVerdict:
    """
    Result of VLM audit by the Critic agent.
    
    Determines if a transition is material-based (good) or object-based (bad).
    """
    
    is_material_transition: bool = False
    confidence: float = 0.0       # 0-1
    reasoning: str = ""           # VLM's explanation
    
    def to_dict(self) -> dict:
        return {
            "is_material_transition": self.is_material_transition,
            "confidence": float(self.confidence),
            "reasoning": self.reasoning,
        }


# ============================================
# Mask Filter Verdict
# ============================================

@dataclass
class MaskFilterVerdict:
    """
    Result of VLM-based mask quality assessment.

    Produced by the MaskFilter agent before scoring.
    """

    passed: bool = False
    reason: str = ""          # NO_BOUNDARY, NOT_GROUND_TRUTH, INCOMPLETE, TOO_COMPLEX, NO_TEXTURE_TRANSITION, PASS
    confidence: float = 0.0   # 0-1
    explanation: str = ""     # VLM's brief reasoning

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "confidence": float(self.confidence),
            "explanation": self.explanation,
        }


# ============================================
# Candidate Record (Full record for one candidate)
# ============================================

@dataclass
class CandidateRecord:
    """
    Complete record for a single candidate image.
    
    Tracks all information from discovery to selection.
    """
    
    # Identity
    id: str
    image_path: Path
    mask_path: Path
    
    # Extracted features (populated by Analyst)
    features: Optional[CandidateFeatures] = None
    
    # Scores (populated by Analyst)
    scores: Optional[ScoreBreakdown] = None
    
    # Mask status
    mask_status: MaskStatus = MaskStatus.PENDING

    # Mask filter verdict (populated by MaskFilter agent)
    mask_filter_verdict: Optional[MaskFilterVerdict] = None

    # Critic verdict (populated by Critic)
    critic_verdict: Optional[CriticVerdict] = None
    
    # Selection flag
    is_selected: bool = False
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "image_path": str(self.image_path),
            "mask_path": str(self.mask_path),
            "features": self.features.to_dict() if self.features else None,
            "scores": self.scores.to_dict() if self.scores else None,
            "mask_status": self.mask_status.value,
            "mask_filter_verdict": self.mask_filter_verdict.to_dict() if self.mask_filter_verdict else None,
            "critic_verdict": self.critic_verdict.to_dict() if self.critic_verdict else None,
            "is_selected": self.is_selected,
        }


# ============================================
# Reports
# ============================================

@dataclass
class CriticReport:
    """
    Summary report from the Critic agent.
    
    Used by Planner to decide whether to proceed or reroute.
    """
    
    samples_reviewed: int = 0
    material_transitions: int = 0
    object_boundaries: int = 0
    
    @property
    def quality_score(self) -> float:
        """Ratio of good transitions (0-1)."""
        if self.samples_reviewed == 0:
            return 0.0
        return self.material_transitions / self.samples_reviewed
    
    issues_found: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "samples_reviewed": self.samples_reviewed,
            "material_transitions": self.material_transitions,
            "object_boundaries": self.object_boundaries,
            "quality_score": float(self.quality_score),
            "issues_found": self.issues_found,
            "recommendations": self.recommendations,
        }


@dataclass
class SelectionReport:
    """
    Summary report from the Optimizer agent.
    """
    
    total_candidates: int = 0
    passed_quality_gate: int = 0
    final_selected: int = 0
    diversity_score: float = 0.0
    mean_quality_score: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "total_candidates": self.total_candidates,
            "passed_quality_gate": self.passed_quality_gate,
            "final_selected": self.final_selected,
            "diversity_score": float(self.diversity_score),
            "mean_quality_score": float(self.mean_quality_score),
        }


# ============================================
# Agent Communication
# ============================================

@dataclass
class AgentMessage:
    """
    A message in the agent conversation history.
    
    Used for debugging and understanding agent decisions.
    """
    
    agent: str           # "planner", "profiler", "analyst", "critic", "optimizer"
    role: str            # "system", "user", "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RerouteRecord:
    """
    Record of a reroute decision.
    
    Tracks why the system rerouted and what adjustments were made.
    """
    
    iteration: int
    reason: str
    threshold_adjustments: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "reason": self.reason,
            "threshold_adjustments": self.threshold_adjustments,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================
# Quick test
# ============================================

if __name__ == "__main__":
    import numpy as np
    
    # Test Distribution
    values = np.random.normal(100, 20, 256)  # Simulate RWTD entropy values
    dist = Distribution.from_values(values)
    print("Distribution test:")
    print(f"  Mean: {dist.mean:.2f}, Std: {dist.std:.2f}")
    print(f"  Score for mean value: {dist.score_value(dist.mean):.2f}")
    print(f"  Score for +2 std: {dist.score_value(dist.mean + 2*dist.std):.2f}")
    print()
    
    # Test CandidateRecord
    candidate = CandidateRecord(
        id="img_001",
        image_path=Path("/data/images/img_001.jpg"),
        mask_path=Path("/data/masks/img_001.png"),
    )
    print("CandidateRecord test:")
    print(f"  ID: {candidate.id}")
    print(f"  Status: {candidate.mask_status}")
    print(f"  Dict: {candidate.to_dict()}")