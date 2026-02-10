"""
Graph State for Texture Curator.

This is the CENTRAL STATE that flows through the entire agent graph.
All agents read from and write to this state.

Think of it as a shared whiteboard that all agents can see and modify.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json

from config.settings import Config, Phase
from state.models import (
    RWTDProfile,
    CandidateRecord,
    CriticReport,
    SelectionReport,
    AgentMessage,
    RerouteRecord,
)


@dataclass
class GraphState:
    """
    Shared state across all agents in the graph.
    
    This is the single source of truth for the entire curation process.
    
    DESIGN PRINCIPLES:
    1. Immutable config (set once at start)
    2. Mutable artifacts (built up by agents)
    3. Full history for debugging
    4. Serializable for checkpoints
    """
    
    # ═══════════════════════════════════════════════════════════════
    # CONFIGURATION (Set once at initialization)
    # ═══════════════════════════════════════════════════════════════
    
    config: Config = field(default_factory=Config)
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 1 OUTPUT: RWTD Profile
    # ═══════════════════════════════════════════════════════════════
    
    rwtd_profile: Optional[RWTDProfile] = None
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 2 OUTPUT: Scored Candidates
    # ═══════════════════════════════════════════════════════════════
    
    # Key = candidate ID, Value = full record
    candidates: Dict[str, CandidateRecord] = field(default_factory=dict)
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 3 OUTPUT: Critic Results
    # ═══════════════════════════════════════════════════════════════
    
    critic_report: Optional[CriticReport] = None
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 4 OUTPUT: Final Selection
    # ═══════════════════════════════════════════════════════════════
    
    selected_ids: List[str] = field(default_factory=list)
    selection_report: Optional[SelectionReport] = None
    
    # ═══════════════════════════════════════════════════════════════
    # CONTROL FLOW
    # ═══════════════════════════════════════════════════════════════
    
    # Current phase in the workflow
    current_phase: Phase = Phase.INIT
    
    # Iteration counter (increments on each reroute)
    iteration: int = 0
    
    # History of reroutes (for debugging)
    reroute_history: List[RerouteRecord] = field(default_factory=list)
    
    # Mask filtering status
    mask_filtering_done: bool = False

    # Error tracking
    last_error: Optional[str] = None
    
    # ═══════════════════════════════════════════════════════════════
    # OBSERVABILITY (For debugging and learning)
    # ═══════════════════════════════════════════════════════════════
    
    # Full conversation history across all agents
    message_history: List[AgentMessage] = field(default_factory=list)
    
    # Timestamps
    started_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # ═══════════════════════════════════════════════════════════════
    # CONVENIENCE PROPERTIES
    # ═══════════════════════════════════════════════════════════════
    
    @property
    def profile_exists(self) -> bool:
        """Check if RWTD profile has been built."""
        return self.rwtd_profile is not None and self.rwtd_profile.is_complete()
    
    @property
    def num_candidates(self) -> int:
        """Total number of candidates discovered."""
        return len(self.candidates)
    
    @property
    def num_scored(self) -> int:
        """Number of candidates that have been scored."""
        return sum(1 for c in self.candidates.values() if c.scores is not None)
    
    @property
    def num_validated(self) -> int:
        """Number of candidates reviewed by Critic."""
        return sum(1 for c in self.candidates.values() if c.critic_verdict is not None)
    
    @property
    def num_valid(self) -> int:
        """Number of candidates that passed Critic review."""
        return sum(
            1 for c in self.candidates.values() 
            if c.critic_verdict is not None and c.critic_verdict.is_material_transition
        )
    
    @property
    def num_mask_filtered(self) -> int:
        """Number of candidates that have been assessed by mask filter."""
        from config.settings import MaskStatus
        return sum(1 for c in self.candidates.values() if c.mask_status != MaskStatus.PENDING)

    @property
    def num_mask_passed(self) -> int:
        """Number of candidates that passed mask filter."""
        from config.settings import MaskStatus
        return sum(1 for c in self.candidates.values() if c.mask_status == MaskStatus.VALID)

    @property
    def num_selected(self) -> int:
        """Number of candidates selected for final dataset."""
        return len(self.selected_ids)
    
    @property
    def is_done(self) -> bool:
        """Check if curation is complete."""
        return self.current_phase == Phase.DONE
    
    @property
    def can_reroute(self) -> bool:
        """Check if we can still reroute (haven't hit max iterations)."""
        return self.iteration < self.config.max_iterations
    
    # ═══════════════════════════════════════════════════════════════
    # METHODS
    # ═══════════════════════════════════════════════════════════════
    
    def get_top_candidates(self, n: int = 10, require_scored: bool = True) -> List[CandidateRecord]:
        """
        Get top N candidates by total score.
        
        Args:
            n: Number of candidates to return
            require_scored: Only include candidates with scores
        
        Returns:
            List of CandidateRecord sorted by score descending
        """
        candidates_list = list(self.candidates.values())
        
        if require_scored:
            candidates_list = [c for c in candidates_list if c.scores is not None]
        
        # Sort by total score descending
        candidates_list.sort(
            key=lambda c: c.scores.total_score if c.scores else 0,
            reverse=True
        )
        
        return candidates_list[:n]
    
    def get_valid_candidates(self) -> List[CandidateRecord]:
        """Get all candidates that passed Critic review."""
        return [
            c for c in self.candidates.values()
            if c.critic_verdict is not None and c.critic_verdict.is_material_transition
        ]
    
    def add_message(self, agent: str, role: str, content: str):
        """Add a message to the conversation history."""
        self.message_history.append(
            AgentMessage(agent=agent, role=role, content=content)
        )
        self.last_updated = datetime.now()
    
    def record_reroute(self, reason: str, adjustments: Dict[str, float]):
        """Record a reroute decision."""
        self.reroute_history.append(
            RerouteRecord(
                iteration=self.iteration,
                reason=reason,
                threshold_adjustments=adjustments,
            )
        )
        self.iteration += 1
        self.last_updated = datetime.now()
    
    def get_status_summary(self) -> str:
        """
        Get a human-readable status summary.
        
        This is what gets sent to the Planner agent.
        """
        lines = [
            f"## Current State (Iteration {self.iteration})",
            f"- Phase: {self.current_phase.value}",
            f"- RWTD Profile: {'✓ Built' if self.profile_exists else '✗ Not built'}",
            f"- Candidates Discovered: {self.num_candidates}",
            f"- Mask Filter: {'done' if self.mask_filtering_done else 'pending'} ({self.num_mask_passed} passed / {self.num_mask_filtered} assessed)",
            f"- Candidates Scored: {self.num_scored}/{self.num_candidates}",
            f"- Candidates Validated: {self.num_validated}",
            f"- Valid (Material Transitions): {self.num_valid}",
            f"- Selected: {self.num_selected}/{self.config.target_n}",
        ]
        
        if self.critic_report:
            lines.append(f"- Quality Score: {self.critic_report.quality_score:.2%}")
        
        if self.reroute_history:
            lines.append(f"- Reroutes: {len(self.reroute_history)}")
        
        if self.last_error:
            lines.append(f"- Last Error: {self.last_error}")
        
        return "\n".join(lines)
    
    def get_recent_actions(self, n: int = 5) -> str:
        """Get summary of recent agent actions."""
        if not self.message_history:
            return "No actions yet."
        
        recent = self.message_history[-n*2:]  # Get last n exchanges
        
        lines = ["## Recent Actions"]
        for msg in recent:
            if msg.role == "assistant":
                # Truncate long content
                content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                lines.append(f"[{msg.agent}] {content}")
        
        return "\n".join(lines)
    
    # ═══════════════════════════════════════════════════════════════
    # SERIALIZATION (For checkpoints)
    # ═══════════════════════════════════════════════════════════════
    
    def to_dict(self) -> dict:
        """
        Serialize state to dictionary.
        
        Note: Large numpy arrays (embeddings) are not included.
        Those are saved separately in checkpoint files.
        """
        return {
            "config": self.config.to_dict(),
            "current_phase": self.current_phase.value,
            "iteration": self.iteration,
            "profile_exists": self.profile_exists,
            "rwtd_profile": self.rwtd_profile.to_dict() if self.rwtd_profile else None,
            "num_candidates": self.num_candidates,
            "num_scored": self.num_scored,
            "num_validated": self.num_validated,
            "num_valid": self.num_valid,
            "num_selected": self.num_selected,
            "selected_ids": self.selected_ids,
            "critic_report": self.critic_report.to_dict() if self.critic_report else None,
            "selection_report": self.selection_report.to_dict() if self.selection_report else None,
            "reroute_history": [r.to_dict() for r in self.reroute_history],
            "message_count": len(self.message_history),
            "started_at": self.started_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "last_error": self.last_error,
        }
    
    def save_checkpoint(self, checkpoint_dir: Path, name: str):
        """
        Save state to checkpoint directory.
        
        Creates:
        - {name}_state.json: Serialized state
        - {name}_candidates.json: Candidate records
        - {name}_embeddings.npz: Numpy arrays (if any)
        """
        import numpy as np
        
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main state
        state_file = checkpoint_dir / f"{name}_state.json"
        with open(state_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        # Save candidates
        candidates_file = checkpoint_dir / f"{name}_candidates.json"
        candidates_data = {k: v.to_dict() for k, v in self.candidates.items()}
        with open(candidates_file, "w") as f:
            json.dump(candidates_data, f, indent=2)
        
        # Save embeddings (if profile exists)
        if self.rwtd_profile and self.rwtd_profile.centroid_embedding is not None:
            embeddings_file = checkpoint_dir / f"{name}_embeddings.npz"
            np.savez(
                embeddings_file,
                centroid=self.rwtd_profile.centroid_embedding,
            )
        
        return state_file


# ═══════════════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════════════

def create_initial_state(
    rwtd_path: str,
    source_pool_path: str,
    target_n: int = 10,
    **config_kwargs
) -> GraphState:
    """
    Create initial GraphState with custom configuration.
    
    Args:
        rwtd_path: Path to RWTD reference dataset
        source_pool_path: Path to source pool
        target_n: Number of samples to select
        **config_kwargs: Additional config overrides
    
    Returns:
        Initialized GraphState
    """
    config = Config(
        rwtd_path=Path(rwtd_path),
        source_pool_path=Path(source_pool_path),
        target_n=target_n,
        **config_kwargs
    )
    
    return GraphState(config=config)


# ═══════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Create a test state
    state = GraphState()
    
    print("=" * 60)
    print("GraphState Test")
    print("=" * 60)
    print()
    print(state.get_status_summary())
    print()
    print("Serialized state:")
    print(json.dumps(state.to_dict(), indent=2))