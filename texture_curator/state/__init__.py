"""
State management module for Texture Curator.
"""

# Handle both package and direct imports
try:
    from state.models import (
        Distribution,
        RWTDProfile,
        CandidateFeatures,
        ScoreBreakdown,
        CriticVerdict,
        CandidateRecord,
        CriticReport,
        SelectionReport,
        AgentMessage,
        RerouteRecord,
    )
    from state.graph_state import (
        GraphState,
        create_initial_state,
    )
except ImportError:
    from .models import (
        Distribution,
        RWTDProfile,
        CandidateFeatures,
        ScoreBreakdown,
        CriticVerdict,
        CandidateRecord,
        CriticReport,
        SelectionReport,
        AgentMessage,
        RerouteRecord,
    )
    from .graph_state import (
        GraphState,
        create_initial_state,
    )

__all__ = [
    # Models
    "Distribution",
    "RWTDProfile",
    "CandidateFeatures",
    "ScoreBreakdown",
    "CriticVerdict",
    "CandidateRecord",
    "CriticReport",
    "SelectionReport",
    "AgentMessage",
    "RerouteRecord",
    # State
    "GraphState",
    "create_initial_state",
]