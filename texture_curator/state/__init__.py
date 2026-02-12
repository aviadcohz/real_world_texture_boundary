"""
State management module for Texture Curator.
"""

try:
    from state.models import (
        RWTDProfile,
        CandidateFeatures,
        ScoreBreakdown,
        CandidateRecord,
        MaskFilterVerdict,
        SelectionReport,
        AgentMessage,
    )
    from state.graph_state import (
        GraphState,
        create_initial_state,
    )
except ImportError:
    from .models import (
        RWTDProfile,
        CandidateFeatures,
        ScoreBreakdown,
        CandidateRecord,
        MaskFilterVerdict,
        SelectionReport,
        AgentMessage,
    )
    from .graph_state import (
        GraphState,
        create_initial_state,
    )

__all__ = [
    "RWTDProfile",
    "CandidateFeatures",
    "ScoreBreakdown",
    "CandidateRecord",
    "MaskFilterVerdict",
    "SelectionReport",
    "AgentMessage",
    "GraphState",
    "create_initial_state",
]
