"""
Agents module for Texture Curator.

Contains all agent implementations:
- BaseAgent: Abstract base class
- PlannerAgent: Orchestrates workflow
- ProfilerAgent: Builds RWTD profile
- AnalystAgent: Scores candidates
- MaskFilterAgent: VLM-based mask quality filter
- OptimizerAgent: Selects diverse final set
"""


from agents.base import BaseAgent, AgentAction, ToolResult, TOOL_DESCRIPTIONS, get_tool_descriptions
from agents.planner import PlannerAgent
from agents.profiler import ProfilerAgent
from agents.analyst import AnalystAgent
from agents.mask_filter import MaskFilterAgent
from agents.optimizer import OptimizerAgent

__all__ = [
    "BaseAgent",
    "AgentAction",
    "ToolResult",
    "TOOL_DESCRIPTIONS",
    "get_tool_descriptions",
    "PlannerAgent",
    "ProfilerAgent",
    "AnalystAgent",
    "MaskFilterAgent",
    "OptimizerAgent",
]
