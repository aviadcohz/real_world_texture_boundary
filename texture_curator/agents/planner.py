"""
Planner Agent for Texture Curator.

Orchestrates the linear workflow:
  1. profiler  → Build RWTD centroid
  2. analyst   → Discover candidates
  3. mask_filter → VLM filter
  4. analyst   → Score (cosine similarity)
  5. optimizer → Select diverse top-N
  6. done
"""

import logging
from typing import List, Tuple, Dict

try:
    from agents.base import BaseAgent, AgentAction, ToolResult
    from state.graph_state import GraphState
    from config.settings import Phase
    from llm.ollama_client import OllamaClient
except ImportError:
    from .base import BaseAgent, AgentAction, ToolResult
    from ..state.graph_state import GraphState
    from ..config.settings import Phase
    from ..llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """
    Central coordinator — simple rule-based routing through the pipeline.
    """

    def __init__(self, llm_client: OllamaClient):
        super().__init__(
            name="planner",
            llm_client=llm_client,
            system_prompt="",
        )

    def get_available_tools(self) -> List[str]:
        return [
            "route_to_profiler",
            "route_to_analyst",
            "route_to_mask_filter",
            "route_to_optimizer",
            "done",
        ]

    def decide_route(self, state: GraphState) -> Tuple[str, Dict[str, float]]:
        """Decide which agent to route to next."""
        return self._rule_based_routing(state)

    def _rule_based_routing(self, state: GraphState) -> Tuple[str, Dict[str, float]]:
        """
        Linear routing:
          1. Profile not built → profiler
          2. No candidates → analyst (discover)
          3. Masks not filtered → mask_filter
          4. Not scored → analyst (score)
          5. Not selected → optimizer
          6. Done
        """
        # 1. Build RWTD profile
        if not state.profile_exists:
            state.current_phase = Phase.PROFILING
            return "profiler", {}

        # 2. Discover candidates
        if state.num_candidates == 0:
            state.current_phase = Phase.SCORING
            return "analyst", {}

        # 3. VLM mask filter
        if not state.mask_filtering_done:
            state.current_phase = Phase.MASK_FILTERING
            return "mask_filter", {}

        # 4. Score candidates (cosine similarity)
        if state.num_scored < state.num_mask_passed:
            state.current_phase = Phase.SCORING
            return "analyst", {}

        # 5. Select top-N
        if state.num_selected < state.config.target_n:
            state.current_phase = Phase.OPTIMIZING
            return "optimizer", {}

        # 6. Done
        state.current_phase = Phase.DONE
        return "done", {}

    def decide(self, state: GraphState, additional_context: str = "") -> AgentAction:
        route_to, adjustments = self.decide_route(state)
        reasoning = (
            f"phase={state.current_phase.value}, scored={state.num_scored}, "
            f"selected={state.num_selected}"
        )
        return AgentAction(
            tool_name=f"route_to_{route_to}" if route_to != "done" else "done",
            params={},
            reasoning=reasoning,
            agent_name=self.name,
        )

    def get_next_agent(self, state: GraphState) -> str:
        route_to, _ = self.decide_route(state)
        return route_to
