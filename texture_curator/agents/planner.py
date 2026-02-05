"""
Planner Agent for Texture Curator.

This is the "BRAIN" of the multi-agent system. It orchestrates the entire
workflow by deciding which agent to invoke next based on the current state.

ROLE:
- Monitor overall progress
- Decide which agent should act next
- Handle rerouting when quality is low
- Track iterations to prevent infinite loops

WORKFLOW:
1. PROFILER: Build RWTD profile (if not exists)
2. ANALYST: Score all candidates
3. CRITIC: Audit top candidates
4. If quality OK → OPTIMIZER
   If quality LOW → Adjust thresholds, back to ANALYST
5. OPTIMIZER: Select diverse final set
6. DONE

OUTPUT:
- Route decisions (which agent to invoke)
- Threshold adjustments (on reroute)
"""

import json
from typing import List, Dict, Any, Optional, Tuple
import logging

# Handle imports
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


# ============================================
# Planner Agent System Prompt
# ============================================

PLANNER_SYSTEM_PROMPT = """You are the PLANNER agent - the central coordinator of a texture dataset curation system.

YOUR ROLE:
Orchestrate the workflow between specialized agents to curate a high-quality dataset.

AGENTS AVAILABLE:
- profiler: Builds RWTD reference profile (must run first)
- analyst: Scores candidates against the profile
- critic: Audits quality using VLM (material vs object transitions)
- optimizer: Selects diverse final subset

WORKFLOW:
1. If profile doesn't exist → route to "profiler"
2. If candidates not scored → route to "analyst"
3. If quality not verified → route to "critic"
4. If critic passed quality gate → route to "optimizer"
5. If critic failed AND can reroute → adjust thresholds, route to "analyst"
6. If selection complete → route to "done"

REROUTE RULES:
- Maximum {max_iterations} reroute iterations
- On reroute, adjust thresholds based on critic feedback
- Common adjustments: increase boundary_sharpness_min, semantic_min

RESPONSE FORMAT:
Always respond with valid JSON:
{{
    "reasoning": "Your step-by-step analysis of current state",
    "route_to": "profiler|analyst|critic|optimizer|done",
    "threshold_adjustments": {{}}  // Only if rerouting
}}
"""


# ============================================
# Planner Agent
# ============================================

class PlannerAgent(BaseAgent):
    """
    Central coordinator that orchestrates the multi-agent workflow.
    """
    
    def __init__(
        self,
        llm_client: OllamaClient,
    ):
        """
        Initialize Planner agent.
        
        Args:
            llm_client: Ollama client for reasoning
        """
        super().__init__(
            name="planner",
            llm_client=llm_client,
            system_prompt=PLANNER_SYSTEM_PROMPT,
        )
    
    def get_available_tools(self) -> List[str]:
        # Planner doesn't use tools directly - it routes to other agents
        return [
            "route_to_profiler",
            "route_to_analyst",
            "route_to_critic",
            "route_to_optimizer",
            "adjust_thresholds",
            "done",
        ]
    
    def format_state_for_prompt(self, state: GraphState) -> str:
        """Format comprehensive state for planner decision."""
        lines = [
            "## Overall State",
            f"- Current Phase: {state.current_phase.value}",
            f"- Iteration: {state.iteration}/{state.config.max_iterations}",
            "",
            "## Progress",
            f"- RWTD Profile: {'✓ Built' if state.profile_exists else '✗ Not built'}",
            f"- Candidates Discovered: {state.num_candidates}",
            f"- Candidates Scored: {state.num_scored}",
            f"- Candidates Validated: {state.num_validated}",
            f"- Valid (Material Transitions): {state.num_valid}",
            f"- Selected: {state.num_selected}/{state.config.target_n}",
        ]
        
        # Add critic report if exists
        if state.critic_report:
            lines.extend([
                "",
                "## Critic Report",
                f"- Quality Score: {state.critic_report.quality_score:.0%}",
                f"- Threshold: {state.config.thresholds.quality_gate_min:.0%}",
                f"- Passed: {'✓' if state.critic_report.quality_score >= state.config.thresholds.quality_gate_min else '✗'}",
            ])
            
            if state.critic_report.issues_found:
                lines.append(f"- Issues: {', '.join(state.critic_report.issues_found)}")
            
            if state.critic_report.recommendations:
                lines.append(f"- Recommendations: {', '.join(state.critic_report.recommendations)}")
        
        # Add reroute history
        if state.reroute_history:
            lines.extend([
                "",
                "## Reroute History",
            ])
            for reroute in state.reroute_history:
                lines.append(f"- Iteration {reroute.iteration}: {reroute.reason}")
        
        return "\n".join(lines)
    
    def decide_route(self, state: GraphState) -> Tuple[str, Dict[str, float]]:
        """
        Decide which agent to route to next.
        
        This can use LLM or rule-based logic.
        
        Returns:
            Tuple of (route_to, threshold_adjustments)
        """
        # Use rule-based logic for reliability
        # (Can be replaced with LLM decision for more flexibility)
        
        return self._rule_based_routing(state)
    
    def _rule_based_routing(self, state: GraphState) -> Tuple[str, Dict[str, float]]:
        """
        Rule-based routing logic.
        
        This is deterministic and predictable, good for debugging.
        """
        adjustments = {}
        
        # Rule 1: If no profile, build it
        if not state.profile_exists:
            state.current_phase = Phase.PROFILING
            return "profiler", adjustments
        
        # Rule 2: If no candidates discovered or scored, run analyst
        if state.num_candidates == 0 or state.num_scored == 0:
            state.current_phase = Phase.SCORING
            return "analyst", adjustments
        
        # Rule 3: If not all candidates scored, continue scoring
        if state.num_scored < state.num_candidates:
            state.current_phase = Phase.SCORING
            return "analyst", adjustments
        
        # Rule 4: If not enough validated, run critic
        if state.num_validated < min(state.config.critic_sample_size, state.num_scored):
            state.current_phase = Phase.CRITIQUING
            return "critic", adjustments
        
        # Rule 5: Check critic results
        if state.critic_report:
            quality = state.critic_report.quality_score
            threshold = state.config.thresholds.quality_gate_min
            
            if quality >= threshold:
                # Quality OK - proceed to optimizer
                if state.num_selected < state.config.target_n:
                    state.current_phase = Phase.OPTIMIZING
                    return "optimizer", adjustments
                else:
                    state.current_phase = Phase.DONE
                    return "done", adjustments
            else:
                # Quality low - consider reroute
                if state.can_reroute:
                    # Adjust thresholds
                    adjustments = {
                        "boundary_sharpness_min": state.config.thresholds.boundary_sharpness_min * 1.2,
                        "semantic_min": min(state.config.thresholds.semantic_min + 0.05, 0.9),
                    }
                    
                    # Record reroute
                    state.record_reroute(
                        reason=f"Quality {quality:.0%} < threshold {threshold:.0%}",
                        adjustments=adjustments
                    )
                    
                    # Apply adjustments
                    state.config.thresholds.boundary_sharpness_min = adjustments["boundary_sharpness_min"]
                    state.config.thresholds.semantic_min = adjustments["semantic_min"]
                    
                    # Re-run analyst with new thresholds
                    state.current_phase = Phase.SCORING
                    return "analyst", adjustments
                else:
                    # Max iterations reached, proceed anyway
                    logger.warning(f"Max iterations reached. Proceeding with quality {quality:.0%}")
                    state.current_phase = Phase.OPTIMIZING
                    return "optimizer", adjustments
        
        # Rule 6: If selected enough, done
        if state.num_selected >= state.config.target_n:
            state.current_phase = Phase.DONE
            return "done", adjustments
        
        # Default: run optimizer
        state.current_phase = Phase.OPTIMIZING
        return "optimizer", adjustments
    
    def decide(self, state: GraphState, additional_context: str = "") -> AgentAction:
        """
        Override base decide to use routing logic.
        """
        route_to, adjustments = self.decide_route(state)
        
        reasoning = f"Based on state: profile={state.profile_exists}, scored={state.num_scored}, validated={state.num_validated}, selected={state.num_selected}"
        
        if adjustments:
            reasoning += f". Adjusting thresholds: {adjustments}"
        
        return AgentAction(
            tool_name=f"route_to_{route_to}" if route_to != "done" else "done",
            params={"adjustments": adjustments} if adjustments else {},
            reasoning=reasoning,
            agent_name=self.name,
        )
    
    def get_next_agent(self, state: GraphState) -> str:
        """
        Simple method to get the name of the next agent to run.
        """
        route_to, _ = self.decide_route(state)
        return route_to


# ============================================
# Quick Test
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Planner Agent Test")
    print("=" * 60)
    
    from llm.ollama_client import OllamaClient
    from state.graph_state import GraphState
    
    # Create planner
    client = OllamaClient(model="qwen2.5:7b")
    planner = PlannerAgent(llm_client=client)
    
    # Test routing decisions
    state = GraphState()
    
    print("\n--- Test 1: Initial State ---")
    route, adjustments = planner.decide_route(state)
    print(f"Route: {route}, Adjustments: {adjustments}")
    assert route == "profiler", "Should route to profiler first"
    print("✓ Correctly routes to profiler")
    
    # Mock profile exists
    print("\n--- Test 2: After Profiling ---")
    import numpy as np
    from state.models import RWTDProfile, Distribution
    
    state.rwtd_profile = RWTDProfile(
        num_samples=256,
        centroid_embedding=np.random.randn(768).astype(np.float32),
        entropy_distribution=Distribution(mean=5.0, std=1.0, min_val=2.0, max_val=8.0),
        boundary_sharpness_distribution=Distribution(mean=100.0, std=30.0, min_val=20.0, max_val=200.0),
    )
    
    route, adjustments = planner.decide_route(state)
    print(f"Route: {route}, Adjustments: {adjustments}")
    assert route == "analyst", "Should route to analyst after profiling"
    print("✓ Correctly routes to analyst")
    
    print("\n✅ Planner Agent test passed!")