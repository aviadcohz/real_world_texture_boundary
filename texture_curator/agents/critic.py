"""
Critic Agent for Texture Curator.

This agent is responsible for quality auditing using VLM (Vision-Language Model).
It verifies that candidates represent MATERIAL transitions, not object boundaries.

ROLE:
- Sample top candidates for review
- Use VLM to analyze each image/mask pair
- Determine if transition is material-based (good) or object-based (bad)
- Produce quality report
- Recommend rerouting if quality is low

KEY DISTINCTION:
✓ GOOD: Material/texture transitions (grass→dirt, brick→concrete, wood→metal)
✗ BAD: Object boundaries (car edge, person silhouette, furniture outline)

OUTPUT:
- CriticVerdict for each reviewed candidate
- CriticReport with quality score and recommendations
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import base64

# Handle imports
try:
    from agents.base import BaseAgent, AgentAction, ToolResult
    from state.graph_state import GraphState
    from state.models import CriticVerdict, CriticReport
    from llm.ollama_client import OllamaClient
except ImportError:
    from .base import BaseAgent, AgentAction, ToolResult
    from ..state.graph_state import GraphState
    from ..state.models import CriticVerdict, CriticReport
    from ..llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


# ============================================
# Critic Agent System Prompt
# ============================================

CRITIC_SYSTEM_PROMPT = """You are the CRITIC agent in a texture dataset curation system.

YOUR ROLE:
Audit the quality of top-scoring candidates to ensure they represent
TEXTURE transitions, not object boundaries.

CRITICAL DISTINCTION:
✓ GOOD (Material Transition): The boundary between two different texture patterns.
  Examples: grass meeting pavement, brick wall meeting concrete, wood floor meeting tile etc.
  
✗ BAD (Object Boundary): The edge of an objects.

AVAILABLE TOOLS:
- sample_candidates: Select top candidates for review
- vlm_audit: Use vision model to analyze a candidate
- compile_report: Generate quality report
- recommend_reroute: Suggest threshold adjustments if quality is low
- done: Mark critique as complete

WORKFLOW:
1. Sample top N candidates for review
2. For each candidate, run VLM audit
3. Compile report with quality score
4. If quality < threshold, recommend reroute

RESPONSE FORMAT:
Always respond with valid JSON:
{
    "reasoning": "Your analysis",
    "tool": "tool_name",
    "params": {}
}
"""


# VLM Audit Prompt Template
VLM_AUDIT_PROMPT = """Analyze this image with its segmentation mask.

The mask (shown as a highlighted region) divides the image into two parts.

QUESTION: Does the mask boundary represent:
A) A TEXTURE TRANSITION - where two different pattern meet
   (e.g., grass meeting concrete, brick meeting plaster, wood meeting tile)

B) An OBJECT BOUNDARY - the edge/outline of an object
   

Look at WHAT is on each side of the boundary:
- If one side at least is surfaces/materials/textures → MATERIAL TRANSITION
- If one side is an object and other is object too → OBJECT BOUNDARY

Respond with JSON:
{
    "classification": "material_transition" or "object_boundary",
    "confidence": 0.0-1.0,
    "left_side": "description of what's on left of boundary",
    "right_side": "description of what's on right of boundary",
    "reasoning": "brief explanation"
}
"""


# ============================================
# Critic Agent
# ============================================

class CriticAgent(BaseAgent):
    """
    Agent responsible for quality auditing using VLM.
    
    Uses a Vision-Language Model to verify that candidates
    represent true material transitions.
    """
    
    def __init__(
        self,
        llm_client: OllamaClient,
        vlm_model: str = "qwen2.5-vl:7b",  # or "llava:7b"
        device: str = "cuda",
    ):
        """
        Initialize Critic agent.
        
        Args:
            llm_client: Ollama client for reasoning
            vlm_model: Vision-Language model for image analysis
            device: Device for VLM
        """
        super().__init__(
            name="critic",
            llm_client=llm_client,
            system_prompt=CRITIC_SYSTEM_PROMPT,
        )
        
        self.vlm_model = vlm_model
        self.device = device
        
        # VLM client (may be same or different from reasoning LLM)
        self._vlm_client = None
        
        # Track reviewed candidates
        self._reviewed_ids = set()
    
    @property
    def vlm_client(self) -> OllamaClient:
        """Get or create VLM client."""
        if self._vlm_client is None:
            self._vlm_client = OllamaClient(model=self.vlm_model)
        return self._vlm_client
    
    def get_available_tools(self) -> List[str]:
        return [
            "sample_candidates",
            "vlm_audit",
            "vlm_audit_batch",
            "compile_report",
            "recommend_reroute",
            "done",
        ]
    
    def format_state_for_prompt(self, state: GraphState) -> str:
        """Format state with critic-specific info."""
        lines = [
            f"## Critic State",
            f"- Candidates Scored: {state.num_scored}",
            f"- Candidates Validated: {state.num_validated}",
            f"- Valid (Material Transitions): {state.num_valid}",
            f"- Quality Gate Threshold: {state.config.thresholds.quality_gate_min:.0%}",
        ]
        
        if state.critic_report:
            lines.append(f"\n## Current Quality Report")
            lines.append(f"- Reviewed: {state.critic_report.samples_reviewed}")
            lines.append(f"- Material Transitions: {state.critic_report.material_transitions}")
            lines.append(f"- Object Boundaries: {state.critic_report.object_boundaries}")
            lines.append(f"- Quality Score: {state.critic_report.quality_score:.0%}")
        
        return "\n".join(lines)
    
    def execute_tool(self, action: AgentAction, state: GraphState) -> ToolResult:
        """Execute a critic tool."""
        tool_name = action.tool_name
        params = action.params
        
        try:
            if tool_name == "sample_candidates":
                n = params.get("n", state.config.critic_sample_size)
                return self._sample_candidates(state, n)
            
            elif tool_name == "vlm_audit":
                candidate_id = params.get("candidate_id")
                return self._vlm_audit(state, candidate_id)
            
            elif tool_name == "vlm_audit_batch":
                candidate_ids = params.get("candidate_ids", [])
                return self._vlm_audit_batch(state, candidate_ids)
            
            elif tool_name == "compile_report":
                return self._compile_report(state)
            
            elif tool_name == "recommend_reroute":
                return self._recommend_reroute(state)
            
            elif tool_name == "done":
                return ToolResult(success=True, data={"message": "Critique complete"})
            
            else:
                return ToolResult(success=False, error_message=f"Unknown tool: {tool_name}")
                
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            import traceback
            traceback.print_exc()
            return ToolResult(success=False, error_message=str(e))
    
    def _sample_candidates(self, state: GraphState, n: int) -> ToolResult:
        """Sample top candidates for review."""
        # Get top candidates that haven't been reviewed
        all_top = state.get_top_candidates(n * 2)  # Get extra in case some reviewed
        
        to_review = []
        for c in all_top:
            if c.id not in self._reviewed_ids and c.critic_verdict is None:
                to_review.append(c)
                if len(to_review) >= n:
                    break
        
        return ToolResult(
            success=True,
            data={
                "sampled": len(to_review),
                "candidate_ids": [c.id for c in to_review],
                "scores": [c.scores.total_score if c.scores else 0 for c in to_review],
            }
        )
    
    def _vlm_audit(self, state: GraphState, candidate_id: str) -> ToolResult:
        """Audit a single candidate using VLM."""
        if candidate_id not in state.candidates:
            return ToolResult(
                success=False,
                error_message=f"Candidate not found: {candidate_id}"
            )
        
        candidate = state.candidates[candidate_id]
        
        # For now, use a simpler heuristic-based approach
        # In production, this would call an actual VLM
        verdict = self._analyze_candidate_heuristic(candidate, state)
        
        # Store verdict
        candidate.critic_verdict = verdict
        self._reviewed_ids.add(candidate_id)
        
        return ToolResult(
            success=True,
            data={
                "candidate_id": candidate_id,
                "is_material_transition": verdict.is_material_transition,
                "confidence": verdict.confidence,
                "reasoning": verdict.reasoning,
            }
        )
    
    def _vlm_audit_batch(self, state: GraphState, candidate_ids: List[str]) -> ToolResult:
        """Audit multiple candidates."""
        results = []
        
        for cid in candidate_ids:
            result = self._vlm_audit(state, cid)
            results.append({
                "id": cid,
                "success": result.success,
                "is_material": result.data.get("is_material_transition") if result.success else None,
            })
        
        material_count = sum(1 for r in results if r.get("is_material"))
        
        return ToolResult(
            success=True,
            data={
                "reviewed": len(results),
                "material_transitions": material_count,
                "object_boundaries": len(results) - material_count,
            }
        )
    
    def _analyze_candidate_heuristic(self, candidate, state: GraphState) -> CriticVerdict:
        """
        Analyze candidate using heuristics (fallback when VLM unavailable).
        
        This uses the features we've already extracted to make a determination.
        In a real system, this would be replaced by actual VLM inference.
        """
        features = candidate.features
        
        if features is None:
            return CriticVerdict(
                is_material_transition=False,
                confidence=0.0,
                reasoning="No features extracted"
            )
        
        # Heuristic: Material transitions tend to have:
        # 1. High edge density (boundary follows real edge)
        # 2. High boundary sharpness (VoL)
        # 3. Moderate entropy (textures are complex but not random)
        
        edge_density = features.edge_density
        vol = features.boundary_sharpness
        entropy = features.entropy
        
        # Score based on heuristics
        score = 0.0
        reasons = []
        
        # Edge density: higher is better
        if edge_density > 0.5:
            score += 0.35
            reasons.append(f"Good edge alignment ({edge_density:.0%})")
        elif edge_density > 0.3:
            score += 0.2
            reasons.append(f"Moderate edge alignment ({edge_density:.0%})")
        else:
            reasons.append(f"Poor edge alignment ({edge_density:.0%})")
        
        # Variance of Laplacian: higher means sharper boundary
        if state.rwtd_profile and state.rwtd_profile.boundary_sharpness_distribution:
            vol_dist = state.rwtd_profile.boundary_sharpness_distribution
            vol_score = vol_dist.score_value(vol)
            score += 0.35 * vol_score
            if vol_score > 0.5:
                reasons.append(f"Sharp boundary (VoL={vol:.1f})")
            else:
                reasons.append(f"Soft boundary (VoL={vol:.1f})")
        
        # Entropy: should be in a reasonable range (not too uniform, not noise)
        if state.rwtd_profile and state.rwtd_profile.entropy_distribution:
            ent_dist = state.rwtd_profile.entropy_distribution
            ent_score = ent_dist.score_value(entropy)
            score += 0.3 * ent_score
            if ent_score > 0.5:
                reasons.append(f"Good texture complexity")
        
        # Determine verdict
        is_material = score > 0.5
        
        return CriticVerdict(
            is_material_transition=is_material,
            confidence=min(score, 1.0),
            reasoning="; ".join(reasons)
        )
    
    def _compile_report(self, state: GraphState) -> ToolResult:
        """Compile quality report from reviewed candidates."""
        # Count verdicts
        material_count = 0
        object_count = 0
        
        for candidate in state.candidates.values():
            if candidate.critic_verdict is not None:
                if candidate.critic_verdict.is_material_transition:
                    material_count += 1
                else:
                    object_count += 1
        
        total_reviewed = material_count + object_count
        quality_score = material_count / total_reviewed if total_reviewed > 0 else 0
        
        # Identify issues
        issues = []
        recommendations = []
        
        if quality_score < state.config.thresholds.quality_gate_min:
            issues.append(f"Quality score ({quality_score:.0%}) below threshold ({state.config.thresholds.quality_gate_min:.0%})")
            recommendations.append("Increase boundary sharpness threshold")
            recommendations.append("Increase edge density threshold")
        
        if object_count > material_count:
            issues.append(f"More object boundaries ({object_count}) than material transitions ({material_count})")
            recommendations.append("Filter candidates with low edge density")
        
        # Create report
        report = CriticReport(
            samples_reviewed=total_reviewed,
            material_transitions=material_count,
            object_boundaries=object_count,
            issues_found=issues,
            recommendations=recommendations,
        )
        
        # Store in state
        state.critic_report = report
        
        return ToolResult(
            success=True,
            data={
                "samples_reviewed": total_reviewed,
                "material_transitions": material_count,
                "object_boundaries": object_count,
                "quality_score": quality_score,
                "passed": quality_score >= state.config.thresholds.quality_gate_min,
                "issues": issues,
                "recommendations": recommendations,
            }
        )
    
    def _recommend_reroute(self, state: GraphState) -> ToolResult:
        """Recommend threshold adjustments for rerouting."""
        if state.critic_report is None:
            return ToolResult(
                success=False,
                error_message="No critic report. Run compile_report first."
            )
        
        report = state.critic_report
        
        # Calculate adjustments
        adjustments = {}
        
        # If too many object boundaries, increase thresholds
        if report.object_boundaries > report.material_transitions:
            # Increase boundary sharpness minimum
            current = state.config.thresholds.boundary_sharpness_min
            adjustments["boundary_sharpness_min"] = current * 1.2
            
            # Increase semantic threshold
            current = state.config.thresholds.semantic_min
            adjustments["semantic_min"] = min(current + 0.05, 0.9)
        
        return ToolResult(
            success=True,
            data={
                "should_reroute": report.quality_score < state.config.thresholds.quality_gate_min,
                "current_quality": report.quality_score,
                "threshold": state.config.thresholds.quality_gate_min,
                "recommended_adjustments": adjustments,
            }
        )
    
    def run_full_critique(self, state: GraphState, n_samples: int = None) -> ToolResult:
        """
        Run the complete critique pipeline without LLM.
        
        Args:
            state: Current graph state
            n_samples: Number of candidates to review (default: config.critic_sample_size)
        """
        logger.info("Running full critique pipeline...")
        
        n = n_samples or state.config.critic_sample_size
        
        # Step 1: Sample candidates
        result = self._sample_candidates(state, n)
        if not result.success:
            return result
        
        candidate_ids = result.data["candidate_ids"]
        logger.info(f"Sampled {len(candidate_ids)} candidates for review")
        
        # Step 2: Audit each candidate
        for cid in candidate_ids:
            result = self._vlm_audit(state, cid)
            if result.success:
                logger.info(f"  {cid}: {'✓ Material' if result.data['is_material_transition'] else '✗ Object'}")
        
        # Step 3: Compile report
        result = self._compile_report(state)
        logger.info(f"Quality report: {result.data}")
        
        return result


# ============================================
# Quick Test
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Critic Agent Test")
    print("=" * 60)
    print("\nNote: This test requires scored candidates.")
    print("Run ProfilerAgent and AnalystAgent first.")