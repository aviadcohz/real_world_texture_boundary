"""
Optimizer Agent for Texture Curator.

This agent is responsible for selecting the final diverse subset
from validated candidates.

ROLE:
- Filter to only valid (material transition) candidates
- Apply coreset/diversity selection
- Balance quality and diversity
- Produce final selection

ALGORITHM:
Greedy coreset selection that balances quality and diversity:
  score(next) = quality_score - λ * max_similarity_to_already_selected

This ensures we don't just pick the top-N most similar images.

OUTPUT:
- selected_ids list in GraphState
- SelectionReport with statistics
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

# Handle imports
try:
    from agents.base import BaseAgent, AgentAction, ToolResult
    from state.graph_state import GraphState
    from state.models import SelectionReport
    from llm.ollama_client import OllamaClient
except ImportError:
    from .base import BaseAgent, AgentAction, ToolResult
    from ..state.graph_state import GraphState
    from ..state.models import SelectionReport
    from ..llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


# ============================================
# Optimizer Agent System Prompt
# ============================================

OPTIMIZER_SYSTEM_PROMPT = """You are the OPTIMIZER agent in a texture dataset curation system.

YOUR ROLE:
Select the final diverse subset from validated candidates.
You must balance QUALITY (high scores) with DIVERSITY (not redundant).

AVAILABLE TOOLS:
- get_valid_candidates: Get candidates that passed critic review
- compute_diversity_matrix: Compute pairwise similarities
- greedy_select: Run greedy coreset selection
- verify_selection: Verify the final selection meets requirements
- export_selection: Export the final dataset
- done: Mark optimization as complete

SELECTION CRITERIA:
1. Must have passed critic review (is_material_transition = True)
2. Balance quality score with diversity contribution
3. Avoid selecting very similar images

DIVERSITY ALGORITHM:
For each selection step:
  score(candidate) = quality - λ * max_similarity_to_selected
  
This penalizes candidates that are too similar to already selected ones.

RESPONSE FORMAT:
Always respond with valid JSON:
{
    "reasoning": "Your analysis",
    "tool": "tool_name",
    "params": {}
}
"""


# ============================================
# Optimizer Agent
# ============================================

class OptimizerAgent(BaseAgent):
    """
    Agent responsible for diverse selection of final candidates.
    """
    
    def __init__(
        self,
        llm_client: OllamaClient,
    ):
        """
        Initialize Optimizer agent.
        
        Args:
            llm_client: Ollama client for reasoning
        """
        super().__init__(
            name="optimizer",
            llm_client=llm_client,
            system_prompt=OPTIMIZER_SYSTEM_PROMPT,
        )
        
        # Cached similarity matrix
        self._similarity_matrix = None
        self._candidate_ids_for_matrix = None
    
    def get_available_tools(self) -> List[str]:
        return [
            "get_valid_candidates",
            "compute_diversity_matrix",
            "greedy_select",
            "verify_selection",
            "export_selection",
            "done",
        ]
    
    def format_state_for_prompt(self, state: GraphState) -> str:
        """Format state with optimizer-specific info."""
        valid_candidates = state.get_valid_candidates()
        
        lines = [
            f"## Optimizer State",
            f"- Total Candidates: {state.num_candidates}",
            f"- Valid Candidates: {len(valid_candidates)}",
            f"- Target Selection: {state.config.target_n}",
            f"- Currently Selected: {state.num_selected}",
            f"- Diversity Weight: {state.config.thresholds.diversity_weight}",
        ]
        
        if state.selection_report:
            lines.append(f"\n## Selection Report")
            lines.append(f"- Diversity Score: {state.selection_report.diversity_score:.3f}")
            lines.append(f"- Mean Quality: {state.selection_report.mean_quality_score:.3f}")
        
        return "\n".join(lines)
    
    def execute_tool(self, action: AgentAction, state: GraphState) -> ToolResult:
        """Execute an optimizer tool."""
        tool_name = action.tool_name
        params = action.params
        
        try:
            if tool_name == "get_valid_candidates":
                return self._get_valid_candidates(state)
            
            elif tool_name == "compute_diversity_matrix":
                return self._compute_diversity_matrix(state)
            
            elif tool_name == "greedy_select":
                n = params.get("n", state.config.target_n)
                diversity_weight = params.get("diversity_weight", state.config.thresholds.diversity_weight)
                return self._greedy_select(state, n, diversity_weight)
            
            elif tool_name == "verify_selection":
                return self._verify_selection(state)
            
            elif tool_name == "export_selection":
                return self._export_selection(state)
            
            elif tool_name == "done":
                return ToolResult(success=True, data={"message": "Optimization complete"})
            
            else:
                return ToolResult(success=False, error_message=f"Unknown tool: {tool_name}")
                
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            import traceback
            traceback.print_exc()
            return ToolResult(success=False, error_message=str(e))
    
    def _get_valid_candidates(self, state: GraphState) -> ToolResult:
        """Get all candidates that passed critic review."""
        valid = state.get_valid_candidates()
        
        # Sort by score
        valid.sort(key=lambda c: c.scores.total_score if c.scores else 0, reverse=True)
        
        return ToolResult(
            success=True,
            data={
                "valid_count": len(valid),
                "target_n": state.config.target_n,
                "can_select": len(valid) >= state.config.target_n,
                "top_10": [
                    {"id": c.id, "score": c.scores.total_score if c.scores else 0}
                    for c in valid[:10]
                ]
            }
        )
    
    def _compute_diversity_matrix(self, state: GraphState) -> ToolResult:
        """Compute pairwise cosine similarity matrix for valid candidates."""
        valid = state.get_valid_candidates()
        
        if not valid:
            return ToolResult(
                success=False,
                error_message="No valid candidates to compute diversity matrix"
            )
        
        # Get embeddings
        embeddings = []
        ids = []
        
        for c in valid:
            if c.features and c.features.dino_embedding is not None:
                embeddings.append(c.features.dino_embedding)
                ids.append(c.id)
        
        if not embeddings:
            return ToolResult(
                success=False,
                error_message="No embeddings found for valid candidates"
            )
        
        embeddings = np.array(embeddings)
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = embeddings / norms
        
        # Compute cosine similarity matrix
        similarity_matrix = normalized @ normalized.T
        
        # Cache for later use
        self._similarity_matrix = similarity_matrix
        self._candidate_ids_for_matrix = ids
        
        # Compute stats
        upper_tri = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        return ToolResult(
            success=True,
            data={
                "matrix_size": similarity_matrix.shape[0],
                "mean_similarity": float(np.mean(upper_tri)),
                "max_similarity": float(np.max(upper_tri)),
                "min_similarity": float(np.min(upper_tri)),
            }
        )
    
    def _greedy_select(
        self,
        state: GraphState,
        n: int,
        diversity_weight: float,
    ) -> ToolResult:
        """
        Greedy coreset selection balancing quality and diversity.
        
        Algorithm:
        1. Start with empty selection
        2. For each step:
           - For each candidate not selected:
             - Compute score = quality - λ * max_similarity_to_selected
           - Select candidate with highest score
        3. Repeat until n selected
        """
        logger.info(f"Running greedy selection: n={n}, diversity_weight={diversity_weight}")
        
        # Get valid candidates
        valid = state.get_valid_candidates()
        
        if len(valid) < n:
            logger.warning(f"Only {len(valid)} valid candidates, requested {n}")
            n = len(valid)
        
        if n == 0:
            return ToolResult(
                success=False,
                error_message="No valid candidates to select"
            )
        
        # Ensure we have similarity matrix
        if self._similarity_matrix is None or self._candidate_ids_for_matrix is None:
            result = self._compute_diversity_matrix(state)
            if not result.success:
                return result
        
        # Create ID to index mapping
        id_to_idx = {cid: i for i, cid in enumerate(self._candidate_ids_for_matrix)}
        
        # Create ID to candidate mapping
        id_to_candidate = {c.id: c for c in valid}
        
        # Filter to only candidates in similarity matrix
        valid_ids = [c.id for c in valid if c.id in id_to_idx]
        
        if len(valid_ids) < n:
            n = len(valid_ids)
        
        # Greedy selection
        selected_ids = []
        selected_indices = []
        remaining_ids = set(valid_ids)
        
        for step in range(n):
            best_id = None
            best_score = float('-inf')
            
            for cid in remaining_ids:
                candidate = id_to_candidate[cid]
                quality = candidate.scores.total_score if candidate.scores else 0
                
                # Compute diversity penalty
                if selected_indices:
                    idx = id_to_idx[cid]
                    similarities = [self._similarity_matrix[idx, sel_idx] for sel_idx in selected_indices]
                    max_sim = max(similarities)
                    diversity_penalty = diversity_weight * max_sim
                else:
                    diversity_penalty = 0
                
                # Combined score
                score = quality - diversity_penalty
                
                if score > best_score:
                    best_score = score
                    best_id = cid
            
            if best_id is None:
                break
            
            # Select this candidate
            selected_ids.append(best_id)
            selected_indices.append(id_to_idx[best_id])
            remaining_ids.remove(best_id)
            
            logger.debug(f"Step {step+1}: Selected {best_id} (score={best_score:.3f})")
        
        # Store in state
        state.selected_ids = selected_ids
        
        # Mark candidates as selected
        for cid in selected_ids:
            state.candidates[cid].is_selected = True
        
        # Compute final diversity score
        if len(selected_indices) > 1:
            selected_sims = []
            for i in range(len(selected_indices)):
                for j in range(i+1, len(selected_indices)):
                    selected_sims.append(self._similarity_matrix[selected_indices[i], selected_indices[j]])
            diversity_score = 1 - np.mean(selected_sims)  # Higher is more diverse
        else:
            diversity_score = 1.0
        
        # Compute mean quality
        selected_candidates = [id_to_candidate[cid] for cid in selected_ids]
        mean_quality = np.mean([c.scores.total_score for c in selected_candidates if c.scores])
        
        # Create report
        state.selection_report = SelectionReport(
            total_candidates=state.num_candidates,
            passed_quality_gate=len(valid),
            final_selected=len(selected_ids),
            diversity_score=diversity_score,
            mean_quality_score=mean_quality,
        )
        
        return ToolResult(
            success=True,
            data={
                "selected_count": len(selected_ids),
                "selected_ids": selected_ids,
                "diversity_score": diversity_score,
                "mean_quality": mean_quality,
            }
        )
    
    def _verify_selection(self, state: GraphState) -> ToolResult:
        """Verify the final selection meets requirements."""
        issues = []
        
        # Check count
        if len(state.selected_ids) < state.config.target_n:
            issues.append(f"Selected {len(state.selected_ids)}, target was {state.config.target_n}")
        
        # Check all are valid
        for cid in state.selected_ids:
            candidate = state.candidates.get(cid)
            if candidate is None:
                issues.append(f"Selected candidate {cid} not found")
            elif candidate.critic_verdict is None:
                issues.append(f"Selected candidate {cid} has no critic verdict")
            elif not candidate.critic_verdict.is_material_transition:
                issues.append(f"Selected candidate {cid} is not a material transition")
        
        # Check diversity
        if state.selection_report and state.selection_report.diversity_score < 0.3:
            issues.append(f"Low diversity score: {state.selection_report.diversity_score:.3f}")
        
        return ToolResult(
            success=len(issues) == 0,
            data={
                "verified": len(issues) == 0,
                "issues": issues,
                "selection_count": len(state.selected_ids),
            }
        )
    
    def _export_selection(self, state: GraphState) -> ToolResult:
        """Export the final selection to output directory."""
        import shutil
        
        output_dir = state.config.output_path / "curated_dataset"
        images_dir = output_dir / "images"
        masks_dir = output_dir / "masks"
        
        # Create directories
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        exported = 0
        for cid in state.selected_ids:
            candidate = state.candidates.get(cid)
            if candidate is None:
                continue
            
            # Copy image
            src_image = candidate.image_path
            dst_image = images_dir / src_image.name
            shutil.copy2(src_image, dst_image)
            
            # Copy mask
            src_mask = candidate.mask_path
            dst_mask = masks_dir / src_mask.name
            shutil.copy2(src_mask, dst_mask)
            
            exported += 1
        
        # Save metadata
        import json
        metadata = {
            "num_images": exported,
            "candidates": [
                {
                    "id": cid,
                    "score": float(state.candidates[cid].scores.total_score) if state.candidates[cid].scores else 0,
                }
                for cid in state.selected_ids
            ],
            "selection_report": state.selection_report.to_dict() if state.selection_report else None,
        }
        
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return ToolResult(
            success=True,
            data={
                "exported": exported,
                "output_dir": str(output_dir),
            }
        )
    
    def run_full_optimization(self, state: GraphState) -> ToolResult:
        """
        Run the complete optimization pipeline without LLM.
        """
        logger.info("Running full optimization pipeline...")
        
        # Step 1: Get valid candidates
        result = self._get_valid_candidates(state)
        if not result.success:
            return result
        logger.info(f"Valid candidates: {result.data}")
        
        if not result.data["can_select"]:
            return ToolResult(
                success=False,
                error_message=f"Not enough valid candidates. Have {result.data['valid_count']}, need {state.config.target_n}"
            )
        
        # Step 2: Compute diversity matrix
        result = self._compute_diversity_matrix(state)
        if not result.success:
            return result
        logger.info(f"Diversity matrix: {result.data}")
        
        # Step 3: Greedy selection
        result = self._greedy_select(
            state,
            state.config.target_n,
            state.config.thresholds.diversity_weight
        )
        if not result.success:
            return result
        logger.info(f"Selection: {result.data}")
        
        # Step 4: Verify
        result = self._verify_selection(state)
        logger.info(f"Verification: {result.data}")
        
        # Step 5: Export
        result = self._export_selection(state)
        logger.info(f"Export: {result.data}")
        
        return result


# ============================================
# Quick Test
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Optimizer Agent Test")
    print("=" * 60)
    print("\nNote: This test requires validated candidates.")
    print("Run ProfilerAgent, AnalystAgent, and CriticAgent first.")