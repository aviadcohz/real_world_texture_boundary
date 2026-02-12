"""
Optimizer Agent for Texture Curator.

Selects the final diverse subset from scored, mask-passed candidates.

ALGORITHM:
Greedy coreset selection balancing quality and diversity:
  score(next) = quality_score - λ * max_similarity_to_already_selected
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

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


class OptimizerAgent(BaseAgent):
    """
    Selects diverse top-N from scored + mask-passed candidates.
    """

    def __init__(self, llm_client: OllamaClient):
        super().__init__(
            name="optimizer",
            llm_client=llm_client,
            system_prompt="",
        )
        self._similarity_matrix = None
        self._candidate_ids_for_matrix = None

    def get_available_tools(self) -> List[str]:
        return [
            "get_candidates",
            "compute_diversity_matrix",
            "greedy_select",
            "export_selection",
            "done",
        ]

    def execute_tool(self, action: AgentAction, state: GraphState) -> ToolResult:
        tool_name = action.tool_name
        params = action.params
        try:
            if tool_name == "get_candidates":
                return self._get_candidates(state)
            elif tool_name == "compute_diversity_matrix":
                return self._compute_diversity_matrix(state)
            elif tool_name == "greedy_select":
                n = params.get("n", state.config.target_n)
                dw = params.get("diversity_weight", state.config.thresholds.diversity_weight)
                return self._greedy_select(state, n, dw)
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

    def _get_candidates(self, state: GraphState) -> ToolResult:
        """Get all scored, mask-passed candidates sorted by score."""
        candidates = state.get_scored_candidates()
        return ToolResult(
            success=True,
            data={
                "count": len(candidates),
                "target_n": state.config.target_n,
                "can_select": len(candidates) >= state.config.target_n,
                "top_10": [
                    {"id": c.id, "score": c.scores.total_score}
                    for c in candidates[:10]
                ],
            },
        )

    def _compute_diversity_matrix(self, state: GraphState) -> ToolResult:
        """Compute pairwise cosine similarity matrix."""
        candidates = state.get_scored_candidates()

        if not candidates:
            return ToolResult(success=False, error_message="No scored candidates")

        embeddings = []
        ids = []
        for c in candidates:
            if c.features and c.features.dino_embedding is not None:
                embeddings.append(c.features.dino_embedding)
                ids.append(c.id)

        if not embeddings:
            return ToolResult(success=False, error_message="No embeddings found")

        embeddings = np.array(embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = embeddings / norms

        similarity_matrix = normalized @ normalized.T

        self._similarity_matrix = similarity_matrix
        self._candidate_ids_for_matrix = ids

        upper_tri = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]

        return ToolResult(
            success=True,
            data={
                "matrix_size": similarity_matrix.shape[0],
                "mean_similarity": float(np.mean(upper_tri)),
                "max_similarity": float(np.max(upper_tri)) if len(upper_tri) > 0 else 0,
                "min_similarity": float(np.min(upper_tri)) if len(upper_tri) > 0 else 0,
            },
        )

    def _greedy_select(self, state: GraphState, n: int, diversity_weight: float) -> ToolResult:
        """Greedy coreset selection balancing quality and diversity."""
        logger.info(f"Running greedy selection: n={n}, diversity_weight={diversity_weight}")

        candidates = state.get_scored_candidates()

        if len(candidates) < n:
            logger.warning(f"Only {len(candidates)} candidates, requested {n}")
            n = len(candidates)

        if n == 0:
            return ToolResult(success=False, error_message="No candidates to select")

        # Ensure similarity matrix
        if self._similarity_matrix is None:
            result = self._compute_diversity_matrix(state)
            if not result.success:
                return result

        id_to_idx = {cid: i for i, cid in enumerate(self._candidate_ids_for_matrix)}
        id_to_candidate = {c.id: c for c in candidates}

        valid_ids = [c.id for c in candidates if c.id in id_to_idx]
        if len(valid_ids) < n:
            n = len(valid_ids)

        selected_ids = []
        selected_indices = []
        remaining_ids = set(valid_ids)

        for step in range(n):
            best_id = None
            best_score = float('-inf')

            for cid in remaining_ids:
                quality = id_to_candidate[cid].scores.total_score

                if selected_indices:
                    idx = id_to_idx[cid]
                    max_sim = max(self._similarity_matrix[idx, si] for si in selected_indices)
                    diversity_penalty = diversity_weight * max_sim
                else:
                    diversity_penalty = 0

                score = quality - diversity_penalty
                if score > best_score:
                    best_score = score
                    best_id = cid

            if best_id is None:
                break

            selected_ids.append(best_id)
            selected_indices.append(id_to_idx[best_id])
            remaining_ids.remove(best_id)

        state.selected_ids = selected_ids

        for cid in selected_ids:
            state.candidates[cid].is_selected = True

        # Diversity score
        if len(selected_indices) > 1:
            sims = []
            for i in range(len(selected_indices)):
                for j in range(i + 1, len(selected_indices)):
                    sims.append(self._similarity_matrix[selected_indices[i], selected_indices[j]])
            diversity_score = 1 - np.mean(sims)
        else:
            diversity_score = 1.0

        selected_candidates = [id_to_candidate[cid] for cid in selected_ids]
        mean_quality = float(np.mean([c.scores.total_score for c in selected_candidates]))

        state.selection_report = SelectionReport(
            total_candidates=state.num_candidates,
            passed_quality_gate=len(candidates),
            final_selected=len(selected_ids),
            diversity_score=diversity_score,
            mean_quality_score=mean_quality,
        )

        return ToolResult(
            success=True,
            data={
                "selected_count": len(selected_ids),
                "diversity_score": diversity_score,
                "mean_quality": mean_quality,
            },
        )

    def _export_selection(self, state: GraphState) -> ToolResult:
        """Export selected images to output directory."""
        import shutil

        output_dir = state.config.output_path / "curated_dataset"
        images_dir = output_dir / "images"
        masks_dir = output_dir / "masks"

        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        exported = 0
        for cid in state.selected_ids:
            candidate = state.candidates.get(cid)
            if candidate is None:
                continue
            shutil.copy2(candidate.image_path, images_dir / candidate.image_path.name)
            shutil.copy2(candidate.mask_path, masks_dir / candidate.mask_path.name)
            exported += 1

        import json
        metadata = {
            "num_images": exported,
            "candidates": [
                {"id": cid, "score": float(state.candidates[cid].scores.total_score)}
                for cid in state.selected_ids
                if state.candidates[cid].scores
            ],
            "selection_report": state.selection_report.to_dict() if state.selection_report else None,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return ToolResult(
            success=True,
            data={"exported": exported, "output_dir": str(output_dir)},
        )

    def run_full_optimization(self, state: GraphState) -> ToolResult:
        """Run the complete optimization pipeline."""
        logger.info("Running full optimization pipeline...")

        result = self._get_candidates(state)
        if not result.success:
            return result
        logger.info(f"Candidates: {result.data}")

        if result.data["count"] == 0:
            return ToolResult(success=False, error_message="No scored candidates available")

        # If fewer candidates than target, select all
        actual_target = min(state.config.target_n, result.data["count"])
        if actual_target < state.config.target_n:
            logger.warning(
                f"Only {result.data['count']} candidates available, "
                f"selecting {actual_target} instead of {state.config.target_n}"
            )

        result = self._compute_diversity_matrix(state)
        if not result.success:
            return result
        logger.info(f"Diversity matrix: {result.data}")

        result = self._greedy_select(
            state, actual_target, state.config.thresholds.diversity_weight
        )
        if not result.success:
            return result
        logger.info(f"Selection: {result.data}")

        result = self._export_selection(state)
        logger.info(f"Export: {result.data}")

        return result
