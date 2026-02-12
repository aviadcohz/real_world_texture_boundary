"""
Analyst Agent for Texture Curator.

Discovers candidates and scores them by cosine similarity to the RWTD centroid.

FLOW:
1. Discover all image/mask pairs in source pool
2. Extract DINOv2 embeddings
3. Score each candidate = cosine_similarity(embedding, centroid)
"""

import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

try:
    from agents.base import BaseAgent, AgentAction, ToolResult
    from state.graph_state import GraphState
    from state.models import CandidateRecord, CandidateFeatures, ScoreBreakdown
    from config.settings import MaskStatus
    from llm.ollama_client import OllamaClient
    from mcp_servers.vision.dino_extractor import DINOv2Extractor
except ImportError:
    from .base import BaseAgent, AgentAction, ToolResult
    from ..state.graph_state import GraphState
    from ..state.models import CandidateRecord, CandidateFeatures, ScoreBreakdown
    from ..config.settings import MaskStatus
    from ..llm.ollama_client import OllamaClient
    from ..mcp_servers.vision.dino_extractor import DINOv2Extractor

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """
    Discovers candidates and scores by cosine similarity to RWTD centroid.
    """

    def __init__(
        self,
        llm_client: OllamaClient,
        dino_extractor: DINOv2Extractor = None,
        device: str = "cuda",
    ):
        super().__init__(
            name="analyst",
            llm_client=llm_client,
            system_prompt="",
        )
        self.device = device
        self._dino_extractor = dino_extractor

    @property
    def dino_extractor(self) -> DINOv2Extractor:
        if self._dino_extractor is None:
            logger.info("Loading DINOv2 extractor...")
            self._dino_extractor = DINOv2Extractor(device=self.device)
        return self._dino_extractor

    def get_available_tools(self) -> List[str]:
        return [
            "discover_candidates",
            "extract_features",
            "score_candidates",
            "get_top_candidates",
            "done",
        ]

    def execute_tool(self, action: AgentAction, state: GraphState) -> ToolResult:
        tool_name = action.tool_name
        params = action.params
        try:
            if tool_name == "discover_candidates":
                return self._discover_candidates(state)
            elif tool_name == "extract_features":
                return self._extract_features(state, params)
            elif tool_name == "score_candidates":
                return self._score_candidates(state)
            elif tool_name == "get_top_candidates":
                n = params.get("n", 10)
                return self._get_top_candidates(state, n)
            elif tool_name == "done":
                return ToolResult(success=True, data={"message": "Analysis complete"})
            else:
                return ToolResult(success=False, error_message=f"Unknown tool: {tool_name}")
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            import traceback
            traceback.print_exc()
            return ToolResult(success=False, error_message=str(e))

    def _discover_candidates(self, state: GraphState) -> ToolResult:
        """Discover all image/mask pairs in source pool."""
        logger.info("Discovering candidates in source pool...")

        source_path = state.config.source_pool_path
        masks_dir = source_path / "masks"
        crops_dir = source_path / "crops"
        images_dir = source_path / "images"
        filter_passed_dir = source_path / "filter" / "passed"

        if crops_dir.exists():
            if state.config.filter_passed:
                if not filter_passed_dir.exists():
                    return ToolResult(
                        success=False,
                        error_message=f"filter/passed directory not found: {filter_passed_dir}"
                    )
                img_root = filter_passed_dir
                logger.info(f"Using filtered-passed crops from: {img_root}")
            else:
                img_root = crops_dir
                logger.info(f"Using all crops from: {img_root}")

            image_paths = sorted(
                list(img_root.glob("**/*.jpg")) +
                list(img_root.glob("**/*.png")) +
                list(img_root.glob("**/*.jpeg"))
            )
        elif images_dir.exists():
            image_paths = sorted(
                list(images_dir.glob("*.jpg")) +
                list(images_dir.glob("*.png")) +
                list(images_dir.glob("*.jpeg"))
            )
        else:
            return ToolResult(
                success=False,
                error_message=f"No images found. Expected 'crops/' or 'images/' in: {source_path}"
            )

        if not masks_dir.exists():
            return ToolResult(
                success=False,
                error_message=f"Masks directory not found: {masks_dir}"
            )

        # Random subsample for quick testing
        max_cand = state.config.max_candidates
        if max_cand > 0 and len(image_paths) > max_cand:
            logger.info(f"Sampling {max_cand} of {len(image_paths)} candidates (max_candidates={max_cand})")
            image_paths = sorted(random.sample(image_paths, max_cand))

        # Build mask lookup
        all_masks = {}
        for mask_path in (
            list(masks_dir.glob("**/*.png")) +
            list(masks_dir.glob("**/*.jpg"))
        ):
            all_masks[mask_path.stem] = mask_path

        discovered = 0
        for img_path in image_paths:
            stem = img_path.stem
            mask_path = all_masks.get(stem)
            if mask_path is None:
                mask_path = all_masks.get(f"{stem}_mask")
            if mask_path:
                candidate_id = stem
                if candidate_id not in state.candidates:
                    state.candidates[candidate_id] = CandidateRecord(
                        id=candidate_id,
                        image_path=img_path,
                        mask_path=mask_path,
                    )
                    discovered += 1

        return ToolResult(
            success=True,
            data={"discovered": discovered, "total_candidates": len(state.candidates)},
        )

    def _extract_features(self, state: GraphState, params: Dict[str, Any]) -> ToolResult:
        """Extract DINOv2 embeddings for candidates."""
        logger.info("Extracting DINOv2 embeddings for candidates...")

        candidates_to_process = [
            c for c in state.candidates.values()
            if c.features is None and c.mask_status != MaskStatus.REJECTED
        ]

        batch_size = params.get("batch_size", len(candidates_to_process))
        candidates_to_process = candidates_to_process[:batch_size]

        if not candidates_to_process:
            return ToolResult(
                success=True,
                data={"message": "All candidates already have features", "processed": 0},
            )

        image_paths = [c.image_path for c in candidates_to_process]
        embeddings, ids, failed = self.dino_extractor.extract_batch(
            image_paths, batch_size=min(16, len(image_paths)),
        )

        embedding_map = {id_: emb for id_, emb in zip(ids, embeddings)}

        processed = 0
        for candidate in candidates_to_process:
            embedding = embedding_map.get(candidate.id)
            if embedding is None:
                continue
            candidate.features = CandidateFeatures(dino_embedding=embedding)
            processed += 1

        return ToolResult(
            success=True,
            data={
                "processed": processed,
                "failed": len(candidates_to_process) - processed,
                "remaining": len([c for c in state.candidates.values() if c.features is None]),
            },
        )

    def _score_candidates(self, state: GraphState) -> ToolResult:
        """Score all candidates by cosine similarity to RWTD centroid."""
        logger.info("Scoring candidates by cosine similarity...")

        if not state.profile_exists:
            return ToolResult(success=False, error_message="RWTD profile not built.")

        centroid = state.rwtd_profile.centroid_embedding

        candidates_to_score = [
            c for c in state.candidates.values()
            if c.features is not None and c.scores is None and c.mask_status != MaskStatus.REJECTED
        ]

        if not candidates_to_score:
            return ToolResult(
                success=True,
                data={"message": "All candidates already scored", "scored": 0},
            )

        scored = 0
        for candidate in candidates_to_score:
            emb = candidate.features.dino_embedding
            if emb is None:
                continue

            sim = self._cosine_similarity(emb, centroid)
            # Normalize from [-1, 1] to [0, 1]
            semantic_score = (sim + 1) / 2

            candidate.scores = ScoreBreakdown(semantic_score=semantic_score)
            scored += 1

        all_scores = [c.scores.total_score for c in state.candidates.values() if c.scores]

        return ToolResult(
            success=True,
            data={
                "scored": scored,
                "total_scored": len(all_scores),
                "score_mean": float(np.mean(all_scores)) if all_scores else 0,
                "score_std": float(np.std(all_scores)) if all_scores else 0,
                "score_max": float(np.max(all_scores)) if all_scores else 0,
                "score_min": float(np.min(all_scores)) if all_scores else 0,
            },
        )

    def _get_top_candidates(self, state: GraphState, n: int = 10) -> ToolResult:
        top = state.get_top_candidates(n)
        return ToolResult(
            success=True,
            data={
                "top_candidates": [
                    {"id": c.id, "score": c.scores.total_score if c.scores else 0}
                    for c in top
                ]
            },
        )

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def run_full_analysis(self, state: GraphState) -> ToolResult:
        """Run the complete analysis pipeline."""
        logger.info("Running full candidate analysis pipeline...")

        result = self._discover_candidates(state)
        if not result.success:
            return result
        logger.info(f"Discovered: {result.data}")

        result = self._extract_features(state, {})
        if not result.success:
            return result
        logger.info(f"Features: {result.data}")

        result = self._score_candidates(state)
        if not result.success:
            return result
        logger.info(f"Scores: {result.data}")

        return result
