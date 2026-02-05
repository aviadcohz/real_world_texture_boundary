"""
Analyst Agent for Texture Curator.

This agent is responsible for scoring candidates from the source pool
against the RWTD profile.

ROLE:
- Load candidates from source pool
- Extract features (DINOv2, texture, boundary)
- Score each candidate against RWTD profile
- Rank candidates by total score

SCORING:
- Semantic score: Cosine similarity to RWTD centroid
- Texture score: How well stats fit RWTD distributions
- Boundary score: Boundary quality metrics

OUTPUT:
- Populated candidates dict in GraphState with scores
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Handle imports
try:
    from agents.base import BaseAgent, AgentAction, ToolResult
    from state.graph_state import GraphState
    from state.models import CandidateRecord, CandidateFeatures, ScoreBreakdown
    from config.settings import MaskStatus
    from llm.ollama_client import OllamaClient
    from mcp_servers.vision.dino_extractor import DINOv2Extractor
    from mcp_servers.vision.texture_stats import TextureStatsExtractor
    from mcp_servers.vision.boundary_metrics import BoundaryMetricsExtractor
except ImportError:
    from .base import BaseAgent, AgentAction, ToolResult
    from ..state.graph_state import GraphState
    from ..state.models import CandidateRecord, CandidateFeatures, ScoreBreakdown
    from ..config.settings import MaskStatus
    from ..llm.ollama_client import OllamaClient
    from ..mcp_servers.vision.dino_extractor import DINOv2Extractor
    from ..mcp_servers.vision.texture_stats import TextureStatsExtractor
    from ..mcp_servers.vision.boundary_metrics import BoundaryMetricsExtractor

logger = logging.getLogger(__name__)


# ============================================
# Analyst Agent System Prompt
# ============================================

ANALYST_SYSTEM_PROMPT = """You are the ANALYST agent in a texture dataset curation system.

YOUR ROLE:
Score candidates from the source pool against the RWTD profile.
Each candidate gets a score based on how well it matches the "Gold Standard".

AVAILABLE TOOLS:
- discover_candidates: Find all image/mask pairs in source pool
- extract_features: Extract DINOv2, texture, and boundary features
- score_candidates: Compute scores against RWTD profile
- get_top_candidates: Get the highest-scoring candidates
- done: Mark analysis as complete

SCORING FORMULA:
total_score = w1 * semantic_score + w2 * texture_score + w3 * boundary_score

Where:
- semantic_score = cosine_similarity(candidate_embedding, rwtd_centroid)
- texture_score = how well entropy/GLCM fit RWTD distributions
- boundary_score = how well boundary metrics fit RWTD distributions

WORKFLOW:
1. First, discover candidates in the source pool
2. Extract features for all candidates
3. Score candidates against the profile
4. Report top candidates

RESPONSE FORMAT:
Always respond with valid JSON:
{
    "reasoning": "Your step-by-step thinking",
    "tool": "tool_name",
    "params": {}
}
"""


# ============================================
# Analyst Agent
# ============================================

class AnalystAgent(BaseAgent):
    """
    Agent responsible for scoring candidates against RWTD profile.
    """
    
    def __init__(
        self,
        llm_client: OllamaClient,
        dino_extractor: DINOv2Extractor = None,
        texture_extractor: TextureStatsExtractor = None,
        boundary_extractor: BoundaryMetricsExtractor = None,
        device: str = "cuda",
    ):
        """
        Initialize Analyst agent.
        
        Args:
            llm_client: Ollama client for reasoning
            dino_extractor: Pre-initialized DINOv2 extractor (optional)
            texture_extractor: Pre-initialized texture stats extractor (optional)
            boundary_extractor: Pre-initialized boundary metrics extractor (optional)
            device: Device for vision models
        """
        super().__init__(
            name="analyst",
            llm_client=llm_client,
            system_prompt=ANALYST_SYSTEM_PROMPT,
        )
        
        self.device = device
        
        # Vision extractors (can be shared with ProfilerAgent)
        self._dino_extractor = dino_extractor
        self._texture_extractor = texture_extractor
        self._boundary_extractor = boundary_extractor
    
    @property
    def dino_extractor(self) -> DINOv2Extractor:
        """Lazy-load DINOv2 extractor."""
        if self._dino_extractor is None:
            logger.info("Loading DINOv2 extractor...")
            self._dino_extractor = DINOv2Extractor(device=self.device)
        return self._dino_extractor
    
    @property
    def texture_extractor(self) -> TextureStatsExtractor:
        """Lazy-load texture stats extractor."""
        if self._texture_extractor is None:
            self._texture_extractor = TextureStatsExtractor()
        return self._texture_extractor
    
    @property
    def boundary_extractor(self) -> BoundaryMetricsExtractor:
        """Lazy-load boundary metrics extractor."""
        if self._boundary_extractor is None:
            self._boundary_extractor = BoundaryMetricsExtractor()
        return self._boundary_extractor
    
    def get_available_tools(self) -> List[str]:
        return [
            "discover_candidates",
            "extract_features",
            "score_candidates",
            "get_top_candidates",
            "done",
        ]
    
    def format_state_for_prompt(self, state: GraphState) -> str:
        """Format state with analyst-specific info."""
        lines = [
            f"## Analysis State",
            f"- Source Pool: {state.config.source_pool_path}",
            f"- Profile Built: {state.profile_exists}",
            f"- Candidates Discovered: {state.num_candidates}",
            f"- Candidates Scored: {state.num_scored}",
        ]
        
        if state.num_scored > 0:
            top = state.get_top_candidates(5)
            lines.append(f"\nTop 5 Candidates:")
            for c in top:
                if c.scores:
                    lines.append(f"  - {c.id}: {c.scores.total_score:.3f}")
        
        return "\n".join(lines)
    
    def execute_tool(self, action: AgentAction, state: GraphState) -> ToolResult:
        """Execute an analyst tool."""
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
        images_dir = source_path / "images"
        masks_dir = source_path / "masks"
        
        if not images_dir.exists():
            return ToolResult(
                success=False,
                error_message=f"Images directory not found: {images_dir}"
            )
        
        # Find all images
        image_paths = sorted(
            list(images_dir.glob("*.jpg")) + 
            list(images_dir.glob("*.png")) +
            list(images_dir.glob("*.jpeg"))
        )
        
        # Find corresponding masks and create candidates
        discovered = 0
        for img_path in image_paths:
            # Try different mask naming conventions
            mask_candidates = [
                masks_dir / f"{img_path.stem}.png",
                masks_dir / f"{img_path.stem}_mask.png",
                masks_dir / f"{img_path.stem}.jpg",
            ]
            
            mask_path = None
            for candidate in mask_candidates:
                if candidate.exists():
                    mask_path = candidate
                    break
            
            if mask_path:
                candidate_id = img_path.stem
                
                # Only add if not already present
                if candidate_id not in state.candidates:
                    state.candidates[candidate_id] = CandidateRecord(
                        id=candidate_id,
                        image_path=img_path,
                        mask_path=mask_path,
                    )
                    discovered += 1
        
        return ToolResult(
            success=True,
            data={
                "discovered": discovered,
                "total_candidates": len(state.candidates),
            }
        )
    
    def _extract_features(
        self,
        state: GraphState,
        params: Dict[str, Any],
    ) -> ToolResult:
        """Extract features for candidates."""
        logger.info("Extracting features for candidates...")
        
        # Get candidates without features
        candidates_to_process = [
            c for c in state.candidates.values()
            if c.features is None
        ]
        
        # Limit batch size if specified
        batch_size = params.get("batch_size", len(candidates_to_process))
        candidates_to_process = candidates_to_process[:batch_size]
        
        if not candidates_to_process:
            return ToolResult(
                success=True,
                data={"message": "All candidates already have features", "processed": 0}
            )
        
        # Extract DINOv2 embeddings
        image_paths = [c.image_path for c in candidates_to_process]
        embeddings, ids, failed = self.dino_extractor.extract_batch(
            image_paths,
            batch_size=min(16, len(image_paths)),
        )
        
        # Create ID to embedding mapping
        embedding_map = {id_: emb for id_, emb in zip(ids, embeddings)}
        
        # Extract texture stats
        texture_stats = self.texture_extractor.compute_batch(image_paths, show_progress=False)
        texture_map = {s.image_id: s for s in texture_stats if s.success}
        
        # Extract boundary metrics
        mask_paths = [c.mask_path for c in candidates_to_process]
        boundary_metrics = self.boundary_extractor.compute_batch(
            image_paths, mask_paths, show_progress=False
        )
        boundary_map = {m.image_id: m for m in boundary_metrics if m.success}
        
        # Update candidates
        processed = 0
        for candidate in candidates_to_process:
            cid = candidate.id
            
            # Get embedding
            embedding = embedding_map.get(cid)
            if embedding is None:
                continue
            
            # Get texture stats
            tex = texture_map.get(cid)
            
            # Get boundary metrics
            bound = boundary_map.get(cid)
            
            # Create features
            candidate.features = CandidateFeatures(
                dino_embedding=embedding,
                entropy=tex.entropy_mean if tex else 0,
                glcm_contrast=tex.glcm_contrast if tex else 0,
                glcm_homogeneity=tex.glcm_homogeneity if tex else 0,
                glcm_energy=tex.glcm_energy if tex else 0,
                glcm_correlation=tex.glcm_correlation if tex else 0,
                boundary_sharpness=bound.variance_of_laplacian if bound else 0,
                edge_density=bound.edge_density if bound else 0,
                gradient_magnitude_mean=bound.gradient_magnitude_mean if bound else 0,
                gradient_magnitude_std=bound.gradient_magnitude_std if bound else 0,
            )
            processed += 1
        
        return ToolResult(
            success=True,
            data={
                "processed": processed,
                "failed": len(candidates_to_process) - processed,
                "remaining": len([c for c in state.candidates.values() if c.features is None]),
            }
        )
    
    def _score_candidates(self, state: GraphState) -> ToolResult:
        """Score all candidates against RWTD profile."""
        logger.info("Scoring candidates...")
        
        if not state.profile_exists:
            return ToolResult(
                success=False,
                error_message="RWTD profile not built. Run profiler first."
            )
        
        profile = state.rwtd_profile
        thresholds = state.config.thresholds
        
        # Get candidates with features but no scores
        candidates_to_score = [
            c for c in state.candidates.values()
            if c.features is not None and c.scores is None
        ]
        
        if not candidates_to_score:
            return ToolResult(
                success=True,
                data={"message": "All candidates already scored", "scored": 0}
            )
        
        scored = 0
        for candidate in candidates_to_score:
            features = candidate.features
            
            # 1. Semantic score (cosine similarity to centroid)
            if features.dino_embedding is not None and profile.centroid_embedding is not None:
                semantic_score = self._cosine_similarity(
                    features.dino_embedding,
                    profile.centroid_embedding
                )
                # Normalize to 0-1 range (cosine is -1 to 1)
                semantic_score = (semantic_score + 1) / 2
            else:
                semantic_score = 0.5
            
            # 2. Texture score (how well fits RWTD distributions)
            texture_scores = []
            
            if profile.entropy_distribution:
                texture_scores.append(profile.entropy_distribution.score_value(features.entropy))
            
            for metric_name in ["contrast", "homogeneity", "energy", "correlation"]:
                if metric_name in profile.glcm_distributions:
                    dist = profile.glcm_distributions[metric_name]
                    value = getattr(features, f"glcm_{metric_name}", 0)
                    texture_scores.append(dist.score_value(value))
            
            texture_score = np.mean(texture_scores) if texture_scores else 0.5
            
            # 3. Boundary score
            boundary_scores = []
            
            if profile.boundary_sharpness_distribution:
                boundary_scores.append(
                    profile.boundary_sharpness_distribution.score_value(features.boundary_sharpness)
                )
            
            if profile.edge_density_distribution:
                boundary_scores.append(
                    profile.edge_density_distribution.score_value(features.edge_density)
                )
            
            boundary_score = np.mean(boundary_scores) if boundary_scores else 0.5
            
            # Compute total score
            total_score = (
                thresholds.weight_semantic * semantic_score +
                thresholds.weight_texture * texture_score +
                thresholds.weight_boundary * boundary_score
            )
            
            # Store scores
            candidate.scores = ScoreBreakdown(
                semantic_score=semantic_score,
                texture_score=texture_score,
                boundary_score=boundary_score,
                total_score=total_score,
            )
            scored += 1
        
        # Compute summary statistics
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
            }
        )
    
    def _get_top_candidates(self, state: GraphState, n: int = 10) -> ToolResult:
        """Get top N candidates by score."""
        top = state.get_top_candidates(n)
        
        return ToolResult(
            success=True,
            data={
                "top_candidates": [
                    {
                        "id": c.id,
                        "total_score": c.scores.total_score if c.scores else 0,
                        "semantic": c.scores.semantic_score if c.scores else 0,
                        "texture": c.scores.texture_score if c.scores else 0,
                        "boundary": c.scores.boundary_score if c.scores else 0,
                    }
                    for c in top
                ]
            }
        )
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def run_full_analysis(self, state: GraphState) -> ToolResult:
        """
        Run the complete analysis pipeline without LLM.
        
        This is a convenience method for direct execution.
        """
        logger.info("Running full candidate analysis pipeline...")
        
        # Step 1: Discover candidates
        result = self._discover_candidates(state)
        if not result.success:
            return result
        logger.info(f"Discovered: {result.data}")
        
        # Step 2: Extract features
        result = self._extract_features(state, {})
        if not result.success:
            return result
        logger.info(f"Features: {result.data}")
        
        # Step 3: Score candidates
        result = self._score_candidates(state)
        if not result.success:
            return result
        logger.info(f"Scores: {result.data}")
        
        # Step 4: Get top candidates
        result = self._get_top_candidates(state, state.config.target_n * 2)
        logger.info(f"Top candidates: {result.data}")
        
        return result


# ============================================
# Quick Test
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Analyst Agent Test")
    print("=" * 60)
    print("\nNote: This test requires a pre-built RWTD profile.")
    print("Run ProfilerAgent first to build the profile.")