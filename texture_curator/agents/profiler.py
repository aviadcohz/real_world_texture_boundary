"""
Profiler Agent for Texture Curator.

This agent is responsible for building the "Gold Standard" profile
from the RWTD (Real-World Texture Dataset).

ROLE:
- Extract DINOv2 embeddings for all RWTD images
- Compute texture statistics (entropy, GLCM)
- Compute boundary metrics (VoL, edge density)
- Build statistical distributions for each metric
- Create FAISS index for fast similarity search

OUTPUT:
- RWTDProfile object stored in GraphState
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

# Handle imports
try:
    from agents.base import BaseAgent, AgentAction, ToolResult
    from state.graph_state import GraphState
    from state.models import RWTDProfile, Distribution
    from llm.ollama_client import OllamaClient
    from mcp_servers.vision.dino_extractor import DINOv2Extractor
    from mcp_servers.vision.texture_stats import TextureStatsExtractor
    from mcp_servers.vision.boundary_metrics import BoundaryMetricsExtractor
except ImportError:
    from .base import BaseAgent, AgentAction, ToolResult
    from ..state.graph_state import GraphState
    from ..state.models import RWTDProfile, Distribution
    from ..llm.ollama_client import OllamaClient
    from ..mcp_servers.vision.dino_extractor import DINOv2Extractor
    from ..mcp_servers.vision.texture_stats import TextureStatsExtractor
    from ..mcp_servers.vision.boundary_metrics import BoundaryMetricsExtractor

logger = logging.getLogger(__name__)


# ============================================
# Profiler Agent System Prompt
# ============================================

PROFILER_SYSTEM_PROMPT = """You are the PROFILER agent in a texture dataset curation system.

YOUR ROLE:
Build a comprehensive statistical profile of the RWTD (Real-World Texture Dataset).
This profile serves as the "Gold Standard" that candidates will be compared against.

AVAILABLE TOOLS:
- extract_dino_embeddings: Extract semantic embeddings using DINOv2
- compute_texture_stats: Compute entropy and GLCM features
- compute_boundary_metrics: Analyze boundary quality (VoL, gradient, edge density)
- build_profile: Combine all features into final profile
- done: Mark profiling as complete

WORKFLOW:
1. First, extract DINOv2 embeddings for all RWTD images
2. Then, compute texture statistics
3. Then, compute boundary metrics
4. Finally, build the complete profile

RESPONSE FORMAT:
Always respond with valid JSON:
{
    "reasoning": "Your step-by-step thinking about what to do next",
    "tool": "tool_name",
    "params": {}
}
"""


# ============================================
# Profiler Agent
# ============================================

class ProfilerAgent(BaseAgent):
    """
    Agent responsible for building the RWTD reference profile.
    
    This agent orchestrates the feature extraction process and
    builds statistical distributions that define what "good"
    texture transitions look like.
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
        Initialize Profiler agent.
        
        Args:
            llm_client: Ollama client for reasoning
            dino_extractor: Pre-initialized DINOv2 extractor (optional)
            texture_extractor: Pre-initialized texture stats extractor (optional)
            boundary_extractor: Pre-initialized boundary metrics extractor (optional)
            device: Device for vision models
        """
        super().__init__(
            name="profiler",
            llm_client=llm_client,
            system_prompt=PROFILER_SYSTEM_PROMPT,
        )
        
        self.device = device
        
        # Vision extractors (lazy initialization)
        self._dino_extractor = dino_extractor
        self._texture_extractor = texture_extractor
        self._boundary_extractor = boundary_extractor
        
        # Internal state for multi-step profiling
        self._embeddings = None
        self._embedding_ids = None
        self._texture_stats = None
        self._boundary_metrics = None
    
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
            "extract_dino_embeddings",
            "compute_texture_stats",
            "compute_boundary_metrics",
            "build_profile",
            "done",
        ]
    
    def format_state_for_prompt(self, state: GraphState) -> str:
        """Format state with profiling-specific info."""
        lines = [
            f"## Profiling State",
            f"- RWTD Path: {state.config.rwtd_path}",
            f"- Profile Built: {state.profile_exists}",
        ]
        
        # Add internal state
        if self._embeddings is not None:
            lines.append(f"- DINOv2 Embeddings: ✓ ({len(self._embedding_ids)} images)")
        else:
            lines.append(f"- DINOv2 Embeddings: Not extracted")
        
        if self._texture_stats is not None:
            lines.append(f"- Texture Stats: ✓ ({len(self._texture_stats)} images)")
        else:
            lines.append(f"- Texture Stats: Not computed")
        
        if self._boundary_metrics is not None:
            lines.append(f"- Boundary Metrics: ✓ ({len(self._boundary_metrics)} images)")
        else:
            lines.append(f"- Boundary Metrics: Not computed")
        
        return "\n".join(lines)
    
    def execute_tool(self, action: AgentAction, state: GraphState) -> ToolResult:
        """
        Execute a profiling tool.
        
        Args:
            action: The action to execute
            state: Current graph state
        
        Returns:
            ToolResult with success status and data
        """
        tool_name = action.tool_name
        
        try:
            if tool_name == "extract_dino_embeddings":
                return self._extract_dino_embeddings(state)
            
            elif tool_name == "compute_texture_stats":
                return self._compute_texture_stats(state)
            
            elif tool_name == "compute_boundary_metrics":
                return self._compute_boundary_metrics(state)
            
            elif tool_name == "build_profile":
                return self._build_profile(state)
            
            elif tool_name == "done":
                return ToolResult(success=True, data={"message": "Profiling complete"})
            
            else:
                return ToolResult(success=False, error_message=f"Unknown tool: {tool_name}")
                
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return ToolResult(success=False, error_message=str(e))
    
    def _get_rwtd_paths(self, state: GraphState) -> Tuple[List[Path], List[Path]]:
        """Get image and mask paths from RWTD directory."""
        rwtd_path = state.config.rwtd_path
        images_dir = rwtd_path / "images"
        masks_dir = rwtd_path / "masks"
        
        # Find all images
        image_paths = sorted(
            list(images_dir.glob("*.jpg")) + 
            list(images_dir.glob("*.png")) +
            list(images_dir.glob("*.jpeg"))
        )
        
        # Find corresponding masks
        mask_paths = []
        valid_image_paths = []
        
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
                valid_image_paths.append(img_path)
                mask_paths.append(mask_path)
        
        return valid_image_paths, mask_paths
    
    def _extract_dino_embeddings(self, state: GraphState) -> ToolResult:
        """Extract DINOv2 embeddings for all RWTD images."""
        logger.info("Extracting DINOv2 embeddings...")
        
        image_paths, _ = self._get_rwtd_paths(state)
        
        if not image_paths:
            return ToolResult(
                success=False,
                error_message=f"No images found in {state.config.rwtd_path}/images"
            )
        
        # Extract embeddings
        embeddings, ids, failed = self.dino_extractor.extract_batch(
            image_paths,
            batch_size=state.config.vision.batch_size,
        )
        
        # Store internally
        self._embeddings = embeddings
        self._embedding_ids = ids
        
        return ToolResult(
            success=True,
            data={
                "num_extracted": len(ids),
                "num_failed": len(failed),
                "embedding_dim": embeddings.shape[1] if len(embeddings) > 0 else 0,
            }
        )
    
    def _compute_texture_stats(self, state: GraphState) -> ToolResult:
        """Compute texture statistics for all RWTD images."""
        logger.info("Computing texture statistics...")
        
        image_paths, _ = self._get_rwtd_paths(state)
        
        if not image_paths:
            return ToolResult(
                success=False,
                error_message=f"No images found in {state.config.rwtd_path}/images"
            )
        
        # Compute stats
        stats_list = self.texture_extractor.compute_batch(image_paths)
        
        # Store internally
        self._texture_stats = stats_list
        
        # Compute summary
        successful = [s for s in stats_list if s.success]
        
        return ToolResult(
            success=True,
            data={
                "num_computed": len(successful),
                "num_failed": len(stats_list) - len(successful),
                "mean_entropy": np.mean([s.entropy_mean for s in successful]) if successful else 0,
            }
        )
    
    def _compute_boundary_metrics(self, state: GraphState) -> ToolResult:
        """Compute boundary metrics for all RWTD image/mask pairs."""
        logger.info("Computing boundary metrics...")
        
        image_paths, mask_paths = self._get_rwtd_paths(state)
        
        if not image_paths:
            return ToolResult(
                success=False,
                error_message=f"No image/mask pairs found in {state.config.rwtd_path}"
            )
        
        # Compute metrics
        metrics_list = self.boundary_extractor.compute_batch(image_paths, mask_paths)
        
        # Store internally
        self._boundary_metrics = metrics_list
        
        # Compute summary
        successful = [m for m in metrics_list if m.success]
        
        return ToolResult(
            success=True,
            data={
                "num_computed": len(successful),
                "num_failed": len(metrics_list) - len(successful),
                "mean_vol": np.mean([m.variance_of_laplacian for m in successful]) if successful else 0,
                "mean_edge_density": np.mean([m.edge_density for m in successful]) if successful else 0,
            }
        )
    
    def _build_profile(self, state: GraphState) -> ToolResult:
        """Build the final RWTD profile from extracted features."""
        logger.info("Building RWTD profile...")
        
        # Check we have all components
        if self._embeddings is None:
            return ToolResult(success=False, error_message="DINOv2 embeddings not extracted")
        
        if self._texture_stats is None:
            return ToolResult(success=False, error_message="Texture stats not computed")
        
        if self._boundary_metrics is None:
            return ToolResult(success=False, error_message="Boundary metrics not computed")
        
        # Compute centroid
        centroid = self.dino_extractor.compute_centroid(self._embeddings)
        
        # Build entropy distribution
        successful_texture = [s for s in self._texture_stats if s.success]
        entropy_values = np.array([s.entropy_mean for s in successful_texture])
        entropy_dist = Distribution.from_values(entropy_values)
        
        # Build GLCM distributions
        glcm_distributions = {
            "contrast": Distribution.from_values(np.array([s.glcm_contrast for s in successful_texture])),
            "homogeneity": Distribution.from_values(np.array([s.glcm_homogeneity for s in successful_texture])),
            "energy": Distribution.from_values(np.array([s.glcm_energy for s in successful_texture])),
            "correlation": Distribution.from_values(np.array([s.glcm_correlation for s in successful_texture])),
        }
        
        # Build boundary distributions
        successful_boundary = [m for m in self._boundary_metrics if m.success]
        vol_values = np.array([m.variance_of_laplacian for m in successful_boundary])
        vol_dist = Distribution.from_values(vol_values)
        
        edge_density_values = np.array([m.edge_density for m in successful_boundary])
        edge_density_dist = Distribution.from_values(edge_density_values)
        
        # Create profile
        profile = RWTDProfile(
            num_samples=len(self._embedding_ids),
            centroid_embedding=centroid,
            entropy_distribution=entropy_dist,
            glcm_distributions=glcm_distributions,
            boundary_sharpness_distribution=vol_dist,
            edge_density_distribution=edge_density_dist,
        )
        
        # Store in state
        state.rwtd_profile = profile
        
        # Clear internal state
        self._embeddings = None
        self._embedding_ids = None
        self._texture_stats = None
        self._boundary_metrics = None
        
        return ToolResult(
            success=True,
            data={
                "num_samples": profile.num_samples,
                "embedding_dim": len(centroid),
                "entropy_mean": entropy_dist.mean,
                "entropy_std": entropy_dist.std,
                "vol_mean": vol_dist.mean,
                "vol_std": vol_dist.std,
                "edge_density_mean": edge_density_dist.mean,
            }
        )
    
    def run_full_profiling(self, state: GraphState) -> ToolResult:
        """
        Run the complete profiling pipeline without LLM.
        
        This is a convenience method for direct execution.
        """
        logger.info("Running full RWTD profiling pipeline...")
        
        # Step 1: Extract embeddings
        result = self._extract_dino_embeddings(state)
        if not result.success:
            return result
        logger.info(f"Embeddings: {result.data}")
        
        # Step 2: Compute texture stats
        result = self._compute_texture_stats(state)
        if not result.success:
            return result
        logger.info(f"Texture stats: {result.data}")
        
        # Step 3: Compute boundary metrics
        result = self._compute_boundary_metrics(state)
        if not result.success:
            return result
        logger.info(f"Boundary metrics: {result.data}")
        
        # Step 4: Build profile
        result = self._build_profile(state)
        if not result.success:
            return result
        logger.info(f"Profile built: {result.data}")
        
        return result


# ============================================
# Quick Test
# ============================================

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Profiler Agent Test")
    print("=" * 60)
    
    # Check for RWTD path
    rwtd_path = Path("/home/aviad/RWTD")
    if not rwtd_path.exists():
        print(f"✗ RWTD not found at {rwtd_path}")
        sys.exit(1)
    
    from llm.ollama_client import OllamaClient
    from state.graph_state import GraphState
    from config.settings import Config
    
    # Create client and agent
    client = OllamaClient(model="qwen2.5:7b")
    agent = ProfilerAgent(llm_client=client, device="cuda")
    
    print(f"✓ Agent created: {agent.name}")
    print(f"  Tools: {agent.get_available_tools()}")
    
    # Create state
    config = Config(rwtd_path=rwtd_path)
    state = GraphState(config=config)
    
    print(f"\n--- Running Full Profiling ---")
    result = agent.run_full_profiling(state)
    
    if result.success:
        print(f"\n✓ Profiling complete!")
        print(f"  Profile: {state.rwtd_profile.num_samples} samples")
        print(f"  Centroid norm: {np.linalg.norm(state.rwtd_profile.centroid_embedding):.3f}")
    else:
        print(f"\n✗ Profiling failed: {result.error_message}")