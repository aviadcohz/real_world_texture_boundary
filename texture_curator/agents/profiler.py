"""
Profiler Agent for Texture Curator.

Builds the RWTD reference profile: extracts DINOv2 embeddings for all RWTD
images and computes the L2-normalized centroid.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple
import logging

try:
    from agents.base import BaseAgent, AgentAction, ToolResult
    from state.graph_state import GraphState
    from state.models import RWTDProfile
    from llm.ollama_client import OllamaClient
    from mcp_servers.vision.dino_extractor import DINOv2Extractor
except ImportError:
    from .base import BaseAgent, AgentAction, ToolResult
    from ..state.graph_state import GraphState
    from ..state.models import RWTDProfile
    from ..llm.ollama_client import OllamaClient
    from ..mcp_servers.vision.dino_extractor import DINOv2Extractor

logger = logging.getLogger(__name__)


class ProfilerAgent(BaseAgent):
    """
    Builds the RWTD centroid from DINOv2 embeddings.
    """

    def __init__(
        self,
        llm_client: OllamaClient,
        dino_extractor: DINOv2Extractor = None,
        device: str = "cuda",
    ):
        super().__init__(
            name="profiler",
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
        return ["extract_dino_embeddings", "build_profile", "done"]

    def execute_tool(self, action: AgentAction, state: GraphState) -> ToolResult:
        tool_name = action.tool_name
        try:
            if tool_name == "extract_dino_embeddings":
                return self._extract_dino_embeddings(state)
            elif tool_name == "build_profile":
                return self._build_profile(state)
            elif tool_name == "done":
                return ToolResult(success=True, data={"message": "Profiling complete"})
            else:
                return ToolResult(success=False, error_message=f"Unknown tool: {tool_name}")
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return ToolResult(success=False, error_message=str(e))

    def _get_rwtd_image_paths(self, state: GraphState) -> List[Path]:
        """Get image paths from RWTD directory."""
        images_dir = state.config.rwtd_path / "images"
        return sorted(
            list(images_dir.glob("*.jpg")) +
            list(images_dir.glob("*.png")) +
            list(images_dir.glob("*.jpeg"))
        )

    def _extract_dino_embeddings(self, state: GraphState) -> ToolResult:
        """Extract DINOv2 embeddings for all RWTD images."""
        logger.info("Extracting DINOv2 embeddings for RWTD...")

        image_paths = self._get_rwtd_image_paths(state)
        if not image_paths:
            return ToolResult(
                success=False,
                error_message=f"No images found in {state.config.rwtd_path}/images"
            )

        embeddings, ids, failed = self.dino_extractor.extract_batch(
            image_paths, batch_size=state.config.vision.batch_size,
        )

        self._embeddings = embeddings
        self._embedding_ids = ids

        return ToolResult(
            success=True,
            data={
                "num_extracted": len(ids),
                "num_failed": len(failed),
                "embedding_dim": embeddings.shape[1] if len(embeddings) > 0 else 0,
            },
        )

    def _build_profile(self, state: GraphState) -> ToolResult:
        """Build RWTD profile from extracted embeddings."""
        logger.info("Building RWTD profile (centroid)...")

        if self._embeddings is None:
            return ToolResult(success=False, error_message="Embeddings not extracted")

        centroid = self.dino_extractor.compute_centroid(self._embeddings)

        profile = RWTDProfile(
            num_samples=len(self._embedding_ids),
            centroid_embedding=centroid,
        )

        state.rwtd_profile = profile

        # Clear internal state
        num_samples = len(self._embedding_ids)
        self._embeddings = None
        self._embedding_ids = None

        return ToolResult(
            success=True,
            data={
                "num_samples": num_samples,
                "embedding_dim": len(centroid),
            },
        )

    def run_full_profiling(self, state: GraphState) -> ToolResult:
        """Run the complete profiling pipeline."""
        logger.info("Running RWTD profiling pipeline...")

        result = self._extract_dino_embeddings(state)
        if not result.success:
            return result
        logger.info(f"Embeddings: {result.data}")

        result = self._build_profile(state)
        if not result.success:
            return result
        logger.info(f"Profile built: {result.data}")

        return result
