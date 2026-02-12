"""
Graph Orchestrator for Texture Curator.

Simplified linear workflow:
  1. Profiler  → RWTD centroid
  2. Analyst   → Discover + score candidates (cosine similarity)
  3. MaskFilter → VLM filter
  4. Analyst   → Score mask-passed candidates
  5. Optimizer → Select diverse top-N
"""

import time
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

try:
    from config.settings import Config, Phase
    from state.graph_state import GraphState
    from llm.ollama_client import OllamaClient
    from agents.planner import PlannerAgent
    from agents.profiler import ProfilerAgent
    from agents.analyst import AnalystAgent
    from agents.optimizer import OptimizerAgent
    from agents.mask_filter import MaskFilterAgent
except ImportError:
    from .config.settings import Config, Phase
    from .state.graph_state import GraphState
    from .llm.ollama_client import OllamaClient
    from .agents.planner import PlannerAgent
    from .agents.profiler import ProfilerAgent
    from .agents.analyst import AnalystAgent
    from .agents.optimizer import OptimizerAgent
    from .agents.mask_filter import MaskFilterAgent


class TextureCuratorOrchestrator:
    """
    Orchestrator for the multi-agent texture curation system.

    Flow: profiler → analyst → mask_filter → analyst → optimizer → done
    """

    def __init__(
        self,
        config: Config,
        llm_model: str = "qwen2.5:7b",
        device: str = "cuda",
        save_checkpoints: bool = True,
    ):
        self.config = config
        self.device = device
        self.save_checkpoints = save_checkpoints

        logger.info(f"Initializing LLM client with model: {llm_model}")
        self.llm_client = OllamaClient(model=llm_model)

        if not self.llm_client.is_available():
            raise RuntimeError("Ollama is not running. Start with: ollama serve")

        self.state = GraphState(config=config)

        logger.info("Initializing agents...")
        self.planner = PlannerAgent(llm_client=self.llm_client)
        self.profiler = ProfilerAgent(llm_client=self.llm_client, device=device)
        self.analyst = AnalystAgent(llm_client=self.llm_client, device=device)
        self.optimizer = OptimizerAgent(llm_client=self.llm_client)
        self.mask_filter = MaskFilterAgent(
            llm_client=self.llm_client,
            vlm_model=config.mask_filter.vlm_model,
            device=device,
        )

        self.agents = {
            "profiler": self.profiler,
            "analyst": self.analyst,
            "mask_filter": self.mask_filter,
            "optimizer": self.optimizer,
        }

        self.step_count = 0
        self.start_time = None
        logger.info("Orchestrator initialized successfully")

    def run(self, max_steps: int = 20) -> GraphState:
        """Run the full curation workflow."""
        logger.info("=" * 60)
        logger.info("TEXTURE CURATOR - Starting Workflow")
        logger.info("=" * 60)
        logger.info(f"RWTD Path: {self.config.rwtd_path}")
        logger.info(f"Source Pool: {self.config.source_pool_path}")
        logger.info(f"Target Selection: {self.config.target_n}")
        logger.info("=" * 60)

        self.start_time = time.time()

        while self.step_count < max_steps:
            self.step_count += 1
            next_agent = self.planner.get_next_agent(self.state)

            logger.info(f"\n[Step {self.step_count}] Phase: {self.state.current_phase.value} → Agent: {next_agent}")

            if next_agent == "done":
                logger.info("Workflow complete!")
                break

            if next_agent not in self.agents:
                logger.error(f"Unknown agent: {next_agent}")
                break

            try:
                result = self._execute_agent(next_agent)
                if result.success:
                    logger.info(f"  ✓ {next_agent} completed: {self._summarize_result(result)}")
                else:
                    logger.warning(f"  ✗ {next_agent} failed: {result.error_message}")
            except Exception as e:
                logger.error(f"  ✗ {next_agent} error: {e}")
                import traceback
                traceback.print_exc()
                self.state.last_error = str(e)

            if self.save_checkpoints:
                self._save_checkpoint()

            self._log_progress()

        self._log_final_summary()
        return self.state

    def _execute_agent(self, agent_name: str):
        if agent_name == "profiler":
            return self.profiler.run_full_profiling(self.state)
        elif agent_name == "mask_filter":
            return self.mask_filter.run_full_filtering(self.state)
        elif agent_name == "analyst":
            return self.analyst.run_full_analysis(self.state)
        elif agent_name == "optimizer":
            return self.optimizer.run_full_optimization(self.state)
        else:
            from agents.base import ToolResult
            return ToolResult(success=False, error_message=f"Unknown agent: {agent_name}")

    def _summarize_result(self, result) -> str:
        if not result.data:
            return "No data"
        keys_to_show = ["num_samples", "discovered", "processed", "scored",
                        "selected_count", "exported"]
        parts = []
        for key in keys_to_show:
            if key in result.data:
                value = result.data[key]
                if isinstance(value, float):
                    parts.append(f"{key}={value:.2f}")
                else:
                    parts.append(f"{key}={value}")
        return ", ".join(parts) if parts else str(result.data)[:100]

    def _save_checkpoint(self):
        checkpoint_name = f"step_{self.step_count:03d}_{self.state.current_phase.value}"
        self.state.save_checkpoint(self.config.checkpoint_path, checkpoint_name)

    def _log_progress(self):
        elapsed = time.time() - self.start_time
        logger.info(
            f"  Progress: candidates={self.state.num_candidates}, "
            f"mask_passed={self.state.num_mask_passed}, "
            f"scored={self.state.num_scored}, "
            f"selected={self.state.num_selected}/{self.config.target_n} "
            f"[{elapsed:.1f}s elapsed]"
        )

    def _log_final_summary(self):
        elapsed = time.time() - self.start_time
        logger.info("\n" + "=" * 60)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Steps: {self.step_count}")
        logger.info(f"Total Time: {elapsed:.1f}s")
        logger.info(f"Candidates Discovered: {self.state.num_candidates}")
        logger.info(f"Mask Passed: {self.state.num_mask_passed}")
        logger.info(f"Candidates Scored: {self.state.num_scored}")
        logger.info(f"Final Selection: {self.state.num_selected}/{self.config.target_n}")

        if self.state.selection_report:
            logger.info(f"Diversity Score: {self.state.selection_report.diversity_score:.3f}")
            logger.info(f"Mean Quality: {self.state.selection_report.mean_quality_score:.3f}")

        logger.info("=" * 60)


def run_texture_curator(
    rwtd_path: str,
    source_pool_path: str,
    target_n: int = 10,
    output_path: str = "./outputs",
    llm_model: str = "qwen2.5:7b",
    device: str = "cuda",
    **kwargs
) -> GraphState:
    """Convenience function to run the texture curator."""
    config = Config(
        rwtd_path=Path(rwtd_path),
        source_pool_path=Path(source_pool_path),
        target_n=target_n,
        output_path=Path(output_path),
        device=device,
        **kwargs
    )
    orchestrator = TextureCuratorOrchestrator(
        config=config, llm_model=llm_model, device=device,
    )
    return orchestrator.run()
