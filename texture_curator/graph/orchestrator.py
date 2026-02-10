"""
Graph Orchestrator for Texture Curator.

This is the main entry point that runs the multi-agent workflow.
It coordinates between all agents and manages the execution flow.

WORKFLOW:
1. Initialize state and agents
2. Loop:
   a. Planner decides next agent
   b. Execute agent's task
   c. Update state
   d. Save checkpoint
   e. Check if done
3. Export results

FEATURES:
- Automatic agent routing
- Checkpoint saving for debugging
- Error handling and recovery
- Progress logging
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Handle imports
try:
    from config.settings import Config, Phase
    from state.graph_state import GraphState, create_initial_state
    from llm.ollama_client import OllamaClient
    from agents.planner import PlannerAgent
    from agents.profiler import ProfilerAgent
    from agents.analyst import AnalystAgent
    from agents.critic import CriticAgent
    from agents.optimizer import OptimizerAgent
    from agents.mask_filter import MaskFilterAgent
except ImportError:
    from .config.settings import Config, Phase
    from .state.graph_state import GraphState, create_initial_state
    from .llm.ollama_client import OllamaClient
    from .agents.planner import PlannerAgent
    from .agents.profiler import ProfilerAgent
    from .agents.analyst import AnalystAgent
    from .agents.critic import CriticAgent
    from .agents.optimizer import OptimizerAgent
    from .agents.mask_filter import MaskFilterAgent


class TextureCuratorOrchestrator:
    """
    Main orchestrator for the multi-agent texture curation system.
    
    This class manages the entire workflow:
    1. Initializes all agents
    2. Routes between agents based on planner decisions
    3. Executes agent tasks
    4. Manages state and checkpoints
    """
    
    def __init__(
        self,
        config: Config,
        llm_model: str = "qwen2.5:7b",
        device: str = "cuda",
        save_checkpoints: bool = True,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            config: System configuration
            llm_model: Model name for Ollama
            device: Device for vision models
            save_checkpoints: Whether to save checkpoints after each step
        """
        self.config = config
        self.device = device
        self.save_checkpoints = save_checkpoints
        
        # Initialize LLM client
        logger.info(f"Initializing LLM client with model: {llm_model}")
        self.llm_client = OllamaClient(model=llm_model)
        
        if not self.llm_client.is_available():
            raise RuntimeError("Ollama is not running. Start with: ollama serve")
        
        # Initialize state
        self.state = GraphState(config=config)
        
        # Initialize agents (lazy loading for vision models)
        logger.info("Initializing agents...")
        self.planner = PlannerAgent(llm_client=self.llm_client)
        self.profiler = ProfilerAgent(llm_client=self.llm_client, device=device)
        self.analyst = AnalystAgent(llm_client=self.llm_client, device=device)
        self.critic = CriticAgent(llm_client=self.llm_client, device=device)
        self.optimizer = OptimizerAgent(llm_client=self.llm_client)
        self.mask_filter = MaskFilterAgent(
            llm_client=self.llm_client,
            vlm_model=config.mask_filter.vlm_model,
            device=device,
        )

        # Agent registry
        self.agents = {
            "profiler": self.profiler,
            "analyst": self.analyst,
            "mask_filter": self.mask_filter,
            "critic": self.critic,
            "optimizer": self.optimizer,
        }
        
        # Execution tracking
        self.step_count = 0
        self.start_time = None
        
        logger.info("Orchestrator initialized successfully")
    
    def run(self, max_steps: int = 50) -> GraphState:
        """
        Run the full curation workflow.
        
        Args:
            max_steps: Maximum number of steps to prevent infinite loops
        
        Returns:
            Final GraphState with results
        """
        logger.info("=" * 60)
        logger.info("TEXTURE CURATOR - Starting Multi-Agent Workflow")
        logger.info("=" * 60)
        logger.info(f"RWTD Path: {self.config.rwtd_path}")
        logger.info(f"Source Pool: {self.config.source_pool_path}")
        logger.info(f"Target Selection: {self.config.target_n}")
        logger.info("=" * 60)
        
        self.start_time = time.time()
        
        while self.step_count < max_steps:
            self.step_count += 1
            
            # Get routing decision from planner
            next_agent = self.planner.get_next_agent(self.state)
            
            logger.info(f"\n[Step {self.step_count}] Phase: {self.state.current_phase.value} → Agent: {next_agent}")
            
            # Check if done
            if next_agent == "done":
                logger.info("Workflow complete!")
                break
            
            # Get and execute agent
            if next_agent not in self.agents:
                logger.error(f"Unknown agent: {next_agent}")
                break
            
            agent = self.agents[next_agent]
            
            # Execute agent's full pipeline
            try:
                result = self._execute_agent(agent, next_agent)
                
                if result.success:
                    logger.info(f"  ✓ {next_agent} completed: {self._summarize_result(result)}")
                else:
                    logger.warning(f"  ✗ {next_agent} failed: {result.error_message}")
                    
            except Exception as e:
                logger.error(f"  ✗ {next_agent} error: {e}")
                import traceback
                traceback.print_exc()
                self.state.last_error = str(e)
            
            # Save checkpoint
            if self.save_checkpoints:
                self._save_checkpoint()
            
            # Log progress
            self._log_progress()
        
        # Final summary
        self._log_final_summary()
        
        return self.state
    
    def _execute_agent(self, agent, agent_name: str):
        """Execute an agent's full pipeline."""
        
        if agent_name == "profiler":
            return self.profiler.run_full_profiling(self.state)

        elif agent_name == "mask_filter":
            return self.mask_filter.run_full_filtering(self.state)

        elif agent_name == "analyst":
            return self.analyst.run_full_analysis(self.state)
        
        elif agent_name == "critic":
            return self.critic.run_full_critique(self.state)
        
        elif agent_name == "optimizer":
            return self.optimizer.run_full_optimization(self.state)
        
        else:
            from agents.base import ToolResult
            return ToolResult(success=False, error_message=f"Unknown agent: {agent_name}")
    
    def _summarize_result(self, result) -> str:
        """Create a short summary of a tool result."""
        if not result.data:
            return "No data"
        
        # Pick key metrics to show
        keys_to_show = ["num_samples", "discovered", "processed", "scored", 
                       "reviewed", "selected_count", "quality_score", "exported"]
        
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
        """Save current state to checkpoint."""
        checkpoint_name = f"step_{self.step_count:03d}_{self.state.current_phase.value}"
        checkpoint_file = self.state.save_checkpoint(
            self.config.checkpoint_path,
            checkpoint_name
        )
        logger.debug(f"Checkpoint saved: {checkpoint_file}")
    
    def _log_progress(self):
        """Log current progress."""
        elapsed = time.time() - self.start_time
        logger.info(f"  Progress: candidates={self.state.num_candidates}, "
                   f"mask_passed={self.state.num_mask_passed}, "
                   f"scored={self.state.num_scored}, "
                   f"validated={self.state.num_validated}, "
                   f"selected={self.state.num_selected}/{self.config.target_n} "
                   f"[{elapsed:.1f}s elapsed]")
    
    def _log_final_summary(self):
        """Log final summary of the run."""
        elapsed = time.time() - self.start_time
        
        logger.info("\n" + "=" * 60)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Steps: {self.step_count}")
        logger.info(f"Total Time: {elapsed:.1f}s")
        logger.info(f"Iterations: {self.state.iteration}")
        logger.info("")
        logger.info(f"Candidates Discovered: {self.state.num_candidates}")
        logger.info(f"Candidates Scored: {self.state.num_scored}")
        logger.info(f"Candidates Validated: {self.state.num_validated}")
        logger.info(f"Valid (Material Transitions): {self.state.num_valid}")
        logger.info(f"Final Selection: {self.state.num_selected}/{self.config.target_n}")
        
        if self.state.selection_report:
            logger.info("")
            logger.info(f"Diversity Score: {self.state.selection_report.diversity_score:.3f}")
            logger.info(f"Mean Quality: {self.state.selection_report.mean_quality_score:.3f}")
        
        if self.state.selected_ids:
            logger.info("")
            logger.info("Selected Images:")
            for i, cid in enumerate(self.state.selected_ids):
                candidate = self.state.candidates.get(cid)
                score = candidate.scores.total_score if candidate and candidate.scores else 0
                logger.info(f"  {i+1}. {cid} (score={score:.3f})")
        
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
    """
    Convenience function to run the texture curator.
    
    Args:
        rwtd_path: Path to RWTD reference dataset
        source_pool_path: Path to source pool
        target_n: Number of samples to select
        output_path: Output directory
        llm_model: Ollama model name
        device: Device for vision models
        **kwargs: Additional config options
    
    Returns:
        Final GraphState with results
    """
    # Create config
    config = Config(
        rwtd_path=Path(rwtd_path),
        source_pool_path=Path(source_pool_path),
        target_n=target_n,
        output_path=Path(output_path),
        device=device,
        **kwargs
    )
    
    # Create and run orchestrator
    orchestrator = TextureCuratorOrchestrator(
        config=config,
        llm_model=llm_model,
        device=device,
    )
    
    return orchestrator.run()


# ============================================
# CLI Entry Point
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Texture Curator - Multi-Agent Dataset Curation")
    parser.add_argument("--rwtd", type=str, default="/home/aviad/RWTD",
                       help="Path to RWTD reference dataset")
    parser.add_argument("--source", type=str, default="/datasets/ade20k/real_texture_boundaries_20260201",
                       help="Path to source pool")
    parser.add_argument("--target-n", type=int, default=10,
                       help="Number of samples to select")
    parser.add_argument("--output", type=str, default="./outputs",
                       help="Output directory")
    parser.add_argument("--model", type=str, default="qwen2.5:7b",
                       help="Ollama model name")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda or cpu)")
    parser.add_argument("--critic-samples", type=int, default=20,
                       help="Number of samples for critic to review")
    
    args = parser.parse_args()
    
    # Run
    state = run_texture_curator(
        rwtd_path=args.rwtd,
        source_pool_path=args.source,
        target_n=args.target_n,
        output_path=args.output,
        llm_model=args.model,
        device=args.device,
        critic_sample_size=args.critic_samples,
    )
    
    print(f"\nDone! Selected {state.num_selected} images.")
    print(f"Results saved to: {args.output}/curated_dataset/")