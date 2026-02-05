#!/usr/bin/env python3
"""
Texture Curator - Multi-Agent Dataset Curation System

This is the main entry point for running the texture curation workflow.

USAGE:
    # Basic usage with defaults
    python main.py
    
    # Custom paths and settings
    python main.py --rwtd /path/to/RWTD --source /path/to/pool --target-n 10
    
    # Use CPU instead of GPU
    python main.py --device cpu

WORKFLOW:
    1. Profiler: Builds RWTD reference profile
    2. Analyst: Scores candidates from source pool
    3. Critic: Audits quality (material vs object transitions)
    4. Optimizer: Selects diverse final subset
    
OUTPUT:
    - ./outputs/curated_dataset/images/  - Selected images
    - ./outputs/curated_dataset/masks/   - Corresponding masks
    - ./outputs/curated_dataset/metadata.json - Selection metadata
    - ./checkpoints/ - Intermediate checkpoints for debugging
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import Config
from graph.orchestrator import TextureCuratorOrchestrator


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Reduce noise from some libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Texture Curator - Multi-Agent Dataset Curation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults
  python main.py
  
  # Custom settings
  python main.py --rwtd /home/aviad/RWTD --source /datasets/ade20k/real_texture_boundaries_20260201 --target-n 15
  
  # CPU mode (no GPU)
  python main.py --device cpu
        """
    )
    
    # Data paths
    parser.add_argument(
        "--rwtd", 
        type=str, 
        default="/home/aviad/RWTD",
        help="Path to RWTD reference dataset"
    )
    parser.add_argument(
        "--source", 
        type=str, 
        default="/datasets/ade20k/real_texture_boundaries_20260201",
        help="Path to source pool to filter"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="./outputs",
        help="Output directory for results"
    )
    
    # Task settings
    parser.add_argument(
        "--target-n", 
        type=int, 
        default=10,
        help="Number of samples to select (default: 10)"
    )
    parser.add_argument(
        "--critic-samples", 
        type=int, 
        default=20,
        help="Number of samples for critic to review (default: 20)"
    )
    parser.add_argument(
        "--max-iterations", 
        type=int, 
        default=3,
        help="Maximum reroute iterations (default: 3)"
    )
    
    # Model settings
    parser.add_argument(
        "--model", 
        type=str, 
        default="qwen2.5:7b",
        help="Ollama model for agent reasoning (default: qwen2.5:7b)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for vision models (default: cuda)"
    )
    
    # Threshold overrides
    parser.add_argument(
        "--quality-threshold", 
        type=float, 
        default=0.7,
        help="Quality gate threshold for critic (default: 0.7)"
    )
    parser.add_argument(
        "--diversity-weight", 
        type=float, 
        default=0.3,
        help="Diversity weight in selection (default: 0.3)"
    )
    
    # Runtime options
    parser.add_argument(
        "--no-checkpoints", 
        action="store_true",
        help="Disable checkpoint saving"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--max-steps", 
        type=int, 
        default=50,
        help="Maximum workflow steps (default: 50)"
    )
    
    return parser.parse_args()


def validate_paths(args) -> bool:
    """Validate that required paths exist."""
    rwtd_path = Path(args.rwtd)
    source_path = Path(args.source)
    
    if not rwtd_path.exists():
        print(f"ERROR: RWTD path does not exist: {rwtd_path}")
        return False
    
    if not (rwtd_path / "images").exists():
        print(f"ERROR: RWTD images directory not found: {rwtd_path / 'images'}")
        return False
    
    if not source_path.exists():
        print(f"ERROR: Source pool path does not exist: {source_path}")
        return False
    
    if not (source_path / "images").exists():
        print(f"ERROR: Source pool images directory not found: {source_path / 'images'}")
        return False
    
    return True


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate paths
    if not validate_paths(args):
        sys.exit(1)
    
    # Create configuration
    config = Config(
        rwtd_path=Path(args.rwtd),
        source_pool_path=Path(args.source),
        output_path=Path(args.output),
        target_n=args.target_n,
        critic_sample_size=args.critic_samples,
        max_iterations=args.max_iterations,
        device=args.device,
        save_checkpoints=not args.no_checkpoints,
        verbose=args.verbose,
    )
    
    # Override thresholds
    config.thresholds.quality_gate_min = args.quality_threshold
    config.thresholds.diversity_weight = args.diversity_weight
    
    # Print banner
    print()
    print("=" * 70)
    print("🎨 TEXTURE CURATOR - Multi-Agent Dataset Curation System")
    print("=" * 70)
    print(f"  RWTD Reference:  {args.rwtd}")
    print(f"  Source Pool:     {args.source}")
    print(f"  Target Selection: {args.target_n} images")
    print(f"  LLM Model:       {args.model}")
    print(f"  Device:          {args.device}")
    print("=" * 70)
    print()
    
    try:
        # Create orchestrator
        logger.info("Initializing orchestrator...")
        orchestrator = TextureCuratorOrchestrator(
            config=config,
            llm_model=args.model,
            device=args.device,
            save_checkpoints=not args.no_checkpoints,
        )
        
        # Run workflow
        logger.info("Starting workflow...")
        state = orchestrator.run(max_steps=args.max_steps)
        
        # Report results
        print()
        print("=" * 70)
        print("✅ CURATION COMPLETE")
        print("=" * 70)
        print(f"  Images Selected: {state.num_selected}/{args.target_n}")
        
        if state.selection_report:
            print(f"  Diversity Score: {state.selection_report.diversity_score:.3f}")
            print(f"  Mean Quality:    {state.selection_report.mean_quality_score:.3f}")
        
        print()
        print(f"  Output: {args.output}/curated_dataset/")
        print("=" * 70)
        
        return 0 if state.num_selected >= args.target_n else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
        
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())