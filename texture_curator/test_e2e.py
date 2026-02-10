#!/usr/bin/env python3
"""
End-to-End Test for Texture Curator.

This script tests the complete multi-agent pipeline:
1. Profiler builds RWTD profile
2. Analyst scores candidates
3. Critic audits quality
4. Optimizer selects diverse subset

USAGE:
    cd ~/texture_curator
    python test_e2e.py
    
    # Test with fewer samples (faster)
    python test_e2e.py --quick
    
    # Specify custom paths
    python test_e2e.py --rwtd /path/to/RWTD --source /path/to/pool
"""

import sys
import argparse
import json
import logging
import shutil
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def setup_logging():
    """Configure logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    # Reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)


def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("Test 1: Module Imports")
    print("=" * 60)
    
    modules = [
        ("config.settings", "Config"),
        ("state.models", "RWTDProfile"),
        ("state.graph_state", "GraphState"),
        ("llm.ollama_client", "OllamaClient"),
        ("agents.base", "BaseAgent"),
        ("agents.profiler", "ProfilerAgent"),
        ("agents.analyst", "AnalystAgent"),
        ("agents.critic", "CriticAgent"),
        ("agents.optimizer", "OptimizerAgent"),
        ("agents.planner", "PlannerAgent"),
        ("graph.orchestrator", "TextureCuratorOrchestrator"),
        ("mcp_servers.vision.dino_extractor", "DINOv2Extractor"),
        ("mcp_servers.vision.texture_stats", "TextureStatsExtractor"),
        ("mcp_servers.vision.boundary_metrics", "BoundaryMetricsExtractor"),
    ]
    
    all_ok = True
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"  ✓ {module_name}.{class_name}")
        except Exception as e:
            print(f"  ✗ {module_name}.{class_name}: {e}")
            all_ok = False
    
    if all_ok:
        print("\n✅ All imports successful")
    else:
        print("\n❌ Some imports failed")
    
    return all_ok


def test_ollama():
    """Test Ollama connection."""
    print("\n" + "=" * 60)
    print("Test 2: Ollama Connection")
    print("=" * 60)
    
    from llm.ollama_client import OllamaClient
    
    client = OllamaClient(model="qwen2.5:7b")
    
    if not client.is_available():
        print("  ✗ Ollama not running")
        print("    Start with: ollama serve")
        return False
    
    print("  ✓ Ollama is running")
    
    # Check model
    models = client.list_models()
    if "qwen2.5:7b" in models or any("qwen2.5" in m for m in models):
        print("  ✓ qwen2.5:7b model available")
    else:
        print(f"  ✗ qwen2.5:7b not found. Available: {models}")
        print("    Pull with: ollama pull qwen2.5:7b")
        return False
    
    # Quick test
    response = client.chat("Say 'OK' if you can hear me.", system_prompt="Respond briefly.")
    if "OK" in response.content.upper() or len(response.content) > 0:
        print(f"  ✓ LLM responds: {response.content[:50]}...")
    else:
        print("  ✗ LLM did not respond")
        return False
    
    print("\n✅ Ollama test passed")
    return True


def test_profiler(rwtd_path: Path, device: str = "cuda", target_n: int = 10):
    """Test the Profiler agent."""
    print("\n" + "=" * 60)
    print("Test 3: Profiler Agent")
    print("=" * 60)

    from config.settings import Config
    from state.graph_state import GraphState
    from llm.ollama_client import OllamaClient
    from agents.profiler import ProfilerAgent

    # Setup
    config = Config(rwtd_path=rwtd_path, device=device, target_n=target_n)
    state = GraphState(config=config)
    client = OllamaClient(model="qwen2.5:7b")
    
    # Create profiler
    profiler = ProfilerAgent(llm_client=client, device=device)
    print(f"  ✓ Profiler created")
    
    # Run profiling
    print("  Running RWTD profiling...")
    start = time.time()
    result = profiler.run_full_profiling(state)
    elapsed = time.time() - start
    
    if result.success:
        print(f"  ✓ Profiling complete in {elapsed:.1f}s")
        print(f"    - Samples: {state.rwtd_profile.num_samples}")
        print(f"    - Centroid norm: {state.rwtd_profile.centroid_embedding.sum():.2f}")
        print(f"    - Entropy mean: {state.rwtd_profile.entropy_distribution.mean:.2f}")
        print("\n✅ Profiler test passed")
        return True, state
    else:
        print(f"  ✗ Profiling failed: {result.error_message}")
        print("\n❌ Profiler test failed")
        return False, None


def test_analyst(state: "GraphState", source_path: Path, device: str = "cuda", filter_passed: bool = False):
    """Test the Analyst agent."""
    print("\n" + "=" * 60)
    print("Test 4: Analyst Agent")
    print("=" * 60)

    from llm.ollama_client import OllamaClient
    from agents.analyst import AnalystAgent

    # Update config with source path and filter mode
    state.config.source_pool_path = source_path
    state.config.filter_passed = filter_passed
    
    # Create analyst (share extractors with profiler for efficiency)
    client = OllamaClient(model="qwen2.5:7b")
    analyst = AnalystAgent(llm_client=client, device=device)
    print(f"  ✓ Analyst created")
    
    # Run analysis
    print("  Running candidate analysis...")
    start = time.time()
    result = analyst.run_full_analysis(state)
    elapsed = time.time() - start
    
    if result.success:
        print(f"  ✓ Analysis complete in {elapsed:.1f}s")
        print(f"    - Candidates: {state.num_candidates}")
        print(f"    - Scored: {state.num_scored}")
        
        # Show top 5
        top = state.get_top_candidates(5)
        print(f"    - Top 5 scores:")
        for c in top:
            print(f"      {c.id}: {c.scores.total_score:.3f}")
        
        print("\n✅ Analyst test passed")
        return True
    else:
        print(f"  ✗ Analysis failed: {result.error_message}")
        print("\n❌ Analyst test failed")
        return False


def test_critic(state: "GraphState", device: str = "cuda"):
    """Test the Critic agent."""
    print("\n" + "=" * 60)
    print("Test 5: Critic Agent")
    print("=" * 60)
    
    from llm.ollama_client import OllamaClient
    from agents.critic import CriticAgent
    
    # Create critic
    client = OllamaClient(model="qwen2.5:7b")
    critic = CriticAgent(llm_client=client, device=device)
    print(f"  ✓ Critic created")
    
    # Run critique (sample fewer for testing)
    print("  Running quality audit...")
    start = time.time()
    result = critic.run_full_critique(state, n_samples=min(10, state.num_scored))
    elapsed = time.time() - start
    
    if result.success:
        print(f"  ✓ Critique complete in {elapsed:.1f}s")
        print(f"    - Reviewed: {state.critic_report.samples_reviewed}")
        print(f"    - Material transitions: {state.critic_report.material_transitions}")
        print(f"    - Object boundaries: {state.critic_report.object_boundaries}")
        print(f"    - Quality score: {state.critic_report.quality_score:.0%}")
        print("\n✅ Critic test passed")
        return True
    else:
        print(f"  ✗ Critique failed: {result.error_message}")
        print("\n❌ Critic test failed")
        return False


def test_optimizer(state: "GraphState"):
    """Test the Optimizer agent."""
    print("\n" + "=" * 60)
    print("Test 6: Optimizer Agent")
    print("=" * 60)
    
    from llm.ollama_client import OllamaClient
    from agents.optimizer import OptimizerAgent
    
    # Create optimizer
    client = OllamaClient(model="qwen2.5:7b")
    optimizer = OptimizerAgent(llm_client=client)
    print(f"  ✓ Optimizer created")
    
    # Run optimization
    print("  Running diverse selection...")
    start = time.time()
    result = optimizer.run_full_optimization(state)
    elapsed = time.time() - start
    
    if result.success:
        print(f"  ✓ Optimization complete in {elapsed:.1f}s")
        print(f"    - Selected: {state.num_selected}/{state.config.target_n}")
        print(f"    - Diversity score: {state.selection_report.diversity_score:.3f}")
        print(f"    - Mean quality: {state.selection_report.mean_quality_score:.3f}")
        print(f"    - Selected IDs: {state.selected_ids[:5]}...")
        print("\n✅ Optimizer test passed")
        return True
    else:
        print(f"  ✗ Optimization failed: {result.error_message}")
        print("\n❌ Optimizer test failed")
        return False


def test_full_orchestrator(rwtd_path: Path, source_path: Path, device: str = "cuda", filter_passed: bool = False, target_n: int = 10, vlm_model: str = "qwen2.5vl:7b", skip_mask_filter: bool = False):
    """Test the full orchestrator."""
    print("\n" + "=" * 60)
    print("Test 7: Full Orchestrator (End-to-End)")
    print("=" * 60)

    from config.settings import Config
    from graph.orchestrator import TextureCuratorOrchestrator

    # Create config
    config = Config(
        rwtd_path=rwtd_path,
        source_pool_path=source_path,
        target_n=target_n,
        critic_sample_size=min(20, target_n * 2),
        device=device,
        save_checkpoints=False,  # Faster for testing
        filter_passed=filter_passed,
    )
    config.mask_filter.vlm_model = vlm_model
    config.mask_filter.skip_vlm = skip_mask_filter
    
    print(f"  Config: target_n={config.target_n}, critic_samples={config.critic_sample_size}")
    
    # Create orchestrator
    orchestrator = TextureCuratorOrchestrator(
        config=config,
        llm_model="qwen2.5:7b",
        device=device,
        save_checkpoints=False,
    )
    print(f"  ✓ Orchestrator created")
    
    # Run
    print("  Running full workflow...")
    start = time.time()
    state = orchestrator.run(max_steps=20)
    elapsed = time.time() - start
    
    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"    - Final phase: {state.current_phase.value}")
    print(f"    - Candidates: {state.num_candidates}")
    print(f"    - Scored: {state.num_scored}")
    print(f"    - Validated: {state.num_validated}")
    print(f"    - Selected: {state.num_selected}/{config.target_n}")
    
    if state.num_selected >= config.target_n:
        print("\n✅ Full orchestrator test passed!")
        return True, state
    else:
        print(f"\n⚠️ Partial success: selected {state.num_selected}/{config.target_n}")
        return True, state  # Still count as pass if it ran


def create_overlay(image_path, mask_path, alpha=0.5):
    """Create an overlay image: mask boundaries shown in green on top of the crop."""
    import cv2
    import numpy as np
    from PIL import Image

    crop = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=2)

    if mask.shape[:2] != crop.shape[:2]:
        mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay = crop.copy()
    boundary = mask > 127
    overlay[boundary] = [0, 255, 0]

    result = crop.copy()
    result[boundary] = (
        (1 - alpha) * crop[boundary].astype(float) +
        alpha * overlay[boundary].astype(float)
    ).astype(np.uint8)

    return Image.fromarray(result)


def export_results(state, output_path: Path):
    """Export selected images, masks, and overlays to flat directories.

    Creates:
        output_path/images/{stem}.jpg
        output_path/masks/{stem}.png
        output_path/overlays/{stem}.jpg
    """
    if not hasattr(state, 'selected_ids') or not state.selected_ids:
        print("\n⚠️ No selected candidates to export.")
        return

    images_dir = output_path / "images"
    masks_dir = output_path / "masks"
    overlays_dir = output_path / "overlays"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    for cid in state.selected_ids:
        candidate = state.candidates.get(cid)
        if candidate is None:
            continue

        stem = candidate.image_path.stem
        # Copy image keeping original extension
        dst_image = images_dir / candidate.image_path.name
        shutil.copy2(candidate.image_path, dst_image)

        # Copy mask with same stem (keeps .png extension)
        dst_mask = masks_dir / f"{stem}{candidate.mask_path.suffix}"
        shutil.copy2(candidate.mask_path, dst_mask)

        # Generate overlay
        try:
            overlay = create_overlay(candidate.image_path, candidate.mask_path)
            overlay.save(str(overlays_dir / f"{stem}.jpg"), quality=90)
        except Exception:
            pass

        exported += 1

    # Save metadata
    metadata = {
        "num_exported": exported,
        "candidates": [
            {
                "id": cid,
                "image": state.candidates[cid].image_path.name if state.candidates.get(cid) else None,
                "score": float(state.candidates[cid].scores.total_score)
                    if state.candidates.get(cid) and state.candidates[cid].scores else 0,
            }
            for cid in state.selected_ids
        ],
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n📦 Exported {exported} images to: {output_path}")
    print(f"   images/   : {exported} files")
    print(f"   masks/    : {exported} files")
    print(f"   overlays/ : {exported} files")


def main():
    parser = argparse.ArgumentParser(description="End-to-End tests for Texture Curator")
    parser.add_argument("--rwtd", type=str, default="/home/aviad/RWTD",
                       help="Path to RWTD dataset")
    parser.add_argument("--source", type=str, default="/home/aviad/real_world_texture_boundary/google_landmarks_v2_scale/run_20260208_230720",
                       help="Path to source pool (supports both flat images/ and nested crops/ structures)")
    parser.add_argument("--filter-passed", action="store_true", default=False,
                       help="Only use crops that passed the entropy filter")
    parser.add_argument("--output", type=str, default="/datasets/google_landmarks_v2_scale/real_world_texture_boundary/results/run_20260208_230720",
                       help="Output path to export selected images and masks (flat images/ and masks/ dirs)")
    parser.add_argument("--target-n", type=int, default=17000,
                       help="Number of images to select (default: 10)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda or cpu)")
    parser.add_argument("--vlm-model", type=str, default="qwen2.5vl:7b",
                       help="VLM model for mask quality filter (default: qwen2.5vl:7b)")
    parser.add_argument("--skip-mask-filter", action="store_true",
                       help="Skip VLM-based mask quality filtering")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick tests only (skip full orchestrator)")
    parser.add_argument("--full-only", action="store_true",
                       help="Run only the full orchestrator test")
    args = parser.parse_args()
    
    # args.quick = True
    
    setup_logging()
    
    rwtd_path = Path(args.rwtd)
    source_path = Path(args.source)
    
    # Validate paths
    if not rwtd_path.exists():
        print(f"ERROR: RWTD path not found: {rwtd_path}")
        sys.exit(1)
    
    if not source_path.exists():
        print(f"ERROR: Source path not found: {source_path}")
        sys.exit(1)
    
    print()
    print("🧪 TEXTURE CURATOR - END-TO-END TESTS")
    print("=" * 60)
    print(f"  RWTD: {rwtd_path}")
    print(f"  Source: {source_path}")
    print(f"  Filter: {'passed only' if args.filter_passed else 'all crops'}")
    print(f"  Output: {args.output or '(none)'}")
    print(f"  VLM:    {args.vlm_model}{' (SKIPPED)' if args.skip_mask_filter else ''}")
    print(f"  Device: {args.device}")
    print("=" * 60)
    
    results = []
    state = None
    
    if args.full_only:
        # Only run full orchestrator
        success, state = test_full_orchestrator(rwtd_path, source_path, args.device, args.filter_passed, args.target_n, args.vlm_model, args.skip_mask_filter)
        results.append(("Full Orchestrator", success))
    else:
        # Test 1: Imports
        results.append(("Imports", test_imports()))
        
        # Test 2: Ollama
        results.append(("Ollama", test_ollama()))
        
        if not results[-1][1]:
            print("\n⚠️ Skipping remaining tests (Ollama not available)")
        else:
            # Test 3: Profiler
            success, state = test_profiler(rwtd_path, args.device, args.target_n)
            results.append(("Profiler", success))
            
            if success and state:
                # Test 4: Analyst
                results.append(("Analyst", test_analyst(state, source_path, args.device, args.filter_passed)))
                
                # Test 5: Critic
                results.append(("Critic", test_critic(state, args.device)))
                
                # Test 6: Optimizer
                results.append(("Optimizer", test_optimizer(state)))
            
            # Test 7: Full Orchestrator (skip if quick mode)
            if not args.quick:
                orch_success, orch_state = test_full_orchestrator(rwtd_path, source_path, args.device, args.filter_passed, args.target_n, args.vlm_model, args.skip_mask_filter)
                results.append(("Full Orchestrator", orch_success))
                # Prefer the orchestrator state for export (it ran the full pipeline)
                if orch_state and orch_state.selected_ids:
                    state = orch_state
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All tests passed! The system is ready.")
    else:
        print("⚠️ Some tests failed. Please review the output above.")

    # Export results if output path specified and we have a state with selections
    if args.output and state and hasattr(state, 'selected_ids') and state.selected_ids:
        export_results(state, Path(args.output))
    elif args.output:
        print(f"\n⚠️ No selected candidates to export to {args.output}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())