#!/usr/bin/env python3
"""
VLM Mask Filter Demo - Test on 100 crops, select top 10.

Shows:
1. Mask filter breakdown (prefilter vs VLM, pass vs fail, reasons)
2. Scoring of surviving candidates
3. Top 10 selection with visual composites saved

Usage:
    python test_vlm_filter_demo.py
    python test_vlm_filter_demo.py --n-candidates 200 --target-n 15
    python test_vlm_filter_demo.py --skip-vlm   # prefilter only (fast)
"""

import sys
import json
import time
import random
import shutil
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import Config, MaskStatus
from state.graph_state import GraphState
from state.models import CandidateRecord
from llm.ollama_client import OllamaClient
from agents.mask_filter import MaskFilterAgent
from agents.analyst import AnalystAgent


def create_overlay(image_path, mask_path, alpha=0.5):
    """Create an overlay image: mask boundaries shown in green on top of the crop."""
    import cv2
    import numpy as np
    from PIL import Image

    crop = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))

    # Dilate mask lines for visibility
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Resize mask to match crop if needed
    if mask.shape[:2] != crop.shape[:2]:
        mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create green overlay where mask is white
    overlay = crop.copy()
    boundary = mask > 127
    overlay[boundary] = [0, 255, 0]

    # Blend
    result = crop.copy()
    result[boundary] = (
        (1 - alpha) * crop[boundary].astype(float) +
        alpha * overlay[boundary].astype(float)
    ).astype(np.uint8)

    return Image.fromarray(result)


def main():
    parser = argparse.ArgumentParser(description="VLM Mask Filter Demo")
    parser.add_argument("--n-candidates", type=int, default=500,
                        help="Number of candidates to sample (default: 100)")
    parser.add_argument("--target-n", type=int, default=10,
                        help="Number of top images to select (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for vision models (default: cuda)")
    parser.add_argument("--skip-vlm", action="store_true",
                        help="Skip VLM, use only math prefilter")
    parser.add_argument("--output", type=str, default="./outputs/vlm_filter_demo",
                        help="Output directory for results")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Step 1: Sample candidates
    # ================================================================
    print()
    print("=" * 70)
    print("  VLM MASK FILTER DEMO")
    print("=" * 70)

    base = Path("/home/aviad/real_world_texture_boundary/google_landmarks_v2_scale/run_20260208_230720")
    rwtd = Path("/home/aviad/RWTD")

    crop_dir = base / "crops/medium"
    mask_dir = base / "masks/medium"
    passed_dir = base / "filter/passed/medium"

    # Use entropy-filter passed crops
    all_passed = sorted(passed_dir.iterdir())
    random.seed(args.seed)
    sample = random.sample(all_passed, min(args.n_candidates, len(all_passed)))

    print(f"  Source:     {base}")
    print(f"  Pool size:  {len(all_passed)} (entropy-passed medium crops)")
    print(f"  Sampled:    {len(sample)} candidates")
    print(f"  Target:     top {args.target_n}")
    print(f"  VLM:        {'SKIPPED' if args.skip_vlm else 'qwen2.5vl:7b'}")
    print(f"  Output:     {output_dir}")
    print("=" * 70)

    # ================================================================
    # Step 2: Build state with candidates
    # ================================================================
    print("\n[1/4] Building candidate pool...")

    config = Config(
        rwtd_path=rwtd,
        source_pool_path=base,
        target_n=args.target_n,
        device=args.device,
        filter_passed=True,
    )
    config.mask_filter.skip_vlm = args.skip_vlm

    state = GraphState(config=config)

    for crop_file in sample:
        stem = crop_file.stem
        crop_path = crop_dir / f"{stem}.jpg"
        mask_path = mask_dir / f"{stem}.png"

        if crop_path.exists() and mask_path.exists():
            state.candidates[stem] = CandidateRecord(
                id=stem,
                image_path=crop_path,
                mask_path=mask_path,
                mask_status=MaskStatus.PENDING,
            )

    print(f"  Registered {len(state.candidates)} candidates")

    # ================================================================
    # Step 3: Run mask filter
    # ================================================================
    print(f"\n[2/4] Running mask filter ({'prefilter only' if args.skip_vlm else 'prefilter + VLM'})...")

    llm = OllamaClient(model="qwen2.5:7b")
    mask_agent = MaskFilterAgent(llm_client=llm, vlm_model="qwen2.5vl:7b", device=args.device)

    t0 = time.time()
    filter_result = mask_agent.run_full_filtering(state)
    filter_time = time.time() - t0

    data = filter_result.data
    print(f"\n  MASK FILTER RESULTS ({filter_time:.1f}s)")
    print(f"  {'─' * 50}")
    print(f"  Total assessed:     {data['assessed']}")
    print(f"  Passed:             {data['passed']}  ({data['passed']/data['assessed']*100:.0f}%)")
    print(f"  Rejected:           {data['failed']}  ({data['failed']/data['assessed']*100:.0f}%)")
    print(f"    - Prefilter:      {data['prefilter_rejected']}")
    print(f"    - VLM:            {data['vlm_failed']}")
    if data.get("vlm_errors", 0):
        print(f"    - VLM errors:     {data['vlm_errors']}")
    print(f"  {'─' * 50}")
    if data["reasons"]:
        print(f"  Rejection reasons:")
        for reason, count in sorted(data["reasons"].items(), key=lambda x: -x[1]):
            pct = count / data["assessed"] * 100
            print(f"    {reason:40s} {count:4d}  ({pct:.1f}%)")
    print()

    # Save rejected composites and overlays for review
    rejected_dir = output_dir / "rejected"
    rejected_overlay_dir = output_dir / "rejected_overlays"
    rejected_dir.mkdir(exist_ok=True)
    rejected_overlay_dir.mkdir(exist_ok=True)
    rejected_count = 0
    for cid, c in state.candidates.items():
        if c.mask_status == MaskStatus.REJECTED and rejected_count < 30:
            reason = c.mask_filter_verdict.reason if c.mask_filter_verdict else "unknown"
            fname = f"{reason}_{cid[:30]}"
            try:
                comp = MaskFilterAgent.create_composite(c.image_path, c.mask_path)
                shutil.copy(str(comp), str(rejected_dir / f"{fname}.jpg"))
                comp.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                overlay = create_overlay(c.image_path, c.mask_path)
                overlay.save(str(rejected_overlay_dir / f"{fname}.jpg"), quality=90)
            except Exception:
                pass
            rejected_count += 1
    print(f"  Saved {rejected_count} rejected composites to: {rejected_dir}")
    print(f"  Saved {rejected_count} rejected overlays to: {rejected_overlay_dir}")

    # ================================================================
    # Step 4: Score surviving candidates (needs RWTD profile)
    # ================================================================
    n_passed = data["passed"]
    if n_passed == 0:
        print("\n  No candidates passed filter. Nothing to score.")
        return 1

    print(f"\n[3/4] Scoring {n_passed} surviving candidates...")
    print(f"  Building RWTD profile first...")

    from agents.profiler import ProfilerAgent
    profiler = ProfilerAgent(llm_client=llm, device=args.device)
    profile_result = profiler.run_full_profiling(state)
    if not profile_result.success:
        print(f"  ERROR: Profiling failed: {profile_result.error_message}")
        return 1
    print(f"  RWTD profile built ({profile_result.data.get('num_samples', '?')} samples)")

    analyst = AnalystAgent(llm_client=llm, device=args.device)

    # Extract features
    print(f"  Extracting features...")
    t0 = time.time()
    feat_result = analyst._extract_features(state, {})
    feat_time = time.time() - t0
    print(f"  Features extracted in {feat_time:.1f}s")

    # Score candidates
    print(f"  Scoring...")
    score_result = analyst._score_candidates(state)
    if not score_result.success:
        print(f"  ERROR: Scoring failed: {score_result.error_message}")
        return 1

    scored = [(cid, c) for cid, c in state.candidates.items()
              if c.scores is not None and c.mask_status != MaskStatus.REJECTED]
    scored.sort(key=lambda x: x[1].scores.total_score, reverse=True)

    print(f"  Scored {len(scored)} candidates")
    if score_result.data:
        print(f"  Score stats: mean={score_result.data.get('mean_score', 0):.3f}, "
              f"max={score_result.data.get('max_score', 0):.3f}, "
              f"min={score_result.data.get('min_score', 0):.3f}")

    # ================================================================
    # Step 5: Select top N
    # ================================================================
    top_n = min(args.target_n, len(scored))
    print(f"\n[4/4] Selecting top {top_n} from {len(scored)} scored candidates...")

    top_dir = output_dir / "top_selected"
    if top_dir.exists():
        shutil.rmtree(top_dir)
    top_dir.mkdir(exist_ok=True)

    top_images_dir = output_dir / "top_selected" / "images"
    top_masks_dir = output_dir / "top_selected" / "masks"
    top_composites_dir = output_dir / "top_selected" / "composites"
    top_overlays_dir = output_dir / "top_selected" / "overlays"
    top_images_dir.mkdir(parents=True, exist_ok=True)
    top_masks_dir.mkdir(parents=True, exist_ok=True)
    top_composites_dir.mkdir(parents=True, exist_ok=True)
    top_overlays_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(f"  {'Rank':<5} {'Score':>6} {'Semantic':>9} {'Texture':>8} {'Boundary':>9}  ID")
    print(f"  {'─' * 70}")

    top_meta = []
    for rank, (cid, c) in enumerate(scored[:top_n], 1):
        s = c.scores
        print(f"  {rank:<5} {s.total_score:>6.3f} {s.semantic_score:>9.3f} "
              f"{s.texture_score:>8.3f} {s.boundary_score:>9.3f}  {cid[:40]}")

        # Copy image and mask
        shutil.copy2(c.image_path, top_images_dir / c.image_path.name)
        shutil.copy2(c.mask_path, top_masks_dir / c.mask_path.name)

        # Save composite
        try:
            comp = MaskFilterAgent.create_composite(c.image_path, c.mask_path)
            shutil.copy(str(comp), str(top_composites_dir / f"rank{rank:02d}_{cid[:30]}.jpg"))
            comp.unlink(missing_ok=True)
        except Exception:
            pass

        # Save overlay
        try:
            overlay = create_overlay(c.image_path, c.mask_path)
            overlay.save(str(top_overlays_dir / f"rank{rank:02d}_{cid[:30]}.jpg"), quality=90)
        except Exception:
            pass

        top_meta.append({
            "rank": rank,
            "id": cid,
            "total_score": round(s.total_score, 4),
            "semantic_score": round(s.semantic_score, 4),
            "texture_score": round(s.texture_score, 4),
            "boundary_score": round(s.boundary_score, 4),
            "image": c.image_path.name,
            "mask": c.mask_path.name,
        })

    # Also show some bottom-ranked for comparison
    if len(scored) > top_n + 5:
        print(f"  {'─' * 70}")
        print(f"  ... {len(scored) - top_n} more ...")
        print(f"  {'─' * 70}")
        for rank, (cid, c) in enumerate(scored[-3:], len(scored) - 2):
            s = c.scores
            print(f"  {rank:<5} {s.total_score:>6.3f} {s.semantic_score:>9.3f} "
                  f"{s.texture_score:>8.3f} {s.boundary_score:>9.3f}  {cid[:40]}")

    # ================================================================
    # Summary
    # ================================================================
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Candidates sampled:     {len(state.candidates)}")
    print(f"  Mask filter rejected:   {data['failed']} ({data['failed']/data['assessed']*100:.0f}%)")
    print(f"  Mask filter passed:     {data['passed']} ({data['passed']/data['assessed']*100:.0f}%)")
    print(f"  Scored:                 {len(scored)}")
    print(f"  Selected top:           {top_n}")
    if scored:
        print(f"  Top score:              {scored[0][1].scores.total_score:.4f}")
        print(f"  #10 score:              {scored[min(9, len(scored)-1)][1].scores.total_score:.4f}")
    print(f"  Filter time:            {filter_time:.1f}s")
    print()
    print(f"  Output:")
    print(f"    Composites:  {top_composites_dir}")
    print(f"    Overlays:    {top_overlays_dir}")
    print(f"    Images:      {top_images_dir}")
    print(f"    Masks:       {top_masks_dir}")
    print(f"    Rejected:    {rejected_dir}")
    print(f"    Rej overlays:{rejected_overlay_dir}")
    print("=" * 70)

    # Save full metadata
    meta = {
        "config": {
            "n_candidates": len(state.candidates),
            "target_n": args.target_n,
            "skip_vlm": args.skip_vlm,
            "seed": args.seed,
        },
        "filter_results": data,
        "filter_time_seconds": round(filter_time, 1),
        "top_selected": top_meta,
    }
    meta_path = output_dir / "demo_results.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Full metadata: {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
