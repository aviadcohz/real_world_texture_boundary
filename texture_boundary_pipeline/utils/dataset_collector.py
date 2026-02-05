"""
Dataset Collector

Collect all TEXTURE TRANSITION crops from iterative pipeline results.
Filters out SEMANTIC crops based on iteration_0 classification.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Set, Union


def collect_texture_crops(
    results_dir: Union[str, Path],
    output_dataset_dir: Union[str, Path],
    copy_filtered: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Collect all TEXTURE TRANSITION crops from iterative pipeline results.

    Pipeline workflow:
    - Iteration 0: Basic pipeline → crops/ → classification (iteration_0/classification_results.json)
      * Only certain crop categories (medium, large, xlarge) are classified
    - Iteration 1+: Refine SEMANTIC crops → iteration_N/refined_crops/
      * classification may be empty if no crops matched the category filter

    Collection strategy:
    1. Read iteration_0/classification_results.json
    2. Identify which PARENT crops from iteration_0 are TEXTURE TRANSITION vs SEMANTIC
    3. Collect ALL crops from iteration_0/crops/ that are TEXTURE TRANSITION
    4. For iterations 1+, collect ALL refined crops that descended from TEXTURE TRANSITION parents
    5. Output to flat directory structure (all images in one folder, no size subdirs)

    Args:
        results_dir: Path to pipeline results directory (e.g., results/run_TIMESTAMP/)
        output_dataset_dir: Path to output dataset directory
        copy_filtered: If True, copy filtered (SEMANTIC) crops to "filtered" folder
        verbose: Print progress

    Returns:
        Dict with collection statistics
    """
    results_dir = Path(results_dir)
    output_dataset_dir = Path(output_dataset_dir)

    if not results_dir.exists():
        raise ValueError(f"Results directory not found: {results_dir}")

    # Auto-detect structure: is this a run directory or base directory?
    iteration_0_dir = results_dir / "iteration_0"

    if not iteration_0_dir.exists():
        # This might be a base directory containing run_* folders
        run_dirs = sorted(results_dir.glob("run_*"))

        if run_dirs:
            if verbose:
                print(f"⚠️  Detected base directory with {len(run_dirs)} run(s)")
                print(f"   Using latest run: {run_dirs[-1].name}")

            results_dir = run_dirs[-1]
            iteration_0_dir = results_dir / "iteration_0"
        else:
            raise ValueError(
                f"Invalid results directory structure.\n"
                f"Expected either:\n"
                f"  1. Directory with iteration_*/ folders\n"
                f"  2. Base directory with run_*/ subdirectories\n"
                f"Got: {results_dir}"
            )

    # Create output directory (flat structure)
    output_dataset_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n" + "="*70)
        print("TEXTURE TRANSITION DATASET COLLECTOR")
        print("="*70)
        print(f"Source: {results_dir}")
        print(f"Output: {output_dataset_dir}")
        print("="*70 + "\n")

    # Track statistics
    stats = {
        'total_images_collected': 0,
        'total_images_filtered': 0,
        'by_iteration': {}
    }

    # Step 1: Read iteration_0 classification to identify TEXTURE TRANSITION parent crops
    classification_file = iteration_0_dir / "classification_results.json"

    if not classification_file.exists():
        if verbose:
            print("⚠️  WARNING: No classification file found at iteration_0/")
            print("   Cannot determine which crops are TEXTURE TRANSITION")
            print("   Collecting ALL crops (no filtering)")

        texture_parent_names = None  # Collect everything
        semantic_parent_names = set()
    else:
        with open(classification_file, 'r') as f:
            classifications = json.load(f)

        if not classifications:
            if verbose:
                print("⚠️  WARNING: Classification file is empty!")
                print("   Classifier found 0 TEXTURE TRANSITION crops")
                print("   No images will be collected")

            # Empty classification = nothing to collect
            stats['total_images_collected'] = 0
            stats['total_images_filtered'] = 0
            return stats

        # Build sets of parent crop names
        texture_parent_names = set()
        semantic_parent_names = set()

        for result in classifications:
            crop_name = Path(result['crop_name']).stem  # Without extension
            classification = result['classification'].upper()

            if classification == 'TEXTURE TRANSITION':
                texture_parent_names.add(crop_name)
            elif classification == 'SEMANTIC':
                semantic_parent_names.add(crop_name)

        if verbose:
            print(f"📋 Classification loaded from iteration_0:")
            print(f"   TEXTURE TRANSITION parents: {len(texture_parent_names)}")
            print(f"   SEMANTIC parents: {len(semantic_parent_names)}")
            print()

    # Step 2: Collect from iteration_0 (crops/ directory)
    crops_dir = results_dir / "crops"

    if crops_dir.exists():
        if verbose:
            print("📁 ITERATION 0: Collecting from crops/")

        iter0_stats = _collect_from_iteration_0(
            crops_dir=crops_dir,
            output_dir=output_dataset_dir,
            texture_parents=texture_parent_names,
            semantic_parents=semantic_parent_names,
            copy_filtered=copy_filtered,
            verbose=verbose
        )

        stats['by_iteration']['iteration_0'] = iter0_stats
        stats['total_images_collected'] += iter0_stats['collected']
        stats['total_images_filtered'] += iter0_stats['filtered']

    # Step 3: Collect from iteration 1+ (refined_crops)
    iteration_dirs = sorted([d for d in results_dir.glob("iteration_*") if d.name != "iteration_0"])

    for iteration_dir in iteration_dirs:
        iteration_name = iteration_dir.name

        if verbose:
            print(f"\n📁 {iteration_name.upper()}: Collecting refined crops")

        refined_crops_dir = iteration_dir / "refined_crops"

        if not refined_crops_dir.exists():
            if verbose:
                print(f"   ⚠️  No refined_crops directory, skipping")
            continue

        # Collect all refined crops that descended from TEXTURE TRANSITION parents
        iter_stats = _collect_from_refined_crops(
            refined_crops_dir=refined_crops_dir,
            output_dir=output_dataset_dir,
            texture_parents=texture_parent_names,
            semantic_parents=semantic_parent_names,
            copy_filtered=copy_filtered,
            verbose=verbose
        )

        stats['by_iteration'][iteration_name] = iter_stats
        stats['total_images_collected'] += iter_stats['collected']
        stats['total_images_filtered'] += iter_stats['filtered']

    # Final summary
    if verbose:
        print("\n" + "="*70)
        print("✅ COLLECTION COMPLETE")
        print("="*70)
        print(f"  Total images collected: {stats['total_images_collected']}")
        print(f"  Total images filtered:  {stats['total_images_filtered']}")
        print(f"  Dataset location: {output_dataset_dir.absolute()}")
        print("="*70 + "\n")

    return stats


def _collect_from_iteration_0(
    crops_dir: Path,
    output_dir: Path,
    texture_parents: Set[str] = None,
    semantic_parents: Set[str] = None,
    copy_filtered: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Collect crops from iteration_0 crops/ directory.

    Args:
        crops_dir: crops/ directory with size subdirectories
        output_dir: Output directory (flat structure)
        texture_parents: Set of TEXTURE TRANSITION parent crop names (None = collect all)
        semantic_parents: Set of SEMANTIC parent crop names
        copy_filtered: If True, copy SEMANTIC crops to filtered/ folder
        verbose: Print progress

    Returns:
        Dict with stats
    """
    stats = {'collected': 0, 'filtered': 0}

    # Iterate through all size categories
    for size_dir in crops_dir.iterdir():
        if not size_dir.is_dir():
            continue

        for crop_file in size_dir.glob("*.jpg"):
            crop_name = crop_file.stem

            # Determine if this crop is TEXTURE TRANSITION or SEMANTIC
            if texture_parents is None:
                # No classification, collect everything
                is_texture = True
            else:
                is_texture = crop_name in texture_parents

            is_semantic = semantic_parents and crop_name in semantic_parents

            if is_texture and not is_semantic:
                # Copy to dataset (flat structure)
                output_path = output_dir / crop_file.name
                shutil.copy2(crop_file, output_path)
                stats['collected'] += 1

            elif is_semantic and copy_filtered:
                # Copy to filtered folder (in parent of crops/)
                filtered_dir = crops_dir.parent / "filtered"
                filtered_dir.mkdir(exist_ok=True)
                filtered_path = filtered_dir / crop_file.name
                shutil.copy2(crop_file, filtered_path)
                stats['filtered'] += 1

    if verbose:
        print(f"   → Collected: {stats['collected']} images")
        if stats['filtered'] > 0:
            print(f"   → Filtered:  {stats['filtered']} images to filtered/")

    return stats


def _collect_from_refined_crops(
    refined_crops_dir: Path,
    output_dir: Path,
    texture_parents: Set[str] = None,
    semantic_parents: Set[str] = None,
    copy_filtered: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Collect crops from iteration_N/refined_crops/ directory.

    Structure: refined_crops/PARENT_CROP_NAME/size/image.jpg

    Only collect refined crops whose PARENT is TEXTURE TRANSITION.

    Args:
        refined_crops_dir: refined_crops/ directory
        output_dir: Output directory (flat structure)
        texture_parents: Set of TEXTURE TRANSITION parent names (None = collect all)
        semantic_parents: Set of SEMANTIC parent names
        copy_filtered: If True, copy SEMANTIC descendants to filtered/
        verbose: Print progress

    Returns:
        Dict with stats
    """
    stats = {'collected': 0, 'filtered': 0}

    # Each subdirectory is named after a parent crop
    for parent_dir in refined_crops_dir.iterdir():
        if not parent_dir.is_dir():
            continue

        parent_crop_name = parent_dir.name

        # Determine if parent is TEXTURE TRANSITION or SEMANTIC
        if texture_parents is None:
            # No classification, collect everything
            is_texture_parent = True
        else:
            is_texture_parent = parent_crop_name in texture_parents

        is_semantic_parent = semantic_parents and parent_crop_name in semantic_parents

        # Collect all images from this parent's subdirectories
        for size_dir in parent_dir.iterdir():
            if not size_dir.is_dir():
                continue

            for crop_file in size_dir.glob("*.jpg"):
                if is_texture_parent and not is_semantic_parent:
                    # Copy to dataset (flat structure)
                    output_path = output_dir / crop_file.name
                    shutil.copy2(crop_file, output_path)
                    stats['collected'] += 1

                elif is_semantic_parent and copy_filtered:
                    # Copy to filtered folder
                    filtered_dir = refined_crops_dir.parent.parent / "filtered"
                    filtered_dir.mkdir(exist_ok=True)
                    filtered_path = filtered_dir / crop_file.name
                    shutil.copy2(crop_file, filtered_path)
                    stats['filtered'] += 1

    if verbose:
        print(f"   → Collected: {stats['collected']} images")
        if stats['filtered'] > 0:
            print(f"   → Filtered:  {stats['filtered']} images to filtered/")

    return stats


def create_dataset_from_multiple_runs(
    results_base_dir: Union[str, Path],
    output_dataset_dir: Union[str, Path],
    run_patterns: List[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Collect crops from multiple pipeline runs into one dataset.

    Args:
        results_base_dir: Base directory containing multiple run_* folders
        output_dataset_dir: Output dataset directory
        run_patterns: List of run folder patterns (default: ['run_*'])
        verbose: Print progress

    Returns:
        Dict with combined statistics
    """
    results_base_dir = Path(results_base_dir)
    output_dataset_dir = Path(output_dataset_dir)

    if run_patterns is None:
        run_patterns = ['run_*']

    # Find all run directories
    run_dirs = []
    for pattern in run_patterns:
        run_dirs.extend(results_base_dir.glob(pattern))

    run_dirs = sorted([d for d in run_dirs if d.is_dir()])

    if not run_dirs:
        raise ValueError(f"No run directories found in {results_base_dir}")

    if verbose:
        print("\n" + "="*70)
        print("MULTI-RUN DATASET COLLECTOR")
        print("="*70)
        print(f"Found {len(run_dirs)} run directories")
        print("="*70)

    combined_stats = {
        'runs_processed': 0,
        'total_images': 0,
        'by_run': {}
    }

    for run_dir in run_dirs:
        if verbose:
            print(f"\n🔄 Processing: {run_dir.name}")

        try:
            run_stats = collect_texture_crops(
                results_dir=run_dir,
                output_dataset_dir=output_dataset_dir,
                copy_filtered=True,
                verbose=False
            )

            combined_stats['runs_processed'] += 1
            combined_stats['total_images'] += run_stats['total_images_collected']
            combined_stats['by_run'][run_dir.name] = run_stats

            if verbose:
                print(f"   ✅ Collected {run_stats['total_images_collected']} images")

        except Exception as e:
            if verbose:
                print(f"   ⚠️  Error: {e}")

    if verbose:
        print("\n" + "="*70)
        print("✅ MULTI-RUN COLLECTION COMPLETE")
        print("="*70)
        print(f"  Runs processed: {combined_stats['runs_processed']}")
        print(f"  Total images:   {combined_stats['total_images']}")
        print(f"  Dataset:        {output_dataset_dir.absolute()}")
        print("="*70 + "\n")

    return combined_stats


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect TEXTURE TRANSITION crops from pipeline results into a dataset"
    )

    parser.add_argument(
        'results_dir',
        type=str,
        help='Path to pipeline results directory (e.g., results/run_TIMESTAMP/)'
    )

    parser.add_argument(
        'output_dir',
        type=str,
        help='Path to output dataset directory'
    )

    parser.add_argument(
        '--no-filter',
        action='store_true',
        help="Don't copy filtered (SEMANTIC) crops to 'filtered' folder"
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    parser.add_argument(
        '--multi-run',
        action='store_true',
        help='Collect from multiple run_* folders in results_dir'
    )

    args = parser.parse_args()

    if args.multi_run:
        # Collect from multiple runs
        stats = create_dataset_from_multiple_runs(
            results_base_dir=args.results_dir,
            output_dataset_dir=args.output_dir,
            verbose=not args.quiet
        )
    else:
        # Collect from single run
        stats = collect_texture_crops(
            results_dir=args.results_dir,
            output_dataset_dir=args.output_dir,
            copy_filtered=not args.no_filter,
            verbose=not args.quiet
        )

    print(f"\n✅ Dataset created: {args.output_dir}")
