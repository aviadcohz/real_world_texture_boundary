"""
Diagnostic script to check results directory structure.

This helps verify your results are in the correct format for dataset collection.
"""

from pathlib import Path
import json


def check_results_structure(results_path: str):
    """Check if results directory has the expected structure."""
    
    results_dir = Path(results_path)
    
    if not results_dir.exists():
        print(f"❌ Directory not found: {results_dir}")
        return False
    
    print("="*70)
    print("CHECKING RESULTS STRUCTURE")
    print("="*70)
    print(f"Directory: {results_dir.absolute()}\n")
    
    # Check if this is a base directory with run_* subdirectories
    run_dirs = sorted(results_dir.glob("run_*"))
    
    if run_dirs:
        print(f"✅ Found {len(run_dirs)} run directory(ies):")
        for run_dir in run_dirs:
            print(f"   - {run_dir.name}")
        print(f"\n📌 Will use: {run_dirs[-1].name} (latest)")
        
        # Use the latest run for checking
        check_dir = run_dirs[-1]
    else:
        print("📌 Checking current directory directly")
        check_dir = results_dir
    
    print(f"\n{'='*70}")
    print(f"CHECKING: {check_dir.name}")
    print(f"{'='*70}\n")
    
    # Check for required directories
    crops_dir = check_dir / "crops"
    iteration_0_dir = check_dir / "iteration_0"
    
    issues = []
    
    # Check crops/
    if crops_dir.exists():
        print("✅ Found crops/ directory")
        
        # Count crops by size
        size_dirs = ['tiny', 'small', 'medium', 'large', 'xlarge']
        for size in size_dirs:
            size_path = crops_dir / size
            if size_path.exists():
                count = len(list(size_path.glob("*.jpg")))
                print(f"   {size:8}: {count:3} crops")
    else:
        print("❌ Missing crops/ directory")
        issues.append("crops/ directory not found")
    
    # Check iteration_0/
    print()
    if iteration_0_dir.exists():
        print("✅ Found iteration_0/ directory")
        
        # Check classification file
        classification_file = iteration_0_dir / "classification_results.json"
        if classification_file.exists():
            print(f"   ✅ Found classification_results.json")
            
            # Load and check
            with open(classification_file) as f:
                classifications = json.load(f)
            
            texture_count = sum(1 for c in classifications if c['classification'].upper() == 'TEXTURE TRANSITION')
            semantic_count = sum(1 for c in classifications if c['classification'].upper() == 'SEMANTIC')
            
            print(f"      TEXTURE TRANSITION: {texture_count}")
            print(f"      SEMANTIC: {semantic_count}")
        else:
            print(f"   ⚠️  Missing classification_results.json")
            issues.append("iteration_0/classification_results.json not found")
    else:
        print("⚠️  Missing iteration_0/ directory (might not have run with new version)")
    
    # Check iteration_1+
    print()
    iteration_dirs = sorted(check_dir.glob("iteration_*"))
    iteration_dirs = [d for d in iteration_dirs if d.name != "iteration_0"]
    
    if iteration_dirs:
        print(f"✅ Found {len(iteration_dirs)} refinement iteration(s):")
        
        for iteration_dir in iteration_dirs:
            print(f"\n   {iteration_dir.name}:")
            
            # Check refined_crops/
            refined_crops_dir = iteration_dir / "refined_crops"
            if refined_crops_dir.exists():
                crop_folders = [d for d in refined_crops_dir.iterdir() if d.is_dir()]
                total_crops = sum(1 for _ in refined_crops_dir.rglob("*.jpg"))
                print(f"      ✅ refined_crops/: {len(crop_folders)} folders, {total_crops} crops")
            else:
                print(f"      ❌ Missing refined_crops/")
                issues.append(f"{iteration_dir.name}/refined_crops not found")
            
            # Check classification
            classification_file = iteration_dir / "classification_results.json"
            if classification_file.exists():
                with open(classification_file) as f:
                    classifications = json.load(f)
                
                texture_count = sum(1 for c in classifications if c['classification'].upper() == 'TEXTURE TRANSITION')
                semantic_count = sum(1 for c in classifications if c['classification'].upper() == 'SEMANTIC')
                
                print(f"      ✅ classification_results.json:")
                print(f"         TEXTURE TRANSITION: {texture_count}")
                print(f"         SEMANTIC: {semantic_count}")
            else:
                print(f"      ⚠️  Missing classification_results.json")
                issues.append(f"{iteration_dir.name}/classification_results.json not found")
    else:
        print("📌 No refinement iterations found (only iteration 0 ran)")
    
    # Summary
    print(f"\n{'='*70}")
    if issues:
        print("⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        print(f"\nNote: Some issues might be expected if using old pipeline version")
    else:
        print("✅ STRUCTURE LOOKS GOOD!")
        print("   Ready for dataset collection!")
    print(f"{'='*70}\n")
    
    return len(issues) == 0


if __name__ == "__main__":

    results_path = "/home/aviad/check_structure"
    check_results_structure(results_path)