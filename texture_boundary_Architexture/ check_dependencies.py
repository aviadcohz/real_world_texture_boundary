#!/usr/bin/env python3
"""
Dependency Checker

Run this to check if all required dependencies are installed.
"""

import sys

def check_models():
    """Check if models package works."""
    print("\n" + "="*70)
    print("CHECKING MODELS PACKAGE")
    print("="*70)
    
    try:
        from models.model_factory import check_dependencies,  list_available_models
        
        deps = check_dependencies()
        
        print("\n✓ Models package imported successfully\n")
        print("Dependency Status:")
        print(f"  Qwen:  {'✓ Available' if deps['qwen'] else '✗ Missing (need: torch, transformers, qwen-vl-utils)'}")
        print(f"  LLaVA: {'✓ Available' if deps['llava'] else '✗ Missing'}")
        
        print(f"\nAvailable models: {', '.join(list_available_models()) or 'None'}")
        
        if not deps['qwen']:
            print("\n⚠️  To use Qwen models, install:")
            print("   pip install torch transformers qwen-vl-utils")
        
        return deps['qwen'] or deps['llava']
    
    except Exception as e:
        print(f"\n✗ Error importing models: {e}")
        return False


def check_utils():
    """Check if utils package works."""
    print("\n" + "="*70)
    print("CHECKING UTILS PACKAGE")
    print("="*70)
    
    try:
        from utils import (
            pad_image_to_standard_size,
            parse_json_format,
            calculate_iou,
            get_prompt
        )
        
        print("\n✓ Utils package imported successfully")
        print("  ✓ image_utils")
        print("  ✓ bbox_utils (calculate_iou included)")
        print("  ✓ io_utils")
        
        return True
    
    except Exception as e:
        print(f"\n✗ Error importing utils: {e}")
        return False


def check_core():
    """Check if core package works."""
    print("\n" + "="*70)
    print("CHECKING CORE PACKAGE")
    print("="*70)
    
    try:
        # Note: This will fail if models aren't available
        # because core imports models
        from core import (
            ground_images,
            process_bboxes,
            extract_all_crops,
            visualize_all_images,
            classify_crops
        )
        
        print("\n✓ Core package imported successfully")
        print("  ✓ grounding")
        print("  ✓ bbox_processing")
        print("  ✓ crop_extraction")
        print("  ✓ visualization")
        print("  ✓ classification")
        
        return True
    
    except ImportError as e:
        print(f"\n⚠️  Core package needs models dependencies: {e}")
        print("   This is OK if you'll install torch in your environment")
        return False
    
    except Exception as e:
        print(f"\n✗ Error importing core: {e}")
        return False


def check_config():
    """Check if config works."""
    print("\n" + "="*70)
    print("CHECKING CONFIG")
    print("="*70)
    
    try:
        from config.prompts import get_prompt, list_prompts
        
        prompts = list_prompts()
        
        print("\n✓ Config imported successfully")
        print(f"  Available prompts: {', '.join(prompts)}")
        
        # Test getting a prompt
        grounding = get_prompt('grounding')
        print(f"\n  Grounding prompt: {len(grounding)} characters")
        
        return True
    
    except Exception as e:
        print(f"\n✗ Error loading config: {e}")
        return False


def main():
    """Run all checks."""
    print("\n" + "="*70)
    print("TEXTURE BOUNDARY PIPELINE - DEPENDENCY CHECK")
    print("="*70)
    
    results = {
        'config': check_config(),
        'utils': check_utils(),
        'models': check_models(),
        'core': check_core()
    }
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for package, status in results.items():
        status_str = "✓ OK" if status else "✗ Issues"
        print(f"  {package:10} {status_str}")
    
    if all(results.values()):
        print("\n✅ All checks passed!")
        return 0
    elif results['config'] and results['utils']:
        print("\n⚠️  Some packages need dependencies (install torch/transformers)")
        print("   This is expected if running tests without a GPU environment")
        return 0
    else:
        print("\n❌ Some checks failed - see errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())