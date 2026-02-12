#!/usr/bin/env python3
"""
Test script for Texture Curator Foundation.

Run this to verify all modules load correctly.

Usage:
    cd ~/texture_curator
    python test_foundation.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_config():
    """Test configuration module."""
    print("=" * 60)
    print("Testing: config module")
    print("=" * 60)
    
    from config.settings import Config, Phase, MaskStatus
    
    # Create default config
    config = Config()
    
    print(f"✓ Config created successfully")
    print(f"  - RWTD Path: {config.rwtd_path}")
    print(f"  - Source Pool: {config.source_pool_path}")
    print(f"  - Target N: {config.target_n}")
    print(f"  - Device: {config.device}")
    print(f"  - LLM Model: {config.llm.model_name}")
    print(f"  - DINOv2 Model: {config.vision.dino_model}")
    print()
    
    # Test enums
    print(f"✓ Phase enum: {Phase.PROFILING.value}")
    print(f"✓ MaskStatus enum: {MaskStatus.VALID.value}")
    print()
    
    return True


def test_models():
    """Test data models."""
    print("=" * 60)
    print("Testing: state.models module")
    print("=" * 60)
    
    import numpy as np
    from state.models import (
        RWTDProfile,
        CandidateFeatures,
        CandidateRecord,
        ScoreBreakdown,
    )

    # Test RWTDProfile
    profile = RWTDProfile(
        num_samples=256,
        centroid_embedding=np.random.randn(768).astype(np.float32),
    )
    print(f"✓ RWTDProfile created")
    print(f"  - Num samples: {profile.num_samples}")
    print(f"  - Is complete: {profile.is_complete()}")
    print()
    
    # Test CandidateRecord
    candidate = CandidateRecord(
        id="test_001",
        image_path=Path("/test/image.jpg"),
        mask_path=Path("/test/mask.png"),
    )
    print(f"✓ CandidateRecord created")
    print(f"  - ID: {candidate.id}")
    print(f"  - Mask status: {candidate.mask_status.value}")
    print()
    
    return True


def test_graph_state():
    """Test graph state."""
    print("=" * 60)
    print("Testing: state.graph_state module")
    print("=" * 60)
    
    from state.graph_state import GraphState, create_initial_state
    from config.settings import Config
    
    # Create with default config
    state = GraphState()
    
    print(f"✓ GraphState created with default config")
    print()
    print("Status Summary:")
    print("-" * 40)
    print(state.get_status_summary())
    print()
    
    # Test properties
    print(f"✓ Properties work:")
    print(f"  - profile_exists: {state.profile_exists}")
    print(f"  - num_candidates: {state.num_candidates}")
    print(f"  - is_done: {state.is_done}")
    print()
    
    # Test serialization
    state_dict = state.to_dict()
    print(f"✓ Serialization works")
    print(f"  - Keys: {list(state_dict.keys())[:5]}...")
    print()
    
    return True


def test_ollama():
    """Test Ollama connection."""
    print("=" * 60)
    print("Testing: Ollama LLM connection")
    print("=" * 60)
    
    try:
        import httpx
        
        # Check if Ollama is running
        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✓ Ollama is running")
            print(f"  - Available models: {len(models)}")
            for model in models[:5]:
                print(f"    • {model.get('name', 'unknown')}")
            print()
            return True
        else:
            print(f"✗ Ollama responded with status {response.status_code}")
            return False
            
    except httpx.ConnectError:
        print("✗ Cannot connect to Ollama at http://localhost:11434")
        print("  Make sure Ollama is running: ollama serve")
        return False
    except ImportError:
        print("✗ httpx not installed. Run: pip install httpx")
        return False


def main():
    """Run all tests."""
    print()
    print("🧪 TEXTURE CURATOR - FOUNDATION TESTS")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    try:
        results.append(("config", test_config()))
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        results.append(("config", False))
    
    try:
        results.append(("models", test_models()))
    except Exception as e:
        print(f"✗ Models test failed: {e}")
        results.append(("models", False))
    
    try:
        results.append(("graph_state", test_graph_state()))
    except Exception as e:
        print(f"✗ GraphState test failed: {e}")
        results.append(("graph_state", False))
    
    try:
        results.append(("ollama", test_ollama()))
    except Exception as e:
        print(f"✗ Ollama test failed: {e}")
        results.append(("ollama", False))
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All tests passed! Foundation is ready.")
    else:
        print("⚠️  Some tests failed. Please fix before continuing.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)