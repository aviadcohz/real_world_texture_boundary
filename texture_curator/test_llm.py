#!/usr/bin/env python3
"""
Test script for Texture Curator - Phase 3: LLM Client & Agents.

Tests:
1. Ollama connection and availability
2. Basic chat functionality
3. JSON response parsing
4. Base agent decision making

Usage:
    cd ~/texture_curator
    python test_llm.py
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_ollama_connection():
    """Test basic Ollama connection."""
    print("=" * 60)
    print("Testing: Ollama Connection")
    print("=" * 60)
    
    from llm.ollama_client import OllamaClient
    
    client = OllamaClient(model="qwen2.5:7b")
    
    # Check availability
    print(f"\nChecking Ollama at {client.base_url}...")
    available = client.is_available()
    
    if not available:
        print("✗ Ollama is NOT running!")
        print("  Start with: ollama serve")
        print("  Or check if service is running: sudo systemctl status ollama")
        return False
    
    print("✓ Ollama is running")
    
    # List models
    models = client.list_models()
    print(f"\nAvailable models ({len(models)}):")
    for m in models[:5]:
        print(f"  • {m}")
    
    # Check if our model is available
    model_names = [m.split(":")[0] for m in models]
    if "qwen2.5" in model_names or "qwen2.5:7b" in models:
        print(f"\n✓ Target model available: qwen2.5:7b")
    else:
        print(f"\n⚠ Model qwen2.5:7b not found. Available: {models}")
        print("  Pull with: ollama pull qwen2.5:7b")
        return False
    
    print("\n✅ Ollama Connection: PASSED")
    return True


def test_basic_chat():
    """Test basic chat functionality."""
    print("\n" + "=" * 60)
    print("Testing: Basic Chat")
    print("=" * 60)
    
    from llm.ollama_client import OllamaClient
    
    client = OllamaClient(model="qwen2.5:7b")
    
    # Simple question
    print("\nTest 1: Simple question")
    response = client.chat("What is 2 + 2? Answer with just the number.")
    print(f"  Q: What is 2 + 2?")
    print(f"  A: {response.content.strip()}")
    print(f"  Tokens: {response.total_tokens}, Time: {response.total_duration_ms:.0f}ms")
    
    # With system prompt
    print("\nTest 2: With system prompt")
    response = client.chat(
        "What should I add next?",
        system_prompt="You are a cooking assistant helping make pasta. Keep responses brief."
    )
    print(f"  System: Cooking assistant for pasta")
    print(f"  Q: What should I add next?")
    print(f"  A: {response.content[:150]}...")
    
    print("\n✅ Basic Chat: PASSED")
    return True


def test_json_response():
    """Test JSON response parsing."""
    print("\n" + "=" * 60)
    print("Testing: JSON Response Parsing")
    print("=" * 60)
    
    from llm.ollama_client import OllamaClient
    
    client = OllamaClient(model="qwen2.5:7b")
    
    # Simple JSON
    print("\nTest 1: Simple JSON")
    response = client.chat_json(
        'What are RGB values for red? Respond with {"r": int, "g": int, "b": int}'
    )
    print(f"  Response: {response}")
    assert "r" in response, "Missing 'r' key"
    print("  ✓ Parsed successfully")
    
    # Complex JSON
    print("\nTest 2: Complex JSON (agent-style)")
    response = client.chat_json(
        """You need to analyze data. Your options are:
        - analyze: Start analysis
        - wait: Wait for more data
        - done: Finish
        
        Current state: No data loaded yet.
        
        Respond with: {"reasoning": "your thinking", "action": "chosen_action", "params": {}}"""
    )
    print(f"  Response: {json.dumps(response, indent=2)}")
    assert "action" in response or "tool" in response, "Missing action key"
    print("  ✓ Parsed successfully")
    
    print("\n✅ JSON Response: PASSED")
    return True


def test_base_agent():
    """Test base agent decision making."""
    print("\n" + "=" * 60)
    print("Testing: Base Agent")
    print("=" * 60)
    
    from llm.ollama_client import OllamaClient
    from agents.base import BaseAgent, AgentAction
    from state.graph_state import GraphState
    
    # Create a simple test agent
    class TestAgent(BaseAgent):
        def get_available_tools(self):
            return ["profile_rwtd", "score_candidates", "select_diverse", "done"]
    
    client = OllamaClient(model="qwen2.5:7b")
    
    agent = TestAgent(
        name="test_agent",
        llm_client=client,
        system_prompt="""You are a dataset curation agent.
        
Your goal is to curate a high-quality dataset by:
1. First building a profile (profile_rwtd)
2. Then scoring candidates (score_candidates)
3. Finally selecting diverse samples (select_diverse)

Based on the current state, decide which tool to use next.
Always respond with valid JSON: {"reasoning": "...", "tool": "...", "params": {}}"""
    )
    
    print(f"\nAgent: {agent.name}")
    print(f"Tools: {agent.get_available_tools()}")
    
    # Test 1: Initial state (no profile)
    print("\n--- Test 1: Initial State (no profile) ---")
    state = GraphState()
    print(f"State: profile={state.profile_exists}, candidates={state.num_candidates}")
    
    action = agent.decide(state)
    print(f"Decision:")
    print(f"  Tool: {action.tool_name}")
    print(f"  Reasoning: {action.reasoning[:100]}...")
    
    # The agent should choose to build profile first
    expected_first = ["profile_rwtd", "profile"]
    if any(exp in action.tool_name.lower() for exp in expected_first):
        print("  ✓ Correctly chose to profile first")
    else:
        print(f"  ⚠ Expected profile_rwtd, got {action.tool_name}")
    
    # Test 2: After profiling
    print("\n--- Test 2: After Profiling (mock profile exists) ---")
    
    # Mock: pretend profile exists
    from state.models import RWTDProfile
    import numpy as np

    state.rwtd_profile = RWTDProfile(
        num_samples=256,
        centroid_embedding=np.random.randn(768).astype(np.float32),
    )
    
    print(f"State: profile={state.profile_exists}, candidates={state.num_candidates}")
    
    # Reset agent history for fresh decision
    agent.reset_history()
    action = agent.decide(state)
    
    print(f"Decision:")
    print(f"  Tool: {action.tool_name}")
    print(f"  Reasoning: {action.reasoning[:100]}...")
    
    print("\n✅ Base Agent: PASSED")
    return True


def test_agent_error_handling():
    """Test agent handles errors gracefully."""
    print("\n" + "=" * 60)
    print("Testing: Agent Error Handling")
    print("=" * 60)
    
    from llm.ollama_client import OllamaClient
    from agents.base import BaseAgent
    from state.graph_state import GraphState
    
    class BadAgent(BaseAgent):
        def get_available_tools(self):
            return ["tool1"]
        
        def build_prompt(self, state, additional_context=""):
            # Return a prompt that might cause issues
            return "Return invalid JSON: {broken"
    
    client = OllamaClient(model="qwen2.5:7b")
    
    agent = BadAgent(
        name="bad_agent",
        llm_client=client,
        system_prompt="You are a test agent."
    )
    
    state = GraphState()
    
    # This should not crash, but return an error action
    try:
        action = agent.decide(state)
        print(f"Action type: {action.tool_name}")
        # If we get here without exception, error was handled
        print("✓ Error handled gracefully")
    except Exception as e:
        print(f"✗ Unhandled exception: {e}")
        return False
    
    print("\n✅ Error Handling: PASSED")
    return True


def main():
    print("\n🧪 TEXTURE CURATOR - PHASE 3: LLM CLIENT & AGENTS")
    print("=" * 60)
    
    results = []
    
    # Test 1: Ollama Connection
    try:
        results.append(("Ollama Connection", test_ollama_connection()))
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Ollama Connection", False))
        # If connection fails, skip other tests
        print("\n⚠ Skipping remaining tests (Ollama not available)")
        return False
    
    # Test 2: Basic Chat
    try:
        results.append(("Basic Chat", test_basic_chat()))
    except Exception as e:
        print(f"✗ Chat test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Basic Chat", False))
    
    # Test 3: JSON Response
    try:
        results.append(("JSON Response", test_json_response()))
    except Exception as e:
        print(f"✗ JSON test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("JSON Response", False))
    
    # Test 4: Base Agent
    try:
        results.append(("Base Agent", test_base_agent()))
    except Exception as e:
        print(f"✗ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Base Agent", False))
    
    # Test 5: Error Handling
    try:
        results.append(("Error Handling", test_agent_error_handling()))
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Error Handling", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All Phase 3 tests passed! LLM Client & Agents ready.")
        print("\nNext step: Phase 4 - Implement specific agents (Profiler, Analyst, etc.)")
    else:
        print("⚠️  Some tests failed. Please fix before continuing.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)