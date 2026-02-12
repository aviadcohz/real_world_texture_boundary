"""
Base Agent Class for Texture Curator.

This module defines the abstract base class for all agents in the system.
Each agent (Planner, Profiler, Analyst, Critic, Optimizer) inherits from this.

AGENT ARCHITECTURE:
- Each agent has a specific ROLE and SYSTEM PROMPT
- Agents read from GraphState and produce ACTIONS
- Actions are either tool calls or state updates
- The orchestrator executes actions and updates state

USAGE:
    class ProfilerAgent(BaseAgent):
        def __init__(self, llm_client):
            super().__init__(
                name="profiler",
                llm_client=llm_client,
                system_prompt="You are the Profiler agent..."
            )
        
        def get_available_tools(self):
            return ["extract_dino", "compute_texture_stats", ...]
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging

# Handle imports
try:
    from state.graph_state import GraphState
    from llm.ollama_client import OllamaClient, Message
except ImportError:
    from ..state.graph_state import GraphState
    from .ollama_client import OllamaClient, Message

logger = logging.getLogger(__name__)


# ============================================
# Action Data Structures
# ============================================

@dataclass
class AgentAction:
    """
    An action decided by an agent.
    
    This represents what the agent wants to do next.
    The orchestrator will execute this action.
    """
    
    # The tool to call (or "done" / "error")
    tool_name: str
    
    # Parameters to pass to the tool
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Agent's reasoning for this action
    reasoning: str = ""
    
    # Which agent produced this action
    agent_name: str = ""
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "params": self.params,
            "reasoning": self.reasoning,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp.isoformat(),
        }
    
    @classmethod
    def done(cls, agent_name: str, reasoning: str = "Task complete") -> "AgentAction":
        """Create a 'done' action."""
        return cls(
            tool_name="done",
            reasoning=reasoning,
            agent_name=agent_name,
        )
    
    @classmethod
    def error(cls, agent_name: str, error_message: str) -> "AgentAction":
        """Create an 'error' action."""
        return cls(
            tool_name="error",
            reasoning=error_message,
            agent_name=agent_name,
        )


@dataclass
class ToolResult:
    """
    Result of executing a tool.
    
    Returned by tools after execution.
    """
    
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error_message,
        }
    
    def summary(self, max_length: int = 200) -> str:
        """Get a short summary for LLM context."""
        if not self.success:
            return f"FAILED: {self.error_message}"
        
        # Summarize data
        summary_parts = []
        for key, value in self.data.items():
            if isinstance(value, (list, dict)):
                summary_parts.append(f"{key}: {len(value)} items")
            else:
                summary_parts.append(f"{key}: {value}")
        
        summary = ", ".join(summary_parts)
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return f"SUCCESS: {summary}"


# ============================================
# Base Agent Class
# ============================================

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Each agent in the system:
    1. Has a specific role (defined by system prompt)
    2. Can read the current state
    3. Decides on actions (tool calls)
    4. Returns structured action objects
    """
    
    def __init__(
        self,
        name: str,
        llm_client: OllamaClient,
        system_prompt: str,
        max_history: int = 10,
    ):
        """
        Initialize base agent.
        
        Args:
            name: Agent name (e.g., "planner", "profiler")
            llm_client: Ollama client for LLM calls
            system_prompt: The agent's role and instructions
            max_history: Maximum conversation history to keep
        """
        self.name = name
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.max_history = max_history
        
        # Conversation history for this agent
        self.conversation_history: List[Message] = []
        
        logger.info(f"Agent initialized: {name}")
    
    @abstractmethod
    def get_available_tools(self) -> List[str]:
        """
        Return list of tools this agent can use.
        
        Each agent has access to specific tools.
        """
        pass
    
    def format_state_for_prompt(self, state: GraphState) -> str:
        """
        Format the current state for the LLM prompt.
        
        Override this in subclasses for agent-specific formatting.
        """
        return state.get_status_summary()
    
    def format_tools_for_prompt(self) -> str:
        """Format available tools for the prompt."""
        tools = self.get_available_tools()
        return "Available tools: " + ", ".join(tools)
    
    def build_prompt(self, state: GraphState, additional_context: str = "") -> str:
        """
        Build the full prompt for the LLM.
        
        Args:
            state: Current graph state
            additional_context: Optional additional context
        
        Returns:
            Complete prompt string
        """
        parts = [
            "## Current State",
            self.format_state_for_prompt(state),
            "",
            "## " + self.format_tools_for_prompt(),
            "",
        ]
        
        if additional_context:
            parts.extend([
                "## Additional Context",
                additional_context,
                "",
            ])
        
        # Add recent actions from state
        if state.message_history:
            recent = state.message_history[-5:]
            parts.append("## Recent Actions")
            for msg in recent:
                if msg.role == "assistant":
                    parts.append(f"[{msg.agent}] {msg.content[:100]}...")
            parts.append("")
        
        parts.extend([
            "## Your Task",
            "Analyze the current state and decide what action to take next.",
            "",
            "Respond with JSON:",
            '{"reasoning": "your step-by-step thinking", "tool": "tool_name", "params": {}}',
        ])
        
        return "\n".join(parts)
    
    def decide(self, state: GraphState, additional_context: str = "") -> AgentAction:
        """
        Decide on the next action based on current state.
        
        This is the main method that orchestrator calls.
        
        Args:
            state: Current graph state
            additional_context: Optional context (e.g., last tool result)
        
        Returns:
            AgentAction to be executed
        """
        # Build prompt
        prompt = self.build_prompt(state, additional_context)
        
        # Add to history
        self.conversation_history.append(Message(role="user", content=prompt))
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
        
        try:
            # Get LLM response
            response = self.llm_client.chat_json(
                prompt=prompt,
                system_prompt=self.system_prompt,
            )
            
            # Add response to history
            self.conversation_history.append(
                Message(role="assistant", content=json.dumps(response))
            )
            
            # Parse action
            action = self._parse_response(response)
            action.agent_name = self.name
            
            # Log to state
            state.add_message(
                agent=self.name,
                role="assistant",
                content=json.dumps(response),
            )
            
            logger.info(f"[{self.name}] Action: {action.tool_name}")
            return action
            
        except Exception as e:
            logger.error(f"[{self.name}] Decision failed: {e}")
            return AgentAction.error(self.name, str(e))
    
    def _parse_response(self, response: Dict[str, Any]) -> AgentAction:
        """
        Parse LLM response into AgentAction.
        
        Expected format:
        {
            "reasoning": "...",
            "tool": "tool_name",
            "params": {}
        }
        """
        # Handle different response formats
        tool_name = response.get("tool") or response.get("action") or response.get("tool_name")
        params = response.get("params") or response.get("parameters") or {}
        reasoning = response.get("reasoning") or response.get("thought") or ""
        
        if not tool_name:
            raise ValueError(f"No tool specified in response: {response}")
        
        # Validate tool
        available_tools = self.get_available_tools() + ["done", "error"]
        if tool_name not in available_tools:
            logger.warning(f"Unknown tool: {tool_name}. Available: {available_tools}")
        
        return AgentAction(
            tool_name=tool_name,
            params=params,
            reasoning=reasoning,
        )
    
    def reset_history(self):
        """Clear conversation history."""
        self.conversation_history = []


# ============================================
# Tool Registry (for reference)
# ============================================

# This maps tool names to their descriptions
# Used by agents to understand what each tool does

TOOL_DESCRIPTIONS = {
    # Profiling tools
    "profile_rwtd": "Build RWTD centroid from DINOv2 embeddings",

    # Feature extraction
    "extract_features": "Extract DINOv2 embeddings for candidates",

    # Scoring
    "score_candidates": "Score candidates by cosine similarity to centroid",
    "get_top_candidates": "Get top N candidates by score",

    # Selection
    "select_diverse": "Select diverse top-N using coreset selection",

    # Utility
    "get_status": "Get current state summary",

    # Control
    "done": "Mark task as complete",
    "route_to": "Route to another agent",
}


def get_tool_descriptions(tools: List[str]) -> str:
    """Get formatted descriptions for a list of tools."""
    lines = []
    for tool in tools:
        desc = TOOL_DESCRIPTIONS.get(tool, "No description available")
        lines.append(f"- {tool}: {desc}")
    return "\n".join(lines)


# ============================================
# Quick Test
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Base Agent Test")
    print("=" * 60)
    
    # Create a simple test agent
    class TestAgent(BaseAgent):
        def get_available_tools(self):
            return ["profile_rwtd", "score_candidates", "done"]
    
    # Check if Ollama is available
    from llm.ollama_client import OllamaClient
    
    client = OllamaClient(model="qwen2.5:7b")
    if not client.is_available():
        print("✗ Ollama not available. Start with: ollama serve")
        exit(1)
    
    print("✓ Ollama available")
    
    # Create test agent
    agent = TestAgent(
        name="test_agent",
        llm_client=client,
        system_prompt="You are a test agent. Choose an appropriate tool based on the state."
    )
    
    print(f"✓ Agent created: {agent.name}")
    print(f"  Tools: {agent.get_available_tools()}")
    
    # Create a mock state
    from state.graph_state import GraphState
    state = GraphState()
    
    print(f"\n--- Testing Decision ---")
    print(f"State: profile_exists={state.profile_exists}, num_candidates={state.num_candidates}")
    
    # Get decision
    action = agent.decide(state)
    
    print(f"\nAgent Decision:")
    print(f"  Tool: {action.tool_name}")
    print(f"  Params: {action.params}")
    print(f"  Reasoning: {action.reasoning}")
    
    print("\n✅ Base Agent test passed!")