"""
Ollama LLM Client for Texture Curator.

This module provides a simple interface to interact with local LLMs
via Ollama. It handles:
- Connection to Ollama server
- Prompt formatting
- JSON response parsing
- Error handling and retries

WHY OLLAMA?
- Free and runs locally
- No API keys needed
- Fast inference on GPU
- Supports many models (Llama, Qwen, Mistral, etc.)

USAGE:
    client = OllamaClient(model="qwen2.5:7b")
    response = client.chat("What is 2+2?")
    json_response = client.chat_json("Return {\"answer\": ...}")
"""

import httpx
import json
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Setup logging
logger = logging.getLogger(__name__)


# ============================================
# Data Structures
# ============================================

@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str
    
    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    """Response from the LLM."""
    content: str
    model: str
    created_at: datetime = field(default_factory=datetime.now)
    
    # Token usage (if available)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Timing
    total_duration_ms: float = 0
    
    # Raw response for debugging
    raw_response: Optional[dict] = None
    
    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "model": self.model,
            "created_at": self.created_at.isoformat(),
            "tokens": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
                "total": self.total_tokens,
            },
            "duration_ms": self.total_duration_ms,
        }


# ============================================
# Ollama Client
# ============================================

class OllamaClient:
    """
    Client for interacting with Ollama LLM server.
    
    Ollama runs locally and provides an OpenAI-compatible API.
    Default endpoint: http://localhost:11434
    """
    
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 120,
    ):
        """
        Initialize Ollama client.
        
        Args:
            model: Model name (e.g., "qwen2.5:7b", "llama3.1:8b")
            base_url: Ollama server URL
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # HTTP client
        self.client = httpx.Client(timeout=timeout)
        
        logger.info(f"OllamaClient initialized: model={model}, url={base_url}")
    
    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """List available models."""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
            return []
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Message]] = None,
    ) -> ChatResponse:
        """
        Send a chat message and get a response.
        
        Args:
            prompt: User message
            system_prompt: Optional system prompt
            conversation_history: Optional list of previous messages
        
        Returns:
            ChatResponse object
        """
        # Build messages list
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if conversation_history:
            for msg in conversation_history:
                messages.append(msg.to_dict())
        
        messages.append({"role": "user", "content": prompt})
        
        # Make request
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }
        
        try:
            response = self.client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            content = data.get("message", {}).get("content", "")
            
            return ChatResponse(
                content=content,
                model=data.get("model", self.model),
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                total_duration_ms=data.get("total_duration", 0) / 1_000_000,  # ns to ms
                raw_response=data,
            )
            
        except httpx.TimeoutException:
            logger.error(f"Request timed out after {self.timeout}s")
            raise TimeoutError(f"Ollama request timed out after {self.timeout}s")
        
        except Exception as e:
            logger.error(f"Chat request failed: {e}")
            raise
    
    def chat_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Message]] = None,
        retry_on_parse_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Send a chat message and parse JSON response.
        
        The LLM is expected to return valid JSON. If parsing fails,
        this will retry with a clarifying prompt.
        
        Args:
            prompt: User message (should ask for JSON output)
            system_prompt: Optional system prompt
            conversation_history: Optional conversation history
            retry_on_parse_error: Whether to retry if JSON parsing fails
        
        Returns:
            Parsed JSON as dictionary
        """
        # Add JSON instruction to system prompt if not present
        if system_prompt and "JSON" not in system_prompt.upper():
            system_prompt += "\n\nIMPORTANT: Always respond with valid JSON only. No markdown, no explanation."
        elif not system_prompt:
            system_prompt = "You are a helpful assistant. Always respond with valid JSON only. No markdown, no explanation."
        
        response = self.chat(prompt, system_prompt, conversation_history)
        
        # Try to parse JSON
        try:
            return self._parse_json(response.content)
        
        except json.JSONDecodeError as e:
            if retry_on_parse_error:
                logger.warning(f"JSON parse failed, retrying: {e}")
                
                # Retry with explicit instruction
                retry_prompt = f"""Your previous response was not valid JSON. 
Please respond with ONLY valid JSON, no markdown code blocks, no explanation.

Original request: {prompt}

Respond with valid JSON:"""
                
                retry_response = self.chat(retry_prompt, system_prompt, conversation_history)
                return self._parse_json(retry_response.content)
            else:
                raise
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response.
        
        Handles common issues like:
        - Markdown code blocks
        - Leading/trailing whitespace
        - Explanation text before/after JSON
        """
        # Clean the text
        text = text.strip()
        
        # Remove markdown code blocks
        if text.startswith("```"):
            # Find the end of the code block
            lines = text.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.startswith("```") and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            text = "\n".join(json_lines)
        
        # Try to find JSON object in text
        # Look for { ... } pattern
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            text = match.group()
        
        # Parse
        return json.loads(text)
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Simple text generation (non-chat mode).
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt (prepended to prompt)
        
        Returns:
            Generated text
        """
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }
        
        response = self.client.post(
            f"{self.base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        
        return data.get("response", "")
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================
# Quick Test
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Ollama LLM Client Test")
    print("=" * 60)
    
    # Create client
    client = OllamaClient(model="qwen2.5:7b")
    
    # Check availability
    print(f"\nOllama available: {client.is_available()}")
    
    if not client.is_available():
        print("✗ Ollama is not running. Start with: ollama serve")
        exit(1)
    
    # List models
    models = client.list_models()
    print(f"Available models: {models}")
    
    if client.model not in [m.split(":")[0] for m in models] and client.model not in models:
        print(f"✗ Model {client.model} not found. Pull with: ollama pull {client.model}")
        exit(1)
    
    # Test simple chat
    print("\n--- Test 1: Simple Chat ---")
    response = client.chat("What is 2 + 2? Answer in one word.")
    print(f"Response: {response.content}")
    print(f"Tokens: {response.total_tokens}")
    print(f"Duration: {response.total_duration_ms:.0f}ms")
    
    # Test JSON response
    print("\n--- Test 2: JSON Response ---")
    json_response = client.chat_json(
        'What are the RGB values for the color "sky blue"? Return as {"r": int, "g": int, "b": int}'
    )
    print(f"JSON Response: {json_response}")
    
    # Test with system prompt
    print("\n--- Test 3: System Prompt ---")
    response = client.chat(
        "What should I do next?",
        system_prompt="You are a helpful cooking assistant. The user is making pasta."
    )
    print(f"Response: {response.content[:200]}...")
    
    # Test agent-style JSON
    print("\n--- Test 4: Agent Decision ---")
    agent_response = client.chat_json(
        """Current state: Profile not built, 0 candidates scored.
        
What should we do next? Choose from: profile_rwtd, score_batch, validate_mask, select_diverse, done

Respond with JSON: {"reasoning": "...", "action": "...", "params": {}}""",
        system_prompt="You are a dataset curation agent. Always respond with valid JSON."
    )
    print(f"Agent Decision: {json.dumps(agent_response, indent=2)}")
    
    print("\n✅ Ollama Client test passed!")