"""
LLM module for Texture Curator.

Provides interfaces to local LLMs via Ollama.
"""


from llm.ollama_client import OllamaClient, Message, ChatResponse


__all__ = [
    "OllamaClient",
    "Message", 
    "ChatResponse",
]