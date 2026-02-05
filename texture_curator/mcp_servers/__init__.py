"""
MCP Servers for Texture Curator.

This package contains tool servers for the multi-agent system:
- vision: Feature extraction (DINOv2, texture stats, boundary metrics)
- vectordb: Vector database operations (FAISS-GPU, embedding store)
- optimization: Selection algorithms (coreset, diversity sampling)
"""

# Vision Module
from .vision import (
    DINOv2Extractor,
    TextureStatsExtractor,
    BoundaryMetricsExtractor,
    TextureStats,
    BoundaryMetrics,
)

# VectorDB Module
from .vectordb import (
    FAISSVectorDB,
    EmbeddingStore,
    EmbeddingMetadata,
    SearchResult,
    create_vector_db,
    HAS_FAISS,
    HAS_GPU,
)

# Optimization Module
from .optimization import (
    CoresetSelector,
    SelectionMethod,
    SelectionResult,
)

__all__ = [
    # Vision
    "DINOv2Extractor",
    "TextureStatsExtractor",
    "BoundaryMetricsExtractor",
    "TextureStats",
    "BoundaryMetrics",
    # VectorDB
    "FAISSVectorDB",
    "EmbeddingStore",
    "EmbeddingMetadata",
    "SearchResult",
    "create_vector_db",
    "HAS_FAISS",
    "HAS_GPU",
    # Optimization
    "CoresetSelector",
    "SelectionMethod",
    "SelectionResult",
]