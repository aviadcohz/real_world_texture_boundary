"""
VectorDB Module - FAISS-GPU Vector Database.

Provides scalable similarity search for large-scale datasets.
"""

from .faiss_index import (
    FAISSVectorDB,
    SearchResult,
    create_vector_db,
    HAS_FAISS,
    HAS_GPU,
)

from .embedding_store import (
    EmbeddingStore,
    EmbeddingMetadata,
)

__all__ = [
    "FAISSVectorDB",
    "SearchResult",
    "create_vector_db",
    "HAS_FAISS",
    "HAS_GPU",
    "EmbeddingStore",
    "EmbeddingMetadata",
]