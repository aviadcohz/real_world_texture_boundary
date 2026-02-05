"""
Embedding Store for Texture Curator.

Combines raw embedding storage with FAISS index for:
- Fast similarity search
- Embedding reconstruction (get original vectors back)
- Metadata storage per embedding
- Persistent save/load

USAGE:
    store = EmbeddingStore(dimension=768)
    store.add_batch(embeddings, ids, metadata_list)
    
    # Search
    results = store.search(query, k=10)
    
    # Get original embedding
    emb = store.get_embedding("img_001")
    
    # Compute pairwise similarity
    sim_matrix = store.compute_similarity_matrix()
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import logging
import json
import pickle

logger = logging.getLogger(__name__)

# Import FAISS index
from .faiss_index import FAISSVectorDB, SearchResult, HAS_FAISS, HAS_GPU


@dataclass
class EmbeddingMetadata:
    """Metadata for a single embedding."""
    id: str
    source_path: Optional[str] = None
    quality_score: float = 0.0
    semantic_score: float = 0.0
    texture_score: float = 0.0
    boundary_score: float = 0.0
    is_valid: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_path": self.source_path,
            "quality_score": self.quality_score,
            "semantic_score": self.semantic_score,
            "texture_score": self.texture_score,
            "boundary_score": self.boundary_score,
            "is_valid": self.is_valid,
            "extra": self.extra,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "EmbeddingMetadata":
        return cls(**d)


class EmbeddingStore:
    """
    Combined embedding storage with FAISS index.
    
    Stores both the raw embeddings and the FAISS index,
    enabling both fast search and reconstruction.
    
    Features:
    - GPU-accelerated similarity search via FAISS
    - Raw embedding storage for reconstruction
    - Per-embedding metadata storage
    - Persistent save/load to disk
    - Batch operations for efficiency
    """
    
    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "flat",
        use_gpu: bool = True,
        normalize: bool = True,
    ):
        """
        Initialize embedding store.
        
        Args:
            dimension: Embedding dimension (768 for DINOv2-base)
            index_type: FAISS index type ("flat", "ivf", "hnsw")
            use_gpu: Whether to use GPU acceleration
            normalize: Whether to normalize embeddings for cosine similarity
        """
        self.dimension = dimension
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.normalize = normalize
        
        # FAISS index for fast search
        self.faiss_index = FAISSVectorDB(
            dimension=dimension,
            index_type=index_type,
            use_gpu=use_gpu,
            normalize=normalize,
        )
        
        # Raw embedding storage (id -> embedding)
        self.embeddings: Dict[str, np.ndarray] = {}
        
        # Ordered list of IDs (for matrix operations)
        self.id_list: List[str] = []
        
        # Metadata storage (id -> metadata)
        self.metadata: Dict[str, EmbeddingMetadata] = {}
        
        logger.info(f"EmbeddingStore initialized: dim={dimension}, type={index_type}, gpu={use_gpu}")
    
    def add(
        self,
        embedding: np.ndarray,
        id: str,
        metadata: EmbeddingMetadata = None,
    ) -> None:
        """
        Add a single embedding.
        
        Args:
            embedding: Vector of shape (dimension,)
            id: String identifier
            metadata: Optional metadata
        """
        self.add_batch(
            embeddings=embedding.reshape(1, -1),
            ids=[id],
            metadata_list=[metadata] if metadata else None,
        )
    
    def add_batch(
        self,
        embeddings: np.ndarray,
        ids: List[str],
        metadata_list: List[EmbeddingMetadata] = None,
    ) -> int:
        """
        Add multiple embeddings.
        
        Args:
            embeddings: Array of shape (N, dimension)
            ids: List of N string identifiers
            metadata_list: Optional list of N metadata objects
        
        Returns:
            Number of embeddings added
        """
        if len(embeddings) != len(ids):
            raise ValueError(f"Embeddings ({len(embeddings)}) and IDs ({len(ids)}) must match")
        
        # Store raw embeddings
        for i, (emb, id_) in enumerate(zip(embeddings, ids)):
            self.embeddings[id_] = emb.copy().astype(np.float32)
            
            if id_ not in self.id_list:
                self.id_list.append(id_)
            
            # Store metadata
            if metadata_list and i < len(metadata_list) and metadata_list[i]:
                self.metadata[id_] = metadata_list[i]
            elif id_ not in self.metadata:
                self.metadata[id_] = EmbeddingMetadata(id=id_)
        
        # Add to FAISS index
        self.faiss_index.add_batch(embeddings, ids)
        
        logger.debug(f"Added {len(embeddings)} embeddings. Total: {len(self.embeddings)}")
        
        return len(embeddings)
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        filter_valid: bool = False,
    ) -> List[SearchResult]:
        """
        Search for similar embeddings.
        
        Args:
            query: Query vector of shape (dimension,)
            k: Number of results
            filter_valid: If True, only return valid embeddings
        
        Returns:
            List of SearchResult objects
        """
        results = self.faiss_index.search(query, k=k * 2 if filter_valid else k)
        
        if filter_valid:
            results = [r for r in results if self.metadata.get(r.id, EmbeddingMetadata(id=r.id)).is_valid]
            results = results[:k]
        
        return results
    
    def search_by_id(
        self,
        query_id: str,
        k: int = 10,
        exclude_self: bool = True,
    ) -> List[SearchResult]:
        """
        Search using an existing embedding as query.
        
        Args:
            query_id: ID of the query embedding
            k: Number of results
            exclude_self: Whether to exclude the query from results
        
        Returns:
            List of SearchResult objects
        """
        if query_id not in self.embeddings:
            raise KeyError(f"ID not found: {query_id}")
        
        query = self.embeddings[query_id]
        k_search = k + 1 if exclude_self else k
        
        results = self.search(query, k=k_search)
        
        if exclude_self:
            results = [r for r in results if r.id != query_id][:k]
        
        return results
    
    def get_embedding(self, id: str) -> Optional[np.ndarray]:
        """Get embedding by ID."""
        return self.embeddings.get(id)
    
    def get_embeddings(self, ids: List[str]) -> np.ndarray:
        """
        Get multiple embeddings by ID.
        
        Args:
            ids: List of IDs
        
        Returns:
            Array of shape (len(ids), dimension)
        """
        embeddings = []
        for id_ in ids:
            if id_ in self.embeddings:
                embeddings.append(self.embeddings[id_])
            else:
                logger.warning(f"ID not found: {id_}")
                embeddings.append(np.zeros(self.dimension, dtype=np.float32))
        
        return np.array(embeddings)
    
    def get_all_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """
        Get all embeddings.
        
        Returns:
            Tuple of (embeddings array, list of IDs)
        """
        ids = self.id_list.copy()
        embeddings = self.get_embeddings(ids)
        return embeddings, ids
    
    def get_metadata(self, id: str) -> Optional[EmbeddingMetadata]:
        """Get metadata by ID."""
        return self.metadata.get(id)
    
    def update_metadata(self, id: str, **kwargs) -> None:
        """
        Update metadata fields.
        
        Args:
            id: Embedding ID
            **kwargs: Fields to update
        """
        if id not in self.metadata:
            self.metadata[id] = EmbeddingMetadata(id=id)
        
        for key, value in kwargs.items():
            if hasattr(self.metadata[id], key):
                setattr(self.metadata[id], key, value)
            else:
                self.metadata[id].extra[key] = value
    
    def compute_similarity_matrix(
        self,
        ids: List[str] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute pairwise similarity matrix.
        
        Args:
            ids: Optional subset of IDs (default: all)
        
        Returns:
            Tuple of (similarity_matrix, list of IDs)
        """
        if ids is None:
            ids = self.id_list.copy()
        
        embeddings = self.get_embeddings(ids)
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = embeddings / norms
        
        # Compute similarity
        similarity = normalized @ normalized.T
        
        return similarity, ids
    
    def get_valid_ids(self) -> List[str]:
        """Get list of IDs marked as valid."""
        return [id_ for id_ in self.id_list if self.metadata.get(id_, EmbeddingMetadata(id=id_)).is_valid]
    
    def get_scores(self, ids: List[str] = None) -> np.ndarray:
        """
        Get quality scores for embeddings.
        
        Args:
            ids: Optional subset of IDs (default: all)
        
        Returns:
            Array of quality scores
        """
        if ids is None:
            ids = self.id_list
        
        return np.array([
            self.metadata.get(id_, EmbeddingMetadata(id=id_)).quality_score
            for id_ in ids
        ])
    
    def save(self, path: Path) -> None:
        """
        Save store to disk.
        
        Creates:
        - embeddings.npz: Raw embeddings
        - metadata.json: Metadata for all embeddings
        - faiss/: FAISS index files
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save raw embeddings
        np.savez_compressed(
            path / "embeddings.npz",
            **{id_: emb for id_, emb in self.embeddings.items()}
        )
        
        # Save ID list
        with open(path / "id_list.json", "w") as f:
            json.dump(self.id_list, f)
        
        # Save metadata
        metadata_dict = {id_: meta.to_dict() for id_, meta in self.metadata.items()}
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata_dict, f, indent=2)
        
        # Save FAISS index
        self.faiss_index.save(path / "faiss")
        
        # Save config
        config = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "use_gpu": self.use_gpu,
            "normalize": self.normalize,
            "num_embeddings": len(self.embeddings),
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"EmbeddingStore saved to {path} ({len(self.embeddings)} embeddings)")
    
    @classmethod
    def load(cls, path: Path, use_gpu: bool = None) -> "EmbeddingStore":
        """
        Load store from disk.
        
        Args:
            path: Path to saved store
            use_gpu: Override GPU setting (None = use saved setting)
        
        Returns:
            Loaded EmbeddingStore
        """
        path = Path(path)
        
        # Load config
        with open(path / "config.json") as f:
            config = json.load(f)
        
        if use_gpu is not None:
            config["use_gpu"] = use_gpu
        
        # Create instance
        store = cls(
            dimension=config["dimension"],
            index_type=config["index_type"],
            use_gpu=config["use_gpu"],
            normalize=config.get("normalize", True),
        )
        
        # Load embeddings
        data = np.load(path / "embeddings.npz")
        store.embeddings = {key: data[key] for key in data.files}
        
        # Load ID list
        with open(path / "id_list.json") as f:
            store.id_list = json.load(f)
        
        # Load metadata
        with open(path / "metadata.json") as f:
            metadata_dict = json.load(f)
        store.metadata = {id_: EmbeddingMetadata.from_dict(meta) for id_, meta in metadata_dict.items()}
        
        # Load FAISS index
        store.faiss_index.load(path / "faiss")
        
        logger.info(f"EmbeddingStore loaded from {path} ({len(store.embeddings)} embeddings)")
        
        return store
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __contains__(self, id: str) -> bool:
        return id in self.embeddings
    
    def __repr__(self) -> str:
        return f"EmbeddingStore(n={len(self)}, dim={self.dimension}, gpu={self.use_gpu})"


# ============================================
# Quick Test
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Embedding Store Test")
    print("=" * 60)
    
    if not HAS_FAISS:
        print("✗ FAISS not installed")
        print("  Install with: pip install faiss-gpu-cu12")
        exit(1)
    
    print(f"✓ FAISS available, GPU: {HAS_GPU}")
    
    # Create store
    store = EmbeddingStore(dimension=768, use_gpu=HAS_GPU)
    print(f"✓ Store created: {store}")
    
    # Add test embeddings
    print("\nAdding 100 test embeddings...")
    np.random.seed(42)
    embeddings = np.random.randn(100, 768).astype(np.float32)
    ids = [f"img_{i:04d}" for i in range(100)]
    
    metadata_list = [
        EmbeddingMetadata(
            id=id_,
            quality_score=np.random.rand(),
            is_valid=(i % 3 != 0),  # Mark every 3rd as invalid
        )
        for i, id_ in enumerate(ids)
    ]
    
    store.add_batch(embeddings, ids, metadata_list)
    print(f"✓ Store now has {len(store)} embeddings")
    
    # Search
    print("\nSearching for similar embeddings...")
    results = store.search(embeddings[0], k=5)
    print(f"Top 5 results for {ids[0]}:")
    for r in results:
        print(f"  {r.id}: score={r.score:.4f}")
    
    # Search by ID
    print("\nSearch by ID...")
    results = store.search_by_id(ids[0], k=5)
    print(f"Similar to {ids[0]} (excluding self):")
    for r in results:
        print(f"  {r.id}: score={r.score:.4f}")
    
    # Get embedding back
    print("\nReconstructing embedding...")
    emb = store.get_embedding(ids[0])
    match = np.allclose(emb, embeddings[0])
    print(f"✓ Embedding reconstruction: {'MATCH' if match else 'MISMATCH'}")
    
    # Similarity matrix
    print("\nComputing similarity matrix for 20 embeddings...")
    sim_matrix, sim_ids = store.compute_similarity_matrix(ids[:20])
    print(f"Matrix shape: {sim_matrix.shape}")
    print(f"Diagonal (self-similarity): {np.diag(sim_matrix)[:5]}")
    
    # Valid IDs
    valid_ids = store.get_valid_ids()
    print(f"\nValid embeddings: {len(valid_ids)}/{len(store)}")
    
    # Save and load
    print("\nTesting save/load...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_store"
        store.save(save_path)
        print(f"✓ Saved to {save_path}")
        
        loaded = EmbeddingStore.load(save_path)
        print(f"✓ Loaded: {loaded}")
        
        # Verify
        loaded_emb = loaded.get_embedding(ids[0])
        match = np.allclose(loaded_emb, embeddings[0])
        print(f"✓ Loaded embedding match: {'YES' if match else 'NO'}")
    
    print("\n✅ Embedding Store test passed!")