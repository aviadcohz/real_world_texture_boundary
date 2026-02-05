"""
FAISS Vector Database for Texture Curator.

Provides GPU-accelerated similarity search for large-scale datasets.
Supports thousands of embeddings with millisecond query times.

FEATURES:
- FAISS-GPU for fast similarity search
- Persistent storage (save/load indexes)
- Batch operations for efficiency
- Multiple index types for different use cases

USAGE:
    db = FAISSIndex(dimension=768, use_gpu=True)
    db.add_embeddings(embeddings, ids)
    similar_ids, scores = db.search(query_embedding, k=10)
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import FAISS
try:
    import faiss
    HAS_FAISS = True
    
    # Check for GPU support
    HAS_FAISS_GPU = faiss.get_num_gpus() > 0
    if HAS_FAISS_GPU:
        logger.info(f"FAISS-GPU available with {faiss.get_num_gpus()} GPU(s)")
    else:
        logger.info("FAISS available (CPU only)")
except ImportError:
    HAS_FAISS = False
    HAS_FAISS_GPU = False
    logger.warning("FAISS not installed. Install with: pip install faiss-gpu-cu12")


@dataclass
class IndexMetadata:
    """Metadata about the FAISS index."""
    dimension: int
    num_embeddings: int
    index_type: str
    use_gpu: bool
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "num_embeddings": self.num_embeddings,
            "index_type": self.index_type,
            "use_gpu": self.use_gpu,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }


class FAISSIndex:
    """
    GPU-accelerated FAISS index for similarity search.
    
    Supports multiple index types:
    - Flat: Exact search (best for < 10K vectors)
    - IVF: Approximate search (best for 10K - 1M vectors)
    - HNSW: Graph-based (good balance of speed/accuracy)
    """
    
    INDEX_TYPES = ["flat", "ivf", "hnsw"]
    
    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "flat",
        use_gpu: bool = True,
        nlist: int = 100,  # For IVF: number of clusters
        nprobe: int = 10,  # For IVF: clusters to search
        M: int = 32,       # For HNSW: connections per node
        ef_search: int = 64,  # For HNSW: search depth
    ):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension (768 for DINOv2-base)
            index_type: "flat", "ivf", or "hnsw"
            use_gpu: Whether to use GPU acceleration
            nlist: Number of IVF clusters
            nprobe: Number of clusters to search
            M: HNSW connections per node
            ef_search: HNSW search depth
        """
        if not HAS_FAISS:
            raise ImportError("FAISS not installed. Run: pip install faiss-gpu-cu12")
        
        self.dimension = dimension
        self.index_type = index_type
        self.use_gpu = use_gpu and HAS_FAISS_GPU
        self.nlist = nlist
        self.nprobe = nprobe
        self.M = M
        self.ef_search = ef_search
        
        # ID management (FAISS uses sequential IDs internally)
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.next_idx = 0
        
        # Create index
        self.index = self._create_index()
        
        # Metadata
        self.metadata = IndexMetadata(
            dimension=dimension,
            num_embeddings=0,
            index_type=index_type,
            use_gpu=self.use_gpu,
        )
        
        logger.info(f"FAISSIndex created: dim={dimension}, type={index_type}, gpu={self.use_gpu}")
    
    def _create_index(self) -> faiss.Index:
        """Create the appropriate FAISS index."""
        
        if self.index_type == "flat":
            # Exact search - best for smaller datasets
            index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine after normalization)
            
        elif self.index_type == "ivf":
            # Inverted file index - good for medium datasets
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
            index.nprobe = self.nprobe
            
        elif self.index_type == "hnsw":
            # Hierarchical Navigable Small World - good balance
            index = faiss.IndexHNSWFlat(self.dimension, self.M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = self.ef_search
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}. Choose from {self.INDEX_TYPES}")
        
        # Move to GPU if available and requested
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("Index moved to GPU")
            except Exception as e:
                logger.warning(f"Failed to move index to GPU: {e}. Using CPU.")
                self.use_gpu = False
        
        return index
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return embeddings / norms
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        ids: List[str],
        normalize: bool = True,
    ) -> int:
        """
        Add embeddings to the index.
        
        Args:
            embeddings: Array of shape (N, dimension)
            ids: List of string IDs for each embedding
            normalize: Whether to L2 normalize embeddings
        
        Returns:
            Number of embeddings added
        """
        if len(embeddings) != len(ids):
            raise ValueError(f"Number of embeddings ({len(embeddings)}) != number of IDs ({len(ids)})")
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension ({embeddings.shape[1]}) != index dimension ({self.dimension})")
        
        # Normalize for cosine similarity
        if normalize:
            embeddings = self._normalize(embeddings)
        
        # Convert to float32 (FAISS requirement)
        embeddings = embeddings.astype(np.float32)
        
        # Map IDs to sequential indices
        for id_ in ids:
            if id_ not in self.id_to_idx:
                self.id_to_idx[id_] = self.next_idx
                self.idx_to_id[self.next_idx] = id_
                self.next_idx += 1
        
        # Train IVF index if needed
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info(f"Training IVF index with {len(embeddings)} vectors...")
            # Need to convert to CPU for training
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                cpu_index.train(embeddings)
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            else:
                self.index.train(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Update metadata
        self.metadata.num_embeddings = self.index.ntotal
        self.metadata.last_updated = datetime.now()
        
        logger.info(f"Added {len(embeddings)} embeddings. Total: {self.index.ntotal}")
        
        return len(embeddings)
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        normalize: bool = True,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Search for similar embeddings.
        
        Args:
            query: Query embedding(s) of shape (dimension,) or (N, dimension)
            k: Number of results per query
            normalize: Whether to normalize query
        
        Returns:
            Tuple of (ids, scores) where:
            - ids: List of string IDs (or list of lists for batch)
            - scores: Similarity scores (higher = more similar)
        """
        # Handle single query
        single_query = query.ndim == 1
        if single_query:
            query = query.reshape(1, -1)
        
        # Normalize
        if normalize:
            query = self._normalize(query)
        
        query = query.astype(np.float32)
        
        # Search
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query, k)
        
        # Convert indices to IDs
        result_ids = []
        for idx_row in indices:
            ids = [self.idx_to_id.get(int(idx), None) for idx in idx_row if idx >= 0]
            result_ids.append(ids)
        
        if single_query:
            return result_ids[0], scores[0]
        
        return result_ids, scores
    
    def search_by_id(
        self,
        query_id: str,
        k: int = 10,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Search for embeddings similar to a given ID.
        
        Note: This requires storing the original embeddings or
        using an index that supports reconstruction.
        """
        raise NotImplementedError("search_by_id requires embedding storage. Use EmbeddingStore instead.")
    
    def compute_all_pairwise_similarities(self) -> np.ndarray:
        """
        Compute full pairwise similarity matrix.
        
        Warning: O(n²) memory and time. Use only for small datasets (<5K).
        
        Returns:
            Similarity matrix of shape (N, N)
        """
        n = self.index.ntotal
        
        if n > 5000:
            logger.warning(f"Computing {n}x{n} similarity matrix. This may be slow/memory intensive.")
        
        # Get all vectors (need to reconstruct)
        if hasattr(self.index, 'reconstruct_n'):
            vectors = self.index.reconstruct_n(0, n)
        else:
            raise NotImplementedError("Index type doesn't support reconstruction")
        
        # Normalize and compute similarity
        vectors = self._normalize(vectors)
        similarity_matrix = vectors @ vectors.T
        
        return similarity_matrix
    
    def save(self, path: Path):
        """Save index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save index (convert to CPU first if on GPU)
        index_path = path / "index.faiss"
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(index_path))
        else:
            faiss.write_index(self.index, str(index_path))
        
        # Save ID mappings and metadata
        meta_path = path / "metadata.pkl"
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'id_to_idx': self.id_to_idx,
                'idx_to_id': self.idx_to_id,
                'next_idx': self.next_idx,
                'metadata': self.metadata,
                'config': {
                    'dimension': self.dimension,
                    'index_type': self.index_type,
                    'use_gpu': self.use_gpu,
                    'nlist': self.nlist,
                    'nprobe': self.nprobe,
                    'M': self.M,
                    'ef_search': self.ef_search,
                }
            }, f)
        
        logger.info(f"Index saved to {path}")
    
    @classmethod
    def load(cls, path: Path, use_gpu: bool = True) -> "FAISSIndex":
        """Load index from disk."""
        path = Path(path)
        
        # Load metadata
        meta_path = path / "metadata.pkl"
        with open(meta_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create instance
        config = data['config']
        instance = cls(
            dimension=config['dimension'],
            index_type=config['index_type'],
            use_gpu=use_gpu,
            nlist=config['nlist'],
            nprobe=config['nprobe'],
            M=config['M'],
            ef_search=config['ef_search'],
        )
        
        # Load index
        index_path = path / "index.faiss"
        instance.index = faiss.read_index(str(index_path))
        
        if use_gpu and HAS_FAISS_GPU:
            try:
                res = faiss.StandardGpuResources()
                instance.index = faiss.index_cpu_to_gpu(res, 0, instance.index)
                instance.use_gpu = True
            except Exception as e:
                logger.warning(f"Failed to move loaded index to GPU: {e}")
                instance.use_gpu = False
        
        # Restore mappings
        instance.id_to_idx = data['id_to_idx']
        instance.idx_to_id = data['idx_to_id']
        instance.next_idx = data['next_idx']
        instance.metadata = data['metadata']
        
        logger.info(f"Index loaded from {path}: {instance.metadata.num_embeddings} embeddings")
        
        return instance
    
    def __len__(self) -> int:
        return self.index.ntotal
    
    def __repr__(self) -> str:
        return f"FAISSIndex(dim={self.dimension}, type={self.index_type}, n={len(self)}, gpu={self.use_gpu})"


class EmbeddingStore:
    """
    Combined embedding storage with FAISS index.
    
    Stores both the raw embeddings and the FAISS index,
    enabling both fast search and reconstruction.
    """
    
    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "flat",
        use_gpu: bool = True,
    ):
        """
        Initialize embedding store.
        
        Args:
            dimension: Embedding dimension
            index_type: FAISS index type
            use_gpu: Use GPU acceleration
        """
        self.dimension = dimension
        self.faiss_index = FAISSIndex(dimension, index_type, use_gpu)
        
        # Raw embedding storage
        self.embeddings: Dict[str, np.ndarray] = {}
        
        # Additional metadata per embedding
        self.embedding_metadata: Dict[str, Dict[str, Any]] = {}
    
    def add(
        self,
        embedding: np.ndarray,
        id_: str,
        metadata: Dict[str, Any] = None,
    ):
        """Add a single embedding."""
        self.add_batch(
            embeddings=embedding.reshape(1, -1),
            ids=[id_],
            metadata_list=[metadata] if metadata else None,
        )
    
    def add_batch(
        self,
        embeddings: np.ndarray,
        ids: List[str],
        metadata_list: List[Dict[str, Any]] = None,
    ):
        """Add a batch of embeddings."""
        # Store raw embeddings
        for i, id_ in enumerate(ids):
            self.embeddings[id_] = embeddings[i].copy()
            
            if metadata_list and metadata_list[i]:
                self.embedding_metadata[id_] = metadata_list[i]
        
        # Add to FAISS index
        self.faiss_index.add_embeddings(embeddings, ids)
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
    ) -> Tuple[List[str], np.ndarray]:
        """Search for similar embeddings."""
        return self.faiss_index.search(query, k)
    
    def search_by_id(
        self,
        query_id: str,
        k: int = 10,
    ) -> Tuple[List[str], np.ndarray]:
        """Search using an existing embedding as query."""
        if query_id not in self.embeddings:
            raise KeyError(f"ID not found: {query_id}")
        
        query = self.embeddings[query_id]
        return self.search(query, k)
    
    def get_embedding(self, id_: str) -> Optional[np.ndarray]:
        """Get embedding by ID."""
        return self.embeddings.get(id_)
    
    def get_embeddings(self, ids: List[str]) -> np.ndarray:
        """Get multiple embeddings by ID."""
        return np.array([self.embeddings[id_] for id_ in ids if id_ in self.embeddings])
    
    def get_all_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """Get all embeddings and their IDs."""
        ids = list(self.embeddings.keys())
        embeddings = np.array([self.embeddings[id_] for id_ in ids])
        return embeddings, ids
    
    def compute_pairwise_similarity(
        self,
        ids: List[str] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute pairwise similarity matrix.
        
        Args:
            ids: Optional subset of IDs (default: all)
        
        Returns:
            Tuple of (similarity_matrix, ids)
        """
        if ids is None:
            embeddings, ids = self.get_all_embeddings()
        else:
            embeddings = self.get_embeddings(ids)
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = embeddings / norms
        
        # Compute similarity
        similarity = normalized @ normalized.T
        
        return similarity, ids
    
    def save(self, path: Path):
        """Save store to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        self.faiss_index.save(path / "faiss")
        
        # Save raw embeddings
        np.savez_compressed(
            path / "embeddings.npz",
            **{id_: emb for id_, emb in self.embeddings.items()}
        )
        
        # Save metadata
        with open(path / "embedding_metadata.pkl", 'wb') as f:
            pickle.dump(self.embedding_metadata, f)
        
        logger.info(f"EmbeddingStore saved to {path}")
    
    @classmethod
    def load(cls, path: Path, use_gpu: bool = True) -> "EmbeddingStore":
        """Load store from disk."""
        path = Path(path)
        
        # Load FAISS index
        faiss_index = FAISSIndex.load(path / "faiss", use_gpu)
        
        # Create instance
        instance = cls(
            dimension=faiss_index.dimension,
            index_type=faiss_index.index_type,
            use_gpu=use_gpu,
        )
        instance.faiss_index = faiss_index
        
        # Load embeddings
        data = np.load(path / "embeddings.npz")
        instance.embeddings = {key: data[key] for key in data.files}
        
        # Load metadata
        meta_path = path / "embedding_metadata.pkl"
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                instance.embedding_metadata = pickle.load(f)
        
        logger.info(f"EmbeddingStore loaded: {len(instance.embeddings)} embeddings")
        
        return instance
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __repr__(self) -> str:
        return f"EmbeddingStore(n={len(self)}, dim={self.dimension})"


# ============================================
# Quick Test
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("FAISS Vector Database Test")
    print("=" * 60)
    
    if not HAS_FAISS:
        print("✗ FAISS not installed")
        print("  Install with: pip install faiss-gpu-cu12")
        exit(1)
    
    print(f"✓ FAISS installed")
    print(f"  GPU available: {HAS_FAISS_GPU}")
    
    # Test with random embeddings
    np.random.seed(42)
    n_embeddings = 1000
    dimension = 768
    
    print(f"\nCreating {n_embeddings} random embeddings...")
    embeddings = np.random.randn(n_embeddings, dimension).astype(np.float32)
    ids = [f"img_{i:04d}" for i in range(n_embeddings)]
    
    # Test FAISSIndex
    print("\n--- Testing FAISSIndex ---")
    index = FAISSIndex(dimension=dimension, index_type="flat", use_gpu=True)
    print(f"Created: {index}")
    
    index.add_embeddings(embeddings, ids)
    print(f"After add: {index}")
    
    # Search
    query = embeddings[0]
    result_ids, scores = index.search(query, k=5)
    print(f"\nSearch results for {ids[0]}:")
    for id_, score in zip(result_ids, scores):
        print(f"  {id_}: {score:.4f}")
    
    # Test EmbeddingStore
    print("\n--- Testing EmbeddingStore ---")
    store = EmbeddingStore(dimension=dimension, use_gpu=True)
    store.add_batch(embeddings[:100], ids[:100])
    print(f"Created: {store}")
    
    # Search by ID
    result_ids, scores = store.search_by_id(ids[0], k=5)
    print(f"\nSearch by ID results:")
    for id_, score in zip(result_ids, scores):
        print(f"  {id_}: {score:.4f}")
    
    # Pairwise similarity
    print("\nComputing pairwise similarity for 50 embeddings...")
    sim_matrix, sim_ids = store.compute_pairwise_similarity(ids[:50])
    print(f"  Matrix shape: {sim_matrix.shape}")
    print(f"  Mean similarity: {sim_matrix.mean():.4f}")
    
    print("\n✅ FAISS Vector Database test passed!")


# ============================================
# Aliases and missing exports for __init__.py
# ============================================

# Alias for backward compatibility
FAISSVectorDB = FAISSIndex
HAS_GPU = HAS_FAISS_GPU


@dataclass
class SearchResult:
    """Result from a similarity search."""
    id: str
    score: float
    rank: int = 0


def create_vector_db(
    dimension: int = 768,
    num_vectors: int = 0,
    use_gpu: bool = True,
) -> FAISSIndex:
    """
    Create an appropriate vector database based on expected size.
    
    Args:
        dimension: Vector dimension
        num_vectors: Expected number of vectors (0 if unknown)
        use_gpu: Whether to use GPU
    
    Returns:
        Configured FAISSIndex instance
    """
    # Choose index type based on size
    if num_vectors < 10000:
        index_type = "flat"
    elif num_vectors < 1000000:
        index_type = "ivf"
    else:
        index_type = "hnsw"
    
    return FAISSIndex(
        dimension=dimension,
        index_type=index_type,
        use_gpu=use_gpu and HAS_FAISS_GPU,
    )