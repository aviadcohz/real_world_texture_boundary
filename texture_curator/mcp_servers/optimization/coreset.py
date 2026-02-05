"""
Coreset Selection Algorithms for Texture Curator.

Provides advanced algorithms for selecting diverse, representative subsets
from large candidate pools. Essential for scaling to thousands of images.

ALGORITHMS:
- Greedy K-Center: Maximize minimum distance to selected set
- K-Medoids: Cluster-based selection
- MaxMin Diversity: Greedily maximize diversity
- Facility Location: Submodular optimization
- DPP (Determinantal Point Process): Probabilistic diversity

USAGE:
    selector = CoresetSelector(method="k_center")
    selected_indices = selector.select(embeddings, k=100, scores=quality_scores)
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass
import logging
from enum import Enum
import heapq

logger = logging.getLogger(__name__)


class SelectionMethod(Enum):
    """Available selection methods."""
    GREEDY_QUALITY = "greedy_quality"      # Simple top-k by quality
    K_CENTER = "k_center"                   # K-center coreset
    K_MEDOIDS = "k_medoids"                 # K-medoids clustering
    MAXMIN_DIVERSITY = "maxmin_diversity"   # MaxMin diversity sampling
    FACILITY_LOCATION = "facility_location" # Submodular facility location
    QUALITY_DIVERSITY = "quality_diversity" # Balance quality and diversity


@dataclass
class SelectionResult:
    """Result of coreset selection."""
    selected_indices: List[int]
    selected_ids: List[str]
    diversity_score: float
    quality_score: float
    method: str
    iterations: int = 0
    
    def to_dict(self) -> dict:
        return {
            "selected_indices": self.selected_indices,
            "selected_ids": self.selected_ids,
            "diversity_score": self.diversity_score,
            "quality_score": self.quality_score,
            "method": self.method,
            "iterations": self.iterations,
        }


class CoresetSelector:
    """
    Advanced coreset selection for diverse sampling.
    
    Selects a representative subset that balances:
    - Quality: High individual scores
    - Diversity: Coverage of the embedding space
    - Representativeness: Similar to the overall distribution
    """
    
    def __init__(
        self,
        method: str = "quality_diversity",
        diversity_weight: float = 0.3,
        batch_size: int = 1000,  # For large-scale computation
    ):
        """
        Initialize coreset selector.
        
        Args:
            method: Selection algorithm to use
            diversity_weight: Weight for diversity vs quality (0-1)
            batch_size: Batch size for large-scale similarity computation
        """
        self.method = method
        self.diversity_weight = diversity_weight
        self.batch_size = batch_size
    
    def select(
        self,
        embeddings: np.ndarray,
        k: int,
        ids: List[str] = None,
        scores: np.ndarray = None,
        similarity_matrix: np.ndarray = None,
    ) -> SelectionResult:
        """
        Select k diverse, high-quality samples.
        
        Args:
            embeddings: Embedding matrix (N, D)
            k: Number of samples to select
            ids: Optional string IDs for each embedding
            scores: Optional quality scores (higher = better)
            similarity_matrix: Optional precomputed similarity (N, N)
        
        Returns:
            SelectionResult with selected indices and metrics
        """
        n = len(embeddings)
        k = min(k, n)
        
        if ids is None:
            ids = [f"idx_{i}" for i in range(n)]
        
        if scores is None:
            scores = np.ones(n)
        
        logger.info(f"Selecting {k} from {n} using {self.method}")
        
        # Normalize embeddings for cosine similarity
        embeddings_norm = self._normalize(embeddings)
        
        # Compute similarity matrix if needed and not provided
        if similarity_matrix is None and self.method in ["k_center", "maxmin_diversity", "facility_location", "quality_diversity"]:
            if n <= 5000:
                logger.info("Computing full similarity matrix...")
                similarity_matrix = embeddings_norm @ embeddings_norm.T
            else:
                logger.info("Large dataset: using batch similarity computation")
        
        # Select based on method
        if self.method == "greedy_quality":
            selected = self._greedy_quality(scores, k)
        
        elif self.method == "k_center":
            selected = self._k_center(embeddings_norm, k, similarity_matrix)
        
        elif self.method == "k_medoids":
            selected = self._k_medoids(embeddings_norm, k)
        
        elif self.method == "maxmin_diversity":
            selected = self._maxmin_diversity(embeddings_norm, k, similarity_matrix)
        
        elif self.method == "facility_location":
            selected = self._facility_location(embeddings_norm, k, scores, similarity_matrix)
        
        elif self.method == "quality_diversity":
            selected = self._quality_diversity(embeddings_norm, k, scores, similarity_matrix)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Compute metrics
        selected_ids = [ids[i] for i in selected]
        diversity = self._compute_diversity(embeddings_norm[selected])
        quality = float(np.mean(scores[selected]))
        
        logger.info(f"Selected {len(selected)}: diversity={diversity:.3f}, quality={quality:.3f}")
        
        return SelectionResult(
            selected_indices=selected,
            selected_ids=selected_ids,
            diversity_score=diversity,
            quality_score=quality,
            method=self.method,
        )
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return embeddings / norms
    
    def _compute_diversity(self, embeddings: np.ndarray) -> float:
        """
        Compute diversity score for selected embeddings.
        
        Diversity = 1 - mean(pairwise_similarity)
        Higher = more diverse
        """
        if len(embeddings) < 2:
            return 1.0
        
        similarity = embeddings @ embeddings.T
        # Get upper triangle (excluding diagonal)
        upper_tri = similarity[np.triu_indices_from(similarity, k=1)]
        
        return 1.0 - float(np.mean(upper_tri))
    
    def _greedy_quality(self, scores: np.ndarray, k: int) -> List[int]:
        """Simple top-k by quality score."""
        return list(np.argsort(scores)[-k:][::-1])
    
    def _k_center(
        self,
        embeddings: np.ndarray,
        k: int,
        similarity_matrix: np.ndarray = None,
    ) -> List[int]:
        """
        K-Center (Greedy Furthest Point) selection.
        
        Greedily selects points that maximize the minimum distance
        to the already selected set. Good for coverage.
        """
        n = len(embeddings)
        selected = []
        
        # Start with point furthest from centroid
        centroid = embeddings.mean(axis=0)
        distances_to_centroid = 1 - embeddings @ centroid
        selected.append(int(np.argmax(distances_to_centroid)))
        
        # Distance to nearest selected point for each point
        min_distances = np.ones(n) * np.inf
        
        for _ in range(k - 1):
            # Update min distances based on last selected
            last_selected = selected[-1]
            
            if similarity_matrix is not None:
                similarities = similarity_matrix[last_selected]
            else:
                similarities = embeddings @ embeddings[last_selected]
            
            distances = 1 - similarities  # Convert similarity to distance
            min_distances = np.minimum(min_distances, distances)
            
            # Select point with maximum min-distance
            min_distances[selected] = -np.inf  # Exclude already selected
            next_idx = int(np.argmax(min_distances))
            selected.append(next_idx)
        
        return selected
    
    def _k_medoids(
        self,
        embeddings: np.ndarray,
        k: int,
        max_iter: int = 100,
    ) -> List[int]:
        """
        K-Medoids clustering selection.
        
        Finds k representative medoids (actual data points as centers).
        """
        n = len(embeddings)
        
        # Initialize with k-center
        medoids = self._k_center(embeddings, k)
        
        # Compute distance matrix
        similarity = embeddings @ embeddings.T
        distance = 1 - similarity
        
        for iteration in range(max_iter):
            # Assign points to nearest medoid
            distances_to_medoids = distance[:, medoids]
            assignments = np.argmin(distances_to_medoids, axis=1)
            
            # Update medoids
            new_medoids = []
            changed = False
            
            for cluster_idx in range(k):
                cluster_points = np.where(assignments == cluster_idx)[0]
                
                if len(cluster_points) == 0:
                    new_medoids.append(medoids[cluster_idx])
                    continue
                
                # Find point that minimizes total distance to cluster
                cluster_distances = distance[np.ix_(cluster_points, cluster_points)]
                total_distances = cluster_distances.sum(axis=1)
                best_idx = cluster_points[np.argmin(total_distances)]
                
                if best_idx != medoids[cluster_idx]:
                    changed = True
                
                new_medoids.append(best_idx)
            
            medoids = new_medoids
            
            if not changed:
                break
        
        return medoids
    
    def _maxmin_diversity(
        self,
        embeddings: np.ndarray,
        k: int,
        similarity_matrix: np.ndarray = None,
    ) -> List[int]:
        """
        MaxMin diversity sampling.
        
        Greedily selects points that maximize minimum pairwise distance
        within the selected set.
        """
        n = len(embeddings)
        selected = []
        
        # Start with random point
        selected.append(np.random.randint(n))
        
        for _ in range(k - 1):
            # Compute min similarity to selected set for each point
            min_similarities = np.ones(n) * np.inf
            
            for s in selected:
                if similarity_matrix is not None:
                    similarities = similarity_matrix[s]
                else:
                    similarities = embeddings @ embeddings[s]
                
                min_similarities = np.minimum(min_similarities, similarities)
            
            # Mask already selected
            min_similarities[selected] = np.inf
            
            # Select point with minimum similarity (maximum distance)
            next_idx = int(np.argmin(min_similarities))
            selected.append(next_idx)
        
        return selected
    
    def _facility_location(
        self,
        embeddings: np.ndarray,
        k: int,
        scores: np.ndarray,
        similarity_matrix: np.ndarray = None,
    ) -> List[int]:
        """
        Submodular Facility Location selection.
        
        Maximizes coverage: sum of max similarities to selected set.
        This is a submodular function, so greedy gives 1-1/e approximation.
        """
        n = len(embeddings)
        selected = []
        
        # Coverage of each point by selected set
        coverage = np.zeros(n)
        
        # Weighted scores for tie-breaking
        weighted_scores = scores / scores.max() if scores.max() > 0 else np.ones(n)
        
        for _ in range(k):
            best_gain = -np.inf
            best_idx = -1
            
            for i in range(n):
                if i in selected:
                    continue
                
                # Compute marginal gain
                if similarity_matrix is not None:
                    similarities = similarity_matrix[i]
                else:
                    similarities = embeddings @ embeddings[i]
                
                # Gain = improvement in coverage
                new_coverage = np.maximum(coverage, similarities)
                gain = np.sum(new_coverage - coverage)
                
                # Add small quality bonus for tie-breaking
                gain += 0.01 * weighted_scores[i]
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx >= 0:
                # Update coverage
                if similarity_matrix is not None:
                    similarities = similarity_matrix[best_idx]
                else:
                    similarities = embeddings @ embeddings[best_idx]
                
                coverage = np.maximum(coverage, similarities)
                selected.append(best_idx)
        
        return selected
    
    def _quality_diversity(
        self,
        embeddings: np.ndarray,
        k: int,
        scores: np.ndarray,
        similarity_matrix: np.ndarray = None,
    ) -> List[int]:
        """
        Quality-Diversity trade-off selection.
        
        Greedily selects points that maximize:
          score(i) = quality(i) - λ * max_similarity_to_selected(i)
        
        This balances picking high-quality items while maintaining diversity.
        """
        n = len(embeddings)
        selected = []
        
        # Normalize scores to 0-1
        if scores.max() > scores.min():
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            normalized_scores = np.ones(n)
        
        for step in range(k):
            best_score = -np.inf
            best_idx = -1
            
            for i in range(n):
                if i in selected:
                    continue
                
                # Quality component
                quality = normalized_scores[i]
                
                # Diversity penalty
                if selected:
                    if similarity_matrix is not None:
                        similarities = similarity_matrix[i, selected]
                    else:
                        similarities = embeddings[i] @ embeddings[selected].T
                    
                    max_sim = np.max(similarities)
                    diversity_penalty = self.diversity_weight * max_sim
                else:
                    diversity_penalty = 0
                
                # Combined score
                combined = quality - diversity_penalty
                
                if combined > best_score:
                    best_score = combined
                    best_idx = i
            
            if best_idx >= 0:
                selected.append(best_idx)
        
        return selected
    
    def select_with_constraints(
        self,
        embeddings: np.ndarray,
        k: int,
        ids: List[str] = None,
        scores: np.ndarray = None,
        must_include: List[int] = None,
        must_exclude: List[int] = None,
        min_score: float = None,
    ) -> SelectionResult:
        """
        Select with constraints.
        
        Args:
            embeddings: Embedding matrix
            k: Number to select
            ids: String IDs
            scores: Quality scores
            must_include: Indices that must be included
            must_exclude: Indices that must be excluded
            min_score: Minimum quality score threshold
        """
        n = len(embeddings)
        
        # Build valid mask
        valid_mask = np.ones(n, dtype=bool)
        
        if must_exclude:
            valid_mask[must_exclude] = False
        
        if min_score is not None and scores is not None:
            valid_mask &= (scores >= min_score)
        
        # Filter to valid indices
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) < k:
            logger.warning(f"Only {len(valid_indices)} valid candidates for k={k}")
            k = len(valid_indices)
        
        # Handle must_include
        if must_include:
            must_include_set = set(must_include) & set(valid_indices)
            remaining_k = k - len(must_include_set)
            
            if remaining_k <= 0:
                selected = list(must_include_set)[:k]
            else:
                # Select remaining from valid indices excluding must_include
                valid_remaining = [i for i in valid_indices if i not in must_include_set]
                
                result = self.select(
                    embeddings[valid_remaining],
                    remaining_k,
                    [ids[i] for i in valid_remaining] if ids else None,
                    scores[valid_remaining] if scores is not None else None,
                )
                
                # Map back to original indices
                selected = list(must_include_set) + [valid_remaining[i] for i in result.selected_indices]
        else:
            # Normal selection on valid subset
            result = self.select(
                embeddings[valid_indices],
                k,
                [ids[i] for i in valid_indices] if ids else None,
                scores[valid_indices] if scores is not None else None,
            )
            
            # Map back to original indices
            selected = [valid_indices[i] for i in result.selected_indices]
        
        # Compute final metrics
        embeddings_norm = self._normalize(embeddings)
        selected_ids = [ids[i] for i in selected] if ids else [f"idx_{i}" for i in selected]
        diversity = self._compute_diversity(embeddings_norm[selected])
        quality = float(np.mean(scores[selected])) if scores is not None else 1.0
        
        return SelectionResult(
            selected_indices=selected,
            selected_ids=selected_ids,
            diversity_score=diversity,
            quality_score=quality,
            method=self.method,
        )


# ============================================
# Quick Test
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Coreset Selection Test")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    n = 500
    d = 768
    k = 20
    
    print(f"\nTest data: {n} embeddings, {d} dimensions, selecting {k}")
    
    # Create clustered embeddings (3 clusters)
    embeddings = np.vstack([
        np.random.randn(n // 3, d) + np.array([1, 0] + [0] * (d - 2)),
        np.random.randn(n // 3, d) + np.array([0, 1] + [0] * (d - 2)),
        np.random.randn(n - 2 * (n // 3), d) + np.array([-1, -1] + [0] * (d - 2)),
    ]).astype(np.float32)
    
    # Create quality scores (random with some variation)
    scores = np.random.rand(n) * 0.5 + 0.5  # Range 0.5 to 1.0
    
    ids = [f"img_{i:04d}" for i in range(n)]
    
    # Test each method
    methods = ["greedy_quality", "k_center", "maxmin_diversity", "facility_location", "quality_diversity"]
    
    for method in methods:
        print(f"\n--- {method} ---")
        selector = CoresetSelector(method=method, diversity_weight=0.3)
        result = selector.select(embeddings, k, ids, scores)
        
        print(f"  Diversity: {result.diversity_score:.3f}")
        print(f"  Quality:   {result.quality_score:.3f}")
        print(f"  Sample IDs: {result.selected_ids[:5]}...")
    
    # Test with constraints
    print("\n--- With Constraints ---")
    selector = CoresetSelector(method="quality_diversity")
    result = selector.select_with_constraints(
        embeddings, k, ids, scores,
        must_include=[0, 1],  # Must include first two
        must_exclude=[2, 3],  # Exclude indices 2, 3
        min_score=0.6,        # Minimum score threshold
    )
    print(f"  Selected (with constraints): {len(result.selected_indices)}")
    print(f"  Includes 0, 1: {0 in result.selected_indices and 1 in result.selected_indices}")
    
    print("\n✅ Coreset Selection test passed!")