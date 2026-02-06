#!/usr/bin/env python3
"""
Scalable Pipeline Runner for Texture Curator.

Handles thousands of images efficiently using:
- Batched GPU processing
- FAISS-GPU for similarity search
- Advanced coreset selection
- Checkpointing for long runs

USAGE:
    python run_scaled.py --source /path/to/images --target-n 100
    
SCALING:
    - 1K images: ~2 minutes
    - 10K images: ~15 minutes
    - 100K images: ~2 hours (with IVF index)
"""

import sys
import argparse
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================
# Scalable State
# ============================================

@dataclass
class ScalableState:
    """State for scalable pipeline."""
    
    # Paths
    rwtd_path: Path = None
    source_path: Path = None
    output_path: Path = None
    
    # Candidate tracking
    candidate_ids: List[str] = field(default_factory=list)
    candidate_paths: Dict[str, Tuple[Path, Path]] = field(default_factory=dict)  # id -> (image, mask)
    
    # Scores (numpy arrays for efficiency)
    embeddings: np.ndarray = None  # (N, 768)
    quality_scores: np.ndarray = None  # (N,)
    semantic_scores: np.ndarray = None  # (N,)
    texture_scores: np.ndarray = None  # (N,)
    boundary_scores: np.ndarray = None  # (N,)
    
    # Validation
    is_valid: np.ndarray = None  # (N,) boolean mask
    
    # Selection
    selected_indices: List[int] = field(default_factory=list)
    selected_ids: List[str] = field(default_factory=list)
    
    # RWTD Profile
    rwtd_centroid: np.ndarray = None
    rwtd_entropy_mean: float = 0.0
    rwtd_entropy_std: float = 0.0
    rwtd_vol_mean: float = 0.0
    rwtd_vol_std: float = 0.0
    
    # Stats
    start_time: datetime = None
    profiling_time: float = 0.0
    extraction_time: float = 0.0
    scoring_time: float = 0.0
    selection_time: float = 0.0
    
    def save_checkpoint(self, path: Path):
        """Save state to checkpoint file."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save numpy arrays separately
        if self.embeddings is not None:
            np.save(path / "embeddings.npy", self.embeddings)
        if self.quality_scores is not None:
            np.save(path / "quality_scores.npy", self.quality_scores)
        if self.is_valid is not None:
            np.save(path / "is_valid.npy", self.is_valid)
        if self.rwtd_centroid is not None:
            np.save(path / "rwtd_centroid.npy", self.rwtd_centroid)
        
        # Save metadata
        meta = {
            "candidate_ids": self.candidate_ids,
            "candidate_paths": {k: (str(v[0]), str(v[1])) for k, v in self.candidate_paths.items()},
            "selected_indices": self.selected_indices,
            "selected_ids": self.selected_ids,
            "rwtd_entropy_mean": self.rwtd_entropy_mean,
            "rwtd_entropy_std": self.rwtd_entropy_std,
            "rwtd_vol_mean": self.rwtd_vol_mean,
            "rwtd_vol_std": self.rwtd_vol_std,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"Checkpoint saved to {path}")
    
    @classmethod
    def load_checkpoint(cls, path: Path) -> "ScalableState":
        """Load state from checkpoint."""
        path = Path(path)
        state = cls()
        
        # Load numpy arrays
        if (path / "embeddings.npy").exists():
            state.embeddings = np.load(path / "embeddings.npy")
        if (path / "quality_scores.npy").exists():
            state.quality_scores = np.load(path / "quality_scores.npy")
        if (path / "is_valid.npy").exists():
            state.is_valid = np.load(path / "is_valid.npy")
        if (path / "rwtd_centroid.npy").exists():
            state.rwtd_centroid = np.load(path / "rwtd_centroid.npy")
        
        # Load metadata
        with open(path / "metadata.json") as f:
            meta = json.load(f)
        
        state.candidate_ids = meta["candidate_ids"]
        state.candidate_paths = {k: (Path(v[0]), Path(v[1])) for k, v in meta["candidate_paths"].items()}
        state.selected_indices = meta["selected_indices"]
        state.selected_ids = meta["selected_ids"]
        state.rwtd_entropy_mean = meta["rwtd_entropy_mean"]
        state.rwtd_entropy_std = meta["rwtd_entropy_std"]
        state.rwtd_vol_mean = meta["rwtd_vol_mean"]
        state.rwtd_vol_std = meta["rwtd_vol_std"]
        
        logger.info(f"Checkpoint loaded from {path}")
        return state


# ============================================
# Scalable Pipeline
# ============================================

class ScalablePipeline:
    """
    Scalable pipeline for processing thousands of images.
    
    Uses:
    - Batched DINOv2 extraction
    - FAISS-GPU for similarity
    - CoresetSelector for diverse selection
    """
    
    def __init__(
        self,
        rwtd_path: Path,
        source_path: Path,
        output_path: Path,
        target_n: int = 100,
        device: str = "cuda",
        batch_size: int = 32,
        quality_threshold: float = 0.3,
        diversity_weight: float = 0.3,
        selection_method: str = "quality_diversity",
    ):
        self.rwtd_path = Path(rwtd_path)
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.target_n = target_n
        self.device = device
        self.batch_size = batch_size
        self.quality_threshold = quality_threshold
        self.diversity_weight = diversity_weight
        self.selection_method = selection_method
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # State
        self.state = ScalableState(
            rwtd_path=self.rwtd_path,
            source_path=self.source_path,
            output_path=self.output_path,
        )
        
        # Lazy-loaded extractors
        self._dino_extractor = None
        self._texture_extractor = None
        self._boundary_extractor = None
        self._faiss_db = None
        self._coreset_selector = None
    
    @property
    def dino_extractor(self):
        """Lazy load DINOv2 extractor."""
        if self._dino_extractor is None:
            from mcp_servers.vision import DINOv2Extractor
            logger.info("Loading DINOv2 extractor...")
            self._dino_extractor = DINOv2Extractor(device=self.device)
        return self._dino_extractor
    
    @property
    def texture_extractor(self):
        """Lazy load texture extractor."""
        if self._texture_extractor is None:
            from mcp_servers.vision import TextureStatsExtractor
            self._texture_extractor = TextureStatsExtractor()
        return self._texture_extractor
    
    @property
    def boundary_extractor(self):
        """Lazy load boundary extractor."""
        if self._boundary_extractor is None:
            from mcp_servers.vision import BoundaryMetricsExtractor
            self._boundary_extractor = BoundaryMetricsExtractor()
        return self._boundary_extractor
    
    @property
    def faiss_db(self):
        """Lazy load FAISS database."""
        if self._faiss_db is None:
            try:
                from mcp_servers.vectordb import FAISSVectorDB
                n_expected = len(self.state.candidate_ids) or 10000
                
                # Choose index type based on size
                if n_expected < 10000:
                    index_type = "flat"
                else:
                    index_type = "ivf"
                
                self._faiss_db = FAISSVectorDB(
                    dimension=768,
                    index_type=index_type,
                    use_gpu=(self.device == "cuda"),
                )
            except ImportError:
                logger.warning("FAISS not available. Using numpy fallback.")
                self._faiss_db = None
        return self._faiss_db
    
    @property
    def coreset_selector(self):
        """Lazy load coreset selector."""
        if self._coreset_selector is None:
            from mcp_servers.optimization import CoresetSelector
            self._coreset_selector = CoresetSelector(
                method=self.selection_method,
                diversity_weight=self.diversity_weight,
            )
        return self._coreset_selector
    
    def discover_candidates(self) -> int:
        """Discover all image/mask pairs in source directory."""
        logger.info(f"Discovering candidates in {self.source_path}...")
        
        images_dir = self.source_path / "images"
        masks_dir = self.source_path / "masks"
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        # Find all images
        image_files = sorted(
            list(images_dir.glob("*.jpg")) +
            list(images_dir.glob("*.png")) +
            list(images_dir.glob("*.jpeg"))
        )
        
        # Find corresponding masks
        for img_path in image_files:
            # Try different mask naming conventions
            mask_candidates = [
                masks_dir / f"{img_path.stem}.png",
                masks_dir / f"{img_path.stem}_mask.png",
                masks_dir / f"{img_path.stem}.jpg",
            ]
            
            mask_path = None
            for candidate in mask_candidates:
                if candidate.exists():
                    mask_path = candidate
                    break
            
            if mask_path:
                cid = img_path.stem
                self.state.candidate_ids.append(cid)
                self.state.candidate_paths[cid] = (img_path, mask_path)
        
        logger.info(f"Discovered {len(self.state.candidate_ids)} candidates")
        return len(self.state.candidate_ids)
    
    def build_rwtd_profile(self) -> None:
        """Build reference profile from RWTD."""
        logger.info("Building RWTD profile...")
        start_time = time.time()
        
        images_dir = self.rwtd_path / "images"
        masks_dir = self.rwtd_path / "masks"
        
        # Find RWTD images
        image_files = sorted(
            list(images_dir.glob("*.jpg")) +
            list(images_dir.glob("*.png")) +
            list(images_dir.glob("*.jpeg"))
        )
        
        # Extract embeddings
        embeddings, ids, _ = self.dino_extractor.extract_batch(
            image_files, batch_size=self.batch_size
        )
        
        # Compute centroid
        self.state.rwtd_centroid = np.mean(embeddings, axis=0)
        
        # Normalize centroid
        norm = np.linalg.norm(self.state.rwtd_centroid)
        if norm > 0:
            self.state.rwtd_centroid /= norm
        
        # Extract texture stats
        texture_stats = self.texture_extractor.compute_batch(image_files, show_progress=True)
        entropies = [s.entropy_mean for s in texture_stats if s.success]
        self.state.rwtd_entropy_mean = np.mean(entropies)
        self.state.rwtd_entropy_std = np.std(entropies)
        
        # Extract boundary metrics (if masks exist)
        mask_paths = []
        valid_images = []
        for img_path in image_files:
            mask_candidates = [
                masks_dir / f"{img_path.stem}.png",
                masks_dir / f"{img_path.stem}_mask.png",
            ]
            for mask_path in mask_candidates:
                if mask_path.exists():
                    mask_paths.append(mask_path)
                    valid_images.append(img_path)
                    break
        
        if mask_paths:
            boundary_metrics = self.boundary_extractor.compute_batch(
                valid_images, mask_paths, show_progress=True
            )
            vols = [m.variance_of_laplacian for m in boundary_metrics if m.success]
            self.state.rwtd_vol_mean = np.mean(vols) if vols else 100.0
            self.state.rwtd_vol_std = np.std(vols) if vols else 50.0
        
        self.state.profiling_time = time.time() - start_time
        logger.info(f"RWTD profile built in {self.state.profiling_time:.1f}s")
        logger.info(f"  Centroid shape: {self.state.rwtd_centroid.shape}")
        logger.info(f"  Entropy: {self.state.rwtd_entropy_mean:.2f} ± {self.state.rwtd_entropy_std:.2f}")
        logger.info(f"  VoL: {self.state.rwtd_vol_mean:.2f} ± {self.state.rwtd_vol_std:.2f}")
    
    def extract_features(self) -> None:
        """Extract features for all candidates."""
        logger.info(f"Extracting features for {len(self.state.candidate_ids)} candidates...")
        start_time = time.time()
        
        n = len(self.state.candidate_ids)
        
        # Get paths in order
        image_paths = [self.state.candidate_paths[cid][0] for cid in self.state.candidate_ids]
        mask_paths = [self.state.candidate_paths[cid][1] for cid in self.state.candidate_ids]
        
        # 1. Extract DINOv2 embeddings
        logger.info("  Extracting DINOv2 embeddings...")
        embeddings, ids, failed = self.dino_extractor.extract_batch(
            image_paths, batch_size=self.batch_size
        )
        
        # Create ID to embedding mapping
        id_to_embedding = {id_: emb for id_, emb in zip(ids, embeddings)}
        
        # Reorder to match candidate_ids
        self.state.embeddings = np.zeros((n, 768), dtype=np.float32)
        for i, cid in enumerate(self.state.candidate_ids):
            if cid in id_to_embedding:
                self.state.embeddings[i] = id_to_embedding[cid]
        
        # 2. Extract texture stats
        logger.info("  Extracting texture stats...")
        texture_stats = self.texture_extractor.compute_batch(image_paths, show_progress=True)
        id_to_texture = {s.image_id: s for s in texture_stats if s.success}
        
        # 3. Extract boundary metrics
        logger.info("  Extracting boundary metrics...")
        boundary_metrics = self.boundary_extractor.compute_batch(
            image_paths, mask_paths, show_progress=True
        )
        id_to_boundary = {m.image_id: m for m in boundary_metrics if m.success}
        
        # 4. Compute scores
        logger.info("  Computing scores...")
        self.state.semantic_scores = np.zeros(n, dtype=np.float32)
        self.state.texture_scores = np.zeros(n, dtype=np.float32)
        self.state.boundary_scores = np.zeros(n, dtype=np.float32)
        
        for i, cid in enumerate(self.state.candidate_ids):
            # Semantic score (cosine similarity to centroid)
            emb = self.state.embeddings[i]
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb_normalized = emb / norm
                self.state.semantic_scores[i] = float(np.dot(emb_normalized, self.state.rwtd_centroid))
                # Convert from [-1, 1] to [0, 1]
                self.state.semantic_scores[i] = (self.state.semantic_scores[i] + 1) / 2
            
            # Texture score
            if cid in id_to_texture:
                tex = id_to_texture[cid]
                # Score based on how close to RWTD mean
                z_score = abs(tex.entropy_mean - self.state.rwtd_entropy_mean) / max(self.state.rwtd_entropy_std, 0.1)
                self.state.texture_scores[i] = np.exp(-z_score / 2)  # Gaussian-like scoring
            
            # Boundary score
            if cid in id_to_boundary:
                bound = id_to_boundary[cid]
                z_score = abs(bound.variance_of_laplacian - self.state.rwtd_vol_mean) / max(self.state.rwtd_vol_std, 10)
                self.state.boundary_scores[i] = np.exp(-z_score / 2)
        
        # Combined quality score
        self.state.quality_scores = (
            0.4 * self.state.semantic_scores +
            0.3 * self.state.texture_scores +
            0.3 * self.state.boundary_scores
        )
        
        self.state.extraction_time = time.time() - start_time
        logger.info(f"Features extracted in {self.state.extraction_time:.1f}s")
        logger.info(f"  Score distribution: min={self.state.quality_scores.min():.3f}, "
                   f"max={self.state.quality_scores.max():.3f}, "
                   f"mean={self.state.quality_scores.mean():.3f}")
    
    def validate_candidates(self) -> int:
        """Validate candidates (simple threshold for now)."""
        logger.info("Validating candidates...")
        
        # Mark as valid if quality score above threshold
        self.state.is_valid = self.state.quality_scores >= self.quality_threshold
        
        n_valid = int(np.sum(self.state.is_valid))
        logger.info(f"Valid candidates: {n_valid}/{len(self.state.candidate_ids)}")
        
        return n_valid
    
    def select_diverse_subset(self) -> int:
        """Select diverse subset using coreset selection."""
        logger.info(f"Selecting {self.target_n} diverse samples...")
        start_time = time.time()
        
        # Get valid candidates only
        valid_indices = np.where(self.state.is_valid)[0]
        valid_embeddings = self.state.embeddings[valid_indices]
        valid_scores = self.state.quality_scores[valid_indices]
        valid_ids = [self.state.candidate_ids[i] for i in valid_indices]
        
        logger.info(f"  Selecting from {len(valid_indices)} valid candidates")
        
        if len(valid_indices) < self.target_n:
            logger.warning(f"  Not enough valid candidates ({len(valid_indices)}) for target ({self.target_n})")
            self.target_n = len(valid_indices)
        
        # Use coreset selector
        result = self.coreset_selector.select(
            embeddings=valid_embeddings,
            k=self.target_n,
            ids=valid_ids,
            scores=valid_scores,
        )
        
        # Map back to original indices
        self.state.selected_indices = [int(valid_indices[i]) for i in result.selected_indices]
        self.state.selected_ids = result.selected_ids
        
        self.state.selection_time = time.time() - start_time
        logger.info(f"Selection completed in {self.state.selection_time:.1f}s")
        logger.info(f"  Selected: {len(self.state.selected_ids)}")
        logger.info(f"  Diversity score: {result.diversity_score:.3f}")
        logger.info(f"  Quality score: {result.quality_score:.3f}")
        
        return len(self.state.selected_ids)
    
    def export_results(self) -> Path:
        """Export selected images to output directory."""
        logger.info("Exporting results...")
        import shutil
        
        export_dir = self.output_path / "curated_dataset"
        images_dir = export_dir / "images"
        masks_dir = export_dir / "masks"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy selected files
        for cid in self.state.selected_ids:
            img_path, mask_path = self.state.candidate_paths[cid]
            
            shutil.copy2(img_path, images_dir / img_path.name)
            shutil.copy2(mask_path, masks_dir / mask_path.name)
        
        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "total_candidates": len(self.state.candidate_ids),
            "valid_candidates": int(np.sum(self.state.is_valid)),
            "selected": len(self.state.selected_ids),
            "target_n": self.target_n,
            "quality_threshold": self.quality_threshold,
            "diversity_weight": self.diversity_weight,
            "selection_method": self.selection_method,
            "timing": {
                "profiling": self.state.profiling_time,
                "extraction": self.state.extraction_time,
                "selection": self.state.selection_time,
            },
            "selected_images": [
                {
                    "id": cid,
                    "score": float(self.state.quality_scores[self.state.candidate_ids.index(cid)]),
                }
                for cid in self.state.selected_ids
            ],
        }
        
        with open(export_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Exported {len(self.state.selected_ids)} images to {export_dir}")
        
        return export_dir
    
    def run(self) -> ScalableState:
        """Run the full pipeline."""
        total_start = time.time()
        self.state.start_time = datetime.now()
        
        print()
        print("=" * 70)
        print("🎨 TEXTURE CURATOR - Scalable Pipeline")
        print("=" * 70)
        print(f"  RWTD Reference:  {self.rwtd_path}")
        print(f"  Source Pool:     {self.source_path}")
        print(f"  Target Selection: {self.target_n}")
        print(f"  Device:          {self.device}")
        print("=" * 70)
        print()
        
        # Step 1: Discover candidates
        self.discover_candidates()
        
        # Step 2: Build RWTD profile
        self.build_rwtd_profile()
        
        # Save checkpoint
        self.state.save_checkpoint(self.output_path / "checkpoints" / "after_profile")
        
        # Step 3: Extract features
        self.extract_features()
        
        # Save checkpoint
        self.state.save_checkpoint(self.output_path / "checkpoints" / "after_extraction")
        
        # Step 4: Validate
        self.validate_candidates()
        
        # Step 5: Select diverse subset
        self.select_diverse_subset()
        
        # Step 6: Export
        export_dir = self.export_results()
        
        total_time = time.time() - total_start
        
        print()
        print("=" * 70)
        print("✅ PIPELINE COMPLETE")
        print("=" * 70)
        print(f"  Total Time: {total_time:.1f}s")
        print(f"  Candidates: {len(self.state.candidate_ids)}")
        print(f"  Valid: {int(np.sum(self.state.is_valid))}")
        print(f"  Selected: {len(self.state.selected_ids)}")
        print()
        print(f"  Output: {export_dir}")
        print("=" * 70)
        
        return self.state


# ============================================
# CLI
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Texture Curator - Scalable Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--rwtd", type=str, default="/home/aviad/RWTD",
                       help="Path to RWTD reference dataset")
    parser.add_argument("--source", type=str, default="google_landmarks_v2/run_20260205_172235",
                       help="Path to source pool")
    parser.add_argument("--output", type=str, default="./outputs_scaled",
                       help="Output directory")
    parser.add_argument("--target-n", type=int, default=24120,
                       help="Number of images to select")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for extraction")
    parser.add_argument("--quality-threshold", type=float, default=0.3,
                       help="Minimum quality score")
    parser.add_argument("--diversity-weight", type=float, default=0.3,
                       help="Diversity weight in selection")
    parser.add_argument("--selection-method", type=str, default="quality_diversity",
                       choices=["greedy_quality", "k_center", "maxmin_diversity", "facility_location", "quality_diversity"],
                       help="Selection algorithm")
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.rwtd).exists():
        print(f"ERROR: RWTD path not found: {args.rwtd}")
        sys.exit(1)
    
    if not Path(args.source).exists():
        print(f"ERROR: Source path not found: {args.source}")
        sys.exit(1)
    
    # Run pipeline
    pipeline = ScalablePipeline(
        rwtd_path=Path(args.rwtd),
        source_path=Path(args.source),
        output_path=Path(args.output),
        target_n=args.target_n,
        device=args.device,
        batch_size=args.batch_size,
        quality_threshold=args.quality_threshold,
        diversity_weight=args.diversity_weight,
        selection_method=args.selection_method,
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()