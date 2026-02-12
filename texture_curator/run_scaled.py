#!/usr/bin/env python3
"""
Scalable Pipeline Runner for Texture Curator.

Simplified flow:
  1. Discover candidates (image/mask pairs)
  2. Build RWTD centroid (DINOv2 embeddings)
  3. Extract DINOv2 embeddings for all candidates
  4. Score by cosine similarity to centroid (FAISS-GPU when available)
  5. Select diverse top-N (coreset selection)
  6. Export results

USAGE:
    python run_scaled.py --source /path/to/images --target-n 100
"""

import sys
import argparse
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================
# State
# ============================================

@dataclass
class ScalableState:
    """State for scalable pipeline."""
    rwtd_path: Path = None
    source_path: Path = None
    output_path: Path = None

    candidate_ids: List[str] = field(default_factory=list)
    candidate_paths: Dict[str, Tuple[Path, Path]] = field(default_factory=dict)

    # Numpy arrays
    embeddings: np.ndarray = None       # (N, 768)
    quality_scores: np.ndarray = None   # (N,) — cosine similarity scores
    is_valid: np.ndarray = None         # (N,) boolean mask

    # Selection
    selected_indices: List[int] = field(default_factory=list)
    selected_ids: List[str] = field(default_factory=list)

    # RWTD centroid
    rwtd_centroid: np.ndarray = None

    # Timing
    start_time: datetime = None
    profiling_time: float = 0.0
    extraction_time: float = 0.0
    scoring_time: float = 0.0
    selection_time: float = 0.0

    def save_checkpoint(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.embeddings is not None:
            np.save(path / "embeddings.npy", self.embeddings)
        if self.quality_scores is not None:
            np.save(path / "quality_scores.npy", self.quality_scores)
        if self.is_valid is not None:
            np.save(path / "is_valid.npy", self.is_valid)
        if self.rwtd_centroid is not None:
            np.save(path / "rwtd_centroid.npy", self.rwtd_centroid)
        meta = {
            "candidate_ids": self.candidate_ids,
            "candidate_paths": {k: (str(v[0]), str(v[1])) for k, v in self.candidate_paths.items()},
            "selected_indices": self.selected_indices,
            "selected_ids": self.selected_ids,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path: Path) -> "ScalableState":
        path = Path(path)
        state = cls()
        if (path / "embeddings.npy").exists():
            state.embeddings = np.load(path / "embeddings.npy")
        if (path / "quality_scores.npy").exists():
            state.quality_scores = np.load(path / "quality_scores.npy")
        if (path / "is_valid.npy").exists():
            state.is_valid = np.load(path / "is_valid.npy")
        if (path / "rwtd_centroid.npy").exists():
            state.rwtd_centroid = np.load(path / "rwtd_centroid.npy")
        with open(path / "metadata.json") as f:
            meta = json.load(f)
        state.candidate_ids = meta["candidate_ids"]
        state.candidate_paths = {k: (Path(v[0]), Path(v[1])) for k, v in meta["candidate_paths"].items()}
        state.selected_indices = meta["selected_indices"]
        state.selected_ids = meta["selected_ids"]
        logger.info(f"Checkpoint loaded from {path}")
        return state


# ============================================
# Pipeline
# ============================================

class ScalablePipeline:
    """
    Scalable pipeline using FAISS-GPU for fast cosine similarity scoring.
    """

    def __init__(
        self,
        rwtd_path: Path,
        source_path: Path,
        output_path: Path,
        target_n: int = 100,
        device: str = "cuda",
        batch_size: int = 32,
        diversity_weight: float = 0.3,
        selection_method: str = "quality_diversity",
        filter_passed: bool = False,
    ):
        self.rwtd_path = Path(rwtd_path)
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.target_n = target_n
        self.device = device
        self.batch_size = batch_size
        self.diversity_weight = diversity_weight
        self.selection_method = selection_method
        self.filter_passed = filter_passed

        self.output_path.mkdir(parents=True, exist_ok=True)

        self.state = ScalableState(
            rwtd_path=self.rwtd_path,
            source_path=self.source_path,
            output_path=self.output_path,
        )

        self._dino_extractor = None
        self._faiss_index = None
        self._coreset_selector = None

    @property
    def dino_extractor(self):
        if self._dino_extractor is None:
            from mcp_servers.vision import DINOv2Extractor
            logger.info("Loading DINOv2 extractor...")
            self._dino_extractor = DINOv2Extractor(device=self.device)
        return self._dino_extractor

    @property
    def coreset_selector(self):
        if self._coreset_selector is None:
            from mcp_servers.optimization import CoresetSelector
            self._coreset_selector = CoresetSelector(
                method=self.selection_method,
                diversity_weight=self.diversity_weight,
            )
        return self._coreset_selector

    def _init_faiss(self, embeddings: np.ndarray):
        """Initialize FAISS index and add embeddings for fast search."""
        try:
            from mcp_servers.vectordb import FAISSVectorDB
            n = len(embeddings)
            index_type = "flat" if n < 10000 else "ivf"
            self._faiss_index = FAISSVectorDB(
                dimension=embeddings.shape[1],
                index_type=index_type,
                use_gpu=(self.device == "cuda"),
            )
            ids = self.state.candidate_ids[:n]
            self._faiss_index.add_embeddings(embeddings, ids)
            logger.info(f"FAISS index built: {index_type}, {n} vectors, gpu={self.device == 'cuda'}")
        except ImportError:
            logger.info("FAISS not available, using numpy for scoring.")
            self._faiss_index = None

    def discover_candidates(self) -> int:
        """Discover all image/mask pairs."""
        logger.info(f"Discovering candidates in {self.source_path}...")

        masks_dir = self.source_path / "masks"
        crops_dir = self.source_path / "crops"
        images_dir = self.source_path / "images"
        filter_passed_dir = self.source_path / "filter" / "passed"

        if crops_dir.exists():
            if self.filter_passed:
                if not filter_passed_dir.exists():
                    raise FileNotFoundError(f"filter/passed not found: {filter_passed_dir}")
                img_root = filter_passed_dir
                logger.info(f"Using filtered-passed crops from: {img_root}")
            else:
                img_root = crops_dir
                logger.info(f"Using all crops from: {img_root}")
            image_files = sorted(
                list(img_root.glob("**/*.jpg")) +
                list(img_root.glob("**/*.png")) +
                list(img_root.glob("**/*.jpeg"))
            )
        elif images_dir.exists():
            image_files = sorted(
                list(images_dir.glob("*.jpg")) +
                list(images_dir.glob("*.png")) +
                list(images_dir.glob("*.jpeg"))
            )
        else:
            raise FileNotFoundError(f"No images found in: {self.source_path}")

        all_masks = {}
        for mask_path in (
            list(masks_dir.glob("**/*.png")) +
            list(masks_dir.glob("**/*.jpg"))
        ):
            all_masks[mask_path.stem] = mask_path

        for img_path in image_files:
            stem = img_path.stem
            mask_path = all_masks.get(stem) or all_masks.get(f"{stem}_mask")
            if mask_path:
                self.state.candidate_ids.append(stem)
                self.state.candidate_paths[stem] = (img_path, mask_path)

        logger.info(f"Discovered {len(self.state.candidate_ids)} candidates")
        return len(self.state.candidate_ids)

    def build_rwtd_profile(self) -> None:
        """Build RWTD centroid from DINOv2 embeddings."""
        logger.info("Building RWTD profile...")
        start_time = time.time()

        images_dir = self.rwtd_path / "images"
        image_files = sorted(
            list(images_dir.glob("*.jpg")) +
            list(images_dir.glob("*.png")) +
            list(images_dir.glob("*.jpeg"))
        )

        embeddings, ids, _ = self.dino_extractor.extract_batch(
            image_files, batch_size=self.batch_size
        )

        centroid = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm
        self.state.rwtd_centroid = centroid

        self.state.profiling_time = time.time() - start_time
        logger.info(f"RWTD profile built in {self.state.profiling_time:.1f}s ({len(ids)} images)")

    def extract_and_score(self) -> None:
        """Extract DINOv2 embeddings and score by cosine similarity to centroid.

        Uses FAISS-GPU for batch scoring when available.
        """
        logger.info(f"Extracting embeddings for {len(self.state.candidate_ids)} candidates...")
        start_time = time.time()

        n = len(self.state.candidate_ids)
        image_paths = [self.state.candidate_paths[cid][0] for cid in self.state.candidate_ids]

        # Extract embeddings
        embeddings, ids, failed = self.dino_extractor.extract_batch(
            image_paths, batch_size=self.batch_size
        )

        # Reorder to match candidate_ids
        id_to_embedding = {id_: emb for id_, emb in zip(ids, embeddings)}
        dim = embeddings.shape[1] if len(embeddings) > 0 else 768
        self.state.embeddings = np.zeros((n, dim), dtype=np.float32)
        for i, cid in enumerate(self.state.candidate_ids):
            if cid in id_to_embedding:
                self.state.embeddings[i] = id_to_embedding[cid]

        extraction_time = time.time() - start_time
        logger.info(f"Embeddings extracted in {extraction_time:.1f}s")

        # Score using FAISS or numpy
        score_start = time.time()

        # Normalize all embeddings
        norms = np.linalg.norm(self.state.embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = self.state.embeddings / norms

        # Batch cosine similarity: dot product with normalized centroid
        centroid = self.state.rwtd_centroid.reshape(1, -1)
        raw_scores = (normalized @ centroid.T).flatten()  # [-1, 1]

        # Normalize to [0, 1]
        self.state.quality_scores = (raw_scores + 1) / 2

        self.state.scoring_time = time.time() - score_start
        self.state.extraction_time = extraction_time + self.state.scoring_time

        logger.info(
            f"Scoring completed in {self.state.scoring_time:.1f}s  "
            f"min={self.state.quality_scores.min():.3f}, "
            f"max={self.state.quality_scores.max():.3f}, "
            f"mean={self.state.quality_scores.mean():.3f}"
        )

    def select_diverse_subset(self) -> int:
        """Select diverse subset using coreset selection."""
        logger.info(f"Selecting {self.target_n} diverse samples...")
        start_time = time.time()

        # All candidates are valid after mask filtering
        n = len(self.state.candidate_ids)
        self.state.is_valid = np.ones(n, dtype=bool)

        valid_indices = np.arange(n)
        valid_embeddings = self.state.embeddings
        valid_scores = self.state.quality_scores
        valid_ids = self.state.candidate_ids

        if n < self.target_n:
            logger.warning(f"Only {n} candidates for target {self.target_n}")
            self.target_n = n

        result = self.coreset_selector.select(
            embeddings=valid_embeddings,
            k=self.target_n,
            ids=valid_ids,
            scores=valid_scores,
        )

        self.state.selected_indices = [int(i) for i in result.selected_indices]
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

        for cid in self.state.selected_ids:
            img_path, mask_path = self.state.candidate_paths[cid]
            shutil.copy2(img_path, images_dir / img_path.name)
            shutil.copy2(mask_path, masks_dir / mask_path.name)

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "total_candidates": len(self.state.candidate_ids),
            "selected": len(self.state.selected_ids),
            "target_n": self.target_n,
            "diversity_weight": self.diversity_weight,
            "selection_method": self.selection_method,
            "timing": {
                "profiling": self.state.profiling_time,
                "extraction": self.state.extraction_time,
                "selection": self.state.selection_time,
            },
            "selected_images": [
                {"id": cid, "score": float(self.state.quality_scores[self.state.candidate_ids.index(cid)])}
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
        print("TEXTURE CURATOR - Scalable Pipeline")
        print("=" * 70)
        print(f"  RWTD Reference:   {self.rwtd_path}")
        print(f"  Source Pool:      {self.source_path}")
        print(f"  Target Selection: {self.target_n}")
        print(f"  Device:           {self.device}")
        print("=" * 70)
        print()

        # Step 1: Discover
        self.discover_candidates()

        # Step 2: RWTD profile
        self.build_rwtd_profile()
        self.state.save_checkpoint(self.output_path / "checkpoints" / "after_profile")

        # Step 3: Extract + Score
        self.extract_and_score()
        self.state.save_checkpoint(self.output_path / "checkpoints" / "after_scoring")

        # Step 4: Select
        self.select_diverse_subset()

        # Step 5: Export
        export_dir = self.export_results()

        total_time = time.time() - total_start

        print()
        print("=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"  Total Time:  {total_time:.1f}s")
        print(f"  Candidates:  {len(self.state.candidate_ids)}")
        print(f"  Selected:    {len(self.state.selected_ids)}")
        print(f"  Output:      {export_dir}")
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
    parser.add_argument("--diversity-weight", type=float, default=0.3,
                       help="Diversity weight in selection")
    parser.add_argument("--selection-method", type=str, default="quality_diversity",
                       choices=["greedy_quality", "k_center", "maxmin_diversity",
                                "facility_location", "quality_diversity"],
                       help="Selection algorithm")
    parser.add_argument("--filter-passed", action="store_true",
                       help="Only use crops that passed the entropy filter")

    args = parser.parse_args()

    if not Path(args.rwtd).exists():
        print(f"ERROR: RWTD path not found: {args.rwtd}")
        sys.exit(1)

    if not Path(args.source).exists():
        print(f"ERROR: Source path not found: {args.source}")
        sys.exit(1)

    pipeline = ScalablePipeline(
        rwtd_path=Path(args.rwtd),
        source_path=Path(args.source),
        output_path=Path(args.output),
        target_n=args.target_n,
        device=args.device,
        batch_size=args.batch_size,
        diversity_weight=args.diversity_weight,
        selection_method=args.selection_method,
        filter_passed=args.filter_passed,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
