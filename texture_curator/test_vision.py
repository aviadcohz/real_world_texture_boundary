#!/usr/bin/env python3
"""
Test script for Vision MCP Server components.

Tests:
1. DINOv2 Feature Extractor
2. Texture Statistics Extractor
3. Boundary Metrics Extractor

Usage:
    cd ~/texture_curator
    python test_vision.py
    
    # Or test with your actual data:
    python test_vision.py --rwtd /home/aviad/RWTD --num-samples 5
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_dino_extractor(use_gpu: bool = True):
    """Test DINOv2 feature extractor."""
    print("=" * 60)
    print("Testing: DINOv2 Feature Extractor")
    print("=" * 60)
    
    from mcp_servers.vision.dino_extractor import DINOv2Extractor
    
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load extractor
    print("\nLoading DINOv2 model...")
    start_time = time.time()
    extractor = DINOv2Extractor(
        model_name="dinov2_vitb14",
        device=device,
    )
    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.2f}s")
    print(f"✓ Embedding dimension: {extractor.embedding_dim}")
    
    # Create test images
    print("\nCreating synthetic test images...")
    test_images = []
    
    # Texture-like patterns
    # 1. Brick-like pattern
    brick = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(0, 256, 32):
        for j in range(0, 256, 64):
            offset = 32 if (i // 32) % 2 else 0
            brick[i:i+30, (j+offset):(j+offset+60)] = [180, 100, 80]
    brick[np.all(brick == 0, axis=2)] = [200, 180, 160]
    test_images.append(("brick", Image.fromarray(brick)))
    
    # 2. Grass-like pattern
    grass = np.random.randint(30, 80, (256, 256, 3), dtype=np.uint8)
    grass[:, :, 1] = np.random.randint(80, 150, (256, 256), dtype=np.uint8)
    test_images.append(("grass", Image.fromarray(grass)))
    
    # 3. Uniform surface
    uniform = np.ones((256, 256, 3), dtype=np.uint8) * 128
    test_images.append(("uniform", Image.fromarray(uniform)))
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings = {}
    for name, img in test_images:
        start = time.time()
        emb = extractor.extract_single(img)
        elapsed = time.time() - start
        embeddings[name] = emb
        print(f"  {name}: shape={emb.shape}, norm={np.linalg.norm(emb):.3f}, time={elapsed:.3f}s")
    
    # Compute similarities
    print("\nComputing cosine similarities:")
    all_emb = np.stack([embeddings[n] for n, _ in test_images])
    
    for i, (name1, _) in enumerate(test_images):
        sims = extractor.cosine_similarity(all_emb, embeddings[name1])
        for j, (name2, _) in enumerate(test_images):
            print(f"  {name1} ↔ {name2}: {sims[j]:.4f}")
    
    print("\n✓ DINOv2 test passed!")
    return True


def test_texture_stats():
    """Test texture statistics extractor."""
    print("\n" + "=" * 60)
    print("Testing: Texture Statistics Extractor")
    print("=" * 60)
    
    from mcp_servers.vision.texture_stats import TextureStatsExtractor
    
    extractor = TextureStatsExtractor()
    print("✓ Extractor initialized")
    
    # Create test images
    print("\nCreating test images with different textures...")
    
    # 1. Uniform (low entropy)
    uniform = Image.new("RGB", (256, 256), color=(128, 128, 128))
    
    # 2. Noisy (high entropy)
    noise = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    noisy = Image.fromarray(noise)
    
    # 3. Structured (medium entropy, high contrast)
    checker = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            if ((i // 32) + (j // 32)) % 2:
                checker[i, j] = [200, 200, 200]
            else:
                checker[i, j] = [50, 50, 50]
    structured = Image.fromarray(checker)
    
    # Compute stats
    test_images = [("uniform", uniform), ("noisy", noisy), ("structured", structured)]
    
    print("\nComputing texture statistics:")
    for name, img in test_images:
        stats = extractor.compute_stats(img)
        print(f"\n  {name}:")
        print(f"    Entropy: {stats.entropy_mean:.2f} (±{stats.entropy_std:.2f})")
        print(f"    GLCM Contrast: {stats.glcm_contrast:.4f}")
        print(f"    GLCM Homogeneity: {stats.glcm_homogeneity:.4f}")
        print(f"    GLCM Energy: {stats.glcm_energy:.6f}")
    
    print("\n✓ Texture stats test passed!")
    return True


def test_boundary_metrics():
    """Test boundary metrics extractor."""
    print("\n" + "=" * 60)
    print("Testing: Boundary Metrics Extractor")
    print("=" * 60)
    
    from mcp_servers.vision.boundary_metrics import BoundaryMetricsExtractor
    
    extractor = BoundaryMetricsExtractor()
    print("✓ Extractor initialized")
    
    # Create test cases
    print("\nCreating test image/mask pairs...")
    
    # Case 1: Good boundary (mask on sharp edge)
    good_image = np.zeros((256, 256), dtype=np.uint8)
    good_image[:, :128] = 50
    good_image[:, 128:] = 200
    good_mask = np.zeros((256, 256), dtype=np.uint8)
    good_mask[:, 128:] = 255
    
    # Case 2: Bad boundary (mask on uniform region)
    bad_image = np.ones((256, 256), dtype=np.uint8) * 128
    bad_mask = good_mask.copy()
    
    # Case 3: Diagonal edge
    diag_image = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            if i + j < 256:
                diag_image[i, j] = 50
            else:
                diag_image[i, j] = 200
    diag_mask = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            if i + j >= 256:
                diag_mask[i, j] = 255
    
    # Compute metrics
    test_cases = [
        ("good_edge", good_image, good_mask),
        ("bad_edge", bad_image, bad_mask),
        ("diagonal", diag_image, diag_mask),
    ]
    
    print("\nComputing boundary metrics:")
    for name, img, mask in test_cases:
        metrics = extractor.compute_metrics(img, mask)
        print(f"\n  {name}:")
        print(f"    Variance of Laplacian: {metrics.variance_of_laplacian:.2f}")
        print(f"    Gradient Mean: {metrics.gradient_magnitude_mean:.2f}")
        print(f"    Edge Density: {metrics.edge_density:.2%}")
        print(f"    Boundary Length: {metrics.boundary_length} pixels")
    
    print("\n✓ Boundary metrics test passed!")
    return True


def test_with_real_data(rwtd_path: str, num_samples: int = 5):
    """Test with actual RWTD data."""
    print("\n" + "=" * 60)
    print("Testing: With Real RWTD Data")
    print("=" * 60)
    
    from mcp_servers.vision.dino_extractor import DINOv2Extractor
    from mcp_servers.vision.texture_stats import TextureStatsExtractor
    from mcp_servers.vision.boundary_metrics import BoundaryMetricsExtractor
    
    rwtd_path = Path(rwtd_path)
    
    # Find image and mask directories
    images_dir = rwtd_path / "images"
    masks_dir = rwtd_path / "masks"
    
    if not images_dir.exists():
        print(f"✗ Images directory not found: {images_dir}")
        return False
    
    if not masks_dir.exists():
        print(f"✗ Masks directory not found: {masks_dir}")
        return False
    
    # Get image files
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))[:num_samples]
    
    if not image_files:
        print(f"✗ No images found in {images_dir}")
        return False
    
    print(f"Found {len(image_files)} images to process")
    
    # Create extractors
    print("\nLoading extractors...")
    dino = DINOv2Extractor(device="cuda" if torch.cuda.is_available() else "cpu")
    texture = TextureStatsExtractor()
    boundary = BoundaryMetricsExtractor()
    
    # Process each image
    print(f"\nProcessing {len(image_files)} samples:")
    
    for img_path in image_files:
        print(f"\n  {img_path.name}:")
        
        # Find corresponding mask
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            # Try with different extension
            mask_path = masks_dir / (img_path.stem + ".png")
        
        if not mask_path.exists():
            print(f"    ⚠ Mask not found, skipping boundary metrics")
            mask_path = None
        
        # DINOv2 embedding
        emb = dino.extract_single(img_path)
        print(f"    DINOv2: shape={emb.shape}, norm={np.linalg.norm(emb):.2f}")
        
        # Texture stats
        tstats = texture.compute_stats(img_path)
        print(f"    Texture: entropy={tstats.entropy_mean:.2f}, contrast={tstats.glcm_contrast:.2f}")
        
        # Boundary metrics (if mask exists)
        if mask_path:
            bmetrics = boundary.compute_metrics(img_path, mask_path)
            print(f"    Boundary: VoL={bmetrics.variance_of_laplacian:.2f}, edge_density={bmetrics.edge_density:.2%}")
    
    print("\n✓ Real data test passed!")
    return True


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test Vision MCP Server")
    parser.add_argument("--rwtd", type=str, help="Path to RWTD dataset for real data test")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to test")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    args = parser.parse_args()
    
    print()
    print("🔬 TEXTURE CURATOR - VISION MODULE TESTS")
    print("=" * 60)
    
    if torch.cuda.is_available() and not args.cpu:
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ Running on CPU (slower)")
    print()
    
    results = []
    
    # Run synthetic tests
    try:
        results.append(("DINOv2 Extractor", test_dino_extractor(not args.cpu)))
    except Exception as e:
        print(f"✗ DINOv2 test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("DINOv2 Extractor", False))
    
    try:
        results.append(("Texture Stats", test_texture_stats()))
    except Exception as e:
        print(f"✗ Texture stats test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Texture Stats", False))
    
    try:
        results.append(("Boundary Metrics", test_boundary_metrics()))
    except Exception as e:
        print(f"✗ Boundary metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Boundary Metrics", False))
    
    # Run real data test if path provided
    if args.rwtd:
        try:
            results.append(("Real RWTD Data", test_with_real_data(args.rwtd, args.num_samples)))
        except Exception as e:
            print(f"✗ Real data test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(("Real RWTD Data", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All vision tests passed!")
    else:
        print("⚠️  Some tests failed. Please fix before continuing.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)