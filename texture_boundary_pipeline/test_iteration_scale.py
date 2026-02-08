"""
Test script for iterative dual-GPU pipeline.

Combines:
- Parallel grounding on local GPU + remote H100 (Qwen)
- Entropy-based filtering with boundary masks (Sa2VA)
- Automatic workload splitting

This is the scaled version of test_iteration.py
"""
from models import create_model
from pipelines import run_iterative_dual_gpu_pipeline

print("="*70)
print("RUNNING ITERATIVE DUAL-GPU PIPELINE")
print("="*70)

# Create local model (Qwen)
model = create_model('qwen-8b', device='cuda')

# Remote H100 server URL
REMOTE_URL = "http://132.66.150.69:8000"

# Number of images to process (None = all images)
num_images = 20

results = run_iterative_dual_gpu_pipeline(
    local_model=model,
    remote_url=REMOTE_URL,
    image_dir='/datasets/google_landmarks_v2/train_subset/images/',
    output_dir="results_scale",
    local_ratio=0.3,              # 100% local, 0% remote (adjust for your needs)
    local_batch_size=4,           # Your local GPU batch size
    remote_batch_size=12,         # H100 can handle larger batches
    num_images=num_images,
    extract_masks=True,
    entropy_threshold=3.5,
    verbose=True
)

print(f"\n📁 Results saved to: {results['output_dir']}")

