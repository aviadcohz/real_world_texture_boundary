"""
Test script for dual-GPU pipeline.

Demonstrates:
- Parallel processing on local GPU + remote H100
- Automatic workload splitting
- Batched inference on both GPUs
"""
from models import create_model
from pipelines import run_dual_gpu_pipeline

print("="*70)
print("RUNNING DUAL-GPU PIPELINE")
print("="*70)

# Create local model
model = create_model('qwen-8b', device='cuda')

# Remote H100 server URL
REMOTE_URL = "http://132.66.150.69:8000"

# Number of images to process (None = all images)
num_images = None

results = run_dual_gpu_pipeline(
    local_model=model,
    remote_url=REMOTE_URL,
    image_dir='/datasets/google_landmarks_v2/train_subset/images/',
    output_dir="google_landmarks_v2",
    local_ratio=0.3,          # 30% local, 70% on H100 (faster)
    local_batch_size=4,       # Your local GPU
    remote_batch_size=12,     # H100 can handle larger batches
    num_images=num_images,
    iou_threshold=0.6,
    fix_boundaries=True,
    verbose=True
)

print(f"\nResults saved to: {results['output_dir']}")
