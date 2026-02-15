"""Run iterative pipeline with entropy-based filtering."""
from models import create_model
from pipelines import run_iterative_pipeline


# Create model (only Qwen - Sa2VA will be loaded after Qwen is unloaded)
model = create_model('qwen-8b', device='cuda')

# Run iterative pipeline
print("="*70)
print("RUNNING ITERATIVE PIPELINE WITH ENTROPY FILTER")
print("="*70)

results = run_iterative_pipeline(
    model=model,
    image_dir='/home/aviad/RWTD/images/',
    output_dir="RWTD_iterative_results_full/",
    dataset_base_dir='/datasets/debug/',
    entropy_threshold=3.5,  # Minimum entropy for both texture regions
    extract_masks=True,
    num_images=None,
    verbose=True
)

print(f"📁 Results saved to: {results['output_dir']}")
