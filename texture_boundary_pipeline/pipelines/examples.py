"""
Example Usage Scripts

Quick examples to get started with the pipeline.
"""

# ==============================================================================
# EXAMPLE 1: Basic Pipeline - Simplest Usage
# ==============================================================================

def example_1_basic_simple():
    """Run basic pipeline on a directory of images."""
    from models import create_model
    from pipelines import run_basic_pipeline
    
    # Create model
    model = create_model('qwen', device='cuda')
    
    # Run pipeline
    results = run_basic_pipeline(
        model=model,
        image_dir='/path/to/your/images',
        output_dir='results'
    )
    
    print(f"✅ Complete! Results in: {results['output_dir']}")


# ==============================================================================
# EXAMPLE 2: Basic Pipeline - Custom Settings
# ==============================================================================

def example_2_basic_custom():
    """Run basic pipeline with custom settings."""
    from models import create_model
    from pipelines import BasicPipeline
    from utils import list_images
    
    # Create model
    model = create_model('qwen', device='cuda')
    
    # List images
    images = list_images('/path/to/your/images')
    
    # Create pipeline with custom settings
    pipeline = BasicPipeline(
        model=model,
        output_dir='my_results',
        iou_threshold=0.7,          # Higher = stricter overlap filtering
        fix_boundaries=True,         # Fix tiny boxes near edges
        target_size=1024,            # Padding size
        verbose=True                 # Show progress
    )
    
    # Run
    results = pipeline.run(images)
    
    print(f"Processed {results['input']['num_images']} images")
    print(f"Found {results['processing']['output_boxes']} boundaries")


# ==============================================================================
# EXAMPLE 3: Iterative Pipeline - Refinement
# ==============================================================================

def example_3_iterative():
    """Run iterative pipeline with classification and refinement."""
    from models import create_model
    from pipelines import run_iterative_pipeline
    
    # Create model
    model = create_model('qwen', device='cuda')
    
    # Run iterative pipeline
    results = run_iterative_pipeline(
        model=model,
        image_dir='/path/to/your/images',
        output_dir='results',
        semantic_threshold=7.0,      # Confidence threshold (0-10)
        max_iterations=2,            # Max refinement iterations
        crop_categories=['medium', 'large', 'xlarge'],  # Which crops to classify
        num_images=10,              # Process only first 10 images
        verbose=True
    )
    
    print(f"Iterations completed: {results['iterations_completed']}")


# ==============================================================================
# EXAMPLE 4: Custom Prompts
# ==============================================================================

def example_4_custom_prompts():
    """Use custom prompts instead of default."""
    from models import create_model
    from pipelines import BasicPipeline
    from utils import list_images
    
    # Custom grounding prompt
    my_prompt = """
    Find texture boundaries in this image.
    Focus on material changes (wood, metal, fabric, stone).
    Output JSON: [{"description": "...", "bbox": [x1,y1,x2,y2]}]
    """
    
    model = create_model('qwen')
    images = list_images('/path/to/images')
    
    pipeline = BasicPipeline(model, output_dir='results')
    results = pipeline.run(images, prompt=my_prompt)


# ==============================================================================
# EXAMPLE 5: Process Single Image
# ==============================================================================

def example_5_single_image():
    """Process a single image."""
    from models import create_model
    from core import ground_single_image, process_bboxes
    from utils import get_prompt, load_image, get_image_size
    
    # Setup
    model = create_model('qwen')
    image_path = '/path/to/single/image.jpg'
    prompt = get_prompt('grounding')
    
    # Ground
    result = ground_single_image(
        model=model,
        image_path=image_path,
        prompt=prompt
    )
    
    print(f"Found {len(result['bboxes'])} bounding boxes")
    
    # Process
    img_width, img_height = get_image_size(image_path)
    processed, stats = process_bboxes(
        result['bboxes'],
        img_width,
        img_height,
        filter_iou=True,
        fix_boundaries=True
    )
    
    print(f"After processing: {stats['output_count']} boxes")


# ==============================================================================
# EXAMPLE 6: Batch Processing with Custom Logic
# ==============================================================================

def example_6_batch_custom():
    """Process multiple images with custom logic."""
    from models import create_model
    from core import ground_images, visualize_all_images
    from utils import list_images, get_prompt
    from pathlib import Path
    
    model = create_model('qwen')
    images = list_images('/path/to/images')
    prompt = get_prompt('grounding')
    
    # Ground all images
    results = ground_images(model, images, prompt, verbose=True)
    
    # Filter: only keep images with 5+ boxes
    filtered = [
        {
            'image': r['image_name'],
            'image_path': str(Path('/path/to/images') / r['image_name']),
            'boxes': r['bboxes']
        }
        for r in results
        if len(r['bboxes']) >= 5
    ]
    
    print(f"Kept {len(filtered)} images with 5+ boxes")
    
    # Visualize filtered results
    visualize_all_images(filtered, output_dir='filtered_results')


# ==============================================================================
# EXAMPLE 7: Use Different Models
# ==============================================================================

def example_7_different_models():
    """Compare results from different model sizes."""
    from models import create_model
    from pipelines import run_basic_pipeline
    
    images_dir = '/path/to/images'
    
    # Try Qwen 8B
    model_8b = create_model('qwen-8b')
    results_8b = run_basic_pipeline(
        model=model_8b,
        image_dir=images_dir,
        output_dir='results_qwen8b'
    )
    
    # Try Qwen 2B
    model_2b = create_model('qwen-2b')
    results_2b = run_basic_pipeline(
        model=model_2b,
        image_dir=images_dir,
        output_dir='results_qwen2b'
    )
    
    print(f"Qwen-8B found {results_8b['processing']['output_boxes']} boxes")
    print(f"Qwen-2B found {results_2b['processing']['output_boxes']} boxes")


# ==============================================================================
# EXAMPLE 8: Extract Specific Crop Sizes
# ==============================================================================

def example_8_specific_crops():
    """Extract only large and xlarge crops."""
    from models import create_model
    from pipelines import BasicPipeline
    from core import get_crops_by_category
    from utils import list_images
    
    model = create_model('qwen')
    images = list_images('/path/to/images')
    
    # Run basic pipeline
    pipeline = BasicPipeline(model, output_dir='results')
    results = pipeline.run(images)
    
    # Get only large/xlarge crops
    from pathlib import Path
    crops_dir = Path(results['output_dir']) / 'crops'
    
    large_crops = get_crops_by_category(
        crops_dir,
        categories=['large', 'xlarge']
    )
    
    print(f"Found {len(large_crops['large'])} large crops")
    print(f"Found {len(large_crops['xlarge'])} xlarge crops")


# ==============================================================================
# EXAMPLE 9: CLI Usage (from terminal)
# ==============================================================================

"""
# Basic pipeline
python main.py basic --images /path/to/images --model qwen

# Iterative pipeline
python main.py iterative --images /path/to/images --model qwen \
    --semantic-threshold 8.0 --max-iterations 2

# With custom settings
python main.py basic --images /path/to/images \
    --model qwen-8b \
    --iou-threshold 0.7 \
    --target-size 1024 \
    --output my_results

# Quiet mode
python main.py basic --images /path/to/images --model qwen --quiet

# Check dependencies
python main.py --check
"""


# ==============================================================================
# HOW TO RUN THESE EXAMPLES
# ==============================================================================

if __name__ == "__main__":
    print("Example Usage Scripts")
    print("=" * 70)
    print()
    print("To run an example:")
    print("  1. Uncomment the function call below")
    print("  2. Update the image paths")
    print("  3. Run: python examples.py")
    print()
    print("=" * 70)
    
    # Uncomment one of these to run:
    # example_1_basic_simple()
    # example_2_basic_custom()
    # example_3_iterative()
    # example_4_custom_prompts()
    # example_5_single_image()
    # example_6_batch_custom()
    # example_7_different_models()
    # example_8_specific_crops()