import argparse
import sys
from pathlib import Path

from models import create_model, list_available_models, check_dependencies
from pipelines import (
    run_basic_pipeline,
    run_iterative_pipeline,
    SCALABLE_AVAILABLE,
    DISTRIBUTED_AVAILABLE
)
from utils import list_images

# Import scalable pipeline if available
if SCALABLE_AVAILABLE:
    from pipelines import run_scalable_pipeline

# Import scale config if available
try:
    from config.scale_config import (
        ScaleConfig,
        get_preset,
        configure_from_file,
        get_config
    )
    SCALE_CONFIG_AVAILABLE = True
except ImportError:
    SCALE_CONFIG_AVAILABLE = False


def check_setup():
    """Check if dependencies are available."""
    deps = check_dependencies()

    print("\n" + "="*70)
    print("DEPENDENCY CHECK")
    print("="*70)
    print(f"  Qwen:  {'Y' if deps['qwen'] else 'N'}")
    print(f"  LLaVA: {'Y' if deps['llava'] else 'N'}")
    print(f"\n  Available models: {', '.join(deps['available_models']) or 'None'}")

    # Check scale utilities
    print(f"\n  Scale utilities:")
    print(f"    Scalable pipeline: {'Y' if SCALABLE_AVAILABLE else 'N'}")
    print(f"    Distributed pipeline: {'Y' if DISTRIBUTED_AVAILABLE else 'N'}")
    print(f"    Scale config: {'Y' if SCALE_CONFIG_AVAILABLE else 'N'}")

    print("="*70 + "\n")

    if not deps['available_models']:
        print("No models available! Install dependencies:")
        print("   pip install torch transformers qwen-vl-utils")
        return False

    return True


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Texture Boundary Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run basic pipeline on a directory
  python main.py basic --images /path/to/images --model qwen

  # Run scalable pipeline with batch processing
  python main.py scalable --images /path/to/images --batch-size 4 --preset large

  # Run with custom scale config
  python main.py scalable --images /path/to/images --scale-config config.yaml

  # Check available models and features
  python main.py --check
        """
    )

    # Global arguments
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check dependencies and available models'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='qwen',
        help='Model to use (default: qwen). Options: ' + ', '.join(list_available_models())
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (default: cuda)'
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Pipeline to run')

    # Basic pipeline
    basic_parser = subparsers.add_parser('basic', help='Run basic pipeline')
    basic_parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='Directory containing images'
    )
    basic_parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory (default: results)'
    )
    basic_parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.6,
        help='IoU threshold for filtering (default: 0.6)'
    )
    basic_parser.add_argument(
        '--no-fix-boundaries',
        action='store_true',
        help='Disable boundary fixing'
    )
    basic_parser.add_argument(
        '--target-size',
        type=int,
        default=1024,
        help='Target size for padding (default: 1024)'
    )
    basic_parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    # Iterative pipeline
    iter_parser = subparsers.add_parser('iterative', help='Run iterative pipeline')
    iter_parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='Directory containing images'
    )
    iter_parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory (default: results)'
    )
    iter_parser.add_argument(
        '--entropy-threshold',
        type=float,
        default=4.5,
        help='Entropy threshold for filtering (default: 4.5)'
    )
    iter_parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.9,
        help='IoU threshold for filtering (default: 0.9)'
    )
    iter_parser.add_argument(
        '--no-fix-boundaries',
        action='store_true',
        help='Disable boundary fixing'
    )
    iter_parser.add_argument(
        '--target-size',
        type=int,
        default=1024,
        help='Target size for padding (default: 1024)'
    )
    iter_parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    iter_parser.add_argument(
        '--num-images',
        type=int,
        default=None,
        help='Number of images to process (default: all)'
    )

    # Scalable pipeline
    if SCALABLE_AVAILABLE:
        scale_parser = subparsers.add_parser(
            'scalable',
            help='Run scalable pipeline with optimizations'
        )
        scale_parser.add_argument(
            '--images',
            type=str,
            required=True,
            help='Directory containing images'
        )
        scale_parser.add_argument(
            '--output',
            type=str,
            default='results',
            help='Output directory (default: results)'
        )
        scale_parser.add_argument(
            '--batch-size',
            type=int,
            default=4,
            help='VLM batch size (default: 4)'
        )
        scale_parser.add_argument(
            '--prefetch-size',
            type=int,
            default=8,
            help='Number of images to prefetch (default: 8)'
        )
        scale_parser.add_argument(
            '--preset',
            type=str,
            choices=['small', 'large', 'multi_gpu', 'low_memory', 'max_throughput'],
            default=None,
            help='Use a preset configuration'
        )
        scale_parser.add_argument(
            '--scale-config',
            type=str,
            default=None,
            help='Path to scale config file (YAML or JSON)'
        )
        scale_parser.add_argument(
            '--iou-threshold',
            type=float,
            default=0.6,
            help='IoU threshold for filtering (default: 0.6)'
        )
        scale_parser.add_argument(
            '--no-fix-boundaries',
            action='store_true',
            help='Disable boundary fixing'
        )
        scale_parser.add_argument(
            '--streaming',
            action='store_true',
            help='Enable streaming JSON output'
        )
        scale_parser.add_argument(
            '--parallel-save',
            action='store_true',
            default=True,
            help='Enable parallel crop saving (default: True)'
        )
        scale_parser.add_argument(
            '--save-workers',
            type=int,
            default=8,
            help='Number of workers for parallel saving (default: 8)'
        )
        scale_parser.add_argument(
            '--quiet',
            action='store_true',
            help='Suppress progress output'
        )

    # Parse arguments
    args = parser.parse_args()

    # Check dependencies
    if args.check or not args.command:
        return 0 if check_setup() else 1

    # Verify model available
    available = list_available_models()
    if args.model not in available:
        print(f"Model '{args.model}' not available!")
        print(f"   Available models: {', '.join(available)}")
        return 1

    # Load model
    print(f"\nLoading model: {args.model}...")
    try:
        model = create_model(args.model, device=args.device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return 1

    print(f"Model loaded: {model}")

    # Verify images directory
    image_dir = Path(args.images)
    if not image_dir.exists():
        print(f"Images directory not found: {image_dir}")
        return 1

    images = list_images(image_dir)
    if not images:
        print(f"No images found in {image_dir}")
        return 1

    print(f"Found {len(images)} images in {image_dir}")

    # Run pipeline
    try:
        if args.command == 'basic':
            print("\nRunning BASIC pipeline...")
            results = run_basic_pipeline(
                model=model,
                image_dir=image_dir,
                output_dir=args.output,
                iou_threshold=args.iou_threshold,
                fix_boundaries=not args.no_fix_boundaries,
                target_size=args.target_size,
                verbose=not args.quiet
            )

        elif args.command == 'iterative':
            print("\nRunning ITERATIVE pipeline...")
            results = run_iterative_pipeline(
                model=model,
                image_dir=image_dir,
                output_dir=args.output,
                entropy_threshold=args.entropy_threshold,
                iou_threshold=args.iou_threshold,
                fix_boundaries=not args.no_fix_boundaries,
                target_size=args.target_size,
                num_images=args.num_images,
                verbose=not args.quiet
            )

        elif args.command == 'scalable' and SCALABLE_AVAILABLE:
            print("\nRunning SCALABLE pipeline...")

            # Build scale config
            config = None
            if SCALE_CONFIG_AVAILABLE:
                if args.scale_config:
                    configure_from_file(args.scale_config)
                    config = get_config()
                elif args.preset:
                    config = get_preset(args.preset)
                else:
                    config = ScaleConfig()
                    config.processing.vlm_batch_size = args.batch_size
                    config.processing.prefetch_size = args.prefetch_size
                    config.processing.parallel_save = args.parallel_save
                    config.processing.save_workers = args.save_workers
                    config.output.streaming = args.streaming

            results = run_scalable_pipeline(
                model=model,
                image_dir=image_dir,
                output_dir=args.output,
                config=config,
                iou_threshold=args.iou_threshold,
                fix_boundaries=not args.no_fix_boundaries,
                verbose=not args.quiet
            )

        else:
            parser.print_help()
            return 1

        print("\nPipeline completed successfully!")
        print(f"Results saved to: {results['output_dir']}")
        return 0

    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
