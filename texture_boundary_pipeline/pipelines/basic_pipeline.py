
from pathlib import Path
from typing import List, Dict, Union
import json


from models import BaseVLM
from core import (
    ground_images,
    process_bboxes,
    visualize_all_images,
    extract_all_crops,
    update_processed_bboxes_with_crops,
    validate_bboxes
)
from utils import (
    list_images,
    get_prompt,
    save_json,
    create_output_directory,
    get_timestamp,
    load_image,
    get_image_size
)


class BasicPipeline:
    """
    Basic texture boundary detection pipeline.
    
    Workflow:
    1. Ground images (extract bboxes)
    2. Process bboxes (IoU filter + boundary fix)
    3. Visualize results
    4. Extract crops
    """
    
    def __init__(
        self,
        model: BaseVLM,
        output_dir: Union[str, Path] = "results",
        sa2va_model = None, 
        iou_threshold: float = 0.6,
        fix_boundaries: bool = True,
        target_size: int = 1024,
        extract_boundaries: bool = True,  
        boundary_thickness: int = 2, 
        verbose: bool = True
    ):
        """
        Initialize pipeline.
        
        Args:
            model: VLM model instance
            output_dir: Base output directory
            iou_threshold: IoU threshold for filtering (0-1)
            fix_boundaries: Enable boundary fixing
            target_size: Target size for image padding
            verbose: Print progress
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.sa2va_model = sa2va_model  
        self.iou_threshold = iou_threshold
        self.fix_boundaries = fix_boundaries
        self.target_size = target_size
        self.extract_boundaries = extract_boundaries  
        self.boundary_thickness = boundary_thickness  
        self.verbose = verbose
        
        # Create output structure
        self.timestamp = get_timestamp()
        self.run_dir = self.output_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    def run(
        self,
        image_paths: List[Union[str, Path]],
        prompt: str = None
    ) -> Dict:
        """
        Run the complete pipeline.
        
        Args:
            image_paths: List of image paths
            prompt: Grounding prompt (if None, loads default)
        
        Returns:
            Dict with results and statistics
        """
        if self.verbose:
            print("\n" + "="*70)
            print("BASIC PIPELINE - STARTING")
            print("="*70)
            print(f"Images: {len(image_paths)}")
            print(f"Output: {self.run_dir}")
            print(f"IoU threshold: {self.iou_threshold}")
            print(f"Fix boundaries: {self.fix_boundaries}")
            print("="*70)
        
        # Load prompt
        if prompt is None:
            prompt = get_prompt('grounding')
        
        # Step 1: Ground images
        if self.verbose:
            print("\n🔍 STEP 1: GROUNDING")
        
        grounding_results = ground_images(
            model=self.model,
            image_paths=image_paths,
            prompt=prompt,
            target_size=self.target_size,
            verbose=self.verbose
        )
        
        # Save raw grounding results
        grounding_file = self.run_dir / "grounding_results.json"
        save_json(
            [{'image': r['image_name'], 'raw_response': r['raw_response']} 
             for r in grounding_results],
            grounding_file
        )
        
        # Step 2: Process bboxes
        if self.verbose:
            print("\n⚙️  STEP 2: PROCESSING BBOXES")
        
        processed_data = []
        total_input = 0
        total_filtered = 0
        total_fixed = 0
        total_output = 0
        
        for result in grounding_results:
            image_path = Path([p for p in image_paths if Path(p).name == result['image_name']][0])
            img_width, img_height = get_image_size(image_path)
            
            # Validate first
            valid_boxes, skipped = validate_bboxes(
                result['bboxes'],
                img_width,
                img_height,
                verbose=False
            )
            
            # Process
            processed_boxes, stats = process_bboxes(
                valid_boxes,
                img_width,
                img_height,
                filter_iou=True,
                iou_threshold=self.iou_threshold,
                fix_boundaries=self.fix_boundaries,
                verbose=self.verbose
            )
            
            processed_data.append({
                'image': result['image_name'],
                'image_path': str(image_path),
                'boxes': processed_boxes
            })
            
            total_input += stats['input_count']
            total_filtered += stats['filtered_count']
            total_fixed += stats['fixed_count']
            total_output += stats['output_count']
        
        if self.verbose:
            print(f"\n  📊 Processing Summary:")
            print(f"     Input boxes:    {total_input}")
            print(f"     Filtered (IoU): {total_filtered}")
            print(f"     Fixed (edges):  {total_fixed}")
            print(f"     Final boxes:    {total_output}")
        
        # Save processed results
        processed_file = self.run_dir / "processed_bboxes.json"
        save_json(processed_data, processed_file)
        
        # Step 3: Visualize
        if self.verbose:
            print("\n🎨 STEP 3: VISUALIZING")
        
        viz_dir = self.run_dir / "visualizations"
        viz_stats = visualize_all_images(
            processed_data,
            output_dir=viz_dir,
            verbose=self.verbose
        )
        
        # Step 4: Extract crops
        if self.verbose:
            print("\n✂️  STEP 4: EXTRACTING CROPS")
            if self.extract_boundaries and self.sa2va_model:
                print("         + Boundaries masks")

        crops_dir = self.run_dir / "crops"
        crop_stats, crops_mapping = extract_all_crops(
            processed_data,
            output_dir=crops_dir,
            sa2va_model=self.sa2va_model,  
            resize_by_category=True,
            extract_boundaries=self.extract_boundaries,  
            boundary_thickness=self.boundary_thickness,  
            verbose=self.verbose
        )

        update_processed_bboxes_with_crops(
            processed_bboxes_path=processed_file,
            crops_mapping=crops_mapping
        )
        
        
        # Final summary
        summary = {
            'pipeline': 'basic',
            'timestamp': self.timestamp,
            'config': {
                'iou_threshold': self.iou_threshold,
                'fix_boundaries': self.fix_boundaries,
                'target_size': self.target_size,
                'extract_boundaries': self.extract_boundaries, 
                'sa2va_enabled': self.sa2va_model is not None 
            },
            'input': {
                'num_images': len(image_paths),
                'image_names': [Path(p).name for p in image_paths]
            },
            'grounding': {
                'total_boxes': total_input
            },
            'processing': {
                'input_boxes': total_input,
                'filtered': total_filtered,
                'fixed': total_fixed,
                'output_boxes': total_output
            },
            'visualization': viz_stats,
            'crops': crop_stats,
            'output_dir': str(self.run_dir)
        }
        
        # Save summary
        summary_file = self.run_dir / "pipeline_summary.json"
        save_json(summary, summary_file)
        
        if self.verbose:
            print("\n" + "="*70)
            print("✅ PIPELINE COMPLETE!")
            print("="*70)
            print(f"  📁 Output directory: {self.run_dir}")
            print(f"  📊 Summary: {summary_file}")
            print(f"  🎨 Visualizations: {viz_dir}")
            print(f"  ✂️  Crops: {crops_dir}")
            if crop_stats.get('total_boundaries', 0) > 0:  # ← חדש!
                masks_dir = self.run_dir / "masks"
                print(f"  🎭 Boundaries: {masks_dir}")
            print("="*70 + "\n")
            
        return summary

def run_basic_pipeline(
    model: BaseVLM,
    image_dir: Union[str, Path],
    output_dir: Union[str, Path] = "results",
    sa2va_model = None,
    extract_boundaries: bool = True,  
    boundary_thickness: int = 2,  
    **kwargs
) -> Dict:
    """
    Convenience function to run basic pipeline on a directory of images.
    
    Args:
        model: VLM model instance
        image_dir: Directory containing images
        output_dir: Output directory
        sa2va_model: Sa2VAModel instance for boundary extraction
        extract_boundaries: Extract boundaries masks
        boundary_thickness: Thickness of boundary lines
        **kwargs: Additional pipeline arguments (iou_threshold, fix_boundaries, etc.)
    
    Returns:
        Summary dict
    """
    # List images
    images = list_images(image_dir)
    
    if not images:
        raise ValueError(f"No images found in {image_dir}")
    
    # Create and run pipeline
    pipeline = BasicPipeline(
        model=model,
        output_dir=output_dir,
        sa2va_model=sa2va_model,  
        extract_boundaries=extract_boundaries,  
        boundary_thickness=boundary_thickness,  
        **kwargs
    )
    return pipeline.run(images)