import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from skimage.morphology import skeletonize, binary_erosion

# ============= Sa2VA Dual Segmentation - MORPHOLOGICAL ONLY =============

class Sa2VAMorphologicalBoundary:
    def __init__(self, device="cuda"):
        self.device = device
        print("Loading Sa2VA model...")
        
        self.model = AutoModel.from_pretrained(
            "ByteDance/Sa2VA-Qwen2_5-VL-7B",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            "ByteDance/Sa2VA-Qwen2_5-VL-7B",
            trust_remote_code=True
        )
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "ByteDance/Sa2VA-Qwen2_5-VL-7B",
                trust_remote_code=True
            )
        except:
            self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else None
        
        print("✓ Sa2VA loaded\n")
    
    def segment_texture(self, image_path, texture_description):
        """Segment a single texture region"""
        
        image = Image.open(image_path).convert("RGB")
        orig_size = image.size
        
        result = self.model.predict_forward(
            image=image,
            text=f"segment all regions with {texture_description}",
            tokenizer=self.tokenizer,
            processor=self.processor
        )
        
        mask = self._convert_to_mask(result['prediction_masks'][0], orig_size)
        return mask
    
    def extract_boundary(self, mask_a, mask_b):
        """Morphological boundary extraction between two masks"""
        
        binary_a = mask_a > 127
        binary_b = mask_b > 127
        
        # Erode both masks
        kernel = np.ones((3, 3), np.uint8)
        eroded_a = binary_erosion(binary_a, footprint=kernel)
        eroded_b = binary_erosion(binary_b, footprint=kernel)
        
        # Find boundaries
        boundary_a = np.logical_and(binary_a, ~eroded_a)
        boundary_b = np.logical_and(binary_b, ~eroded_b)
        
        # Union of boundaries
        boundary = np.logical_or(boundary_a, boundary_b)
        
        # Thin to skeleton
        boundary_thin = skeletonize(boundary)
        
        return (boundary_thin * 255).astype(np.uint8)
    
    def _convert_to_mask(self, mask, orig_size):
        """Convert to binary mask"""
        if len(mask.shape) == 3:
            mask = mask[0]
        
        if mask.dtype == bool or mask.dtype == np.bool_:
            mask = mask.astype(np.uint8)
        
        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        
        width, height = orig_size
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    def process_image(self, image_path, texture_a, texture_b, gt_mask_path=None):
        """
        Complete pipeline for one image
        
        Args:
            image_path: path to original image
            texture_a: description of first texture
            texture_b: description of second texture
            gt_mask_path: path to ground truth annotation mask (optional)
        """
        
        print(f"\nProcessing: {image_path.split('/')[-1]}")
        print(f"  Texture A: {texture_a}")
        print(f"  Texture B: {texture_b}")
        
        # Segment both textures
        print(f"  → Segmenting textures...")
        mask_a = self.segment_texture(image_path, texture_a)
        mask_b = self.segment_texture(image_path, texture_b)
        print(f"    Mask A: {(mask_a>127).sum()} pixels")
        print(f"    Mask B: {(mask_b>127).sum()} pixels")
        
        # Extract boundary
        print(f"  → Extracting boundary (morphological)...")
        boundary = self.extract_boundary(mask_a, mask_b)
        print(f"    Boundary: {(boundary>127).sum()} pixels")
        
        # Load ground truth if provided
        gt_mask = None
        if gt_mask_path is not None:
            print(f"  → Loading ground truth mask...")
            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                # Resize to match image size if needed
                img = cv2.imread(image_path)
                if gt_mask.shape != (img.shape[0], img.shape[1]):
                    gt_mask = cv2.resize(gt_mask, (img.shape[1], img.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
                print(f"    Ground truth: {(gt_mask>127).sum()} pixels")
            else:
                print(f"    ⚠ Failed to load ground truth")
        
        return mask_a, mask_b, boundary, gt_mask

# ============= VISUALIZATION WITH GROUND TRUTH =============

def visualize_with_ground_truth(image_data):
    """
    Visualize results with ground truth comparison
    
    image_data: list of (image_path, texture_a, texture_b, gt_mask_path)
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    extractor = Sa2VAMorphologicalBoundary(device)
    
    print("="*70)
    print("PROCESSING ALL IMAGES")
    print("="*70)
    
    results = []
    for i, data in enumerate(image_data):
        if len(data) == 4:
            img_path, texture_a, texture_b, gt_path = data
        else:
            img_path, texture_a, texture_b = data
            gt_path = None
        
        print(f"\n[{i+1}/{len(image_data)}]", end=" ")
        
        try:
            mask_a, mask_b, boundary, gt_mask = extractor.process_image(
                img_path, texture_a, texture_b, gt_path
            )
            results.append({
                'path': img_path,
                'texture_a': texture_a,
                'texture_b': texture_b,
                'mask_a': mask_a,
                'mask_b': mask_b,
                'boundary': boundary,
                'gt_mask': gt_mask,
                'error': None
            })
            print("  ✓ Success")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'path': img_path,
                'error': str(e)
            })
    
    # Create visualization
    print("\n" + "="*70)
    print("CREATING VISUALIZATION")
    print("="*70)
    
    num_images = len(results)
    # 6 columns if we have ground truth, 5 if not
    has_gt = any(r.get('gt_mask') is not None for r in results if r.get('error') is None)
    num_cols = 6 if has_gt else 5
    
    fig, axes = plt.subplots(num_images, num_cols, figsize=(4*num_cols, 4*num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        if result.get('error') is not None:
            for j in range(num_cols):
                axes[i, j].text(0.5, 0.5, f"Failed\n{result['error'][:30]}", 
                              ha='center', va='center', fontsize=9, color='red')
                axes[i, j].axis('off')
            continue
        
        img = cv2.imread(result['path'])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Column 1: Original
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"[{i+1}] Original\n{result['path'].split('/')[-1]}", 
                            fontsize=10, weight='bold')
        axes[i, 0].axis('off')
        
        # Column 2: Mask A
        axes[i, 1].imshow(result['mask_a'], cmap='Blues')
        axes[i, 1].set_title(f"Mask A\n{result['texture_a'][:30]}\n{(result['mask_a']>127).sum()} px", 
                            fontsize=9)
        axes[i, 1].axis('off')
        
        # Column 3: Mask B
        axes[i, 2].imshow(result['mask_b'], cmap='Oranges')
        axes[i, 2].set_title(f"Mask B\n{result['texture_b'][:30]}\n{(result['mask_b']>127).sum()} px", 
                            fontsize=9)
        axes[i, 2].axis('off')
        
        # Column 4: Predicted Boundary (Morphological)
        axes[i, 3].imshow(result['boundary'], cmap='gray', vmin=0, vmax=255)
        axes[i, 3].set_title(f"Predicted Boundary\n(Morphological)\n{(result['boundary']>127).sum()} px", 
                            fontsize=10, weight='bold')
        axes[i, 3].axis('off')
        
        # Column 5: Overlay of prediction
        overlay_pred = img_rgb.copy()
        overlay_pred[result['boundary'] > 127] = [255, 0, 0]
        axes[i, 4].imshow(overlay_pred)
        axes[i, 4].set_title("Prediction Overlay\n(Red)", fontsize=10, weight='bold')
        axes[i, 4].axis('off')
        
        # Column 6: Ground Truth (if available)
        if has_gt:
            if result['gt_mask'] is not None:
                # Show ground truth mask
                axes[i, 5].imshow(result['gt_mask'], cmap='gray', vmin=0, vmax=255)
                axes[i, 5].set_title(f"Ground Truth\n{(result['gt_mask']>127).sum()} px", 
                                    fontsize=10, weight='bold', color='green')
                axes[i, 5].axis('off')
            else:
                axes[i, 5].text(0.5, 0.5, "No GT", ha='center', va='center', fontsize=12)
                axes[i, 5].set_title("Ground Truth\n(Not provided)", fontsize=10)
                axes[i, 5].axis('off')
    
    title = "Sa2VA Morphological Boundary Extraction"
    if has_gt:
        title += " - With Ground Truth Comparison"
    plt.suptitle(title, fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = 'morphological_results_with_gt.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Saved: {output_path}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    successful = [r for r in results if r.get('error') is None]
    print(f"Successful: {len(successful)}/{len(results)}\n")
    
    if successful:
        print(f"{'Image':<35} {'Predicted':>12} {'Ground Truth':>15}")
        print("-" * 70)
        
        for r in successful:
            img_name = r['path'].split('/')[-1][:33]
            pred_px = (r['boundary']>127).sum()
            gt_px = (r['gt_mask']>127).sum() if r['gt_mask'] is not None else "N/A"
            
            print(f"{img_name:<35} {pred_px:>12} {str(gt_px):>15}")
        
        print("-" * 70)
    
    print("="*70)

# ============= RUN =============


if __name__ == "__main__":
   
    test_data = [("/home/aviad/results_debug_refinment_1/run_20260114_163952/crops/medium/ADE_val_00000001_380_39_460_209.jpg", "smooth roof tiles", "rough brick chimney","/home/aviad/results_debug_refinment_1/run_20260114_163952/masks/medium/ADE_val_00000001_380_39_460_209.png"),
                 ("/home/aviad/results_debug_refinment_1/run_20260114_163952/crops/medium/ADE_val_00000001_0_364_335_424.jpg", "smooth grass", "rough stone wall", "/home/aviad/results_debug_refinment_1/run_20260114_163952/masks/medium/ADE_val_00000001_0_364_335_424.png"),
                 ("/home/aviad/results_debug_refinment_1/run_20260114_163952/crops/small/ADE_val_00000001_160_274_280_369.jpg", "smooth window frame", "rough stone wall", "/home/aviad/results_debug_refinment_1/run_20260114_163952/masks/small/ADE_val_00000001_160_274_280_369.png"),
                 ("/home/aviad/results_debug_refinment_1/run_20260114_163952/crops/large/ADE_val_00000001_380_154_660_294.jpg","smooth roof tiles", "stone wall", "/home/aviad/results_debug_refinment_1/run_20260114_163952/masks/large/ADE_val_00000001_380_154_660_294.png")
    ]
    visualize_with_ground_truth(test_data)
