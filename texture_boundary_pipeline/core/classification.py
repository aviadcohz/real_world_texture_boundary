import json
import re
from pathlib import Path
from typing import List, Dict, Union, Tuple
from PIL import Image

from models.base_vlm import BaseVLM
from utils.image_utils import load_image
from utils.io_utils import save_json


def parse_classification_response(response: str) -> Dict:
    """
    Parse classification response from model.
    
    Expected format:
    {
      "classification": "SEMANTIC" or "TEXTURE",
      "reasoning": "..."
    }
    
    Args:
        response: Raw model response
    
    Returns:
        Dict with classification, reasoning
    """
    try:
        # Try direct JSON parse
        data = json.loads(response.strip())
        
        return {
            'classification': data.get('classification', 'UNKNOWN').upper(),
            'reasoning': data.get('reasoning', ''),
            'raw_response': response
        }
    
    except json.JSONDecodeError:
        # Fallback: try to extract from text
        classification = None
        reasoning = ""
        
        # Extract classification
        class_match = re.search(r'classification["\s:]+([A-Z]+)', response, re.IGNORECASE)
        if class_match:
            classification = class_match.group(1).upper()
        
        
        # Extract reasoning
        reason_match = re.search(r'reasoning["\s:]+["\']?([^"\'}\]]+)', response, re.IGNORECASE)
        if reason_match:
            reasoning = reason_match.group(1).strip()
        
        return {
            'classification': classification or 'UNKNOWN',
            'reasoning': reasoning,
            'raw_response': response
        }


def classify_crop(
    model: BaseVLM,
    crop_image: Union[str, Path, Image.Image],
    prompt: str,
    max_tokens: int = 256
) -> Dict:
    """
    Classify a single crop as semantic or texture.
    
    Args:
        model: VLM model instance
        crop_image: Crop image path or PIL Image
        prompt: Classification prompt
        max_tokens: Max tokens for generation
    
    Returns:
        Dict with classification result
    """
    # Generate response
    response = model.generate(
        image=crop_image,
        prompt=prompt,
        max_tokens=max_tokens
    )
    
    # Parse response
    result = parse_classification_response(response)
    
    return result


def classify_crops(
    model: BaseVLM,
    crop_paths: List[Path],
    prompt: str,
    max_tokens: int = 256,
    verbose: bool = True
) -> Tuple[List[Path], List[Path], List[Dict]]:
    """
    Classify multiple crops.
    
    Args:
        model: VLM model instance
        crop_paths: List of crop image paths
        prompt: Classification prompt
        max_tokens: Max tokens for generation
        verbose: Print progress
    
    Returns:
        semantic_crops: List of paths classified as semantic
        texture_crops: List of paths classified as texture
        all_results: List of classification dicts
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"CLASSIFYING {len(crop_paths)} CROPS")
        print(f"{'='*70}\n")
    
    semantic_crops = []
    texture_crops = []
    all_results = []
    
    for idx, crop_path in enumerate(crop_paths, 1):
        if verbose:
            print(f"[{idx}/{len(crop_paths)}] {crop_path.name}")
        
        result = classify_crop(
            model=model,
            crop_image=str(crop_path),
            prompt=prompt,
            max_tokens=max_tokens
        )
        
        # Add crop info
        result['crop_path'] = str(crop_path)
        result['crop_name'] = crop_path.name
        
        # Categorize
        is_semantic = (
            result['classification'] == 'SEMANTIC'
        )
        
        if is_semantic:
            semantic_crops.append(crop_path)
            category = "SEMANTIC"
        else:
            texture_crops.append(crop_path)
            category = "TEXTURE TRANSITION"
        
        if verbose:
            print(f"     {result['reasoning'][:80]}...")
            print()
        
        all_results.append(result)
    
    if verbose:
        print("="*70)
        print("✅ CLASSIFICATION COMPLETE")
        print("="*70)
        print(f"  Total:    {len(crop_paths)}")
        print(f"  Semantic: {len(semantic_crops)}")
        print(f"  Texture:  {len(texture_crops)}")
        print("="*70 + "\n")
    
    return semantic_crops, texture_crops, all_results


def filter_crops_by_category(
    crops_dir: Union[str, Path],
    categories: List[str] = None
) -> List[Path]:
    """
    Get crops from specific size categories.
    
    Args:
        crops_dir: Root crops directory
        categories: List of categories (default: ['medium', 'large', 'xlarge'])
    
    Returns:
        List of crop paths
    """
    if categories is None:
        categories = ['medium', 'large', 'xlarge']
    
    crops_dir = Path(crops_dir)
    all_crops = []
    
    for category in categories:
        category_dir = crops_dir / category
        if category_dir.exists():
            crops = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
            all_crops.extend(crops)
    
    return sorted(all_crops)


def save_classification_results(
    results: List[Dict],
    output_path: Union[str, Path]
) -> None:
    """
    Save classification results to JSON.
    
    Args:
        results: List of classification result dicts
        output_path: Output JSON path
    """
    
    save_json(results, output_path, indent=2)