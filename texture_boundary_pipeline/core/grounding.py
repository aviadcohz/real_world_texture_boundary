from pathlib import Path
from typing import List, Dict, Union
from PIL import Image
import re 
from models.base_vlm import BaseVLM
from utils.image_utils import pad_image_to_standard_size
from utils.bbox_utils import adjust_bboxes_after_padding, parse_json_format, format_bbox_for_json




def ground_single_image(
    model: BaseVLM,
    image_path: Union[str, Path],
    prompt: str,
    target_size: int = 1024,
    max_tokens: int = 512
) -> Dict:
    """
    Ground texture boundaries in a single image.
    
    Args:
        model: VLM model instance
        image_path: Path to image
        prompt: Grounding prompt
        target_size: Target size for padding
        max_tokens: Max tokens for generation
    
    Returns:
        Dict with:
            - image_name: str
            - bboxes: List of bbox dicts
            - raw_response: str
            - metadata: Dict with padding info
    """
    image_path = Path(image_path)
    
    # Pad image
    padded_image, offset_x, offset_y, orig_size = pad_image_to_standard_size(
        image_path,
        target_size=target_size
    )
    
    # Generate response
    raw_response = model.generate(
        image=padded_image,
        prompt=prompt,
        max_tokens=max_tokens
    )
    
    # Parse bboxes
    bboxes_padded = parse_json_format(raw_response)
    
    # Adjust to original coordinates
    bboxes_adjusted = adjust_bboxes_after_padding(
        bboxes_padded,
        offset_x,
        offset_y
    )
    
    return {
        'image_name': image_path.name,
        'bboxes': bboxes_adjusted,
        'raw_response': raw_response,
        'metadata': {
            'offset_x': offset_x,
            'offset_y': offset_y,
            'original_size': orig_size,
            'padded_size': target_size
        }
    }


def ground_images(
    model: BaseVLM,
    image_paths: List[Union[str, Path]],
    prompt: str,
    target_size: int = 1024,
    max_tokens: int = 512,
    verbose: bool = True
) -> List[Dict]:
    """
    Ground texture boundaries in multiple images.
    
    Args:
        model: VLM model instance
        image_paths: List of image paths
        prompt: Grounding prompt
        target_size: Target size for padding
        max_tokens: Max tokens for generation
        verbose: Print progress
    
    Returns:
        List of result dicts (one per image)
    """
    results = []
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"GROUNDING {len(image_paths)} IMAGES")
        print(f"{'='*70}\n")
    
    for idx, img_path in enumerate(image_paths, 1):
        if verbose:
            print(f"[{idx}/{len(image_paths)}] Processing: {Path(img_path).name}")
        
        result = ground_single_image(
            model=model,
            image_path=img_path,
            prompt=prompt,
            target_size=target_size,
            max_tokens=max_tokens
        )
        
        if verbose:
            print(f"  ✅ Found {len(result['bboxes'])} bboxes")
            print(f"  📐 Original size: {result['metadata']['original_size']}")
            print()
        
        results.append(result)
    
    if verbose:
        total_bboxes = sum(len(r['bboxes']) for r in results)
        print(f"{'='*70}")
        print(f"✅ Complete! Total bboxes: {total_bboxes}")
        print(f"{'='*70}\n")
    
    return results

def refine_semantic_crop_with_padding(
    model: BaseVLM,
    crop_path: Union[str, Path],
    prompt: str,
    target_size: int = 512,  
    max_tokens: int = 256
) -> Dict:
    """
    Refine a semantic crop - ask model for focused bbox.
    
    Similar to ground_single_image but:
    - Uses smaller padding (512 instead of 1024)
    - Expects single bbox or special response
    
    Args:
        model: VLM model instance
        crop_path: Path to crop image
        prompt: Refinement prompt
        target_size: Target size for padding (default 512)
        max_tokens: Max tokens for generation
    
    Returns:
        Dict with:
            - action: 'refine' | 'keep' | 'discard'
            - bbox: Adjusted bbox coords (if action='refine')
            - raw_response: Model response
            - metadata: Padding info
    """
    crop_path = Path(crop_path)

    original_img = Image.open(crop_path)
    width, height = original_img.size

    prompt_with_size = f"""Crop dimensions: {width}×{height}
    IMPORTANT: If you desicide to get a bbox, coordinates must be within these bounds:
    - x coordinates: [0, {width}]
    - y coordinates: [0, {height}]

    {prompt}"""
    
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": original_img,
                    "resized_height": height,  
                    "resized_width": width     
                },
                {"type": "text", "text": prompt_with_size}
            ]
        }
    ]

     # Use custom_generate instead of generate
    raw_response = model.custom_generate(
        messages=messages,
        max_tokens=max_tokens
    )
    
    response_text = raw_response.strip().upper()
    
    # Parse response
    result = {
        'crop_name': crop_path.name,
        'raw_response': raw_response,
        'metadata': {
            'original_size': original_img.size
        }
    }
    
    # Check for GOOD_AS_IS
    if 'GOOD_AS_IS' in response_text or 'GOOD AS IS' in response_text:
        result['action'] = 'keep'
        result['bbox'] = None
        return result
    
    # Check for NO_TRANSITION
    if 'NO_TRANSITION' in response_text or 'NO TRANSITION' in response_text:
        result['action'] = 'discard'
        result['bbox'] = None
        return result
    
    # Pattern 1: [[x1,y1,x2,y2]]
    bbox_pattern_double = r'\[\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]\]'
    matches = re.findall(bbox_pattern_double, raw_response)
    
    if not matches:
        # Pattern 2: [x1,y1,x2,y2] (fallback)
        bbox_pattern_single = r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]'
        matches = re.findall(bbox_pattern_single, raw_response)
    
    if matches:
        x1, y1, x2, y2 = map(int, matches[0])
        
        # Check if bbox is FULLY within image bounds
        if (0 <= x1 < x2 <= width and 
            0 <= y1 < y2 <= height):
            
            # Valid bbox - within bounds!
            result['action'] = 'refine'
            result['bbox'] = [[x1, y1, x2, y2]]
            return result
        else:
            # Bbox out of bounds - discard!
            result['action'] = 'discard'
            result['bbox'] = None
            return result
    
    # No bbox found - keep as is
    result['action'] = 'keep'
    result['bbox'] = None
    return result


def format_results_for_json(results: List[Dict]) -> List[Dict]:
    """
    Format grounding results for JSON export.
    
    Converts internal format to JSON-serializable format.
    
    Args:
        results: List of grounding results
    
    Returns:
        List of dicts with 'image' and 'raw_response' keys
    """
    import json
    
    formatted = []
    
    for result in results:
        formatted.append({
            'image': result['image_name'],
            'raw_response': json.dumps([
                format_bbox_for_json(bbox)
                for bbox in result['bboxes']
            ])
        })
    
    return formatted

