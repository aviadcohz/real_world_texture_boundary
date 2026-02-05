import json
import re
from typing import List, Dict, Tuple, Union


def parse_json_format(text: str) -> List[Dict]:
    """
    Parse JSON format from model output.
    
    Supports both:
    1. Simple list: [[x1,y1,x2,y2], [x3,y3,x4,y4], ...]
    2. With descriptions: [{"description": "...", "bbox": [x1,y1,x2,y2]}, ...]
    
    Args:
        text: JSON text or text containing JSON
    
    Returns:
        List of dicts with 'id', 'description', 'coords' keys
    """
    try:
        data = json.loads(text.strip())
        
        transitions = []
        
        # Check if first element is a dict (with descriptions) or list (simple)
        if data and isinstance(data[0], dict):
            # Format with descriptions
            for idx, item in enumerate(data, 1):
                if 'bbox' in item and len(item['bbox']) == 4:
                    transitions.append({
                        'id': idx,
                        'description': item.get('description', f'Transition {idx}'),
                        'coords': tuple(item['bbox'])
                    })
        else:
            # Simple list format
            for idx, box in enumerate(data, 1):
                if len(box) == 4:
                    transitions.append({
                        'id': idx,
                        'description': None,
                        'coords': tuple(box)
                    })
        
        return transitions
    
    except json.JSONDecodeError:
        # Fallback: try to extract JSON from text
        # Try dict format first
        match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                transitions = []
                for idx, item in enumerate(data, 1):
                    if 'bbox' in item and len(item['bbox']) == 4:
                        transitions.append({
                            'id': idx,
                            'description': item.get('description', f'Transition {idx}'),
                            'coords': tuple(item['bbox'])
                        })
                return transitions
            except:
                pass
        
        # Try simple list format
        match = re.search(r'\[\s*\[[\d,\s]+\]\s*(?:,\s*\[[\d,\s]+\]\s*)*\]', text)
        if match:
            try:
                boxes = json.loads(match.group(0))
                transitions = []
                for idx, box in enumerate(boxes, 1):
                    if len(box) == 4:
                        transitions.append({
                            'id': idx,
                            'description': None,
                            'coords': tuple(box)
                        })
                return transitions
            except:
                pass
        
        return []


def adjust_bboxes_after_padding(
    bboxes: List[Dict],
    offset_x: int,
    offset_y: int
) -> List[Dict]:
    """
    Adjust bboxes from padded coordinates back to original image coordinates.
    
    Args:
        bboxes: List of [x1, y1, x2, y2] or list of dicts with 'bbox' or 'coords' key
        offset_x: Padding on left
        offset_y: Padding on top
    
    Returns:
        adjusted_bboxes: Bboxes in original coordinates (same format as input)
    """
    adjusted = []
    
    for bbox in bboxes:
        if isinstance(bbox, dict):
            # Check which key it uses
            if 'coords' in bbox:
                # Format from parse_json_format: {"description": "...", "coords": (x1, y1, x2, y2)}
                x1, y1, x2, y2 = bbox['coords']
                adjusted.append({
                    **bbox,  # Keep other fields (id, description)
                    'coords': (
                        max(0, x1 - offset_x),
                        max(0, y1 - offset_y),
                        max(0, x2 - offset_x),
                        max(0, y2 - offset_y)
                    )
                })
            elif 'bbox' in bbox:
                # Format: {"description": "...", "bbox": [x1, y1, x2, y2]}
                x1, y1, x2, y2 = bbox['bbox']
                adjusted.append({
                    **bbox,  # Keep other fields
                    'bbox': [
                        max(0, x1 - offset_x),
                        max(0, y1 - offset_y),
                        max(0, x2 - offset_x),
                        max(0, y2 - offset_y)
                    ]
                })
        else:
            # Format: [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            adjusted.append([
                max(0, x1 - offset_x),
                max(0, y1 - offset_y),
                max(0, x2 - offset_x),
                max(0, y2 - offset_y)
            ])
    
    return adjusted


def clamp_bbox(
    bbox: Tuple[int, int, int, int],
    image_width: int,
    image_height: int
) -> Tuple[int, int, int, int]:
    """
    Clamp bounding box to image boundaries.
    
    Args:
        bbox: (x1, y1, x2, y2)
        image_width: Image width
        image_height: Image height
    
    Returns:
        Clamped (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    
    x1 = max(0, min(x1, image_width - 1))
    x2 = max(0, min(x2, image_width))
    y1 = max(0, min(y1, image_height - 1))
    y2 = max(0, min(y2, image_height))
    
    return (x1, y1, x2, y2)


def bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    """
    Calculate bounding box area.
    
    Args:
        bbox: (x1, y1, x2, y2)
    
    Returns:
        Area in pixels
    """
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def bbox_dimensions(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Get bounding box dimensions.
    
    Args:
        bbox: (x1, y1, x2, y2)
    
    Returns:
        (width, height)
    """
    x1, y1, x2, y2 = bbox
    return (x2 - x1, y2 - y1)


def is_valid_bbox(bbox: Tuple[int, int, int, int]) -> bool:
    """
    Check if bounding box is valid (positive area).
    
    Args:
        bbox: (x1, y1, x2, y2)
    
    Returns:
        True if valid
    """
    x1, y1, x2, y2 = bbox
    return x1 < x2 and y1 < y2


def calculate_iou(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int]
) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)
    
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection rectangle
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if there's no intersection
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    # Calculate intersection area
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate areas of both boxes
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate union
    union = area1 + area2 - intersection
    
    # Calculate IoU
    if union == 0:
        return 0.0
    
    return intersection / union


def convert_coords_format(
    bboxes: List[Dict],
    from_format: str = 'coords',
    to_format: str = 'bbox'
) -> List[Dict]:
    """
    Convert between 'coords' and 'bbox' key formats.
    
    Args:
        bboxes: List of bbox dicts
        from_format: Source format ('coords' or 'bbox')
        to_format: Target format ('coords' or 'bbox')
    
    Returns:
        Converted bboxes
    """
    if from_format == to_format:
        return bboxes
    
    converted = []
    for box in bboxes:
        if isinstance(box, dict) and from_format in box:
            new_box = {**box}
            new_box[to_format] = list(box[from_format]) if to_format == 'bbox' else tuple(box[from_format])
            del new_box[from_format]
            converted.append(new_box)
        else:
            converted.append(box)
    
    return converted


def format_bbox_for_json(bbox: Union[Tuple, List, Dict]) -> Dict:
    """
    Format bbox for JSON output.
    
    Args:
        bbox: Bbox in any format
    
    Returns:
        Dict with 'description' and 'bbox' keys
    """
    if isinstance(bbox, dict):
        if 'coords' in bbox:
            return {
                'description': bbox.get('description'),
                'bbox': list(bbox['coords'])
            }
        elif 'bbox' in bbox:
            return {
                'description': bbox.get('description'),
                'bbox': bbox['bbox']
            }
    
    # Assume it's a tuple or list
    return {
        'description': None,
        'bbox': list(bbox)
    }


def get_bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Get center point of bounding box.
    
    Args:
        bbox: (x1, y1, x2, y2)
    
    Returns:
        (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)