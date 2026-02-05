from typing import List, Dict, Tuple
from utils.bbox_utils import calculate_iou


def filter_overlapping_boxes(
    transitions: List[Dict],
    iou_threshold: float = 0.6,
    keep_larger: bool = True,
    verbose: bool = True
) -> Tuple[List[Dict], int]:
    """
    Filter out boxes with high IoU (overlapping boxes).
    
    Args:
        transitions: List of transition dicts with 'coords'
        iou_threshold: IoU threshold for filtering (default 0.6 = 60%)
        keep_larger: If True, keep larger box; if False, keep first box
        verbose: Print filtering info
    
    Returns:
        filtered_transitions: List of transitions after filtering
        removed_count: Number of boxes removed
    """
    if len(transitions) <= 1:
        return transitions, 0
    
    # Create list to track which boxes to keep
    keep_indices = set(range(len(transitions)))
    removed_pairs = []
    
    # Compare each pair of boxes
    for i in range(len(transitions)):
        if i not in keep_indices:
            continue
        
        for j in range(i + 1, len(transitions)):
            if j not in keep_indices:
                continue
            
            # Calculate IoU
            box1 = transitions[i]['coords']
            box2 = transitions[j]['coords']
            iou = calculate_iou(box1, box2)
            
            # If IoU is above threshold, remove one box
            if iou >= iou_threshold:
                if keep_larger:
                    # Keep the larger box
                    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                    
                    if area1 >= area2:
                        remove_idx = j
                        keep_idx = i
                    else:
                        remove_idx = i
                        keep_idx = j
                else:
                    # Keep first box
                    remove_idx = j
                    keep_idx = i
                
                keep_indices.discard(remove_idx)
                removed_pairs.append({
                    'kept': keep_idx + 1,
                    'removed': remove_idx + 1,
                    'iou': iou
                })
    
    # Create filtered list
    filtered = [transitions[i] for i in sorted(keep_indices)]
    removed_count = len(transitions) - len(filtered)
    
    # Print filtering info
    if verbose and removed_count > 0:
        print(f"       🔍 IoU Filtering: Removed {removed_count} overlapping boxes (IoU > {iou_threshold:.0%})")
        for pair in removed_pairs:
            print(f"          Kept Box {pair['kept']}, Removed Box {pair['removed']} (IoU={pair['iou']:.2%})")
    
    return filtered, removed_count


def fix_tiny_boundary_bbox(
    bbox: Tuple[int, int, int, int],
    image_width: int,
    image_height: int
) -> Tuple[Tuple[int, int, int, int], bool]:
    """
    Fix tiny boxes near boundaries - snap to edge and extend inward.
    
    CONSTRAINT: Fixed bbox limited to max 1/4 of image width/height.
    Constraint applies ONLY to boxes that are actually fixed (near boundaries).
    
    Args:
        bbox: (x1, y1, x2, y2)
        image_width: Width of image
        image_height: Height of image
    
    Returns:
        fixed_bbox: (x1, y1, x2, y2)
        was_fixed: bool
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    # First, check if box is near ANY boundary
    near_bottom = y2 >= image_height - 10
    near_top = y1 <= 10
    near_right = x2 >= image_width - 10
    near_left = x1 <= 10
    
    # If NOT near any boundary, return original (no fix, no constraint)
    if not (near_bottom or near_top or near_right or near_left):
        return bbox, False
    
    # Box IS near boundary, so we will fix it
    # NOW apply constraint (only for boxes we're fixing)
    max_width = image_width // 4
    max_height = image_height // 4
    
    if width > max_width:
        width = max_width
    if height > max_height:
        height = max_height
    
    fixed = False
    
    # Fix BOTTOM
    if near_bottom:
        y2 = image_height
        y1 = y2 - height  # Use constrained height
        if y1 < 0:
            y1 = 0
            y2 = height
        fixed = True
    
    # Fix TOP
    if near_top:
        y1 = 0
        y2 = height  # Use constrained height
        if y2 > image_height:
            y2 = image_height
            y1 = image_height - height
        fixed = True
    
    # Fix RIGHT
    if near_right:
        x2 = image_width
        x1 = x2 - width  # Use constrained width
        if x1 < 0:
            x1 = 0
            x2 = width
        fixed = True
    
    # Fix LEFT
    if near_left:
        x1 = 0
        x2 = width  # Use constrained width
        if x2 > image_width:
            x2 = image_width
            x1 = image_width - width
        fixed = True
    
    return (x1, y1, x2, y2), fixed


def process_bboxes(
    bboxes: List[Dict],
    image_width: int,
    image_height: int,
    filter_iou: bool = True,
    iou_threshold: float = 0.6,
    fix_boundaries: bool = True,
    verbose: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    Process bboxes: IoU filtering + boundary fixing.
    
    Args:
        bboxes: List of bbox dicts with 'coords'
        image_width: Image width
        image_height: Image height
        filter_iou: Enable IoU filtering
        iou_threshold: IoU threshold
        fix_boundaries: Enable boundary fixing
        verbose: Print info
    
    Returns:
        processed_bboxes: Processed bbox list
        stats: Dict with processing statistics
    """
    stats = {
        'input_count': len(bboxes),
        'filtered_count': 0,
        'fixed_count': 0,
        'output_count': 0
    }
    
    # IoU filtering
    if filter_iou and len(bboxes) > 1:
        bboxes, filtered_count = filter_overlapping_boxes(
            bboxes,
            iou_threshold=iou_threshold,
            verbose=verbose
        )
        stats['filtered_count'] = filtered_count
    
    # Boundary fixing
    if fix_boundaries:
        fixed_bboxes = []
        fixed_count = 0
        
        for bbox in bboxes:
            coords = bbox['coords']
            
            # Check if would be tiny after clamping
            x1_preview = max(0, min(coords[0], image_width - 1))
            x2_preview = max(0, min(coords[2], image_width))
            y1_preview = max(0, min(coords[1], image_height - 1))
            y2_preview = max(0, min(coords[3], image_height))
            preview_width = x2_preview - x1_preview
            preview_height = y2_preview - y1_preview
            would_be_tiny = preview_width < 20 or preview_height < 20
            
            # Fix if tiny
            if would_be_tiny:
                fixed_coords, was_fixed = fix_tiny_boundary_bbox(
                    coords,
                    image_width,
                    image_height
                )
                
                if was_fixed:
                    bbox = {**bbox, 'coords': fixed_coords, 'was_fixed': True}
                    fixed_count += 1
                else:
                    bbox = {**bbox, 'was_fixed': False}
            else:
                bbox = {**bbox, 'was_fixed': False}
            
            fixed_bboxes.append(bbox)
        
        bboxes = fixed_bboxes
        stats['fixed_count'] = fixed_count
    
    stats['output_count'] = len(bboxes)
    
    return bboxes, stats


def validate_bboxes(
    bboxes: List[Dict],
    image_width: int,
    image_height: int,
    verbose: bool = True
) -> Tuple[List[Dict], int]:
    """
    Validate and filter invalid bboxes.
    
    Args:
        bboxes: List of bbox dicts
        image_width: Image width
        image_height: Image height
        verbose: Print info
    
    Returns:
        valid_bboxes: List of valid bboxes
        skipped_count: Number of invalid boxes
    """
    valid = []
    skipped = 0
    
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox['coords']
        
        # Check validity
        if x1 >= x2 or y1 >= y2:
            skipped += 1
            if verbose:
                print(f"  ⚠️  Skipped invalid bbox: {bbox['coords']}")
            continue
        
        # Clamp to image bounds
        x1 = max(0, min(x1, image_width - 1))
        x2 = max(0, min(x2, image_width))
        y1 = max(0, min(y1, image_height - 1))
        y2 = max(0, min(y2, image_height))
        
        # Check if completely outside
        if x1 >= image_width or x2 <= 0 or y1 >= image_height or y2 <= 0:
            skipped += 1
            if verbose:
                print(f"  ⚠️  Skipped out-of-bounds bbox: {bbox['coords']}")
            continue
        
        # Check if zero area after clamping
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            skipped += 1
            if verbose:
                print(f"  ⚠️  Skipped zero-area bbox: {bbox['coords']}")
            continue
        
        # Update coords
        bbox = {**bbox, 'coords': (x1, y1, x2, y2)}
        valid.append(bbox)
    
    return valid, skipped