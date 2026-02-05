from pathlib import Path
from typing import List, Dict, Tuple, Union
from PIL import Image, ImageDraw, ImageFont

from utils.image_utils import load_image, save_image


def get_label_text(bbox_dict: Dict) -> str:
    """
    Format label text for bbox.
    
    Args:
        bbox_dict: Bbox dict with 'id', 'description', etc.
    
    Returns:
        Label text
    """
    label_parts = [f"{bbox_dict['id']}"]
    
    if bbox_dict.get('description'):
        desc = bbox_dict['description']
        if len(desc) > 40:
            desc = desc[:37] + "..."
        label_parts.append(desc)
    
    if bbox_dict.get('was_fixed'):
        label_parts.append("[FIXED]")
    
    return ": ".join(label_parts)


def get_bbox_color(
    bbox_dict: Dict,
    color_palette: List[str],
    index: int
) -> str:
    """
    Get color for bbox.
    
    Args:
        bbox_dict: Bbox dict
        color_palette: List of color hex codes
        index: Bbox index
    
    Returns:
        Color hex code
    """
    if bbox_dict.get('was_fixed'):
        return '#FF0000'  # Red for fixed boxes
    else:
        return color_palette[index % len(color_palette)]


def draw_boxes_on_image(
    image_path: Union[str, Path],
    bboxes: List[Dict],
    output_path: Union[str, Path],
    line_thickness: int = 2,
    font_size: int = 10,
    color_palette: List[str] = None
) -> int:
    """
    Draw bounding boxes on image.
    
    Args:
        image_path: Path to image
        bboxes: List of bbox dicts with 'coords'
        output_path: Output path for annotated image
        line_thickness: Box border thickness
        font_size: Label font size
        color_palette: List of color hex codes
    
    Returns:
        Number of boxes drawn
    """
    # Default color palette
    if color_palette is None:
        color_palette = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
            '#F7DC6F', '#BB8FCE', '#85C1E2', '#F06292', '#AED581'
        ]
    
    # Load image
    img = load_image(image_path)
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size
    
    # Load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    drawn_count = 0
    
    for idx, bbox_dict in enumerate(bboxes):
        x1, y1, x2, y2 = bbox_dict['coords']
        
        # Get color
        color = get_bbox_color(bbox_dict, color_palette, idx)
        
        # Determine if tiny (for thick border)
        w = x2 - x1
        h = y2 - y1
        is_tiny = w < 20 or h < 20
        thickness = line_thickness + 1 if is_tiny else line_thickness
        
        # Draw rectangle
        for offset in range(thickness):
            draw.rectangle(
                [(x1 - offset, y1 - offset), (x2 + offset, y2 + offset)],
                outline=color,
                width=1
            )
        
        # Draw label
        label = get_label_text(bbox_dict)
        
        # Get text size
        bbox_text = draw.textbbox((0, 0), label, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        padding = 2
        
        # Smart label placement
        if w < 30 and h > 50:
            # Narrow tall box - label on right
            label_x = x2 + 3
            label_y = y1 + (h // 2) - (text_height // 2)
        elif h < 30 and w > 50:
            # Short wide box - label above/below
            label_x = x1 + (w // 2) - (text_width // 2)
            label_y = y1 - text_height - padding * 2 - 3 if y1 > text_height + 10 else y2 + 3
        else:
            # Normal - label above
            label_x = x1
            label_y = y1 - text_height - padding * 2 - 3 if y1 > text_height + 10 else y2 + 3
        
        # Clamp label to image bounds
        label_x = max(2, min(label_x, img_width - text_width - padding * 2))
        label_y = max(2, min(label_y, img_height - text_height - padding * 2))
        
        # Draw label background
        draw.rectangle(
            [(label_x - padding, label_y - padding),
             (label_x + text_width + padding, label_y + text_height + padding)],
            fill=color
        )
        
        # Draw label text
        draw.text((label_x, label_y), label, fill='white', font=font)
        
        drawn_count += 1
    
    # Save
    save_image(img, output_path, quality=95)
    
    return drawn_count


def visualize_all_images(
    images_data: List[Dict],
    output_dir: Union[str, Path],
    verbose: bool = True
) -> Dict:
    """
    Visualize bounding boxes for all images.
    
    Args:
        images_data: List of dicts with 'image_path' and 'boxes'
        output_dir: Output directory
        verbose: Print progress
    
    Returns:
        Summary dict with statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"VISUALIZING BOUNDING BOXES")
        print(f"Output: {output_dir}")
        print(f"{'='*70}\n")
    
    total_boxes = 0
    successful = 0
    
    for img_data in images_data:
        image_name = img_data['image']
        image_path = Path(img_data['image_path'])
        bboxes = img_data['boxes']
        
        if verbose:
            print(f"🖼️  {image_name}")
        
        if not image_path.exists():
            if verbose:
                print(f"   ⚠️  Image not found")
                print()
            continue
        
        output_path = output_dir / f"annotated_{image_name}"
        
        num_drawn = draw_boxes_on_image(
            image_path=image_path,
            bboxes=bboxes,
            output_path=output_path
        )
        
        if verbose:
            print(f"   ✅ Drew {num_drawn} boxes → {output_path.name}")
            print()
        
        total_boxes += num_drawn
        successful += 1
    
    if verbose:
        print("="*70)
        print("✅ COMPLETE!")
        print("="*70)
        print(f"  Images annotated: {successful}")
        print(f"  Total boxes drawn: {total_boxes}")
        print(f"  Output dir: {output_dir.absolute()}")
        print("="*70 + "\n")
    
    return {
        'images_annotated': successful,
        'total_boxes': total_boxes,
        'output_dir': str(output_dir)
    }