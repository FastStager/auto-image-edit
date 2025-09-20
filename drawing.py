import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import NAMED_COLORS

def get_color_name_from_rgb(rgb_color: list[int]) -> str:
    min_distance, closest_color = float('inf'), "unknown"
    for name, data in NAMED_COLORS.items():
        dist = sum((a - b) ** 2 for a, b in zip(rgb_color, data["rgb"]))
        if dist < min_distance:
            min_distance, closest_color = dist, name
    return closest_color

def draw_contours_with_selection(image_pil: Image.Image, annotations: list[dict], selected_id: int = None) -> Image.Image | None:
    if image_pil is None: return None
    if not annotations: return image_pil.copy()
    
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    for annot in annotations:
        contour = annot.get('contour')
        if contour is None: continue

        rgb_color = annot.get('color', [255, 0, 0])
        bgr_color = tuple(rgb_color[::-1]) 
        obj_id = annot.get('id', 0)
        is_selected = (obj_id == selected_id)
        
        thickness = 5 if is_selected else 3
        cv2.drawContours(image_cv, [contour], -1, bgr_color, thickness)
        
        if is_selected:
            cv2.drawContours(image_cv, [contour], -1, (0, 255, 255), thickness + 4,-1)
            cv2.drawContours(image_cv, [contour], -1, bgr_color, thickness)


    image_pil_with_contours = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil_with_contours)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    for annot in annotations:
        x, y, w, h = annot['bbox']
        color_name = annot.get('color_name', 'unknown')
        obj_id = annot.get('id', 0)
        is_selected = (obj_id == selected_id)
        
        label_text = f"ID: {obj_id} ({color_name})"
        label_bbox = draw.textbbox((0,0), label_text, font=font)
        label_w, label_h = label_bbox[2] - label_bbox[0], label_bbox[3] - label_bbox[1]
        
        label_x, label_y = x, (y - label_h - 10 if y > label_h + 10 else y + h + 10)
        
        bg_color = (255, 255, 100) if is_selected else "white"
        text_color = "darkblue" if is_selected else "black"
        
        draw.rectangle([label_x, label_y, label_x + label_w + 10, label_y + label_h + 10], fill=bg_color, outline=tuple(annot['color']), width=2)
        draw.text((label_x + 5, label_y + 5), label_text, fill=text_color, font=font)
        
    return image_pil_with_contours

def draw_circles_on_image(image_pil: Image.Image, annotations: list[dict], radius: int = 25) -> Image.Image:
    if not image_pil or not annotations: return image_pil
    img_with_circles = image_pil.copy()
    draw = ImageDraw.Draw(img_with_circles)
    for annot in annotations:
        x, y, w, h = annot['bbox']
        center_x, center_y = x + w // 2, y + h // 2
        color = tuple(annot.get('color', (255, 0, 0)))
        draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius], fill=color, outline="black", width=3)
    return img_with_circles