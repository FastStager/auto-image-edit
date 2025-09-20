import cv2
import numpy as np
from PIL import Image
import gradio as gr

from config import detection_results, NAMED_COLORS
from drawing import draw_contours_with_selection

def get_next_id() -> int:
    current_id = detection_results["next_id"]
    detection_results["next_id"] += 1
    return current_id

def find_contour_by_click(annotations: list[dict], click_x: int, click_y: int) -> dict | None:
    for annot in reversed(annotations): 
        contour = annot.get('contour')
        if contour is None: continue
        if cv2.pointPolygonTest(contour, (click_x, click_y), False) >= 0:
            return annot
    return None

def handle_click(evt, image_type: str):
    if evt is None:
        image = detection_results.get(f"{image_type}_image")
        annotations = detection_results.get(f"{image_type}_annotations", [])
        selected_id = detection_results.get(f"selected_{image_type}")
        updated_image = draw_contours_with_selection(image, annotations, selected_id)
        return updated_image, {}, "No click detected. Please try again."

    click_x, click_y = evt.index
    annotations = detection_results[f"{image_type}_annotations"]
    image = detection_results.get(f"{image_type}_image")
    
    clicked_annot = find_contour_by_click(annotations, click_x, click_y)
    
    if clicked_annot:
        selected_id = clicked_annot['id']
        x, y, w, h = clicked_annot['bbox']
        color_name = clicked_annot.get('color_name', 'unknown')
        selected_info = {'id': selected_id, 'x': x, 'y': y, 'width': w, 'height': h, 'color': color_name}
        detection_results[f"selected_{image_type}"] = selected_id
        updated_image = draw_contours_with_selection(image, annotations, selected_id)
        info_text = f"✅ Selected ID {selected_id}"
        return updated_image, selected_info, info_text
    else:
        detection_results[f"selected_{image_type}"] = None
        updated_image = draw_contours_with_selection(image, annotations, None)
        return updated_image, {}, "❌ Click inside a contour to select it."

def handle_click_and_populate_edit_fields(evt, image_type: str):
    """Wrapper that populates editor fields for the clicked object."""
    updated_image, selected_info, info_text = handle_click(evt, image_type)
    if selected_info:
        obj_id = selected_info.get('id')
        x = selected_info.get('x')
        y = selected_info.get('y')
        w = selected_info.get('width')
        h = selected_info.get('height')
        color = selected_info.get('color')
        return updated_image, selected_info, info_text, obj_id, x, y, w, h, color, gr.update(open=True)
    else:
        return updated_image, selected_info, info_text, None, None, None, None, None, None, gr.update(open=False)

def load_by_id(rect_id: int, image_type: str):
    rect_id = int(rect_id)
    annotations = detection_results[f"{image_type}_annotations"]
    image = detection_results.get(f"{image_type}_image")
    
    for annot in annotations:
        if annot['id'] == rect_id:
            x, y, w, h = annot['bbox']
            data = {'id': rect_id, 'x': x, 'y': y, 'width': w, 'height': h, 'color': annot['color_name']}
            detection_results[f"selected_{image_type}"] = rect_id
            updated_image = draw_contours_with_selection(image, annotations, rect_id)
            return updated_image, data, f"✅ Loaded ID {rect_id}"
            
    return image, {}, f"❌ ID {rect_id} not found"

def transfer_to_empty_editor() -> tuple:
    empty_img = detection_results.get("empty_image")
    if not empty_img:
        return None, "❌ Run detection first.", ""
    
    annotations = [
        dict(a, id=get_next_id(), mask=a['mask'].copy(), contour=a['contour'].copy())
        for a in detection_results["staged_annotations"]
    ]
    detection_results.update({"empty_annotations": annotations, "selected_empty": None})
    empty_with_contours = draw_contours_with_selection(empty_img, annotations)
    return empty_with_contours, f"✅ Transferred {len(annotations)} objects.", get_annotations_info("empty")

def apply_editor_changes(image_type: str, obj_id: int, new_x: int, new_y: int, new_w: int, new_h: int, new_color: str) -> tuple:
    """Applies geometry and color changes to the selected object in the specified image (staged or empty)."""
    if obj_id is None:
        return None, "❌ No object selected to apply changes to.", ""
        
    obj_id = int(obj_id)
    image_key = f"{image_type}_image"
    annots_key = f"{image_type}_annotations"
    
    image = detection_results.get(image_key)
    annotations = detection_results.get(annots_key, [])
    
    target_annot = next((a for a in annotations if a['id'] == obj_id), None)

    if not target_annot:
        updated_image = draw_contours_with_selection(image, annotations, obj_id)
        return updated_image, f"❌ Could not find object with ID {obj_id}.", get_annotations_info(image_type)

    x_orig, y_orig, w_orig, h_orig = target_annot['bbox']
    new_x, new_y, new_w, new_h = int(new_x), int(new_y), int(new_w), int(new_h)
    
    if w_orig > 0 and h_orig > 0:
        scale_x = new_w / w_orig
        scale_y = new_h / h_orig
        original_contour = target_annot['contour']
        new_contour = original_contour.copy().astype(np.float32)
        new_contour[:, :, 0] = new_x + (new_contour[:, :, 0] - x_orig) * scale_x
        new_contour[:, :, 1] = new_y + (new_contour[:, :, 1] - y_orig) * scale_y
        target_annot['contour'] = new_contour.astype(np.int32)
    
    target_annot['bbox'] = (new_x, new_y, new_w, new_h)

    if new_color and new_color in NAMED_COLORS:
        color_data = NAMED_COLORS[new_color]
        target_annot.update({
            "color": color_data["rgb"],
            "color_name": new_color,
            "hex_color": color_data["hex"]
        })

    updated_image = draw_contours_with_selection(image, annotations, obj_id)
    status_msg = f"✅ Applied changes to ID {obj_id} in {image_type} image."
    
    return updated_image, status_msg, get_annotations_info(image_type)

def get_annotations_info(image_type: str) -> str:
    annotations = detection_results[f"{image_type}_annotations"]
    return "\n".join([f"ID {a['id']}: {a['color_name']} at {a['bbox']}" for a in annotations]) or "No objects"