import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.ops import nms
from segment_anything import SamPredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from config import DEVICE, NAMED_COLORS, detection_results
from drawing import draw_contours_with_selection
from editor_logic import get_next_id, get_annotations_info

def detect_hf_grounding_dino_raw(image_pil: Image.Image, text_prompt: str, processor, model) -> dict:
    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    return processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, target_sizes=[image_pil.size[::-1]]
    )[0]

def segment_sam(image_rgb_numpy: np.ndarray, sam_predictor: SamPredictor, boxes_xyxy: torch.Tensor) -> torch.Tensor:
    sam_predictor.set_image(image_rgb_numpy)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy.to(DEVICE), image_rgb_numpy.shape[:2])
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False,
    )
    return masks.cpu()

def run_detection_and_populate_editor(empty_img_path, staged_img_pil: Image.Image, prompts_str, processor, model, predictor):
    empty_img_pil = Image.open(empty_img_path).convert("RGB")
    detection_results.update({"empty_image": empty_img_pil, "staged_image": staged_img_pil, "selected_staged": None})
    
    staged_img_np = np.array(staged_img_pil)
    
    diff_mask = cv2.dilate(cv2.absdiff(np.array(empty_img_pil), staged_img_np), np.ones((5,5),np.uint8))
    changed_pixels_mask = np.any(diff_mask > 30, axis=2)

    prompts = [p.strip() for p in prompts_str.split(',') if p.strip()] or ["object"]
    gd_results = detect_hf_grounding_dino_raw(staged_img_pil, ". ".join(prompts) + ".", processor, model)
    gd_boxes = gd_results.get('boxes', torch.tensor([])) 
    
    if len(gd_boxes) == 0:
        detection_results["staged_annotations"] = []
        return staged_img_pil, "❌ No objects detected", "", {}

    sam_masks = segment_sam(staged_img_np, predictor, gd_boxes)
    
    annotations = []
    colors = list(NAMED_COLORS.keys())
    for i, mask_tensor in enumerate(sam_masks):
        mask = mask_tensor[0].numpy()
        
        if np.sum(mask) < 200 or np.sum(np.logical_and(mask, changed_pixels_mask)) / np.sum(mask) < 0.6:
            continue
            
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        
        main_contour = max(contours, key=cv2.contourArea)
        bbox = cv2.boundingRect(main_contour)
        
        color_name = colors[len(annotations) % len(colors)]
        color_data = NAMED_COLORS[color_name]
        
        annotations.append({
            "id": get_next_id(),
            "mask": mask,
            "contour": main_contour,
            "bbox": bbox,
            "color": color_data["rgb"],
            "color_name": color_name,
            "hex_color": color_data["hex"],
            "manual": False
        })
        
    detection_results["staged_annotations"] = annotations
    staged_with_contours = draw_contours_with_selection(staged_img_pil, annotations)
    
    return staged_with_contours, f"✅ Detected {len(annotations)} objects.", get_annotations_info("staged"), {}

def remove_background_and_add_border():
    """First BG removal step: Creates a transparent image with only the detected objects and their colored contour borders."""
    staged_img = detection_results.get("staged_image")
    annotations = detection_results.get("staged_annotations")
    if not staged_img or not annotations:
        return None, "❌ Run detection first."
    
    staged_np = np.array(staged_img)
    rgba_image = cv2.cvtColor(staged_np, cv2.COLOR_RGB2RGBA)
    
    combined_mask = np.zeros(staged_np.shape[:2], dtype=np.uint8)
    for annot in annotations:
        combined_mask[annot['mask']] = 255
        
    rgba_image[:, :, 3] = combined_mask
    
    for annot in annotations:
        bgr_color = tuple(annot['color'][::-1]) + (255,) 
        cv2.drawContours(rgba_image, [annot['contour']], -1, bgr_color, 3)

    result_pil = Image.fromarray(rgba_image)
    detection_results["staged_image_bg_removed"] = result_pil
    
    return result_pil, f"✅ Background removed for {len(annotations)} objects."