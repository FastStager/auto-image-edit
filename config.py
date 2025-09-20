import torch

DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HF_GD_MODEL_ID: str = "IDEA-Research/grounding-dino-tiny"
SAM_CHECKPOINT_PATH: str = 'sam_vit_h_4b8939.pth'

GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    print("WARNING: google-generativeai not found. The 'Enhanced AI Edit' feature will be disabled.")

NAMED_COLORS: dict[str, dict[str, object]] = {
    "red": {"rgb": [255, 0, 0], "hex": "#FF0000"},
    "blue": {"rgb": [0, 0, 255], "hex": "#0000FF"},
    "green": {"rgb": [0, 255, 0], "hex": "#00FF00"},
    "yellow": {"rgb": [255, 255, 0], "hex": "#FFFF00"},
    "orange": {"rgb": [255, 165, 0], "hex": "#FFA500"},
    "purple": {"rgb": [128, 0, 128], "hex": "#800080"},
    "pink": {"rgb": [255, 192, 203], "hex": "#FFC0CB"},
    "brown": {"rgb": [139, 69, 19], "hex": "#8B4513"},
    "black": {"rgb": [0, 0, 0], "hex": "#000000"},
    "white": {"rgb": [255, 255, 255], "hex": "#FFFFFF"},
}

detection_results: dict[str, object] = {
    "staged_annotations": [],
    "empty_annotations": [],
    "staged_image": None,
    "empty_image": None,
    "staged_image_bg_removed": None, 
    "selected_staged": None,
    "selected_empty": None,
    "next_id": 1,
    "generated_images": [],
    "original_cutouts": [] 
}