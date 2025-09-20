import os
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from segment_anything import build_sam, SamPredictor
from config import DEVICE, HF_GD_MODEL_ID, SAM_CHECKPOINT_PATH

def load_models():
    """Loads and initializes all required AI models."""
    print("\n--- Loading Models ---")

    hf_gd_processor = AutoProcessor.from_pretrained(HF_GD_MODEL_ID)
    hf_gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(HF_GD_MODEL_ID).to(DEVICE)
    hf_gd_model.eval()
    print(f"Hugging Face Grounding DINO model '{HF_GD_MODEL_ID}' loaded.")

    if not os.path.exists(SAM_CHECKPOINT_PATH):
        print(f"SAM checkpoint '{SAM_CHECKPOINT_PATH}' not found. Downloading...")
        os.system(f"wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O {SAM_CHECKPOINT_PATH}")
        print("SAM checkpoint downloaded.")
    sam_predictor = SamPredictor(build_sam(checkpoint=SAM_CHECKPOINT_PATH).to(DEVICE))
    print(f"SAM Predictor model loaded from {SAM_CHECKPOINT_PATH}")

    print("--- Models Loaded ---")
    return hf_gd_processor, hf_gd_model, sam_predictor