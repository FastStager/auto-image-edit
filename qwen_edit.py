import os
import base64
import requests
from PIL import Image
from io import BytesIO

CAMERA_PROMPT_MAP = {
    "Rotate camera 45° left": "Rotate camera 45° left",
    "Rotate camera 45° right": "Rotate camera 45° right",
    "Rotate camera 90° left": "Rotate camera 90° left",
    "Rotate camera 90° right": "Rotate camera 90° right",
    "Switch to close-up lens": "Switch to close-up lens",
    "Switch to zoom out lens": "Switch to zoom out lens",
}

def run_qwen_edit(image_b64: str, prompt: str) -> str | None:
    """
    Makes a synchronous API call to a RunPod endpoint to perform a multi-angle view edit on an image.

    Args:
        image_b64: The base64-encoded string of the input PNG image.
        prompt: The user-selected prompt from the UI (e.g., "Rotate camera 90° right").

    Returns:
        The base64-encoded string of the edited image if successful, otherwise None.
    """
    api_key = os.getenv("RUNPOD_API_KEY")
    endpoint_id = os.getenv("QWEN_ENDPOINT_ID")

    if not api_key or not endpoint_id:
        print("🔴 ERROR: Missing required environment variables: RUNPOD_API_KEY or QWEN_ENDPOINT_ID.")

        return None

    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"

    camera_option = CAMERA_PROMPT_MAP.get(prompt)
    if not camera_option:
        print(f"🔴 ERROR: Invalid camera prompt received: '{prompt}'")
        return None

    payload = {
        "input": {
            "image": image_b64,
            "prompt": "",  
            "camera_work_option": camera_option,
            "num_inference_steps": 8,
            "true_guidance_scale": 1.5,
        }
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    print(f"🚀 Sending request to Qwen Edit endpoint for '{camera_option}'...")

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        
        data = response.json()

        if data.get("status") == "COMPLETED" and data.get("output", {}).get("image"):
            print("✅ Request successful. Received edited image.")
            return data["output"]["image"]
        else:
            error_message = data.get("output", {}).get("error", "Unknown API error in response.")
            print(f"🔴 API Error: {error_message}")
            return None

    except requests.exceptions.Timeout:
        print("🔴 ERROR: The request to the AI service timed out.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"🔴 ERROR: An HTTP request error occurred: {e}")
        return None
    except Exception as e:
        print(f"🔴 An unexpected error occurred during the API call: {e}")
        return None