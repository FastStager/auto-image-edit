import os
from io import BytesIO
from PIL import Image
from config import GENAI_AVAILABLE

if GENAI_AVAILABLE:
    import google.generativeai as genai


def run_enhanced_ai_edit(reference_collage_image: Image.Image, furniture_assets_image: Image.Image, user_prompt: str):
    if not GENAI_AVAILABLE:
        return None, "❌ Gemini library not found."

    api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        return None, "❌ GOOGLE_AI_STUDIO_API_KEY not found."

    base_prompt = (
        "You are a professional virtual staging artist. You will receive two images:\n"
        "1.  **Reference Collage:** This image shows a room with crudely pasted furniture. This collage defines the **final, non-negotiable position, scale, and layering** of each object.\n"
        "2.  **Furniture Assets:** This image contains the clean, original, high-quality versions of the furniture on a transparent background.\n\n"
        "**Your task is to create a photorealistic version of the 'Reference Collage'. Follow these steps:**\n"
        "1.  For each piece of furniture in the collage, identify its clean version from the 'Furniture Assets' image.\n"
        "2.  Re-render the clean asset in the **exact same position and layer** as shown in the collage.\n"
        "3.  **Crucially, you MUST intelligently adjust the 3D perspective, orientation, and rotation** of each piece to make it look natural and physically correct within the room's 3D space.\n"
        "4.  Integrate each piece seamlessly by adding perfect lighting, highlights, and shadows that match the room's environment.\n\n"
        "**Key Objective:** The final image must respect the layout of the 'Reference Collage' but look like a real photograph, not a flat composition. The furniture must look like it truly 'sits' in the room.\n\n"
        "**Critical Rules:**\n"
        "- **DO NOT change the floor position** of any object from where it is in the collage.\n"
        "- **DO NOT change the layering** (which object is in front of another).\n"
        "- **DO NOT add or remove** any furniture."
    )

    final_prompt = (
        f"{base_prompt}\n\n"
        f"Apply this user's creative direction to the final image: \"{user_prompt}\""
        if user_prompt else base_prompt
    )
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-image-preview")

        response = model.generate_content(
            [reference_collage_image, furniture_assets_image, final_prompt]
        )

        if not (response.candidates and response.candidates[from __future__ import annotations
0].content and response.candidates[0].content.parts):
            return None, "❌ AI response is empty or malformed."

        image_parts = [p.inline_data.data for p in response.candidates[0].content.parts if p.inline_data]
        if image_parts:
            result_image = Image.open(BytesIO(image_parts[0]))
            return result_image, "✅ Enhanced AI Edit successful!"
        else:
            refusal_text = response.text if hasattr(response, 'text') else str(response.prompt_feedback)
            return None, f"❌ AI did not return an image. Reason: {refusal_text}"

    except Exception as e:
        return None, f"❌ An error occurred with the AI Edit: {type(e).__name__}: {e}"