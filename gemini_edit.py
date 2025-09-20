import os
from io import BytesIO
from PIL import Image
from config import GENAI_AVAILABLE

if GENAI_AVAILABLE:
    import google.generativeai as genai

def run_enhanced_ai_edit(staged_sheet_image: Image.Image, marker_image: Image.Image, user_prompt: str):
    if not GENAI_AVAILABLE:
        return None, "❌ Gemini library not found."
    api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        return None, "❌ GOOGLE_AI_STUDIO_API_KEY not found."

    text_prompt = """
    You are an expert AI virtual staging assistant. Your task is to perform a high-fidelity furniture placement and integration task with absolute precision.

    You will receive TWO primary images and a text prompt:
    1.  **[Staged Sheet]:** This image contains the furniture assets you must use. Each piece of furniture is marked with a uniquely colored circle. This is your asset and color-key library.
    2.  **[Marker Image]:** This is the empty room. It contains colored circles that are targets. Each target circle's color corresponds to a piece of furniture from the [Staged Sheet].
    3.  **[User Prompt]:** A text instruction for final lighting and mood adjustments.

    **YOUR MANDATORY, UNBREAKABLE WORKFLOW:**

    1.  **EXTRACT ASSETS:** From the [Staged Sheet], individually extract each piece of furniture, ignoring the circles on top of them. Note the color of the circle associated with each piece.
    2.  **MATCH TARGETS:** For each extracted piece of furniture, find the circle of the *same color* in the [Marker Image]. This is your target.
    3.  **PLACE AND RENDER IN 3D:** Place the furniture asset at the exact location and scale of its corresponding target circle in the [Marker Image]. You MUST render the furniture to match the room's 3D perspective, camera angle, and geometry. It must look physically grounded in that spot.
    4.  **REMOVE ALL CIRCLES:** After placing the furniture, you must completely remove ALL colored circles from BOTH the furniture and the background using a seamless, content-aware fill. No trace of any circle should remain.
    5.  **INTEGRATE LIGHTING & SHADOWS:** Re-light the newly placed furniture to perfectly match the empty room's light sources. Generate realistic and physically accurate shadows.
    6.  **APPLY USER PROMPT:** Apply the user's text prompt for final adjustments.

    The final output must be a single, photorealistic image with no circles, where the furniture from the Staged Sheet has been perfectly rendered into the Marker Image's room at the specified locations.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-image-preview")

        response = model.generate_content(
            [text_prompt, staged_sheet_image, marker_image, user_prompt]
        )
        
        if not (response.candidates and response.candidates[0].content and response.candidates[0].content.parts):
            return None, f"❌ AI response is empty or malformed."
        image_parts = [p.inline_data.data for p in response.candidates[0].content.parts if p.inline_data]
        if image_parts:
            result_image = Image.open(BytesIO(image_parts[0]))
            return result_image, "✅ AI Edit successful!"
        else:
            refusal_text = response.text if hasattr(response, 'text') else str(response.prompt_feedback)
            return None, f"❌ AI did not return an image. Reason: {refusal_text}"
        
    except Exception as e:
        return None, f"❌ An error occurred with the AI Edit: {type(e).__name__}: {e}"