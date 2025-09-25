import os
from io import BytesIO
from PIL import Image
from config import GENAI_AVAILABLE

if GENAI_AVAILABLE:
    import google.generativeai as genai


def run_enhanced_ai_edit(empty_room_with_circles: Image.Image, furniture_image: Image.Image, user_prompt: str):
    if not GENAI_AVAILABLE:
        return None, "❌ Gemini library not found."

    api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        return None, "❌ GOOGLE_AI_STUDIO_API_KEY not found."

    base_prompt = (
        "You are a precise photo-editing tool. Your ONLY job is to composite images according to strict instructions. You are not an interior designer. "
        "Your task has two parts:\n\n"
        "1.  **PLACEMENT (Strict):** For each piece of furniture in the second image (the 'furniture image'), you must place it into the first image (the 'empty room'). The placement for each piece is explicitly defined by a colored circle. The center of the furniture MUST be placed EXACTLY on the center of its corresponding colored circle. This is the ABSOLUTE FINAL location. Do not change it for any reason.\n\n"
        "2.  **RENDERING (Realistic):** After placing the furniture in its exact required spot, you will then adjust its scale, perspective, lighting, and shadows to make it look perfectly photorealistic in that location. Finally, you will completely remove all colored circles from the image, leaving no trace."
    )

    prohibitions = (
        "**CRITICAL PROHIBITIONS:**\n"
        "- **DO NOT ADD NEW OBJECTS:** You are forbidden from adding any furniture, decorations, or objects that are not explicitly provided in the furniture image.\n"
        "- **DO NOT MOVE OR REARRANGE:** You are forbidden from moving the furniture from its designated circle location. Do not 'improve' the layout. Your only job is to place and render.\n"
        "- **DO NOT CORRECT THE USER:** If the user's chosen placement blocks a doorway, window, or seems illogical, you MUST render it there anyway. Follow the placement instruction above all else."
    )

    final_prompt = (
        f"{base_prompt}\n\n{prohibitions}\n\n"
        f"After following all the above rules, apply this final styling instruction: \"{user_prompt}\""
        if user_prompt else f"{base_prompt}\n\n{prohibitions}"
    )
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-image-preview")

        response = model.generate_content(
            [empty_room_with_circles, furniture_image, final_prompt]
        )

        if not (response.candidates and response.candidates[0].content and response.candidates[0].content.parts):
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