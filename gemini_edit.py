import os
from io import BytesIO
from PIL import Image
from config import GENAI_AVAILABLE

if GENAI_AVAILABLE:
    import google.generativeai as genai


def run_enhanced_ai_edit(empty_room_image: Image.Image, furniture_reference_image: Image.Image, user_prompt: str):
    if not GENAI_AVAILABLE:
        return None, "❌ Gemini library not found."

    api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        return None, "❌ GOOGLE_AI_STUDIO_API_KEY not found."

    base_prompt = (
        "You are a photorealistic image editor. You will receive two images: an 'empty room' and a 'furniture reference'.\n"
        "The 'furniture reference' image contains furniture placed on top of colored floor disks. Each disk has a white pointer line.\n\n"
        "**Your instructions are strict and must be followed precisely:**\n"
        "1.  **TRANSFER & POSITION:** For each furniture piece in the 'furniture reference' image, transfer it to the 'empty room'. The colored disk indicates the **exact floor position** where the furniture's base should be.\n"
        "2.  **ORIENT:** The white line on the disk is a pointer. It indicates the **exact 'front-facing' direction** for the furniture. You MUST orient the furniture to face this direction.\n"
        "3.  **RENDER:** After placing and orienting the furniture correctly, adjust its scale, perspective, lighting, and shadows to blend it seamlessly and photorealistically into the room.\n"
        "4.  **CLEANUP:** You MUST completely remove the colored disk and its pointer line from the final image. There should be absolutely no trace of these visual guides.\n\n"
        "**CRITICAL RULES:**\n"
        "- The disk's position and the pointer's direction are NON-NEGOTIABLE commands.\n"
        "- DO NOT add any new objects, furniture, or decorations.\n"
        "- DO NOT change the position or orientation of the furniture from its visual guide.\n"
        "- Render ONLY the furniture provided in the reference image."
    )

    final_prompt = (
        f"{base_prompt}\n\n"
        f"Apply this final user styling: \"{user_prompt}\""
        if user_prompt else base_prompt
    )
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-image-preview")

        response = model.generate_content(
            [empty_room_image, furniture_reference_image, final_prompt]
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