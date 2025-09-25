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
        "You are a photorealistic image editor. You will receive an 'empty room' image and a 'furniture reference' image.\n"
        "The 'furniture reference' image contains furniture, each with a colored disk on the floor beneath it. This disk is your **primary instruction**.\n\n"
        "**Your instructions are strict and must be followed precisely:**\n"
        "1.  **POSITION:** The center of the disk indicates the **exact floor position** for the furniture in the empty room.\n"
        "2.  **ORIENTATION:** The disk has a white pointer line. This pointer indicates the **'front-facing' direction**. You MUST orient the furniture to match this precise direction.\n"
        "3.  **RENDER:** After placing and orienting the furniture correctly, blend it photorealistically into the room with correct lighting, scale, and shadows.\n"
        "4.  **CLEANUP:** You MUST completely remove the colored disk and its pointer from the final image. No trace of the instructional disk should remain.\n\n"
        "**CRITICAL RULES:**\n"
        "- DO NOT add any new objects, furniture, or decorations.\n"
        "- DO NOT deviate from the specified position or orientation.\n"
        "- Render ONLY the furniture provided in the reference image."
    )

    final_prompt = (
        f"{base_prompt}\n\n"
        f"After following all rules, apply this final user styling: \"{user_prompt}\""
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