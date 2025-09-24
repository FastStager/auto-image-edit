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
        "Place each furniture item from the staged image at the location of its corresponding "
        "colored circle in the empty room. Scale, rotate, and orient each piece to match the room’s "
        "perspective and layout. After placement, remove all colored circles and clean the background completely. "
        "Re-render furniture with correct lighting, shadows, reflections, and seamless blending so the final "
        "image looks like a natural photograph."
    )

    final_prompt = (
        f"{base_prompt}\n\n Additionally, apply this creative instruction: "
        f"\"{user_prompt}. Blend seamlessly into the environment as hyper-real. Enhance final result.\""
        if user_prompt else base_prompt
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