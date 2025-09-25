import os
from io import BytesIO
from PIL import Image
from config import GENAI_AVAILABLE

if GENAI_AVAILABLE:
    import google.generativeai as genai


def run_enhanced_ai_edit(crude_collage_image: Image.Image, user_prompt: str):
    if not GENAI_AVAILABLE:
        return None, "❌ Gemini library not found."

    api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        return None, "❌ GOOGLE_AI_STUDIO_API_KEY not found."

    base_prompt = (
        "You are a photorealistic rendering specialist. You will receive a single image that is a crude, flat collage. "
        "The placement, scale, and layering of objects in this collage are the final, desired composition. \n\n"
        "**Your task is to transform this crude collage into a photorealistic photograph.**\n\n"
        "**Instructions:**\n"
        "1.  For each pasted object in the image, you must **intelligently re-render it in 3D.** This means you must adjust its perspective, 'face angle', and orientation to make it look physically correct and naturally placed within the room's 3D space.\n"
        "2.  Create realistic lighting, highlights, and shadows for each object that perfectly match the room's light sources.\n"
        "3.  Seamlessly blend all objects into the scene.\n\n"
        "**CRITICAL RULE:** Do NOT move the objects from their floor positions. The composition is fixed. Your entire job is the 3D re-rendering and lighting to achieve realism."
    )

    final_prompt = (
        f"{base_prompt}\n\n"
        f"Apply this final user instruction to the overall mood and style: \"{user_prompt}\""
        if user_prompt else base_prompt
    )
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-image-preview")

        response = model.generate_content(
            [crude_collage_image, final_prompt]
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