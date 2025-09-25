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
        "The 'furniture reference' image contains one or more pieces of furniture. Each piece of furniture is positioned on top of a colored floor disk.\n\n"
        "**Your instructions are strict and simple:**\n"
        "1.  **TRANSFER:** For each furniture piece in the 'furniture reference' image, you must transfer it into the 'empty room'.\n"
        "2.  **POSITION:** The colored disk underneath each piece of furniture indicates its **exact floor position** in the empty room. You MUST place the furniture at this precise location.\n"
        "3.  **RENDER:** After placing the furniture correctly, adjust its scale, perspective, lighting, and shadows to blend it seamlessly and photorealistically into the room.\n"
        "4.  **CLEANUP:** You MUST completely remove the colored floor disk from the final image. There should be absolutely no trace of the colored disk left.\n\n"
        "**CRITICAL RULES:**\n"
        "- DO NOT add any new objects, furniture, or decorations.\n"
        "- DO NOT change the position of the furniture from its disk location.\n"
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