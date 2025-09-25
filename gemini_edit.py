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
        "You are a photorealistic rendering engine. You will receive a single image that is a crude collage of furniture pasted into a room. The furniture has been placed exactly where the user wants it, but it looks fake.\n\n"
        "**Your ONLY task is to make this collage look like a real photograph.**\n\n"
        "**What to do:**\n"
        "1.  **Integrate Lighting:** Analyze the room's light sources and re-render the pasted furniture with perfectly matching lighting, highlights, and shadows.\n"
        "2.  **Add Realistic Shadows:** Create soft, physically accurate shadows cast by the furniture onto the floor and other objects.\n"
        "3.  **Correct Perspective:** Make minor adjustments to the furniture's perspective to ensure it perfectly matches the room's geometry.\n"
        "4.  **Blend Edges:** Seamlessly blend the edges of the furniture into the background.\n\n"
        "**CRITICAL RULES (FAILURE IF NOT FOLLOWED):**\n"
        "- **DO NOT MOVE a single piece of furniture.** The position and layering are final and must be preserved.\n"
        "- **DO NOT ADD any new objects, furniture, or decorations.**\n"
        "- **DO NOT REMOVE any of the furniture provided in the collage.**\n"
        "- The final output must be a single, coherent, photorealistic image."
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