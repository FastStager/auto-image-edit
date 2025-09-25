import os
from io import BytesIO
from PIL import Image
from config import GENAI_AVAILABLE

if GENAI_AVAILABLE:
    import google.generativeai as genai


def run_enhanced_ai_edit(blueprint_image: Image.Image, furniture_palette_image: Image.Image, user_prompt: str):
    if not GENAI_AVAILABLE:
        return None, "❌ Gemini library not found."

    api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        return None, "❌ GOOGLE_AI_STUDIO_API_KEY not found."

    base_prompt = (
        "You are a graphics rendering engine. You will receive two images:\n"
        "1. **Blueprint Image:** An empty room with colored vector shapes drawn on it. These are instructions.\n"
        "2. **Furniture Palette:** A collection of clean furniture cutouts on a transparent background. These are your assets.\n\n"
        "**Your task is to execute the blueprint instructions precisely:**\n"
        "1. For each colored rectangle on the blueprint, take a matching furniture piece from the palette.\n"
        "2. Place the asset into the room, scaling it to fit the **exact bounds of its colored rectangle**.\n"
        "3. Orient the asset so its 'front' faces the direction of the **white line** inside its rectangle.\n"
        "4. After placing all assets, re-render them with realistic perspective, lighting, and shadows to match the room.\n"
        "5. The final image must be a photorealistic scene. **Completely remove all colored rectangles and white lines** from the final output. They are instructions, not objects."
    )

    final_prompt = (
        f"{base_prompt}\n\nApply this final instruction: "
        f"\"{user_prompt}. Ensure the final result is hyper-realistic and perfectly blended.\""
        if user_prompt else base_prompt
    )
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-image-preview")

        response = model.generate_content(
            [blueprint_image, furniture_palette_image, final_prompt]
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