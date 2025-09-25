import osfrom io 
import BytesIO
from PIL import Image
from config import GENAI_AVAILABLE

if GENAI_AVAILABLE:
    import google.generativeai as genai


def run_enhanced_ai_edit(ghostly_collage_image: Image.Image, user_prompt: str):
    if not GENAI_AVAILABLE:
        return None, "❌ Gemini library not found."

    api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        return None, "❌ GOOGLE_AI_STUDIO_API_KEY not found."

    base_prompt = (
        "You are a photorealistic rendering expert. The user provides an image that is a 'ghostly collage' where semi-transparent furniture has been placed into a room. This shows you **what** furniture to use and **where** it must be located.\n\n"
        "Your task is to **replace the faint, ghostly furniture with fully solid, hyper-realistic versions.**\n\n"
        "**Execution Steps:**\n"
        "1.  **Identify:** Locate the semi-transparent 'ghost' objects in the image.\n"
        "2.  **Re-render:** For each ghost, re-render it from scratch in the exact same position, but make it fully opaque and solid.\n"
        "3.  **Correct Perspective:** This is critical. You MUST adjust the perspective, rotation, and warping of each piece to make it look truly three-dimensional and perfectly aligned with the room's geometry and vanishing points.\n"
        "4.  **Integrate:** Add perfect, physically-accurate lighting, highlights, and shadows so the new furniture blends seamlessly into the room's environment.\n\n"
        "**Strict Rules:**\n"
        "- Maintain the general position and size of the ghostly placeholders.\n"
        "- DO NOT add any new furniture not indicated by a ghost.\n"
- DO NOT remove any furniture indicated by a ghost."
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
            [ghostly_collage_image, final_prompt]
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