import os
from io import BytesIO
from PIL import Image
from config import GENAI_AVAILABLE

if GENAI_AVAILABLE:
    import google.generativeai as genai

def run_enhanced_ai_edit(composite_image: Image.Image, user_prompt: str):
    if not GENAI_AVAILABLE:
        return None, "❌ Gemini library not found."
    api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        return None, "❌ GOOGLE_AI_STUDIO_API_KEY not found."

    text_prompt = f"""You are an expert AI photo editor and VFX compositor.
I am providing you with a single image that is a crude 'rough draft' collage. Furniture has been digitally placed into a room, but it is not integrated.

Your task is to transform this rough draft into a FLAWLESS, PHOTOREALISTIC final image.

**USER'S CREATIVE DIRECTION:** "{user_prompt}"

**YOUR MANDATORY TECHNICAL EXECUTION:**

1.  **PRESERVE THE CORE COMPOSITION:** The specific furniture models, their 2D screen positions, and their scale in the rough draft are INTENTIONAL. You are strictly forbidden from moving, resizing, or replacing the furniture with a different style.

2.  **PERFORM GENERATIVE INTEGRATION:** Your primary job is to completely re-render the furniture in place to make it look real. This is a generative task, not a simple blend. You MUST:
    -   **Fix 3D Perspective:** Analyze the room's geometry and re-render the furniture with the correct perspective and angle, so it sits naturally on the floor. If a leg is not touching the ground, you must fix it.
    -   **Fix Lighting:** Analyze the room's light sources and completely re-light the furniture to match.
    -   **Fix Shadows:** Generate physically-accurate shadows on the floor and walls.

The final output must be a single, seamless photograph that honors the user's composition and prompt."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-image-preview")

        response = model.generate_content(
            [composite_image, text_prompt]
        )
        
        if not (response.candidates and response.candidates[0].content and response.candidates[0].content.parts):
            return None, f"❌ AI response is empty or malformed."
        image_parts = [p.inline_data.data for p in response.candidates[0].content.parts if p.inline_data]
        if image_parts:
            result_image = Image.open(BytesIO(image_parts[0]))
            return result_image, "✅ Enhanced AI Edit successful!"
        else:
            refusal_text = response.text if hasattr(response, 'text') else str(response.prompt_feedback)
            return None, f"❌ AI did not return an image. Reason: {refusal_text}"
        
    except Exception as e:
        return None, f"❌ An error occurred with the AI Edit: {type(e).__name__}: {e}"