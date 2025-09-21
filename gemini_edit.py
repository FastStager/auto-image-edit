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

    text_prompt = f"""You are a world-class AI Photo Editor and VFX Compositor.

The provided image is a rough draft composition where furniture has been crudely cut out and placed into a room. The furniture contour/outline is visible and should be preserved. Your task is to re-render and integrate it realistically.

### Core Instructions
1. **Geometry Correction**
   - Redraw and re-render the furniture object(s) so they align perfectly with the room’s perspective, vanishing points, and floor plane.
   - Correct scale, angle, and proportions so all legs touch the ground naturally.
   - Remove skew, floating, or distorted geometry.

2. **Identity Preservation**
   - Keep the furniture’s essential identity, design, color, and texture.
   - Do not replace with a new model — only fix the existing one.

3. **Lighting & Integration**
   - Add physically accurate **contact shadows** where objects meet the floor.
   - Match ambient light and reflections to the room’s lighting conditions.
   - Remove cutout edges and blend seamlessly into the scene.

4. **Constraints**
   - Do **not** change the camera position, room architecture, or global composition.
   - Furniture should remain in the same 2D screen position.

### Final Creative Step
After geometry and lighting corrections, apply this user instruction: "{user_prompt}".

Execute with precision as if using a professional rendering engine.
"""

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-image-preview")

        response = model.generate_content(
            [composite_image, text_prompt]
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
