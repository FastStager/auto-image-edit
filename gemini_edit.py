import os
from io import BytesIO
from PIL import Image
from config import GENAI_AVAILABLE

if GENAI_AVAILABLE:
    import google.generativeai as genai

def run_enhanced_ai_edit(composite_image: Image.Image, marker_image: Image.Image, user_prompt: str):
    if not GENAI_AVAILABLE:
        return None, "❌ Gemini library not found."
    api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        return None, "❌ GOOGLE_AI_STUDIO_API_KEY not found."

    text_prompt = """
    You are a high-end VFX Compositor and Technical Director AI. Your task is to perform a photorealistic integration by interpreting a set of visual instructions with absolute precision.

    You will receive TWO primary images and ONE text prompt:
    1.  **[Composite Image]:** This is your GROUND TRUTH. It shows the final scene with the correct furniture assets already placed at their final 2D screen positions and scales. The design, color, and texture of the furniture in this image are non-negotiable and MUST be preserved.
    2.  **[Marker Image]:** This is your TECHNICAL BLUEPRINT. It shows the empty room with colored circles. These circles are 3D grounding targets for the furniture seen in the [Composite Image].
    3.  **[User Prompt]:** A final instruction for lighting and mood.

    **YOUR MANDATORY, UNBREAKABLE WORKFLOW:**

    1.  **ANALYZE THE SCENE:** From the [Composite Image], analyze the empty room's geometry, perspective, and lighting.
    2.  **CORRELATE ASSETS:** For each piece of furniture in the [Composite Image], find its corresponding colored circle in the [Marker Image].
    3.  **PERFORM 3D GROUNDING:** This is your primary task. You must take the furniture asset from the [Composite Image] and re-render it so that it becomes a believable 3D object physically grounded at the location of its corresponding circle. This is not a 2D paste. You MUST:
        -   Fix the perspective to match the room.
        -   Ensure all legs or bases are firmly touching the floor plane.
        -   Correct any awkward angles from the original cutout.
    4.  **INTEGRATE REALISTICALLY:** Re-light the newly grounded furniture to match the room's light sources. Generate physically-accurate shadows.
    5.  **REMOVE MARKERS:** The circles in the [Marker Image] are a guide for you. They must be completely absent from the final output.

    **CRITICAL CONSTRAINTS:**
    -   **IDENTITY IS SACRED:** You are strictly forbidden from changing the furniture's design, style, color, or texture. Use the pixels from the [Composite Image] as your absolute reference.
    -   **2D PLACEMENT IS LOCKED:** The final 2D position and scale on the screen must match the [Composite Image]. Your job is to fix the 3D form *within* that 2D boundary.

    Execute this with the precision of a senior VFX artist. The final image must be a flawless, photorealistic photograph.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-image-preview")


        response = model.generate_content(
            [text_prompt, composite_image, marker_image, user_prompt]
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