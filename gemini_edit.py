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
    text_prompt = f"""
**ROLE:** You are a world-class AI Photo Editor and VFX Compositor.

**CONTEXT:** The image I am providing is a "rough draft" composition. It contains furniture that has been crudely cut out and placed into a room. This is NOT a finished image. It is flawed and needs to be fixed.

**PRIMARY OBJECTIVE:** Your main task is to identify the flawed furniture and perform a complete generative re-rendering of it in place.

**YOUR MANDATORY STEP-BY-STEP WORKFLOW:**
1.  **Analyze and Identify:** Analyze the entire composition. Identify the furniture object(s) that are roughly placed in the room.
2.  **Redraw and Re-render (The Core Task):** You MUST redraw and re-render the furniture. This is a generative action, not a simple blend. Your re-rendering must:
    -   Perfectly match the room’s perspective, floor plane, and vanishing points.
    -   Preserve the essential identity, color, and texture of the original furniture.
    -   Adjust the scale, angle, and proportions for absolute realism. A leg not touching the ground is an unacceptable failure.
3.  **Integrate with Scene:** After re-rendering, you must perfectly integrate the new furniture into the room with:
    -   Physically accurate contact shadows, reflections, and ambient light.
    -   Seamless edges with no cutout artifacts.
4.  **Apply User Direction:** After all technical corrections are complete, apply this final creative instruction from the user: "{user_prompt}"

**ABSOLUTE CONSTRAINTS (DO NOT VIOLATE):**
-   **AVOID LAZY BLENDING:** You are strictly forbidden from simply blurring the edges, adding a drop shadow, or making minor adjustments. This is a generative re-rendering task.
-   **PRESERVE COMPOSITION:** Do not change the camera position, room architecture, or the global composition. The furniture's final 2D screen position must be the same as in the rough draft.
-   **AVOID REPLACEMENT:** Do not replace the furniture with a different model. Re-render the *same* piece of furniture correctly.

Execute this with the precision of a high-end rendering engine.
"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash-image-preview")

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