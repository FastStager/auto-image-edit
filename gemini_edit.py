import os
from io import BytesIO
from PIL import Image
from config import GENAI_AVAILABLE

if GENAI_AVAILABLE:
    import google.generativeai as genai

def run_enhanced_ai_edit(reference_sheet: Image.Image, marker_image: Image.Image, user_prompt: str):
    if not GENAI_AVAILABLE:
        return None, "❌ Gemini library not found."
    api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        return None, "❌ GOOGLE_AI_STUDIO_API_KEY not found."

    text_prompt = """
    You are an expert virtual staging AI. Your task is to perform a high-fidelity furniture placement and integration task. You will be given TWO images and ONE text prompt.

    **INPUTS:**
    1.  **[Reference Sheet]:** An image containing one or more clean furniture cutouts on a transparent background. This is your "asset library."
    2.  **[Marker Image]:** An image of an empty room with one or more colored circles. These circles are precise targets, defining the desired location, scale, and position for the furniture.
    3.  **[User Prompt]:** A text instruction for final lighting and mood adjustments.

    **YOUR MANDATORY STEP-BY-STEP WORKFLOW:**

    1.  **EXTRACT FURNITURE:** Isolate each individual piece of furniture from the [Reference Sheet].
    2.  **MATCH AND PLACE:** For each colored circle in the [Marker Image], find the corresponding furniture item in the [Reference Sheet] and place it at that exact location.
    3.  **PERSPECTIVE ALIGNMENT:** This is the most critical step. You MUST realistically render the furniture to match the empty room's camera angle, perspective, and geometry. The furniture must look like it is physically sitting on the floor in that spot, not like a flat image pasted on top.
    4.  **REMOVE MARKERS:** After placing the furniture, you must completely remove the colored circles from the final image using a seamless, content-aware fill. There should be no trace of them.
    5.  **LIGHTING AND SHADOWS:** Re-light each piece of furniture to perfectly match the lighting sources (direction, color, softness) of the empty room. Generate realistic soft shadows and contact shadows.
    6.  **APPLY USER PROMPT:** Use the user's text prompt for final adjustments to lighting or mood.

    **FINAL OUTPUT:** The final image must be a single, photorealistic composite. It must not contain any circles. The background must come ONLY from the [Marker Image].
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-image-preview")

        response = model.generate_content(
            [text_prompt, reference_sheet, marker_image, user_prompt]
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