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
        "You are a photorealistic integration specialist. You will receive a single image that is a crude collage. "
        "This collage contains the **exact assets** to be used and their **final, correct composition.**\n\n"
        "**Your task is to make this collage photorealistic by treating it as a final render that needs post-processing.**\n\n"
        "**Instructions:**\n"
        "1.  **Adjust Perspective & Orientation:** Subtly modify each object's 3D angle and perspective so it sits naturally in the room's 3D space.\n"
        "2.  **Integrate Lighting & Shadows:** Re-light each object to perfectly match the room's light sources. Add physically accurate shadows and reflections.\n"
        "3.  **Blend Edges:** Seamlessly blend the objects into the scene.\n\n"
        "**CRITICAL RULES (FAILURE IF NOT FOLLOWED):**\n"
        "1.  **PRESERVE FURNITURE IDENTITY:** You MUST use the *exact* furniture provided. **DO NOT change its color, style, fabric, or design.** If a blue armchair is provided, the final image must have that *exact same blue armchair*, not a different one.\n"
        "2.  **DO NOT MOVE** the furniture. The floor position and layering are final.\n"
        "3.  **DO NOT ADD OR REMOVE** any objects."
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