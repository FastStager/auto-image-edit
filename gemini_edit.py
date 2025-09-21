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

    text_prompt = f"""Analyze this composition. Identify the furniture object that is roughly placed in the room. 
Redraw and re-render the furniture so it perfectly matches the room’s perspective and floor plane. 
Adjust its scale, angle, and proportions for realism while preserving its identity, color, and texture. 
Integrate it into the room with correct contact shadows, reflections, and ambient light. 
Remove cutout edges and blend seamlessly. 
Do not change the camera position, room architecture, or global composition.

Apply this final user instruction: "{user_prompt}"
"""
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