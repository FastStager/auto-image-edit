import os
from io import BytesIO
from PIL import Image
from config import GENAI_AVAILABLE

if GENAI_AVAILABLE:
    import google.generativeai as genai


def run_enhanced_ai_edit(empty_room_image: Image.Image, furniture_only_image: Image.Image, control_mask_image: Image.Image, user_prompt: str):
    if not GENAI_AVAILABLE:
        return None, "❌ Gemini library not found."

    api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        return None, "❌ GOOGLE_AI_STUDIO_API_KEY not found."

    base_prompt = (
        "You are a deterministic rendering engine. Your task is to composite furniture into a room with perfect precision, following a technical control mask. You will receive three images:\n"
        "1.  **Empty Room:** The background scene.\n"
        "2.  **Furniture Cutouts:** An image with one or more pieces of furniture on a transparent background. These are the only objects you are allowed to use.\n"
        "3.  **Control Mask:** A black image with colored rectangles and white lines. This is a non-photorealistic map that provides strict instructions.\n\n"
        "**EXECUTION STEPS (Follow Exactly):**\n"
        "1.  **Match:** For each piece of furniture from the 'Furniture Cutouts' image, find its corresponding colored rectangle in the 'Control Mask'.\n"
        "2.  **Place:** Position the furniture piece inside the 'Empty Room' so it perfectly occupies the area defined by its colored rectangle in the mask.\n"
        "3.  **Orient:** The white line inside each rectangle indicates the 'forward' direction. You MUST orient the furniture to match this line's direction.\n"
        "4.  **Render:** Blend the furniture into the scene with photorealistic lighting, shadows, and perspective. The furniture should look like it is naturally part of the room.\n\n"
        "**FAILURE CONDITIONS (Do NOT do these):**\n"
        "- **DO NOT ADD OBJECTS:** You are forbidden from adding any objects not present in the 'Furniture Cutouts' image.\n"
        "- **DO NOT DEVIATE:** The position, scale, and orientation provided by the Control Mask are absolute and must not be altered.\n"
        "- **CRITICAL:** The final output image MUST NOT contain any elements from the 'Control Mask'. No black background, no colored rectangles, no white lines. The appearance of any of these abstract control elements in the final photorealistic image is a complete failure."
    )

    final_prompt = (
        f"{base_prompt}\n\n"
        f"After adhering to all technical instructions, apply this final creative styling: \"{user_prompt}\""
        if user_prompt else base_prompt
    )
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-image-preview")

        response = model.generate_content(
            [empty_room_image, furniture_only_image, control_mask_image, final_prompt]
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