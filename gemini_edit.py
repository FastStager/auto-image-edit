import os
from io import BytesIO
from PIL import Image
from config import GENAI_AVAILABLE

if GENAI_AVAILABLE:
    import google.generativeai as genai


def run_enhanced_ai_edit(empty_room_image: Image.Image, furniture_only_image: Image.Image, guidance_map_image: Image.Image, user_prompt: str):
    if not GENAI_AVAILABLE:
        return None, "❌ Gemini library not found."

    api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        return None, "❌ GOOGLE_AI_STUDIO_API_KEY not found."

    base_prompt = (
        "You are a photorealistic image editor. You will receive three input images:\n"
        "1.  **Empty Room Image:** This is the background where furniture needs to be placed.\n"
        "2.  **Furniture Only Image:** This image contains transparent cutouts of all furniture pieces. These pieces are already scaled and rotated by the user to their desired individual orientation. Do NOT change their internal rotation or scaling.\n"
        "3.  **Guidance Map Image:** This image contains 'floor compasses'. Each compass is a colored disk with a white pointer line. Each compass corresponds to a piece of furniture in the 'Furniture Only Image' (implied by proximity and color, though strict matching by color is not needed; simply use each compass for a furniture piece).\n\n"
        "**Your instructions are extremely strict and must be followed with pixel-perfect precision:**\n"
        "1.  **TRANSFER & POSITION:** For each furniture cutout in the 'Furniture Only Image', take it and place it into the 'Empty Room Image'. The colored disk in the 'Guidance Map Image' indicates the **exact floor position** where the center of the furniture's base should be placed.\n"
        "2.  **ORIENT:** The white pointer line extending from the colored disk in the 'Guidance Map Image' shows the **exact forward-facing direction** for the furniture. You MUST orient the furniture (as it appears in the 'Furniture Only Image', already rotated by the user) such that its 'front' aligns perfectly with this white pointer direction on the floor.\n"
        "3.  **RENDERING:** Once positioned and oriented, integrate the furniture into the 'Empty Room Image'. Render it with realistic scale, perspective, lighting, and shadows to ensure it blends seamlessly and photorealistically into the scene. Make sure the furniture appears to be a natural part of the room.\n"
        "4.  **CLEANUP:** After all furniture is placed and rendered, you MUST completely remove ALL elements from the 'Guidance Map Image' (every colored disk and white pointer line) from the final output. There must be absolutely no trace of them.\n\n"
        "**CRITICAL PROHIBITIONS (DO NOT DO ANY OF THESE):**\n"
        "- **DO NOT ADD NEW OBJECTS:** You are strictly forbidden from adding any furniture, decorations, or any other objects not explicitly provided in the 'Furniture Only Image'.\n"
        "- **DO NOT MOVE OR REARRANGE:** You MUST NOT move any furniture from the exact position indicated by its corresponding colored disk in the 'Guidance Map Image'. Do NOT 'improve' the layout or try to act as an interior designer.\n"
        "- **DO NOT ALTER ORIENTATION:** You MUST NOT change the floor-level orientation from that indicated by the white pointer line in the 'Guidance Map Image'.\n"
        "- **DO NOT GENERATE ANY OTHER SCENE:** Focus solely on compositing and rendering the provided furniture in the specified way."
    )

    final_prompt = (
        f"{base_prompt}\n\n"
        f"After following all the above rules and prohibitions, apply this final user styling instruction: \"{user_prompt}. Ensure the final result is hyper-realistic and perfectly blended, as if it were a real photograph.\""
        if user_prompt else f"{base_prompt}\n\nEnsure the final result is hyper-realistic and perfectly blended, as if it were a real photograph."
    )
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-image-preview")

        response = model.generate_content(
            [empty_room_image, furniture_only_image, guidance_map_image, final_prompt]
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