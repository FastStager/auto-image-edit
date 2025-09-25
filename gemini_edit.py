import os
from io import BytesIO
from PIL import Image
from config import GENAI_AVAILABLE

if GENAI_AVAILABLE:
    import google.generativeai as genai


def run_enhanced_ai_edit(empty_room_image: Image.Image, furniture_assets_image: Image.Image, placement_map_image: Image.Image, user_prompt: str):
    if not GENAI_AVAILABLE:
        return None, "❌ Gemini library not found."

    api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        return None, "❌ GOOGLE_AI_STUDIO_API_KEY not found."

    base_prompt = (
        "You are an expert photo editor. Your task is to perform a high-fidelity 'in-painting' operation. You will be given three images:\n"
        "1.  **Background Image:** A clean, empty room.\n"
        "2.  **Furniture Assets:** An image containing one or more pieces of furniture on a transparent background. These are the assets to be placed.\n"
        "3.  **Placement Map:** An image of the same empty room, but with solid colored 'ghost' silhouettes drawn on it. This map shows the exact location, size, and orientation for each furniture asset.\n\n"
        "**Your instructions are simple and strict:**\n"
        "1.  **Identify:** For each colored silhouette in the 'Placement Map', find the corresponding furniture piece in the 'Furniture Assets' image.\n"
        "2.  **Replace and Render:** Your main task is to **replace** each colored silhouette with its corresponding furniture piece. The final image should look like the 'Background Image', but with the furniture from 'Furniture Assets' seamlessly rendered in the exact positions indicated by the silhouettes.\n"
        "3.  **Blend:** Ensure the lighting, shadows, perspective, and reflections on the placed furniture are perfectly matched to the room's environment to create a photorealistic result.\n\n"
        "**CRITICAL RULES:**\n"
        "- The final image MUST NOT contain any of the colored silhouettes. They are placeholders to be filled, not objects to be kept.\n"
        "- DO NOT add any new furniture or objects not provided in the 'Furniture Assets' image.\n"
        "- DO NOT alter the position, scale, or orientation from what is defined by the silhouettes in the 'Placement Map'."
    )

    final_prompt = (
        f"{base_prompt}\n\n"
        f"Finally, apply this user-specified style: \"{user_prompt}\""
        if user_prompt else base_prompt
    )
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-image-preview")

        # The order of images is critical for the logic of the prompt
        response = model.generate_content(
            [empty_room_image, furniture_assets_image, placement_map_image, final_prompt]
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