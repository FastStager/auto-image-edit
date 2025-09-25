import os
from io import BytesIO
from PIL import Image
from config import GENAI_AVAILABLE

if GENAI_AVAILABLE:
    import google.generativeai as genai


def run_enhanced_ai_edit(empty_room_with_circles: Image.Image, furniture_image: Image.Image, user_prompt: str):
    if not GENAI_AVAILABLE:
        return None, "❌ Gemini library not found."

    api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        return None, "❌ GOOGLE_AI_STUDIO_API_KEY not found."

    base_prompt = (
        "Your task is to place furniture from the second image (the 'furniture image') into the first image (the 'empty room') with extreme precision. "
        "Follow these steps strictly:\n"
        "1. For each piece of furniture provided, locate its corresponding colored circle in the empty room.\n"
        "2. Place the furniture so that its center is **exactly** on the center of its corresponding colored circle. The position of the colored circle is an absolute instruction and must not be altered or ignored.\n"
        "3. Adjust the scale and perspective of the furniture piece to look realistic *in that specific location*, matching the room's perspective, but do not change its position.\n"
        "4. After positioning all furniture, completely remove all colored circles from the final image. There should be no trace of them.\n"
        "5. Re-render the furniture with perfectly integrated lighting, shadows, and reflections to create a seamless, photorealistic final image."
    )

    final_prompt = (
        f"{base_prompt}\n\nApply this final user instruction: "
        f"\"{user_prompt}. Ensure the final result is hyper-realistic and perfectly blended.\""
        if user_prompt else base_prompt
    )
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-image-preview")

        response = model.generate_content(
            [empty_room_with_circles, furniture_image, final_prompt]
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