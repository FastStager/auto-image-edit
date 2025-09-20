import os
from io import BytesIO
from PIL import Image
from config import detection_results, GENAI_AVAILABLE
from drawing import draw_circles_on_image
from PIL import Image, ImageFilter

if GENAI_AVAILABLE:
    import google.generativeai as genai

def run_enhanced_ai_edit(composite_image: Image.Image, inpainting_mask: Image.Image, user_prompt: str):
    if not GENAI_AVAILABLE:
        return None, "❌ Gemini library not found. Please install google-generativeai."

    api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        return None, "❌ GOOGLE_AI_STUDIO_API_KEY environment variable not found."
        
    if composite_image is None:
        return None, "❌ No composite image was provided to the AI."


    text_prompt = """You are a master image‐editor using Nano Banana (Gemini 2.5 Flash Image).  

CONTEXT:  
I have an image of an empty room, and a furniture cutout from a staged room pasted into it. The paste is imperfect: perspective, lighting, sometimes missing parts (e.g. legs), sharp edges, etc. A precise mask defines the furniture region.  

YOUR TASK:  
Repair only the masked furniture region. Use the cutout as the *true visual reference*—style, color, texture, design must remain exactly. Do **not** replace with a different piece.  

KEY INSTRUCTIONS:  
1. Preserve furniture identity: you must keep the same material, color, textures, shape, pattern, trim, etc.  
2. Correct perspective: align furniture’s angle with the room’s geometry—floor lines, wall vanishing points, camera viewpoint.  
3. Fix missing elements: reconstruct missing legs, supports, edges that are cut off, based on logical consistency with the rest of furniture.  
4. Soften/repair edges: smooth out rough or jagged mask edges, remove unnatural cutout outlines.  
5. Lighting & shadow matching: match light direction, intensity, color temperature, ambient shading of the room. Add realistic shadows where furniture meets floor or interacts with light sources.  
6. Maintain the rest of the image (outside the masked furniture region) exactly as is—no changes to walls, floors, room content, etc.  

ADDITIONAL STYLE ADJUSTMENTS (if user indicates):  
- (e.g. "make lighting warmer", "increase ambient contrast", etc.)  

OUTPUT:  
A seamless, photorealistic composite where the furniture looks as though it were originally part of the room—proper perspective, full form, correct lighting, perfect integration."""
    try:
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel("gemini-2.5-flash-image-preview")


        response = model.generate_content(
            [text_prompt, composite_image, inpainting_mask, user_prompt]
        )
        
        if not (response.candidates and response.candidates[0].content and response.candidates[0].content.parts):
            return None, f"❌ AI response is empty or malformed. Response: {response.text}"

        image_parts = [p.inline_data.data for p in response.candidates[0].content.parts if p.inline_data]

        if image_parts:
            image_bytes = image_parts[0]
            result_image = Image.open(BytesIO(image_bytes))
            detection_results["generated_images"].append(result_image)
            return result_image, "✅ Enhanced AI Edit successful!"
        else:
            refusal_text = response.text if hasattr(response, 'text') else str(response.prompt_feedback)
            return None, f"❌ AI did not return an image. Reason: {refusal_text}"
        
    except Exception as e:
        return None, f"❌ An error occurred with the AI Edit: {type(e).__name__}: {e}"

def use_generated_as_staged():
    if not detection_results["generated_images"]:
        return None, "❌ No generated images available."
    latest_generated = detection_results["generated_images"][-1]
    empty_img_size = detection_results.get("empty_image").size
    
    resized = latest_generated.resize(empty_img_size)
    detection_results.update({
        "staged_image": resized,
        "staged_annotations": [],
        "selected_staged": None
    })
    return resized, "✅ Generated image loaded as new staged image."