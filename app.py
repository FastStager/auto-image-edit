import os
import uuid
import math
from flask import Flask, render_template, request, jsonify, send_from_directory
from pyngrok import ngrok
import json
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import cv2

from gemini_edit import run_enhanced_ai_edit
from drawing import draw_circles_on_image
try:
    from kaggle_secrets import UserSecretsClient
    KAGGLE_ENV = True
except ImportError:
    KAGGLE_ENV = False
    print("Kagle secrets not found. Assuming local environment.")

import config
from models import load_models
from image_processing import run_detection_and_populate_editor, remove_background_and_add_border

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

print("Loading AI models, this might take a moment...")
hf_gd_processor, hf_gd_model, sam_predictor = load_models()
print("Models loaded successfully.")

def save_uploaded_file(file_storage):
    if not file_storage: return None, None
    filename = f"{uuid.uuid4()}_{os.path.basename(file_storage.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file_storage.save(filepath)
    return filepath, f"/{UPLOAD_FOLDER}/{filename}"

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html', genai_available=config.GENAI_AVAILABLE)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'empty_image' not in request.files or 'staged_image' not in request.files:
        return jsonify({'error': 'Missing images'}), 400

    empty_path, empty_url = save_uploaded_file(request.files['empty_image'])
    staged_path, _ = save_uploaded_file(request.files['staged_image'])
    prompts = request.form.get('prompts', 'furniture, object')

    if not empty_path or not staged_path:
        return jsonify({'error': 'Failed to save files'}), 500

    empty_img_pil = Image.open(empty_path)
    staged_img_pil = Image.open(staged_path).convert("RGB").resize(empty_img_pil.size)
    staged_np = np.array(staged_img_pil)

    resized_staged_filename = f"resized_{os.path.basename(staged_path)}"
    resized_staged_filepath = os.path.join(app.config['UPLOAD_FOLDER'], resized_staged_filename)
    staged_img_pil.save(resized_staged_filepath)
    resized_staged_url = f"/{UPLOAD_FOLDER}/{resized_staged_filename}"
    
    run_detection_and_populate_editor(
        empty_path, resized_staged_filepath, prompts,
        processor=hf_gd_processor, model=hf_gd_model, predictor=sam_predictor
    )
    
    remove_background_and_add_border()
    
    annotations = config.detection_results.get("staged_annotations", [])
    
    cutout_objects = []
    original_cutouts_by_id = {}
    for annot in annotations:
        mask = annot['mask']
        x, y, w, h = annot['bbox']
        
        cutout_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        region_pixels = staged_np[y:y+h, x:x+w]
        region_mask = mask[y:y+h, x:x+w]
        
        cutout_rgba[:, :, :3] = region_pixels
        cutout_rgba[:, :, 3] = region_mask * 255
        
        cutout_pil = Image.fromarray(cutout_rgba, 'RGBA')
        cutout_filename = f"cutout_{annot['id']}.png"
        cutout_filepath = os.path.join(app.config['UPLOAD_FOLDER'], cutout_filename)
        cutout_pil.save(cutout_filepath)
        
        original_cutouts_by_id[annot['id']] = cutout_filepath
        
        cutout_objects.append({
            'url': f"/{UPLOAD_FOLDER}/{cutout_filename}",
            'bbox': annot['bbox'],
            'id': annot['id']
        })
    
    config.detection_results['original_cutouts_by_id'] = original_cutouts_by_id

    return jsonify({
        'staged_image_url': resized_staged_url,
        'empty_image_url': empty_url,
        'objects': cutout_objects 
    })

@app.route('/run_ai', methods=['POST'])
def run_ai_edit_endpoint():
    data = request.json
    if not data or 'objects' not in data:
        return jsonify({'error': 'Missing object data in request.'}), 400

    final_objects = data.get('objects', [])
    user_prompt = data.get('user_prompt', 'Make the final image photorealistic.')

    empty_room_img = config.detection_results.get("empty_image")
    if not empty_room_img:
        return jsonify({'error': 'Base images not found. Please run detection first.'}), 400

    original_cutouts_by_id = config.detection_results.get("original_cutouts_by_id", {})
    if not original_cutouts_by_id:
        return jsonify({'error': 'Cutout object data not found. Please run detection first.'}), 400
        
    # The Blueprint: Empty room + vector instructions
    blueprint_img = empty_room_img.copy()
    draw = ImageDraw.Draw(blueprint_img, "RGBA")

    # The Palette: Clean assets on a transparent background
    furniture_palette_img = Image.new('RGBA', empty_room_img.size, (0, 0, 0, 0))

    canvas_width = 800 
    scale_factor = empty_room_img.width / canvas_width
    
    original_annots_by_id = {
        annot['id']: annot for annot in config.detection_results.get("staged_annotations", [])
    }

    # We need to place assets on the palette without overlap so the AI sees them clearly
    palette_x_offset, palette_y_offset = 10, 10

    for obj_data in final_objects:
        obj_id = obj_data.get('id')
        if obj_id is None: continue
        
        original_annot = original_annots_by_id.get(obj_id)
        cutout_path = original_cutouts_by_id.get(obj_id)
        if not cutout_path or not original_annot: continue
        
        try:
            original_piece = Image.open(cutout_path).convert("RGBA")
        except FileNotFoundError:
            continue
            
        unscaled_left = int(obj_data['left'] * scale_factor)
        unscaled_top = int(obj_data['top'] * scale_factor)
        unscaled_width = int(obj_data['width'] * scale_factor)
        unscaled_height = int(obj_data['height'] * scale_factor)
        
        if unscaled_width <= 0 or unscaled_height <= 0: continue

        # Draw Blueprint Instructions
        color = tuple(original_annot['color'])
        # Use a semi-transparent fill for the rectangle
        rect_fill_color = color + (100,) 
        rect_coords = (unscaled_left, unscaled_top, unscaled_left + unscaled_width, unscaled_top + unscaled_height)
        draw.rectangle(rect_coords, fill=rect_fill_color, outline=color, width=3)
        
        # Draw orientation line
        center_x = unscaled_left + unscaled_width / 2
        center_y = unscaled_top + unscaled_height / 2
        angle_deg = obj_data.get('angle', 0)
        angle_rad = math.radians(angle_deg)
        line_length = (unscaled_width / 2) * 0.8
        end_x = center_x + line_length * math.cos(angle_rad)
        end_y = center_y + line_length * math.sin(angle_rad)
        draw.line([(center_x, center_y), (end_x, end_y)], fill="white", width=5)

        # Add clean asset to the palette
        furniture_palette_img.paste(original_piece, (palette_x_offset, palette_y_offset), original_piece)
        palette_y_offset += original_piece.height + 10


    result_image, status_message = run_enhanced_ai_edit(
        blueprint_img,
        furniture_palette_img,
        user_prompt
    )

    if result_image is None:
        return jsonify({'error': status_message}), 500

    result_filename = f"result_{uuid.uuid4()}.png"
    result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    result_image.save(result_filepath)
    result_url = f"/{UPLOAD_FOLDER}/{result_filename}"

    return jsonify({'result_image_url': result_url, 'status': status_message})


if __name__ == '__main__':
    port = 7860
    try:
        if KAGGLE_ENV:
            user_secrets = UserSecretsClient()
            NGROK_TOKEN = user_secrets.get_secret("NGROK_AUTHTOKEN")
        else:
            NGROK_TOKEN = os.environ.get("NGROK_AUTHTOKEN")
            if NGROK_TOKEN is None: raise ValueError("NGROK_AUTHTOKEN not found")
        ngrok.set_auth_token(NGROK_TOKEN)
    except Exception as e:
        print(f"🚨 ERROR: Could not authenticate ngrok: {e}")
        exit()
    for tunnel in ngrok.get_tunnels(): ngrok.disconnect(tunnel.public_url)
    public_url = ngrok.connect(port).public_url
    print(f"\n✅ Your app is live at: {public_url}\n")
    app.run(port=port)