import os
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory
from pyngrok import ngrok
import json
from PIL import Image, ImageDraw
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
    user_prompt = data.get('user_prompt', 'Ensure lighting is realistic.')

    empty_room_img = config.detection_results.get("empty_image")
    if not empty_room_img:
        return jsonify({'error': 'Base images not found. Please run detection first.'}), 400

    original_cutouts_by_id = config.detection_results.get("original_cutouts_by_id", {})
    if not original_cutouts_by_id:
        return jsonify({'error': 'Cutout object data not found. Please run detection first.'}), 400
        
    user_arranged_furniture_img = Image.new('RGBA', empty_room_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(user_arranged_furniture_img)
    canvas_width = 800 
    scale_factor = canvas_width / empty_room_img.width
    
    original_annots_by_id = {
        annot['id']: annot for annot in config.detection_results.get("staged_annotations", [])
    }

    for obj_data in final_objects:
        obj_id = obj_data.get('id')
        if obj_id is None: continue
        
        original_annot = original_annots_by_id.get(obj_id)
        cutout_path = original_cutouts_by_id.get(obj_id)
        if not cutout_path or not original_annot: continue
        
        try:
            furniture_piece_img = Image.open(cutout_path).convert("RGBA")
        except FileNotFoundError:
            continue
            
        unscaled_width = int(obj_data['width'] / scale_factor)
        unscaled_height = int(obj_data['height'] / scale_factor)
        
        if unscaled_width <= 0 or unscaled_height <= 0: continue

        piece = furniture_piece_img.resize((unscaled_width, unscaled_height), Image.Resampling.LANCZOS)

        if obj_data.get('flipX'):
            piece = piece.transpose(Image.FLIP_LEFT_RIGHT)
        
        angle = obj_data.get('angle', 0)
        if angle != 0:
            piece = piece.rotate(-angle, expand=True, resample=Image.BICUBIC)

        unscaled_left = int(obj_data['left'] / scale_factor)
        unscaled_top = int(obj_data['top'] / scale_factor)
        
        center_x = unscaled_left + unscaled_width / 2
        center_y = unscaled_top + unscaled_height / 2
        
        paste_x = int(center_x - piece.width / 2)
        paste_y = int(center_y - piece.height / 2)
        
        # Draw colored disk on the floor first
        disk_radius = 30
        disk_center_x = paste_x + piece.width // 2
        disk_center_y = paste_y + piece.height - (disk_radius // 2) # Place it near the bottom
        color = tuple(original_annot['color'])
        draw.ellipse(
            [disk_center_x - disk_radius, disk_center_y - disk_radius, disk_center_x + disk_radius, disk_center_y + disk_radius],
            fill=color
        )
        
        # Paste furniture on top of the disk
        user_arranged_furniture_img.paste(piece, (paste_x, paste_y), piece)

    
    result_image, status_message = run_enhanced_ai_edit(
        empty_room_img,
        user_arranged_furniture_img,
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