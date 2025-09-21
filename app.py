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
    print("Kaggle secrets not found. Assuming local environment.")

import config
from models import load_models
from image_processing import run_detection_and_populate_editor

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
    
    annotations = config.detection_results.get("staged_annotations", [])
    
    cutout_objects = []
    original_cutout_paths = []
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
        
        original_cutout_paths.append(cutout_filepath)
        
        cutout_objects.append({
            'url': f"/{UPLOAD_FOLDER}/{cutout_filename}",
            'bbox': annot['bbox'],
            'id': annot['id']
        })
    
    config.detection_results['original_cutouts'] = original_cutout_paths 

    return jsonify({
        'staged_image_url': resized_staged_url,
        'empty_image_url': empty_url,
        'objects': cutout_objects 
    })

@app.route('/prepare_ai', methods=['POST'])
def prepare_for_ai():
    data = request.json
    final_objects = data.get('objects', [])

    staged_img = config.detection_results.get("staged_image")
    if not staged_img:
        return jsonify({'error': 'Staged image not found. Run detection first.'}), 400
    
    staged_np = np.array(staged_img)
    transparent_bg = np.zeros((staged_np.shape[0], staged_np.shape[1], 4), dtype=np.uint8)

    final_annotations_for_gemini = []
    for obj_data in final_objects:
        contour = np.array(obj_data['contour']).astype(np.int32)
        
        mask = np.zeros(staged_np.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        object_pixels = np.where(mask[..., None] == 255, staged_np, 0)
        
        transparent_bg = np.where(mask[..., None] == 255, cv2.cvtColor(object_pixels, cv2.COLOR_RGB2RGBA), transparent_bg)

        final_annotations_for_gemini.append({
            'bbox': cv2.boundingRect(contour),
            'color': obj_data['color']
        })
        
    result_pil = Image.fromarray(transparent_bg)
    config.detection_results["staged_image_bg_removed"] = result_pil
    config.detection_results["empty_annotations"] = final_annotations_for_gemini

    return jsonify({'status': f'Prepared {len(final_annotations_for_gemini)} objects for AI.'})
@app.route('/run_ai', methods=['POST'])
def run_ai_edit_endpoint():
    if 'composite_image' not in request.files:
        return jsonify({'error': 'Missing composite image.'}), 400

    composite_img_file = request.files['composite_image']
    user_prompt = request.form.get('user_prompt', 'Fix lighting and shadows.')
    composite_pil = Image.open(composite_img_file.stream).convert("RGB")

    result_image, status_message = run_enhanced_ai_edit(composite_pil, user_prompt)

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