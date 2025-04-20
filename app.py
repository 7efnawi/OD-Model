from flask import Flask, request, jsonify
import os
import threading
import torch
from PIL import Image
import io

app = Flask(__name__)

# Global variables
model = None
model_loading = False
model_error = None

def load_model_in_background():
    global model, model_loading, model_error
    try:
        print("Starting to load YOLOv5 model...")
        # This import is slow, so it's in the function
        from yolov5.models.common import DetectMultiBackend
        
        # Get the current directory and model path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'best.pt')
        
        # Load the model (this is what takes time)
        device = torch.device('cpu')  # Always use CPU for Railway
        loaded_model = DetectMultiBackend(model_path, device=device)
        loaded_model.eval()
        
        # Only update the global model once fully loaded
        model = loaded_model
        print("Model loaded successfully!")
    except Exception as e:
        model_error = str(e)
        print(f"Error loading model: {str(e)}")
    finally:
        model_loading = False

# Start model loading in background when app starts
@app.before_first_request
def before_first_request():
    global model_loading
    if not model and not model_loading:
        model_loading = True
        thread = threading.Thread(target=load_model_in_background)
        thread.daemon = True
        thread.start()

@app.route('/')
def index():
    return jsonify({
        "message": "YOLOv5 Object Detection API",
        "model_status": "loading" if model_loading else "loaded" if model else "failed",
        "error": model_error
    })

@app.route('/health')
def health():
    # Always return OK for health checks
    return jsonify({"status": "ok"})

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        if model_loading:
            return jsonify({"error": "Model is still loading"}), 503
        else:
            return jsonify({"error": f"Model failed to load: {model_error}"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Read and process the image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Make prediction
        results = model(img)
        
        # Convert results to JSON format
        results_json = []
        for pred in results.xyxy[0]:  # xyxy format: x1, y1, x2, y2, confidence, class
            results_json.append({
                'xmin': float(pred[0]),
                'ymin': float(pred[1]),
                'xmax': float(pred[2]),
                'ymax': float(pred[3]),
                'confidence': float(pred[4]),
                'class': int(pred[5]),
                'name': results.names[int(pred[5])]
            })
        
        return jsonify({"results": results_json})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port) 