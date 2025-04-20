from flask import Flask, request, jsonify
import os
import sys
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
        
        # Get the current directory and set up paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set up Python paths for YOLOv5
        yolov5_dir = os.path.join(current_dir, 'yolov5')
        
        # Add all necessary directories to sys.path
        for path in [current_dir, yolov5_dir, os.path.join(yolov5_dir, 'models'), os.path.join(yolov5_dir, 'utils')]:
            if path not in sys.path:
                sys.path.insert(0, path)
                print(f"Added {path} to Python path")
        
        # Verify paths
        print(f"Current working directory: {os.getcwd()}")
        print(f"sys.path: {sys.path}")
        
        # Create a simple models/__init__.py file if it doesn't exist
        models_init = os.path.join(yolov5_dir, 'models', '__init__.py')
        if not os.path.exists(models_init):
            os.makedirs(os.path.dirname(models_init), exist_ok=True)
            with open(models_init, 'w') as f:
                f.write('# Initialize models package\n')
            print(f"Created {models_init}")
        
        # Make sure utils has __init__.py
        utils_init = os.path.join(yolov5_dir, 'utils', '__init__.py')
        if not os.path.exists(utils_init):
            os.makedirs(os.path.dirname(utils_init), exist_ok=True)
            with open(utils_init, 'w') as f:
                f.write('# Initialize utils package\n')
            print(f"Created {utils_init}")
        
        try:
            # Try importing directly
            print("Trying import path 1...")
            from yolov5.models.common import DetectMultiBackend
            print("Successfully imported via yolov5.models.common")
        except ImportError as e1:
            print(f"Import path 1 failed: {e1}")
            try:
                # Try alternative import
                print("Trying import path 2...")
                from models.common import DetectMultiBackend
                print("Successfully imported via models.common")
            except ImportError as e2:
                print(f"Import path 2 failed: {e2}")
                # Last resort, try modifying sys.modules
                print("Trying import path 3...")
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "common", os.path.join(yolov5_dir, 'models', 'common.py'))
                common = importlib.util.module_from_spec(spec)
                sys.modules["common"] = common
                spec.loader.exec_module(common)
                DetectMultiBackend = common.DetectMultiBackend
                print("Successfully imported via direct file loading")
        
        # Load the model
        model_path = os.path.join(current_dir, 'best.pt')
        print(f"Loading model from: {model_path}")
        
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
        # Print the traceback for debugging
        import traceback
        traceback.print_exc()
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