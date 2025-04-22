from flask import Flask, request, jsonify
import os
import sys
import threading
import torch
from PIL import Image
import io
import numpy as np
import cv2

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
        
        # Try a test warmup inference to catch any issues
        dummy_img = torch.zeros((1, 3, 640, 640)).to(device)
        print("Running warmup inference...")
        _ = loaded_model(dummy_img)
        print("Warmup successful")
        
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
def load_model_on_startup():
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
        # Very simple approach - just get the image tensor
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Save original dimensions for scaling back the bounding boxes
        orig_width, orig_height = img.size
        
        # Resize the image to YOLOv5 standard input size
        img = img.resize((640, 640))
        
        # Convert to tensor
        img_tensor = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float()
        img_tensor = img_tensor.unsqueeze(0) / 255.0  # Add batch dimension and normalize
        
        # Process with the model
        with torch.no_grad():
            results = model(img_tensor)
        
        # Process the results
        results_json = []
        
        # Extract detections from output
        detections = results[0].cpu().numpy()  # Access first element and convert to numpy
        
        # Standard YOLOv5 output format: xywh, confidence, class
        if len(detections.shape) == 3:  # [batch, num_detections, detection_data]
            detections = detections[0]  # Get the first batch
        
        # Process each detection
        for detection in detections:
            # Standard YOLOv5 detection has 6 values: x, y, w, h, confidence, class
            if len(detection) < 6:
                continue
                
            x_center, y_center, width, height, confidence, class_id = detection[:6]
            
            # Convert normalized coordinates to pixel values for the original image
            x1 = (x_center - width/2) * orig_width / 640
            y1 = (y_center - height/2) * orig_height / 640
            x2 = (x_center + width/2) * orig_width / 640
            y2 = (y_center + height/2) * orig_height / 640
            
            # Get class name if available
            class_name = f"class_{int(class_id)}"
            if hasattr(model, 'names') and int(class_id) in model.names:
                class_name = model.names[int(class_id)]
            
            # Add detection to results
            results_json.append({
                'xmin': float(x1),
                'ymin': float(y1),
                'xmax': float(x2),
                'ymax': float(y2),
                'confidence': float(confidence),
                'class': int(class_id),
                'name': class_name
            })
        
        return jsonify({
            "status": "success",
            "message": "Image processed successfully",
            "detections": results_json,
            "count": len(results_json)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Start the model loading after app has been fully initialized
load_model_on_startup()

if __name__ == '__main__':
    # Model loading is already started at the top level
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port) 