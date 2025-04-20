from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import torch
from pathlib import Path
from PIL import Image
import io
import os
import torchvision.transforms as T
import threading

app = FastAPI()

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables
model = None
model_loading = False
model_error = None

def load_model_in_background():
    global model, model_loading, model_error
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'best.pt')
        
        print(f"Loading model from: {model_path}")
        from yolov5.models.common import DetectMultiBackend
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

@app.on_event("startup")
async def startup_event():
    global model_loading
    model_loading = True
    # Start loading model in a background thread
    thread = threading.Thread(target=load_model_in_background)
    thread.daemon = True
    thread.start()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        if model_loading:
            return JSONResponse(status_code=503, content={
                "error": "Model is still loading. Please try again in a moment."
            })
        else:
            return JSONResponse(status_code=500, content={
                "error": f"Model failed to load: {model_error or 'Unknown error'}"
            })
        
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
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
            
        return JSONResponse(content={"results": results_json})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def root():
    return {
        "message": "YOLOv5 Object Detection API is running.",
        "model_status": "loading" if model_loading else "loaded" if model is not None else "failed",
        "device": str(device),
        "model_error": model_error
    }

@app.get("/health")
def health():
    # Always respond with 200 OK for Railway health checks
    return {"status": "ok"}

if __name__ == "__main__":
    # Use MAIN_PORT if set by health_check.py, otherwise fallback to PORT or 8000
    port = int(os.environ.get("MAIN_PORT", os.environ.get("PORT", "8000")))
    print(f"Starting main application on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
