from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import torch
from pathlib import Path
from PIL import Image
import io
import os
import torchvision.transforms as T

app = FastAPI()

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'best.pt')
    
    # Load YOLOv5 model
    print(f"Loading model from: {model_path}")
    from yolov5.models.common import DetectMultiBackend
    model = DetectMultiBackend(model_path, device=device)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")
        
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
        "model_loaded": model is not None,
        "device": str(device)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
