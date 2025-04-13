from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import torch
from pathlib import Path
from PIL import Image
import io
import os
import torchvision.transforms as T

app = FastAPI()

# Load YOLOv5 model (assumes best.pt is in the same directory)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        results = model(img)
        detections = results.pandas().xyxy[0].to_dict(orient="records")
        return JSONResponse(content={"results": detections})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def root():
    return {"message": "YOLOv5 Object Detection API is running."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
