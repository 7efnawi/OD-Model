# 🧠 YOLOv5 Object Detection on Custom COCO Dataset

This project is a full pipeline for training a YOLOv5 model on a custom subset of the COCO dataset. It includes data preparation, model training, evaluation, and deployment instructions.

---

## 📂 Project Structure

```
project/
│
├── data/                         # Custom YAML config files
│   └── MY_coco_yolov5.yaml       # Training configuration
│
├── models/                       # YOLOv5 model architecture
│
├── runs/                         # Training runs, weights, and logs
│
├── utils/                        # Utility scripts for training
│
├── yolov5s.pt                    # Pre-trained weights
├── train.py                      # Main training script
├── detect.py                     # Inference script
├── requirements.txt              # Python dependencies
├── README.md                     # You're here!
```

---

## ✅ Model Information

- **Base Model**: YOLOv5s
- **Image Size**: 640x640
- **Classes**: 80 (COCO format)
- **Initial Dataset Size**: 30% of COCO
- **Final Dataset Size**: ~60% of COCO (balanced across classes)
- **Training Epochs**: 50 + Continued Training
- **Batch Size**: 16 (then reduced to 4 due to resource limits)
- **Device**: Trained on RTX 3050 (6GB)
- **Metrics Achieved**:
  - `mAP@0.5`: ~0.54+
  - `Precision`: ~0.63
  - `Recall`: ~0.47+

---

## 🛠️ Installation

```bash
# Clone YOLOv5 repo
git clone https://github.com/ultralytics/yolov5.git
cd yolov5

# Install dependencies
pip install -r requirements.txt

# Optional: Install TensorBoard
pip install tensorboard==2.10.1 numpy==1.23.5
```

---

## 🏋️‍♂️ Training

To train using 60% of the dataset:

```bash
python train.py --img 640 --batch 4 --epochs 50 \
  --data data/MY_coco_yolov5.yaml \
  --weights yolov5s.pt --device 0
```

To resume training from best checkpoint:

```bash
python train.py --weights runs/train/exp6/weights/best.pt \
  --data data/MY_coco_yolov5.yaml --device 0
```

---

## 🔍 Inference

Test the model on an image:

```bash
python detect.py --weights runs/train/exp6/weights/best.pt \
  --source path/to/image.jpg --conf 0.4 --device 0
```

Test on webcam:

```bash
python detect.py --weights runs/train/exp6/weights/best.pt \
  --source 0 --conf 0.4 --device 0
```

---

## 📊 TensorBoard

To monitor training:

```bash
tensorboard --logdir runs/train
```

Then open [http://localhost:6006](http://localhost:6006) in your browser.

---

## 🚄 Deployment on Railway

This project provides a REST API for object detection using YOLOv5, with easy deployment on Railway.

### Features

- REST API built with Flask
- Object detection using pre-trained YOLOv5 model
- Background model loading allowing the server to respond immediately
- Full compatibility with Railway deployment platform

### Requirements

- Python 3.9 or newer
- Dependencies listed in `requirements.txt`

### Local Installation

1. Clone the repository:

```bash
git clone https://github.com/7efnawi/OD-Model.git
cd OD-Model
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python app.py
```

4. The application will be available at `http://localhost:8000`

### Deployment on Railway

1. Create an account on Railway.app
2. Create a new project and link it to your GitHub repository
3. Railway will automatically detect the presence of a Dockerfile and use it for building and deployment

### API Endpoints

#### 1. Main Endpoint

- **URL**: `/`
- **Method**: GET
- **Response**: Model status and application information

```json
{
  "message": "YOLOv5 Object Detection API",
  "model_status": "loaded",
  "error": null
}
```

#### 2. Health Check

- **URL**: `/health`
- **Method**: GET
- **Response**: Application status

```json
{
  "status": "ok"
}
```

#### 3. Object Detection

- **URL**: `/predict`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Data**:
  - `file`: Image file to be analyzed
- **Response**: Object detection results

```json
{
  "results": [
    {
      "xmin": 120.5,
      "ymin": 220.3,
      "xmax": 250.8,
      "ymax": 380.1,
      "confidence": 0.85,
      "class": 0,
      "name": "person"
    }
  ]
}
```

## 💡 Technical Challenges and Solutions

We faced several challenges during the deployment of this project on Railway. Here's a record of the problems and how we solved them:

### 1. "Application failed to respond" Issue

**Problem**: After deploying the application on Railway, health checks failed because the model takes a long time to load.

**Solution**: Implemented background model loading using threading, allowing the server to respond to health checks immediately while the model loads.

```python
@app.before_first_request
def before_first_request():
    global model_loading
    if not model and not model_loading:
        model_loading = True
        thread = threading.Thread(target=load_model_in_background)
        thread.daemon = True
        thread.start()
```

### 2. "The executable `uvicorn` could not be found" Issue

**Problem**: Encountered an error when trying to run FastAPI using uvicorn.

**Solution**:

1. Switched from FastAPI to Flask to simplify the process
2. Added uvicorn to requirements.txt as a backup
3. Modified Dockerfile and railway.json file

### 3. "No module named 'utils'" Issue

**Problem**: Encountered an issue importing the utils module from YOLOv5.

**Solution**:

1. Added all necessary paths to `sys.path`
2. Ensured the presence of `__init__.py` files in relevant directories
3. Implemented a multi-stage import strategy with fallback mechanisms

```python
for path in [current_dir, yolov5_dir, os.path.join(yolov5_dir, 'models'), os.path.join(yolov5_dir, 'utils')]:
    if path not in sys.path:
        sys.path.insert(0, path)
```

### 4. Docker Environment Variables Issue

**Problem**: Failure in interpreting environment variables in Dockerfile.

**Solution**:

1. Simplified Dockerfile and removed CMD commands
2. Added `startCommand` to railway.json file
3. Used Python directly instead of a command-line tool

### 5. Final Integration Issues

**Problem**: Integrating all previous solutions.

**Solution**:

1. Unified all solutions into a simple Flask application
2. Used background model loading mechanism
3. Properly configured health check endpoints
4. Updated configuration files for Railway

### Other Known Issues & Fixes

- **OMP Error**: Set the following before running training

  ```bash
  set KMP_DUPLICATE_LIB_OK=TRUE
  ```

- **Numpy bool8 Error**: Downgrade numpy to a stable version

  ```bash
  pip install numpy==1.23.5
  ```

- **TensorBoard compatibility**: Downgrade to `tensorboard==2.10.1` to match TensorFlow-GPU 2.10

## 🖥️ Usage Examples

### Using cURL

```bash
curl -X POST -F "file=@path/to/image.jpg" https://od-model-production.up.railway.app/predict
```

### Using Python

```python
import requests

url = "https://od-model-production.up.railway.app/predict"
image_path = "path/to/image.jpg"

with open(image_path, "rb") as image_file:
    files = {"file": image_file}
    response = requests.post(url, files=files)

print(response.json())
```

## 📋 Tips for Railway Deployment

1. Ensure a simple and straightforward Dockerfile
2. Use railway.json file to provide deployment settings
3. Implement health checks properly
4. Use background model loading to avoid health check timeout
5. Proactively handle Python import issues

## 📥 Dataset Notes

- Images + Labels in YOLO format
- Folder: `train2017`, `val2017`
- Balanced across classes
- Label files located alongside each image

## 👨‍💻 Maintainer

**Capstone Project - NCT 2025**  
Model trained and optimized locally on limited hardware  
Feel free to fork or contribute!

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---
