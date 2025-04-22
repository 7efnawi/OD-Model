# 🚀 YOLOv5 Object Detection API

<div align="center">
  
  ![Object Detection Banner](https://miro.medium.com/v2/resize:fit:1400/1*QOGcQM9G4dFAYJq-RK0YYg.png)

  <p>
    <b>Advanced Object Detection System using YOLOv5</b><br>
    Professional API service for object detection in images with easy deployment
  </p>

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![YOLOv5](https://img.shields.io/badge/Model-YOLOv5-brightgreen.svg)](https://github.com/ultralytics/yolov5)
[![Flask](https://img.shields.io/badge/Framework-Flask-red.svg)](https://flask.palletsprojects.com/)
[![Railway](https://img.shields.io/badge/Deployment-Railway-blueviolet.svg)](https://railway.app/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

## 🔍 Overview

This project is an integrated system for detecting objects in images using the YOLOv5 algorithm trained on the COCO dataset. The project provides an easy-to-use REST API built with Flask with the ability to quickly deploy on the Railway platform.

### ✨ Key Features

- ⚡️ **Lightning Fast**: Real-time object detection
- 🧠 **Advanced Model**: Using YOLOv5 trained on COCO dataset
- 📊 **High Accuracy**: mAP@0.5 reaching up to 0.54+
- 🌐 **User-Friendly API**: Simple and fully documented programming interface
- 🔄 **Background Loading**: No server response delay during model loading
- 🚂 **Deployment Ready**: Full integration with Railway platform

<br>

## 📋 Model Information

<table>
  <tr>
    <td><b>Base Model</b></td>
    <td>YOLOv5s</td>
  </tr>
  <tr>
    <td><b>Image Size</b></td>
    <td>640×640</td>
  </tr>
  <tr>
    <td><b>Number of Classes</b></td>
    <td>80 (COCO format)</td>
  </tr>
  <tr>
    <td><b>Final Dataset Size</b></td>
    <td>~60% of COCO (balanced across classes)</td>
  </tr>
  <tr>
    <td><b>Training Epochs</b></td>
    <td>50 + additional training</td>
  </tr>
  <tr>
    <td><b>Precision</b></td>
    <td>~0.63</td>
  </tr>
  <tr>
    <td><b>Recall</b></td>
    <td>~0.47+</td>
  </tr>
</table>

<br>

## 🛠️ Local Installation

```bash
# Clone the repository
git clone https://github.com/7efnawi/OD-Model.git
cd OD-Model

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The application will be available at: `http://localhost:8000`

<br>

## 🌐 Deployment on Railway

<div align="center">
  <img src="https://railway.app/brand/logo-light.svg" width="250" alt="Railway Logo">
</div>

1. Create an account on [Railway.app](https://railway.app)
2. Create a new project and link it to your GitHub repository
3. Railway will automatically detect the presence of a Dockerfile and use it for building and deployment

<br>

## 🔌 API Interface

### 1. Main Endpoint

- **Path**: `/`
- **Method**: GET
- **Response**: Model status and application information

```json
{
  "message": "YOLOv5 Object Detection API",
  "model_status": "loaded",
  "error": null
}
```

### 2. Health Check

- **Path**: `/health`
- **Method**: GET
- **Response**: Application status

```json
{
  "status": "ok"
}
```

### 3. Object Detection

- **Path**: `/predict`
- **Method**: POST
- **Content Type**: multipart/form-data
- **Data**:
  - `file`: Image file to be analyzed
- **Response**: Object detection results

```json
{
  "status": "success",
  "message": "Image processed successfully",
  "results": [
    {
      "class": 0,
      "confidence": 0.85,
      "name": "person",
      "xmin": 120.5,
      "ymin": 220.3,
      "xmax": 250.8,
      "ymax": 380.1
    }
  ]
}
```

<br>

## 💻 Usage Examples

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

<br>

## 🧩 Project Structure

```
OD-Model/
│
├── app.py                        # Main Flask application
├── test_predict.py               # Prediction test script
├── Dockerfile                    # Docker file for deployment
│
├── yolov5/                       # Embedded YOLOv5 library
│
├── MY_coco30_yolov5.yaml         # Custom YAML configuration file
├── best.pt                       # Trained model weights
│
├── requirements.txt              # Python dependencies
├── railway.json                  # Railway deployment settings
├── .gitignore                    # List of ignored files
└── README.md                     # You are here!
```

<br>

## 🔧 Technical Challenges and Solutions

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

<br>

## 📊 Training and Evaluation

### Model Training

To train the model using 60% of the dataset:

```bash
python train.py --img 640 --batch 4 --epochs 50 \
  --data data/MY_coco_yolov5.yaml \
  --weights yolov5s.pt --device 0
```

To resume training from the best checkpoint:

```bash
python train.py --weights runs/train/exp6/weights/best.pt \
  --data data/MY_coco_yolov5.yaml --device 0
```

### View Training Results with TensorBoard

```bash
tensorboard --logdir runs/train
```

Then open [http://localhost:6006](http://localhost:6006) in your browser.

<br>

## 📋 Solutions for Common Issues

- **OMP Error**: Set the following before running training

  ```bash
  set KMP_DUPLICATE_LIB_OK=TRUE
  ```

- **Numpy bool8 Error**: Downgrade to a stable version

  ```bash
  pip install numpy==1.23.5
  ```

- **TensorBoard compatibility**: Downgrade to `tensorboard==2.10.1` to match TensorFlow-GPU 2.10

<br>

## 👨‍💻 The Project

**Capstone Project - NCT 2025**  
Model trained and optimized locally on limited hardware  
Feel free to fork or contribute!

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  <p>🔍 Discover. 🤖 Predict. 🚀 Deploy.</p>
</div>
