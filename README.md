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

## 🚄 Deployment on Railway (Quick Instructions)

1. Create a free account on [Railway](https://railway.app/).
2. Create a new project and connect it to your project repository (GitHub or upload files manually).
3. In the service settings:
   - **Environment**: Python 3.10 or higher
   - **Start Command**:
     ```bash
     uvicorn main:app --host 0.0.0.0 --port $PORT
     ```
   - **Install Command**: `pip install -r requirements.txt`
   - **Port**: Leave as default (Railway sets the PORT variable automatically)
4. Upload the model file `best.pt` along with other files.
5. After deployment, use the URL provided by Railway to access the API interface.

> **Note:** If you encounter issues loading the model, ensure the file size doesn't exceed Railway's limit (typically 500MB for individual files on the free plan).

---

## 💡 Known Issues & Fixes

- **OMP Error**: Set the following before running training

  ```bash
  set KMP_DUPLICATE_LIB_OK=TRUE
  ```

- **Numpy bool8 Error**: Downgrade numpy to a stable version

  ```bash
  pip install numpy==1.23.5
  ```

- **TensorBoard compatibility**: Downgrade to `tensorboard==2.10.1` to match TensorFlow-GPU 2.10

---

## 📥 Dataset Notes

- Images + Labels in YOLO format
- Folder: `train2017`, `val2017`
- Balanced across classes
- Label files located alongside each image

---

## 👨‍💻 Maintainer

**Capstone Project - NCT 2025**  
Model trained and optimized locally on limited hardware  
Feel free to fork or contribute!

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🖥️ Using the API (FastAPI)

After deploying the project on Railway or any server that supports Python, you can send an image to the `/predict` endpoint to get object detection results.

### Example request using `curl`:

```bash
curl -X POST "https://YOUR-RAILWAY-URL/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

### Example response:

```json
{
  "results": [
    {
      "xmin": 123.4,
      "ymin": 56.7,
      "xmax": 234.5,
      "ymax": 167.8,
      "confidence": 0.92,
      "class": 0,
      "name": "person"
    },
    ...
  ]
}
```

- **YOUR-RAILWAY-URL**: Replace with your service URL on Railway.
- Each item in results represents a detected object with box coordinates, confidence, and class number/name.

---
