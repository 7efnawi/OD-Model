# YOLOv5 Object Detection API

This project provides a FastAPI-based REST API for object detection using YOLOv5. It allows users to upload images and receive detection results in JSON format.

## Features

- FastAPI-based REST API
- YOLOv5 object detection
- Support for image uploads
- JSON response format
- Error handling
- Hot reload for development

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

## Installation

1. Clone the repository:

```bash
git clone <https://github.com/7efnawi/OD-Model.git>
cd <D:/NCT/NCT-2/S2/Capston/OD Model>
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Make sure you have the YOLOv5 model file (`best.pt`) in the project directory.

## Usage

1. Start the server:

```bash
python main.py
```

2. The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Root Endpoint

- **URL**: `/`
- **Method**: GET
- **Response**: Status message

```json
{
  "message": "YOLOv5 Object Detection API is running."
}
```

#### 2. Prediction Endpoint

- **URL**: `/predict`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Parameters**:
  - `file`: Image file to process
- **Response**: Detection results in JSON format

```json
{
    "results": [
        {
            "xmin": float,
            "ymin": float,
            "xmax": float,
            "ymax": float,
            "confidence": float,
            "class": int,
            "name": string
        }
    ]
}
```

## Error Handling

The API includes error handling for:

- Invalid file uploads
- Processing errors
- Server errors

## Development

- The server runs on port 8000
- Hot reload is enabled for development
- The server binds to all interfaces (0.0.0.0)

