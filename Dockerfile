FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

# Explicitly expose port 8000
EXPOSE 8000

# Run the health check server which will then start the main app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]