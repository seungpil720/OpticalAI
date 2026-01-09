# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for OpenCV (Fixes "libgl1" error)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# [핵심 수정] 빌드 시점에 YOLO 모델을 미리 다운로드 (서버 시작 지연/타임아웃 방지)
RUN python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt')"

# Copy the rest of the app code
COPY . .

# Run the app using Gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
