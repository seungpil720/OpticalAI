# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies with better error handling
# Added --fix-missing to prevent build failures on network glitches
RUN apt-get update --fix-missing && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to use Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLO model to prevent runtime timeout
RUN python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt')"

# Copy the rest of the app code
COPY . .

# Run the app using Gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
