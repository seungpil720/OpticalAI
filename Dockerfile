# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for OpenCV
# This fixes the "libgl1-mesa-glx not found" error from your logs
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to use Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app.py into the container
COPY . .

# Run the app using Gunicorn for production stability
# Gunicorn handles the $PORT variable automatically for Google Cloud Run
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
