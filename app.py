import os
import io
import base64
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, render_template_string
from ultralytics import YOLO

app = Flask(__name__)

# ==========================================
# 1. Configuration & Constants
# ==========================================
model = YOLO('yolov8n.pt')

# Extensive Real-World Height Map (Meters)
REAL_HEIGHTS = {
    # --- PEOPLE & ANIMALS ---
    "person": 1.70, "child": 1.20, "dog": 0.50, "cat": 0.25, "bird": 0.15,
    # --- INDOOR FURNITURE ---
    "couch": 0.85, "sofa": 0.85, "chair": 0.90, "armchair": 1.00,
    "bed": 0.60, "dining table": 0.75, "desk": 0.75, "coffee table": 0.45,
    "side table": 0.55, "shelf": 1.60, "bookcase": 1.80, "wardrobe": 2.00,
    "cabinet": 0.80, "stool": 0.50, "bench": 0.45, "toilet": 0.45,
    # --- HOUSEHOLD ITEMS ---
    "tv": 0.55, "monitor": 0.45, "laptop": 0.25, "refrigerator": 1.75,
    "microwave": 0.35, "oven": 0.85, "stove": 0.85, "sink": 0.85,
    "washing machine": 0.85, "vacuum cleaner": 1.00, "clock": 0.30,
    "vase": 0.35, "potted plant": 0.70, "lamp": 0.50, "chandelier": 0.70,
    "trash can": 0.50, "bucket": 0.35, "backpack": 0.50, "handbag": 0.30,
    # --- OUTDOOR ---
    "car": 1.50, "suv": 1.70, "van": 2.00, "truck": 3.00, "bus": 3.20,
    "bicycle": 1.00, "motorcycle": 1.10, "scooter": 1.00,
    "fire hydrant": 0.80, "stop sign": 2.50, "traffic light": 4.00,
    "parking meter": 1.30, "bench (outdoor)": 0.80, "mailbox": 1.20,
    "street light": 6.00, "tree": 5.00, "fence": 1.20,
    # --- ARCHITECTURE ---
    "door": 2.05, "window": 1.20, "stairs": 1.00, "elevator": 2.20
}

IMAGE_SIZE = 640
VFOV_DEG = 60
FOCAL_LENGTH = (IMAGE_SIZE / 2) / np.tan(np.deg2rad(VFOV_DEG / 2))

# ==========================================
# 2. Analysis Logic
# ==========================================
def analyze_hazards(pil_image):
    # Resize to 640x640 to match your Focal Length math
    img_resized = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # Convert PIL (RGB) to OpenCV (BGR)
    frame = np.array(img_resized)[:, :, ::-1].copy()

    # Inference (Lower confidence to catch obstacles)
    results = model(frame, verbose=False, conf=0.20)[0]
    
    detections = []
    
    for box in results.boxes:
        label = model.names[int(box.cls[0])]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        pixel_h = y2 - y1
        center_x = (x1 + x2) / 2
        
        # 1. Height Lookup
        real_h = REAL_HEIGHTS.get(label, 0.6) # Fallback 0.6m
        
        # 2. Distance Calculation
        distance = (real_h * FOCAL_LENGTH) / pixel_h
        
        # 3. Ground Contact Bias
        if y2 > 576: # Bottom 10% of 640px
            distance *= 0.75
            
        # 4. Safety Classification
        if distance < 1.3:
            status = "STOP"
            color = (0, 0, 255) # Red
        elif 1.3 <= distance <= 4.0:
            status = "WARNING"
            color = (0, 165, 255) # Orange
        else:
            status = "SAFE"
            color = (0, 255, 0) # Green
            
        detections.append({
            "label": label,
            "dist": distance,
            "status": status,
            "x": center_x,
            "box": (int(x1), int(y1), int(x2), int(y2)),
            "color": color
        })
        
        # Draw on frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        text = f"{label.upper()} {distance:.1f}m"
        cv2.putText(frame, text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Sort by distance (Nearest first)
    detections.sort(key=lambda x: x['dist'])
    
    return frame, detections

# ==========================================
# 3. Web Routes
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files: return "No file uploaded"
        file = request.files['file']
        if file.filename == '': return "No file selected"
        
        try:
            # Load Image
            image = Image.open(file.stream).convert('RGB')
            
            # Process
            processed_frame, detection_data = analyze_hazards(image)
            
            # Encode image for HTML
            _, buffer = cv2.imencode('.jpg', processed_frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Generate Report Text
            report_lines = []
            if not detection_data:
                report_lines.append("✅ Area clear. No obstacles detected.")
            else:
                for d in detection_data:
                    # Direction Logic
                    if d['x'] < 213: pos = "LEFT"
                    elif d['x'] > 427: pos = "RIGHT"
                    else: pos = "CENTER"
                    
                    icon = "🔴" if d['status'] == "STOP" else "🟠" if d['status'] == "WARNING" else "🟢"
                    line = f"{icon} <strong>{d['status']}</strong>: {d['label']} ({pos}) at {d['dist']:.1f}m"
                    
                    if d['status'] == "STOP":
                        step_dir = "RIGHT" if d['x'] < 320 else "LEFT"
                        line += f" <br>&nbsp;&nbsp;&nbsp;&nbsp;↪ ACTION: Step {step_dir} immediately!"
                    
                    report_lines.append(line)
            
            return render_template_string(RESULT_HTML, img_data=img_base64, report="<br>".join(report_lines))
            
        except Exception as e:
            return f"Error: {str(e)}"

    return render_template_string(UPLOAD_HTML)

# ==========================================
# 4. Templates
# ==========================================
UPLOAD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Safety Hazard Detector</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: sans-serif; background: #f0f2f5; text-align: center; padding: 20px; }
        .box { background: white; padding: 40px; border-radius: 12px; max-width: 500px; margin: 0 auto; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
        button { background: #d93025; color: white; border: none; padding: 12px 24px; font-size: 16px; border-radius: 6px; cursor: pointer; width: 100%; margin-top: 20px;}
        button:hover { background: #b02015; }
    </style>
</head>
<body>
    <div class="box">
        <h1>🚧 Hazard & Distance Detector</h1>
        <p>Upload a photo to scan for obstacles and safety hazards.</p>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required style="width:100%">
            <button type="submit">Analyze Safety</button>
        </form>
    </div>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Safety Report</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: sans-serif; background: #222; color: white; text-align: center; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        img { width: 100%; max-width: 640px; border: 2px solid #555; border-radius: 8px; }
        .report { background: #333; text-align: left; padding: 20px; margin-top: 20px; border-radius: 8px; line-height: 1.6; }
        .btn { display: inline-block; margin-top: 20px; padding: 10px 20px; background: #555; color: white; text-decoration: none; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Safety Analysis Report</h1>
        <img src="data:image/jpeg;base64,{{ img_data }}">
        <div class="report">
            {{ report|safe }}
        </div>
        <a href="/" class="btn">Scan Another</a>
    </div>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
