import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template_string, request
from PIL import Image
from ultralytics import YOLO
from werkzeug.serving import make_server

# ======================================================
# Flask App
# ======================================================
app = Flask(__name__)

# ======================================================
# 설정 및 거리 계산
# ======================================================
IMAGE_SIZE = 640
VFOV_DEG = 60
FOCAL_LENGTH = (IMAGE_SIZE / 2) / np.tan(np.deg2rad(VFOV_DEG / 2))

REAL_HEIGHTS = {
    "person": 1.70,
    "chair": 0.90,
    "couch": 0.85,
    "sofa": 0.85,
    "dining table": 0.75,
    "tv": 0.55,
    "laptop": 0.30,
    "bottle": 0.25,
    "vase": 0.40,
    "potted plant": 0.70,
    "fire hydrant": 0.80
}

# ======================================================
# YOLO Model
# ======================================================
model = YOLO("yolov8n.pt")

# ======================================================
# HTML Template
# ======================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Distance & Hazard Detector</title>
    <style>
        body { font-family: sans-serif; background: #f0f2f5; text-align: center; padding: 20px; }
        .container { background: white; max-width: 800px; margin: auto; padding: 20px;
                     border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .danger { background: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; }
        .safe { background: #d4edda; color: #155724; padding: 10px; border-radius: 5px; }
        img { max-width: 100%; border-radius: 5px; margin-top: 20px; }
    </style>
</head>
<body>
<div class="container">
    <h1>📸 AI Distance & Hazard Detector</h1>

    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <br><br>
        <button type="submit">Analyze</button>
    </form>

    {% if result_image %}
        <h2>Result</h2>
        <img src="data:image/jpeg;base64,{{ result_image }}">
        <h3>Distance Info</h3>
        {% for item in detections %}
            <div class="{{ 'danger' if item.dist < 2.0 else 'safe' }}">
                <b>{{ item.label }}</b> : {{ item.dist }} m
            </div>
        {% endfor %}
    {% endif %}
</div>
</body>
</html>
"""

# ======================================================
# Main Route
# ======================================================
@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None
    detections = []

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            img = Image.open(file).convert("RGB")
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            results = model(frame, verbose=False)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    pixel_h = max(1, y2 - y1)
                    if pixel_h < 20:
                        continue

                    label = model.names[int(box.cls[0])]
                    real_h = REAL_HEIGHTS.get(label, 0.5)
                    dist = (real_h * FOCAL_LENGTH) / pixel_h

                    detections.append({
                        "label": label.upper(),
                        "dist": round(dist, 2)
                    })

                    color = (0, 0, 255) if dist < 2.0 else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {dist:.1f}m",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            detections.sort(key=lambda x: x["dist"])

            _, buf = cv2.imencode(".jpg", frame)
            result_image = base64.b64encode(buf).decode()

    return render_template_string(
        HTML_TEMPLATE,
        result_image=result_image,
        detections=detections
    )

# ======================================================
# Auto-Port Runner (Jupyter-safe)
# ======================================================
def run_with_auto_port(app, host="0.0.0.0", start_port=8081, max_tries=50):
    last_err = None
    for port in range(start_port, start_port + max_tries):
        try:
            server = make_server(host, port, app)
            print(f"✅ Running at http://{host}:{port}")
            server.serve_forever()
            return
        except (OSError, SystemExit) as e:
            print(f"⚠️ Port {port} busy, trying next...")
            last_err = e
    raise RuntimeError("No available port found") from last_err

# ======================================================
# Entry Point
# ======================================================
if __name__ == "__main__":
    base_port = int(os.environ.get("PORT", 8081))
    run_with_auto_port(app, start_port=base_port)
