import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template_string, request
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# ======================================================
# 설정 및 거리 계산 데이터
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
# YOLO Model 로드
# ======================================================
# 모델 로드 (전역 변수로 한 번만 로드)
model = YOLO("yolov8n.pt")

# ======================================================
# HTML Template
# ======================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Distance & Hazard Detector</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: sans-serif; background: #f0f2f5; text-align: center; padding: 20px; }
        .container { background: white; max-width: 800px; margin: auto; padding: 20px;
                     border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .danger { background: #ffebee; color: #c62828; padding: 10px; border-radius: 5px; margin: 5px 0; border: 1px solid #ffcdd2; }
        .safe { background: #e8f5e9; color: #2e7d32; padding: 10px; border-radius: 5px; margin: 5px 0; border: 1px solid #c8e6c9; }
        img { max-width: 100%; border-radius: 5px; margin-top: 20px; }
        button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        button:hover { background-color: #0056b3; }
    </style>
</head>
<body>
<div class="container">
    <h1>📸 AI Distance & Hazard Detector</h1>
    <p>Upload a photo to detect objects and estimate distance.</p>

    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <br><br>
        <button type="submit">Analyze Image</button>
    </form>

    {% if result_image %}
        <h2>Analysis Result</h2>
        <img src="data:image/jpeg;base64,{{ result_image }}">
        
        <div style="text-align: left; margin-top: 20px;">
            <h3>📊 Detailed Info</h3>
            {% for item in detections %}
                <div class="{{ 'danger' if item.dist < 2.0 else 'safe' }}">
                    <strong>{{ item.label }}</strong> : 약 {{ item.dist }}m 
                    {% if item.dist < 2.0 %} (⚠️ WARNING) {% else %} (SAFE) {% endif %}
                </div>
            {% endfor %}
        </div>
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
            try:
                # 이미지 읽기 및 변환
                img = Image.open(file).convert("RGB")
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # YOLO 추론
                results = model(frame, verbose=False)

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        pixel_h = max(1, y2 - y1)
                        
                        # 너무 작은 박스는 무시 (노이즈 제거)
                        if pixel_h < 20:
                            continue

                        cls_id = int(box.cls[0])
                        label = model.names[cls_id]
                        
                        # 거리 계산
                        real_h = REAL_HEIGHTS.get(label, 0.5)
                        dist = (real_h * FOCAL_LENGTH) / pixel_h

                        detections.append({
                            "label": label.upper(),
                            "dist": round(dist, 2)
                        })

                        # 이미지에 그리기
                        color = (0, 0, 255) if dist < 2.0 else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label} {dist:.1f}m",
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 결과 정렬
                detections.sort(key=lambda x: x["dist"])

                # 이미지 인코딩
                _, buf = cv2.imencode(".jpg", frame)
                result_image = base64.b64encode(buf).decode("utf-8")

            except Exception as e:
                print(f"Error processing image: {e}")
                return f"Error: {e}", 500

    return render_template_string(
        HTML_TEMPLATE,
        result_image=result_image,
        detections=detections
    )

# ======================================================
# Entry Point (Production vs Local)
# ======================================================
if __name__ == "__main__":
    # 로컬 테스트용 (python app.py 실행 시에만 작동)
    # Cloud Run(Gunicorn)에서는 이 블록이 실행되지 않음
    port = int(os.environ.get("PORT", 8081))
    app.run(host="0.0.0.0", port=port, debug=True)
