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
# 1. 설정 및 데이터 (Configuration)
# ==========================================
model = YOLO('yolov8n.pt')

# 실제 사물 평균 높이 데이터 (단위: m)
REAL_HEIGHTS = {
    # 사람/동물
    "person": 1.70, "child": 1.20, "dog": 0.50, "cat": 0.25,
    # 실내 가구
    "couch": 0.85, "sofa": 0.85, "chair": 0.90, "bed": 0.60, 
    "dining table": 0.75, "desk": 0.75, "refrigerator": 1.75, 
    "cabinet": 0.80, "potted plant": 0.70, "tv": 0.55,
    # 실외/도로
    "car": 1.50, "bus": 3.20, "truck": 3.00, "bicycle": 1.00, 
    "motorcycle": 1.10, "traffic light": 4.00, "stop sign": 2.50, 
    "fire hydrant": 0.80, "bench": 0.80, "street light": 6.00
}

IMAGE_SIZE = 640
VFOV_DEG = 60
FOCAL_LENGTH = (IMAGE_SIZE / 2) / np.tan(np.deg2rad(VFOV_DEG / 2))

# ==========================================
# 2. 분석 로직 (Analysis Logic)
# ==========================================
def analyze_hazards(pil_image):
    # 1. 이미지 크기 조정 및 포맷 변환 (Pillow -> OpenCV)
    img_resized = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    frame = np.array(img_resized)[:, :, ::-1].copy()

    # 2. YOLO 추론 (낮은 신뢰도 허용)
    results = model(frame, verbose=False, conf=0.20)[0]
    
    detections = []
    
    for box in results.boxes:
        label = model.names[int(box.cls[0])]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        pixel_h = y2 - y1
        center_x = (x1 + x2) / 2
        
        # 높이 기반 거리 계산
        real_h = REAL_HEIGHTS.get(label, 0.6)
        distance = (real_h * FOCAL_LENGTH) / pixel_h
        
        # 바닥 보정 (화면 아래쪽 사물은 더 가깝게)
        if y2 > 576: 
            distance *= 0.75
            
        # 위험도 분류
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
            "label": label, "dist": distance, "status": status,
            "x": center_x, "color": color, "box": (int(x1), int(y1), int(x2), int(y2))
        })
        
        # 그리기
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        text = f"{label.upper()} {distance:.1f}m"
        cv2.putText(frame, text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    detections.sort(key=lambda x: x['dist'])
    return frame, detections

# ==========================================
# 3. 웹 라우팅 (Web Routes)
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def home():
    # 현재 폴더의 이미지 파일 검색 (.png, .jpg, .jpeg)
    all_files = os.listdir('.')
    image_list = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_list.sort()
    
    selected_image = None
    img_data = None
    report_html = ""
    
    if request.method == 'POST':
        selected_image = request.form.get('filename')
        
        if selected_image and selected_image in image_list:
            try:
                # Pillow로 이미지 열기 (다양한 포맷 지원)
                image = Image.open(selected_image).convert('RGB')
                
                # 분석 수행
                processed_frame, detection_data = analyze_hazards(image)
                
                # 결과 이미지 인코딩
                _, buffer = cv2.imencode('.jpg', processed_frame)
                img_data = base64.b64encode(buffer).decode('utf-8')
                
                # 리포트 생성
                lines = []
                if not detection_data:
                    lines.append("✅ 감지된 장애물이 없습니다.")
                else:
                    for d in detection_data:
                        pos = "LEFT" if d['x'] < 213 else "RIGHT" if d['x'] > 427 else "CENTER"
                        icon = "🔴" if d['status'] == "STOP" else "🟠" if d['status'] == "WARNING" else "🟢"
                        
                        line = f"{icon} <strong>{d['status']}</strong>: {d['label']} ({pos}) - 약 {d['dist']:.1f}m"
                        if d['status'] == "STOP":
                            step = "RIGHT" if d['x'] < 320 else "LEFT"
                            line += f"<br>&nbsp;&nbsp;&nbsp;&nbsp;↪ 🚨 <strong>즉시 {step}쪽으로 피하세요!</strong>"
                        lines.append(line)
                report_html = "<br><br>".join(lines)
                
            except Exception as e:
                report_html = f"Error: {str(e)}"

    return render_template_string(HTML_TEMPLATE, 
                                  images=image_list, 
                                  selected=selected_image, 
                                  img_data=img_data, 
                                  report=report_html)

# ==========================================
# 4. HTML 템플릿
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Safety Scanner</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #222; color: #eee; text-align: center; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; background: #333; padding: 30px; border-radius: 15px; }
        h1 { color: #f1c40f; }
        select { padding: 10px; font-size: 16px; width: 70%; margin-bottom: 20px; border-radius: 5px; }
        button { padding: 10px 20px; font-size: 16px; background: #e74c3c; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #c0392b; }
        .result-img { width: 100%; max-width: 640px; border: 3px solid #555; border-radius: 10px; margin-top: 20px; }
        .report-box { background: #444; text-align: left; padding: 20px; margin-top: 20px; border-radius: 10px; line-height: 1.6; border-left: 5px solid #e74c3c; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚧 AI 장애물 & 거리 탐지기</h1>
        <p>서버에 저장된 사진을 선택하여 위험 요소를 분석합니다.</p>
        
        <form method="POST">
            <select name="filename">
                <option value="" disabled {% if not selected %}selected{% endif %}>-- 사진 선택 --</option>
                {% for img in images %}
                <option value="{{ img }}" {% if img == selected %}selected{% endif %}>{{ img }}</option>
                {% endfor %}
            </select>
            <br>
            <button type="submit">🔍 안전 분석 시작</button>
        </form>

        {% if img_data %}
            <img src="data:image/jpeg;base64,{{ img_data }}" class="result-img">
            <div class="report-box">
                {{ report|safe }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
