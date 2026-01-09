import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template_string, request
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# ======================================================
# 1. 설정 및 거리 데이터 (Colab 코드와 동일)
# ======================================================
IMAGE_SIZE = 640
VFOV_DEG = 60
FOCAL_LENGTH = (IMAGE_SIZE / 2) / np.tan(np.deg2rad(VFOV_DEG / 2))

REAL_HEIGHTS = {
    "person": 1.70, "child": 1.20, "dog": 0.50, "cat": 0.25,
    "chair": 0.90, "couch": 0.85, "sofa": 0.85, "dining table": 0.75,
    "tv": 0.55, "laptop": 0.30, "mouse": 0.05, "keyboard": 0.03,
    "cell phone": 0.15, "bottle": 0.25, "cup": 0.10, "vase": 0.40,
    "potted plant": 0.70, "fire hydrant": 0.80, "car": 1.50,
    "bus": 3.20, "truck": 3.00, "traffic light": 3.00, "stop sign": 2.50
}

# 모델 미리 로드 (서버 시작 시 1회만 실행됨)
print("Load YOLO model...")
model = YOLO('yolov8n.pt')

# ======================================================
# 2. HTML UI 템플릿
# ======================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Vision Guard AI</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: sans-serif; background: #f4f4f9; text-align: center; padding: 20px; }
        .container { background: white; max-width: 800px; margin: auto; padding: 20px;
                     border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        h1 { color: #333; margin-bottom: 10px; }
        .btn { background: #4a90e2; color: white; padding: 12px 25px; border: none;
               border-radius: 8px; font-size: 16px; cursor: pointer; transition: 0.3s; }
        .btn:hover { background: #357abd; }
        .result-box { margin-top: 20px; text-align: left; }
        .item { padding: 10px; margin: 5px 0; border-radius: 5px; color: white; font-weight: bold; }
        .danger { background-color: #ff5252; } /* 빨강 */
        .safe { background-color: #66bb6a; }   /* 초록 */
        img { max-width: 100%; border-radius: 10px; margin-top: 15px; border: 2px solid #ddd; }
        input[type=file] { margin-bottom: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>👁️ Vision Guard AI</h1>
        <p>사진을 업로드하면 위험 요소와 거리를 분석합니다.</p>
        
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <br>
            <button type="submit" class="btn">분석 시작 (Analyze)</button>
        </form>

        {% if result_image %}
            <div class="result-box">
                <h2>📊 분석 결과</h2>
                <img src="data:image/jpeg;base64,{{ result_image }}">
                
                <div style="margin-top: 20px;">
                    {% for item in detections %}
                        <div class="item {{ 'danger' if item.is_danger else 'safe' }}">
                            {{ item.label }} : 약 {{ item.dist }}m 
                            [{{ '⚠️ 접근 금지' if item.is_danger else '✅ 안전' }}]
                        </div>
                    {% endfor %}
                </div>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

# ======================================================
# 3. 메인 로직
# ======================================================
@app.route('/', methods=['GET', 'POST'])
def index():
    result_image = None
    detections = []

    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            try:
                # 1. 이미지 읽기
                img_pil = Image.open(file).convert("RGB")
                
                # 2. 리사이즈 (Colab 코드와 동일하게 640x640)
                img_pil = img_pil.resize((IMAGE_SIZE, IMAGE_SIZE))
                frame = np.array(img_pil)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # 3. YOLO 추론
                results = model(frame, verbose=False)[0]

                # 4. 결과 처리 및 그리기
                temp_detections = []
                
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    pixel_h = max(1, y2 - y1)

                    if pixel_h < 15: continue # 노이즈 제거

                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]

                    # 거리 계산
                    real_h = REAL_HEIGHTS.get(label, 0.5)
                    distance = (real_h * FOCAL_LENGTH) / pixel_h

                    # 바닥 보정 (Colab 코드 동일 적용)
                    if y2 > (IMAGE_SIZE * 0.9):
                        distance *= 0.8

                    is_danger = distance < 2.0
                    
                    temp_detections.append({
                        "label": label.upper(),
                        "dist": round(distance, 2),
                        "is_danger": is_danger,
                        "coords": (x1, y1, x2, y2)
                    })

                # 거리순 정렬 (먼 곳부터 그려야 텍스트가 안 겹침)
                temp_detections.sort(key=lambda x: x['dist'], reverse=True)

                for item in temp_detections:
                    x1, y1, x2, y2 = item['coords']
                    color = (0, 0, 255) if item['is_danger'] else (0, 255, 0)
                    
                    # 박스
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # 라벨
                    text = f"{item['label']} {item['dist']}m"
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # 프론트엔드 전달용 리스트 (역순 정렬된 것을 다시 가까운 순으로 보이게 하려면 여기서 조정 가능)
                    detections.insert(0, item) 

                # 5. 이미지 인코딩 (Base64)
                _, buffer = cv2.imencode('.jpg', frame)
                result_image = base64.b64encode(buffer).decode('utf-8')

            except Exception as e:
                print(f"Error: {e}")
                return f"서버 처리 중 오류 발생: {e}", 500

    return render_template_string(HTML_TEMPLATE, result_image=result_image, detections=detections)

if __name__ == '__main__':
    # 로컬 테스트용
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
