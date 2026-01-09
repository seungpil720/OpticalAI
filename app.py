import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template_string, request
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# --- 설정 및 거리 계산용 데이터 ---
# 이미지 크기 (YOLO 표준 입력 크기)
IMAGE_SIZE = 640

# 가상의 카메라 초점 거리 (Focal Length) 설정
# 일반적인 스마트폰 카메라의 수직 화각(VFOV)을 약 60도로 가정하고 계산
VFOV_DEG = 60
FOCAL_LENGTH = (IMAGE_SIZE / 2) / np.tan(np.deg2rad(VFOV_DEG / 2))

# 사물별 평균 실제 높이 (단위: 미터) - 거리 추정용 데이터베이스
# 제공해주신 사진 속 물체들(의자, 사람, 소화기, 테이블 등)을 포함
REAL_HEIGHTS = {
    "person": 1.70,
    "chair": 0.90,
    "couch": 0.85,
    "sofa": 0.85,
    "dining table": 0.75,
    "tv": 0.55,          # 모니터/키오스크 등
    "laptop": 0.30,
    "bottle": 0.25,      # 소화기를 bottle로 인식할 경우를 대비
    "vase": 0.40,        # 화분 등
    "potted plant": 0.70,
    "fire hydrant": 0.80 # 소화전/소화기
}

# YOLO 모델 로드 (처음 실행 시 자동으로 다운로드 됩니다)
model = YOLO('yolov8n.pt') 

# --- HTML 템플릿 (프론트엔드) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Distance & Hazard Detector</title>
    <style>
        body { font-family: sans-serif; background: #f0f2f5; text-align: center; padding: 20px; }
        .container { background: white; max-width: 800px; margin: auto; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        .upload-btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .upload-btn:hover { background: #0056b3; }
        img { max-width: 100%; height: auto; border-radius: 5px; margin-top: 20px; }
        .alert { padding: 10px; margin: 10px 0; border-radius: 5px; text-align: left; }
        .danger { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .safe { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📸 사물 인식 및 거리 측정</h1>
        <p>사진을 업로드하면 AI가 사물을 인식하고 거리를 계산합니다.</p>
        
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <br><br>
            <button type="submit" class="upload-btn">분석 시작</button>
        </form>

        {% if result_image %}
            <h2>분석 결과</h2>
            <img src="data:image/jpeg;base64,{{ result_image }}" alt="Analyzed Image">
            
            <div style="margin-top: 20px;">
                <h3>📊 거리 상세 정보</h3>
                {% for item in detections %}
                    <div class="alert {% if item.dist < 2.0 %}danger{% else %}safe{% endif %}">
                        <strong>{{ item.label }}</strong>: 약 {{ item.dist }}m 
                        {% if item.dist < 2.0 %} (⚠️ 가까움 - 주의!) {% else %} (안전 거리) {% endif %}
                    </div>
                {% endfor %}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    result_image = None
    detections = []

    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            # 이미지 읽기 및 변환
            img = Image.open(file).convert('RGB')
            # YOLO 분석을 위해 리사이즈 (속도 최적화)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE)) 
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # YOLO 추론
            results = model(frame)
            
            for result in results:
                for box in result.boxes:
                    # 클래스 ID 및 이름
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    
                    # 바운딩 박스 좌표
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    pixel_height = y2 - y1
                    
                    # 거리 계산 (알려진 물체만 계산)
                    real_height = REAL_HEIGHTS.get(label, 0.5) # 모르는 물체는 0.5m 가정
                    
                    # 거리 공식: Distance = (Real_H * Focal_Length) / Pixel_H
                    distance = (real_height * FOCAL_LENGTH) / pixel_height
                    
                    # 결과 리스트 저장
                    detections.append({
                        "label": label.upper(),
                        "dist": round(distance, 2)
                    })

                    # 이미지에 박스 및 텍스트 그리기
                    color = (0, 0, 255) if distance < 2.0 else (0, 255, 0) # 2m 미만 빨간색
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {distance:.1f}m", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 결과 이미지를 HTML에 표시하기 위해 Base64로 인코딩
            _, buffer = cv2.imencode('.jpg', frame)
            result_image = base64.b64encode(buffer).decode('utf-8')
            
            # 거리 순으로 정렬
            detections.sort(key=lambda x: x['dist'])

    return render_template_string(HTML_TEMPLATE, result_image=result_image, detections=detections)

if __name__ == '__main__':
    # Cloud Run은 $PORT 환경변수를 사용합니다.
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
