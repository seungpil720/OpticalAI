import os
import cv2
import base64
import numpy as np
from flask import Flask, render_template_string, request
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# --- CONFIGURATION & HEIGHT DATA ---
IMAGE_SIZE = 640
VFOV_DEG = 60
FOCAL_LENGTH = (IMAGE_SIZE / 2) / np.tan(np.deg2rad(VFOV_DEG / 2))

REAL_HEIGHTS = {
    "person": 1.70, "child": 1.20, "dog": 0.50, "cat": 0.25, "bird": 0.15,
    "couch": 0.85, "sofa": 0.85, "chair": 0.90, "armchair": 1.00,
    "bed": 0.60, "dining table": 0.75, "desk": 0.75, "coffee table": 0.45,
    "side table": 0.55, "shelf": 1.60, "bookcase": 1.80, "wardrobe": 2.00,
    "cabinet": 0.80, "stool": 0.50, "bench": 0.45, "toilet": 0.45,
    "tv": 0.55, "monitor": 0.45, "laptop": 0.25, "refrigerator": 1.75,
    "microwave": 0.35, "oven": 0.85, "stove": 0.85, "sink": 0.85,
    "washing machine": 0.85, "vacuum cleaner": 1.00, "clock": 0.30,
    "vase": 0.35, "potted plant": 0.70, "lamp": 0.50, "chandelier": 0.70,
    "trash can": 0.50, "bucket": 0.35, "backpack": 0.50, "handbag": 0.30,
    "car": 1.50, "suv": 1.70, "van": 2.00, "truck": 3.00, "bus": 3.20,
    "bicycle": 1.00, "motorcycle": 1.10, "scooter": 1.00,
    "fire hydrant": 0.80, "stop sign": 2.50, "traffic light": 4.00,
    "parking meter": 1.30, "bench (outdoor)": 0.80, "mailbox": 1.20,
    "street light": 6.00, "tree": 5.00, "fence": 1.20,
    "door": 2.05, "window": 1.20, "stairs": 1.00, "elevator": 2.20
}

# 모델 로드
model = YOLO('yolov8n.pt')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Vision Guard AI</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: sans-serif; background: #121212; color: white; text-align: center; padding: 20px; }
        .box { max-width: 800px; margin: 0 auto; background: #1e1e1e; padding: 20px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
        h1 { margin-bottom: 20px; color: #4fc3f7; }
        .item { padding: 15px; margin: 10px 0; border-radius: 10px; text-align: left; border-left: 10px solid; background: #2c2c2c; }
        .STOP { background: #3d0b13; border-color: #ff1744; }
        .WARNING { background: #3d3b0b; border-color: #ffea00; }
        .SAFE { background: #0b3d1c; border-color: #00e676; }
        select { margin-bottom: 20px; width: 100%; padding: 10px; font-size: 16px; border-radius: 5px; }
        button { width: 100%; padding: 15px; font-weight: bold; background: #4fc3f7; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; color: #121212; }
        button:hover { background: #29b6f6; }
        img.result-img { width: 100%; max-width: 640px; border-radius: 10px; margin-top: 20px; border: 2px solid #555; }
    </style>
</head>
<body>
    <div class="box">
        <h1>Vision Guard AI</h1>
        <p style="color:#aaa; font-size:0.9em;">Select an image to detect obstacles and estimate distance.</p>
        
        <form method="POST">
            <select name="filename">
                {% for img in images %}
                    <option value="{{ img }}" {% if selected_image == img %}selected{% endif %}>{{ img }}</option>
                {% endfor %}
            </select>
            <br>
            <button type="submit">ANALYZE IMAGE</button>
        </form>

        {% if img_data %}
            <div style="margin-top:30px;">
                <h3>Analysis Result</h3>
                <img src="data:image/jpeg;base64,{{ img_data }}" class="result-img">
            </div>
        {% endif %}

        {% if detections %}
            <div style="margin-top:30px;">
                <h3 style="text-align:left; border-bottom:1px solid #444; padding-bottom:10px;">Safety Report</h3>
                {% for d in detections %}
                    <div class="item {{ d.status }}">
                        <div style="font-size:1.1em; font-weight:bold;">{{ d.label | upper }}</div>
                        <div style="display:flex; justify-content:space-between; margin-top:5px; color:#ddd;">
                            <span>Position: {{ d.pos }}</span>
                            <span>Distance: {{ d.dist }}m</span>
                        </div>
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
    # 1. 파일 목록 가져오기
    try:
        all_files = os.listdir('.')
        image_list = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_list.sort()
    except Exception as e:
        image_list = []
        print(f"Error reading directory: {e}")

    detections = []
    img_data = None
    selected_image = image_list[0] if image_list else None

    if request.method == 'POST':
        selected_image = request.form.get('filename')
        
        if selected_image and selected_image in image_list:
            try:
                # 2. 이미지 로드 및 리사이즈
                pil_img = Image.open(selected_image).convert('RGB')
                pil_img = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE))
                frame = np.array(pil_img)[:, :, ::-1].copy() # RGB to BGR for OpenCV
                
                # 3. YOLO 추론
                results = model(frame, verbose=False, conf=0.20)[0]
                
                for box in results.boxes:
                    label = model.names[int(box.cls[0])]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    pixel_h = y2 - y1
                    center_x = (x1 + x2) / 2
                    
                    # 거리 계산
                    real_h = REAL_HEIGHTS.get(label, 0.6)
                    distance = (real_h * FOCAL_LENGTH) / pixel_h
                    
                    # 바닥 보정
                    if y2 > 576: 
                        distance *= 0.75
                    
                    # 상태 판단
                    if distance < 1.3: 
                        status = "STOP"
                        color = (0, 0, 255) # Red
                    elif 1.3 <= distance <= 4.0: 
                        status = "WARNING"
                        color = (0, 255, 255) # Yellow
                    else: 
                        status = "SAFE"
                        color = (0, 255, 0) # Green
                    
                    # 위치 판단
                    if center_x < 213: pos = "Left"
                    elif center_x > 427: pos = "Right"
                    else: pos = "Ahead"
                    
                    detections.append({
                        "label": label, 
                        "dist": round(distance, 1), 
                        "status": status, 
                        "pos": pos
                    })

                    # 4. 이미지에 그리기 (Bounding Box & Text)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{label.upper()} {distance:.1f}m"
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 5. 결과 이미지 인코딩 (Base64)
                _, buffer = cv2.imencode('.jpg', frame)
                img_data = base64.b64encode(buffer).decode('utf-8')
                
                detections.sort(key=lambda x: x['dist'])
                
            except Exception as e:
                print(f"Error processing image: {e}")
                
    return render_template_string(HTML_TEMPLATE, 
                                  detections=detections, 
                                  images=image_list, 
                                  selected_image=selected_image,
                                  img_data=img_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
