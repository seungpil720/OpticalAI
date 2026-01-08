# 🚧 AI Hazard & Distance Detector (안전 거리 탐지기)

이 프로젝트는 **Python Flask**와 **YOLOv8**을 활용하여 사진 속 장애물을 인식하고, **실제 사물의 평균 높이 데이터**를 기반으로 거리를 정밀하게 추정하여 위험도를 알려주는 웹 애플리케이션입니다.

시각 장애인 보조, 로봇 주행, 혹은 일반적인 안전 거리 확보를 돕기 위해 설계되었으며 **Google Cloud Run** 배포에 최적화되어 있습니다.

## 📋 주요 기능 (Key Features)

1.  **광범위한 사물 인식 & 높이 매핑:**
    * 사람, 가구, 가전제품, 도로 시설물 등 **50여 가지 이상의 사물**에 대한 실제 평균 높이 데이터(`REAL_HEIGHTS`)를 내장하고 있습니다.
    * 예: `Person`(1.7m), `Chair`(0.9m), `Car`(1.5m), `Traffic Light`(4.0m) 등.
2.  **정밀 거리 계산 (Distance Estimation):**
    * **삼각형 닮음비(Triangle Similarity)** 원리를 사용합니다.
    * **지면 접촉 보정(Ground-Contact Bias):** 화면 하단(발 밑)에 위치한 사물은 더 가깝게 인식하도록 보정 알고리즘이 적용되어 있습니다.
3.  **위험 단계 분류 (Safety Classification):**
    * 거리에 따라 3단계로 위험을 알립니다.
    * 🔴 **STOP (< 1.3m):** 즉시 정지 필요 (충돌 위험)
    * 🟠 **WARNING (1.3m ~ 4.0m):** 주의 필요
    * 🟢 **SAFE (> 4.0m):** 안전 거리 확보됨
4.  **행동 가이드 (Actionable Advice):**
    * 위험 감지 시, 장애물의 위치(좌/우)를 파악하여 "왼쪽으로 피하세요(Step LEFT)"와 같은 구체적인 행동 지침을 제공합니다.

## 🛠 기술 스택 (Tech Stack)

* **Language:** Python 3.10
* **Web Framework:** Flask
* **AI Model:** Ultralytics YOLOv8 (`yolov8n.pt`)
* **Image Processing:** OpenCV (Headless), Pillow, NumPy
* **Deployment:** Docker, Google Cloud Run

## 📂 프로젝트 구조 (File Structure)

```bash
.
├── app.py                 # 메인 웹 애플리케이션 (거리 계산 및 위험 분석 로직 포함)
├── Dockerfile             # Google Cloud 배포용 도커 설정
├── requirements.txt       # 의존성 라이브러리 목록
├── yolov8n.pt             # YOLO AI 모델 (최초 실행 시 자동 다운로드)
└── README.md              # 프로젝트 설명서
