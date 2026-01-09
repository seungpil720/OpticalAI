# 1. 파이썬 가벼운 버전 사용
FROM python:3.9-slim

# 2. 작업 폴더 설정
WORKDIR /app

# 3. OpenCV 실행에 필요한 시스템 라이브러리 설치 (필수)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 복사
COPY . .

# 6. Gunicorn으로 서버 실행 (Cloud Run의 $PORT 환경변수 사용)
# 타임아웃을 120초로 늘려 AI 모델 로딩 시간을 확보합니다.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 120 app:app
