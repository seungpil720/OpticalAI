FROM python:3.10-slim

# 파이썬 로그 즉시 출력
ENV PYTHONUNBUFFERED True
ENV APP_HOME /app
WORKDIR $APP_HOME

# [중요] 시스템 패키지 설치 (libgl1-mesa-glx 대신 libgl1 사용)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

COPY . ./
RUN pip install --no-cache-dir -r requirements.txt
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
