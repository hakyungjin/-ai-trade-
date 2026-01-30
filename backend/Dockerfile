# Python 3.11 기반 이미지
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 환경 변수 설정
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# 포트 노출
EXPOSE 8080

# 서버 실행 (Cloud Run의 PORT 환경 변수 사용)
# Python으로 PORT 환경 변수 읽기
CMD python -c "import os; import uvicorn; port = int(os.environ.get('PORT', 8080)); uvicorn.run('app.main:app', host='0.0.0.0', port=port)"



