# Build stage
FROM node:20-alpine as build

WORKDIR /app

# 패키지 파일 복사
COPY package*.json ./

# 의존성 설치
RUN npm ci

# 소스 복사
COPY . .

# .env.production 파일이 포함되어 있어야 Vite가 빌드 시 자동으로 읽음
# --set-build-env-vars로 전달된 환경 변수도 사용 가능
# Vite는 다음 순서로 환경 변수를 읽음:
# 1. process.env (시스템 환경 변수, --set-build-env-vars로 전달)
# 2. .env.production 파일
# 3. .env 파일

# 빌드
RUN npm run build

# Production stage
FROM nginx:alpine

# nginx 설정
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 빌드된 파일 복사
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]




