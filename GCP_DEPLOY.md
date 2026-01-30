# GCP Cloud Run 배포 가이드

## 🚀 빠른 시작 (5분)

```powershell
# 1. gcloud CLI 설치 및 로그인
gcloud auth login
gcloud config set project gen-lang-client-0823293183

# 2. 환경 변수 설정
$env:DATABASE_URL = "mysql+aiomysql://user:pass@host:3306/crypto_trader"
$env:BINANCE_API_KEY = "your_api_key"
$env:BINANCE_SECRET_KEY = "your_secret_key"

# 3. 배포 실행 (프로젝트 ID는 기본값으로 설정됨)
.\deploy-gcp.ps1

# 또는 다른 프로젝트 사용 시
.\deploy-gcp.ps1 -ProjectId "your-project-id" -Region "asia-northeast3"
```

배포 완료 후 프론트엔드 URL이 표시됩니다! 🎉

---

## 사전 준비

### 1. GCP 프로젝트 설정
현재 프로젝트 ID: `gen-lang-client-0823293183`

다른 프로젝트를 사용하려면:
1. https://console.cloud.google.com 접속
2. 새 프로젝트 생성 또는 기존 프로젝트 선택
3. 프로젝트 ID 확인

### 2. gcloud CLI 설치
```powershell
# Windows (PowerShell 관리자 권한)
(New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")
& $env:Temp\GoogleCloudSDKInstaller.exe
```

### 3. gcloud 로그인 및 설정
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

---

## 백엔드 배포

### 방법 1: 배포 스크립트 사용 (추천)

```powershell
# 환경 변수 설정
$env:DATABASE_URL = "mysql+aiomysql://user:pass@host:3306/crypto_trader"
$env:BINANCE_API_KEY = "your_api_key"
$env:BINANCE_SECRET_KEY = "your_secret_key"
$env:BINANCE_TESTNET = "false"

# 배포 실행
.\deploy-gcp.ps1 -ProjectId "your-project-id" -Region "asia-northeast3"
```

### 방법 2: 수동 배포

#### 1. 환경 변수 설정
```powershell
# PowerShell에서 환경 변수 설정
$env:DATABASE_URL = "mysql+aiomysql://user:pass@host:3306/crypto_trader"
$env:BINANCE_API_KEY = "your_api_key"
$env:BINANCE_SECRET_KEY = "your_secret_key"
$env:BINANCE_TESTNET = "false"
```

#### 2. Cloud Run 배포
```powershell
cd backend

# 이미지 빌드 및 배포 (한 번에)
gcloud run deploy crypto-backend `
  --source . `
  --platform managed `
  --region asia-northeast3 `
  --allow-unauthenticated `
  --memory 1Gi `
  --cpu 1 `
  --timeout 300 `
  --set-env-vars="DATABASE_URL=$env:DATABASE_URL,BINANCE_API_KEY=$env:BINANCE_API_KEY,BINANCE_SECRET_KEY=$env:BINANCE_SECRET_KEY"
```

#### 3. 배포 URL 확인
배포 완료 후 표시되는 URL 기억 (예: https://crypto-backend-xxxxx-du.a.run.app)

---

## 프론트엔드 배포

### 방법 1: 프론트엔드 전용 스크립트 사용 (추천)

```powershell
# 백엔드 URL 자동 감지 또는 수동 입력
.\deploy-frontend.ps1

# 또는 백엔드 URL 직접 지정
.\deploy-frontend.ps1 -BackendUrl "https://crypto-backend-xxxxx-du.a.run.app"
```

### 방법 2: 통합 배포 스크립트 사용
백엔드 배포 후 자동으로 프론트엔드도 배포됩니다:
```powershell
.\deploy-gcp.ps1
```

### 방법 3: 수동 배포
```powershell
cd frontend

# 백엔드 URL을 빌드 인자로 전달
$backendUrl = "https://crypto-backend-xxxxx-du.a.run.app"

gcloud run deploy crypto-frontend `
  --source . `
  --platform managed `
  --region asia-northeast3 `
  --allow-unauthenticated `
  --memory 512Mi `
  --cpu 1 `
  --build-arg="VITE_API_URL=$backendUrl"
```

---

## 데이터베이스 옵션

### Option 1: Cloud SQL MySQL (추천 - GCP 통합)

#### 1. Cloud SQL 인스턴스 생성
```powershell
# MySQL 8.0 인스턴스 생성
gcloud sql instances create crypto-db `
  --database-version=MYSQL_8_0 `
  --tier=db-f1-micro `
  --region=asia-northeast3 `
  --root-password=YOUR_ROOT_PASSWORD

# 데이터베이스 생성
gcloud sql databases create crypto_trader --instance=crypto-db

# 사용자 생성 (선택사항)
gcloud sql users create crypto_user `
  --instance=crypto-db `
  --password=YOUR_USER_PASSWORD
```

#### 2. Cloud Run과 연결
```powershell
# Cloud Run 서비스에 Cloud SQL 연결 추가
gcloud run services update crypto-backend `
  --region asia-northeast3 `
  --add-cloudsql-instances=PROJECT_ID:asia-northeast3:crypto-db

# DATABASE_URL 설정 (Unix 소켓 사용)
# 형식: mysql+aiomysql://user:password@/database?unix_socket=/cloudsql/PROJECT_ID:REGION:INSTANCE_NAME
$env:DATABASE_URL = "mysql+aiomysql://crypto_user:password@/crypto_trader?unix_socket=/cloudsql/PROJECT_ID:asia-northeast3:crypto-db"
```

### Option 2: Supabase (추천 - 무료 PostgreSQL)

#### 1. Supabase 프로젝트 생성
1. https://supabase.com 접속 및 가입
2. 새 프로젝트 생성
3. Settings > Database > Connection string 복사

#### 2. 연결 문자열 설정
Supabase는 PostgreSQL을 사용하므로 연결 문자열 형식이 다릅니다:

```powershell
# Supabase 연결 문자열 형식
# postgresql+asyncpg://postgres:[YOUR-PASSWORD]@[PROJECT-REF].supabase.co:5432/postgres

# 예시
$env:DATABASE_URL = "postgresql+asyncpg://postgres:your_password@xxxxx.supabase.co:5432/postgres"
```

#### 3. 배포 시 환경 변수 전달
```powershell
# 환경 변수 설정
$env:DATABASE_URL = "postgresql+asyncpg://postgres:your_password@xxxxx.supabase.co:5432/postgres"
$env:BINANCE_API_KEY = "your_api_key"
$env:BINANCE_SECRET_KEY = "your_secret_key"

# 배포
.\deploy-gcp.ps1
```

#### 4. Supabase 연결 정보 찾기
1. Supabase 대시보드 접속
2. Settings > Database
3. Connection string 섹션에서 "URI" 복사
4. `postgres://` → `postgresql+asyncpg://`로 변경
5. 비밀번호 부분 `[YOUR-PASSWORD]`를 실제 비밀번호로 교체

**예시 변환:**
```
원본: postgres://postgres:[YOUR-PASSWORD]@db.xxxxx.supabase.co:5432/postgres
변환: postgresql+asyncpg://postgres:actual_password@db.xxxxx.supabase.co:5432/postgres
```

### Option 3: 다른 외부 MySQL 서버
- PlanetScale (무료 티어 제공)
- Aiven (무료 티어 제공)

연결 문자열 형식:
```
mysql+aiomysql://user:password@host:3306/database
```

### Option 3: 로컬 개발용 (Cloud Run에서 사용 불가)
로컬 개발 시에만 사용:
```
mysql+aiomysql://root:password@localhost:3306/crypto_trader
```

---

## 비용 예상 (월)

### Supabase 사용 시 (추천)

| 서비스 | 무료 한도 | 예상 비용 |
|--------|----------|----------|
| Cloud Run | 200만 요청/월, 360,000 GiB-초 | $0 (무료 한도 내) |
| Supabase | 500MB DB, 5GB 대역폭 | $0 (무료 티어) |
| **총합** | | **$0** ✅ |

### Cloud SQL 사용 시

| 서비스 | 무료 한도 | 예상 비용 |
|--------|----------|----------|
| Cloud Run | 200만 요청/월, 360,000 GiB-초 | $0 (무료 한도 내) |
| Cloud SQL (db-f1-micro) | 없음 | 약 $7.67/월 |
| **총합** | | **약 $8/월** |

> 💡 **비용 절감 팁**
> - Cloud SQL 대신 외부 무료 MySQL 서비스 사용 (PlanetScale, Aiven 등)
> - Cloud Run은 사용한 만큼만 과금 (무료 한도 충분)
> - 트래픽이 적으면 월 $0 가능

---

## 유용한 명령어

```powershell
# 로그 확인
gcloud run services logs read crypto-backend --region asia-northeast3

# 실시간 로그 스트리밍
gcloud run services logs tail crypto-backend --region asia-northeast3

# 서비스 목록
gcloud run services list --region asia-northeast3

# 서비스 상세 정보
gcloud run services describe crypto-backend --region asia-northeast3

# 서비스 삭제
gcloud run services delete crypto-backend --region asia-northeast3

# 환경 변수 업데이트
gcloud run services update crypto-backend `
  --region asia-northeast3 `
  --set-env-vars="NEW_VAR=value"

# 환경 변수 추가
gcloud run services update crypto-backend `
  --region asia-northeast3 `
  --update-env-vars="NEW_VAR=value"

# 메모리/CPU 업데이트
gcloud run services update crypto-backend `
  --region asia-northeast3 `
  --memory 2Gi `
  --cpu 2
```

---

## 트러블슈팅

### 1. WebSocket 연결 안 됨
Cloud Run은 WebSocket을 지원하지만 타임아웃이 있음 (최대 15분)
- 해결: 클라이언트에서 재연결 로직 구현 (이미 되어있음)

### 2. 콜드 스타트 느림
- 해결: `--min-instances=1` 옵션으로 항상 1개 인스턴스 유지 (비용 발생)
```powershell
gcloud run services update crypto-backend `
  --region asia-northeast3 `
  --min-instances=1
```

### 3. 메모리 부족
- 해결: `--memory=1Gi` 옵션으로 메모리 늘리기
```powershell
gcloud run services update crypto-backend `
  --region asia-northeast3 `
  --memory=2Gi
```

### 4. 컨테이너가 포트에서 리스닝하지 못함
**에러**: `The user-provided container failed to start and listen on the port defined provided by the PORT=8080 environment variable`

**원인**:
- Dockerfile이 하드코딩된 포트를 사용
- Cloud Run의 PORT 환경 변수를 읽지 않음

**해결**:
- Dockerfile이 이미 수정됨 (PORT 환경 변수 사용)
- 재배포 시 정상 작동해야 함

### 5. 데이터베이스 연결 실패
- Cloud SQL 사용 시: `--add-cloudsql-instances` 옵션 확인
- 외부 DB 사용 시: 방화벽 규칙 확인 (Cloud Run IP 허용)
- 연결 문자열 형식 확인: `mysql+aiomysql://user:pass@host:3306/db`
- **중요**: 환경 변수 없이도 애플리케이션은 시작되지만, DB 기능은 사용 불가

### 6. 빌드 실패
- Dockerfile 확인
- `.gcloudignore` 파일 확인 (불필요한 파일 제외)
- 로그 확인: `gcloud builds log --stream`

### 6. 환경 변수 적용 안 됨
- 환경 변수는 재배포 시에만 적용됨
- 업데이트 후 서비스 재시작 필요


