# Supabase 설정 가이드

## 🚀 빠른 시작

### 1. Supabase 프로젝트 생성

1. https://supabase.com 접속
2. "Start your project" 클릭
3. GitHub 계정으로 로그인 (또는 이메일 가입)
4. "New Project" 클릭
5. 프로젝트 정보 입력:
   - **Name**: crypto-ai-trader (원하는 이름)
   - **Database Password**: 강력한 비밀번호 설정 (기억해두세요!)
   - **Region**: Northeast Asia (Seoul) - 한국과 가까운 리전
6. "Create new project" 클릭 (약 2분 소요)

### 2. 연결 정보 가져오기

1. 프로젝트 대시보드에서 **Settings** (왼쪽 메뉴) 클릭
2. **Database** 메뉴 클릭
3. **Connection string** 섹션에서 **URI** 복사

**연결 문자열 예시:**
```
postgres://postgres:[YOUR-PASSWORD]@db.xxxxx.supabase.co:5432/postgres
```

### 3. 연결 문자열 변환

Supabase는 PostgreSQL을 사용하므로 연결 문자열을 변환해야 합니다:

**변환 규칙:**
- `postgres://` → `postgresql+asyncpg://`
- `[YOUR-PASSWORD]` → 실제 비밀번호로 교체

**예시:**
```powershell
# 원본 (Supabase에서 복사)
postgres://postgres:[YOUR-PASSWORD]@db.abcdefgh.supabase.co:5432/postgres

# 변환 후 (PowerShell에서 사용)
$env:DATABASE_URL = "postgresql+asyncpg://postgres:your_actual_password@db.abcdefgh.supabase.co:5432/postgres"
```

### 4. 환경 변수 설정 및 배포

#### 방법 1: Supabase 전용 배포 스크립트 사용 (추천)

```powershell
# Supabase 비밀번호와 Binance API 키로 한 번에 배포
.\deploy-with-supabase.ps1 `
  -SupabasePassword "your_supabase_password" `
  -BinanceApiKey "your_binance_api_key" `
  -BinanceSecretKey "your_binance_secret_key"
```

#### 방법 2: 수동 환경 변수 설정

```powershell
# 1. Supabase 연결 문자열 설정 (비밀번호만 교체)
$env:DATABASE_URL = "postgresql+asyncpg://postgres:YOUR_PASSWORD@db.vmiinfjxpnoevsehhzey.supabase.co:5432/postgres"

# 2. Binance API 키 설정
$env:BINANCE_API_KEY = "your_binance_api_key"
$env:BINANCE_SECRET_KEY = "your_binance_secret_key"
$env:BINANCE_TESTNET = "false"

# 3. 배포 실행
.\deploy-gcp.ps1
```

> 💡 **현재 Supabase 프로젝트**: `db.vmiinfjxpnoevsehhzey.supabase.co`

## 📊 Supabase 무료 티어 제한

| 항목 | 제한 |
|------|------|
| 데이터베이스 크기 | 500 MB |
| 대역폭 | 5 GB/월 |
| API 요청 | 무제한 |
| 동시 연결 | 60개 |

> 💡 **참고**: 소규모 프로젝트에는 충분합니다. 필요 시 유료 플랜으로 업그레이드 가능.

## 🔒 보안 설정

### 1. 비밀번호 보호

Supabase 비밀번호는 환경 변수로만 관리:
- 코드에 하드코딩하지 않기
- `.env` 파일을 `.gitignore`에 추가 (이미 되어있음)

### 2. 연결 풀링

Supabase는 자동으로 연결 풀링을 관리합니다. 추가 설정 불필요.

## 🛠️ 트러블슈팅

### 연결 실패

**에러**: `connection refused` 또는 `authentication failed`

**해결 방법:**
1. 비밀번호 확인 (Supabase 대시보드 > Settings > Database)
2. 연결 문자열 형식 확인 (`postgresql+asyncpg://` 사용)
3. 방화벽 설정 확인 (Supabase는 기본적으로 모든 IP 허용)

### 테이블이 생성되지 않음

**해결 방법:**
1. 애플리케이션 로그 확인:
   ```powershell
   gcloud run services logs read crypto-backend --region asia-northeast3
   ```
2. `init_db()` 함수가 실행되는지 확인
3. Supabase 대시보드 > Table Editor에서 테이블 확인

### 연결 타임아웃

**해결 방법:**
- `database.py`에서 이미 `connect_timeout=10` 설정됨
- Supabase 리전이 너무 멀면 리전 변경 고려

## 📝 유용한 명령어

### Supabase 대시보드에서 확인

1. **Table Editor**: 생성된 테이블 확인
2. **SQL Editor**: 직접 SQL 쿼리 실행
3. **Database > Connection Pooling**: 연결 상태 확인

### 로컬에서 테스트

```powershell
# 환경 변수 설정
$env:DATABASE_URL = "postgresql+asyncpg://postgres:password@db.xxxxx.supabase.co:5432/postgres"

# 백엔드 실행
cd backend
python -m uvicorn app.main:app --reload
```

## 🔄 MySQL에서 PostgreSQL로 마이그레이션

현재 코드는 MySQL과 PostgreSQL을 모두 지원합니다. 연결 문자열만 변경하면 자동으로 감지됩니다:

- **MySQL**: `mysql+aiomysql://...`
- **PostgreSQL**: `postgresql+asyncpg://...`

SQLAlchemy가 자동으로 적절한 드라이버를 사용합니다.

## 💰 비용

- **무료 티어**: $0/월 (500MB, 5GB 대역폭)
- **Pro 플랜**: $25/월 (8GB, 50GB 대역폭)

소규모 프로젝트는 무료 티어로 충분합니다!

