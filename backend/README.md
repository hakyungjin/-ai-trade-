# Crypto AI Trader - Backend API

AI 기반 암호화폐 자동매매 시스템의 FastAPI 백엔드 서버입니다.

## 주요 기능

- **실시간 가격 조회**: Binance API 연동
- **AI 신호 생성**: Gemini AI 또는 기존 PyTorch 모델
- **거래 관리**: 주문 생성, 포지션 조회, 거래 내역
- **실시간 WebSocket**: 시장 데이터 및 AI 분석 스트리밍
- **거래 설정 관리**: 스탑로스, 익절, 포지션 크기 등
- **코인 모니터링**: 모니터링 중인 코인 관리 및 통계 조회

## 프로젝트 구조

```
backend/
├── app/
│   ├── main.py              # FastAPI 진입점
│   ├── config.py            # 설정 (환경변수)
│   ├── api/
│   │   ├── ai_signal.py     # AI 예측 엔드포인트
│   │   ├── trading.py       # 거래 엔드포인트
│   │   ├── realtime.py      # WebSocket 스트리밍
│   │   └── settings.py      # 거래 설정 엔드포인트
│   ├── models/              # DB 모델
│   │   ├── coin.py          # 코인 모니터링 모델
│   │   ├── market_data.py   # 시장 데이터 모델
│   │   └── trade.py         # 거래 모델
│   └── services/
│       ├── ai_service.py    # AI 모델 서비스
│       ├── binance_service.py # Binance API 래퍼
│       └── gemini_service.py  # Gemini AI 서비스
├── requirements.txt         # Python 의존성
└── trading_settings.json    # 거래 설정 (런타임)
```

## 설치 및 실행

### ⚡ 빠른 시작 (권장)

루트 디렉토리에서 한 명령어로 모든 서버 시작:

```bash
# 백엔드 + 프론트엔드 동시 실행
npm run dev

# 또는 개별 실행
npm run backend        # 백엔드만
npm run frontend       # 프론트엔드만
```

### 📦 설치

```bash
# 처음 한 번만 - 모든 의존성 설치
npm run install:all

# 또는 개별 설치
npm run backend:install   # 백엔드 Python 패키지
npm run frontend:install  # 프론트엔드 npm 패키지
```

### 1. 가상환경 생성 및 활성화

```bash
# 가상환경 생성 (한 번만)
python -m venv .venv

# 활성화 (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 활성화 (Windows CMD)
.venv\Scripts\activate.bat

# 활성화 (macOS/Linux)
source .venv/bin/activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 환경변수 설정 (.env 파일)

백엔드 루트에 `.env` 파일 생성:

```env
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
BINANCE_TESTNET=true

GEMINI_API_KEY=your_gemini_api_key

MODEL_PATH=../ai-model/models/price_predictor.pt
```

**주의**: `.env` 파일은 Git에 커밋하지 마세요. 대신 `.env.example` 사용.

### 4. 서버 실행

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- `--reload`: 파일 변경시 자동 재시작 (개발용)
- `--host 0.0.0.0`: 모든 네트워크 인터페이스 수신
- `--port 8000`: 포트 번호

## API 문서

서버 시작 후 브라우저에서 접속:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API 엔드포인트

### 1. 기본 확인

#### GET `/`
루트 API 정보
```bash
curl http://localhost:8000/
```

**응답:**
```json
{
  "message": "Crypto AI Trader API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

#### GET `/health`
헬스 체크 및 설정 상태
```bash
curl http://localhost:8000/health
```

**응답:**
```json
{
  "status": "healthy",
  "testnet": true,
  "binance_configured": true,
  "gemini_configured": false
}
```

---

### 2. 거래 (Trading)

#### GET `/api/trading/price/{symbol}`
현재가 조회 (예: BTCUSDT, ETHUSDT)
```bash
curl http://localhost:8000/api/trading/price/BTCUSDT
```

#### GET `/api/trading/balance`
계좌 잔고 조회
```bash
curl http://localhost:8000/api/trading/balance
```

#### GET `/api/trading/positions`
현재 포지션 조회
```bash
curl http://localhost:8000/api/trading/positions
```

#### POST `/api/trading/order`
주문 생성 (매수/매도)

**요청:**
```bash
curl -X POST http://localhost:8000/api/trading/order \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "side": "BUY",
    "quantity": 0.01,
    "order_type": "MARKET",
    "stop_loss": 40000,
    "take_profit": 45000
  }'
```

#### GET `/api/trading/history`
거래 내역 조회
```bash
curl "http://localhost:8000/api/trading/history?symbol=BTCUSDT&limit=50"
```

#### DELETE `/api/trading/order/{symbol}/{order_id}`
주문 취소

---

### 3. AI 신호 (AI Signal)

#### POST `/api/ai/predict`
AI 예측 신호 조회

**요청:**
```bash
curl -X POST http://localhost:8000/api/ai/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "timeframe": "1h"
  }'
```

**응답:**
```json
{
  "symbol": "BTCUSDT",
  "signal": "BUY",
  "confidence": 0.75,
  "predicted_direction": "UP",
  "current_price": 43500.0,
  "analysis": "강한 상승 신호. RSI 과매도 상태에서 회복. MACD 양수."
}
```

#### GET `/api/ai/market-analysis/{symbol}`
시장 종합 분석
```bash
curl "http://localhost:8000/api/ai/market-analysis/BTCUSDT?timeframe=1h"
```

#### GET `/api/ai/signals`
현재 활성화된 AI 신호 목록
```bash
curl http://localhost:8000/api/ai/signals
```

#### POST `/api/ai/parse-prompt`
자연어 프롬프트를 거래 규칙으로 변환

**요청:**
```bash
curl -X POST http://localhost:8000/api/ai/parse-prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": "스탑로스 3%로 설정하고 익절은 5%로 해줘"}'
```

#### POST `/api/ai/gemini/analyze` (Gemini 전용)
Gemini AI 상세 차트 분석
```bash
curl -X POST http://localhost:8000/api/ai/gemini/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSDT", "timeframe": "1h"}'
```

#### GET `/api/ai/gemini/history`
Gemini 분석 히스토리
```bash
curl "http://localhost:8000/api/ai/gemini/history?limit=20"
```

#### GET `/api/ai/gemini/market-sentiment`
전체 시장 심리 분석 (주요 코인 종합)
```bash
curl http://localhost:8000/api/ai/gemini/market-sentiment
```

---

### 5. 코인 모니터링 (Coin Monitoring)

#### GET `/api/v1/coins/monitoring`
모니터링 중인 코인 목록 조회
```bash
curl http://localhost:8000/api/v1/coins/monitoring
```

**응답:**
```json
{
  "success": true,
  "total": 2,
  "data": [
    {
      "id": 1,
      "symbol": "BTCUSDT",
      "base_asset": "BTC",
      "quote_asset": "USDT",
      "is_monitoring": true,
      "current_price": 43500.0,
      "price_change_24h": 2.5,
      "volume_24h": 1500000000.0,
      "market_cap": 850000000000.0,
      "candle_count": 10000,
      "monitoring_timeframes": ["1h", "4h", "1d"]
    }
  ]
}
```

#### POST `/api/v1/coins/add-monitoring/{symbol}`
모니터링 코인 추가
```bash
curl -X POST "http://localhost:8000/api/v1/coins/add-monitoring/BTCUSDT?timeframes=1h&timeframes=4h"
```

#### POST `/api/v1/coins/add`
코인 추가 (수동)
```bash
curl -X POST http://localhost:8000/api/v1/coins/add \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "ETHUSDT",
    "base_asset": "ETH",
    "quote_asset": "USDT",
    "is_monitoring": true,
    "full_name": "Ethereum"
  }'
```

#### GET `/api/v1/coins/list`
모든 코인 목록 조회
```bash
curl http://localhost:8000/api/v1/coins/list
```

#### GET `/api/v1/coins/{coin_id}/stats`
코인별 통계 조회
```bash
curl http://localhost:8000/api/v1/coins/1/stats
```

#### GET `/api/v1/coins/{coin_id}/config`
코인별 분석 설정 조회
```bash
curl http://localhost:8000/api/v1/coins/1/config
```

---

### 4. 설정 (Settings)

#### GET `/api/settings/`
현재 거래 설정 조회
```bash
curl http://localhost:8000/api/settings/
```

**응답:**
```json
{
  "default_stop_loss": 0.02,
  "default_take_profit": 0.05,
  "max_position_size": 100.0,
  "auto_trading_enabled": false,
  "prediction_threshold": 0.6,
  "max_daily_trades": 10,
  "max_daily_loss": 0.1,
  "trailing_stop_enabled": false,
  "trailing_stop_percent": 0.01,
  "telegram_enabled": false,
  "telegram_chat_id": null
}
```

#### PUT `/api/settings/`
전체 설정 업데이트
```bash
curl -X PUT http://localhost:8000/api/settings/ \
  -H "Content-Type: application/json" \
  -d '{
    "default_stop_loss": 0.03,
    "default_take_profit": 0.08,
    "auto_trading_enabled": true,
    "prediction_threshold": 0.7
  }'
```

#### PATCH `/api/settings/`
설정 부분 업데이트
```bash
curl -X PATCH http://localhost:8000/api/settings/ \
  -H "Content-Type: application/json" \
  -d '{"auto_trading_enabled": true, "prediction_threshold": 0.65}'
```

#### POST `/api/settings/reset`
설정 초기화
```bash
curl -X POST http://localhost:8000/api/settings/reset
```

---

### 6. 실시간 데이터 (WebSocket)

#### WS `/api/realtime/ws/market/{symbol}`
특정 심볼 실시간 스트림 (가격 + AI 분석)

**JavaScript 예제:**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/realtime/ws/market/BTCUSDT');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
  
  // type: "price" - 가격 업데이트 (5초마다)
  // type: "analysis" - AI 분석 (60초마다)
  // type: "error" - 에러
};

ws.onerror = (error) => console.error(error);
ws.onclose = () => console.log("disconnected");
```

#### WS `/api/realtime/ws/multi`
여러 심볼 동시 구독 (고급)

**JavaScript 예제:**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/realtime/ws/multi');

// 심볼 구독
ws.send(JSON.stringify({
  action: 'subscribe',
  symbols: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
}));

// 심볼 구독 해제
ws.send(JSON.stringify({
  action: 'unsubscribe',
  symbols: ['ETHUSDT']
}));

// 즉시 분석 요청
ws.send(JSON.stringify({
  action: 'analyze',
  symbol: 'BTCUSDT'
}));

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
};
```

---

## 데이터베이스 마이그레이션

### 빠른 시작

```bash
# 모든 마이그레이션 적용
cd backend
alembic upgrade head
```

### 자동 마이그레이션 워크플로우

모델 변경 후 새 마이그레이션 파일 자동 생성:

```bash
# 1. 모델 파일 수정 (app/models/*.py)
# 2. 마이그레이션 생성
cd backend
alembic revision --autogenerate -m "Add new table or column description"

# 3. 생성된 파일 검토 (alembic/versions/xxx_*.py)
# 4. 적용
alembic upgrade head
```

### 주요 마이그레이션 명령어

| 명령어 | 설명 |
|--------|------|
| `alembic upgrade head` | 최신 버전까지 마이그레이션 |
| `alembic downgrade -1` | 이전 버전으로 롤백 |
| `alembic current` | 현재 버전 확인 |
| `alembic history` | 마이그레이션 히스토리 |
| `alembic revision --autogenerate -m "message"` | 변경사항 자동 감지 후 마이그레이션 생성 |

### npm 스크립트 명령어 (루트에서 실행)

| 명령어 | 설명 |
|--------|------|
| `npm run migration:upgrade` | 마이그레이션 적용 (npm run dev 대신 사용 시) |
| `npm run migration:downgrade` | 이전 버전으로 롤백 |
| `npm run migration:new "설명"` | 새 마이그레이션 생성 |

### 마이그레이션 히스토리

| 마이그레이션 | 설명 | 상태 |
|-------------|------|------|
| `2a5a9ea01389_initial_migration` | 초기 테이블 생성 (trades, market_candles) | ✅ |
| `7d936f1f64d7_add_ai_training_data` | AI 학습 데이터 테이블 추가 | ✅ |
| `optimize_candle_v1` | market_candles 인덱스 최적화 | ✅ |
| `add_coin_metadata_v1` | 코인 모니터링 테이블 (coins, coin_statistics, coin_analysis_configs, coin_price_history) | ✅ |

### 코인 모니터링 데이터베이스 구조

코인 모니터링 기능을 위한 데이터베이스 테이블:

#### 1. `coins` - 코인 기본 정보
- 코인 심볼, 기본 정보, 모니터링 상태
- 최신 시세 정보 (캐시)
- 모니터링 타임프레임 설정

#### 2. `coin_statistics` - 코인별 통계
- 캔들 통계 (총 개수, 타임프레임별 개수)
- 분석 통계 (신호 개수, 신호 타입별 개수)
- 성능 지표 (평균 확신도, 승률 등)

#### 3. `coin_analysis_configs` - 코인별 분석 설정
- 기술적 지표 사용 여부 (RSI, MACD, Bollinger 등)
- AI 분석 설정 (Gemini, 로컬 모델, 벡터 패턴)
- 신호 임계값 설정

#### 4. `coin_price_history` - 코인 시세 이력
- 시세 이력 저장 (캐시용)
- 가격 변동 추적

**모델 파일**: `app/models/coin.py`

### 트러블슈팅

#### 마이그레이션 오류: "table already exists"
```bash
# 기존 테이블 확인
alembic current

# 강제로 최신 버전으로 표시 (이미 수동으로 생성된 경우)
alembic stamp head
```

#### 마이그레이션 감지 안됨
Alembic auto-detect가 모든 변경사항을 감지하지 못하는 경우:
```bash
# 1. alembic/env.py에 모델 임포트 확인
# from app.models import trade, market_data, coin  # coin 모델 포함 확인

# 2. 수동으로 마이그레이션 파일 생성 후 수정
alembic revision -m "Manual migration description"
# alembic/versions/xxx_manual_migration_description.py 파일 수정
```

**참고**: 코인 모니터링 모델(`coin.py`)을 사용하려면 `alembic/env.py`에 `coin` 모델이 임포트되어 있어야 합니다.

#### MySQL 문법 오류
MySQL 특화 문법으로 마이그레이션 작성:
- `IF NOT EXISTS` 사용 금지 (DROP, CREATE INDEX에서)
- `VARCHAR` 길이 명시 필수
- `COLLATE utf8mb4_unicode_ci` 사용

### 개발 환경 데이터 초기화

```bash
# 모든 테이블 제거 (주의!)
alembic downgrade base

# 다시 생성
alembic upgrade head
```

### 더 자세한 정보

- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - 상세 마이그레이션 가이드
- [CANDLE_DB_OPTIMIZATION.md](CANDLE_DB_OPTIMIZATION.md) - 데이터베이스 최적화 전략

---

## 주요 설정 (config.py)

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `BINANCE_API_KEY` | "" | 바이낸스 API 키 |
| `BINANCE_SECRET_KEY` | "" | 바이낸스 시크릿 키 |
| `BINANCE_TESTNET` | true | 테스트넷 사용 여부 |
| `GEMINI_API_KEY` | "" | Gemini AI API 키 |
| `MODEL_PATH` | ../ai-model/models/price_predictor.pt | PyTorch 모델 경로 |
| `PREDICTION_THRESHOLD` | 0.6 | 예측 신뢰도 기준 |

---

## 오류 해결

### pip install 실패 (네트워크)
인터넷 연결 확인 후 재시도:
```bash
pip install --upgrade pip
pip install -r requirements.txt -v
```

### Binance API 오류
- 테스트넷이 유효한지 확인
- API 키/시크릿 검증
- 시간 동기화 확인

### Gemini API 오류
- API 키 유효성 확인
- API 할당량 검증

---

## 개발

### 코드 스타일
- Python 3.9+
- FastAPI 비동기 지원
- Pydantic V2 데이터 검증

### 로깅
기본으로 stdout에 로깅됩니다. 프로덕션에서는 파일 로깅 추가 권장.

---

## 다음 단계

- [ ] 데이터베이스 모델 구현 (거래 히스토리 영구 저장)
- [ ] 인증 (JWT)
- [ ] Rate Limiting
- [ ] Prometheus 메트릭
- [ ] Docker 컨테이너화
- [ ] 배포 (AWS/GCP)

---

## 라이센스

MIT
