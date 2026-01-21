# Crypto AI Trader - Backend API

AI 기반 암호화폐 자동매매 시스템의 FastAPI 백엔드 서버입니다.

## 주요 기능

- **실시간 가격 조회**: Binance API 연동
- **AI 신호 생성**: Gemini AI 또는 기존 PyTorch 모델
- **거래 관리**: 주문 생성, 포지션 조회, 거래 내역
- **실시간 WebSocket**: 시장 데이터 및 AI 분석 스트리밍
- **거래 설정 관리**: 스탑로스, 익절, 포지션 크기 등

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
│   ├── models/              # DB 모델 (미구현)
│   └── services/
│       ├── ai_service.py    # AI 모델 서비스
│       ├── binance_service.py # Binance API 래퍼
│       └── gemini_service.py  # Gemini AI 서비스
├── requirements.txt         # Python 의존성
└── trading_settings.json    # 거래 설정 (런타임)
```

## 설치 및 실행

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

### 5. 실시간 데이터 (WebSocket)

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
