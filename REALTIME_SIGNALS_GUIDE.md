# 🎯 실시간 트레이딩 신호 시스템

## 📊 개요

프론트엔드에서 심볼을 검색하고, 실시간으로 AI와 기술적 분석을 결합한 매매 신호를 받아볼 수 있는 시스템입니다.

## 🚀 주요 기능

### 1. 심볼 검색 및 등록
- 바이낸스의 모든 USDT 페어 검색
- 인기 코인 우선 정렬 (BTC, ETH, BNB 등)
- 실시간으로 심볼 추가/제거

### 2. 실시간 신호 생성
- **5가지 신호 타입:**
  - 🚀 **강한 매수** (Strong Buy)
  - 📈 **매수** (Buy)
  - ⏸️ **횡보** (Neutral)
  - 📉 **매도** (Sell)
  - 🔴 **강한 매도** (Strong Sell)

- **5단계 신호 강도:**
  - 💪 매우 강함 (90-100%)
  - 💪 강함 (70-90%)
  - 💪 보통 (50-70%)
  - 💪 약함 (30-50%)
  - 💪 매우 약함 (0-30%)

### 3. 종합 분석
- **가중치 기반 전략** (60%)
  - RSI (20%)
  - MACD (25%)
  - Bollinger Bands (15%)
  - EMA Cross (20%)
  - Stochastic (10%)
  - Volume (10%)

- **AI 기반 전략** (40%)
  - LSTM 가격 예측
  - 기술적 지표와 결합

### 4. 실시간 업데이트
- WebSocket을 통한 실시간 신호 전송
- 30초마다 자동 업데이트
- 연결 끊김 시 자동 재연결

---

## 🛠️ 사용 방법

### 백엔드 실행

```bash
cd backend

# 의존성 설치
pip install -r requirements.txt

# 서버 시작
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 프론트엔드 실행

```bash
cd frontend

# 의존성 설치
npm install

# 개발 서버 시작
npm run dev
```

### 환경 변수 설정

```bash
# backend/.env
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
BINANCE_TESTNET=true  # 테스트넷 사용 시
```

---

## 📡 API 엔드포인트

### REST API

#### 심볼 검색
```http
GET /api/signals/symbols/search?query=BTC&limit=50
```

**응답:**
```json
{
  "success": true,
  "symbols": [
    {
      "symbol": "BTCUSDT",
      "baseAsset": "BTC",
      "quoteAsset": "USDT"
    }
  ],
  "total": 50
}
```

#### 심볼 추가
```http
POST /api/signals/symbols/add?symbol=BTCUSDT
```

**응답:**
```json
{
  "success": true,
  "message": "BTCUSDT added and monitoring started",
  "symbol": "BTCUSDT",
  "signal": {
    "symbol": "BTCUSDT",
    "price": 50000.0,
    "signal": "buy",
    "strength": "strong",
    "confidence": 0.75,
    "score": 0.45,
    "recommendation": {
      "action": "buy",
      "message": "📈 매수 신호. 신중한 진입을 권장합니다.",
      "action_text": "매수"
    }
  }
}
```

#### 심볼 제거
```http
DELETE /api/signals/symbols/BTCUSDT
```

#### 특정 심볼 신호 조회
```http
GET /api/signals/signal/BTCUSDT
```

#### 모든 신호 조회
```http
GET /api/signals/signals/all
```

#### 신호 업데이트 트리거
```http
POST /api/signals/signals/update
```

### WebSocket

#### 모든 활성 심볼의 실시간 신호
```javascript
const ws = new WebSocket('ws://localhost:8000/api/signals/ws/signals');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.signals); // 모든 심볼의 신호
};
```

#### 특정 심볼의 실시간 신호
```javascript
const ws = new WebSocket('ws://localhost:8000/api/signals/ws/signal/BTCUSDT');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.signal); // BTCUSDT 신호
};
```

---

## 🎨 프론트엔드 컴포넌트

### SignalsPage
메인 페이지. 심볼 검색, 신호 목록 표시, WebSocket 연결 관리

```typescript
import { SignalsPage } from '@/components/signals/SignalsPage';

// 사용
<SignalsPage />
```

### SymbolSearch
심볼 검색 및 등록 컴포넌트

```typescript
import { SymbolSearch } from '@/components/signals/SymbolSearch';

<SymbolSearch onSymbolAdd={(symbol) => console.log('Added:', symbol)} />
```

### SignalDisplay
개별 신호 표시 카드

```typescript
import { SignalDisplay } from '@/components/signals/SignalDisplay';

<SignalDisplay
  signal={signalData}
  onClick={() => console.log('Clicked')}
/>
```

---

## 🧪 테스트

### 수동 테스트

1. **심볼 추가**
   ```bash
   curl -X POST "http://localhost:8000/api/signals/symbols/add?symbol=BTCUSDT"
   ```

2. **신호 조회**
   ```bash
   curl "http://localhost:8000/api/signals/signal/BTCUSDT"
   ```

3. **WebSocket 연결 (JavaScript)**
   ```javascript
   const ws = new WebSocket('ws://localhost:8000/api/signals/ws/signals');
   ws.onmessage = (e) => console.log(JSON.parse(e.data));
   ```

### 프론트엔드 테스트

1. 브라우저에서 `http://localhost:5173/signals` 접속
2. 검색창에 "BTC" 입력
3. BTCUSDT 선택하여 추가
4. 실시간 신호 확인

---

## 📊 신호 해석

### 신호 점수 (Score)
- **-1.0 ~ -0.6**: 강한 매도
- **-0.6 ~ -0.3**: 매도
- **-0.3 ~ 0.3**: 횡보
- **0.3 ~ 0.6**: 매수
- **0.6 ~ 1.0**: 강한 매수

### 신뢰도 (Confidence)
- 여러 지표의 일치도
- 높을수록 신뢰할 수 있는 신호

### 강도 (Strength)
- 신호의 강도를 5단계로 표시
- 신뢰도와 일치도를 기반으로 계산

---

## 🔧 커스터마이징

### 가중치 조정

`backend/app/services/weighted_strategy.py`에서 가중치 수정:

```python
default_weights = {
    'rsi': 0.20,
    'macd': 0.25,
    'bollinger': 0.15,
    'ema_cross': 0.20,
    'stochastic': 0.10,
    'volume': 0.10
}
```

### 업데이트 주기 조정

`backend/app/api/signals.py`의 WebSocket 핸들러에서:

```python
# 30초 -> 10초로 변경
await asyncio.sleep(10)
```

### AI 모델 사용/미사용

```python
# signal_service.py
signal_service = RealTimeSignalService(binance, use_ai=False)
```

---

## ⚠️ 주의사항

1. **API 레이트 리밋**
   - 바이낸스 API 호출 제한 준수
   - 너무 많은 심볼을 동시에 모니터링하지 말 것 (권장: 10개 이하)

2. **메모리 사용**
   - 각 심볼마다 과거 데이터 캐싱
   - 사용하지 않는 심볼은 제거

3. **WebSocket 연결**
   - 네트워크 불안정 시 자동 재연결
   - 여러 탭에서 동시 접속 가능

4. **신호 지연**
   - 실시간이지만 몇 초의 지연 있을 수 있음
   - 급격한 가격 변동 시 신호가 늦을 수 있음

---

## 🐛 문제 해결

### WebSocket 연결 실패

```bash
# CORS 설정 확인
# backend/app/main.py
allow_origins=["http://localhost:3000", "http://localhost:5173"]
```

### 신호가 업데이트되지 않음

```bash
# 로그 확인
tail -f backend/logs/*.log

# 수동으로 업데이트 트리거
curl -X POST "http://localhost:8000/api/signals/signals/update"
```

### 심볼 추가 실패

```bash
# 심볼 형식 확인 (대문자, USDT 페어)
# 올바른 예: BTCUSDT
# 잘못된 예: btcusdt, BTC/USDT, BTCUSD
```

---

## 📈 성능 최적화

1. **캐싱 활용**
   - 신호는 자동으로 캐싱
   - 불필요한 API 호출 최소화

2. **병렬 처리**
   - 여러 심볼 동시 처리
   - asyncio 활용

3. **데이터베이스 사용**
   - 과거 데이터는 SQLite에 저장
   - 중복 조회 방지

---

## 🔮 향후 개선 계획

- [ ] 차트 통합 (TradingView)
- [ ] 알림 기능 (텔레그램, 이메일)
- [ ] 신호 이력 추적
- [ ] 백테스팅 결과 표시
- [ ] 사용자별 커스텀 가중치
- [ ] 멀티 타임프레임 분석
- [ ] 신호 정확도 통계

---

## 📚 참고 자료

- [FastAPI WebSocket](https://fastapi.tiangolo.com/advanced/websockets/)
- [React WebSocket](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
- [Binance API](https://binance-docs.github.io/apidocs/spot/en/)
- [기술적 분석 지표](https://www.investopedia.com/technical-analysis-4689657)

---

**마지막 업데이트**: 2026-01-21
