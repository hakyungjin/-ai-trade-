# AI Trader Architecture Guide

## 프로젝트 구조

```
backend/app/
├── api/
│   ├── crypto/                 # Crypto-specific APIs
│   │   ├── __init__.py
│   │   ├── coins.py            # Coin monitoring endpoints
│   │   └── realtime_crypto.py  # WebSocket for crypto
│   │
│   ├── stocks/                 # Stock-specific APIs
│   │   ├── __init__.py
│   │   ├── stocks.py           # Stock monitoring endpoints
│   │   └── realtime_stocks.py  # WebSocket for stocks
│   │
│   └── trading.py              # Common trading endpoints
│
├── models/
│   ├── crypto/                 # Crypto data models
│   │   ├── __init__.py
│   │   └── coin.py
│   │
│   └── stocks/                 # Stock data models
│       ├── __init__.py
│       └── stock.py
│
├── services/
│   ├── crypto/                 # Crypto-specific services
│   │   ├── __init__.py
│   │   ├── coin_service.py
│   │   └── binance_service.py  # Binance API wrapper
│   │
│   ├── stocks/                 # Stock-specific services
│   │   ├── __init__.py
│   │   ├── stock_service.py
│   │   ├── alpha_vantage_service.py  # Alpha Vantage API
│   │   └── finnhub_service.py       # Finnhub API (선택)
│   │
│   └── common/                 # Shared services
│       ├── __init__.py
│       ├── model_training_service.py   # XGBoost, LSTM 학습
│       ├── technical_indicators.py     # RSI, MACD, Bollinger
│       ├── weighted_strategy.py        # 가중치 기반 신호
│       └── gemini_service.py          # AI 분석
│
└── main.py                     # FastAPI app & router registration

ai-model/
├── data/
│   ├── crypto/                 # Crypto training data
│   │   └── *.csv
│   └── stocks/                 # Stock training data
│       └── *.csv
│
└── models/
    ├── crypto/                 # Crypto trained models
    │   ├── xgboost_BTCUSDT_*.joblib
    │   └── lstm_BTCUSDT_*.pt
    └── stocks/                 # Stock trained models
        ├── xgboost_AAPL_*.joblib
        └── lstm_AAPL_*.pt
```

---

## API 엔드포인트 설계

### Crypto APIs
```
POST   /api/v1/crypto/coins/add-monitoring/{symbol}
GET    /api/v1/crypto/coins/monitoring
POST   /api/v1/crypto/coins/remove/{id}
WS     /api/realtime/crypto/ws/{symbol}
POST   /api/ai/crypto/predict
```

### Stock APIs
```
POST   /api/v1/stocks/add-monitoring/{symbol}
GET    /api/v1/stocks/monitoring
POST   /api/v1/stocks/remove/{id}
WS     /api/realtime/stocks/ws/{symbol}
POST   /api/ai/stocks/predict
```

### Common Trading APIs (공용)
```
POST   /api/trading/order              # 주문 (암호/주식 구분)
GET    /api/trading/balance            # 잔고 (계정별)
GET    /api/trading/positions          # 포지션
GET    /api/trading/history            # 거래 내역
```

---

## 데이터베이스 모델

### Crypto (기존)
```
- Coin
- CoinStatistics
- CoinAnalysisConfig
- CoinPriceHistory
- MarketCandles
```

### Stocks (신규)
```
- Stock
- StockStatistics
- StockAnalysisConfig
- StockPriceHistory
- StockCandles
```

### Common (공용)
```
- Trade              # 모든 거래 기록 (crypto/stock 구분)
- StrategyWeight     # 가중치 설정
- VectorPattern      # 패턴 학습
```

---

## 설정 관리 (.env)

```env
# ===== Crypto =====
BINANCE_API_KEY=...
BINANCE_SECRET_KEY=...
BINANCE_TESTNET=true

# ===== Stocks =====
ALPHA_VANTAGE_API_KEY=...
FINNHUB_API_KEY=...        # Optional

# ===== AI =====
GEMINI_API_KEY=...
MODEL_PATH=./ai-model/models

# ===== Database =====
DATABASE_URL=mysql+aiomysql://...
```

---

## 핵심 고려사항

### 1. **마켓 타입 구분**
```python
# Trade 모델에 market_type 필드 추가
class Trade(Base):
    market_type: Literal["crypto", "stock"]  # 구분자
    symbol: str       # BTCUSDT, AAPL
```

### 2. **데이터 수집 일정 차이**
| 항목 | Crypto | Stock |
|------|--------|-------|
| 시장시간 | 24/7 | 미국 동장 9:30 ~ 16:00 |
| 최소 봉 | 1분 | 1분 |
| 데이터 주기 | 5분, 15분, 1시간 | 15분, 1시간, 일봉 |

### 3. **API 비용 관리**
- **Alpha Vantage**: 무료(5req/min), 유료 선택
- **Finnhub**: 무료(60req/min)
- **Binance**: 무료 (권장)

### 4. **모델 학습 차이**
- **Crypto**: 변동성 높음 (짧은 timeframe 추천)
- **Stock**: 상대적 안정 (일봉/주봉 추천)

### 5. **백그라운드 작업 스케줄링**
```python
# APScheduler로 시간대별 작업 분리
- 매 5분: Crypto 데이터 수집 (24/7)
- 미국 장 중: Stock 데이터 수집 (9:30-16:00)
- 매일 자정: 모델 재학습
- 매주 금요일: 심층 분석
```

### 6. **포트폴리오 관리**
```python
class Portfolio(Base):
    user_id: int
    crypto_holdings: List[Dict]  # {symbol, quantity, avg_price}
    stock_holdings: List[Dict]   # {symbol, quantity, avg_price}
    total_value: float           # 합계
    allocation: Dict             # crypto 40%, stock 60% 등
```

### 7. **리스크 관리**
```python
class RiskSettings(Base):
    market_type: Literal["crypto", "stock"]
    max_position_size: float
    max_daily_loss: float
    max_correlation: float  # 같은 시장 내 상관계수 제한
    leverage: float         # 주식: 1.0, 선물: 1-10배
```

### 8. **실시간 분석 우선순위**
```
높음: Crypto (24/7, 변동성 큼)
중간: Stock (시장시간만, 기관 거래 영향)
낮음: Gemini AI (비용 고려)
```

---

## 마이그레이션 전략

1. **Phase 1**: 코인 기능 완성 및 테스트
2. **Phase 2**: 공통 서비스 추출 (모델 학습, 기술지표)
3. **Phase 3**: 주식 모델/서비스 구현
4. **Phase 4**: 포트폴리오 통합 기능
5. **Phase 5**: 고급 리스크 관리

---

## 배포 전략

### 로컬 테스트
```bash
# 터미널 1: 백엔드
uvicorn app.main:app --reload

# 터미널 2: 프론트엔드
npm run dev
```

### 프로덕션
- **Crypto**: Cloud Run (기존)
- **Stocks**: 동일한 Cloud Run 인스턴스
- **DB**: MySQL (공유)
- **AI Model**: GCS 저장소

---

## 시작 체크리스트

- [ ] Alpha Vantage API Key 발급
- [ ] 주식 DB 모델 생성
- [ ] stock_service.py 작성
- [ ] StockDataService 구현
- [ ] /api/v1/stocks 엔드포인트 작성
- [ ] 모델 학습 파이프라인 (crypto와 동일)
- [ ] WebSocket 추가 (/api/realtime/stocks)
- [ ] 프론트엔드 UI 통합 (Crypto + Stocks 탭)
