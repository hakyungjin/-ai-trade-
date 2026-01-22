# AI 학습용 데이터 DB 시스템 구축 완료

## 📋 개요

AI 모델 학습을 위한 데이터를 체계적으로 관리하고 저장하는 데이터베이스 구조가 완성되었습니다.

## 🗄️ 데이터베이스 테이블 구조

### 1. **market_candles** - 시장 캔들 데이터
```python
- symbol: 거래쌍 (예: BTCUSDT)
- timeframe: 시간프레임 (1m, 5m, 15m, 1h, 4h, 1d)
- open_time: 캔들 시작 시간
- open, high, low, close: OHLC 가격
- volume: 거래량
- quote_volume: 거래대금
- trades_count: 거래 횟수
```

**인덱스:**
- `idx_candle_symbol_time`: symbol, timeframe, open_time 조합 인덱스로 빠른 조회

---

### 2. **technical_indicators** - 기술적 지표
```python
# 이동평균선
- sma_5, sma_10, sma_20, sma_50
- ema_12, ema_26

# 모멘텀 지표
- rsi_14: 상대강도지수
- macd, macd_signal, macd_histogram

# 변동성 지표
- bb_upper, bb_middle, bb_lower: 볼린저 밴드
- bb_width: 밴드 폭
- atr_14: 평균진정범위

# 거래량
- volume_ratio: 평균 대비 거래량
```

---

### 3. **ai_training_data** - AI 학습용 데이터 (핵심!)
```python
# 입력 데이터
- symbol: 거래쌍
- timeframe: 시간프레임
- timestamp: 데이터 시점
- features: JSON 형식의 모든 기술적 지표
- current_price: 현재 가격

# 레이블 데이터 (예측 대상)
- future_price_1h: 1시간 후 가격
- future_price_4h: 4시간 후 가격
- future_price_24h: 24시간 후 가격

# 분류 레이블
- label_1h, label_4h, label_24h: 0(하락), 1(횡보), 2(상승)
  * 0: 가격 하락 (< -1%)
  * 1: 횡보 (-1% ~ 1%)
  * 2: 가격 상승 (> 1%)
```

---

### 4. **ai_analyses** - AI 분석 결과
```python
- symbol: 거래쌍
- timestamp: 분석 시간
- signal: BUY, SELL, HOLD
- confidence: 신뢰도 (0~1)
- direction: UP, DOWN, NEUTRAL
- analysis_text: 분석 내용
- source: 분석 출처 (gemini, pytorch, rule_based)
- model_version: 사용한 모델 버전
- raw_response: AI 전체 응답 (JSON)
- price_at_analysis: 분석 시점 가격
```

---

### 5. **signal_history** - 매매 신호 히스토리
```python
- symbol: 거래쌍
- timestamp: 신호 발생 시간
- signal: BUY, SELL, HOLD
- confidence: 신뢰도
- source: 신호 출처
- price_at_signal: 신호 시점 가격
- executed: 실행 여부

# 결과 추적
- price_after_1h, 4h, 24h: 각 시간 후 가격
- result: profit, loss, neutral
- profit_percent: 수익률
```

---

## 🔧 마이그레이션 시스템 (Alembic)

### 설정된 마이그레이션
1. **Initial migration** - 초기 구조
2. **Add AI training data models** - 학습 데이터 테이블 추가

### 마이그레이션 명령어
```bash
# 마이그레이션 적용
.venv\Scripts\alembic upgrade head

# 마이그레이션 상태 확인
.venv\Scripts\alembic current

# 새 마이그레이션 생성 (모델 변경 후)
.venv\Scripts\alembic revision --autogenerate -m "설명"

# 마이그레이션 롤백
.venv\Scripts\alembic downgrade -1
```

---

## 📚 서비스 사용법

### 1. TrainingDataService - 학습 데이터 관리

```python
from app.services.training_data_service import TrainingDataService
from sqlalchemy.ext.asyncio import AsyncSession

# 초기화
training_service = TrainingDataService(db_session)

# 학습 데이터 저장
await training_service.save_training_data(
    symbol="BTCUSDT",
    timeframe="1h",
    candles=[...],
    analysis={...}  # 선택사항
)

# 학습 데이터 조회
training_data = await training_service.get_training_data(
    symbol="BTCUSDT",
    timeframe="1h",
    limit=1000
)

# 신호 히스토리 저장
await training_service.save_signal_history(
    symbol="BTCUSDT",
    signal="BUY",
    confidence=0.85,
    price=42000.0,
    source="gemini"
)

# 신호 통계 조회
stats = await training_service.get_signal_statistics(
    symbol="BTCUSDT",
    days=30
)
```

### 2. MarketDataService - 시장 데이터 관리

```python
from app.services.market_data_service import MarketDataService

# 초기화
market_service = MarketDataService(db_session)

# 캔들 데이터 저장
saved_count = await market_service.save_candles(
    symbol="BTCUSDT",
    timeframe="1h",
    candles=[...]
)

# 기술적 지표 계산 및 저장
await market_service.calculate_and_save_indicators(
    symbol="BTCUSDT",
    timeframe="1h",
    candles=[...]
)

# 캔들 데이터 조회
candles = await market_service.get_candles(
    symbol="BTCUSDT",
    timeframe="1h",
    limit=500
)

# 기술적 지표 조회
indicator = await market_service.get_latest_indicator(
    symbol="BTCUSDT",
    timeframe="1h"
)

# 시장 통계 조회
stats = await market_service.get_market_statistics(
    symbol="BTCUSDT",
    timeframe="1h"
)
```

---

## 🤖 AI 학습 데이터 구조

### features 필드 예시 (JSON)
```json
{
  "price": 42000.0,
  "sma_5": 41950.0,
  "sma_10": 41900.0,
  "sma_20": 41800.0,
  "sma_50": 41500.0,
  "high_50": 42500.0,
  "low_50": 40500.0,
  "volume_avg": 150.0,
  "volatility": 2.5,
  "rsi": 65.0,
  "macd": 150.0,
  "price_change_5": 1.2,
  "price_change_20": 2.5,
  "price_change_50": 5.0,
  "volume_change": 15.0
}
```

---

## 🔄 데이터 흐름

```
실시간 캔들 데이터
    ↓
market_candles 테이블에 저장
    ↓
기술적 지표 계산
    ↓
technical_indicators 테이블에 저장
    ↓
학습 데이터 생성 (features + 미래 레이블)
    ↓
ai_training_data 테이블에 저장
    ↓
AI 모델 학습용 데이터로 활용
```

---

## 📊 데이터 조회 쿼리 예시

### 1. 최근 1000개의 학습 데이터 조회 (레이블 있는 것만)
```python
query = select(AITrainingData).where(
    AITrainingData.symbol == "BTCUSDT",
    AITrainingData.timeframe == "1h",
    AITrainingData.label_1h.isnot(None)
).order_by(desc(AITrainingData.timestamp)).limit(1000)
```

### 2. 특정 기간의 캔들 데이터
```python
from datetime import datetime, timedelta

since = datetime.now() - timedelta(days=7)
query = select(MarketCandle).where(
    MarketCandle.symbol == "BTCUSDT",
    MarketCandle.timeframe == "1h",
    MarketCandle.open_time >= since
).order_by(MarketCandle.open_time)
```

### 3. 신호 별 수익성 분석
```python
query = select(SignalHistory).where(
    SignalHistory.symbol == "BTCUSDT",
    SignalHistory.signal == "BUY"
).order_by(desc(SignalHistory.timestamp)).limit(100)
```

---

## 🚀 다음 단계

1. **API 엔드포인트 추가**: 학습 데이터 조회 및 관리 API
2. **자동 레이블 업데이트**: 미래 가격 데이터로 레이블 자동 업데이트
3. **데이터 검증**: 이상치 탐지 및 데이터 품질 관리
4. **AI 모델 통합**: PyTorch/TensorFlow 모델과 연동
5. **백테스팅**: 과거 데이터로 전략 성능 평가

---

## 💾 현재 설정

- **DB**: SQLite (trading.db)
- **마이그레이션**: Alembic
- **ORM**: SQLAlchemy (async)
- **Python**: 3.10+

---

## ⚠️ 주의사항

1. **데이터 일관성**: 캔들 데이터 → 지표 → 학습 데이터 순서대로 저장
2. **레이블 생성**: 미래 데이터가 있어야만 레이블 생성 가능
3. **마이그레이션**: 모델 변경 후 반드시 새 마이그레이션 생성
4. **성능**: 대량 데이터 저장 시 배치 처리 권장

---

## 📝 데이터 저장 체크리스트

- [ ] 실시간 캔들 데이터 저장 (MarketDataService)
- [ ] 기술적 지표 계산 (TechnicalIndicators)
- [ ] 학습 데이터 생성 (TrainingDataService)
- [ ] AI 분석 결과 저장
- [ ] 신호 히스토리 기록
- [ ] 정기적인 데이터 정리 (오래된 데이터 제거)

---

## 🔗 참고 파일

- 모델 정의: `app/models/market_data.py`
- 서비스: `app/services/training_data_service.py`, `app/services/market_data_service.py`
- 마이그레이션: `alembic/versions/`
- DB 설정: `app/database.py`

---

생성일: 2026-01-22
