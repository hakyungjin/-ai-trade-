# 가중치 기반 분석 (Weighted Strategy) 가이드

## 📊 전체 흐름

```
프론트엔드 요청
    ↓
/api/ai/combined-analysis (POST)
    ↓
캔들 데이터 수집 (100개)
    ↓
┌─────────────────────────────────────┐
│  1️⃣ AI 예측 (Gemini 또는 로컬 모델)   │
│   - signal: BUY/SELL/HOLD           │
│   - confidence: 0~1                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  2️⃣ 기술적 지표 계산                 │
│   - RSI (14)                        │
│   - MACD                            │
│   - Bollinger Bands                 │
│   - EMA (12, 26)                    │
│   - Stochastic                      │
│   - Volume Ratio                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  3️⃣ 가중치 적용 & 점수 계산          │
│   최종 점수 = Σ(지표값 × 가중치)    │
│   범위: -1.0 ~ 1.0                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  4️⃣ 신호 생성                       │
│   score > 0.6  → STRONG_BUY         │
│   score > 0.3  → BUY                │
│   score > -0.3 → NEUTRAL            │
│   score > -0.6 → SELL               │
│   score ≤ -0.6 → STRONG_SELL        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  5️⃣ 최종 종합 신호                   │
│   AI 신호 50% + 기술적 신호 50%     │
│   → final_signal & final_confidence │
└─────────────────────────────────────┘
    ↓
프론트엔드에 응답
```

## 🔢 가중치 설정

기본 가중치 (backend/app/data/weight_config.json):

```json
{
  "weights": {
    "rsi": 0.20,      // RSI 20%
    "macd": 0.25,     // MACD 25%
    "bollinger": 0.15,// 볼린저밴드 15%
    "ema_cross": 0.20,// EMA 교차 20%
    "stochastic": 0.10,// 스토캐스틱 10%
    "volume": 0.10    // 거래량 10%
  }
}
```

## 📈 각 지표별 분석

### 1. RSI (Relative Strength Index) - 20%
- **목적**: 과매수/과매도 상태 판단
- **계산**: 14일 기준
- **점수 변환**:
  - RSI > 70 → +1.0 (과매수)
  - RSI < 30 → -1.0 (과매도)
  - 30~70 → 0.0 (중립)

### 2. MACD (Moving Average Convergence Divergence) - 25%
- **목적**: 추세 변화 감지
- **계산**: EMA(12) - EMA(26)
- **신호**: MACD가 Signal 위에 있으면 양수, 아래면 음수

### 3. Bollinger Bands - 15%
- **목적**: 변동성과 지지/저항 파악
- **계산**: 중간선 ±2 표준편차
- **신호**: 
  - 상단 터치 → -1.0 (과매수)
  - 하단 터치 → +1.0 (과매도)

### 4. EMA Cross (12, 26) - 20%
- **목적**: 단기/중기 추세 교차
- **신호**:
  - EMA12 > EMA26 → +1.0 (상승)
  - EMA12 < EMA26 → -1.0 (하강)

### 5. Stochastic - 10%
- **목적**: 모멘텀 측정
- **계산**: (Close - Low) / (High - Low)
- **신호**: K선 기반

### 6. Volume Ratio - 10%
- **목적**: 거래량 확인
- **신호**: 증가 거래량 = 신호 강화

## 🎯 최종 계산 예시

```
RSI 점수: 0.8 × 0.20 = 0.16
MACD 점수: 0.7 × 0.25 = 0.175
BB 점수: 0.5 × 0.15 = 0.075
EMA 점수: 0.9 × 0.20 = 0.18
Stoch 점수: 0.6 × 0.10 = 0.06
Volume 점수: 0.4 × 0.10 = 0.04
───────────────────────────
최종 점수 = 0.725 (약 0.73)

신호: BUY (0.3 < 0.73 < 0.6)
신뢰도: 72.5%
```

## 📍 응답 구조

```json
{
  "symbol": "BTCUSDT",
  "current_price": 45000,
  "timeframe": "1h",
  "ai_prediction": {
    "symbol": "BTCUSDT",
    "signal": "HOLD",
    "confidence": 0.5,
    "predicted_direction": "NEUTRAL",
    "current_price": 45000,
    "analysis": "..."
  },
  "weighted_signal": {
    "signal": "buy",
    "score": 0.725,
    "confidence": 0.725,
    "indicators": {
      "rsi_score": 0.8,
      "macd_score": 0.7,
      "bollinger_score": 0.5,
      "ema_cross_score": 0.9,
      "stochastic_score": 0.6,
      "volume_score": 0.4
    },
    "recommendation": "기술적 지표 종합 신호: buy"
  },
  "final_signal": "BUY",
  "final_confidence": 0.6125  // (AI 0.5 + 기술적 0.725) / 2
}
```

## 🔄 데이터 흐름 (코드 레벨)

### 1. API 요청
```python
POST /api/ai/combined-analysis
{
  "symbol": "BTCUSDT",
  "timeframe": "1h"
}
```

### 2. 캔들 데이터 수집
```python
candles = await unified_service.get_klines_with_cache(
    symbol="BTCUSDT",
    timeframe="1h",
    limit=100
)
```

### 3. 기술적 지표 계산
```python
df = pd.DataFrame(candles)
tech_data = TechnicalIndicators.calculate_all_indicators(df)
```

### 4. 가중치 전략 분석
```python
strategy = WeightedStrategy()
signal, score, indicator_scores = strategy.analyze(tech_data)
```

### 5. 최종 신호 결정
```python
final_score = (ai_score + tech_score) / 2
final_signal = "BUY" if final_score > 0.3 else "SELL" if final_score < -0.3 else "HOLD"
```

## 🎛️ 가중치 조정 (Admin)

### Admin 페이지에서 실시간 조정 가능:
```python
PUT /api/admin/weights
{
  "weights": {
    "rsi": 0.25,        // 20% → 25%로 증가
    "macd": 0.20,       // 25% → 20%로 감소
    "bollinger": 0.15,
    "ema_cross": 0.20,
    "stochastic": 0.10,
    "volume": 0.10
  }
}
```

### 변경 후 자동 적용:
```python
strategy.refresh_weights()
```

## ⚡ 성능 최적화

1. **캐싱**: DB에서 최근 캔들 캐시
2. **비동기 처리**: 모든 API 호출 async/await
3. **배치 계산**: 여러 지표 동시 계산
4. **TTL**: 1분 단위 캐시 갱신

## 🔐 주의사항

1. **과최적화 방지**: 너무 많은 지표 사용 금지
2. **신뢰도 확인**: confidence > 0.5 일 때만 거래 권장
3. **리스크 관리**: 최종_신호만으로 거래하지 말 것
4. **테스트**: 백테스트 필수

## 📊 프론트엔드 표시

대시보드에서:
- ✅ AI 신호 + 신뢰도
- ✅ 기술적 지표 신호 + 점수
- ✅ 최종 종합 신호
- ✅ 전체 신뢰도 진행률 바
- ✅ 각 지표 상세 정보 (호버)

---

**마지막 업데이트**: 2026-01-22
**상태**: 프로덕션 배포 완료
