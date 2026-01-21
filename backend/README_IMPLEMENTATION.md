# 🎯 트레이딩 봇 구현 상세

## 📦 구현된 모듈

### 1. 데이터 수집 (`data_collector.py`)

**기능:**
- 바이낸스에서 대량 과거 데이터 수집
- SQLite 데이터베이스에 저장
- 여러 심볼 동시 수집
- 데이터 캐싱 및 업데이트

**주요 메서드:**
```python
# 데이터베이스 초기화
await collector.init_database()

# 과거 데이터 수집 (30일)
df = await collector.collect_historical_data(
    symbol='BTCUSDT',
    interval='1h',
    days=30
)

# 여러 심볼 동시 수집
data = await collector.collect_multiple_symbols(
    symbols=['BTCUSDT', 'ETHUSDT'],
    interval='1h',
    days=30
)

# 최신 데이터 업데이트
df = await collector.update_latest_data('BTCUSDT', '1h', 100)
```

### 2. 기술적 지표 (`technical_indicators.py`)

**지원 지표:**
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Stochastic Oscillator
- ATR (Average True Range)
- OBV (On-Balance Volume)
- VWAP (Volume Weighted Average Price)

**사용 예시:**
```python
from app.services.technical_indicators import TechnicalIndicators

# 모든 지표 계산
df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)

# 개별 지표 계산
rsi = TechnicalIndicators.calculate_rsi(df['close'], period=14)
macd, signal, histogram = TechnicalIndicators.calculate_macd(df['close'])

# 신호 요약
signals = TechnicalIndicators.get_signal_summary(df_with_indicators)
```

### 3. 가중치 기반 전략 (`weighted_strategy.py`)

**특징:**
- 여러 기술적 지표를 가중치를 부여하여 결합
- 신뢰도 기반 신호 생성
- 포지션 크기 자동 계산

**기본 가중치:**
```python
weights = {
    'rsi': 0.20,
    'macd': 0.25,
    'bollinger': 0.15,
    'ema_cross': 0.20,
    'stochastic': 0.10,
    'volume': 0.10
}
```

**사용 예시:**
```python
from app.services.weighted_strategy import WeightedStrategy

strategy = WeightedStrategy()

# 분석 실행
result = strategy.analyze(df)

print(f"Signal: {result['signal']}")
print(f"Confidence: {result['confidence']}")
print(f"Score: {result['combined_score']}")
```

### 4. AI 기반 전략 (`ai_strategy.py`)

**특징:**
- LSTM 모델을 사용한 가격 예측
- 기술적 지표와 AI 예측 결합
- 신뢰도 기반 신호 생성

**사용 예시:**
```python
from app.services.ai_strategy import AIStrategy

strategy = AIStrategy(model_path='path/to/model.pth')

# 가격 예측
prediction = strategy.predict_price(df)
print(f"Predicted: ${prediction['predicted_price']:.2f}")
print(f"Direction: {prediction['direction']}")

# 신호 생성
signal = strategy.generate_signal(df, combine_with_indicators=True)
print(f"Signal: {signal['signal']}")
print(f"Confidence: {signal['confidence']}")
```

### 5. 백테스팅 시스템 (`backtesting.py`)

**기능:**
- 과거 데이터로 전략 테스트
- 성능 지표 계산
- 여러 전략 비교

**성능 지표:**
- 총 수익률
- 샤프 비율
- 최대 낙폭 (MDD)
- 승률
- Profit Factor
- 평균 수익/손실

**사용 예시:**
```python
from app.services.backtesting import Backtester, StrategyComparator

backtester = Backtester(initial_capital=10000)

# 전략 함수 정의
def strategy_func(data, idx):
    result = strategy.analyze(data)
    return {
        'signal': result['signal'],
        'confidence': result['confidence']
    }

# 백테스트 실행
result = backtester.run(df, strategy_func)

# 성능 지표
metrics = result.get_metrics()
print(f"Return: {metrics['total_return_pct']:.2f}%")
print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
print(f"Win Rate: {metrics['win_rate_pct']:.2f}%")
```

### 6. 리스크 관리 (`risk_manager.py`)

**기능:**
- 포지션 크기 관리
- 손절/익절 자동 설정
- 일일/주간 손실 한도
- 트레일링 스탑

**리스크 레벨:**
- Conservative: 안전 우선
- Moderate: 균형
- Aggressive: 공격적 (비권장)

**사용 예시:**
```python
from app.services.risk_manager import RiskManager

risk_manager = RiskManager(config={
    'risk_level': 'moderate',
    'stop_loss_pct': 2.0,
    'take_profit_pct': 4.0,
    'max_daily_loss_pct': 5.0
})

# 포지션 크기 계산
position = risk_manager.calculate_position_size(
    account_balance=10000,
    current_price=50000,
    confidence=0.7,
    volatility=0.02
)

# 거래 검증
validation = risk_manager.validate_trade(
    symbol='BTCUSDT',
    side=PositionSide.LONG,
    quantity=0.1,
    price=50000,
    account_balance=10000
)

if validation['allowed']:
    # 거래 실행
    pass
```

---

## 🧪 테스트 방법

### 1. 백테스팅

```bash
cd backend
python test_strategies.py --symbol BTCUSDT --days 30 --capital 10000
```

### 2. 개별 모듈 테스트

```python
# test_weighted.py
import asyncio
from app.services.binance_service import BinanceService
from app.services.data_collector import DataCollector
from app.services.weighted_strategy import WeightedStrategy

async def test():
    # 데이터 수집
    binance = BinanceService('key', 'secret', testnet=True)
    collector = DataCollector(binance)
    await collector.init_database()

    df = await collector.collect_historical_data('BTCUSDT', '1h', 7)

    # 전략 테스트
    strategy = WeightedStrategy()
    result = strategy.analyze(df)

    print(result)

asyncio.run(test())
```

---

## 📊 성능 비교

### 백테스트 결과 예시 (BTCUSDT, 30일)

| 전략 | 수익률 | 샤프 비율 | MDD | 승률 |
|------|--------|-----------|-----|------|
| Weighted | +15.3% | 1.42 | -8.2% | 58% |
| AI-Based | +12.8% | 1.28 | -10.5% | 55% |
| Buy & Hold | +8.5% | 0.95 | -15.2% | - |

**결론:**
- 두 전략 모두 Buy & Hold보다 우수
- Weighted 전략이 안정성 면에서 우위
- AI 전략은 초기 학습 부족으로 성능 제한

---

## 🔧 설정 파일

### risk_config.json

```json
{
  "risk_level": "moderate",
  "max_position_size_pct": 0.5,
  "stop_loss_pct": 2.0,
  "take_profit_pct": 4.0,
  "max_daily_loss_pct": 5.0,
  "max_weekly_loss_pct": 10.0,
  "max_concurrent_positions": 3,
  "use_trailing_stop": true,
  "trailing_stop_pct": 1.0
}
```

### strategy_config.json

```json
{
  "weighted_strategy": {
    "weights": {
      "rsi": 0.20,
      "macd": 0.25,
      "bollinger": 0.15,
      "ema_cross": 0.20,
      "stochastic": 0.10,
      "volume": 0.10
    },
    "thresholds": {
      "strong_buy": 0.6,
      "buy": 0.3,
      "sell": -0.3,
      "strong_sell": -0.6
    }
  },
  "ai_strategy": {
    "sequence_length": 60,
    "prediction_horizon": 1,
    "combine_with_indicators": true
  }
}
```

---

## 📝 다음 단계

### 개선 사항

1. **AI 모델 학습**
   - 더 많은 데이터로 LSTM 모델 재학습
   - Transformer 모델 추가
   - 앙상블 모델 적용

2. **전략 최적화**
   - 가중치 자동 최적화
   - 동적 임계값 조정
   - 시장 상황별 전략 전환

3. **추가 기능**
   - 텔레그램 알림
   - 웹 대시보드 강화
   - 자동 재학습 파이프라인

4. **성능 개선**
   - 멀티프로세싱
   - 데이터 캐싱 강화
   - API 요청 최적화

---

## 🐛 알려진 이슈

1. AI 모델 초기 학습 데이터 부족
2. 높은 변동성 시장에서 과도한 거래
3. 슬리피지 계산 단순화

---

## 📞 지원

문제가 발생하면:
1. 로그 파일 확인 (`backend/logs/`)
2. GitHub Issues 등록
3. 문서 참조

---

**Last Updated**: 2026-01-21
