# Crypto AI Trader - AI Model Training Pipeline

암호화폐 가격 예측을 위한 XGBoost 기반 AI 모델 학습 파이프라인입니다.

---

## 🚀 빠른 시작 (Quick Start)

### ai-model 폴더에서 실행 (권장)

```powershell
# 1. ai-model 폴더로 이동 & venv 활성화
cd ai-model
.\.venv\Scripts\Activate.ps1

# 2. (최초 1회) 패키지 설치
pip install -r requirements.txt

# 3. 데이터 수집 (모니터링 코인 전체 - 현물/선물 자동)
python scripts/collect_monitored_coins.py --timeframe 5m --target 10000

# 4. 데이터 수집 (단일 코인 대량)
python scripts/collect_large_dataset.py --symbol BTCUSDT --timeframe 5m --target 50000

# 5. 학습 데이터 준비 (3클래스: BUY/HOLD/SELL)
python scripts/prepare_training_data.py --symbol BTCUSDT --timeframe 5m --threshold 0.01 --future 6 --limit 50000 --classes 3

# 6. 모델 학습
python scripts/train_model.py --input data/btcusdt_5m_training.csv --model-name xgboost_btcusdt_5m
```

### 선물 코인 예시 (BEATUSDT)

```powershell
# 1. 선물 데이터 수집
python scripts/collect_monitored_coins.py --symbols BEATUSDT --market futures --timeframe 5m --target 50000

# 2. 학습 데이터 준비 (3클래스)
python scripts/prepare_training_data.py --symbol BEATUSDT --timeframe 5m --threshold 0.01 --future 3 --limit 50000 --classes 3

# 3. 모델 학습
python scripts/train_model.py --input data/beatusdt_5m_training.csv --model-name xgboost_beatusdt_5m
```

---

## 🧠 LSTM 시계열 모델 (PyTorch)

XGBoost는 **현재 봉 1개**만 보지만, LSTM은 **20봉 시퀀스 전체 패턴**을 학습합니다.

### LSTM 학습 명령어

```powershell
# 1. 데이터 준비 (이미 했으면 스킵)
python scripts/prepare_training_data.py --symbol BTCUSDT --timeframe 5m --limit 10000

# 2. LSTM 학습 (PyTorch)
python scripts/train_lstm.py --symbol BTCUSDT --timeframe 5m --seq-length 20 --epochs 100

# 옵션 설명:
#   --seq-length 20   : 20봉 시퀀스로 패턴 학습 (기본값)
#   --epochs 100      : 최대 학습 에포크 (early stopping 있음)
#   --batch-size 32   : 배치 크기
#   --lr 0.001        : 학습률
```

### XGBoost vs LSTM 비교

| 항목 | XGBoost | LSTM |
|------|---------|------|
| 입력 | 현재 봉 피처 1개 | 20봉 시퀀스 패턴 |
| 시계열 | ❌ 못 봄 | ✅ 패턴 학습 |
| 학습 속도 | ⚡ 빠름 (분) | 🐢 느림 (시간) |
| 데이터 필요량 | 1K~10K | 10K+ |
| 해석 가능성 | ✅ 피처 중요도 | ❌ 블랙박스 |

### 앙상블 (XGBoost + LSTM)

```python
from ensemble_predictor import EnsemblePredictor

predictor = EnsemblePredictor('BTCUSDT', '5m', xgb_weight=0.6, lstm_weight=0.4)
result = predictor.predict(candles_df)

# 결과:
# {
#     'signal': 'BUY',
#     'confidence': 0.72,
#     'agreement': True,  # 두 모델 동의 여부
#     'xgb_signal': 'BUY',
#     'lstm_signal': 'BUY'
# }
```

---

## 📊 추가된 고급 피처

| 카테고리 | 피처 | 설명 |
|----------|------|------|
| **OBV** | `obv_slope`, `obv_divergence` | 스마트머니 흐름, 다이버전스 |
| **MFI** | `mfi_normalized`, `mfi_overbought/oversold` | 거래량 가중 RSI |
| **Williams %R** | `williams_r`, `overbought/oversold` | 모멘텀 과매수/과매도 |
| **ATR 비율** | `atr_ratio` | 변동성 정규화 |
| **캔들 패턴** | `pattern_doji`, `hammer`, `engulfing` | 반전 신호 감지 |
| **복합 신호** | `strong_buy/sell_signal` | OBV+거래량+가격 종합 |

### backend 폴더에서 서버 실행

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```

---

## 📋 스크립트 설명

| 스크립트 | 용도 |
|---------|------|
| `collect_large_dataset.py` | 단일 코인 대량 데이터 수집 (Binance API) |
| `collect_monitored_coins.py` | 모니터링 중인 코인 전체 데이터 수집 |
| `prepare_training_data.py` | 기술적 지표 계산 + 라벨링 |
| `train_model.py` | XGBoost 모델 학습 |

## 📊 파라미터 설명

### collect_monitored_coins.py / collect_large_dataset.py
| 파라미터 | 설명 | 예시 |
|----------|------|------|
| `--symbol` / `--symbols` | 코인 심볼 | `BTCUSDT`, `ETHUSDT` |
| `--timeframe` | 캔들 봉 간격 | `1m`, `5m`, `15m`, `1h` |
| `--target` | 수집할 캔들 개수 | `10000`, `50000` |
| `--market` | 마켓 타입 (현물/선물) | `spot`, `futures` |

### prepare_training_data.py
| 파라미터 | 설명 | 예시 |
|----------|------|------|
| `--symbol` | 코인 심볼 | `BTCUSDT` |
| `--timeframe` | 캔들 봉 간격 | `5m` |
| `--threshold` | BUY/SELL 분류 기준 (변동률) | `0.02` = 2% |
| `--future` | 몇 개 봉 뒤 가격으로 라벨 | `6` = 30분 후 (5m 봉 기준) |
| `--limit` | 가져올 캔들 개수 | `50000` |
| `--classes` | 클래스 수 (3 또는 5) | `3` = BUY/HOLD/SELL (권장) |

### train_model.py
| 파라미터 | 설명 | 예시 |
|----------|------|------|
| `--input` | 학습 데이터 CSV 경로 **(필수)** | `data/btcusdt_5m_training.csv` |
| `--model-name` | 저장할 모델 이름 | `xgboost_btcusdt_5m` |
| `--output-dir` | 모델 저장 디렉토리 | `models/` |
| `--test-size` | 테스트셋 비율 | `0.2` (20%) |

## 📂 결과물 위치

| 항목 | 경로 |
|------|------|
| 가공된 데이터 | `ai-model/data/{symbol}_{timeframe}_training.csv` |
| 학습된 모델 | `ai-model/models/{model-name}.json` |
| 스케일러 | `ai-model/models/scaler_{model-name}.pkl` |

## 차트 데이터 학습 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│ 1️⃣ 데이터 수집 (data_collector.py)                             │
│   - Binance API에서 과거 OHLCV 데이터 수집                      │
│   - 심볼: BTCUSDT, ETHUSDT 등                                   │
│   - 봉: 1분, 5분, 15분, 1시간, 4시간, 1일 등                   │
│   - 예: 1년치 1시간봉 8760개 캔들 데이터                        │
└─────────────────────────────────────────────────────────────────┘
                           ⬇️
┌─────────────────────────────────────────────────────────────────┐
│ 2️⃣ 피처 엔지니어링 (feature_engineering.py)                     │
│   - 기술적 지표 계산:                                           │
│     • 이동평균: SMA 5/10/20/50, EMA 12/26                       │
│     • 모멘텀: RSI, MACD, 볼린저 밴드, ATR                       │
│     • 거래량: 거래량 비율, 변화율                               │
│   - 레이블 생성: 향후 5캔들 가격 변화                           │
│     • 상승(+2%): 클래스 2                                       │
│     • 횡보(-2%~+2%): 클래스 1                                   │
│     • 하락(-2%): 클래스 0                                       │
│   - 결과: (샘플, 18개 피처) 배열                                │
└─────────────────────────────────────────────────────────────────┘
                           ⬇️
┌─────────────────────────────────────────────────────────────────┐
│ 3️⃣ 시퀀스 생성 (train.py)                                      │
│   - LSTM 학습용 시계열 시퀀스:                                  │
│     • 입력: 과거 20개 캔들 (20 timesteps × 18 features)       │
│     • 출력: 다음 캔들 레이블 (3개 클래스)                       │
│   - 예: 8760개 → 8740개 학습 샘플                              │
│   - Train/Test 분할: 80%/20%                                   │
│   - 정규화: StandardScaler로 평균 0, 표준편차 1                │
└─────────────────────────────────────────────────────────────────┘
                           ⬇️
┌─────────────────────────────────────────────────────────────────┐
│ 4️⃣ 모델 학습 (model.py + train.py)                             │
│   - 모델 선택:                                                  │
│     • LSTM: 2개 레이어, 128개 히든 유닛, Attention            │
│     • Transformer: 4개 헤드, 128차원 임베딩                    │
│     • MLP: 512→256→128 완전 연결층                             │
│   - 손실함수: CrossEntropyLoss (클래스 불균형 가중치)          │
│   - 옵티마이저: Adam (학습률 0.001)                            │
│   - Early Stopping: 10 에포크 patience                          │
│   - 결과: 최고 성능 모델 저장                                  │
└─────────────────────────────────────────────────────────────────┘
                           ⬇️
┌─────────────────────────────────────────────────────────────────┐
│ 5️⃣ 평가 및 저장 (train.py)                                     │
│   - 검증 손실, 정확도 추적                                      │
│   - 체크포인트: 모델 가중치 + Scaler + 메타데이터 저장         │
│   - 위치: models/best_model.pt                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 프로젝트 구조

```
ai-model/
├── pipeline.py                    # CLI 학습 진입점
├── requirements.txt               # Python 의존성
├── run_training.py                # 간단한 학습 스크립트
├── data/                          # 다운로드한 OHLCV CSV
│   └── btcusdt_1h.csv
├── models/                        # 학습된 모델 저장
│   ├── best_model.pt              # 최고 성능 모델 (체크포인트)
│   └── price_predictor.pt         # 프로덕션 모델
├── training/                      # 학습 모듈
│   ├── __init__.py
│   ├── data_collector.py          # Binance 데이터 수집
│   ├── feature_engineering.py     # 기술적 지표 + 레이블
│   ├── train.py                   # Trainer 클래스
│   └── model.py                   # LSTM/Transformer/MLP 모델 정의
└── README.md
```

## 설치 및 실행

### 1. 가상환경 설정

```bash
cd ai-model

# 가상환경 생성
python -m venv .venv

# 활성화 (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 활성화 (macOS/Linux)
source .venv/bin/activate
```

### 2. 의존성 설치

```bash
pip install --prefer-binary -r requirements.txt
```

### 3. 환경변수 설정 (.env)

프로젝트 루트에 `.env` 파일 생성:

```env
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
```

> 주의: 데이터 수집만 하려면 API 키가 필수입니다. (Binance 무료 계정 가능)

---

## 학습 실행

### 방법 1: CLI 파이프라인 (권장)

간단한 명령으로 전체 파이프라인 실행:

```bash
# 기본 설정 (BTCUSDT, 1시간봉, LSTM, 50 에포크)
python pipeline.py

# 커스텀 설정
python pipeline.py \
  --symbol ETHUSDT \
  --start_date 2023-01-01 \
  --interval 1h \
  --model_type lstm \
  --epochs 100 \
  --batch_size 64 \
  --learning_rate 0.0005

# 모든 옵션
python pipeline.py --help
```

**주요 파라미터:**

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--symbol` | BTCUSDT | 거래 페어 |
| `--interval` | 1h | 캔들 간격 (1m, 5m, 15m, 1h, 4h, 1d) |
| `--start_date` | 2023-01-01 | 데이터 시작 날짜 |
| `--model_type` | lstm | 모델 종류 (lstm, transformer, mlp) |
| `--epochs` | 50 | 학습 에포크 |
| `--batch_size` | 32 | 배치 크기 |
| `--learning_rate` | 0.001 | 학습률 |
| `--threshold` | 0.02 | 상승/하락 판정 기준 (2%) |
| `--lookahead` | 5 | 미래 몇 캔들 예측 |

**실행 예:**
```bash
# BTC 1년치 데이터로 Transformer 학습
python pipeline.py \
  --symbol BTCUSDT \
  --start_date 2024-01-01 \
  --model_type transformer \
  --epochs 100

# ETH 6개월 데이터로 빠른 테스트 (MLP)
python pipeline.py \
  --symbol ETHUSDT \
  --start_date 2024-07-01 \
  --model_type mlp \
  --epochs 30 \
  --batch_size 128
```

### 방법 2: Python 스크립트

`run_training.py` 파일을 수정해서 커스텀 학습:

```python
from training.train import Trainer

# 트레이너 초기화
trainer = Trainer(model_type="lstm", sequence_length=20, num_classes=3)

# 데이터 준비 (자동 수집 + 전처리)
X, y = trainer.prepare_data(
    symbol="BTCUSDT",
    interval="1h",
    start_date="2023-01-01",
    threshold=0.02,
    lookahead=5
)

# 데이터로더 생성
train_loader, test_loader = trainer.create_dataloaders(
    X, y, test_size=0.2, batch_size=32
)

# 학습
history = trainer.train(
    train_loader, test_loader,
    epochs=100,
    learning_rate=0.001,
    early_stopping_patience=10
)

# 최종 평가
trainer.load_model("./models/best_model.pt")
test_loss, test_acc = trainer.evaluate(test_loader)
print(f"Test Accuracy: {test_acc:.4f}")
```

### 방법 3: Jupyter Notebook (탐색 중심)

```python
import pandas as pd
from training.data_collector import DataCollector
from training.feature_engineering import FeatureEngineer
import matplotlib.pyplot as plt

# 데이터 수집
collector = DataCollector()
df = collector.fetch_historical_data(
    symbol="BTCUSDT",
    interval="1h",
    start_date="2023-01-01"
)
print(f"수집된 캔들: {len(df)}")

# 기술적 지표 추가
engineer = FeatureEngineer()
df = engineer.add_technical_indicators(df)
df = engineer.create_labels(df, threshold=0.02, lookahead=5)

# 시각화
plt.figure(figsize=(15, 5))
plt.plot(df.index, df['close'], label='Close')
plt.plot(df.index, df['sma_20'], label='SMA 20', alpha=0.7)
plt.plot(df.index, df['sma_50'], label='SMA 50', alpha=0.7)
plt.legend()
plt.title(f"BTCUSDT with Technical Indicators")
plt.show()

# 레이블 분포
print("\nLabel Distribution:")
print(df['label'].value_counts())
```

---

## 학습 결과 분석

### 학습 후 파일

- `models/best_model.pt`: 최고 성능 모델 (체크포인트)
- `history.json`: 학습 곡선 데이터
  ```json
  {
    "train_loss": [0.85, 0.75, 0.65, ...],
    "val_loss": [0.80, 0.74, 0.68, ...],
    "val_acc": [0.55, 0.58, 0.62, ...]
  }
  ```

### 성능 해석

- **정확도 55~65%**: 기본 랜덤보다 나음 (3 클래스 기준: 33%)
- **손실 감소**: 학습이 수렴하는지 확인
- **과적합**: 검증 손실이 증가하면 early stopping 동작

### 시각화 (선택)

```python
import json
import matplotlib.pyplot as plt

with open("history.json") as f:
    history = json.load(f)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train")
plt.plot(history["val_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(history["val_acc"], label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Validation Accuracy")

plt.tight_layout()
plt.show()
```

---

## 모델 사용 (백엔드 연동)

학습된 모델은 백엔드 API에서 자동으로 로드됩니다:

```python
# backend/app/services/ai_service.py
from training.train import Trainer

trainer = Trainer()
trainer.load_model("../ai-model/models/best_model.pt")

# 예측
predicted_class, probabilities = trainer.predict(X_new)
# 0: 하락, 1: 횡보, 2: 상승
```

백엔드 API:
```bash
curl -X POST http://localhost:8000/api/ai/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTCUSDT","timeframe":"1h"}'
```

---

## 데이터 수집 심화

### 여러 심볼 동시 수집

```python
from training.data_collector import DataCollector

collector = DataCollector()
data = collector.fetch_multiple_symbols(
    symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
    interval="1h",
    start_date="2023-01-01",
    days=365
)

for symbol, df in data.items():
    collector.save_data(df, f"./data/{symbol}_1h.csv")
```

### CSV 저장/로드

```python
# 저장
collector.save_data(df, "./data/btcusdt_1h.csv")

# 로드
df = collector.load_data("./data/btcusdt_1h.csv")
```

---

## 모델 선택 가이드

| 모델 | 장점 | 단점 | 추천 상황 |
|-----|------|------|---------|
| **LSTM** | 장기 의존성 학습, 시계열에 최적 | 느린 학습, 높은 메모리 | 장기 트렌드 예측 |
| **Transformer** | 병렬 처리 빠름, Attention | 더 많은 데이터 필요 | 고주파 데이터 (1m, 5m) |
| **MLP** | 매우 빠름, 간단 | 시계열 특성 무시 | 빠른 프로토타이핑 |

---

## 성능 최적화 팁

### 1. 하이퍼파라미터 튜닝

```python
# Grid Search 예제
learning_rates = [0.0001, 0.0005, 0.001]
batch_sizes = [16, 32, 64]

for lr in learning_rates:
    for bs in batch_sizes:
        trainer = Trainer()
        X, y = trainer.prepare_data()
        train_loader, test_loader = trainer.create_dataloaders(
            X, y, batch_size=bs
        )
        history = trainer.train(
            train_loader, test_loader,
            epochs=50, learning_rate=lr
        )
        # 결과 비교
```

### 2. 데이터 증강

```python
# 노이즈 추가 (과적합 방지)
X_train_noise = X_train + np.random.normal(0, 0.01, X_train.shape)
```

### 3. 앙상블

여러 모델 학습 후 앙상블:
```python
models = [
    load_lstm_model(),
    load_transformer_model(),
    load_mlp_model()
]

ensemble_prediction = np.mean([
    model.predict(X) for model in models
], axis=0)
```

---

## 오류 해결

### CUDA 메모리 부족

```bash
# GPU 비활성화 (CPU 사용)
python pipeline.py --device cpu

# 또는 코드에서
trainer = Trainer(device="cpu")
```

### 데이터 부족 (너무 적은 샘플)

- 더 오래된 데이터 사용: `--start_date 2022-01-01`
- 더 짧은 간격 사용: `--interval 15m` (더 많은 캔들)
- 시퀀스 길이 감소: `--sequence_length 10`

### 모델 수렴 안함 (손실 증가)

- 학습률 감소: `--learning_rate 0.0001`
- 배치 크기 감소: `--batch_size 16`
- Early Stopping Patience 증가: `--early_stopping_patience 20`

---

## 다음 단계

- [ ] 자동 하이퍼파라미터 튜닝 (Optuna)
- [ ] 앙상블 모델
- [ ] 시간대별 모델 (시간, 요일 특성)
- [ ] 실시간 스트림 학습 (온라인 러닝)
- [ ] 거래량/호가 데이터 활용
- [ ] 뉴스/감정 분석 피처 추가
- [ ] 모델 해석성 (SHAP, 어텐션 시각화)

---

## 라이센스

MIT
