# AI 학습 데이터 파이프라인 가이드

## 개요

저장된 캔들 데이터 + 기술적 지표를 가공하여 AI 모델 학습에 사용하는 방법입니다.

```
[DB: market_candles] → [기술적 지표 계산] → [레이블링] → [학습 데이터셋] → [AI 모델 학습]
```

---

## 1. 데이터 소스

### 1.1 저장된 캔들 데이터 (market_candles 테이블)

```sql
SELECT * FROM market_candles 
WHERE symbol = 'BTCUSDT' AND timeframe = '1h'
ORDER BY open_time DESC
LIMIT 1000;
```

| 컬럼 | 설명 |
|------|------|
| symbol | 거래쌍 (BTCUSDT) |
| timeframe | 시간프레임 (1h, 4h, 1d) |
| open_time | 캔들 시작 시간 |
| open, high, low, close | OHLC 가격 |
| volume | 거래량 |
| quote_volume | 견적 거래량 |
| trades_count | 거래 횟수 |

---

## 2. 기술적 지표 계산

### 2.1 사용 가능한 지표 (TechnicalIndicators 클래스)

```python
from app.services.technical_indicators import TechnicalIndicators

# DataFrame에 모든 지표 추가
df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)
```

**계산되는 지표:**

| 지표 | 컬럼명 | 설명 |
|------|--------|------|
| RSI | `rsi_14` | 상대강도지수 (14기간) |
| MACD | `macd`, `macd_signal`, `macd_histogram` | 이동평균수렴확산 |
| Bollinger Bands | `bb_upper`, `bb_middle`, `bb_lower` | 볼린저 밴드 |
| EMA | `ema_12`, `ema_26`, `ema_50`, `ema_200` | 지수이동평균 |
| SMA | `sma_20`, `sma_50`, `sma_200` | 단순이동평균 |
| Stochastic | `stoch_k`, `stoch_d` | 스토캐스틱 |
| ATR | `atr_14` | 평균진폭 |
| OBV | `obv` | 거래량 기반 지표 |
| ADX | `adx` | 추세 강도 |

---

## 3. 학습 데이터 생성

### 3.1 데이터 가공 스크립트

```python
# backend/scripts/prepare_training_data.py

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import select
from app.database import AsyncSessionLocal
from app.models.market_data import MarketCandle
from app.services.technical_indicators import TechnicalIndicators

async def load_candles(symbol: str, timeframe: str, limit: int = 5000) -> pd.DataFrame:
    """DB에서 캔들 데이터 로드"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(MarketCandle)
            .where(MarketCandle.symbol == symbol)
            .where(MarketCandle.timeframe == timeframe)
            .order_by(MarketCandle.open_time.asc())
            .limit(limit)
        )
        candles = result.scalars().all()
        
        data = [{
            'timestamp': c.open_time,
            'open': float(c.open),
            'high': float(c.high),
            'low': float(c.low),
            'close': float(c.close),
            'volume': float(c.volume),
        } for c in candles]
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표 추가"""
    return TechnicalIndicators.calculate_all_indicators(df)

def create_labels(df: pd.DataFrame, future_periods: int = 5, threshold: float = 0.02) -> pd.DataFrame:
    """
    레이블 생성 (미래 가격 변화 기반)
    
    Args:
        future_periods: 몇 캔들 후 가격을 볼 것인지
        threshold: 상승/하락 판단 기준 (2% = 0.02)
    
    Labels:
        2: STRONG_BUY (5% 이상 상승)
        1: BUY (2% 이상 상승)
        0: HOLD (-2% ~ 2%)
        -1: SELL (2% 이상 하락)
        -2: STRONG_SELL (5% 이상 하락)
    """
    df = df.copy()
    
    # 미래 가격
    df['future_close'] = df['close'].shift(-future_periods)
    
    # 가격 변화율
    df['price_change'] = (df['future_close'] - df['close']) / df['close']
    
    # 레이블 생성
    def get_label(change):
        if pd.isna(change):
            return np.nan
        if change >= 0.05:
            return 2  # STRONG_BUY
        elif change >= threshold:
            return 1  # BUY
        elif change <= -0.05:
            return -2  # STRONG_SELL
        elif change <= -threshold:
            return -1  # SELL
        else:
            return 0  # HOLD
    
    df['label'] = df['price_change'].apply(get_label)
    
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    학습용 피처 생성
    """
    df = df.copy()
    
    # 가격 관련 피처
    df['price_change_1'] = df['close'].pct_change(1)
    df['price_change_5'] = df['close'].pct_change(5)
    df['price_change_10'] = df['close'].pct_change(10)
    
    # 거래량 관련 피처
    df['volume_change_1'] = df['volume'].pct_change(1)
    df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # 가격 위치 피처
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    # 볼린저 밴드 위치
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    
    # RSI 정규화
    if 'rsi_14' in df.columns:
        df['rsi_normalized'] = df['rsi_14'] / 100
    
    # MACD 정규화
    if 'macd' in df.columns:
        df['macd_normalized'] = df['macd'] / df['close'] * 100
    
    # EMA 크로스 피처
    if 'ema_12' in df.columns and 'ema_26' in df.columns:
        df['ema_cross'] = (df['ema_12'] - df['ema_26']) / df['close'] * 100
    
    return df

async def prepare_training_dataset(
    symbol: str = 'BTCUSDT',
    timeframe: str = '1h',
    limit: int = 5000,
    future_periods: int = 5,
    threshold: float = 0.02
) -> pd.DataFrame:
    """
    학습 데이터셋 준비
    """
    print(f"📊 Loading candles for {symbol} {timeframe}...")
    df = await load_candles(symbol, timeframe, limit)
    print(f"   Loaded {len(df)} candles")
    
    print("📈 Adding technical indicators...")
    df = add_technical_indicators(df)
    
    print("🏷️ Creating labels...")
    df = create_labels(df, future_periods, threshold)
    
    print("🔧 Creating features...")
    df = create_features(df)
    
    # NaN 제거
    df = df.dropna()
    print(f"✅ Final dataset: {len(df)} samples")
    
    # 레이블 분포 출력
    print("\n📊 Label distribution:")
    print(df['label'].value_counts().sort_index())
    
    return df

def save_dataset(df: pd.DataFrame, filename: str):
    """데이터셋 저장"""
    df.to_csv(filename, index=True)
    print(f"💾 Saved to {filename}")

# 실행
if __name__ == "__main__":
    df = asyncio.run(prepare_training_dataset(
        symbol='BTCUSDT',
        timeframe='1h',
        limit=10000,
        future_periods=5,
        threshold=0.02
    ))
    
    save_dataset(df, 'data/btcusdt_1h_training.csv')
```

### 3.2 피처 목록

| 카테고리 | 피처 | 설명 |
|----------|------|------|
| **가격** | `open`, `high`, `low`, `close` | OHLC |
| **변화율** | `price_change_1`, `price_change_5`, `price_change_10` | 1/5/10 캔들 전 대비 변화율 |
| **거래량** | `volume`, `volume_change_1`, `volume_ma_ratio` | 거래량 관련 |
| **RSI** | `rsi_14`, `rsi_normalized` | 과매수/과매도 |
| **MACD** | `macd`, `macd_signal`, `macd_histogram`, `macd_normalized` | 추세 |
| **볼린저** | `bb_upper`, `bb_middle`, `bb_lower`, `bb_position` | 변동성 |
| **이동평균** | `ema_12`, `ema_26`, `ema_50`, `ema_200`, `ema_cross` | 추세 |
| **스토캐스틱** | `stoch_k`, `stoch_d` | 모멘텀 |
| **기타** | `atr_14`, `adx`, `obv` | 변동성, 추세강도, 거래량 |

---

## 4. AI 모델 학습

### 4.1 간단한 분류 모델 (XGBoost)

```python
# backend/scripts/train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib

def load_dataset(filename: str) -> pd.DataFrame:
    """데이터셋 로드"""
    return pd.read_csv(filename, index_col=0, parse_dates=True)

def prepare_features_and_labels(df: pd.DataFrame):
    """피처와 레이블 분리"""
    
    # 학습에 사용할 피처 컬럼
    feature_columns = [
        # 기술적 지표
        'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
        'bb_position', 'stoch_k', 'stoch_d', 'atr_14', 'adx',
        
        # 가격 변화
        'price_change_1', 'price_change_5', 'price_change_10',
        
        # 거래량
        'volume_change_1', 'volume_ma_ratio',
        
        # 추가 피처
        'ema_cross', 'rsi_normalized', 'macd_normalized', 'price_position'
    ]
    
    # 존재하는 컬럼만 선택
    available_features = [col for col in feature_columns if col in df.columns]
    
    X = df[available_features].values
    y = df['label'].values
    
    return X, y, available_features

def train_xgboost(X_train, y_train, X_test, y_test):
    """XGBoost 모델 학습"""
    
    # 레이블을 0부터 시작하도록 조정 (-2,-1,0,1,2 → 0,1,2,3,4)
    y_train_adjusted = y_train + 2
    y_test_adjusted = y_test + 2
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softmax',
        num_class=5,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
    
    model.fit(
        X_train, y_train_adjusted,
        eval_set=[(X_test, y_test_adjusted)],
        early_stopping_rounds=10,
        verbose=True
    )
    
    return model

def evaluate_model(model, X_test, y_test, feature_names):
    """모델 평가"""
    y_test_adjusted = y_test + 2
    y_pred = model.predict(X_test)
    
    # 레이블 매핑
    label_names = ['STRONG_SELL', 'SELL', 'HOLD', 'BUY', 'STRONG_BUY']
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test_adjusted, y_pred, target_names=label_names))
    
    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test_adjusted, y_pred))
    
    # 피처 중요도
    print("\n📊 Feature Importance:")
    importance = model.feature_importances_
    for name, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {imp:.4f}")

def main():
    # 1. 데이터 로드
    print("📂 Loading dataset...")
    df = load_dataset('data/btcusdt_1h_training.csv')
    print(f"   Loaded {len(df)} samples")
    
    # 2. 피처/레이블 분리
    print("🔧 Preparing features...")
    X, y, feature_names = prepare_features_and_labels(df)
    print(f"   Features: {len(feature_names)}")
    
    # 3. 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. 학습/테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=False
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 5. 모델 학습
    print("\n🚀 Training XGBoost model...")
    model = train_xgboost(X_train, y_train, X_test, y_test)
    
    # 6. 평가
    evaluate_model(model, X_test, y_test, feature_names)
    
    # 7. 모델 저장
    print("\n💾 Saving model...")
    joblib.dump(model, 'models/xgboost_btcusdt_1h.joblib')
    joblib.dump(scaler, 'models/scaler_btcusdt_1h.joblib')
    joblib.dump(feature_names, 'models/features_btcusdt_1h.joblib')
    print("✅ Model saved!")

if __name__ == "__main__":
    main()
```

### 4.2 LSTM 딥러닝 모델

```python
# backend/scripts/train_lstm.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def create_sequences(X, y, sequence_length=50):
    """시계열 시퀀스 생성"""
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def build_lstm_model(input_shape, num_classes=5):
    """LSTM 모델 구축"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        BatchNormalization(),
        
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # 데이터 로드
    df = pd.read_csv('data/btcusdt_1h_training.csv', index_col=0, parse_dates=True)
    
    # 피처 선택
    feature_columns = [
        'rsi_14', 'macd', 'macd_signal', 'stoch_k', 'stoch_d',
        'bb_position', 'ema_cross', 'price_change_1', 'volume_ma_ratio'
    ]
    
    X = df[feature_columns].values
    y = (df['label'].values + 2).astype(int)  # 0-4로 변환
    
    # 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 시퀀스 생성
    sequence_length = 50
    X_seq, y_seq = create_sequences(X_scaled, y, sequence_length)
    
    # 학습/테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
    )
    
    # 모델 구축
    model = build_lstm_model(
        input_shape=(sequence_length, len(feature_columns)),
        num_classes=5
    )
    
    # 콜백
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint('models/lstm_best.h5', save_best_only=True)
    ]
    
    # 학습
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks
    )
    
    # 평가
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\n✅ Test Accuracy: {accuracy:.4f}")
    
    # 모델 저장
    model.save('models/lstm_btcusdt_1h.h5')

if __name__ == "__main__":
    main()
```

---

## 5. 학습된 모델 서비스 연동

### 5.1 예측 서비스

```python
# backend/app/services/trained_model_service.py

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from app.services.technical_indicators import TechnicalIndicators

class TrainedModelService:
    def __init__(self, model_path: str, scaler_path: str, features_path: str):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(features_path)
        
        # 레이블 매핑
        self.label_map = {
            0: ('STRONG_SELL', -2),
            1: ('SELL', -1),
            2: ('HOLD', 0),
            3: ('BUY', 1),
            4: ('STRONG_BUY', 2)
        }
    
    def predict(self, candles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        캔들 데이터로 예측 수행
        """
        # DataFrame 생성
        df = pd.DataFrame(candles)
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        
        # 기술적 지표 계산
        df = TechnicalIndicators.calculate_all_indicators(df)
        
        # 추가 피처 생성
        df = self._create_features(df)
        
        # 피처 추출
        X = df[self.feature_names].iloc[-1:].values
        
        # 정규화
        X_scaled = self.scaler.transform(X)
        
        # 예측
        pred_class = self.model.predict(X_scaled)[0]
        pred_proba = self.model.predict_proba(X_scaled)[0]
        
        label_name, label_value = self.label_map[pred_class]
        confidence = float(pred_proba[pred_class])
        
        return {
            'signal': label_name,
            'signal_value': label_value,
            'confidence': confidence,
            'probabilities': {
                self.label_map[i][0]: float(p) 
                for i, p in enumerate(pred_proba)
            }
        }
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """추가 피처 생성"""
        df = df.copy()
        
        df['price_change_1'] = df['close'].pct_change(1)
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)
        df['volume_change_1'] = df['volume'].pct_change(1)
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        
        if 'rsi_14' in df.columns:
            df['rsi_normalized'] = df['rsi_14'] / 100
        
        if 'macd' in df.columns:
            df['macd_normalized'] = df['macd'] / df['close'] * 100
        
        if 'ema_12' in df.columns and 'ema_26' in df.columns:
            df['ema_cross'] = (df['ema_12'] - df['ema_26']) / df['close'] * 100
        
        return df
```

---

## 6. 실행 순서

```bash
# 1. 데이터 준비
cd backend
python scripts/prepare_training_data.py

# 2. 모델 학습 (XGBoost)
python scripts/train_model.py

# 3. (선택) LSTM 모델 학습
python scripts/train_lstm.py

# 4. 서버에서 학습된 모델 사용
# config에 모델 경로 설정 후 서버 재시작
```

---

## 7. 팁

### 7.1 데이터 품질
- **충분한 데이터**: 최소 1000개 이상의 캔들 권장
- **다양한 시장 상황**: 상승장, 하락장, 횡보장 모두 포함
- **이상치 처리**: 극단적인 변동 데이터 제거

### 7.2 피처 엔지니어링
- **시간 피처**: 요일, 시간대 추가
- **외부 데이터**: 비트코인 도미넌스, 공포/탐욕 지수
- **상관관계**: 다른 코인 가격 변화

### 7.3 모델 개선
- **하이퍼파라미터 튜닝**: GridSearchCV 사용
- **앙상블**: 여러 모델 조합
- **시계열 검증**: TimeSeriesSplit 사용

### 7.4 실전 적용
- **백테스팅**: 과거 데이터로 전략 검증
- **슬리피지/수수료**: 실제 거래 비용 반영
- **리스크 관리**: 포지션 크기 조절




