"""
전체 학습 파이프라인 실행 스크립트
- 데이터 수집
- 피처 엔지니어링
- 모델 학습
- 모델 저장
"""
import os
import sys

# 현재 디렉토리를 training 폴더로 설정
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))

from training.data_collector import DataCollector
from training.feature_engineering import FeatureEngineer
from training.train import Trainer
import numpy as np


def main():
    print("=" * 60)
    print("Crypto AI Trader - Model Training Pipeline")
    print("=" * 60)

    # 설정
    SYMBOL = "BTCUSDT"
    INTERVAL = "1h"
    START_DATE = "2023-01-01"  # 약 2년치 데이터
    THRESHOLD = 0.02  # 2% 변동을 상승/하락으로 판단
    LOOKAHEAD = 5  # 5캔들 후 가격 예측
    MODEL_TYPE = "lstm"  # lstm, transformer, mlp
    EPOCHS = 50
    BATCH_SIZE = 32

    # 1. 데이터 수집
    print("\n[1/4] Collecting historical data from Binance...")
    collector = DataCollector()

    try:
        df = collector.fetch_historical_data(
            symbol=SYMBOL,
            interval=INTERVAL,
            start_date=START_DATE
        )
        print(f"  - Collected {len(df)} candles")
        print(f"  - Date range: {df.index[0]} ~ {df.index[-1]}")

        # 데이터 저장
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        data_path = os.path.join(data_dir, f'{SYMBOL.lower()}_{INTERVAL}.csv')
        collector.save_data(df, data_path)
    except Exception as e:
        print(f"  - Error collecting data: {e}")
        print("  - Trying with public API (no API key)...")
        # Binance public API로 재시도
        from binance.client import Client
        client = Client("", "")  # 공개 API
        klines = client.get_historical_klines(
            SYMBOL,
            Client.KLINE_INTERVAL_1HOUR,
            START_DATE
        )
        import pandas as pd
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df.set_index("timestamp", inplace=True)
        print(f"  - Collected {len(df)} candles via public API")

    # 2. 피처 엔지니어링
    print("\n[2/4] Engineering features...")
    engineer = FeatureEngineer()
    df = engineer.add_technical_indicators(df)
    df = engineer.create_labels(df, threshold=THRESHOLD, lookahead=LOOKAHEAD)

    print(f"  - Added technical indicators")
    print(f"  - Features: {len([c for c in df.columns if c not in ['label', 'label_binary', 'future_return']])}")

    # 레이블 분포 확인
    label_counts = df['label'].value_counts().sort_index()
    print(f"  - Label distribution:")
    print(f"    - Down (0): {label_counts.get(0, 0)}")
    print(f"    - Neutral (1): {label_counts.get(1, 0)}")
    print(f"    - Up (2): {label_counts.get(2, 0)}")

    # 3. 데이터셋 준비
    print("\n[3/4] Preparing dataset...")
    X, y = engineer.prepare_dataset(df, target_column="label")
    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")

    # NaN 체크
    if np.isnan(X).any():
        print("  - Warning: NaN values found, removing...")
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]
        print(f"  - After cleaning: X shape: {X.shape}")

    # 4. 모델 학습
    print("\n[4/4] Training model...")
    trainer = Trainer(
        model_type=MODEL_TYPE,
        sequence_length=20,
        num_classes=3
    )
    trainer.scaler.fit(X)
    trainer.feature_columns = [
        "sma_5", "sma_10", "sma_20", "sma_50",
        "ema_12", "ema_26",
        "rsi", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_lower", "bb_width",
        "atr", "volume_ratio",
        "price_change_1", "price_change_5", "price_change_10",
        "high_low_ratio"
    ]

    train_loader, test_loader = trainer.create_dataloaders(
        X, y, test_size=0.2, batch_size=BATCH_SIZE
    )

    print(f"  - Model type: {MODEL_TYPE}")
    print(f"  - Train samples: {len(train_loader.dataset)}")
    print(f"  - Test samples: {len(test_loader.dataset)}")

    # 학습 실행
    history = trainer.train(
        train_loader,
        test_loader,
        epochs=EPOCHS,
        learning_rate=0.001,
        early_stopping_patience=10
    )

    # 5. 최종 평가 및 저장
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # 최고 모델 로드 및 평가
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, 'best_model.pt')

    if os.path.exists(best_model_path):
        trainer.load_model(best_model_path)
        test_loss, test_acc = trainer.evaluate(test_loader)
        print(f"\nFinal Test Results:")
        print(f"  - Loss: {test_loss:.4f}")
        print(f"  - Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")

    # 백엔드용 모델 저장
    backend_model_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'backend', 'models', 'price_predictor.pt'
    )
    os.makedirs(os.path.dirname(backend_model_path), exist_ok=True)
    trainer.save_model(backend_model_path)
    print(f"\nModel saved for backend: {backend_model_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
