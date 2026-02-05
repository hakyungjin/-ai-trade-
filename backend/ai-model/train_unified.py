"""
통합 LSTM 모델 학습 스크립트
- 여러 자산(코인)을 하나의 모델로 학습
- Asset Embedding으로 각 자산의 특성 학습
- 성능 평가 (Accuracy, Precision, Recall, F1, Confusion Matrix)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
from datetime import datetime

# 경로 설정
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from training.data_collector import DataCollector
from training.feature_engineering import FeatureEngineer
from models.unified_lstm_model import UnifiedLSTMModel, create_unified_model


class UnifiedTrainer:
    """통합 모델 학습기"""

    def __init__(
        self,
        sequence_length: int = 60,
        num_classes: int = 3,
        embedding_dim: int = 16,
        hidden_size: int = 64,
        device: str = None
    ):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.scaler = StandardScaler()
        self.asset_to_id = {}  # 자산명 -> ID 매핑
        self.id_to_asset = {}  # ID -> 자산명 매핑
        self.feature_columns = []

    def collect_multi_asset_data(
        self,
        symbols: list = None,
        interval: str = "1h",
        start_date: str = "2024-01-01",
        threshold: float = 0.02,
        lookahead: int = 5
    ) -> tuple:
        """
        여러 자산의 데이터 수집 및 전처리

        Returns:
            X_all: 전체 피처 배열
            y_all: 전체 레이블 배열
            asset_ids_all: 각 샘플의 자산 ID 배열
        """
        if symbols is None:
            symbols = [
                "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
                "DOGEUSDT", "SOLUSDT", "DOTUSDT", "MATICUSDT", "LTCUSDT"
            ]

        print(f"\n{'='*60}")
        print(f"Collecting data for {len(symbols)} assets...")
        print(f"{'='*60}")

        collector = DataCollector()
        engineer = FeatureEngineer(use_unified=False)  # 기본 피처 사용

        all_data = []

        for idx, symbol in enumerate(symbols):
            print(f"\n[{idx+1}/{len(symbols)}] Processing {symbol}...")

            # Asset ID 매핑
            self.asset_to_id[symbol] = idx
            self.id_to_asset[idx] = symbol

            try:
                # 데이터 수집
                df = collector.fetch_historical_data(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date
                )
                print(f"  - Collected {len(df)} candles")

                if len(df) < self.sequence_length + lookahead + 50:
                    print(f"  - WARNING: Not enough data, skipping...")
                    continue

                # 기술적 지표 추가
                df = engineer.add_technical_indicators(df)
                df = engineer.create_labels(df, threshold=threshold, lookahead=lookahead)

                # NaN 제거
                df = df.dropna()

                # 피처 준비
                X, y, feature_names = engineer.prepare_dataset(df, target_column="label")

                if len(self.feature_columns) == 0:
                    self.feature_columns = feature_names

                print(f"  - Features: {len(feature_names)}, Samples: {len(X)}")

                # 자산 ID 배열 생성
                asset_ids = np.full(len(X), idx, dtype=np.int64)

                all_data.append({
                    'symbol': symbol,
                    'X': X,
                    'y': y,
                    'asset_ids': asset_ids
                })

                # 레이블 분포
                label_counts = np.bincount(y.astype(int), minlength=self.num_classes)
                print(f"  - Labels: Down={label_counts[0]}, Hold={label_counts[1]}, Up={label_counts[2]}")

            except Exception as e:
                print(f"  - ERROR: {e}")
                continue

        if len(all_data) == 0:
            raise ValueError("No data collected!")

        # 모든 데이터 결합
        X_all = np.vstack([d['X'] for d in all_data])
        y_all = np.hstack([d['y'] for d in all_data])
        asset_ids_all = np.hstack([d['asset_ids'] for d in all_data])

        print(f"\n{'='*60}")
        print(f"Total dataset: {len(X_all)} samples from {len(all_data)} assets")
        print(f"Features: {X_all.shape[1]}")
        print(f"{'='*60}")

        return X_all, y_all, asset_ids_all

    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        asset_ids: np.ndarray
    ) -> tuple:
        """시퀀스 데이터 생성"""
        X_seq = []
        y_seq = []
        asset_seq = []

        # 자산별로 시퀀스 생성 (자산 경계에서 시퀀스가 섞이지 않도록)
        unique_assets = np.unique(asset_ids)

        for asset_id in unique_assets:
            mask = asset_ids == asset_id
            X_asset = X[mask]
            y_asset = y[mask]

            for i in range(len(X_asset) - self.sequence_length):
                X_seq.append(X_asset[i:i + self.sequence_length])
                y_seq.append(y_asset[i + self.sequence_length])
                asset_seq.append(asset_id)

        return np.array(X_seq), np.array(y_seq), np.array(asset_seq)

    def create_dataloaders(
        self,
        X: np.ndarray,
        y: np.ndarray,
        asset_ids: np.ndarray,
        test_size: float = 0.2,
        batch_size: int = 32
    ) -> tuple:
        """데이터로더 생성"""
        # 스케일링
        X_flat = X.reshape(-1, X.shape[-1])
        self.scaler.fit(X_flat)
        X_scaled = self.scaler.transform(X_flat).reshape(X.shape)

        # 시퀀스 생성
        X_seq, y_seq, asset_seq = self.create_sequences(X_scaled, y, asset_ids)

        print(f"\nSequence data: {X_seq.shape}")

        # Train/Test 분할 (시간순 유지)
        split_idx = int(len(X_seq) * (1 - test_size))

        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        asset_train, asset_test = asset_seq[:split_idx], asset_seq[split_idx:]

        # 텐서 변환
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train.astype(int))
        y_test = torch.LongTensor(y_test.astype(int))
        asset_train = torch.LongTensor(asset_train)
        asset_test = torch.LongTensor(asset_test)

        # 데이터로더
        train_dataset = TensorDataset(X_train, asset_train, y_train)
        test_dataset = TensorDataset(X_test, asset_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

        return train_loader, test_loader

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 15
    ) -> dict:
        """모델 학습"""
        num_features = train_loader.dataset[0][0].shape[-1]
        num_assets = len(self.asset_to_id)

        print(f"\n{'='*60}")
        print(f"Training Unified LSTM Model")
        print(f"{'='*60}")
        print(f"Assets: {num_assets}")
        print(f"Features: {num_features}")
        print(f"Sequence Length: {self.sequence_length}")
        print(f"Classes: {self.num_classes}")
        print(f"Device: {self.device}")

        # 모델 생성
        self.model = create_unified_model(
            num_assets=num_assets,
            num_features=num_features,
            num_classes=self.num_classes,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            device=str(self.device)
        )

        # 클래스 가중치 계산 (불균형 처리)
        all_labels = []
        for _, _, labels in train_loader:
            all_labels.extend(labels.numpy())
        class_counts = np.bincount(all_labels, minlength=self.num_classes)
        class_weights = 1.0 / (class_counts + 1)
        class_weights = class_weights / class_weights.sum() * self.num_classes
        class_weights = torch.FloatTensor(class_weights).to(self.device)

        print(f"Class weights: {class_weights.cpu().numpy()}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        best_loss = float("inf")
        best_f1 = 0
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for X_batch, asset_batch, y_batch in pbar:
                X_batch = X_batch.to(self.device)
                asset_batch = asset_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch, asset_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

                pbar.set_postfix({'loss': loss.item()})

            train_loss /= len(train_loader)

            # Validation
            val_loss, val_acc, val_f1, _, _ = self.evaluate(test_loader, criterion)
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")

            # Early stopping (F1 기준)
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_loss = val_loss
                patience_counter = 0
                self.save_model("models/unified_best_model.pt")
                print(f"  -> New best model saved! (F1: {best_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        return history

    def evaluate(self, test_loader: DataLoader, criterion=None) -> tuple:
        """모델 평가"""
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, asset_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                asset_batch = asset_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch, asset_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(test_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return avg_loss, accuracy, f1, all_preds, all_labels

    def full_evaluation(self, test_loader: DataLoader):
        """상세 성능 평가"""
        print(f"\n{'='*60}")
        print("Model Evaluation Report")
        print(f"{'='*60}")

        _, accuracy, f1, all_preds, all_labels = self.evaluate(test_loader)

        # Classification Report
        target_names = ['SELL (0)', 'HOLD (1)', 'BUY (2)']
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=target_names))

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        print("\nConfusion Matrix:")
        print(f"{'':12} | Pred SELL | Pred HOLD | Pred BUY")
        print("-" * 50)
        for i, row in enumerate(cm):
            print(f"True {target_names[i]:6} | {row[0]:9} | {row[1]:9} | {row[2]:8}")

        # 자산별 성능
        print(f"\n{'='*60}")
        print("Per-Asset Performance")
        print(f"{'='*60}")

        self.model.eval()
        asset_results = {asset: {'preds': [], 'labels': []} for asset in self.asset_to_id}

        with torch.no_grad():
            for X_batch, asset_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                asset_batch = asset_batch.to(self.device)

                outputs = self.model(X_batch, asset_batch)
                _, predicted = torch.max(outputs, 1)

                for i, asset_id in enumerate(asset_batch.cpu().numpy()):
                    asset_name = self.id_to_asset[asset_id]
                    asset_results[asset_name]['preds'].append(predicted[i].item())
                    asset_results[asset_name]['labels'].append(y_batch[i].item())

        print(f"{'Asset':12} | {'Accuracy':10} | {'F1 Score':10} | {'Samples':8}")
        print("-" * 50)

        for asset, data in asset_results.items():
            if len(data['preds']) > 0:
                acc = np.mean(np.array(data['preds']) == np.array(data['labels']))
                f1 = f1_score(data['labels'], data['preds'], average='weighted', zero_division=0)
                print(f"{asset:12} | {acc:10.4f} | {f1:10.4f} | {len(data['preds']):8}")

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'per_asset': {
                asset: {
                    'accuracy': np.mean(np.array(data['preds']) == np.array(data['labels'])) if data['preds'] else 0,
                    'samples': len(data['preds'])
                }
                for asset, data in asset_results.items()
            }
        }

    def save_model(self, filepath: str):
        """모델 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "num_assets": len(self.asset_to_id),
                "embedding_dim": self.embedding_dim,
                "num_features": len(self.feature_columns),
                "hidden_size": self.hidden_size,
                "num_classes": self.num_classes,
                "sequence_length": self.sequence_length
            },
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "asset_to_id": self.asset_to_id,
            "id_to_asset": {str(k): v for k, v in self.id_to_asset.items()},
            "feature_columns": self.feature_columns,
            "saved_at": datetime.now().isoformat()
        }

        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)

        config = checkpoint["model_config"]
        self.embedding_dim = config["embedding_dim"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        self.sequence_length = config["sequence_length"]
        self.feature_columns = checkpoint["feature_columns"]
        self.asset_to_id = checkpoint["asset_to_id"]
        self.id_to_asset = {int(k): v for k, v in checkpoint["id_to_asset"].items()}

        # Scaler 복원
        self.scaler.mean_ = np.array(checkpoint["scaler_mean"])
        self.scaler.scale_ = np.array(checkpoint["scaler_scale"])

        # 모델 복원
        self.model = create_unified_model(
            num_assets=config["num_assets"],
            num_features=config["num_features"],
            num_classes=config["num_classes"],
            embedding_dim=config["embedding_dim"],
            hidden_size=config["hidden_size"],
            device=str(self.device)
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"Model loaded from {filepath}")
        print(f"  - Assets: {list(self.asset_to_id.keys())}")
        print(f"  - Features: {len(self.feature_columns)}")


def main():
    """메인 학습 파이프라인"""
    print("=" * 60)
    print("Unified LSTM Model Training Pipeline")
    print("=" * 60)

    # 설정
    SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
        "SOLUSDT", "DOGEUSDT", "DOTUSDT", "MATICUSDT", "AVAXUSDT"
    ]
    INTERVAL = "1h"
    START_DATE = "2024-01-01"
    THRESHOLD = 0.015  # 1.5% 변동 기준
    LOOKAHEAD = 6  # 6시간 후 예측
    SEQUENCE_LENGTH = 60  # 60시간 시퀀스
    EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001

    # Trainer 초기화
    trainer = UnifiedTrainer(
        sequence_length=SEQUENCE_LENGTH,
        num_classes=3,
        embedding_dim=16,
        hidden_size=64
    )

    # 1. 데이터 수집
    print("\n[1/4] Collecting multi-asset data...")
    X, y, asset_ids = trainer.collect_multi_asset_data(
        symbols=SYMBOLS,
        interval=INTERVAL,
        start_date=START_DATE,
        threshold=THRESHOLD,
        lookahead=LOOKAHEAD
    )

    # 2. 데이터로더 생성
    print("\n[2/4] Creating dataloaders...")
    train_loader, test_loader = trainer.create_dataloaders(
        X, y, asset_ids,
        test_size=0.2,
        batch_size=BATCH_SIZE
    )

    # 3. 학습
    print("\n[3/4] Training model...")
    history = trainer.train(
        train_loader,
        test_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        early_stopping_patience=15
    )

    # 4. 최종 평가
    print("\n[4/4] Final evaluation...")
    trainer.load_model("models/unified_best_model.pt")
    eval_results = trainer.full_evaluation(test_loader)

    # 히스토리 저장
    history_path = "models/unified_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    # 평가 결과 저장
    eval_path = "models/unified_evaluation.json"
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"Evaluation results saved to {eval_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final Accuracy: {eval_results['accuracy']:.4f}")
    print(f"Final F1 Score: {eval_results['f1_score']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
