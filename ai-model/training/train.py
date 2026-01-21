"""
모델 학습 스크립트
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm
import json

from data_collector import DataCollector
from feature_engineering import FeatureEngineer
from model import get_model


class Trainer:
    def __init__(
        self,
        model_type: str = "lstm",
        sequence_length: int = 20,
        num_classes: int = 3,
        device: str = None
    ):
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.num_classes = num_classes

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None

    def prepare_data(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        start_date: str = "2023-01-01",
        threshold: float = 0.02,
        lookahead: int = 5
    ):
        """데이터 수집 및 전처리"""
        print("Collecting data...")
        collector = DataCollector()
        df = collector.fetch_historical_data(
            symbol=symbol,
            interval=interval,
            start_date=start_date
        )
        print(f"Collected {len(df)} candles")

        print("Adding technical indicators...")
        engineer = FeatureEngineer()
        df = engineer.add_technical_indicators(df)
        df = engineer.create_labels(df, threshold=threshold, lookahead=lookahead)

        print("Preparing dataset...")
        X, y = engineer.prepare_dataset(df, target_column="label")

        # 피처 컬럼 저장 (추론시 필요)
        self.feature_columns = [
            "sma_5", "sma_10", "sma_20", "sma_50",
            "ema_12", "ema_26",
            "rsi", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_lower", "bb_width",
            "atr", "volume_ratio",
            "price_change_1", "price_change_5", "price_change_10",
            "high_low_ratio"
        ]

        return X, y

    def create_dataloaders(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        batch_size: int = 32
    ):
        """데이터로더 생성"""
        # 스케일링
        X_scaled = self.scaler.fit_transform(X)

        # 시퀀스 생성
        engineer = FeatureEngineer()
        X_seq, y_seq = engineer.create_sequences(
            X_scaled, y, sequence_length=self.sequence_length
        )

        # Train/Test 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=test_size, shuffle=False
        )

        # 텐서 변환
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)

        # 데이터로더
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10
    ):
        """모델 학습"""
        input_size = train_loader.dataset[0][0].shape[-1]

        # 모델 초기화
        self.model = get_model(
            model_type=self.model_type,
            input_size=input_size,
            sequence_length=self.sequence_length,
            num_classes=self.num_classes
        ).to(self.device)

        # 클래스 불균형 처리를 위한 가중치
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.numpy())
        class_counts = np.bincount(all_labels, minlength=self.num_classes)
        class_weights = 1.0 / (class_counts + 1)
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.FloatTensor(class_weights).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        print(f"\nTraining on {self.device}...")
        print(f"Model: {self.model_type}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0

            for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            val_loss, val_acc = self.evaluate(test_loader, criterion)

            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                self.save_model("../models/best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        return history

    def evaluate(self, test_loader: DataLoader, criterion=None):
        """모델 평가"""
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def save_model(self, filepath: str):
        """모델 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_type": self.model_type,
            "sequence_length": self.sequence_length,
            "num_classes": self.num_classes,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "feature_columns": self.feature_columns
        }

        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model_type = checkpoint["model_type"]
        self.sequence_length = checkpoint["sequence_length"]
        self.num_classes = checkpoint["num_classes"]
        self.feature_columns = checkpoint["feature_columns"]

        # Scaler 복원
        self.scaler.mean_ = np.array(checkpoint["scaler_mean"])
        self.scaler.scale_ = np.array(checkpoint["scaler_scale"])

        # 모델 복원
        input_size = len(self.feature_columns)
        self.model = get_model(
            model_type=self.model_type,
            input_size=input_size,
            sequence_length=self.sequence_length,
            num_classes=self.num_classes
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"Model loaded from {filepath}")

    def predict(self, X: np.ndarray) -> tuple:
        """
        예측 수행

        Returns:
            predicted_class: 예측 클래스 (0: 하락, 1: 횡보, 2: 상승)
            probabilities: 각 클래스의 확률
        """
        self.model.eval()

        # 스케일링
        X_scaled = self.scaler.transform(X)

        # 시퀀스 형태로 변환
        X_seq = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)

        return predicted_class, probabilities


if __name__ == "__main__":
    # 학습 실행
    trainer = Trainer(model_type="lstm", sequence_length=20, num_classes=3)

    # 데이터 준비
    X, y = trainer.prepare_data(
        symbol="BTCUSDT",
        interval="1h",
        start_date="2023-01-01",
        threshold=0.02,
        lookahead=5
    )

    print(f"Dataset: X shape={X.shape}, y shape={y.shape}")
    print(f"Label distribution: {np.bincount(y.astype(int))}")

    # 데이터로더 생성
    train_loader, test_loader = trainer.create_dataloaders(
        X, y, test_size=0.2, batch_size=32
    )

    # 학습
    history = trainer.train(
        train_loader,
        test_loader,
        epochs=50,
        learning_rate=0.001,
        early_stopping_patience=10
    )

    # 최종 평가
    trainer.load_model("../models/best_model.pt")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
