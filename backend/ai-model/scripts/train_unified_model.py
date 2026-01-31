"""
Train Unified LSTM Model
통합 LSTM 모델 학습 스크립트

Features:
- 통합 데이터셋 로드
- Sliding Window 시퀀스 생성
- PyTorch LSTM + Embedding 모델 학습
- Class Weight Balancing
- Early Stopping
- DB 메타데이터 저장
"""

import os
import sys
import asyncio
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any
import logging
import joblib

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import AsyncSessionLocal
from app.services.asset_mapping_service import AssetMappingService
from app.services.model_training_service import ModelTrainingService
from models.unified_lstm_model import UnifiedLSTMModel, create_unified_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedTimeSeriesDataset(Dataset):
    """통합 시계열 데이터셋 (Sliding Window)"""

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: list,
        sequence_length: int = 60
    ):
        """
        Args:
            data: 통합 데이터프레임 (symbol, asset_id, features, label 포함)
            feature_columns: 사용할 피처 목록
            sequence_length: 시퀀스 길이 (과거 몇 개 캔들 사용)
        """
        self.data = data
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length

        # 심볼별로 시퀀스 생성 (서로 다른 자산의 데이터가 섞이지 않도록)
        self.sequences = []
        self.asset_ids = []
        self.labels = []

        symbols = data['symbol'].unique()
        logger.info(f"Creating sequences for {len(symbols)} assets...")

        for symbol in symbols:
            symbol_data = data[data['symbol'] == symbol].reset_index(drop=True)

            if len(symbol_data) < sequence_length + 1:
                logger.warning(f"⚠️ {symbol}: Not enough data ({len(symbol_data)} < {sequence_length + 1})")
                continue

            asset_id = symbol_data['asset_id'].iloc[0]
            features = symbol_data[feature_columns].values
            labels_arr = symbol_data['label'].values

            # Sliding Window
            for i in range(len(symbol_data) - sequence_length):
                seq = features[i:i + sequence_length]  # (seq_len, num_features)
                label = labels_arr[i + sequence_length]  # 다음 시점 레이블

                self.sequences.append(seq)
                self.asset_ids.append(asset_id)
                self.labels.append(label)

            logger.info(f"  {symbol}: {len(symbol_data) - sequence_length} sequences")

        logger.info(f"✅ Total sequences: {len(self.sequences)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        asset_id = torch.LongTensor([self.asset_ids[idx]])
        label = torch.LongTensor([self.labels[idx]])

        return sequence, asset_id.squeeze(), label.squeeze()


def calculate_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """클래스 가중치 계산 (불균형 데이터 처리)"""
    class_counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)

    # 가중치 = total / (num_classes * count)
    weights = total / (num_classes * class_counts)

    logger.info(f"📊 Class distribution:")
    for i, count in enumerate(class_counts):
        logger.info(f"   Class {i}: {count} samples ({count / total * 100:.1f}%) → weight: {weights[i]:.3f}")

    return torch.FloatTensor(weights)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str
) -> Tuple[float, float]:
    """한 epoch 학습"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for sequences, asset_ids, labels in dataloader:
        sequences = sequences.to(device)
        asset_ids = asset_ids.to(device)
        labels = labels.to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(sequences, asset_ids)

        # Loss
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """검증/테스트"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, asset_ids, labels in dataloader:
            sequences = sequences.to(device)
            asset_ids = asset_ids.to(device)
            labels = labels.to(device)

            outputs = model(sequences, asset_ids)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


async def train_unified_model(
    data_path: str,
    output_dir: str,
    num_assets: int = 500,
    embedding_dim: int = 16,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    sequence_length: int = 60,
    batch_size: int = 64,
    epochs: int = 50,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 10,
    device: str = None
):
    """통합 모델 학습 메인 함수"""
    logger.info("🚀 Starting Unified LSTM Model Training...")

    # Device 설정
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"📱 Using device: {device}")

    # ===== 1. 데이터 로드 =====
    logger.info(f"📂 Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    logger.info(f"✅ Loaded {len(df)} samples")
    logger.info(f"   Assets: {df['symbol'].nunique()}")
    logger.info(f"   Date range: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

    # ===== 2. 피처 선택 =====
    # 통합 피처 엔지니어링으로 생성된 피처 사용
    exclude_cols = ['symbol', 'asset_id', 'timestamp', 'label', 'future_return',
                    'open_time', 'close_time', 'created_at', 'market_type', 'is_market_open']
    feature_columns = [col for col in df.columns if col not in exclude_cols]

    logger.info(f"📊 Using {len(feature_columns)} features")

    # NaN 처리
    df[feature_columns] = df[feature_columns].fillna(0)
    df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], 0)

    # ===== 3. 정규화 (StandardScaler) =====
    logger.info("🔧 Normalizing features...")
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    # ===== 4. Train/Test Split (시간순 유지) =====
    logger.info("✂️ Splitting train/test...")
    # 심볼별로 시간순으로 80/20 분할
    train_dfs = []
    test_dfs = []

    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].reset_index(drop=True)
        split_idx = int(len(symbol_df) * 0.8)

        train_dfs.append(symbol_df[:split_idx])
        test_dfs.append(symbol_df[split_idx:])

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    logger.info(f"   Train: {len(train_df)} samples")
    logger.info(f"   Test: {len(test_df)} samples")

    # ===== 5. Dataset 생성 =====
    logger.info("📦 Creating datasets...")
    train_dataset = UnifiedTimeSeriesDataset(train_df, feature_columns, sequence_length)
    test_dataset = UnifiedTimeSeriesDataset(test_df, feature_columns, sequence_length)

    # Class Weights
    train_labels = np.array([label for _, _, label in train_dataset])
    num_classes = len(np.unique(train_labels))
    class_weights = calculate_class_weights(train_labels, num_classes)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # ===== 6. 모델 생성 =====
    logger.info("🧠 Creating model...")
    model = create_unified_model(
        num_assets=num_assets,
        num_features=len(feature_columns),
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=False,
        device=device
    )

    # ===== 7. Loss & Optimizer =====
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ===== 8. 학습 루프 =====
    logger.info("🏋️ Training...")
    best_val_accuracy = 0
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, test_loader, criterion, device)

        logger.info(f"Epoch {epoch + 1}/{epochs}:")
        logger.info(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early Stopping
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            patience_counter = 0

            # 모델 저장
            model_path = Path(output_dir) / "unified_model_best.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_accuracy': val_acc,
            }, model_path)

            logger.info(f"   ✅ Best model saved (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            logger.info(f"⏹️ Early stopping at epoch {epoch + 1}")
            break

    # ===== 9. 최종 평가 =====
    logger.info("\n📊 Final Evaluation...")
    checkpoint = torch.load(Path(output_dir) / "unified_model_best.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    val_loss, val_acc, val_preds, val_labels = evaluate(model, test_loader, criterion, device)

    # Confusion Matrix
    cm = confusion_matrix(val_labels, val_preds)
    logger.info(f"\nConfusion Matrix:\n{cm}")

    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='weighted')
    logger.info(f"\nMetrics:")
    logger.info(f"   Accuracy: {val_acc:.4f}")
    logger.info(f"   Precision: {precision:.4f}")
    logger.info(f"   Recall: {recall:.4f}")
    logger.info(f"   F1 Score: {f1:.4f}")

    # ===== 10. 메타데이터 저장 =====
    # Scaler 저장
    scaler_path = Path(output_dir) / "unified_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    # Feature names 저장
    features_path = Path(output_dir) / "unified_features.joblib"
    joblib.dump(feature_columns, features_path)

    # 학습 자산 목록
    trained_assets = df['symbol'].unique().tolist()

    # DB 저장
    logger.info("\n💾 Saving metadata to database...")
    async with AsyncSessionLocal() as db:
        await ModelTrainingService.save_model_metadata(
            db=db,
            symbol="UNIFIED",
            timeframe="1h",
            model_type="unified_lstm",
            version=1,
            model_path=str(Path(output_dir) / "unified_model_best.pt"),
            scaler_path=str(scaler_path),
            features_path=str(features_path),
            num_classes=num_classes,
            num_features=len(feature_columns),
            feature_names=feature_columns,
            accuracy=val_acc,
            f1_score=f1,
            training_samples=len(train_dataset),
            test_samples=len(test_dataset),
        )

        # 통합 모델 필드 업데이트 (별도 쿼리)
        from app.models.trained_model import TrainedModel
        from sqlalchemy import select, update

        stmt = select(TrainedModel).where(
            TrainedModel.symbol == "UNIFIED",
            TrainedModel.model_type == "unified_lstm"
        ).order_by(TrainedModel.version.desc()).limit(1)

        result = await db.execute(stmt)
        model_record = result.scalar_one_or_none()

        if model_record:
            stmt = update(TrainedModel).where(TrainedModel.id == model_record.id).values(
                is_unified=True,
                supported_assets=trained_assets,
                asset_count=len(trained_assets),
                embedding_dim=embedding_dim,
                sequence_length=sequence_length
            )
            await db.execute(stmt)
            await db.commit()

    logger.info("✅ Training completed successfully!")


async def main():
    parser = argparse.ArgumentParser(description="Train Unified LSTM Model")
    parser.add_argument('--data', type=str, required=True, help='Path to unified dataset CSV')
    parser.add_argument('--output', type=str, default=None, help='Output directory for model')
    parser.add_argument('--num-assets', type=int, default=500, help='Max number of assets (default: 500)')
    parser.add_argument('--embedding-dim', type=int, default=16, help='Asset embedding dimension (default: 16)')
    parser.add_argument('--hidden-size', type=int, default=64, help='LSTM hidden size (default: 64)')
    parser.add_argument('--num-layers', type=int, default=2, help='LSTM layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (default: 0.3)')
    parser.add_argument('--sequence-length', type=int, default=60, help='Sequence length (default: 60)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (default: 10)')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda)')

    args = parser.parse_args()

    # 출력 디렉토리 설정
    if args.output is None:
        models_dir = PROJECT_ROOT / "ai-model" / "models"
        models_dir.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = str(models_dir / f"unified_{timestamp}")

    Path(args.output).mkdir(exist_ok=True, parents=True)

    # 학습 시작
    await train_unified_model(
        data_path=args.data,
        output_dir=args.output,
        num_assets=args.num_assets,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        early_stopping_patience=args.patience,
        device=args.device
    )


if __name__ == "__main__":
    asyncio.run(main())
