"""
Unified LSTM Model with Asset Embedding
단일 모델로 여러 자산(코인, 주식)을 학습하고 예측

Architecture:
- Input 1: Time Series (60 candles × features)
- Input 2: Asset ID (embedded as vector)
- LSTM Layer: Process time series
- Concatenate: LSTM output + Asset embedding
- Output: Classification (BUY/HOLD/SELL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class UnifiedLSTMModel(nn.Module):
    """
    통합 LSTM 모델 (Asset Embedding 포함)

    Args:
        num_assets: 총 자산 개수 (embedding lookup table 크기)
        embedding_dim: Asset embedding 벡터 차원
        num_features: 입력 피처 개수
        hidden_size: LSTM hidden state 크기
        num_classes: 출력 클래스 개수 (2, 3, 5)
        num_layers: LSTM 레이어 개수
        dropout: Dropout 비율 (과적합 방지)
        bidirectional: 양방향 LSTM 사용 여부
    """

    def __init__(
        self,
        num_assets: int,
        embedding_dim: int = 16,
        num_features: int = 100,
        hidden_size: int = 64,
        num_classes: int = 3,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False
    ):
        super(UnifiedLSTMModel, self).__init__()

        self.num_assets = num_assets
        self.embedding_dim = embedding_dim
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # ===== Asset Embedding Layer =====
        # 자산 ID를 벡터로 변환 (학습 가능한 lookup table)
        self.asset_embedding = nn.Embedding(
            num_embeddings=num_assets,
            embedding_dim=embedding_dim,
            padding_idx=None
        )

        # ===== Time Series Processing: LSTM =====
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # LSTM 출력 크기 (양방향이면 2배)
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # ===== Dropout Layer =====
        self.dropout = nn.Dropout(dropout)

        # ===== Fully Connected Layers =====
        # LSTM 출력 + Asset Embedding 결합
        combined_size = lstm_output_size + embedding_dim

        self.fc1 = nn.Linear(combined_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

        # Batch Normalization (선택적)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)

    def forward(
        self,
        time_series: torch.Tensor,
        asset_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            time_series: (batch_size, sequence_length, num_features)
            asset_ids: (batch_size,) - 각 샘플의 자산 ID

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = time_series.size(0)

        # ===== 1. Asset Embedding =====
        # (batch_size,) → (batch_size, embedding_dim)
        asset_embed = self.asset_embedding(asset_ids)

        # ===== 2. LSTM Processing =====
        # LSTM forward: (batch_size, seq_len, num_features) → (batch_size, seq_len, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(time_series)

        # 마지막 시점의 hidden state 사용
        if self.bidirectional:
            # 양방향 LSTM: forward와 backward의 마지막 hidden state 결합
            h_forward = h_n[-2, :, :]  # (batch_size, hidden_size)
            h_backward = h_n[-1, :, :]  # (batch_size, hidden_size)
            lstm_final = torch.cat([h_forward, h_backward], dim=1)
        else:
            # 단방향 LSTM: 마지막 레이어의 hidden state
            lstm_final = h_n[-1, :, :]  # (batch_size, hidden_size)

        # Dropout 적용
        lstm_final = self.dropout(lstm_final)

        # ===== 3. Concatenate LSTM + Embedding =====
        # (batch_size, lstm_output_size) + (batch_size, embedding_dim)
        # → (batch_size, lstm_output_size + embedding_dim)
        combined = torch.cat([lstm_final, asset_embed], dim=1)

        # ===== 4. Fully Connected Layers =====
        x = F.relu(self.bn1(self.fc1(combined)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        logits = self.fc3(x)  # (batch_size, num_classes)

        return logits

    def predict(
        self,
        time_series: torch.Tensor,
        asset_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction (evaluation mode)

        Returns:
            predictions: (batch_size,) - 예측 클래스 (0, 1, 2, ...)
            probabilities: (batch_size, num_classes) - 각 클래스 확률
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(time_series, asset_ids)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        return predictions, probabilities

    def get_embedding(self, asset_id: int) -> torch.Tensor:
        """특정 자산의 embedding 벡터 조회 (분석용)"""
        asset_id_tensor = torch.tensor([asset_id], dtype=torch.long)
        return self.asset_embedding(asset_id_tensor).squeeze(0)


# ===== 모델 생성 헬퍼 함수 =====

def create_unified_model(
    num_assets: int,
    num_features: int,
    num_classes: int = 3,
    embedding_dim: int = 16,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    bidirectional: bool = False,
    device: str = 'cpu'
) -> UnifiedLSTMModel:
    """
    통합 모델 생성

    Example:
        model = create_unified_model(
            num_assets=200,  # 최대 200개 자산
            num_features=100,  # 100개 피처
            num_classes=3,  # BUY/HOLD/SELL
            embedding_dim=16,
            hidden_size=64
        )
    """
    model = UnifiedLSTMModel(
        num_assets=num_assets,
        embedding_dim=embedding_dim,
        num_features=num_features,
        hidden_size=hidden_size,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional
    )

    model = model.to(device)

    # 파라미터 개수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✅ Unified LSTM Model created:")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Device: {device}")

    return model


if __name__ == "__main__":
    # 테스트
    print("🧪 Testing Unified LSTM Model...")

    # 모델 생성
    model = create_unified_model(
        num_assets=200,
        num_features=100,
        num_classes=3,
        embedding_dim=16,
        hidden_size=64
    )

    # 더미 입력 생성
    batch_size = 8
    sequence_length = 60
    num_features = 100

    dummy_time_series = torch.randn(batch_size, sequence_length, num_features)
    dummy_asset_ids = torch.randint(0, 200, (batch_size,))

    print(f"\n📊 Input shapes:")
    print(f"   - Time series: {dummy_time_series.shape}")
    print(f"   - Asset IDs: {dummy_asset_ids.shape}")

    # Forward pass
    logits = model(dummy_time_series, dummy_asset_ids)
    print(f"\n📈 Output shape: {logits.shape}")

    # Prediction
    predictions, probabilities = model.predict(dummy_time_series, dummy_asset_ids)
    print(f"\n🎯 Predictions: {predictions}")
    print(f"   Probabilities shape: {probabilities.shape}")

    # Embedding 확인
    btc_embedding = model.get_embedding(0)  # BTC = asset_id 0
    print(f"\n🪙 BTC Embedding: {btc_embedding[:5]}... (first 5 dims)")

    print("\n✅ Test passed!")
