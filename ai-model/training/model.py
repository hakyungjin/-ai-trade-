"""
가격 예측 AI 모델 정의
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class LSTMClassifier(nn.Module):
    """
    LSTM 기반 가격 방향 예측 모델

    입력: 과거 N개 캔들의 기술적 지표
    출력: 상승/하락/횡보 확률
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)

        # Attention
        attn_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1), dim=1
        )  # (batch, seq_len)

        # Weighted sum
        context = torch.bmm(
            attn_weights.unsqueeze(1), lstm_out
        ).squeeze(1)  # (batch, hidden)

        # Classification
        out = self.classifier(context)
        return out


class TransformerClassifier(nn.Module):
    """
    Transformer 기반 가격 예측 모델
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection
        x = self.input_projection(x)

        # Positional encoding
        x = self.positional_encoding(x)

        # Transformer
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classification
        out = self.classifier(x)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SimpleMLP(nn.Module):
    """
    간단한 MLP 모델 (빠른 학습/추론용)
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list = [256, 128, 64],
        num_classes: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten if sequence input
        if len(x.shape) == 3:
            x = x.reshape(x.size(0), -1)
        return self.network(x)


def get_model(
    model_type: str,
    input_size: int,
    sequence_length: int = 20,
    num_classes: int = 3
) -> nn.Module:
    """
    모델 타입에 따라 적절한 모델 반환
    """
    if model_type == "lstm":
        return LSTMClassifier(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            num_classes=num_classes
        )
    elif model_type == "transformer":
        return TransformerClassifier(
            input_size=input_size,
            d_model=128,
            nhead=4,
            num_layers=2,
            num_classes=num_classes
        )
    elif model_type == "mlp":
        return SimpleMLP(
            input_size=input_size * sequence_length,
            hidden_sizes=[512, 256, 128],
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # 모델 테스트
    batch_size = 32
    sequence_length = 20
    input_size = 18  # 피처 수

    # LSTM 테스트
    model = LSTMClassifier(input_size=input_size)
    x = torch.randn(batch_size, sequence_length, input_size)
    out = model(x)
    print(f"LSTM output shape: {out.shape}")  # (32, 3)

    # Transformer 테스트
    model = TransformerClassifier(input_size=input_size)
    out = model(x)
    print(f"Transformer output shape: {out.shape}")  # (32, 3)

    # MLP 테스트
    model = SimpleMLP(input_size=input_size * sequence_length)
    x_flat = x.reshape(batch_size, -1)
    out = model(x_flat)
    print(f"MLP output shape: {out.shape}")  # (32, 3)
