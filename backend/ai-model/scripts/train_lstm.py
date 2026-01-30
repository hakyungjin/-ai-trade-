"""
PyTorch LSTM 시계열 모델 학습
- 시퀀스(20봉)를 입력으로 받아 패턴 학습
- XGBoost와 앙상블 가능
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# 경로 설정
AI_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AI_MODEL_DIR)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class FocalLoss(nn.Module):
    """Focal Loss - 어려운 샘플에 집중 (BUY/SELL 강화)"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 클래스 가중치
        self.gamma = gamma  # focusing parameter (2.0 권장)
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # 예측 확률
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ===== 시퀀스 피처 (LSTM에 적합한 피처만 선택) =====
SEQUENCE_FEATURES = [
    # 가격 관련
    'price_change_1', 'price_change_5',
    'price_position', 'price_position_20',
    
    # 기술적 지표 (정규화된 것들)
    'rsi_normalized', 'bb_position', 'stoch_k', 'stoch_d',
    'macd_normalized', 'ema_cross',
    
    # 거래량
    'volume_ma_ratio', 'volume_spike',
    
    # OBV, MFI
    'obv_slope', 'mfi_normalized', 'williams_r',
    
    # 변동성
    'volatility_5', 'atr_ratio',
    
    # 캔들 패턴
    'candle_body', 'upper_shadow', 'lower_shadow', 'is_bullish',
]


class SequenceDataset(Dataset):
    """PyTorch 시퀀스 데이터셋"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Attention(nn.Module):
    """Self-Attention 레이어 - 중요한 시점에 집중"""
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        attention_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # 가중 평균
        context = torch.sum(lstm_output * attention_weights, dim=1)  # (batch, hidden_size)
        return context, attention_weights


class BiLSTMClassifier(nn.Module):
    """양방향 LSTM + Attention 분류 모델"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=3, dropout=0.4):
        super(BiLSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 양방향 LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention 레이어 (어떤 시점이 중요한지 학습)
        self.attention = Attention(hidden_size * 2)
        
        # BatchNorm
        self.bn = nn.BatchNorm1d(hidden_size * 2)
        
        # Fully Connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention - 중요한 시점에 집중
        context, _ = self.attention(lstm_out)  # (batch, hidden*2)
        
        # BatchNorm
        context = self.bn(context)
        
        # FC layers
        out = self.fc(context)
        return out


def load_training_data(filepath: str) -> pd.DataFrame:
    """학습 데이터 로드"""
    df = pd.read_csv(filepath, index_col=0)
    print(f"✅ Loaded {len(df)} samples from {filepath}")
    return df


def create_sequences(df: pd.DataFrame, sequence_length: int = 20):
    """시퀀스 데이터 생성"""
    # 사용할 피처만 선택 (존재하는 것만)
    available_features = [f for f in SEQUENCE_FEATURES if f in df.columns]
    print(f"📊 Using {len(available_features)} features: {available_features[:5]}...")
    
    # NaN 처리
    df = df.dropna(subset=available_features + ['label'])
    
    # 피처 데이터
    feature_data = df[available_features].values
    labels = df['label'].values
    
    # 시퀀스 생성
    X, y = [], []
    for i in range(len(feature_data) - sequence_length):
        X.append(feature_data[i:i + sequence_length])
        y.append(labels[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"📦 Sequence shape: X={X.shape}, y={y.shape}")
    return X, y, available_features


def normalize_sequences(X_train, X_test):
    """시퀀스 데이터 정규화"""
    n_samples, seq_len, n_features = X_train.shape
    
    X_train_flat = X_train.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    X_train_scaled = X_train_scaled.reshape(n_samples, seq_len, n_features)
    X_test_scaled = X_test_scaled.reshape(X_test.shape[0], seq_len, n_features)
    
    return X_train_scaled, X_test_scaled, scaler


def train_lstm(
    symbol: str = 'BTCUSDT',
    timeframe: str = '5m',
    sequence_length: int = 20,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001
):
    """LSTM 모델 학습"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    # 1. 데이터 로드
    data_path = os.path.join(AI_MODEL_DIR, 'data', f'{symbol.lower()}_{timeframe}_training.csv')
    if not os.path.exists(data_path):
        print(f"❌ Training data not found: {data_path}")
        print(f"   Run: python scripts/prepare_training_data.py --symbol {symbol} --timeframe {timeframe}")
        return None
    
    df = load_training_data(data_path)
    
    # 2. 시퀀스 생성
    X, y, feature_names = create_sequences(df, sequence_length)
    
    # 레이블 변환 (3클래스: -1,0,1 → 0,1,2)
    unique_labels = sorted(set(y))
    n_classes = len(unique_labels)
    print(f"🏷️ Classes: {unique_labels} → {n_classes} classes")
    
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_mapped = np.array([label_map[label] for label in y])
    
    # 3. Train/Test 분할 (시계열이므로 섞지 않음!)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_mapped[:split_idx], y_mapped[split_idx:]
    
    print(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 4. 정규화
    X_train, X_test, scaler = normalize_sequences(X_train, X_test)
    
    # 5. 클래스 가중치 (불균형 보정)
    class_counts = np.bincount(y_train)
    total = len(y_train)
    class_weights = torch.FloatTensor([total / (n_classes * count) for count in class_counts]).to(device)
    print(f"⚖️ Class weights: {class_weights.tolist()}")
    
    # 6. DataLoader 생성
    train_dataset = SequenceDataset(X_train, y_train)
    test_dataset = SequenceDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 7. 모델 생성
    model = BiLSTMClassifier(
        input_size=len(feature_names),
        hidden_size=64,
        num_layers=2,
        num_classes=n_classes,
        dropout=0.3
    ).to(device)
    
    print(f"\n🏗️ Model architecture:")
    print(model)
    
    # 8. Loss & Optimizer (단순하게!)
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # Focal Loss 대신 일반 CE
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # AdamW 대신 Adam
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)  # accuracy 기준
    
    # 9. 학습
    print(f"\n🚀 Training LSTM model (Focal Loss + 과적합 방지)...")
    best_accuracy = 0
    patience = 10  # 더 빠른 Early Stopping
    patience_counter = 0
    
    model_dir = os.path.join(AI_MODEL_DIR, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'lstm_{symbol.lower()}_{timeframe}.pt')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        train_acc = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        
        val_acc = 100. * correct / total
        avg_val_loss = val_loss / len(test_loader)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping & Model saving
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"   ✅ Best model saved! (Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                break
    
    # 10. 최종 평가
    print(f"\n📊 Final Evaluation:")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    label_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY'} if n_classes == 3 else {
        0: 'STRONG_SELL', 1: 'SELL', 2: 'HOLD', 3: 'BUY', 4: 'STRONG_BUY'
    }
    target_names = [label_names.get(i, str(i)) for i in range(n_classes)]
    
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    # 11. 메타데이터 저장 (모델 하이퍼파라미터 포함)
    meta = {
        'symbol': symbol,
        'timeframe': timeframe,
        'sequence_length': sequence_length,
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'n_classes': n_classes,
        'label_map': label_map,
        'inverse_label_map': {v: k for k, v in label_map.items()},
        'scaler': scaler,
        'created_at': datetime.now().isoformat(),
        'final_accuracy': best_accuracy,
        # 모델 하이퍼파라미터
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.4
    }
    
    meta_path = os.path.join(model_dir, f'lstm_{symbol.lower()}_{timeframe}_meta.joblib')
    joblib.dump(meta, meta_path)
    
    print(f"\n✅ Model saved: {model_path}")
    print(f"✅ Metadata saved: {meta_path}")
    print(f"🎯 Best accuracy: {best_accuracy:.2f}%")
    
    return model, meta


def predict_with_lstm(symbol: str, timeframe: str, candles: pd.DataFrame) -> dict:
    """LSTM 모델로 예측"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_dir = os.path.join(AI_MODEL_DIR, 'models')
    model_path = os.path.join(model_dir, f'lstm_{symbol.lower()}_{timeframe}.pt')
    meta_path = os.path.join(model_dir, f'lstm_{symbol.lower()}_{timeframe}_meta.joblib')
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return {'error': f'LSTM model not found for {symbol} {timeframe}'}
    
    # 메타데이터 로드
    meta = joblib.load(meta_path)
    
    # 모델 로드 (메타에서 파라미터 읽기)
    model = BiLSTMClassifier(
        input_size=meta['n_features'],
        hidden_size=meta.get('hidden_size', 32),
        num_layers=meta.get('num_layers', 2),
        num_classes=meta['n_classes'],
        dropout=meta.get('dropout', 0.5)
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 피처 추출
    feature_names = meta['feature_names']
    seq_length = meta['sequence_length']
    scaler = meta['scaler']
    inverse_label_map = meta['inverse_label_map']
    
    available_features = [f for f in feature_names if f in candles.columns]
    if len(available_features) < len(feature_names) * 0.8:
        return {'error': f'Not enough features'}
    
    if len(candles) < seq_length:
        return {'error': f'Need at least {seq_length} candles'}
    
    # 시퀀스 준비
    sequence = candles[available_features].iloc[-seq_length:].values
    
    # 정규화
    sequence_flat = sequence.reshape(-1, len(available_features))
    sequence_scaled = scaler.transform(sequence_flat)
    sequence_scaled = sequence_scaled.reshape(1, seq_length, len(available_features))
    
    # 예측
    with torch.no_grad():
        X = torch.FloatTensor(sequence_scaled).to(device)
        outputs = model(X)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
    predicted_class = np.argmax(probabilities)
    original_label = inverse_label_map[predicted_class]
    
    signal_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY', -2: 'STRONG_SELL', 2: 'STRONG_BUY'}
    signal = signal_map.get(original_label, 'HOLD')
    
    return {
        'signal': signal,
        'confidence': float(probabilities[predicted_class]),
        'probabilities': {
            signal_map.get(inverse_label_map[i], str(i)): float(p)
            for i, p in enumerate(probabilities)
        },
        'model_type': 'LSTM',
        'sequence_length': seq_length
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LSTM model (PyTorch)')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair')
    parser.add_argument('--timeframe', type=str, default='5m', help='Timeframe')
    parser.add_argument('--seq-length', type=int, default=20, help='Sequence length')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    train_lstm(
        symbol=args.symbol,
        timeframe=args.timeframe,
        sequence_length=args.seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
