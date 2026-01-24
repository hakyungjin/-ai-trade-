"""
AI 모델 학습 스크립트
준비된 데이터셋으로 XGBoost 분류 모델 학습
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse

# ai-model 디렉토리 경로
AI_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(AI_MODEL_DIR, 'models')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# XGBoost 설치 확인
try:
    import xgboost as xgb
except ImportError:
    print("[ERROR] XGBoost not installed. Run: pip install xgboost")
    sys.exit(1)


def load_dataset(filename: str) -> pd.DataFrame:
    """데이터셋 로드"""
    if not os.path.exists(filename):
        print(f"[ERROR] File not found: {filename}")
        sys.exit(1)
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
    missing_features = [col for col in feature_columns if col not in df.columns]
    
    if missing_features:
        print(f"[WARN] Missing features (ignored): {missing_features}")
    
    print(f"[OK] Using {len(available_features)} features: {available_features}")
    
    # 무한대(inf) 및 NaN 처리
    df_clean = df[available_features].copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.fillna(0)
    
    # 이상치 확인
    inf_count = np.isinf(df[available_features].values).sum()
    nan_count = np.isnan(df[available_features].values).sum()
    if inf_count > 0 or nan_count > 0:
        print(f"[WARN] Cleaned {inf_count} inf values and {nan_count} NaN values")
    
    X = df_clean.values
    y = df['label'].values
    
    return X, y, available_features


def train_xgboost(X_train, y_train, X_test, y_test, use_class_weight=True):
    """XGBoost 모델 학습 (클래스 가중치 지원)"""
    from collections import Counter
    
    # 실제 존재하는 클래스 확인
    unique_labels = np.unique(np.concatenate([y_train, y_test]))
    print(f"[INFO] Unique labels in data: {unique_labels}")
    
    # 레이블을 0부터 시작하도록 매핑
    label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    y_train_adjusted = np.array([label_to_idx[y] for y in y_train])
    y_test_adjusted = np.array([label_to_idx[y] for y in y_test])
    
    num_classes = len(unique_labels)
    print(f"[INFO] Number of classes: {num_classes}")
    
    # 클래스 가중치 계산 (불균형 데이터 보정)
    sample_weights = None
    if use_class_weight:
        class_counts = Counter(y_train_adjusted)
        total = len(y_train_adjusted)
        n_classes = len(class_counts)
        
        # 클래스별 가중치: total / (n_classes * count)
        class_weights = {cls: total / (n_classes * count) for cls, count in class_counts.items()}
        sample_weights = np.array([class_weights[y] for y in y_train_adjusted])
        
        print(f"[INFO] Class weights applied:")
        all_label_names = {-2: 'STRONG_SELL', -1: 'SELL', 0: 'HOLD', 1: 'BUY', 2: 'STRONG_BUY'}
        for cls, weight in sorted(class_weights.items()):
            orig_label = idx_to_label[cls]
            name = all_label_names.get(orig_label, f'CLASS_{cls}')
            count = class_counts[cls]
            print(f"   {name}: {weight:.2f}x (count: {count})")
    
    model = xgb.XGBClassifier(
        n_estimators=200,       # 더 많은 트리
        max_depth=8,            # 더 깊은 트리
        learning_rate=0.05,     # 더 낮은 학습률
        objective='multi:softmax',
        num_class=num_classes,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        min_child_weight=3,     # 과적합 방지
        subsample=0.8,          # 행 샘플링
        colsample_bytree=0.8,   # 피처 샘플링
        reg_alpha=0.1,          # L1 정규화
        reg_lambda=1.0          # L2 정규화
    )
    
    print("\n[TRAIN] Training XGBoost model (with class weights)...")
    model.fit(
        X_train, y_train_adjusted,
        eval_set=[(X_test, y_test_adjusted)],
        sample_weight=sample_weights,
        verbose=True
    )
    
    # 레이블 매핑 정보 저장
    model.label_mapping = {'label_to_idx': label_to_idx, 'idx_to_label': idx_to_label}
    
    return model


def evaluate_model(model, X_test, y_test, feature_names):
    """모델 평가"""
    # 레이블 매핑 정보 가져오기
    label_mapping = getattr(model, 'label_mapping', None)
    
    if label_mapping:
        label_to_idx = label_mapping['label_to_idx']
        idx_to_label = label_mapping['idx_to_label']
        y_test_adjusted = np.array([label_to_idx[y] for y in y_test])
        
        # 실제 레이블 이름 생성
        all_label_names = {-2: 'STRONG_SELL', -1: 'SELL', 0: 'HOLD', 1: 'BUY', 2: 'STRONG_BUY'}
        label_names = [all_label_names.get(idx_to_label[i], f'CLASS_{i}') for i in range(len(idx_to_label))]
    else:
        y_test_adjusted = y_test + 2
        label_names = ['STRONG_SELL', 'SELL', 'HOLD', 'BUY', 'STRONG_BUY']
    
    y_pred = model.predict(X_test)
    
    print("\n" + "="*60)
    print("=== Model Evaluation Results ===")
    print("="*60)
    
    # 정확도
    accuracy = accuracy_score(y_test_adjusted, y_pred)
    print(f"\n[OK] Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 예측 분포
    unique_preds, pred_counts = np.unique(y_pred, return_counts=True)
    print("\n[PRED] Prediction Distribution:")
    for idx, count in zip(unique_preds, pred_counts):
        if label_mapping:
            name = all_label_names.get(idx_to_label.get(int(idx), idx), f'CLASS_{idx}')
        else:
            name = label_names[int(idx)] if int(idx) < len(label_names) else f'CLASS_{idx}'
        print(f"   {name}: {count} ({count/len(y_pred)*100:.1f}%)")
    
    # 분류 리포트 (labels 파라미터로 명시)
    print("\n[REPORT] Classification Report:")
    unique_labels = np.unique(np.concatenate([y_test_adjusted, y_pred]))
    actual_label_names = [label_names[i] if i < len(label_names) else f'CLASS_{i}' for i in unique_labels]
    print(classification_report(y_test_adjusted, y_pred, labels=unique_labels, target_names=actual_label_names, zero_division=0))
    
    # 혼동 행렬
    print("[MATRIX] Confusion Matrix:")
    cm = confusion_matrix(y_test_adjusted, y_pred, labels=unique_labels)
    print(pd.DataFrame(cm, index=actual_label_names, columns=actual_label_names))
    
    # 피처 중요도
    print("\n[IMPORTANCE] Feature Importance (Top 10):")
    importance = model.feature_importances_
    sorted_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    for i, (name, imp) in enumerate(sorted_features[:10]):
        bar = '#' * int(imp * 50)
        print(f"  {i+1}. {name:25s} {imp:.4f} {bar}")
    
    return accuracy


def save_model(model, scaler, feature_names, output_dir: str, model_name: str):
    """모델 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, f'{model_name}.joblib')
    scaler_path = os.path.join(output_dir, f'{model_name}_scaler.joblib')
    features_path = os.path.join(output_dir, f'{model_name}_features.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_names, features_path)
    
    print(f"\n[SAVED] Model saved:")
    print(f"   - Model: {model_path}")
    print(f"   - Scaler: {scaler_path}")
    print(f"   - Features: {features_path}")


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost model for crypto trading signals')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_MODEL_DIR, help='Output directory for model')
    parser.add_argument('--model-name', type=str, default='xgboost_model', help='Model name')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set ratio')
    
    args = parser.parse_args()
    
    # 1. 데이터 로드
    print("[LOAD] Loading dataset...")
    df = load_dataset(args.input)
    print(f"   Loaded {len(df)} samples")
    
    # 2. 피처/레이블 분리
    print("\n[PREP] Preparing features...")
    X, y, feature_names = prepare_features_and_labels(df)
    
    # 3. 정규화
    print("[SCALE] Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. 학습/테스트 분리 (시계열이므로 shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=args.test_size, random_state=42, shuffle=False
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 5. 모델 학습
    model = train_xgboost(X_train, y_train, X_test, y_test)
    
    # 6. 평가
    accuracy = evaluate_model(model, X_test, y_test, feature_names)
    
    # 7. 모델 저장
    save_model(model, scaler, feature_names, args.output_dir, args.model_name)
    
    print("\n" + "="*60)
    print("[DONE] Training Complete!")
    print(f"   Final Accuracy: {accuracy*100:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()

