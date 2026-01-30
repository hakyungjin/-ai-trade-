"""
피드백 기반 모델 재학습 스크립트
- 실제 거래 결과를 학습 데이터로 활용
- 기존 모델을 fine-tuning
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import requests
import joblib

AI_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AI_MODEL_DIR)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def fetch_feedback_data(api_url: str, symbol: str = None, timeframe: str = '5m') -> pd.DataFrame:
    """API에서 피드백 데이터 가져오기"""
    params = {'timeframe': timeframe}
    if symbol:
        params['symbol'] = symbol
    
    response = requests.get(f"{api_url}/api/feedback/training-data", params=params)
    
    if response.status_code != 200:
        print(f"❌ Failed to fetch feedback data: {response.status_code}")
        return pd.DataFrame()
    
    result = response.json()
    print(f"✅ Fetched {result['count']} feedback records")
    
    if result['count'] == 0:
        return pd.DataFrame()
    
    return pd.DataFrame(result['data'])


def combine_with_original_data(feedback_df: pd.DataFrame, original_path: str) -> pd.DataFrame:
    """피드백 데이터와 원본 학습 데이터 결합"""
    
    # 원본 데이터 로드
    if os.path.exists(original_path):
        original_df = pd.read_csv(original_path, index_col=0)
        print(f"📊 Original data: {len(original_df)} samples")
    else:
        original_df = pd.DataFrame()
        print("⚠️ No original data found")
    
    # 피드백 데이터에서 학습 피처 추출
    feedback_features = []
    for _, row in feedback_df.iterrows():
        if pd.notna(row.get('actual_label')) and row['actual_label'] >= 0:
            # 지표 스냅샷이 있으면 사용
            feature_row = dict(row)
            feature_row['label'] = row['actual_label']  # 실제 결과가 레이블
            feedback_features.append(feature_row)
    
    feedback_train_df = pd.DataFrame(feedback_features)
    print(f"📝 Feedback data with labels: {len(feedback_train_df)} samples")
    
    # 공통 피처만 사용하여 결합
    if len(original_df) > 0 and len(feedback_train_df) > 0:
        common_cols = list(set(original_df.columns) & set(feedback_train_df.columns))
        combined = pd.concat([
            original_df[common_cols],
            feedback_train_df[common_cols]
        ], ignore_index=True)
        print(f"🔗 Combined data: {len(combined)} samples")
        return combined
    elif len(feedback_train_df) > 0:
        return feedback_train_df
    else:
        return original_df


def fine_tune_xgboost(
    model_path: str,
    data: pd.DataFrame,
    feature_cols: list,
    output_path: str
):
    """기존 XGBoost 모델 fine-tuning"""
    import xgboost as xgb
    
    # 기존 모델 로드
    if os.path.exists(model_path):
        print(f"📥 Loading existing model: {model_path}")
        model = joblib.load(model_path)
    else:
        print("⚠️ No existing model, creating new one")
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            objective='binary:logistic',
            random_state=42
        )
    
    # 피처 준비
    available_features = [col for col in feature_cols if col in data.columns]
    X = data[available_features].fillna(0).values
    y = data['label'].values
    
    # 레이블을 0, 1로 변환
    unique_labels = np.unique(y)
    print(f"📊 Labels in data: {unique_labels}")
    
    if len(unique_labels) < 2:
        print("❌ Need at least 2 classes for training")
        return False
    
    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n🎯 Fine-tuning with {len(X_train)} samples...")
    
    # Fine-tuning (기존 모델 위에 추가 학습)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=True
    )
    
    # 평가
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 Fine-tuned Model Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 모델 저장
    joblib.dump(model, output_path)
    print(f"\n✅ Fine-tuned model saved: {output_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Fine-tune model with feedback data')
    parser.add_argument('--api', type=str, default='http://localhost:8000', help='Backend API URL')
    parser.add_argument('--symbol', type=str, default=None, help='Symbol to filter (optional)')
    parser.add_argument('--timeframe', type=str, default='5m', help='Timeframe')
    parser.add_argument('--original-data', type=str, default=None, help='Original training data CSV')
    parser.add_argument('--model', type=str, required=True, help='Existing model path')
    parser.add_argument('--output', type=str, required=True, help='Output model path')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🧠 Feedback-based Model Fine-tuning")
    print("="*60)
    
    # 1. 피드백 데이터 가져오기
    print("\n[1/4] Fetching feedback data...")
    feedback_df = fetch_feedback_data(args.api, args.symbol, args.timeframe)
    
    if len(feedback_df) == 0:
        print("❌ No feedback data available. Trade more and record feedback!")
        return
    
    # 2. 원본 데이터와 결합
    print("\n[2/4] Combining with original data...")
    if args.original_data:
        combined_df = combine_with_original_data(feedback_df, args.original_data)
    else:
        combined_df = feedback_df
        combined_df['label'] = combined_df['actual_label']
    
    if len(combined_df) < 50:
        print(f"⚠️ Only {len(combined_df)} samples. Recommend at least 50 for reliable training.")
    
    # 3. 피처 목록 (모델에서 사용하는 피처들)
    feature_cols = [
        'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
        'bb_position', 'stoch_k', 'stoch_d', 'atr_14', 'adx',
        'price_change_1', 'price_change_5', 'price_change_10',
        'volume_change_1', 'volume_ma_ratio', 'volume_spike',
        'obv_slope', 'mfi_normalized', 'williams_r',
        'pump_6', 'drawdown_from_high_12', 'volatility_spike',
        'ai_confidence'  # AI 자체 신뢰도도 피처로!
    ]
    
    # 4. Fine-tuning
    print("\n[3/4] Fine-tuning model...")
    success = fine_tune_xgboost(
        model_path=args.model,
        data=combined_df,
        feature_cols=feature_cols,
        output_path=args.output
    )
    
    if success:
        print("\n[4/4] ✅ Fine-tuning complete!")
        print(f"   New model: {args.output}")
        print("\n💡 Tip: Copy to ai-model/models/ and restart backend to use")
    else:
        print("\n❌ Fine-tuning failed")


if __name__ == '__main__':
    main()

