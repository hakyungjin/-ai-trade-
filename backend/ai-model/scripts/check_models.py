"""
모델 성능 확인 스크립트
- 모든 XGBoost/LSTM 모델 정보 출력
- 정확도, 클래스 수, 피처 수 등 한눈에 확인
"""

import os
import sys
import glob
import joblib
import numpy as np

AI_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(AI_MODEL_DIR, 'models')


def check_xgboost_models():
    """XGBoost 모델 정보 출력"""
    print("\n" + "="*70)
    print("🌲 XGBoost Models")
    print("="*70)
    
    xgb_files = glob.glob(os.path.join(MODELS_DIR, 'xgboost_*.joblib'))
    xgb_files = [f for f in xgb_files if '_scaler' not in f and '_features' not in f]
    
    if not xgb_files:
        print("❌ No XGBoost models found")
        return
    
    for model_path in sorted(xgb_files):
        model_name = os.path.basename(model_path).replace('.joblib', '')
        
        try:
            model = joblib.load(model_path)
            
            # 피처 파일 확인
            features_path = model_path.replace('.joblib', '_features.joblib')
            features = joblib.load(features_path) if os.path.exists(features_path) else []
            
            # 모델 정보 추출
            n_classes = getattr(model, 'n_classes_', len(getattr(model, 'classes_', [])))
            n_estimators = getattr(model, 'n_estimators', 'N/A')
            max_depth = getattr(model, 'max_depth', 'N/A')
            
            # 클래스 레이블 확인
            if n_classes == 2:
                class_names = "SELL / BUY (횡보 제거)"
            elif n_classes == 3:
                class_names = "SELL / HOLD / BUY"
            else:
                class_names = f"{n_classes} classes"
            
            print(f"\n📊 {model_name}")
            print(f"   Classes: {n_classes} ({class_names})")
            print(f"   Features: {len(features)}")
            print(f"   Trees: {n_estimators}, Max Depth: {max_depth}")
            
            # Feature Importance Top 10
            if hasattr(model, 'feature_importances_') and len(features) > 0:
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:10]
                print(f"   Top Features:")
                for i, idx in enumerate(indices):
                    if idx < len(features):
                        print(f"      {i+1}. {features[idx]}: {importances[idx]*100:.1f}%")
        
        except Exception as e:
            print(f"\n❌ {model_name}: Error - {e}")


def check_lstm_models():
    """LSTM 모델 정보 출력"""
    print("\n" + "="*70)
    print("🧠 LSTM Models")
    print("="*70)
    
    lstm_files = glob.glob(os.path.join(MODELS_DIR, 'lstm_*.pt'))
    
    if not lstm_files:
        print("❌ No LSTM models found")
        return
    
    for model_path in sorted(lstm_files):
        model_name = os.path.basename(model_path).replace('.pt', '')
        
        try:
            # 메타 정보 로드
            meta_path = model_path.replace('.pt', '_meta.joblib')
            
            if os.path.exists(meta_path):
                meta = joblib.load(meta_path)
                
                n_classes = meta.get('num_classes', 'N/A')
                features = meta.get('features', [])
                seq_length = meta.get('seq_length', 'N/A')
                train_acc = meta.get('train_accuracy', 'N/A')
                val_acc = meta.get('val_accuracy', 'N/A')
                
                # 클래스 레이블
                if n_classes == 2:
                    class_names = "SELL / BUY"
                elif n_classes == 3:
                    class_names = "SELL / HOLD / BUY"
                else:
                    class_names = f"{n_classes} classes"
                
                print(f"\n🔮 {model_name}")
                print(f"   Classes: {n_classes} ({class_names})")
                print(f"   Sequence Length: {seq_length} candles")
                print(f"   Features: {len(features)}")
                
                if train_acc != 'N/A':
                    print(f"   📈 Train Accuracy: {train_acc*100:.1f}%")
                if val_acc != 'N/A':
                    print(f"   📊 Val Accuracy: {val_acc*100:.1f}%")
                
                # 추가 메타 정보
                if 'best_epoch' in meta:
                    print(f"   Best Epoch: {meta['best_epoch']}")
            else:
                print(f"\n🔮 {model_name}")
                print(f"   ⚠️ No meta file (accuracy unknown)")
                
                # 모델 파일 크기
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"   Model Size: {size_mb:.2f} MB")
        
        except Exception as e:
            print(f"\n❌ {model_name}: Error - {e}")


def show_summary():
    """모델 요약"""
    print("\n" + "="*70)
    print("📋 Model Summary")
    print("="*70)
    
    xgb_count = len([f for f in glob.glob(os.path.join(MODELS_DIR, 'xgboost_*.joblib'))
                    if '_scaler' not in f and '_features' not in f])
    lstm_count = len(glob.glob(os.path.join(MODELS_DIR, 'lstm_*.pt')))
    
    print(f"\n   🌲 XGBoost models: {xgb_count}")
    print(f"   🧠 LSTM models: {lstm_count}")
    print(f"   📁 Models directory: {MODELS_DIR}")
    
    # 심볼별 모델 현황
    print("\n   📊 Models by Symbol:")
    
    all_files = glob.glob(os.path.join(MODELS_DIR, '*.joblib')) + glob.glob(os.path.join(MODELS_DIR, '*.pt'))
    symbols = set()
    for f in all_files:
        name = os.path.basename(f).lower()
        # 심볼 추출 (xgboost_btcusdt_5m... -> btcusdt)
        parts = name.replace('xgboost_', '').replace('lstm_', '').split('_')
        if parts:
            symbols.add(parts[0].upper())
    
    for symbol in sorted(symbols):
        xgb = len([f for f in glob.glob(os.path.join(MODELS_DIR, f'xgboost_{symbol.lower()}_*.joblib'))
                   if '_scaler' not in f and '_features' not in f])
        lstm = len(glob.glob(os.path.join(MODELS_DIR, f'lstm_{symbol.lower()}_*.pt')))
        
        status = []
        if xgb > 0:
            status.append(f"XGB×{xgb}")
        if lstm > 0:
            status.append(f"LSTM×{lstm}")
        
        ensemble = "🎯 Ensemble" if xgb > 0 and lstm > 0 else ""
        print(f"      {symbol}: {' + '.join(status)} {ensemble}")


if __name__ == '__main__':
    print("\n" + "🔍 AI Model Status Check ".center(70, "="))
    
    show_summary()
    check_xgboost_models()
    check_lstm_models()
    
    print("\n" + "="*70)
    print("✅ Done!")
    print("="*70 + "\n")

