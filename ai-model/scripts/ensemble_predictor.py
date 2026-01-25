"""
앙상블 예측기
- XGBoost (피처 기반) + LSTM (시계열 기반) 결합
- 두 모델의 예측을 가중 평균하여 최종 신호 생성
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

AI_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AI_MODEL_DIR)

import torch
import torch.nn as nn
from train_lstm import BiLSTMClassifier


class EnsemblePredictor:
    """XGBoost + LSTM 앙상블 예측기"""
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        xgb_weight: float = 0.6,  # XGBoost 가중치
        lstm_weight: float = 0.4   # LSTM 가중치
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.xgb_weight = xgb_weight
        self.lstm_weight = lstm_weight
        
        self.xgb_model = None
        self.xgb_scaler = None
        self.xgb_features = None
        
        self.lstm_model = None
        self.lstm_meta = None
        
        self._load_models()
    
    def _load_models(self):
        """모델 로드"""
        model_dir = os.path.join(AI_MODEL_DIR, 'models')
        
        # XGBoost 모델 로드
        xgb_pattern = f'xgboost_{self.symbol.lower()}_{self.timeframe}'
        for filename in sorted(os.listdir(model_dir), reverse=True):
            if filename.startswith(xgb_pattern) and filename.endswith('.joblib'):
                if '_meta' not in filename and '_scaler' not in filename:
                    xgb_path = os.path.join(model_dir, filename)
                    self.xgb_model = joblib.load(xgb_path)
                    print(f"✅ XGBoost loaded: {filename}")
                    
                    # 스케일러 로드
                    scaler_path = xgb_path.replace('.joblib', '_scaler.joblib')
                    if os.path.exists(scaler_path):
                        self.xgb_scaler = joblib.load(scaler_path)
                    
                    # 메타데이터에서 피처 목록 로드
                    meta_path = xgb_path.replace('.joblib', '_meta.joblib')
                    if os.path.exists(meta_path):
                        meta = joblib.load(meta_path)
                        self.xgb_features = meta.get('feature_names', [])
                    break
        
        # LSTM 모델 로드 (PyTorch)
        lstm_path = os.path.join(model_dir, f'lstm_{self.symbol.lower()}_{self.timeframe}.pt')
        meta_path = os.path.join(model_dir, f'lstm_{self.symbol.lower()}_{self.timeframe}_meta.joblib')
        
        if os.path.exists(lstm_path) and os.path.exists(meta_path):
            self.lstm_meta = joblib.load(meta_path)
            self.lstm_model = BiLSTMClassifier(
                input_size=self.lstm_meta['n_features'],
                hidden_size=self.lstm_meta.get('hidden_size', 32),
                num_layers=self.lstm_meta.get('num_layers', 2),
                num_classes=self.lstm_meta['n_classes'],
                dropout=self.lstm_meta.get('dropout', 0.5)
            )
            self.lstm_model.load_state_dict(torch.load(lstm_path, map_location='cpu', weights_only=True))
            self.lstm_model.eval()
            print(f"✅ LSTM loaded: lstm_{self.symbol.lower()}_{self.timeframe}.pt")
    
    def predict(self, df: pd.DataFrame) -> dict:
        """
        앙상블 예측
        
        Args:
            df: 피처가 포함된 캔들 데이터 (최소 20봉 이상)
        
        Returns:
            예측 결과 dict
        """
        results = {
            'xgb': None,
            'lstm': None,
            'ensemble': None
        }
        
        # XGBoost 예측
        if self.xgb_model is not None:
            results['xgb'] = self._predict_xgb(df)
        
        # LSTM 예측
        if self.lstm_model is not None and self.lstm_meta is not None:
            results['lstm'] = self._predict_lstm(df)
        
        # 앙상블 결합
        results['ensemble'] = self._combine_predictions(results['xgb'], results['lstm'])
        
        return results
    
    def _predict_xgb(self, df: pd.DataFrame) -> dict:
        """XGBoost 예측"""
        try:
            # 마지막 봉의 피처 추출
            available_features = [f for f in self.xgb_features if f in df.columns]
            features = df[available_features].iloc[-1:].values
            
            # 스케일링
            if self.xgb_scaler:
                features = self.xgb_scaler.transform(features)
            
            # 예측
            probabilities = self.xgb_model.predict_proba(features)[0]
            predicted_class = np.argmax(probabilities)
            
            # 신호 매핑 (3클래스: 0=SELL, 1=HOLD, 2=BUY)
            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            
            return {
                'signal': signal_map.get(predicted_class, 'HOLD'),
                'confidence': float(probabilities[predicted_class]),
                'probabilities': {
                    'SELL': float(probabilities[0]) if len(probabilities) > 0 else 0,
                    'HOLD': float(probabilities[1]) if len(probabilities) > 1 else 0,
                    'BUY': float(probabilities[2]) if len(probabilities) > 2 else 0
                }
            }
        except Exception as e:
            print(f"⚠️ XGBoost prediction error: {e}")
            return None
    
    def _predict_lstm(self, df: pd.DataFrame) -> dict:
        """LSTM 예측 (PyTorch)"""
        try:
            feature_names = self.lstm_meta['feature_names']
            seq_length = self.lstm_meta['sequence_length']
            scaler = self.lstm_meta['scaler']
            inverse_label_map = self.lstm_meta['inverse_label_map']
            
            # 피처 추출
            available_features = [f for f in feature_names if f in df.columns]
            
            if len(df) < seq_length:
                return None
            
            # 시퀀스 준비
            sequence = df[available_features].iloc[-seq_length:].values
            
            # 정규화
            sequence_flat = sequence.reshape(-1, len(available_features))
            sequence_scaled = scaler.transform(sequence_flat)
            sequence_scaled = sequence_scaled.reshape(1, seq_length, len(available_features))
            
            # PyTorch 예측
            with torch.no_grad():
                X = torch.FloatTensor(sequence_scaled)
                outputs = self.lstm_model(X)
                probabilities = torch.softmax(outputs, dim=1).numpy()[0]
            
            predicted_class = np.argmax(probabilities)
            original_label = inverse_label_map[predicted_class]
            
            # 신호 변환
            signal_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
            
            return {
                'signal': signal_map.get(original_label, 'HOLD'),
                'confidence': float(probabilities[predicted_class]),
                'probabilities': {
                    'SELL': float(probabilities[0]) if len(probabilities) > 0 else 0,
                    'HOLD': float(probabilities[1]) if len(probabilities) > 1 else 0,
                    'BUY': float(probabilities[2]) if len(probabilities) > 2 else 0
                }
            }
        except Exception as e:
            print(f"⚠️ LSTM prediction error: {e}")
            return None
    
    def _combine_predictions(self, xgb_result: dict, lstm_result: dict) -> dict:
        """두 모델의 예측을 결합"""
        
        # 둘 다 없으면
        if xgb_result is None and lstm_result is None:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'probabilities': {'SELL': 0.33, 'HOLD': 0.34, 'BUY': 0.33},
                'models_used': []
            }
        
        # 하나만 있으면 그것만 사용
        if xgb_result is None:
            lstm_result['models_used'] = ['LSTM']
            return lstm_result
        
        if lstm_result is None:
            xgb_result['models_used'] = ['XGBoost']
            return xgb_result
        
        # 둘 다 있으면 가중 평균
        xgb_probs = xgb_result['probabilities']
        lstm_probs = lstm_result['probabilities']
        
        # 가중 평균 확률
        combined_probs = {}
        for signal in ['SELL', 'HOLD', 'BUY']:
            combined_probs[signal] = (
                self.xgb_weight * xgb_probs.get(signal, 0) +
                self.lstm_weight * lstm_probs.get(signal, 0)
            )
        
        # 최종 신호
        final_signal = max(combined_probs, key=combined_probs.get)
        final_confidence = combined_probs[final_signal]
        
        # 두 모델이 동의하면 신뢰도 부스트
        agreement_bonus = 0.0
        if xgb_result['signal'] == lstm_result['signal']:
            agreement_bonus = 0.1  # 동의 시 +10%
        
        return {
            'signal': final_signal,
            'confidence': min(final_confidence + agreement_bonus, 1.0),
            'probabilities': combined_probs,
            'models_used': ['XGBoost', 'LSTM'],
            'agreement': xgb_result['signal'] == lstm_result['signal'],
            'xgb_signal': xgb_result['signal'],
            'lstm_signal': lstm_result['signal']
        }
    
    def get_status(self) -> dict:
        """모델 상태 확인"""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'xgb_loaded': self.xgb_model is not None,
            'lstm_loaded': self.lstm_model is not None,
            'xgb_weight': self.xgb_weight,
            'lstm_weight': self.lstm_weight
        }


if __name__ == "__main__":
    # 테스트
    print("🔧 Ensemble Predictor Test")
    
    predictor = EnsemblePredictor('BTCUSDT', '5m')
    print(predictor.get_status())

