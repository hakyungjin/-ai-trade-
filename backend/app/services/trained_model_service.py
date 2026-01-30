"""
학습된 모델 예측 서비스
저장된 XGBoost/LSTM 모델을 로드하여 실시간 예측 수행

통합 피처 엔지니어링 모듈(unified_feature_engineering)을 사용하여
학습/추론 간 피처 일관성 보장
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from app.services.technical_indicators import TechnicalIndicators
from app.services.unified_feature_engineering import compute_all_features

logger = logging.getLogger(__name__)


class TrainedModelService:
    """학습된 모델을 사용한 예측 서비스"""
    
    # 2클래스 레이블 매핑 (횡보 제거!)
    LABEL_MAP_2 = {
        0: ('SELL', -1),
        1: ('BUY', 1)
    }
    
    # 3클래스 레이블 매핑
    LABEL_MAP_3 = {
        0: ('SELL', -1),
        1: ('HOLD', 0),
        2: ('BUY', 1)
    }
    
    # 5클래스 레이블 매핑
    LABEL_MAP_5 = {
        0: ('STRONG_SELL', -2),
        1: ('SELL', -1),
        2: ('HOLD', 0),
        3: ('BUY', 1),
        4: ('STRONG_BUY', 2)
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        features_path: Optional[str] = None
    ):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_loaded = False
        self.num_classes = 5  # 기본값
        
        # 레이블 매핑 (모델 로드 시 동적으로 설정)
        self.label_map = self.LABEL_MAP_5
        
        self.signal_to_simple = {
            'STRONG_SELL': 'SELL',
            'SELL': 'SELL',
            'HOLD': 'HOLD',
            'BUY': 'BUY',
            'STRONG_BUY': 'BUY'
        }
        
        if model_path and scaler_path and features_path:
            self.load_model(model_path, scaler_path, features_path)
    
    def load_model(self, model_path: str, scaler_path: str, features_path: str) -> bool:
        """모델 로드"""
        try:
            import joblib
            
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                return False
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_names = joblib.load(features_path)
            self.is_loaded = True
            
            # 모델의 클래스 수 확인
            if hasattr(self.model, 'n_classes_'):
                self.num_classes = self.model.n_classes_
            elif hasattr(self.model, 'classes_'):
                self.num_classes = len(self.model.classes_)
            else:
                self.num_classes = 5  # 기본값
            
            # 클래스 수에 따라 레이블 매핑 설정
            if self.num_classes == 2:
                self.label_map = self.LABEL_MAP_2
                logger.info(f"   Using 2-class model (SELL/BUY - 횡보 제거!)")
            elif self.num_classes == 3:
                self.label_map = self.LABEL_MAP_3
                logger.info(f"   Using 3-class model (SELL/HOLD/BUY)")
            else:
                self.label_map = self.LABEL_MAP_5
                logger.info(f"   Using 5-class model (STRONG_SELL/SELL/HOLD/BUY/STRONG_BUY)")
            
            logger.info(f"✅ Loaded trained model from {model_path}")
            logger.info(f"   Features: {len(self.feature_names)}, Classes: {self.num_classes}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.is_loaded = False
            return False
    
    def predict(self, candles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        캔들 데이터로 예측 수행
        
        Args:
            candles: 캔들 데이터 리스트 (최소 50개 이상 권장)
        
        Returns:
            {
                'signal': 'BUY' | 'SELL' | 'HOLD',
                'detailed_signal': 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL',
                'confidence': 0.0 ~ 1.0,
                'probabilities': {...}
            }
        """
        if not self.is_loaded:
            return self._default_response("Model not loaded")
        
        if len(candles) < 50:
            return self._default_response(f"Insufficient data: {len(candles)} candles (min 50)")
        
        try:
            # DataFrame 생성
            df = pd.DataFrame(candles)
            logger.info(f"📊 Input candles: {len(df)}, columns: {df.columns.tolist()[:5]}...")
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
                if df['timestamp'].notna().any():
                    df.set_index('timestamp', inplace=True)
            
            # 기술적 지표 계산
            df = TechnicalIndicators.calculate_all_indicators(df)
            logger.info(f"📊 After indicators: {len(df)} rows, columns: {len(df.columns)}")
            
            # 추가 피처 생성
            df = self._create_features(df)
            logger.info(f"📊 After features: {len(df)} rows, columns: {len(df.columns)}")
            
            # NaN 제거 - 마지막 행만 사용하므로 마지막 행의 NaN만 체크
            last_row = df.iloc[-1:]
            nan_cols = last_row.columns[last_row.isna().any()].tolist()
            
            if nan_cols:
                logger.warning(f"⚠️ NaN columns in last row: {nan_cols[:10]}...")
                # NaN 값을 0 또는 이전 값으로 채움
                df = df.fillna(method='ffill').fillna(0)
            
            if df.empty:
                return self._default_response("No valid data after processing")
            
            # 피처 추출 (마지막 행만)
            missing_features = [f for f in self.feature_names if f not in df.columns]
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # 없는 피처는 0으로 채움
                for f in missing_features:
                    df[f] = 0
            
            X = df[self.feature_names].iloc[-1:].values
            
            # 정규화
            X_scaled = self.scaler.transform(X)
            
            # 예측
            pred_class = int(self.model.predict(X_scaled)[0])
            pred_proba = self.model.predict_proba(X_scaled)[0]
            
            detailed_signal, signal_value = self.label_map[pred_class]
            simple_signal = self.signal_to_simple[detailed_signal]
            confidence = float(pred_proba[pred_class])
            
            return {
                'signal': simple_signal,
                'detailed_signal': detailed_signal,
                'signal_value': signal_value,
                'confidence': confidence,
                'direction': 'UP' if signal_value > 0 else 'DOWN' if signal_value < 0 else 'NEUTRAL',
                'probabilities': {
                    self.label_map[i][0]: float(p) 
                    for i, p in enumerate(pred_proba)
                },
                'analysis': self._generate_analysis(detailed_signal, confidence, pred_proba)
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return self._default_response(f"Prediction error: {str(e)}")
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """통합 피처 엔지니어링 모듈 사용 (학습/추론 피처 일관성 보장)"""
        return compute_all_features(df)
    
    def _default_response(self, reason: str) -> Dict[str, Any]:
        """기본 응답 (예측 실패 시)"""
        return {
            'signal': 'HOLD',
            'detailed_signal': 'HOLD',
            'signal_value': 0,
            'confidence': 0.0,
            'direction': 'NEUTRAL',
            'probabilities': {},
            'analysis': f"Unable to predict: {reason}"
        }
    
    def _generate_analysis(
        self,
        signal: str,
        confidence: float,
        probabilities: np.ndarray
    ) -> str:
        """분석 텍스트 생성 (2클래스/3클래스/5클래스 모델 모두 지원)"""
        
        # 클래스 수에 따라 확률 계산
        if len(probabilities) == 2:
            # 2클래스: SELL(0), BUY(1) - 횡보 제거!
            sell_prob = probabilities[0]
            hold_prob = 0  # 횡보 없음
            buy_prob = probabilities[1]
        elif len(probabilities) == 3:
            # 3클래스: SELL(0), HOLD(1), BUY(2)
            sell_prob = probabilities[0]
            hold_prob = probabilities[1]
            buy_prob = probabilities[2]
        elif len(probabilities) == 5:
            # 5클래스: STRONG_SELL(0), SELL(1), HOLD(2), BUY(3), STRONG_BUY(4)
            sell_prob = probabilities[0] + probabilities[1]
            hold_prob = probabilities[2]
            buy_prob = probabilities[3] + probabilities[4]
        else:
            # 알 수 없는 클래스 수
            return f"신호: {signal} (신뢰도: {confidence*100:.1f}%)"
        
        analysis_parts = []
        
        # 신호 설명
        if signal in ['STRONG_BUY', 'BUY']:
            analysis_parts.append(f"매수 신호 감지 (신뢰도: {confidence*100:.1f}%)")
        elif signal in ['STRONG_SELL', 'SELL']:
            analysis_parts.append(f"매도 신호 감지 (신뢰도: {confidence*100:.1f}%)")
        else:
            analysis_parts.append(f"관망 권장 (신뢰도: {confidence*100:.1f}%)")
        
        # 확률 분포
        analysis_parts.append(
            f"확률 분포 - 매수: {buy_prob*100:.1f}%, 관망: {hold_prob*100:.1f}%, 매도: {sell_prob*100:.1f}%"
        )
        
        # 강도 설명 (5클래스 모델만)
        if 'STRONG' in signal:
            analysis_parts.append("강한 신호입니다. 포지션 크기 조절에 참고하세요.")
        
        return " | ".join(analysis_parts)


# 심볼별 모델 캐시
_model_cache: Dict[str, TrainedModelService] = {}


def get_model_dir() -> str:
    """모델 디렉토리 경로 반환"""
    current_file = os.path.abspath(__file__)
    services_dir = os.path.dirname(current_file)      # backend/app/services
    app_dir = os.path.dirname(services_dir)           # backend/app
    backend_dir = os.path.dirname(app_dir)            # backend
    project_root = os.path.dirname(backend_dir)       # project root
    model_dir = os.path.join(project_root, 'ai-model', 'models')
    
    # 디버그 로그
    if not hasattr(get_model_dir, '_logged'):
        logger.info(f"📁 Current file: {current_file}")
        logger.info(f"📁 Project root: {project_root}")
        logger.info(f"📁 Model dir: {model_dir}")
        logger.info(f"📁 Model dir exists: {os.path.exists(model_dir)}")
        if os.path.exists(model_dir):
            files = os.listdir(model_dir)
            logger.info(f"📁 Model files: {[f for f in files if f.endswith('.joblib') and '_scaler' not in f and '_features' not in f]}")
        get_model_dir._logged = True
    
    return model_dir


def get_available_models() -> List[str]:
    """사용 가능한 모델 목록 반환"""
    model_dir = get_model_dir()
    if not os.path.exists(model_dir):
        return []
    
    models = []
    for f in os.listdir(model_dir):
        # xgboost_btcusdt_5m_v2.joblib 형태에서 심볼 추출
        if f.startswith('xgboost_') and f.endswith('.joblib') and '_scaler' not in f and '_features' not in f:
            # xgboost_btcusdt_5m_v2.joblib -> btcusdt_5m_v2
            model_name = f.replace('xgboost_', '').replace('.joblib', '')
            models.append(model_name)
    
    return models


def find_highest_version_model(symbol: str, timeframe: str) -> Optional[str]:
    """
    심볼과 타임프레임에 맞는 가장 높은 버전의 모델을 찾음
    
    예: xgboost_btcusdt_5m_v3.joblib > xgboost_btcusdt_5m_v2.joblib > xgboost_btcusdt_5m.joblib
    """
    import re
    
    model_dir = get_model_dir()
    if not os.path.exists(model_dir):
        return None
    
    symbol_lower = symbol.lower()
    pattern = f'xgboost_{symbol_lower}_{timeframe}'
    
    # 해당 심볼/타임프레임의 모든 모델 파일 찾기
    matching_models = []
    for f in os.listdir(model_dir):
        if f.startswith(pattern) and f.endswith('.joblib') and '_scaler' not in f and '_features' not in f:
            # 버전 번호 추출 (예: xgboost_btcusdt_5m_v2.joblib -> 2, xgboost_btcusdt_5m.joblib -> 0)
            version_match = re.search(r'_v(\d+)\.joblib$', f)
            if version_match:
                version = int(version_match.group(1))
            else:
                version = 0  # v 없으면 버전 0 (v1 이전)
            
            model_key = f.replace('xgboost_', '').replace('.joblib', '')
            matching_models.append((version, model_key, f))
    
    if not matching_models:
        return None
    
    # 버전이 가장 높은 것 선택
    matching_models.sort(key=lambda x: x[0], reverse=True)
    best_version, best_model_key, best_file = matching_models[0]
    
    logger.info(f"🏆 Found {len(matching_models)} model(s) for {symbol} {timeframe}")
    logger.info(f"   Best: {best_file} (v{best_version})")
    
    return best_model_key


def get_trained_model_service(symbol: str, timeframe: str = '5m') -> TrainedModelService:
    """
    심볼별 학습된 모델 서비스 인스턴스 반환
    가장 높은 버전의 모델을 자동으로 선택
    
    모델 명명 규칙: xgboost_{symbol}_{timeframe}_v{N}.joblib
    예: xgboost_btcusdt_5m_v3.joblib
    """
    # 가장 높은 버전 모델 찾기
    model_key = find_highest_version_model(symbol, timeframe)
    
    if not model_key:
        logger.warning(f"⚠️ No model found for {symbol} {timeframe}")
        empty_service = TrainedModelService()
        empty_service.is_loaded = False
        return empty_service
    
    # 캐시에 있으면 반환
    if model_key in _model_cache:
        cached = _model_cache[model_key]
        if cached.is_loaded:
            return cached
    
    # 모델 로드
    model_dir = get_model_dir()
    model_name = f'xgboost_{model_key}'
    model_path = os.path.join(model_dir, f'{model_name}.joblib')
    scaler_path = os.path.join(model_dir, f'{model_name}_scaler.joblib')
    features_path = os.path.join(model_dir, f'{model_name}_features.joblib')
    
    service = TrainedModelService()
    if service.load_model(model_path, scaler_path, features_path):
        logger.info(f"✅ Loaded model for {symbol}: {model_name}")
        _model_cache[model_key] = service
        return service
    
    # 로드 실패
    logger.warning(f"⚠️ Failed to load model: {model_name}")
    empty_service = TrainedModelService()
    empty_service.is_loaded = False
    return empty_service


def check_model_exists(symbol: str, timeframe: str = '5m') -> bool:
    """특정 심볼의 모델이 존재하는지 확인 (가장 높은 버전 기준)"""
    model_key = find_highest_version_model(symbol, timeframe)
    exists = model_key is not None
    
    logger.info(f"🔍 Model check for {symbol} ({timeframe}): {'✅ Found' if exists else '❌ Not found'}")
    
    return exists


# ==================== 앙상블 모델 서비스 ====================

class EnsembleModelService:
    """XGBoost + LSTM 앙상블 예측 서비스"""
    
    def __init__(self, symbol: str, timeframe: str = '5m'):
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.xgb_service = None
        self.lstm_model = None
        self.lstm_scaler = None
        self.lstm_features = None
        self.lstm_seq_length = 20
        self.is_loaded = False
        
        # 가중치: XGBoost는 안정적, LSTM은 패턴 감지에 강함
        self.xgb_weight = 0.6
        self.lstm_weight = 0.4
        
        self._load_models()
    
    def _load_models(self):
        """모델들 로드"""
        import os
        
        model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            'ai-model', 'models'
        )
        
        # 1. XGBoost 모델 로드
        self.xgb_service = get_trained_model_service(self.symbol, self.timeframe)
        xgb_loaded = self.xgb_service and self.xgb_service.is_loaded
        
        # 2. LSTM 모델 로드
        lstm_loaded = self._load_lstm(model_dir)
        
        self.is_loaded = xgb_loaded or lstm_loaded
        
        if xgb_loaded and lstm_loaded:
            logger.info(f"🎯 Ensemble ready: XGBoost + LSTM for {self.symbol}")
        elif xgb_loaded:
            logger.info(f"📊 XGBoost only for {self.symbol} (LSTM not found)")
            self.xgb_weight = 1.0
            self.lstm_weight = 0.0
        elif lstm_loaded:
            logger.info(f"🧠 LSTM only for {self.symbol} (XGBoost not found)")
            self.xgb_weight = 0.0
            self.lstm_weight = 1.0
        else:
            logger.warning(f"⚠️ No models found for {self.symbol}")
    
    def _load_lstm(self, model_dir: str) -> bool:
        """LSTM 모델 로드"""
        try:
            import torch
            import joblib
            
            symbol_lower = self.symbol.lower()
            lstm_path = os.path.join(model_dir, f'lstm_{symbol_lower}_{self.timeframe}.pt')
            meta_path = os.path.join(model_dir, f'lstm_{symbol_lower}_{self.timeframe}_meta.joblib')
            
            if not os.path.exists(lstm_path):
                logger.info(f"LSTM model not found: {lstm_path}")
                return False
            
            # 메타 정보 로드
            if os.path.exists(meta_path):
                meta = joblib.load(meta_path)
                self.lstm_scaler = meta.get('scaler')
                self.lstm_features = meta.get('features', self._default_lstm_features())
                self.lstm_seq_length = meta.get('seq_length', 20)
                num_classes = meta.get('num_classes', 3)
            else:
                self.lstm_features = self._default_lstm_features()
                num_classes = 3
            
            # LSTM 모델 정의 (BiLSTMClassifier)
            class Attention(torch.nn.Module):
                def __init__(self, hidden_size):
                    super().__init__()
                    self.attention = torch.nn.Sequential(
                        torch.nn.Linear(hidden_size, hidden_size // 2),
                        torch.nn.Tanh(),
                        torch.nn.Linear(hidden_size // 2, 1)
                    )
                
                def forward(self, lstm_output):
                    attention_weights = self.attention(lstm_output)
                    attention_weights = torch.softmax(attention_weights, dim=1)
                    context = torch.sum(lstm_output * attention_weights, dim=1)
                    return context, attention_weights
            
            class BiLSTMClassifier(torch.nn.Module):
                def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=3, dropout=0.4):
                    super().__init__()
                    self.lstm = torch.nn.LSTM(
                        input_size=input_size, hidden_size=hidden_size,
                        num_layers=num_layers, batch_first=True,
                        bidirectional=True, dropout=dropout if num_layers > 1 else 0
                    )
                    self.attention = Attention(hidden_size * 2)
                    self.bn = torch.nn.BatchNorm1d(hidden_size * 2)
                    self.fc = torch.nn.Sequential(
                        torch.nn.Linear(hidden_size * 2, 64),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(dropout),
                        torch.nn.Linear(64, 32),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(dropout),
                        torch.nn.Linear(32, num_classes)
                    )
                
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    context, _ = self.attention(lstm_out)
                    context = self.bn(context)
                    return self.fc(context)
            
            # 모델 로드
            input_size = len(self.lstm_features)
            self.lstm_model = BiLSTMClassifier(input_size, num_classes=num_classes)
            self.lstm_model.load_state_dict(torch.load(lstm_path, map_location='cpu'))
            self.lstm_model.eval()
            
            logger.info(f"✅ LSTM loaded: {lstm_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ LSTM load error: {e}")
            return False
    
    def _default_lstm_features(self):
        """기본 LSTM 피처 목록 (통합 모듈의 핵심 피처 사용)"""
        from app.services.unified_feature_engineering import get_core_feature_names
        return get_core_feature_names()
    
    def predict(self, candles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """앙상블 예측"""
        if not self.is_loaded:
            return self._default_response("No models loaded")
        
        xgb_proba = None
        lstm_proba = None
        
        # 1. XGBoost 예측
        if self.xgb_service and self.xgb_service.is_loaded:
            xgb_result = self.xgb_service.predict(candles)
            if xgb_result.get('confidence', 0) > 0:
                xgb_proba = np.array(list(xgb_result.get('probabilities', {}).values()))
        
        # 2. LSTM 예측
        if self.lstm_model is not None:
            lstm_proba = self._predict_lstm(candles)
        
        # 3. 앙상블
        if xgb_proba is not None and lstm_proba is not None:
            # 클래스 수 맞추기
            if len(xgb_proba) != len(lstm_proba):
                # 다르면 XGBoost 우선
                ensemble_proba = xgb_proba
                model_used = "XGBoost (class mismatch)"
            else:
                ensemble_proba = self.xgb_weight * xgb_proba + self.lstm_weight * lstm_proba
                model_used = f"Ensemble (XGB:{self.xgb_weight:.0%} + LSTM:{self.lstm_weight:.0%})"
        elif xgb_proba is not None:
            ensemble_proba = xgb_proba
            model_used = "XGBoost only"
        elif lstm_proba is not None:
            ensemble_proba = lstm_proba
            model_used = "LSTM only"
        else:
            return self._default_response("Prediction failed")
        
        # 4. 결과 생성
        pred_class = int(np.argmax(ensemble_proba))
        confidence = float(ensemble_proba[pred_class])
        
        # 레이블 매핑 (클래스 수에 따라)
        num_classes = len(ensemble_proba)
        if num_classes == 2:
            label_map = {0: ('SELL', -1), 1: ('BUY', 1)}
        elif num_classes == 3:
            label_map = {0: ('SELL', -1), 1: ('HOLD', 0), 2: ('BUY', 1)}
        else:
            label_map = {0: ('STRONG_SELL', -2), 1: ('SELL', -1), 2: ('HOLD', 0), 3: ('BUY', 1), 4: ('STRONG_BUY', 2)}
        
        detailed_signal, signal_value = label_map.get(pred_class, ('HOLD', 0))
        simple_signal = 'BUY' if signal_value > 0 else ('SELL' if signal_value < 0 else 'HOLD')
        
        # 확률 딕셔너리
        prob_dict = {label_map[i][0]: float(ensemble_proba[i]) for i in range(len(ensemble_proba))}
        
        # direction 결정
        direction = 'UP' if signal_value > 0 else ('DOWN' if signal_value < 0 else 'NEUTRAL')
        
        return {
            'signal': simple_signal,
            'detailed_signal': detailed_signal,
            'confidence': confidence,
            'signal_value': signal_value,
            'direction': direction,  # 추가!
            'probabilities': prob_dict,
            'model_used': model_used,
            'analysis': self._generate_analysis(detailed_signal, confidence, ensemble_proba, model_used)
        }
    
    def _predict_lstm(self, candles: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """LSTM 예측"""
        try:
            import torch
            
            # 데이터프레임 변환
            df = pd.DataFrame(candles)
            
            # 기술적 지표 + 통합 피처 생성
            df = TechnicalIndicators.calculate_all_indicators(df)
            df = compute_all_features(df)
            
            # 필요한 피처만 추출
            available_features = [f for f in self.lstm_features if f in df.columns]
            if len(available_features) < len(self.lstm_features) * 0.5:
                logger.warning("Not enough features for LSTM")
                return None
            
            # NaN 처리
            df = df[available_features].fillna(0)
            
            # 시퀀스 데이터 준비
            if len(df) < self.lstm_seq_length:
                logger.warning(f"Not enough candles for LSTM (need {self.lstm_seq_length})")
                return None
            
            # 마지막 시퀀스 추출
            seq_data = df.iloc[-self.lstm_seq_length:].values
            
            # 스케일링
            if self.lstm_scaler is not None:
                seq_data = self.lstm_scaler.transform(seq_data)
            
            # 텐서 변환
            X = torch.FloatTensor(seq_data).unsqueeze(0)  # (1, seq_len, features)
            
            # 예측
            with torch.no_grad():
                output = self.lstm_model(X)
                proba = torch.softmax(output, dim=1).numpy()[0]
            
            return proba
            
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return None
    
    def _generate_analysis(self, signal: str, confidence: float, proba: np.ndarray, model_used: str) -> str:
        """분석 텍스트 생성"""
        parts = []
        
        parts.append(f"🎯 {model_used}")
        
        if signal in ['BUY', 'STRONG_BUY']:
            parts.append(f"📈 매수 신호 (신뢰도: {confidence*100:.1f}%)")
        elif signal in ['SELL', 'STRONG_SELL']:
            parts.append(f"📉 매도 신호 (신뢰도: {confidence*100:.1f}%)")
        else:
            parts.append(f"⏸️ 관망 (신뢰도: {confidence*100:.1f}%)")
        
        # 확률 분포
        if len(proba) == 2:
            parts.append(f"확률: BUY {proba[1]*100:.1f}% / SELL {proba[0]*100:.1f}%")
        elif len(proba) == 3:
            parts.append(f"확률: BUY {proba[2]*100:.1f}% / HOLD {proba[1]*100:.1f}% / SELL {proba[0]*100:.1f}%")
        
        return " | ".join(parts)
    
    def _default_response(self, reason: str) -> Dict[str, Any]:
        """기본 응답"""
        return {
            'signal': 'HOLD',
            'detailed_signal': 'HOLD',
            'confidence': 0.0,
            'signal_value': 0,
            'direction': 'NEUTRAL',  # 추가!
            'probabilities': {'HOLD': 1.0},
            'model_used': 'None',
            'analysis': f"⚠️ {reason}"
        }


# 앙상블 서비스 캐시
_ensemble_cache: Dict[str, EnsembleModelService] = {}


def get_ensemble_service(symbol: str, timeframe: str = '5m') -> EnsembleModelService:
    """앙상블 서비스 인스턴스 가져오기 (캐시)"""
    cache_key = f"{symbol.upper()}_{timeframe}"
    
    if cache_key not in _ensemble_cache:
        _ensemble_cache[cache_key] = EnsembleModelService(symbol, timeframe)
    
    return _ensemble_cache[cache_key]

