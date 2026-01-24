"""
학습된 모델 예측 서비스
저장된 XGBoost/LSTM 모델을 로드하여 실시간 예측 수행
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from app.services.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class TrainedModelService:
    """학습된 모델을 사용한 예측 서비스"""
    
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
        
        # 레이블 매핑
        self.label_map = {
            0: ('STRONG_SELL', -2),
            1: ('SELL', -1),
            2: ('HOLD', 0),
            3: ('BUY', 1),
            4: ('STRONG_BUY', 2)
        }
        
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
            
            logger.info(f"✅ Loaded trained model from {model_path}")
            logger.info(f"   Features: {len(self.feature_names)}")
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
        """추가 피처 생성 (연속 패턴 포함)"""
        df = df.copy()
        
        # ===== 가격 변화 =====
        df['price_change_1'] = df['close'].pct_change(1)
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)
        df['price_change_20'] = df['close'].pct_change(20)
        
        # ===== 거래량 관련 =====
        df['volume_change_1'] = df['volume'].pct_change(1)
        df['volume_change_5'] = df['volume'].pct_change(5)
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_ma_ratio_5'] = df['volume'] / df['volume'].rolling(5).mean()
        df['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
        
        # ===== 가격 위치 =====
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        df['price_position_20'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)
        
        # ===== 변동성 =====
        df['volatility_5'] = df['close'].rolling(5).std() / df['close'].rolling(5).mean()
        df['volatility_20'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        
        # ===== 캔들 패턴 =====
        df['candle_body'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-8)
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-8)
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        
        # ===== 볼린저 밴드 =====
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        
        # ===== RSI =====
        if 'rsi_14' in df.columns:
            df['rsi_normalized'] = df['rsi_14'] / 100
            df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
            df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        
        # ===== MACD =====
        if 'macd' in df.columns:
            df['macd_normalized'] = df['macd'] / df['close'] * 100
        if 'macd_histogram' in df.columns:
            df['macd_hist_change'] = df['macd_histogram'].diff()
        
        # ===== EMA 크로스 =====
        if 'ema_12' in df.columns and 'ema_26' in df.columns:
            df['ema_cross'] = (df['ema_12'] - df['ema_26']) / df['close'] * 100
            df['ema_cross_signal'] = (df['ema_12'] > df['ema_26']).astype(int)
        
        # ===== Stochastic =====
        if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
            df['stoch_cross'] = (df['stoch_k'] - df['stoch_d'])
            df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
            df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        
        # ===== 연속 패턴 (모멘텀) =====
        df['is_green'] = (df['close'] > df['open']).astype(int)
        df['is_red'] = (df['close'] < df['open']).astype(int)
        df['price_up'] = (df['close'] > df['close'].shift(1)).astype(int)
        df['price_down'] = (df['close'] < df['close'].shift(1)).astype(int)
        
        # 연속 상승/하락 카운트
        def count_consecutive(series, max_count=10):
            result = []
            count = 0
            for val in series:
                if val == 1:
                    count = min(count + 1, max_count)
                else:
                    count = 0
                result.append(count)
            return result
        
        df['consecutive_green'] = count_consecutive(df['is_green'].values)
        df['consecutive_red'] = count_consecutive(df['is_red'].values)
        df['consecutive_up'] = count_consecutive(df['price_up'].values)
        df['consecutive_down'] = count_consecutive(df['price_down'].values)
        
        # 연속 패턴 강도
        df['streak_bullish'] = df['consecutive_green'] / 5
        df['streak_bearish'] = df['consecutive_red'] / 5
        df['streak_up_momentum'] = df['consecutive_up'] / 5
        df['streak_down_momentum'] = df['consecutive_down'] / 5
        
        # 최근 N봉 상승 비율
        df['green_ratio_5'] = df['is_green'].rolling(5).mean()
        df['green_ratio_10'] = df['is_green'].rolling(10).mean()
        df['up_ratio_5'] = df['price_up'].rolling(5).mean()
        df['up_ratio_10'] = df['price_up'].rolling(10).mean()
        
        # 모멘텀 신호
        df['bullish_momentum'] = (
            (df['consecutive_up'] >= 3) & 
            (df['volume'] > df['volume'].shift(1))
        ).astype(int) * df['consecutive_up']
        
        df['bearish_momentum'] = (
            (df['consecutive_down'] >= 3) & 
            (df['volume'] > df['volume'].shift(1))
        ).astype(int) * df['consecutive_down']
        
        df['strong_bullish_signal'] = (
            (df['consecutive_up'] >= 3) & 
            (df['price_change_1'] > 0.02)
        ).astype(int)
        
        df['strong_bearish_signal'] = (
            (df['consecutive_down'] >= 3) & 
            (df['price_change_1'] < -0.02)
        ).astype(int)
        
        # 누적 변화율
        df['cumulative_change_3'] = df['price_change_1'].rolling(3).sum()
        df['cumulative_change_5'] = df['price_change_1'].rolling(5).sum()
        
        return df
    
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
        """분석 텍스트 생성"""
        buy_prob = probabilities[3] + probabilities[4]  # BUY + STRONG_BUY
        sell_prob = probabilities[0] + probabilities[1]  # STRONG_SELL + SELL
        hold_prob = probabilities[2]  # HOLD
        
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
        
        # 강도 설명
        if 'STRONG' in signal:
            analysis_parts.append("강한 신호입니다. 포지션 크기 조절에 참고하세요.")
        
        return " | ".join(analysis_parts)


# 싱글톤 인스턴스
_trained_model_service: Optional[TrainedModelService] = None


def get_trained_model_service(model_name: str = 'xgboost_btcusdt_5m_v2') -> TrainedModelService:
    """학습된 모델 서비스 인스턴스 반환 (기본: 5분봉 v2 모델)"""
    global _trained_model_service
    
    if _trained_model_service is None:
        _trained_model_service = TrainedModelService()
        
        # ai-model 폴더에서 모델 로드 (backend 기준 상대 경로)
        # __file__ = backend/app/services/trained_model_service.py
        current_file = os.path.abspath(__file__)
        services_dir = os.path.dirname(current_file)      # backend/app/services
        app_dir = os.path.dirname(services_dir)           # backend/app
        backend_dir = os.path.dirname(app_dir)            # backend
        project_root = os.path.dirname(backend_dir)       # project root
        
        model_dir = os.path.join(project_root, 'ai-model', 'models')
        
        logger.info(f"📁 Looking for models in: {model_dir}")
        
        # 5분봉 v2 모델만 사용
        model_path = os.path.join(model_dir, f'{model_name}.joblib')
        scaler_path = os.path.join(model_dir, f'{model_name}_scaler.joblib')
        features_path = os.path.join(model_dir, f'{model_name}_features.joblib')
        
        logger.info(f"📁 Model path: {model_path}")
        logger.info(f"📁 Model exists: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            if _trained_model_service.load_model(model_path, scaler_path, features_path):
                logger.info(f"✅ Using trained model: {model_name}")
            else:
                logger.warning(f"⚠️ Failed to load model: {model_name}")
        else:
            logger.warning(f"📝 Model not found: {model_path}")
            # 디렉토리 내용 확인
            if os.path.exists(model_dir):
                files = os.listdir(model_dir)
                logger.info(f"📁 Available files in model_dir: {files}")
            else:
                logger.warning(f"📁 Model directory doesn't exist: {model_dir}")
            logger.info(f"   Train it using: python ai-model/scripts/train_model.py")
    
    return _trained_model_service

