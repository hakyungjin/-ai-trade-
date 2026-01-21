"""
AI 기반 트레이딩 전략
- LSTM 모델을 사용한 가격 예측
- 기술적 지표와 AI 예측 결합
- 신뢰도 기반 신호 생성
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging

from .technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class LSTMPredictor(nn.Module):
    """LSTM 가격 예측 모델"""

    def __init__(
        self,
        input_size: int = 20,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(LSTMPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # 완전 연결 레이어
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)

        # 마지막 타임스텝의 출력 사용
        last_output = lstm_out[:, -1, :]

        # 완전 연결 레이어
        prediction = self.fc(last_output)

        return prediction


class AIStrategy:
    """AI 기반 트레이딩 전략"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        sequence_length: int = 60,
        prediction_horizon: int = 1
    ):
        """
        Args:
            model_path: 학습된 모델 파일 경로
            sequence_length: 입력 시퀀스 길이
            prediction_horizon: 예측 기간 (캔들 수)
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 모델 로드
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.warning("No model loaded. Using default model.")
            self.model = LSTMPredictor().to(self.device)

        logger.info(f"AIStrategy initialized on {self.device}")

    def load_model(self, model_path: str):
        """모델 로드"""
        try:
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = LSTMPredictor().to(self.device)

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        특징 준비

        Args:
            df: OHLCV + 기술적 지표 데이터프레임

        Returns:
            특징 배열
        """
        # 기술적 지표 계산 (없는 경우)
        if 'rsi_14' not in df.columns:
            df = TechnicalIndicators.calculate_all_indicators(df)

        # 사용할 특징 선택
        feature_columns = [
            'close', 'volume',
            'rsi_14', 'macd', 'macd_signal',
            'ema_12', 'ema_26',
            'bb_upper', 'bb_middle', 'bb_lower',
            'stoch_k', 'stoch_d',
            'atr_14'
        ]

        # 특징 추출
        features = []
        for col in feature_columns:
            if col in df.columns:
                features.append(df[col].values)

        # (samples, features) 형태로 변환
        features_array = np.column_stack(features)

        # 정규화 (0-1)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        features_normalized = scaler.fit_transform(features_array)

        return features_normalized

    def create_sequences(
        self,
        data: np.ndarray,
        sequence_length: int
    ) -> np.ndarray:
        """
        시퀀스 생성

        Args:
            data: 특징 배열
            sequence_length: 시퀀스 길이

        Returns:
            시퀀스 배열 (samples, sequence_length, features)
        """
        sequences = []

        for i in range(len(data) - sequence_length):
            seq = data[i:i + sequence_length]
            sequences.append(seq)

        return np.array(sequences)

    def predict_price(
        self,
        df: pd.DataFrame,
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        가격 예측

        Args:
            df: OHLCV 데이터프레임
            return_confidence: 신뢰도 반환 여부

        Returns:
            예측 결과
        """
        if self.model is None:
            return {
                'prediction': df.iloc[-1]['close'],
                'confidence': 0.0,
                'direction': 'neutral'
            }

        try:
            # 특징 준비
            features = self.prepare_features(df)

            # 시퀀스 생성
            if len(features) < self.sequence_length:
                logger.warning(f"Not enough data for prediction (need {self.sequence_length}, got {len(features)})")
                return {
                    'prediction': df.iloc[-1]['close'],
                    'confidence': 0.0,
                    'direction': 'neutral'
                }

            # 마지막 시퀀스 추출
            last_sequence = features[-self.sequence_length:]
            last_sequence = last_sequence.reshape(1, self.sequence_length, -1)

            # 텐서 변환
            x = torch.FloatTensor(last_sequence).to(self.device)

            # 예측
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(x)
                predicted_price = prediction.item()

            # 현재 가격
            current_price = df.iloc[-1]['close']

            # 가격 변화 예측
            price_change_pct = (predicted_price - current_price) / current_price * 100

            # 방향 및 신뢰도
            if abs(price_change_pct) < 0.5:
                direction = 'neutral'
                confidence = 0.3
            elif price_change_pct > 0:
                direction = 'bullish'
                confidence = min(abs(price_change_pct) / 5.0, 1.0)  # 최대 5% 변화 기준
            else:
                direction = 'bearish'
                confidence = min(abs(price_change_pct) / 5.0, 1.0)

            result = {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change_pct': price_change_pct,
                'direction': direction,
                'confidence': confidence
            }

            logger.debug(f"AI Prediction: {direction} ({price_change_pct:.2f}%), confidence={confidence:.2f}")

            return result

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'prediction': df.iloc[-1]['close'],
                'confidence': 0.0,
                'direction': 'neutral'
            }

    def generate_signal(
        self,
        df: pd.DataFrame,
        combine_with_indicators: bool = True
    ) -> Dict[str, Any]:
        """
        AI 기반 매매 신호 생성

        Args:
            df: OHLCV 데이터프레임
            combine_with_indicators: 기술적 지표와 결합 여부

        Returns:
            매매 신호
        """
        # AI 예측
        prediction = self.predict_price(df)

        # 기본 신호
        ai_signal = {
            'signal': 'neutral',
            'confidence': prediction['confidence'],
            'ai_prediction': prediction
        }

        # 방향에 따른 신호
        if prediction['direction'] == 'bullish' and prediction['confidence'] > 0.5:
            ai_signal['signal'] = 'strong_buy' if prediction['confidence'] > 0.7 else 'buy'
        elif prediction['direction'] == 'bearish' and prediction['confidence'] > 0.5:
            ai_signal['signal'] = 'strong_sell' if prediction['confidence'] > 0.7 else 'sell'

        # 기술적 지표와 결합
        if combine_with_indicators:
            indicator_signals = TechnicalIndicators.get_signal_summary(
                TechnicalIndicators.calculate_all_indicators(df)
            )

            # 지표 신호 점수 계산
            indicator_score = self._calculate_indicator_score(indicator_signals)

            # AI 신호와 가중 평균
            combined_score = (
                prediction['confidence'] * (1 if prediction['direction'] == 'bullish' else -1) * 0.6 +
                indicator_score * 0.4
            )

            # 최종 신호 결정
            if combined_score > 0.6:
                final_signal = 'strong_buy'
            elif combined_score > 0.3:
                final_signal = 'buy'
            elif combined_score < -0.6:
                final_signal = 'strong_sell'
            elif combined_score < -0.3:
                final_signal = 'sell'
            else:
                final_signal = 'neutral'

            ai_signal['signal'] = final_signal
            ai_signal['combined_score'] = combined_score
            ai_signal['indicator_score'] = indicator_score
            ai_signal['indicator_signals'] = indicator_signals

        return ai_signal

    def _calculate_indicator_score(self, signals: Dict[str, Any]) -> float:
        """
        기술적 지표 신호 점수 계산

        Args:
            signals: 지표 신호 딕셔너리

        Returns:
            점수 (-1 ~ 1)
        """
        score = 0.0
        count = 0

        # RSI
        if 'rsi' in signals:
            rsi_signal = signals['rsi']['signal']
            if rsi_signal == 'oversold':
                score += 0.8
            elif rsi_signal == 'overbought':
                score -= 0.8
            count += 1

        # MACD
        if 'macd' in signals:
            macd_signal = signals['macd']['signal']
            if macd_signal == 'bullish':
                score += 1.0
            elif macd_signal == 'bearish':
                score -= 1.0
            count += 1

        # Bollinger Bands
        if 'bollinger_bands' in signals:
            bb_signal = signals['bollinger_bands']['signal']
            if bb_signal == 'oversold':
                score += 0.6
            elif bb_signal == 'overbought':
                score -= 0.6
            count += 1

        # EMA Cross
        if 'ema_cross' in signals:
            ema_signal = signals['ema_cross']['signal']
            if ema_signal == 'bullish':
                score += 0.7
            elif ema_signal == 'bearish':
                score -= 0.7
            count += 1

        # 평균 점수
        return score / count if count > 0 else 0.0

    def backtest_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        AI 신호 백테스팅

        Args:
            df: OHLCV 데이터프레임

        Returns:
            신호가 포함된 데이터프레임
        """
        result = df.copy()

        # 기술적 지표 계산
        result = TechnicalIndicators.calculate_all_indicators(result)

        signals = []
        confidences = []

        # 각 시점마다 예측
        for i in range(self.sequence_length, len(result)):
            current_data = result.iloc[:i+1]
            signal_data = self.generate_signal(current_data, combine_with_indicators=True)

            signals.append(signal_data['signal'])
            confidences.append(signal_data['confidence'])

        # 신호 추가
        result.loc[result.index[self.sequence_length:], 'ai_signal'] = signals
        result.loc[result.index[self.sequence_length:], 'ai_confidence'] = confidences

        return result
