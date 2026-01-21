"""
가중치 기반 트레이딩 전략
- 여러 기술적 지표를 조합하여 매매 신호 생성
- 각 지표에 가중치를 부여하여 종합 점수 계산
- 신뢰도 기반 포지션 크기 결정
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from enum import Enum
import logging

from .technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

# 가중치 설정 지연 로드 (순환 참조 방지)
def _get_weight_config():
    from .weight_config import get_weight_config
    return get_weight_config()


class Signal(Enum):
    """매매 신호 열거형"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class WeightedStrategy:
    """가중치 기반 트레이딩 전략"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 전략 설정 (가중치, 임계값 등)
        """
        # 기본 가중치 설정
        self.default_weights = {
            'rsi': 0.20,
            'macd': 0.25,
            'bollinger': 0.15,
            'ema_cross': 0.20,
            'stochastic': 0.10,
            'volume': 0.10
        }

        self.config = config or {}

        # 가중치 로드: config > weight_config > default
        if 'weights' in self.config:
            self.weights = self.config['weights']
        else:
            try:
                weight_config = _get_weight_config()
                self.weights = weight_config.get_weights()
            except:
                self.weights = self.default_weights

        # 신호 임계값
        self.thresholds = self.config.get('thresholds', {
            'strong_buy': 0.6,
            'buy': 0.3,
            'sell': -0.3,
            'strong_sell': -0.6
        })

        logger.info(f"Initialized WeightedStrategy with weights: {self.weights}")

    def refresh_weights(self):
        """가중치 새로고침 (Admin에서 변경 후)"""
        try:
            weight_config = _get_weight_config()
            self.weights = weight_config.get_weights()
            logger.info(f"Weights refreshed: {self.weights}")
        except Exception as e:
            logger.error(f"Failed to refresh weights: {e}")

    def calculate_indicator_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        각 지표의 점수 계산 (-1 ~ 1)

        Args:
            df: 기술적 지표가 포함된 데이터프레임

        Returns:
            점수가 추가된 데이터프레임
        """
        result = df.copy()

        # RSI 점수
        if 'rsi_14' in df.columns:
            result['rsi_score'] = self._calculate_rsi_score(df['rsi_14'])

        # MACD 점수
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            result['macd_score'] = self._calculate_macd_score(
                df['macd'], df['macd_signal'], df['macd_histogram']
            )

        # Bollinger Bands 점수
        if all(col in df.columns for col in ['close', 'bb_upper', 'bb_lower', 'bb_middle']):
            result['bollinger_score'] = self._calculate_bollinger_score(
                df['close'], df['bb_upper'], df['bb_lower'], df['bb_middle']
            )

        # EMA Cross 점수
        if 'ema_12' in df.columns and 'ema_26' in df.columns:
            result['ema_cross_score'] = self._calculate_ema_cross_score(
                df['ema_12'], df['ema_26']
            )

        # Stochastic 점수
        if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
            result['stochastic_score'] = self._calculate_stochastic_score(
                df['stoch_k'], df['stoch_d']
            )

        # Volume 점수
        if 'volume' in df.columns:
            result['volume_score'] = self._calculate_volume_score(df['volume'])

        return result

    def _calculate_rsi_score(self, rsi: pd.Series) -> pd.Series:
        """RSI 점수 계산 (-1 ~ 1)"""
        score = pd.Series(0.0, index=rsi.index)

        # 과매도 구간 (0-30): 매수 신호
        mask_oversold = rsi < 30
        score[mask_oversold] = (30 - rsi[mask_oversold]) / 30

        # 과매수 구간 (70-100): 매도 신호
        mask_overbought = rsi > 70
        score[mask_overbought] = -(rsi[mask_overbought] - 70) / 30

        # 중립 구간 (30-70)
        mask_neutral = (rsi >= 30) & (rsi <= 70)
        score[mask_neutral] = (50 - rsi[mask_neutral]) / 20 * 0.5

        return score

    def _calculate_macd_score(
        self,
        macd: pd.Series,
        signal: pd.Series,
        histogram: pd.Series
    ) -> pd.Series:
        """MACD 점수 계산"""
        score = pd.Series(0.0, index=macd.index)

        # 히스토그램의 변화 방향
        histogram_change = histogram.diff()

        # MACD가 시그널선 위에 있고 상승 중: 강한 매수
        mask_strong_buy = (macd > signal) & (histogram_change > 0)
        score[mask_strong_buy] = 1.0

        # MACD가 시그널선 위에 있지만 하락 중: 약한 매수
        mask_weak_buy = (macd > signal) & (histogram_change <= 0)
        score[mask_weak_buy] = 0.3

        # MACD가 시그널선 아래 있고 하락 중: 강한 매도
        mask_strong_sell = (macd < signal) & (histogram_change < 0)
        score[mask_strong_sell] = -1.0

        # MACD가 시그널선 아래 있지만 상승 중: 약한 매도
        mask_weak_sell = (macd < signal) & (histogram_change >= 0)
        score[mask_weak_sell] = -0.3

        return score

    def _calculate_bollinger_score(
        self,
        close: pd.Series,
        upper: pd.Series,
        lower: pd.Series,
        middle: pd.Series
    ) -> pd.Series:
        """Bollinger Bands 점수 계산"""
        score = pd.Series(0.0, index=close.index)

        # 밴드 폭
        band_width = upper - lower

        # 가격의 밴드 내 위치 (0 ~ 1)
        position = (close - lower) / band_width

        # 하단 밴드 근처: 매수 신호
        mask_lower = position < 0.2
        score[mask_lower] = (0.2 - position[mask_lower]) * 5

        # 상단 밴드 근처: 매도 신호
        mask_upper = position > 0.8
        score[mask_upper] = -(position[mask_upper] - 0.8) * 5

        # 중간: 중립
        mask_middle = (position >= 0.2) & (position <= 0.8)
        score[mask_middle] = (0.5 - position[mask_middle]) * 0.5

        return score

    def _calculate_ema_cross_score(
        self,
        ema_fast: pd.Series,
        ema_slow: pd.Series
    ) -> pd.Series:
        """EMA 크로스오버 점수 계산"""
        score = pd.Series(0.0, index=ema_fast.index)

        # EMA 차이의 비율
        diff_pct = (ema_fast - ema_slow) / ema_slow * 100

        # 골든 크로스 확인 (빠른 EMA가 느린 EMA를 상향 돌파)
        cross_up = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        score[cross_up] = 1.0

        # 데드 크로스 확인 (빠른 EMA가 느린 EMA를 하향 돌파)
        cross_down = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
        score[cross_down] = -1.0

        # 크로스가 아닌 경우: 차이에 비례
        no_cross = ~(cross_up | cross_down)
        score[no_cross] = np.clip(diff_pct[no_cross] / 2, -1, 1)

        return score

    def _calculate_stochastic_score(
        self,
        k: pd.Series,
        d: pd.Series
    ) -> pd.Series:
        """Stochastic 점수 계산"""
        score = pd.Series(0.0, index=k.index)

        # 과매도 구간 (K < 20): 매수 신호
        mask_oversold = k < 20
        score[mask_oversold] = (20 - k[mask_oversold]) / 20

        # 과매수 구간 (K > 80): 매도 신호
        mask_overbought = k > 80
        score[mask_overbought] = -(k[mask_overbought] - 80) / 20

        # K선이 D선을 상향 돌파: 강한 매수
        cross_up = (k > d) & (k.shift(1) <= d.shift(1))
        score[cross_up] = 1.0

        # K선이 D선을 하향 돌파: 강한 매도
        cross_down = (k < d) & (k.shift(1) >= d.shift(1))
        score[cross_down] = -1.0

        return score

    def _calculate_volume_score(self, volume: pd.Series) -> pd.Series:
        """거래량 점수 계산"""
        # 거래량 이동평균
        volume_ma = volume.rolling(window=20).mean()

        # 거래량 비율
        volume_ratio = volume / volume_ma

        # 거래량 증가: 긍정적 신호
        score = np.clip((volume_ratio - 1) * 0.5, -1, 1)

        return score

    def calculate_combined_score(self, df: pd.DataFrame) -> pd.Series:
        """
        가중치를 적용한 종합 점수 계산

        Args:
            df: 지표 점수가 포함된 데이터프레임

        Returns:
            종합 점수 시리즈 (-1 ~ 1)
        """
        combined_score = pd.Series(0.0, index=df.index)

        # 각 지표 점수에 가중치 적용
        score_columns = {
            'rsi_score': 'rsi',
            'macd_score': 'macd',
            'bollinger_score': 'bollinger',
            'ema_cross_score': 'ema_cross',
            'stochastic_score': 'stochastic',
            'volume_score': 'volume'
        }

        for col, weight_key in score_columns.items():
            if col in df.columns:
                weight = self.weights.get(weight_key, 0)
                combined_score += df[col] * weight

        return combined_score

    def generate_signal(self, score: float) -> Signal:
        """
        점수를 기반으로 매매 신호 생성

        Args:
            score: 종합 점수

        Returns:
            매매 신호
        """
        if score >= self.thresholds['strong_buy']:
            return Signal.STRONG_BUY
        elif score >= self.thresholds['buy']:
            return Signal.BUY
        elif score <= self.thresholds['strong_sell']:
            return Signal.STRONG_SELL
        elif score <= self.thresholds['sell']:
            return Signal.SELL
        else:
            return Signal.NEUTRAL

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        전체 분석 실행

        Args:
            df: OHLCV 데이터프레임

        Returns:
            분석 결과
        """
        # 기술적 지표 계산
        df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)

        # 지표 점수 계산
        df_with_scores = self.calculate_indicator_scores(df_with_indicators)

        # 종합 점수 계산
        combined_score = self.calculate_combined_score(df_with_scores)
        df_with_scores['combined_score'] = combined_score

        # 최신 데이터
        latest = df_with_scores.iloc[-1]
        latest_score = latest['combined_score']

        # 신호 생성
        signal = self.generate_signal(latest_score)

        # 신뢰도 계산 (점수의 절대값)
        confidence = abs(latest_score)

        # 개별 지표 점수
        indicator_scores = {}
        for col in df_with_scores.columns:
            if col.endswith('_score') and col != 'combined_score':
                indicator_scores[col] = latest[col]

        result = {
            'timestamp': df_with_scores.index[-1],
            'price': latest['close'],
            'signal': signal.value,
            'combined_score': latest_score,
            'confidence': confidence,
            'indicator_scores': indicator_scores,
            'indicators': {
                'rsi': latest.get('rsi_14'),
                'macd': latest.get('macd'),
                'macd_signal': latest.get('macd_signal'),
                'ema_12': latest.get('ema_12'),
                'ema_26': latest.get('ema_26'),
                'stoch_k': latest.get('stoch_k'),
            },
            'recommendation': self._get_recommendation(signal, confidence, latest['close'])
        }

        logger.info(f"Analysis complete: {signal.value} (score: {latest_score:.3f})")

        return result

    def _get_recommendation(self, signal: Signal, confidence: float, price: float) -> Dict[str, Any]:
        """
        매매 권장사항 생성

        Args:
            signal: 매매 신호
            confidence: 신뢰도
            price: 현재 가격

        Returns:
            권장사항
        """
        recommendation = {
            'action': signal.value,
            'confidence_level': 'high' if confidence > 0.7 else 'medium' if confidence > 0.4 else 'low',
            'position_size_pct': confidence * 100,  # 신뢰도에 비례한 포지션 크기
        }

        # 진입가/익절가/손절가 제안
        if signal in [Signal.STRONG_BUY, Signal.BUY]:
            recommendation['entry_price'] = price
            recommendation['take_profit'] = price * 1.02  # 2% 익절
            recommendation['stop_loss'] = price * 0.98  # 2% 손절

        elif signal in [Signal.STRONG_SELL, Signal.SELL]:
            recommendation['entry_price'] = price
            recommendation['take_profit'] = price * 0.98  # 2% 익절 (숏)
            recommendation['stop_loss'] = price * 1.02  # 2% 손절 (숏)

        return recommendation

    def backtest_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        과거 데이터에 대한 신호 생성 (백테스팅용)

        Args:
            df: OHLCV 데이터프레임

        Returns:
            신호가 포함된 데이터프레임
        """
        # 기술적 지표 계산
        df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)

        # 지표 점수 계산
        df_with_scores = self.calculate_indicator_scores(df_with_indicators)

        # 종합 점수 계산
        combined_score = self.calculate_combined_score(df_with_scores)
        df_with_scores['combined_score'] = combined_score

        # 신호 생성
        df_with_scores['signal'] = combined_score.apply(
            lambda x: self.generate_signal(x).value
        )

        # 신뢰도
        df_with_scores['confidence'] = combined_score.abs()

        return df_with_scores
