"""
기술적 지표 계산 모듈
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- EMA (Exponential Moving Average)
- SMA (Simple Moving Average)
- Stochastic Oscillator
- ATR (Average True Range)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """기술적 지표 계산 클래스"""

    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average (단순 이동 평균)

        Args:
            data: 가격 데이터 (보통 종가)
            period: 기간

        Returns:
            SMA 시리즈
        """
        return data.rolling(window=period).mean()

    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average (지수 이동 평균)

        Args:
            data: 가격 데이터
            period: 기간

        Returns:
            EMA 시리즈
        """
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (상대 강도 지수)

        Args:
            data: 가격 데이터 (종가)
            period: 기간 (기본 14)

        Returns:
            RSI 시리즈 (0-100)
        """
        # 가격 변화 계산
        delta = data.diff()

        # 상승/하락 분리
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # 평균 상승/하락 계산
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # RS 계산
        rs = avg_gain / avg_loss

        # RSI 계산
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_macd(
        data: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence)

        Args:
            data: 가격 데이터
            fast_period: 빠른 EMA 기간
            slow_period: 느린 EMA 기간
            signal_period: 시그널선 기간

        Returns:
            (MACD선, 시그널선, 히스토그램)
        """
        # 빠른 EMA와 느린 EMA 계산
        ema_fast = TechnicalIndicators.calculate_ema(data, fast_period)
        ema_slow = TechnicalIndicators.calculate_ema(data, slow_period)

        # MACD선 계산
        macd_line = ema_fast - ema_slow

        # 시그널선 계산
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal_period)

        # 히스토그램 계산
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(
        data: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands (볼린저 밴드)

        Args:
            data: 가격 데이터
            period: 기간
            std_dev: 표준편차 배수

        Returns:
            (상단 밴드, 중간선, 하단 밴드)
        """
        # 중간선 (SMA)
        middle_band = TechnicalIndicators.calculate_sma(data, period)

        # 표준편차 계산
        std = data.rolling(window=period).std()

        # 상단/하단 밴드
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)

        return upper_band, middle_band, lower_band

    @staticmethod
    def calculate_stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator (스토캐스틱)

        Args:
            high: 고가 데이터
            low: 저가 데이터
            close: 종가 데이터
            k_period: %K 기간
            d_period: %D 기간

        Returns:
            (%K, %D)
        """
        # 최저/최고가 계산
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        # %K 계산
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)

        # %D 계산 (K의 이동평균)
        d_percent = k_percent.rolling(window=d_period).mean()

        return k_percent, d_percent

    @staticmethod
    def calculate_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Average True Range (평균 진폭)

        Args:
            high: 고가
            low: 저가
            close: 종가
            period: 기간

        Returns:
            ATR 시리즈
        """
        # True Range 계산
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR 계산 (True Range의 이동평균)
        atr = true_range.rolling(window=period).mean()

        return atr

    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume (거래량 지표)

        Args:
            close: 종가
            volume: 거래량

        Returns:
            OBV 시리즈
        """
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

    @staticmethod
    def calculate_vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Volume Weighted Average Price (거래량 가중 평균가)

        Args:
            high: 고가
            low: 저가
            close: 종가
            volume: 거래량

        Returns:
            VWAP 시리즈
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 기술적 지표 계산

        Args:
            df: OHLCV 데이터프레임 (open, high, low, close, volume 컬럼 필요)

        Returns:
            지표가 추가된 데이터프레임
        """
        result = df.copy()

        try:
            # 이동평균선
            result['sma_20'] = TechnicalIndicators.calculate_sma(df['close'], 20)
            result['sma_50'] = TechnicalIndicators.calculate_sma(df['close'], 50)
            result['sma_200'] = TechnicalIndicators.calculate_sma(df['close'], 200)

            result['ema_12'] = TechnicalIndicators.calculate_ema(df['close'], 12)
            result['ema_26'] = TechnicalIndicators.calculate_ema(df['close'], 26)

            # RSI
            result['rsi_14'] = TechnicalIndicators.calculate_rsi(df['close'], 14)

            # MACD
            macd, signal, histogram = TechnicalIndicators.calculate_macd(df['close'])
            result['macd'] = macd
            result['macd_signal'] = signal
            result['macd_histogram'] = histogram

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(df['close'])
            result['bb_upper'] = bb_upper
            result['bb_middle'] = bb_middle
            result['bb_lower'] = bb_lower

            # Stochastic
            k_percent, d_percent = TechnicalIndicators.calculate_stochastic(
                df['high'], df['low'], df['close']
            )
            result['stoch_k'] = k_percent
            result['stoch_d'] = d_percent

            # ATR
            result['atr_14'] = TechnicalIndicators.calculate_atr(
                df['high'], df['low'], df['close'], 14
            )

            # OBV
            result['obv'] = TechnicalIndicators.calculate_obv(df['close'], df['volume'])

            # VWAP
            result['vwap'] = TechnicalIndicators.calculate_vwap(
                df['high'], df['low'], df['close'], df['volume']
            )

            logger.info(f"Successfully calculated all indicators for {len(df)} rows")

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")

        return result

    @staticmethod
    def get_signal_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """
        기술적 지표 기반 시그널 요약

        Args:
            df: 지표가 포함된 데이터프레임

        Returns:
            시그널 요약 딕셔너리
        """
        if df.empty:
            return {}

        # 최신 데이터
        latest = df.iloc[-1]

        signals = {
            'timestamp': df.index[-1],
            'price': latest['close'],
        }

        # RSI 시그널
        if 'rsi_14' in latest:
            rsi = latest['rsi_14']
            signals['rsi'] = {
                'value': rsi,
                'signal': 'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'
            }

        # MACD 시그널
        if 'macd' in latest and 'macd_signal' in latest:
            macd = latest['macd']
            macd_signal = latest['macd_signal']
            signals['macd'] = {
                'value': macd,
                'signal_line': macd_signal,
                'signal': 'bullish' if macd > macd_signal else 'bearish'
            }

        # Bollinger Bands 시그널
        if all(k in latest for k in ['bb_upper', 'bb_lower', 'close']):
            bb_upper = latest['bb_upper']
            bb_lower = latest['bb_lower']
            price = latest['close']

            if price > bb_upper:
                bb_signal = 'overbought'
            elif price < bb_lower:
                bb_signal = 'oversold'
            else:
                bb_signal = 'neutral'

            signals['bollinger_bands'] = {
                'upper': bb_upper,
                'lower': bb_lower,
                'signal': bb_signal
            }

        # 이동평균 크로스오버
        if 'ema_12' in latest and 'ema_26' in latest:
            ema_12 = latest['ema_12']
            ema_26 = latest['ema_26']
            signals['ema_cross'] = {
                'ema_12': ema_12,
                'ema_26': ema_26,
                'signal': 'bullish' if ema_12 > ema_26 else 'bearish'
            }

        # Stochastic 시그널
        if 'stoch_k' in latest:
            stoch_k = latest['stoch_k']
            signals['stochastic'] = {
                'value': stoch_k,
                'signal': 'oversold' if stoch_k < 20 else 'overbought' if stoch_k > 80 else 'neutral'
            }

        return signals
