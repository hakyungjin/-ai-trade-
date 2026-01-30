"""
통합 피처 엔지니어링 모듈
학습(training)과 추론(inference) 모두에서 동일한 피처를 사용하기 위한 단일 소스

주요 목적:
- 학습 파이프라인과 추론 서비스 간 피처 불일치 방지
- 모든 피처를 한 곳에서 관리
- 피처 카테고리별 그룹화
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ===== 피처 카테고리 정의 =====

# 기본 기술적 지표 (TechnicalIndicators.calculate_all_indicators에서 생성)
BASE_INDICATOR_FEATURES = [
    'sma_20', 'sma_50',
    'ema_12', 'ema_26',
    'rsi_14',
    'macd', 'macd_signal', 'macd_histogram',
    'bb_upper', 'bb_middle', 'bb_lower',
    'stoch_k', 'stoch_d',
    'atr_14',
    'obv',
]

# 가격 변화율 피처
PRICE_CHANGE_FEATURES = [
    'price_change_1', 'price_change_5', 'price_change_10', 'price_change_20',
]

# 거래량 관련 피처
VOLUME_FEATURES = [
    'volume_change_1', 'volume_change_5', 'volume_change_10',
    'volume_ma_ratio', 'volume_ma_ratio_5', 'volume_ma_ratio_10',
    'volume_spike', 'volume_spike_3x',
    'volume_surge_intensity',
    'volume_price_trend', 'volume_price_correlation',
    'volume_momentum_5', 'volume_momentum_10',
    'volume_breakout', 'volume_trend',
]

# OBV 관련 피처
OBV_FEATURES = [
    'obv_ma_20', 'obv_slope', 'obv_divergence',
]

# MFI/Williams 관련 피처
MOMENTUM_OSCILLATOR_FEATURES = [
    'mfi', 'mfi_normalized', 'mfi_overbought', 'mfi_oversold',
    'williams_r', 'williams_overbought', 'williams_oversold',
]

# 볼린저 밴드 파생 피처
BB_DERIVED_FEATURES = [
    'bb_position', 'bb_width',
]

# RSI 파생 피처
RSI_DERIVED_FEATURES = [
    'rsi_normalized', 'rsi_overbought', 'rsi_oversold',
]

# MACD 파생 피처
MACD_DERIVED_FEATURES = [
    'macd_normalized', 'macd_hist_change',
]

# EMA 크로스 피처
EMA_CROSS_FEATURES = [
    'ema_cross', 'ema_cross_signal',
]

# Stochastic 파생 피처
STOCH_DERIVED_FEATURES = [
    'stoch_cross', 'stoch_overbought', 'stoch_oversold',
]

# 가격 위치 피처
PRICE_POSITION_FEATURES = [
    'price_position', 'price_position_20',
    'high_20', 'low_20',
]

# 변동성 피처
VOLATILITY_FEATURES = [
    'volatility_5', 'volatility_20', 'atr_ratio',
]

# 캔들 패턴 피처
CANDLE_PATTERN_FEATURES = [
    'candle_body', 'upper_shadow', 'lower_shadow', 'is_bullish',
    'pattern_doji', 'pattern_hammer', 'pattern_inverted_hammer',
    'pattern_bullish_engulfing', 'pattern_bearish_engulfing',
    'pattern_shooting_star',
]

# 펌프 앤 덤프 감지 피처
PUMP_DUMP_FEATURES = [
    'pump_3', 'pump_6', 'pump_12',
    'drawdown_from_high_12', 'drawdown_from_high_24',
    'pump_then_dump', 'dump_then_pump',
    'overheated', 'oversold_bounce',
    'volatility_spike',
    'near_high', 'near_low',
    'pump_strength', 'dump_strength',
]

# 연속 패턴 (모멘텀) 피처
STREAK_FEATURES = [
    'consecutive_green', 'consecutive_red',
    'consecutive_up', 'consecutive_down',
    'streak_bullish', 'streak_bearish',
    'green_ratio_5', 'green_ratio_10',
    'up_ratio_5', 'up_ratio_10',
    'bullish_momentum', 'bearish_momentum',
    'cumulative_change_3', 'cumulative_change_5',
]

# 복합 신호 피처
COMPOSITE_SIGNAL_FEATURES = [
    'strong_buy_signal', 'strong_sell_signal',
]


def get_all_feature_names() -> List[str]:
    """모든 피처 이름 반환 (학습 및 추론에서 사용할 전체 피처 목록)"""
    all_features = (
        PRICE_CHANGE_FEATURES +
        VOLUME_FEATURES +
        OBV_FEATURES +
        MOMENTUM_OSCILLATOR_FEATURES +
        BB_DERIVED_FEATURES +
        RSI_DERIVED_FEATURES +
        MACD_DERIVED_FEATURES +
        EMA_CROSS_FEATURES +
        STOCH_DERIVED_FEATURES +
        PRICE_POSITION_FEATURES +
        VOLATILITY_FEATURES +
        CANDLE_PATTERN_FEATURES +
        PUMP_DUMP_FEATURES +
        STREAK_FEATURES +
        COMPOSITE_SIGNAL_FEATURES
    )
    return all_features


def get_core_feature_names() -> List[str]:
    """핵심 피처 이름 반환 (가볍고 빠른 학습에 적합)"""
    core_features = (
        PRICE_CHANGE_FEATURES +
        ['volume_ma_ratio', 'volume_surge_intensity', 'volume_spike'] +
        OBV_FEATURES +
        ['mfi_normalized', 'williams_r'] +
        BB_DERIVED_FEATURES +
        RSI_DERIVED_FEATURES +
        MACD_DERIVED_FEATURES +
        EMA_CROSS_FEATURES +
        STOCH_DERIVED_FEATURES +
        ['price_position', 'price_position_20'] +
        VOLATILITY_FEATURES +
        ['candle_body', 'upper_shadow', 'lower_shadow', 'is_bullish'] +
        ['pump_6', 'drawdown_from_high_12', 'overheated', 'oversold_bounce'] +
        ['consecutive_up', 'consecutive_down', 'green_ratio_5', 'up_ratio_5'] +
        COMPOSITE_SIGNAL_FEATURES
    )
    return core_features


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    전체 피처 계산 (통합 피처 엔지니어링)

    입력: OHLCV 데이터프레임 (open, high, low, close, volume)
    + 이미 TechnicalIndicators.calculate_all_indicators()가 적용된 데이터도 지원

    출력: 모든 파생 피처가 추가된 데이터프레임
    """
    df = df.copy()

    # ===== 가격 변화율 =====
    df['price_change_1'] = df['close'].pct_change(1)
    df['price_change_5'] = df['close'].pct_change(5)
    df['price_change_10'] = df['close'].pct_change(10)
    df['price_change_20'] = df['close'].pct_change(20)

    # ===== OBV (On Balance Volume) =====
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv

    df['obv_ma_20'] = df['obv'].rolling(20).mean()
    df['obv_slope'] = (df['obv'] - df['obv'].shift(5)) / (df['obv'].shift(5).abs() + 1e-8)

    price_direction = np.sign(df['close'] - df['close'].shift(5))
    obv_direction = np.sign(df['obv'] - df['obv'].shift(5))
    df['obv_divergence'] = (price_direction != obv_direction).astype(int)

    # ===== MFI (Money Flow Index) =====
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']

    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    positive_mf = positive_flow.rolling(14).sum()
    negative_mf = negative_flow.rolling(14).sum()

    mfi_ratio = positive_mf / (negative_mf + 1e-8)
    df['mfi'] = 100 - (100 / (1 + mfi_ratio))
    df['mfi_normalized'] = df['mfi'] / 100
    df['mfi_overbought'] = (df['mfi'] > 80).astype(int)
    df['mfi_oversold'] = (df['mfi'] < 20).astype(int)

    # ===== Williams %R =====
    highest_high = df['high'].rolling(14).max()
    lowest_low = df['low'].rolling(14).min()
    df['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low + 1e-8)
    df['williams_overbought'] = (df['williams_r'] > -20).astype(int)
    df['williams_oversold'] = (df['williams_r'] < -80).astype(int)

    # ===== ATR 비율 =====
    if 'atr_14' in df.columns:
        df['atr_ratio'] = df['atr_14'] / df['close'] * 100
    else:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        df['atr_ratio'] = df['atr_14'] / df['close'] * 100

    # ===== 캔들 패턴 =====
    body = df['close'] - df['open']
    body_abs = abs(body)
    upper_shadow_val = df['high'] - df[['open', 'close']].max(axis=1)
    lower_shadow_val = df[['open', 'close']].min(axis=1) - df['low']
    candle_range = df['high'] - df['low']

    df['pattern_doji'] = (body_abs < candle_range * 0.1).astype(int)
    df['pattern_hammer'] = (
        (lower_shadow_val > body_abs * 2) &
        (upper_shadow_val < body_abs * 0.5) &
        (df['close'] > df['open'])
    ).astype(int)
    df['pattern_inverted_hammer'] = (
        (upper_shadow_val > body_abs * 2) &
        (lower_shadow_val < body_abs * 0.5) &
        (df['close'] > df['open'])
    ).astype(int)

    prev_body = (df['close'].shift(1) - df['open'].shift(1)).abs()
    df['pattern_bullish_engulfing'] = (
        (df['close'].shift(1) < df['open'].shift(1)) &
        (df['close'] > df['open']) &
        (body_abs > prev_body * 1.5) &
        (df['open'] < df['close'].shift(1)) &
        (df['close'] > df['open'].shift(1))
    ).astype(int)
    df['pattern_bearish_engulfing'] = (
        (df['close'].shift(1) > df['open'].shift(1)) &
        (df['close'] < df['open']) &
        (body_abs > prev_body * 1.5) &
        (df['open'] > df['close'].shift(1)) &
        (df['close'] < df['open'].shift(1))
    ).astype(int)
    df['pattern_shooting_star'] = (
        (upper_shadow_val > body_abs * 2) &
        (lower_shadow_val < candle_range * 0.1) &
        (df['close'] < df['open'])
    ).astype(int)

    # ===== 거래량 관련 =====
    df['volume_change_1'] = df['volume'].pct_change(1)
    df['volume_change_5'] = df['volume'].pct_change(5)
    df['volume_change_10'] = df['volume'].pct_change(10)
    df['volume_ma_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)
    df['volume_ma_ratio_5'] = df['volume'] / (df['volume'].rolling(5).mean() + 1e-8)
    df['volume_ma_ratio_10'] = df['volume'] / (df['volume'].rolling(10).mean() + 1e-8)

    vol_ma = df['volume'].rolling(20).mean()
    df['volume_spike'] = (df['volume'] > vol_ma * 2).astype(int)
    df['volume_spike_3x'] = (df['volume'] > vol_ma * 3).astype(int)

    df['volume_surge_intensity'] = (df['volume'] / (vol_ma + 1e-8)).clip(upper=10)
    df['volume_price_trend'] = df['volume_change_1'] * df['price_change_1'] * 100
    df['volume_price_correlation'] = df['volume'].rolling(10).corr(df['close'])
    df['volume_momentum_5'] = df['volume'].rolling(5).mean() / (vol_ma + 1e-8)
    df['volume_momentum_10'] = df['volume'].rolling(10).mean() / (vol_ma + 1e-8)

    df['volume_breakout'] = (
        (df['volume'] > vol_ma * 2) &
        (abs(df['price_change_1']) > 0.01)
    ).astype(int)
    df['volume_trend'] = (df['volume'].rolling(5).mean() - vol_ma) / (vol_ma + 1e-8)

    # ===== 가격 위치 =====
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    df['price_position_20'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)

    # ===== 변동성 =====
    df['volatility_5'] = df['close'].rolling(5).std() / (df['close'].rolling(5).mean() + 1e-8)
    df['volatility_20'] = df['close'].rolling(20).std() / (df['close'].rolling(20).mean() + 1e-8)

    # ===== 캔들 비율 =====
    df['candle_body'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-8)
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['is_bullish'] = (df['close'] > df['open']).astype(int)

    # ===== 볼린저 밴드 파생 =====
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['close'] + 1e-8)

    # ===== RSI 파생 =====
    if 'rsi_14' in df.columns:
        df['rsi_normalized'] = df['rsi_14'] / 100
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)

    # ===== MACD 파생 =====
    if 'macd' in df.columns:
        df['macd_normalized'] = df['macd'] / (df['close'] + 1e-8) * 100
    if 'macd_histogram' in df.columns:
        df['macd_hist_change'] = df['macd_histogram'].diff()

    # ===== EMA 크로스 =====
    if 'ema_12' in df.columns and 'ema_26' in df.columns:
        df['ema_cross'] = (df['ema_12'] - df['ema_26']) / (df['close'] + 1e-8) * 100
        df['ema_cross_signal'] = (df['ema_12'] > df['ema_26']).astype(int)

    # ===== Stochastic 파생 =====
    if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
        df['stoch_cross'] = (df['stoch_k'] - df['stoch_d'])
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)

    # ===== 펌프 앤 덤프 패턴 감지 =====
    df['pump_3'] = df['close'].pct_change(3)
    df['pump_6'] = df['close'].pct_change(6)
    df['pump_12'] = df['close'].pct_change(12)

    df['high_12'] = df['high'].rolling(12).max()
    df['high_24'] = df['high'].rolling(24).max()
    df['drawdown_from_high_12'] = (df['close'] - df['high_12']) / (df['high_12'] + 1e-8)
    df['drawdown_from_high_24'] = (df['close'] - df['high_24']) / (df['high_24'] + 1e-8)

    df['pump_then_dump'] = (
        (df['pump_6'] > 0.03) &
        (df['price_change_1'] < -0.01)
    ).astype(int)
    df['dump_then_pump'] = (
        (df['drawdown_from_high_12'] < -0.05) &
        (df['price_change_1'] > 0.005)
    ).astype(int)

    rsi_condition = df['rsi_14'] > 70 if 'rsi_14' in df.columns else df['pump_12'] > 0.08
    df['overheated'] = (
        (df['pump_12'] > 0.05) & rsi_condition
    ).astype(int)
    df['oversold_bounce'] = (
        (df['drawdown_from_high_24'] < -0.08) &
        (df['price_change_1'] > 0)
    ).astype(int)

    df['volatility_spike'] = (
        df['volatility_5'] > df['volatility_20'] * 1.5
    ).astype(int)
    df['near_high'] = (df['close'] > df['high_24'] * 0.98).astype(int)
    df['near_low'] = (df['close'] < df['low_20'] * 1.02).astype(int)

    df['pump_strength'] = df['pump_6'] * df['volume_surge_intensity']
    df['dump_strength'] = abs(df['drawdown_from_high_12']) * df['volume_surge_intensity']

    # ===== 연속 패턴 (모멘텀) =====
    df['is_green'] = (df['close'] > df['open']).astype(int)
    df['is_red'] = (df['close'] < df['open']).astype(int)
    df['price_up'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['price_down'] = (df['close'] < df['close'].shift(1)).astype(int)

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

    df['streak_bullish'] = df['consecutive_green'] / 5
    df['streak_bearish'] = df['consecutive_red'] / 5
    df['green_ratio_5'] = df['is_green'].rolling(5).mean()
    df['green_ratio_10'] = df['is_green'].rolling(10).mean()
    df['up_ratio_5'] = df['price_up'].rolling(5).mean()
    df['up_ratio_10'] = df['price_up'].rolling(10).mean()

    df['bullish_momentum'] = (
        (df['consecutive_up'] >= 3) &
        (df['volume'] > df['volume'].shift(1))
    ).astype(int) * df['consecutive_up']
    df['bearish_momentum'] = (
        (df['consecutive_down'] >= 3) &
        (df['volume'] > df['volume'].shift(1))
    ).astype(int) * df['consecutive_down']

    df['cumulative_change_3'] = df['price_change_1'].rolling(3).sum()
    df['cumulative_change_5'] = df['price_change_1'].rolling(5).sum()

    # ===== 복합 신호 =====
    df['strong_buy_signal'] = (
        (df['obv_slope'] > 0.1) &
        (df['volume'] > vol_ma * 1.5) &
        (df['close'] > df['close'].shift(1))
    ).astype(int)
    df['strong_sell_signal'] = (
        (df['obv_slope'] < -0.1) &
        (df['volume'] > vol_ma * 1.5) &
        (df['close'] < df['close'].shift(1))
    ).astype(int)

    # 임시 컬럼 제거
    temp_cols = ['is_green', 'is_red', 'price_up', 'price_down', 'high_12', 'high_24']
    df.drop(columns=[c for c in temp_cols if c in df.columns], inplace=True, errors='ignore')

    return df


def create_labels(
    df: pd.DataFrame,
    threshold: float = 0.02,
    lookahead: int = 5,
    num_classes: int = 3
) -> pd.DataFrame:
    """
    학습용 레이블 생성

    Args:
        df: 가격 데이터
        threshold: 상승/하락 판단 기준
        lookahead: 미래 캔들 수
        num_classes: 2 (BUY/SELL), 3 (BUY/HOLD/SELL), 5 (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL)

    Returns:
        DataFrame with 'label' column
    """
    df = df.copy()
    future_return = df['close'].shift(-lookahead) / df['close'] - 1

    if num_classes == 2:
        df['label'] = (future_return > 0).astype(int)
    elif num_classes == 3:
        df['label'] = 1  # HOLD
        df.loc[future_return > threshold, 'label'] = 2  # BUY
        df.loc[future_return < -threshold, 'label'] = 0  # SELL
    elif num_classes == 5:
        strong_threshold = threshold * 2
        df['label'] = 2  # HOLD
        df.loc[future_return > threshold, 'label'] = 3  # BUY
        df.loc[future_return > strong_threshold, 'label'] = 4  # STRONG_BUY
        df.loc[future_return < -threshold, 'label'] = 1  # SELL
        df.loc[future_return < -strong_threshold, 'label'] = 0  # STRONG_SELL

    df['future_return'] = future_return

    return df


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    target_column: str = 'label'
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    학습/추론용 피처 매트릭스 준비

    Args:
        df: 피처가 계산된 데이터프레임
        feature_names: 사용할 피처 목록 (None이면 전체 피처)
        target_column: 레이블 컬럼 이름

    Returns:
        (X, y, feature_names) - 피처 배열, 레이블 배열, 실제 사용된 피처 이름
    """
    if feature_names is None:
        feature_names = get_all_feature_names()

    # 존재하는 피처만 선택
    available_features = [f for f in feature_names if f in df.columns]
    missing_features = [f for f in feature_names if f not in df.columns]

    if missing_features:
        logger.warning(f"Missing features ({len(missing_features)}): {missing_features[:10]}...")

    # NaN 행 제거
    df_clean = df.dropna(subset=available_features + ([target_column] if target_column in df.columns else []))

    X = df_clean[available_features].values

    if target_column in df_clean.columns:
        y = df_clean[target_column].values
    else:
        y = np.zeros(len(df_clean))

    logger.info(f"Feature matrix: X={X.shape}, y={y.shape}, features={len(available_features)}")

    return X, y, available_features
