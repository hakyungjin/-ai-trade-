"""
학습 데이터 준비 스크립트
DB에서 캔들 데이터를 로드하고 기술적 지표를 추가하여 학습용 데이터셋 생성
"""

import asyncio
import pandas as pd
import numpy as np
import sys
import os

# backend 디렉토리를 경로에 추가 (app 모듈 import용)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
backend_path = os.path.join(project_root, 'backend')
sys.path.insert(0, backend_path)

# ai-model 디렉토리 경로
AI_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import select
from app.database import AsyncSessionLocal
from app.models.market_data import MarketCandle
from app.services.technical_indicators import TechnicalIndicators


async def load_candles(symbol: str, timeframe: str, limit: int = 5000) -> pd.DataFrame:
    """DB에서 캔들 데이터 로드"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(MarketCandle)
            .where(MarketCandle.symbol == symbol)
            .where(MarketCandle.timeframe == timeframe)
            .order_by(MarketCandle.open_time.asc())
            .limit(limit)
        )
        candles = result.scalars().all()
        
        if not candles:
            print(f"[ERROR] No candles found for {symbol} {timeframe}")
            return pd.DataFrame()
        
        data = [{
            'timestamp': c.open_time,
            'open': float(c.open),
            'high': float(c.high),
            'low': float(c.low),
            'close': float(c.close),
            'volume': float(c.volume),
        } for c in candles]
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표 추가"""
    return TechnicalIndicators.calculate_all_indicators(df)


def create_labels(df: pd.DataFrame, future_periods: int = 5, threshold: float = 0.02, num_classes: int = 2) -> pd.DataFrame:
    """
    레이블 생성 (미래 가격 변화 기반)
    
    Args:
        future_periods: 몇 캔들 후 가격을 볼 것인지
        threshold: 상승/하락 판단 기준 (2% = 0.02)
        num_classes: 클래스 수 (2, 3 또는 5)
    
    2-class Labels (횡보 제거 - 추천!):
        1: BUY (상승)
        0: SELL (하락)
    
    3-class Labels:
        1: BUY (threshold 이상 상승)
        0: HOLD (-threshold ~ threshold)
        -1: SELL (threshold 이상 하락)
    
    5-class Labels:
        2: STRONG_BUY (5% 이상 상승)
        1: BUY (threshold 이상 상승)
        0: HOLD (-threshold ~ threshold)
        -1: SELL (threshold 이상 하락)
        -2: STRONG_SELL (5% 이상 하락)
    """
    df = df.copy()
    
    # 미래 가격
    df['future_close'] = df['close'].shift(-future_periods)
    
    # 가격 변화율
    df['price_change'] = (df['future_close'] - df['close']) / df['close']
    
    if num_classes == 2:
        # 2클래스: BUY / SELL (횡보 제거!)
        # 상승이면 BUY(1), 하락이면 SELL(0)
        def get_label_2(change):
            if pd.isna(change):
                return np.nan
            return 1 if change >= 0 else 0  # 상승/보합=BUY, 하락=SELL
        
        df['label'] = df['price_change'].apply(get_label_2)
    elif num_classes == 3:
        # 3클래스: BUY / HOLD / SELL
        def get_label_3(change):
            if pd.isna(change):
                return np.nan
            if change >= threshold:
                return 1  # BUY
            elif change <= -threshold:
                return -1  # SELL
            else:
                return 0  # HOLD
        
        df['label'] = df['price_change'].apply(get_label_3)
    else:
        # 5클래스: STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL
        strong_threshold = threshold * 2.5  # 기본적으로 threshold의 2.5배
        
        def get_label_5(change):
            if pd.isna(change):
                return np.nan
            if change >= strong_threshold:
                return 2  # STRONG_BUY
            elif change >= threshold:
                return 1  # BUY
            elif change <= -strong_threshold:
                return -2  # STRONG_SELL
            elif change <= -threshold:
                return -1  # SELL
            else:
                return 0  # HOLD
        
        df['label'] = df['price_change'].apply(get_label_5)
    
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """학습용 피처 생성 (확장 버전 + 고급 지표)"""
    df = df.copy()
    
    # ===== OBV (On Balance Volume) - 스마트머니 추적 =====
    # 가격 상승 시 거래량 누적, 하락 시 차감
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    
    # OBV 이동평균 및 기울기
    df['obv_ma_20'] = df['obv'].rolling(20).mean()
    df['obv_slope'] = (df['obv'] - df['obv'].shift(5)) / (df['obv'].shift(5).abs() + 1e-8)
    
    # OBV 다이버전스 감지 (가격 vs OBV 방향 불일치)
    price_direction = np.sign(df['close'] - df['close'].shift(5))
    obv_direction = np.sign(df['obv'] - df['obv'].shift(5))
    df['obv_divergence'] = (price_direction != obv_direction).astype(int)
    
    # ===== MFI (Money Flow Index) - 거래량 가중 RSI =====
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
    
    # ===== Williams %R - 모멘텀 =====
    highest_high = df['high'].rolling(14).max()
    lowest_low = df['low'].rolling(14).min()
    df['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low + 1e-8)
    df['williams_overbought'] = (df['williams_r'] > -20).astype(int)
    df['williams_oversold'] = (df['williams_r'] < -80).astype(int)
    
    # ===== ATR 비율 (변동성 정규화) =====
    if 'atr_14' in df.columns:
        df['atr_ratio'] = df['atr_14'] / df['close'] * 100  # ATR을 가격 대비 %로
    else:
        # ATR 직접 계산
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        df['atr_ratio'] = df['atr_14'] / df['close'] * 100
    
    # ===== 캔들 패턴 감지 =====
    body = df['close'] - df['open']
    body_abs = abs(body)
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    candle_range = df['high'] - df['low']
    
    # 도지 (Doji) - 몸통이 매우 작음 → 추세 반전 가능성
    df['pattern_doji'] = (body_abs < candle_range * 0.1).astype(int)
    
    # 망치형 (Hammer) - 하락 추세 후 반전 신호
    df['pattern_hammer'] = (
        (lower_shadow > body_abs * 2) &  # 긴 아래꼬리
        (upper_shadow < body_abs * 0.5) &  # 짧은 위꼬리
        (df['close'] > df['open'])  # 양봉
    ).astype(int)
    
    # 역망치형 (Inverted Hammer)
    df['pattern_inverted_hammer'] = (
        (upper_shadow > body_abs * 2) &
        (lower_shadow < body_abs * 0.5) &
        (df['close'] > df['open'])
    ).astype(int)
    
    # 잉걸핑 (Engulfing) - 강한 반전 신호
    prev_body = (df['close'].shift(1) - df['open'].shift(1)).abs()
    df['pattern_bullish_engulfing'] = (
        (df['close'].shift(1) < df['open'].shift(1)) &  # 이전봉 음봉
        (df['close'] > df['open']) &  # 현재봉 양봉
        (body_abs > prev_body * 1.5) &  # 현재봉이 이전봉보다 큼
        (df['open'] < df['close'].shift(1)) &  # 갭 다운
        (df['close'] > df['open'].shift(1))  # 이전 시가 돌파
    ).astype(int)
    
    df['pattern_bearish_engulfing'] = (
        (df['close'].shift(1) > df['open'].shift(1)) &
        (df['close'] < df['open']) &
        (body_abs > prev_body * 1.5) &
        (df['open'] > df['close'].shift(1)) &
        (df['close'] < df['open'].shift(1))
    ).astype(int)
    
    # 십자형 (Shooting Star / Morning Star 간소화)
    df['pattern_shooting_star'] = (
        (upper_shadow > body_abs * 2) &
        (lower_shadow < candle_range * 0.1) &
        (df['close'] < df['open'])  # 음봉
    ).astype(int)
    
    # ===== 복합 신호 =====
    # OBV 상승 + 거래량 급증 + 가격 상승 = 강한 매수 신호
    df['strong_buy_signal'] = (
        (df['obv_slope'] > 0.1) &
        (df['volume'] > df['volume'].rolling(20).mean() * 1.5) &
        (df['close'] > df['close'].shift(1))
    ).astype(int)
    
    # OBV 하락 + 거래량 급증 + 가격 하락 = 강한 매도 신호
    df['strong_sell_signal'] = (
        (df['obv_slope'] < -0.1) &
        (df['volume'] > df['volume'].rolling(20).mean() * 1.5) &
        (df['close'] < df['close'].shift(1))
    ).astype(int)
    
    # ===== 가격 관련 피처 =====
    df['price_change_1'] = df['close'].pct_change(1)
    df['price_change_5'] = df['close'].pct_change(5)
    df['price_change_10'] = df['close'].pct_change(10)
    df['price_change_20'] = df['close'].pct_change(20)
    
    # 고가/저가 대비 현재가 위치
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    # 최근 N봉 고점/저점 대비 위치
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    df['price_position_20'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)
    
    # ===== 거래량 관련 피처 (알트코인 강화) =====
    df['volume_change_1'] = df['volume'].pct_change(1)
    df['volume_change_5'] = df['volume'].pct_change(5)
    df['volume_change_10'] = df['volume'].pct_change(10)
    df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_ma_ratio_5'] = df['volume'] / df['volume'].rolling(5).mean()
    df['volume_ma_ratio_10'] = df['volume'] / df['volume'].rolling(10).mean()
    
    # 거래량 급증 감지 (다단계)
    df['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
    df['volume_spike_3x'] = (df['volume'] > df['volume'].rolling(20).mean() * 3).astype(int)
    df['volume_spike_5x'] = (df['volume'] > df['volume'].rolling(20).mean() * 5).astype(int)
    
    # 거래량 급증 강도 (연속)
    df['volume_surge_intensity'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)
    df['volume_surge_intensity'] = df['volume_surge_intensity'].clip(upper=10)  # 최대 10배
    
    # 거래량 + 가격 상관관계 (알트코인 핵심!)
    df['volume_price_trend'] = df['volume_change_1'] * df['price_change_1'] * 100  # 같은 방향이면 +
    df['volume_price_correlation'] = df['volume'].rolling(10).corr(df['close'])
    
    # 거래량 모멘텀
    df['volume_momentum_5'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
    df['volume_momentum_10'] = df['volume'].rolling(10).mean() / df['volume'].rolling(20).mean()
    
    # 거래량 기반 브레이크아웃 신호
    vol_ma = df['volume'].rolling(20).mean()
    df['volume_breakout'] = (
        (df['volume'] > vol_ma * 2) &  # 거래량 2배
        (abs(df['price_change_1']) > 0.01)  # 가격 1% 이상 변동
    ).astype(int)
    
    # 거래량 급증 + 상승 (강력 매수 신호)
    df['volume_up_signal'] = (
        (df['volume'] > vol_ma * 2) &
        (df['close'] > df['open']) &
        (df['price_change_1'] > 0.005)
    ).astype(int) * df['volume_surge_intensity']
    
    # 거래량 급증 + 하락 (강력 매도 신호)
    df['volume_down_signal'] = (
        (df['volume'] > vol_ma * 2) &
        (df['close'] < df['open']) &
        (df['price_change_1'] < -0.005)
    ).astype(int) * df['volume_surge_intensity']
    
    # 최근 거래량 트렌드 (증가/감소)
    df['volume_trend'] = (df['volume'].rolling(5).mean() - df['volume'].rolling(20).mean()) / (df['volume'].rolling(20).mean() + 1e-8)
    
    # ===== 펌프 앤 덤프 패턴 감지 (알트코인 핵심!) =====
    
    # 급등 감지 (최근 N봉 최고 상승률)
    df['pump_3'] = df['close'].pct_change(3)   # 3봉 전 대비
    df['pump_6'] = df['close'].pct_change(6)   # 6봉 전 대비
    df['pump_12'] = df['close'].pct_change(12) # 12봉 전 대비
    
    # 최근 고점 대비 하락률 (덤프 감지)
    df['high_12'] = df['high'].rolling(12).max()
    df['high_24'] = df['high'].rolling(24).max()
    df['drawdown_from_high_12'] = (df['close'] - df['high_12']) / df['high_12']
    df['drawdown_from_high_24'] = (df['close'] - df['high_24']) / df['high_24']
    
    # 급등 후 급락 패턴 (펌프 앤 덤프 신호)
    df['pump_then_dump'] = (
        (df['pump_6'] > 0.03) &  # 6봉 전 대비 3% 이상 급등했었고
        (df['price_change_1'] < -0.01)  # 현재 1% 이상 하락 중
    ).astype(int)
    
    # 급락 후 반등 패턴 (매수 기회)
    df['dump_then_pump'] = (
        (df['drawdown_from_high_12'] < -0.05) &  # 고점 대비 5% 이상 하락했고
        (df['price_change_1'] > 0.005)  # 현재 반등 중
    ).astype(int)
    
    # 과열 감지 (너무 많이 올랐을 때 - 매도 신호)
    df['overheated'] = (
        (df['pump_12'] > 0.05) &  # 12봉 동안 5% 이상 상승
        (df['rsi_14'] > 70 if 'rsi_14' in df.columns else df['pump_12'] > 0.08)  # RSI 과매수
    ).astype(int)
    
    # 과매도 감지 (너무 많이 떨어졌을 때 - 매수 신호)
    df['oversold_bounce'] = (
        (df['drawdown_from_high_24'] < -0.08) &  # 24봉 고점 대비 8% 이상 하락
        (df['price_change_1'] > 0)  # 반등 시작
    ).astype(int)
    
    # ===== 변동성 피처 (먼저 정의!) =====
    df['volatility_5'] = df['close'].rolling(5).std() / df['close'].rolling(5).mean()
    df['volatility_20'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
    
    # 변동성 급증 (큰 움직임 예고)
    df['volatility_spike'] = (
        df['volatility_5'] > df['volatility_20'] * 1.5
    ).astype(int)
    
    # 고점 근처 vs 저점 근처
    df['near_high'] = (df['close'] > df['high_24'] * 0.98).astype(int)  # 고점 2% 이내
    df['near_low'] = (df['close'] < df['low_20'] * 1.02).astype(int)   # 저점 2% 이내
    
    # 급등 강도 (거래량 동반)
    df['pump_strength'] = df['pump_6'] * df['volume_surge_intensity']
    
    # 급락 강도
    df['dump_strength'] = abs(df['drawdown_from_high_12']) * df['volume_surge_intensity']
    
    # 캔들 패턴 피처
    df['candle_body'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-8)
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['is_bullish'] = (df['close'] > df['open']).astype(int)
    
    # ===== 기술적 지표 관련 피처 =====
    # 볼린저 밴드 위치
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
    
    # RSI 관련
    if 'rsi_14' in df.columns:
        df['rsi_normalized'] = df['rsi_14'] / 100
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
    
    # MACD 관련
    if 'macd' in df.columns:
        df['macd_normalized'] = df['macd'] / df['close'] * 100
    if 'macd_histogram' in df.columns:
        df['macd_hist_change'] = df['macd_histogram'].diff()
    
    # EMA 크로스 피처
    if 'ema_12' in df.columns and 'ema_26' in df.columns:
        df['ema_cross'] = (df['ema_12'] - df['ema_26']) / df['close'] * 100
        df['ema_cross_signal'] = (df['ema_12'] > df['ema_26']).astype(int)
    
    # Stochastic 관련
    if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
        df['stoch_cross'] = (df['stoch_k'] - df['stoch_d'])
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
    
    # ===== 시간 관련 피처 (선택적) =====
    if 'open_time' in df.columns:
        try:
            df['hour'] = pd.to_datetime(df['open_time']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['open_time']).dt.dayofweek
            # 사인/코사인 변환 (순환 특성 반영)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        except:
            pass
    
    # ===== 연속 패턴 피처 (모멘텀 감지) =====
    # 양봉/음봉 여부
    df['is_green'] = (df['close'] > df['open']).astype(int)
    df['is_red'] = (df['close'] < df['open']).astype(int)
    
    # 가격 상승/하락 여부
    df['price_up'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['price_down'] = (df['close'] < df['close'].shift(1)).astype(int)
    
    # 연속 양봉 카운트 (최대 10봉까지)
    def count_consecutive(series, max_count=10):
        """연속된 1의 개수를 카운트"""
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
    
    # 연속 패턴 강도 (정규화)
    df['streak_bullish'] = df['consecutive_green'] / 5  # 5봉 연속이면 1.0
    df['streak_bearish'] = df['consecutive_red'] / 5
    df['streak_up_momentum'] = df['consecutive_up'] / 5
    df['streak_down_momentum'] = df['consecutive_down'] / 5
    
    # 최근 N봉 중 양봉 비율
    df['green_ratio_5'] = df['is_green'].rolling(5).mean()
    df['green_ratio_10'] = df['is_green'].rolling(10).mean()
    
    # 최근 N봉 중 상승 비율
    df['up_ratio_5'] = df['price_up'].rolling(5).mean()
    df['up_ratio_10'] = df['price_up'].rolling(10).mean()
    
    # 연속 상승 + 거래량 증가 (강한 모멘텀 신호)
    df['bullish_momentum'] = (
        (df['consecutive_up'] >= 3) & 
        (df['volume'] > df['volume'].shift(1))
    ).astype(int) * df['consecutive_up']
    
    df['bearish_momentum'] = (
        (df['consecutive_down'] >= 3) & 
        (df['volume'] > df['volume'].shift(1))
    ).astype(int) * df['consecutive_down']
    
    # 연속 상승 후 급등 감지 (3봉 연속 상승 + 현재 2% 이상 상승)
    df['strong_bullish_signal'] = (
        (df['consecutive_up'] >= 3) & 
        (df['price_change_1'] > 0.02)
    ).astype(int)
    
    df['strong_bearish_signal'] = (
        (df['consecutive_down'] >= 3) & 
        (df['price_change_1'] < -0.02)
    ).astype(int)
    
    # 누적 변화율 (최근 N봉 합계)
    df['cumulative_change_3'] = df['price_change_1'].rolling(3).sum()
    df['cumulative_change_5'] = df['price_change_1'].rolling(5).sum()
    
    return df


async def prepare_training_dataset(
    symbol: str = 'BTCUSDT',
    timeframe: str = '1h',
    limit: int = 5000,
    future_periods: int = 5,
    threshold: float = 0.02,
    num_classes: int = 3
) -> pd.DataFrame:
    """학습 데이터셋 준비"""
    print(f"[1/4] Loading candles for {symbol} {timeframe}...")
    df = await load_candles(symbol, timeframe, limit)
    
    if df.empty:
        return df
    
    print(f"      Loaded {len(df)} candles")
    
    print("[2/4] Adding technical indicators...")
    df = add_technical_indicators(df)
    
    print(f"[3/4] Creating labels ({num_classes} classes)...")
    df = create_labels(df, future_periods, threshold, num_classes)
    
    print("[4/4] Creating features...")
    df = create_features(df)
    
    # NaN 제거
    df = df.dropna()
    print(f"[OK] Final dataset: {len(df)} samples")
    
    # 레이블 분포 출력
    print(f"\nLabel distribution ({num_classes} classes):")
    label_counts = df['label'].value_counts().sort_index()
    if num_classes == 2:
        label_names = {0: 'SELL ⬇️', 1: 'BUY ⬆️'}
    elif num_classes == 3:
        label_names = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
    else:
        label_names = {-2: 'STRONG_SELL', -1: 'SELL', 0: 'HOLD', 1: 'BUY', 2: 'STRONG_BUY'}
    for label, count in label_counts.items():
        print(f"   {label_names.get(int(label), label)}: {count} ({count/len(df)*100:.1f}%)")
    
    return df


def save_dataset(df: pd.DataFrame, filename: str):
    """데이터셋 저장"""
    # 디렉토리 생성
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=True)
    print(f"[SAVED] {filename}")


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare training data from candle data')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--timeframe', type=str, default='1h', help='Candle timeframe')
    parser.add_argument('--limit', type=int, default=10000, help='Max candles to load')
    parser.add_argument('--future', type=int, default=5, help='Future periods for labeling')
    parser.add_argument('--threshold', type=float, default=0.02, help='Price change threshold')
    parser.add_argument('--classes', type=int, default=2, choices=[2, 3, 5], help='Number of classes (2: BUY/SELL 횡보제거, 3: BUY/HOLD/SELL, 5: with STRONG)')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    
    args = parser.parse_args()
    
    df = await prepare_training_dataset(
        symbol=args.symbol,
        timeframe=args.timeframe,
        limit=args.limit,
        future_periods=args.future,
        threshold=args.threshold,
        num_classes=args.classes
    )
    
    if df.empty:
        print("[ERROR] No data to save")
        return
    
    # ai-model/data 폴더에 저장
    default_output = os.path.join(AI_MODEL_DIR, 'data', f'{args.symbol.lower()}_{args.timeframe}_training.csv')
    output_file = args.output or default_output
    save_dataset(df, output_file)


if __name__ == "__main__":
    asyncio.run(main())

