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


def create_labels(df: pd.DataFrame, future_periods: int = 5, threshold: float = 0.02, num_classes: int = 3) -> pd.DataFrame:
    """
    레이블 생성 (미래 가격 변화 기반)
    
    Args:
        future_periods: 몇 캔들 후 가격을 볼 것인지
        threshold: 상승/하락 판단 기준 (2% = 0.02)
        num_classes: 클래스 수 (3 또는 5)
    
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
    
    if num_classes == 3:
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
    """학습용 피처 생성 (확장 버전)"""
    df = df.copy()
    
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
    
    # ===== 거래량 관련 피처 =====
    df['volume_change_1'] = df['volume'].pct_change(1)
    df['volume_change_5'] = df['volume'].pct_change(5)
    df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_ma_ratio_5'] = df['volume'] / df['volume'].rolling(5).mean()
    
    # 거래량 급증 감지 (2배 이상)
    df['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
    
    # ===== 변동성 피처 =====
    df['volatility_5'] = df['close'].rolling(5).std() / df['close'].rolling(5).mean()
    df['volatility_20'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
    
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
    if num_classes == 3:
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
    parser.add_argument('--classes', type=int, default=3, choices=[3, 5], help='Number of classes (3: BUY/HOLD/SELL, 5: with STRONG)')
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

