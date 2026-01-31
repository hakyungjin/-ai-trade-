"""
Prepare Unified Dataset for Multi-Asset Model Training
여러 자산(코인, 주식)의 데이터를 하나의 통합 데이터셋으로 병합

Features:
- 여러 심볼의 OHLCV 데이터 병합
- 로그 수익률 계산
- 시장 휴장 시간 처리 (나스닥 등)
- 통합 피처 엔지니어링 적용
- Asset ID 매핑
"""

import os
import sys
import asyncio
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import AsyncSessionLocal
from app.models.market_data import MarketCandle
from app.services.asset_mapping_service import AssetMappingService
from app.services.unified_feature_engineering import compute_all_features, create_labels
from app.services.technical_indicators import TechnicalIndicators
from sqlalchemy import select, and_
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def fetch_candles_from_db(
    symbol: str,
    timeframe: str,
    limit: int = 10000
) -> pd.DataFrame:
    """DB에서 캔들 데이터 가져오기"""
    async with AsyncSessionLocal() as db:
        stmt = (
            select(MarketCandle)
            .where(
                and_(
                    MarketCandle.symbol == symbol,
                    MarketCandle.timeframe == timeframe
                )
            )
            .order_by(MarketCandle.open_time.desc())
            .limit(limit)
        )

        result = await db.execute(stmt)
        candles = result.scalars().all()

        if not candles:
            logger.warning(f"No candles found for {symbol} {timeframe}")
            return pd.DataFrame()

        # DataFrame 변환
        data = []
        for c in candles:
            data.append({
                'timestamp': c.open_time,
                'open': c.open,
                'high': c.high,
                'low': c.low,
                'close': c.close,
                'volume': c.volume,
                'market_type': c.market_type if hasattr(c, 'market_type') else 'crypto',
                'is_market_open': c.is_market_open if hasattr(c, 'is_market_open') else 1,
            })

        df = pd.DataFrame(data)
        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"✅ Fetched {len(df)} candles for {symbol} {timeframe}")
        return df


def preprocess_market_gaps(df: pd.DataFrame, market_type: str) -> pd.DataFrame:
    """시장 휴장 시간 처리 (Forward Fill)"""
    if market_type == 'nasdaq':
        # 나스닥: 주말과 밤시간 데이터 Forward Fill
        logger.info(f"📊 Processing market gaps for nasdaq...")

        # is_market_open == 0인 행의 가격을 직전 값으로 채움
        df['close'] = df['close'].ffill()
        df['open'] = df['open'].ffill()
        df['high'] = df['high'].ffill()
        df['low'] = df['low'].ffill()

        # 거래량은 0으로
        df.loc[df['is_market_open'] == 0, 'volume'] = 0

        logger.info(f"✅ Market gaps processed")

    return df


def calculate_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """로그 수익률 계산"""
    # 로그 수익률: r_t = ln(P_t / P_{t-1})
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # 첫 행은 NaN이므로 제거하거나 0으로 채움
    df['log_return'] = df['log_return'].fillna(0)

    return df


async def prepare_unified_dataset(
    symbols: List[str],
    timeframe: str = "1h",
    limit_per_symbol: int = 10000,
    output_path: Optional[str] = None,
    threshold: float = 0.02,
    lookahead: int = 5,
    num_classes: int = 3
) -> pd.DataFrame:
    """통합 데이터셋 생성"""
    logger.info(f"🚀 Starting unified dataset preparation for {len(symbols)} assets...")

    all_data = []

    async with AsyncSessionLocal() as db:
        # Asset ID 초기화 (기본 매핑 생성)
        await AssetMappingService.initialize_default_mappings(db)

        for symbol in symbols:
            logger.info(f"\n📊 Processing {symbol}...")

            # 1. DB에서 데이터 가져오기
            df = await fetch_candles_from_db(symbol, timeframe, limit_per_symbol)
            if df.empty:
                logger.warning(f"⚠️ No data for {symbol}, skipping...")
                continue

            # 2. Asset ID 조회
            asset_id = await AssetMappingService.get_asset_id(db, symbol, create_if_missing=True)
            if asset_id is None:
                logger.error(f"❌ Failed to get asset_id for {symbol}")
                continue

            df['symbol'] = symbol
            df['asset_id'] = asset_id

            # 3. 시장 휴장 처리
            market_type = df['market_type'].iloc[0] if 'market_type' in df.columns else 'crypto'
            df = preprocess_market_gaps(df, market_type)

            # 4. 로그 수익률 계산
            df = calculate_log_returns(df)

            # 5. 기술적 지표 계산
            logger.info(f"📈 Calculating technical indicators...")
            df = TechnicalIndicators.calculate_all_indicators(df)

            # 6. 통합 피처 엔지니어링
            logger.info(f"🔧 Computing unified features...")
            df = compute_all_features(df)

            # 7. 레이블 생성
            logger.info(f"🏷️ Creating labels (threshold={threshold}, lookahead={lookahead})...")
            df = create_labels(df, threshold=threshold, lookahead=lookahead, num_classes=num_classes)

            # 8. NaN 제거
            df = df.dropna()

            logger.info(f"✅ {symbol}: {len(df)} samples prepared")
            all_data.append(df)

    # 모든 데이터 병합
    if not all_data:
        logger.error("❌ No data to merge!")
        return pd.DataFrame()

    unified_df = pd.concat(all_data, ignore_index=True)

    # 시간순 정렬 (중요!)
    unified_df = unified_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

    logger.info(f"\n✅ Unified dataset created:")
    logger.info(f"   - Total samples: {len(unified_df)}")
    logger.info(f"   - Assets: {unified_df['symbol'].nunique()}")
    logger.info(f"   - Features: {len(unified_df.columns)}")
    logger.info(f"   - Label distribution:\n{unified_df['label'].value_counts()}")

    # 저장
    if output_path:
        unified_df.to_csv(output_path, index=False)
        logger.info(f"💾 Saved to {output_path}")

    return unified_df


async def main():
    parser = argparse.ArgumentParser(description="Prepare unified dataset for multi-asset model")
    parser.add_argument('--symbols', nargs='+', required=True, help='List of symbols (e.g., BTCUSDT ETHUSDT AAPL)')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe (default: 1h)')
    parser.add_argument('--limit', type=int, default=10000, help='Max candles per symbol (default: 10000)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV path')
    parser.add_argument('--threshold', type=float, default=0.02, help='Label threshold (default: 0.02)')
    parser.add_argument('--lookahead', type=int, default=5, help='Lookahead periods (default: 5)')
    parser.add_argument('--classes', type=int, default=3, choices=[2, 3, 5], help='Number of classes (default: 3)')

    args = parser.parse_args()

    # 출력 경로 설정
    if args.output is None:
        data_dir = PROJECT_ROOT / "ai-model" / "data"
        data_dir.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = str(data_dir / f"unified_dataset_{args.timeframe}_{timestamp}.csv")

    # 데이터셋 생성
    df = await prepare_unified_dataset(
        symbols=args.symbols,
        timeframe=args.timeframe,
        limit_per_symbol=args.limit,
        output_path=args.output,
        threshold=args.threshold,
        lookahead=args.lookahead,
        num_classes=args.classes
    )

    if not df.empty:
        logger.info(f"\n✅ SUCCESS! Unified dataset ready for training.")
        logger.info(f"   File: {args.output}")
        logger.info(f"   Samples: {len(df)}")
        logger.info(f"   Assets: {df['symbol'].nunique()}")
    else:
        logger.error("❌ Failed to create dataset")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
