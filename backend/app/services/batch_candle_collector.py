"""
배치 캔들 수집 서비스
- 모든 모니터링 코인의 캔들을 주기적으로 수집
- 벡터 DB 학습 데이터 축적용
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .binance_service import BinanceService
from ..models.market_data import MarketCandle
from ..database import AsyncSessionLocal

logger = logging.getLogger(__name__)


class BatchCandleCollector:
    """배치 캔들 수집 서비스"""

    def __init__(self, binance_service: BinanceService):
        self.binance = binance_service
        self.is_running = False
        self.active_symbols: List[str] = []

    async def get_active_symbols(self, db: AsyncSession) -> List[str]:
        """활성 심볼 목록 조회"""
        try:
            # market_candles 테이블에서 고유 심볼 조회
            stmt = select(MarketCandle.symbol).distinct()
            result = await db.execute(stmt)
            symbols = result.scalars().all()
            return list(set(symbols)) if symbols else []
        except Exception as e:
            logger.error(f"❌ Error getting active symbols: {e}")
            return []

    async def collect_candles_for_symbol(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500,
        db: Optional[AsyncSession] = None
    ) -> int:
        """
        단일 심볼의 캔들 수집 및 저장

        Args:
            symbol: 거래 쌍 (예: BTCUSDT)
            interval: 캔들 주기 (기본값: 1h)
            limit: 수집할 캔들 개수 (기본값: 500)
            db: 데이터베이스 세션

        Returns:
            저장된 캔들 개수
        """
        try:
            logger.info(f"📊 Collecting candles for {symbol} ({interval})...")

            # 바이낸스에서 캔들 수집
            candles = await self.binance.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )

            if not candles:
                logger.warning(f"⚠️ No candles returned for {symbol}")
                return 0

            logger.info(f"✅ Fetched {len(candles)} candles for {symbol}")

            # 데이터베이스 세션 없으면 생성
            if db is None:
                db = AsyncSessionLocal()

            # 캔들 데이터 저장
            saved_count = 0
            for candle in candles:
                try:
                    # 중복 체크
                    stmt = select(MarketCandle).where(
                        MarketCandle.symbol == symbol,
                        MarketCandle.timeframe == interval,
                        MarketCandle.open_time == candle.get("timestamp")
                    )
                    existing = await db.execute(stmt)
                    if existing.scalar():
                        continue  # 이미 존재하면 스킵

                    # 새로운 마켓 캔들 생성
                    market_candle = MarketCandle(
                        symbol=symbol,
                        timeframe=interval,
                        open_time=candle.get("timestamp"),
                        open=float(candle.get("open", 0)),
                        high=float(candle.get("high", 0)),
                        low=float(candle.get("low", 0)),
                        close=float(candle.get("close", 0)),
                        volume=float(candle.get("volume", 0)),
                        quote_volume=float(candle.get("quote_volume", 0)),
                        trades_count=int(candle.get("trades", 0))
                    )

                    db.add(market_candle)
                    saved_count += 1

                except Exception as e:
                    logger.error(f"❌ Error saving candle for {symbol}: {e}")
                    continue

            # 배치 커밋
            if saved_count > 0:
                await db.commit()
                logger.info(f"💾 Saved {saved_count} new candles for {symbol}")
            else:
                logger.info(f"ℹ️ No new candles to save for {symbol}")

            return saved_count

        except Exception as e:
            logger.error(f"❌ Error collecting candles for {symbol}: {e}")
            return 0

    async def collect_all_symbols(
        self,
        interval: str = "1h",
        limit: int = 500
    ) -> dict:
        """
        모든 활성 심볼의 캔들 수집

        Args:
            interval: 캔들 주기
            limit: 각 심볼당 수집 캔들 개수

        Returns:
            수집 결과 {'symbol': saved_count}
        """
        try:
            async with AsyncSessionLocal() as db:
                # 활성 심볼 조회
                symbols = await self.get_active_symbols(db)
                
                if not symbols:
                    logger.warning("⚠️ No active symbols to collect")
                    return {}

                logger.info(f"🔄 Starting batch collection for {len(symbols)} symbols")

                results = {}
                for symbol in symbols:
                    count = await self.collect_candles_for_symbol(
                        symbol=symbol,
                        interval=interval,
                        limit=limit,
                        db=db
                    )
                    results[symbol] = count

                    # 레이트 리미팅: 심볼당 1초 대기
                    await asyncio.sleep(1)

                logger.info(f"✅ Batch collection complete: {results}")
                return results

        except Exception as e:
            logger.error(f"❌ Error in batch collection: {e}")
            return {}

    async def start_periodic_collection(
        self,
        interval: str = "1h",
        limit: int = 500,
        collect_interval_hours: int = 1
    ):
        """
        주기적 캔들 수집 시작

        Args:
            interval: 캔들 주기 (1m, 5m, 15m, 1h 등)
            limit: 각 심볼당 수집 캔들 개수
            collect_interval_hours: 수집 주기 (시간)
        """
        self.is_running = True
        logger.info(
            f"🚀 Starting periodic candle collection "
            f"(interval: {interval}, every {collect_interval_hours}h)"
        )

        while self.is_running:
            try:
                await self.collect_all_symbols(interval=interval, limit=limit)
                
                # 다음 수집까지 대기
                wait_seconds = collect_interval_hours * 3600
                logger.info(f"⏳ Next collection in {collect_interval_hours}h")
                await asyncio.sleep(wait_seconds)

            except Exception as e:
                logger.error(f"❌ Error in periodic collection: {e}")
                # 에러 발생 시 30초 후 재시도
                await asyncio.sleep(30)

    def stop_periodic_collection(self):
        """주기적 수집 중지"""
        self.is_running = False
        logger.info("⏹️ Periodic collection stopped")

    async def collect_daily_snapshot(self) -> dict:
        """
        일일 스냅샷 수집 (모든 심볼의 1일 봉 데이터)

        Returns:
            수집 결과
        """
        logger.info("📅 Starting daily snapshot collection...")
        return await self.collect_all_symbols(interval="1d", limit=365)

    async def collect_historical_data(
        self,
        symbol: str,
        interval: str = "1h",
        days: int = 365
    ) -> int:
        """
        특정 심볼의 과거 데이터 수집

        Args:
            symbol: 거래 쌍
            interval: 캔들 주기
            days: 과거 몇 일 데이터를 수집할지

        Returns:
            저장된 캔들 개수
        """
        try:
            logger.info(f"📈 Collecting {days} days of historical data for {symbol}...")

            # 최대 수집 가능한 캔들 개수 계산
            if interval == "1m":
                limit = min(days * 24 * 60, 1000)
            elif interval == "5m":
                limit = min(days * 24 * 12, 1000)
            elif interval == "15m":
                limit = min(days * 24 * 4, 1000)
            elif interval == "1h":
                limit = min(days * 24, 1000)
            elif interval == "4h":
                limit = min(days * 6, 1000)
            elif interval == "1d":
                limit = min(days, 1000)
            else:
                limit = 1000

            logger.info(f"Collecting {limit} candles with {interval} interval...")
            return await self.collect_candles_for_symbol(symbol, interval, limit)

        except Exception as e:
            logger.error(f"❌ Error collecting historical data for {symbol}: {e}")
            return 0
