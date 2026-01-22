"""
데이터 통합 서비스
- DB에서 캐시된 데이터 먼저 조회
- DB에 없는 부분은 자동으로 증분 수집
- 항상 최신 데이터 보장
"""

from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from datetime import datetime
import logging

from app.models.market_data import MarketCandle
from app.services.binance_service import BinanceService
from app.services.incremental_collector import IncrementalDataCollector

logger = logging.getLogger(__name__)


class UnifiedDataService:
    """DB 캐시 + 증분 수집을 통합하는 서비스"""

    def __init__(self, db: AsyncSession, binance_service: BinanceService):
        self.db = db
        self.binance = binance_service
        self.collector = IncrementalDataCollector(db, binance_service)

    async def get_klines_with_cache(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        캐시 + 증분 수집을 이용한 캔들 데이터 조회

        우선순위:
        1. DB에서 필요한 데이터 확인
        2. 부족하면 Binance에서 증분 수집
        3. DB + 새 데이터 조합하여 반환

        Args:
            symbol: 거래쌍
            timeframe: 시간프레임
            limit: 필요한 캔들 개수

        Returns:
            캔들 데이터 리스트 (시간순)
        """
        try:
            # 1단계: DB에 저장된 마지막 캔들 시간 확인
            last_saved_time = await self.collector.get_last_saved_time(symbol, timeframe)

            # 2단계: 필요한 데이터 개수 계산
            candles_needed = await self.collector.get_required_candles_count(last_saved_time, timeframe)

            # 3단계: Binance에서 데이터 수집 (증분)
            logger.info(f"Collecting {candles_needed} candles for {symbol} {timeframe}")
            try:
                binance_klines = await self.binance.get_klines(
                    symbol=symbol,
                    interval=timeframe,
                    limit=candles_needed
                )
            except Exception as e:
                logger.error(f"Error fetching from Binance: {e}")
                binance_klines = []

            # 4단계: 신규 데이터 필터링 및 DB 저장
            if binance_klines and last_saved_time:
                last_timestamp_ms = int(last_saved_time.timestamp() * 1000)
                new_klines = [k for k in binance_klines if k["timestamp"] > last_timestamp_ms]

                if new_klines:
                    try:
                        saved_count = await self._save_klines(symbol, timeframe, new_klines)
                        logger.info(f"Saved {saved_count} new candles for {symbol} {timeframe}")
                    except Exception as e:
                        logger.error(f"Error saving new klines: {e}")

            # 5단계: DB에서 최근 데이터 조회
            cached_candles = await self._get_candles_from_db(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )

            logger.info(f"Returned {len(cached_candles)} candles for {symbol} {timeframe}")
            return cached_candles

        except Exception as e:
            logger.error(f"Error in get_klines_with_cache: {e}")
            # Fallback: Binance에서 직접 조회
            return await self.binance.get_klines(
                symbol=symbol,
                interval=timeframe,
                limit=limit
            )

    async def _get_candles_from_db(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        DB에서 캔들 데이터 조회

        Args:
            symbol: 거래쌍
            timeframe: 시간프레임
            limit: 조회할 캔들 개수

        Returns:
            캔들 데이터 리스트
        """
        try:
            result = await self.db.execute(
                select(MarketCandle)
                .where(
                    MarketCandle.symbol == symbol,
                    MarketCandle.timeframe == timeframe
                )
                .order_by(desc(MarketCandle.open_time))
                .limit(limit)
            )

            candles = result.scalars().all()
            candles.reverse()  # 시간순으로 정렬

            return [
                {
                    "timestamp": int(c.open_time.timestamp() * 1000),
                    "open_time": c.open_time.isoformat(),
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                    "close_time": int(c.close_time.timestamp() * 1000) if c.close_time else c.open_time.timestamp() * 1000,
                    "quote_volume": c.quote_volume or 0,
                    "trades": c.trades_count or 0
                }
                for c in candles
            ]

        except Exception as e:
            logger.error(f"Error retrieving candles from DB: {e}")
            return []

    async def _save_klines(
        self,
        symbol: str,
        timeframe: str,
        klines: List[Dict[str, Any]]
    ) -> int:
        """
        캔들 데이터 DB 저장

        Args:
            symbol: 거래쌍
            timeframe: 시간프레임
            klines: 캔들 데이터

        Returns:
            저장된 캔들 개수
        """
        saved_count = 0

        try:
            for kline in klines:
                try:
                    # timestamp를 datetime으로 변환
                    open_time = datetime.utcfromtimestamp(kline["timestamp"] / 1000)
                    close_time = datetime.utcfromtimestamp(kline.get("close_time", kline["timestamp"]) / 1000)

                    # 이미 존재하는지 확인
                    existing = await self.db.execute(
                        select(MarketCandle).where(
                            MarketCandle.symbol == symbol,
                            MarketCandle.timeframe == timeframe,
                            MarketCandle.open_time == open_time
                        )
                    )

                    if existing.scalar_one_or_none():
                        continue

                    # 새 캔들 생성
                    market_candle = MarketCandle(
                        symbol=symbol,
                        timeframe=timeframe,
                        open_time=open_time,
                        open=float(kline["open"]),
                        high=float(kline["high"]),
                        low=float(kline["low"]),
                        close=float(kline["close"]),
                        volume=float(kline["volume"]),
                        close_time=close_time,
                        quote_volume=float(kline.get("quote_volume", 0)),
                        trades_count=int(kline.get("trades", 0)),
                    )

                    self.db.add(market_candle)
                    saved_count += 1

                except Exception as e:
                    logger.error(f"Error processing kline: {e}")
                    continue

            if saved_count > 0:
                await self.db.commit()

            return saved_count

        except Exception as e:
            logger.error(f"Error saving klines: {e}")
            await self.db.rollback()
            return 0

    async def get_market_data_for_analysis(
        self,
        symbol: str,
        timeframe: str = "1h",
        analysis_candles: int = 100,
        lookback_candles: int = 50
    ) -> Dict[str, Any]:
        """
        AI 분석용 시장 데이터 통합 제공

        Args:
            symbol: 거래쌍
            timeframe: 시간프레임
            analysis_candles: 분석에 사용할 캔들 개수
            lookback_candles: 기술적 지표 계산용 과거 캔들 개수

        Returns:
            {
                "current_price": float,
                "candles": [...],
                "basic_stats": {...}
            }
        """
        try:
            # 캔들 데이터 조회 (DB 캐시 활용)
            total_needed = max(analysis_candles, lookback_candles) + 50

            candles = await self.get_klines_with_cache(
                symbol=symbol,
                timeframe=timeframe,
                limit=total_needed
            )

            if not candles:
                logger.warning(f"No candles available for {symbol} {timeframe}")
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "current_price": 0,
                    "candles": [],
                    "error": "No data available"
                }

            # 현재가
            current_price = candles[-1]["close"]

            # 기본 통계
            closes = [c["close"] for c in candles[-lookback_candles:]]
            volumes = [c["volume"] for c in candles[-lookback_candles:]]

            import statistics
            avg_volume = statistics.mean(volumes) if volumes else 0
            volatility = statistics.stdev(closes) / statistics.mean(closes) * 100 if len(closes) > 1 else 0

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.utcnow().isoformat(),
                "current_price": current_price,
                "candles": candles[-analysis_candles:],  # 최근 데이터만
                "basic_stats": {
                    "high_50": max(c["high"] for c in candles[-50:]) if len(candles) >= 50 else current_price,
                    "low_50": min(c["low"] for c in candles[-50:]) if len(candles) >= 50 else current_price,
                    "avg_volume": avg_volume,
                    "volatility": volatility,
                    "total_cached_candles": len(candles),
                }
            }

        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "error": str(e)
            }


# 편의 함수들

async def get_analysis_candles(
    db: AsyncSession,
    binance_service: BinanceService,
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    분석용 캔들 데이터 조회 (DB 캐시 활용)

    사용 예:
    candles = await get_analysis_candles(db, binance, "BTCUSDT", "1h", 100)
    """
    service = UnifiedDataService(db, binance_service)
    return await service.get_klines_with_cache(symbol, timeframe, limit)


async def get_all_market_data(
    db: AsyncSession,
    binance_service: BinanceService,
    symbol: str,
    timeframe: str = "1h"
) -> Dict[str, Any]:
    """AI 분석용 모든 시장 데이터 조회"""
    service = UnifiedDataService(db, binance_service)
    return await service.get_market_data_for_analysis(symbol, timeframe)
