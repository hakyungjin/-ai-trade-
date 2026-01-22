"""
개선된 데이터 수집 및 저장 서비스
- DB에 없는 데이터만 선택적으로 수집
- 마지막 저장된 시간 이후의 데이터만 수집
- 효율적인 캐싱 및 증분 업데이트
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
import logging

from app.models.market_data import MarketCandle
from app.services.binance_service import BinanceService
from app.services.market_data_service import MarketDataService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IncrementalDataCollector:
    """증분 방식의 데이터 수집 서비스"""

    def __init__(self, db: AsyncSession, binance_service: BinanceService):
        self.db = db
        self.binance = binance_service
        self.market_service = MarketDataService(db)

    async def get_last_saved_time(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[datetime]:
        """
        DB에 저장된 마지막 캔들의 시간 조회

        Args:
            symbol: 거래쌍
            timeframe: 시간프레임

        Returns:
            마지막 저장 시간, 없으면 None
        """
        try:
            result = await self.db.execute(
                select(func.max(MarketCandle.open_time)).where(
                    MarketCandle.symbol == symbol,
                    MarketCandle.timeframe == timeframe
                )
            )

            last_time = result.scalar()
            return last_time

        except Exception as e:
            logger.error(f"Error getting last saved time: {e}")
            return None

    async def get_required_candles_count(
        self,
        last_saved_time: Optional[datetime],
        timeframe: str
    ) -> int:
        """
        마지막 저장 시간 이후의 필요한 캔들 개수 계산

        Args:
            last_saved_time: 마지막 저장 시간
            timeframe: 시간프레임

        Returns:
            필요한 캔들 개수
        """
        # 시간프레임을 분으로 변환
        timeframe_minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
            "1w": 10080,
        }

        minutes = timeframe_minutes.get(timeframe, 60)

        if last_saved_time is None:
            # 처음 수집하는 경우 30일치 데이터
            minutes_needed = 30 * 1440
        else:
            # 마지막 저장 시간 이후의 분 계산
            now = datetime.utcnow()
            minutes_needed = int((now - last_saved_time).total_seconds() / 60)

            # 최소 1개 캔들, 최대 1000개
            minutes_needed = max(minutes_needed, minutes)

        # 필요한 캔들 개수
        candles_needed = minutes_needed // minutes + 2  # 여유분

        # 바이낸스 API 제한 (한번에 최대 1000개)
        return min(candles_needed, 1000)

    async def collect_incremental_data(
        self,
        symbol: str,
        timeframe: str,
        force_full: bool = False
    ) -> Tuple[bool, int]:
        """
        증분 방식으로 데이터 수집

        Args:
            symbol: 거래쌍
            timeframe: 시간프레임
            force_full: 전체 재수집 여부 (True면 DB 무시하고 새로 수집)

        Returns:
            (성공 여부, 저장된 캔들 개수)
        """
        try:
            # 마지막 저장 시간 확인
            if force_full:
                last_saved_time = None
                logger.info(f"Force full collection for {symbol} {timeframe}")
            else:
                last_saved_time = await self.get_last_saved_time(symbol, timeframe)

            if last_saved_time:
                logger.info(f"Last saved time for {symbol} {timeframe}: {last_saved_time}")
            else:
                logger.info(f"No existing data for {symbol} {timeframe}, collecting from start")

            # 필요한 캔들 개수 계산
            candles_needed = await self.get_required_candles_count(last_saved_time, timeframe)
            logger.info(f"Need to collect {candles_needed} candles for {symbol} {timeframe}")

            # 바이낸스에서 캔들 데이터 가져오기
            klines = await self.binance.get_klines(
                symbol=symbol,
                interval=timeframe,
                limit=candles_needed
            )

            if not klines:
                logger.warning(f"No klines returned for {symbol} {timeframe}")
                return False, 0

            # 필터링: 마지막 저장 시간 이후의 데이터만
            new_klines = klines
            if last_saved_time and not force_full:
                # 이미 저장된 데이터를 제외
                last_timestamp_ms = int(last_saved_time.timestamp() * 1000)
                new_klines = [
                    k for k in klines
                    if k["timestamp"] > last_timestamp_ms
                ]
                logger.info(f"Filtered to {len(new_klines)} new candles out of {len(klines)}")

            if not new_klines:
                logger.info(f"No new candles for {symbol} {timeframe}")
                return True, 0

            # 캔들 데이터 저장
            saved_count = await self.market_service.save_candles(
                symbol=symbol,
                timeframe=timeframe,
                candles=new_klines
            )

            # 기술적 지표 계산 및 저장
            await self.market_service.calculate_and_save_indicators(
                symbol=symbol,
                timeframe=timeframe,
                candles=klines  # 전체 데이터로 지표 계산
            )

            logger.info(f"Successfully saved {saved_count} candles for {symbol} {timeframe}")
            return True, saved_count

        except Exception as e:
            logger.error(f"Error in incremental data collection: {e}")
            return False, 0

    async def collect_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        force_full: bool = False
    ) -> Dict[str, int]:
        """
        여러 심볼의 데이터를 병렬로 수집

        Args:
            symbols: 거래쌍 리스트
            timeframe: 시간프레임
            force_full: 전체 재수집 여부

        Returns:
            {symbol: saved_count} 딕셔너리
        """
        results = {}

        for symbol in symbols:
            try:
                success, saved_count = await self.collect_incremental_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    force_full=force_full
                )

                results[symbol] = saved_count if success else -1

                # API 레이트 리밋 방지
                import asyncio
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error collecting {symbol}: {e}")
                results[symbol] = -1

        return results

    async def sync_all_data(
        self,
        symbols: List[str],
        timeframes: List[str] = ["1h", "4h", "1d"]
    ) -> Dict[str, Dict[str, int]]:
        """
        모든 심볼의 모든 타임프레임 데이터 동기화

        Args:
            symbols: 거래쌍 리스트
            timeframes: 시간프레임 리스트

        Returns:
            {symbol: {timeframe: saved_count}} 구조의 딕셔너리
        """
        results = {}

        for symbol in symbols:
            results[symbol] = {}

            for timeframe in timeframes:
                try:
                    success, saved_count = await self.collect_incremental_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        force_full=False
                    )

                    results[symbol][timeframe] = saved_count if success else -1

                    import asyncio
                    await asyncio.sleep(0.5)

                except Exception as e:
                    logger.error(f"Error syncing {symbol} {timeframe}: {e}")
                    results[symbol][timeframe] = -1

        return results

    async def get_data_coverage(
        self,
        symbol: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """
        심볼의 데이터 커버리지 정보 조회

        Args:
            symbol: 거래쌍
            timeframe: 시간프레임

        Returns:
            커버리지 정보 (최초, 최신, 개수 등)
        """
        try:
            # 첫 번째 캔들
            first_result = await self.db.execute(
                select(MarketCandle)
                .where(
                    MarketCandle.symbol == symbol,
                    MarketCandle.timeframe == timeframe
                )
                .order_by(MarketCandle.open_time)
                .limit(1)
            )
            first_candle = first_result.scalar_one_or_none()

            # 마지막 캔들
            last_result = await self.db.execute(
                select(MarketCandle)
                .where(
                    MarketCandle.symbol == symbol,
                    MarketCandle.timeframe == timeframe
                )
                .order_by(desc(MarketCandle.open_time))
                .limit(1)
            )
            last_candle = last_result.scalar_one_or_none()

            # 캔들 개수
            count_result = await self.db.execute(
                select(func.count(MarketCandle.id)).where(
                    MarketCandle.symbol == symbol,
                    MarketCandle.timeframe == timeframe
                )
            )
            count = count_result.scalar() or 0

            if first_candle and last_candle:
                # 예상 개수 계산
                timeframe_minutes = {
                    "1m": 1,
                    "5m": 5,
                    "15m": 15,
                    "30m": 30,
                    "1h": 60,
                    "4h": 240,
                    "1d": 1440,
                }

                minutes = timeframe_minutes.get(timeframe, 60)
                duration_minutes = int((last_candle.open_time - first_candle.open_time).total_seconds() / 60)
                expected_count = (duration_minutes // minutes) + 1

                coverage_percent = (count / expected_count * 100) if expected_count > 0 else 0

                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "first_time": first_candle.open_time.isoformat(),
                    "last_time": last_candle.open_time.isoformat(),
                    "total_candles": count,
                    "expected_candles": expected_count,
                    "coverage_percent": coverage_percent,
                    "gap_hours": duration_minutes // 60,
                }
            else:
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "first_time": None,
                    "last_time": None,
                    "total_candles": 0,
                    "expected_candles": 0,
                    "coverage_percent": 0.0,
                    "gap_hours": 0,
                }

        except Exception as e:
            logger.error(f"Error getting data coverage: {e}")
            return {}


# 편의 함수들

async def quick_sync(
    db: AsyncSession,
    binance_service: BinanceService,
    symbols: List[str] = None,
    timeframes: List[str] = ["1h"]
) -> Dict[str, Any]:
    """
    빠른 데이터 동기화 함수

    사용 예:
    await quick_sync(db, binance_service, ["BTCUSDT", "ETHUSDT"])
    """
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT"]

    collector = IncrementalDataCollector(db, binance_service)
    return await collector.sync_all_data(symbols, timeframes)


async def check_coverage(
    db: AsyncSession,
    binance_service: BinanceService,
    symbol: str,
    timeframe: str = "1h"
) -> Dict[str, Any]:
    """데이터 커버리지 확인"""
    collector = IncrementalDataCollector(db, binance_service)
    return await collector.get_data_coverage(symbol, timeframe)
