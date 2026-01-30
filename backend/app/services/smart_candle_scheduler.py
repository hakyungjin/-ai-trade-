"""
스마트 캔들 스케줄러
- 모니터링 코인 기준으로 타임프레임별 최적 주기로 캔들 수집
- 분석 시 API 호출 없이 DB에서 바로 조회 가능하도록 데이터 미리 준비
- 서버 부담 최소화
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.services.binance_service import BinanceService
from app.services.binance_futures_service import BinanceFuturesService, get_futures_service
from app.models.market_data import MarketCandle
from app.models.coin import Coin
from app.database import AsyncSessionLocal

logger = logging.getLogger(__name__)


# 타임프레임별 수집 주기 (초)
TIMEFRAME_INTERVALS = {
    "1m": 60,           # 1분마다
    "5m": 300,          # 5분마다
    "15m": 900,         # 15분마다
    "30m": 1800,        # 30분마다
    "1h": 3600,         # 1시간마다
    "4h": 14400,        # 4시간마다
    "1d": 86400,        # 1일마다
}


class SmartCandleScheduler:
    """
    스마트 캔들 스케줄러
    - 모니터링 중인 코인만 수집
    - 타임프레임별 적절한 주기로 수집
    - 최소한의 API 호출로 DB 최신화
    """

    def __init__(self, binance_service: BinanceService):
        self.binance = binance_service
        self.futures_service: Optional[BinanceFuturesService] = None
        self.is_running = False
        self._tasks: Dict[str, asyncio.Task] = {}
        self._last_collection_time: Dict[str, datetime] = {}
        
    async def get_monitoring_coins(self) -> List[Dict[str, Any]]:
        """모니터링 중인 코인 목록 조회"""
        async with AsyncSessionLocal() as db:
            stmt = select(Coin).where(
                and_(
                    Coin.is_active == True,
                    Coin.is_monitoring == True
                )
            )
            result = await db.execute(stmt)
            coins = result.scalars().all()
            
            return [
                {
                    "symbol": coin.symbol,
                    "market_type": coin.market_type,
                    "timeframes": coin.monitoring_timeframes or ["1h"],
                    "id": coin.id
                }
                for coin in coins
            ]
    
    async def collect_single_candle(
        self,
        symbol: str,
        timeframe: str,
        market_type: str = "spot",
        limit: int = 5  # 최근 5개만 - 최신 데이터 유지용
    ) -> int:
        """
        단일 심볼의 최신 캔들 수집 (최소한의 API 호출)
        
        Args:
            symbol: 심볼
            timeframe: 타임프레임
            market_type: spot 또는 futures
            limit: 가져올 캔들 개수 (적을수록 빠름)
        
        Returns:
            저장된 캔들 개수
        """
        try:
            # 마켓 타입에 따라 적절한 서비스 사용
            if market_type == "futures":
                if self.futures_service is None:
                    self.futures_service = get_futures_service()
                klines = await self.futures_service.get_futures_klines(
                    symbol=symbol,
                    interval=timeframe,
                    limit=limit
                )
            else:
                klines = await self.binance.get_klines(
                    symbol=symbol,
                    interval=timeframe,
                    limit=limit
                )
            
            if not klines:
                return 0
            
            async with AsyncSessionLocal() as db:
                saved_count = 0
                
                for kline in klines:
                    try:
                        # timestamp 처리
                        ts = kline.get("timestamp")
                        if isinstance(ts, str):
                            open_time = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        else:
                            open_time = datetime.utcfromtimestamp(ts / 1000)
                        
                        # 중복 체크
                        existing = await db.execute(
                            select(MarketCandle).where(
                                MarketCandle.symbol == symbol,
                                MarketCandle.timeframe == timeframe,
                                MarketCandle.open_time == open_time
                            )
                        )
                        
                        if existing.scalar_one_or_none():
                            # 기존 캔들 업데이트 (최신 close 가격 반영)
                            continue
                        
                        # 새 캔들 생성
                        close_time_val = kline.get("close_time", ts)
                        if isinstance(close_time_val, str):
                            close_time = datetime.fromisoformat(close_time_val.replace('Z', '+00:00'))
                        else:
                            close_time = datetime.utcfromtimestamp(close_time_val / 1000)
                        
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
                        
                        db.add(market_candle)
                        saved_count += 1
                        
                    except Exception as e:
                        logger.debug(f"Skipping candle: {e}")
                        continue
                
                if saved_count > 0:
                    await db.commit()
                    logger.debug(f"💾 Saved {saved_count} candles for {symbol} ({timeframe})")
                
                return saved_count
                
        except Exception as e:
            logger.error(f"❌ Error collecting candle for {symbol} ({timeframe}): {e}")
            return 0
    
    async def _timeframe_collector(self, timeframe: str):
        """
        특정 타임프레임의 캔들 수집 루프
        
        타임프레임 종료 시점에 맞춰 수집하여 완성된 캔들만 저장
        """
        interval_seconds = TIMEFRAME_INTERVALS.get(timeframe, 3600)
        
        logger.info(f"🚀 Starting collector for {timeframe} (interval: {interval_seconds}s)")
        
        while self.is_running:
            try:
                # 모니터링 코인 목록 조회
                coins = await self.get_monitoring_coins()
                
                # 해당 타임프레임을 사용하는 코인만 필터링
                target_coins = [
                    c for c in coins 
                    if timeframe in c.get("timeframes", ["1h"])
                ]
                
                if target_coins:
                    logger.info(f"📊 [{timeframe}] Collecting for {len(target_coins)} coins...")
                    
                    for coin in target_coins:
                        await self.collect_single_candle(
                            symbol=coin["symbol"],
                            timeframe=timeframe,
                            market_type=coin["market_type"],
                            limit=3  # 최근 3개만 - 최신 데이터 + 약간의 여유
                        )
                        # API 레이트 리밋 방지
                        await asyncio.sleep(0.2)
                    
                    logger.info(f"✅ [{timeframe}] Collection completed")
                
                # 다음 수집까지 대기
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                logger.info(f"⏹️ Collector for {timeframe} cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Error in {timeframe} collector: {e}")
                await asyncio.sleep(30)  # 에러 시 30초 후 재시도
    
    async def start(self, timeframes: List[str] = None):
        """
        스케줄러 시작
        
        Args:
            timeframes: 수집할 타임프레임 목록 (기본: 1h, 4h)
        """
        if self.is_running:
            logger.warning("⚠️ Scheduler already running")
            return
        
        self.is_running = True
        
        if timeframes is None:
            # 기본값: 주요 타임프레임만 수집
            timeframes = ["1h", "4h"]
        
        logger.info(f"🚀 Starting Smart Candle Scheduler for timeframes: {timeframes}")
        
        # 각 타임프레임별 수집 태스크 시작
        for tf in timeframes:
            if tf in TIMEFRAME_INTERVALS:
                task = asyncio.create_task(self._timeframe_collector(tf))
                self._tasks[tf] = task
                logger.info(f"📌 Started collector for {tf}")
        
        # 초기 데이터 수집 (스케줄러 시작 시 한 번 전체 수집)
        await self._initial_collection(timeframes)
    
    async def _initial_collection(self, timeframes: List[str]):
        """
        초기 데이터 수집 - 히스토리 데이터 확보
        """
        try:
            coins = await self.get_monitoring_coins()
            
            if not coins:
                logger.info("ℹ️ No monitoring coins found for initial collection")
                return
            
            logger.info(f"📥 Initial collection for {len(coins)} coins...")
            
            for coin in coins:
                coin_timeframes = coin.get("timeframes", ["1h"])
                
                for tf in coin_timeframes:
                    if tf in timeframes:
                        # 초기 수집은 더 많은 캔들 (분석에 충분한 양)
                        await self.collect_single_candle(
                            symbol=coin["symbol"],
                            timeframe=tf,
                            market_type=coin["market_type"],
                            limit=100  # 초기엔 100개 수집
                        )
                        await asyncio.sleep(0.3)
            
            logger.info("✅ Initial collection completed")
            
        except Exception as e:
            logger.error(f"❌ Error in initial collection: {e}")
    
    async def stop(self):
        """스케줄러 중지"""
        self.is_running = False
        
        # 모든 태스크 취소
        for tf, task in self._tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._tasks.clear()
        logger.info("⏹️ Smart Candle Scheduler stopped")
    
    async def add_timeframe(self, timeframe: str):
        """새 타임프레임 수집 추가"""
        if timeframe in self._tasks:
            logger.warning(f"⚠️ Timeframe {timeframe} already being collected")
            return
        
        if timeframe not in TIMEFRAME_INTERVALS:
            logger.error(f"❌ Invalid timeframe: {timeframe}")
            return
        
        task = asyncio.create_task(self._timeframe_collector(timeframe))
        self._tasks[timeframe] = task
        logger.info(f"📌 Added collector for {timeframe}")
    
    async def remove_timeframe(self, timeframe: str):
        """타임프레임 수집 제거"""
        if timeframe not in self._tasks:
            return
        
        self._tasks[timeframe].cancel()
        try:
            await self._tasks[timeframe]
        except asyncio.CancelledError:
            pass
        
        del self._tasks[timeframe]
        logger.info(f"🗑️ Removed collector for {timeframe}")
    
    def get_status(self) -> Dict[str, Any]:
        """스케줄러 상태 조회"""
        return {
            "is_running": self.is_running,
            "active_timeframes": list(self._tasks.keys()),
            "last_collection": {
                tf: time.isoformat() if time else None
                for tf, time in self._last_collection_time.items()
            }
        }


# 전역 스케줄러 인스턴스
_scheduler: Optional[SmartCandleScheduler] = None


def get_scheduler() -> Optional[SmartCandleScheduler]:
    """스케줄러 인스턴스 반환"""
    return _scheduler


def init_scheduler(binance_service: BinanceService) -> SmartCandleScheduler:
    """스케줄러 초기화"""
    global _scheduler
    _scheduler = SmartCandleScheduler(binance_service)
    return _scheduler





