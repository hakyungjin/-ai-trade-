"""
데이터 수집 및 관리 API
- 증분 데이터 수집
- 데이터 커버리지 확인
- 수동 동기화
- 배치 캔들 수집
"""

import asyncio
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.binance_service import BinanceService
from app.services.incremental_collector import IncrementalDataCollector
from app.services.batch_candle_collector import BatchCandleCollector
from app.config import get_settings

settings = get_settings()

router = APIRouter(
    prefix="/api/v1/data",
    tags=["data-collection"]
)

# Binance 서비스 인스턴스
binance_service = BinanceService(
    api_key=settings.binance_api_key,
    secret_key=settings.binance_secret_key,
    testnet=False
)

# 배치 수집 모델
class BatchCollectRequest(BaseModel):
    """배치 캔들 수집 요청"""
    interval: str = "1h"  # 캔들 주기
    limit: int = 500  # 각 심볼당 수집 개수
    symbols: Optional[List[str]] = None  # 특정 심볼만 수집 (None이면 모두)


class HistoricalDataRequest(BaseModel):
    """과거 데이터 수집 요청"""
    symbol: str
    interval: str = "1h"
    days: int = 365


@router.post("/sync/{symbol}")
async def sync_symbol_data(
    symbol: str,
    timeframe: str = "1h",
    force_full: bool = False,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    특정 심볼의 데이터 증분 수집

    Query Parameters:
    - timeframe: 시간프레임 (1m, 5m, 15m, 1h, 4h, 1d) - 기본값: 1h
    - force_full: 전체 재수집 여부 - 기본값: False

    Example:
    - POST /api/v1/data/sync/BTCUSDT?timeframe=1h
    - POST /api/v1/data/sync/ETHUSDT?timeframe=4h&force_full=true
    """
    try:
        collector = IncrementalDataCollector(db, binance_service)

        # 데이터 수집
        success, saved_count = await collector.collect_incremental_data(
            symbol=symbol,
            timeframe=timeframe,
            force_full=force_full
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to collect data")

        # 데이터 커버리지 정보
        coverage = await collector.get_data_coverage(symbol, timeframe)

        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "saved_candles": saved_count,
            "coverage": coverage,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync-all")
async def sync_all_data(
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    모든 심볼의 전체 데이터 동기화

    Request Body (JSON):
    {
        "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        "timeframes": ["1h", "4h", "1d"]
    }

    기본값:
    - symbols: ["BTCUSDT", "ETHUSDT"]
    - timeframes: ["1h", "4h", "1d"]
    """
    try:
        if symbols is None:
            symbols = ["BTCUSDT", "ETHUSDT"]

        if timeframes is None:
            timeframes = ["1h", "4h", "1d"]

        collector = IncrementalDataCollector(db, binance_service)

        # 전체 데이터 동기화
        results = await collector.sync_all_data(symbols, timeframes)

        return {
            "success": True,
            "symbols": symbols,
            "timeframes": timeframes,
            "results": results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/coverage/{symbol}")
async def get_data_coverage(
    symbol: str,
    timeframe: str = "1h",
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    데이터 커버리지 정보 조회

    Response:
    {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "first_time": "2025-01-01T00:00:00",
        "last_time": "2026-01-22T16:00:00",
        "total_candles": 8760,
        "expected_candles": 8760,
        "coverage_percent": 100.0,
        "gap_hours": 8760
    }
    """
    try:
        collector = IncrementalDataCollector(db, binance_service)
        coverage = await collector.get_data_coverage(symbol, timeframe)

        if not coverage:
            raise HTTPException(status_code=404, detail="No data found")

        return coverage

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/coverage")
async def get_all_coverage(
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    모든 심볼의 데이터 커버리지 정보 조회

    기본 심볼: BTCUSDT, ETHUSDT
    기본 타임프레임: 1h, 4h, 1d
    """
    try:
        symbols = ["BTCUSDT", "ETHUSDT"]
        timeframes = ["1h", "4h", "1d"]

        collector = IncrementalDataCollector(db, binance_service)

        coverage_info = {}
        for symbol in symbols:
            coverage_info[symbol] = {}
            for timeframe in timeframes:
                coverage = await collector.get_data_coverage(symbol, timeframe)
                coverage_info[symbol][timeframe] = coverage

        return {
            "success": True,
            "coverage": coverage_info,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-last-saved")
async def check_last_saved_time(
    symbol: str,
    timeframe: str = "1h",
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    마지막 저장된 캔들 시간 확인

    Response:
    {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "last_saved_time": "2026-01-22T16:00:00",
        "minutes_until_next": 44
    }
    """
    try:
        collector = IncrementalDataCollector(db, binance_service)
        last_time = await collector.get_last_saved_time(symbol, timeframe)

        if not last_time:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "last_saved_time": None,
                "message": "No data in database yet",
            }

        # 다음 캔들까지 남은 시간 계산
        from datetime import datetime, timedelta

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
        next_candle_time = last_time + timedelta(minutes=minutes)
        now = datetime.utcnow()
        minutes_until_next = int((next_candle_time - now).total_seconds() / 60)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "last_saved_time": last_time.isoformat(),
            "next_candle_time": next_candle_time.isoformat(),
            "minutes_until_next": max(minutes_until_next, 0),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-collect")
async def batch_collect_candles(
    request: BatchCollectRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    모든 활성 심볼의 캔들을 배치로 수집
    
    벡터 DB 학습 데이터 축적용
    
    Args:
        request.interval: 캔들 주기 (1m, 5m, 15m, 1h, 4h, 1d)
        request.limit: 각 심볼당 수집할 캔들 개수
        request.symbols: 특정 심볼만 수집할 경우 지정 (기본값: 모든 활성 심볼)
    
    Example:
    - POST /api/v1/data/batch-collect
    - POST /api/v1/data/batch-collect
      Body: {
        "interval": "1h",
        "limit": 500,
        "symbols": ["BTCUSDT", "ETHUSDT"]
      }
    """
    try:
        collector = BatchCandleCollector(binance_service)
        
        # 특정 심볼 수집 또는 모든 활성 심볼 수집
        if request.symbols:
            results = {}
            for symbol in request.symbols:
                count = await collector.collect_candles_for_symbol(
                    symbol=symbol,
                    interval=request.interval,
                    limit=request.limit,
                    db=db
                )
                results[symbol] = count
                await asyncio.sleep(0.5)  # 레이트 리미팅
            
            return {
                "success": True,
                "message": f"Collected candles for {len(request.symbols)} symbols",
                "interval": request.interval,
                "results": results,
                "total_saved": sum(results.values())
            }
        else:
            results = await collector.collect_all_symbols(
                interval=request.interval,
                limit=request.limit
            )
            
            return {
                "success": True,
                "message": f"Collected candles for {len(results)} active symbols",
                "interval": request.interval,
                "results": results,
                "total_saved": sum(results.values())
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch collection error: {str(e)}")


@router.post("/collect-historical")
async def collect_historical_data(
    request: HistoricalDataRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    특정 심볼의 과거 데이터 수집
    
    Args:
        request.symbol: 거래 쌍 (예: BTCUSDT)
        request.interval: 캔들 주기 (기본값: 1h)
        request.days: 과거 몇 일 데이터 (기본값: 365)
    
    Example:
    - POST /api/v1/data/collect-historical
      Body: {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "days": 365
      }
    """
    try:
        collector = BatchCandleCollector(binance_service)
        count = await collector.collect_historical_data(
            symbol=request.symbol,
            interval=request.interval,
            days=request.days
        )
        
        return {
            "success": True,
            "message": f"Collected historical data for {request.symbol}",
            "symbol": request.symbol,
            "interval": request.interval,
            "days": request.days,
            "saved_count": count
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Historical collection error: {str(e)}")


@router.post("/collect-daily-snapshot")
async def collect_daily_snapshot(
    db: AsyncSession = Depends(get_db)
):
    """
    모든 활성 심볼의 일일 스냅샷 수집 (1d 봉)
    
    Example:
    - POST /api/v1/data/collect-daily-snapshot
    """
    try:
        collector = BatchCandleCollector(binance_service)
        results = await collector.collect_daily_snapshot()
        
        return {
            "success": True,
            "message": "Collected daily snapshots for all active symbols",
            "interval": "1d",
            "results": results,
            "total_saved": sum(results.values())
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Snapshot collection error: {str(e)}")


# 사용 예시
"""
1. 단일 심볼 데이터 동기화:
   POST /api/v1/data/sync/BTCUSDT?timeframe=1h

2. 모든 심볼 동기화:
   POST /api/v1/data/sync-all
   Body: {
       "symbols": ["BTCUSDT", "ETHUSDT"],
       "timeframes": ["1h", "4h", "1d"]
   }

3. 데이터 커버리지 확인:
   GET /api/v1/data/coverage/BTCUSDT?timeframe=1h

4. 마지막 저장 시간 확인:
   POST /api/v1/data/check-last-saved?symbol=BTCUSDT&timeframe=1h

5. 배치 캔들 수집 (모든 활성 심볼):
   POST /api/v1/data/batch-collect

6. 배치 캔들 수집 (특정 심볼):
   POST /api/v1/data/batch-collect
   Body: {
       "interval": "1h",
       "limit": 500,
       "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
   }

7. 과거 데이터 수집:
   POST /api/v1/data/collect-historical
   Body: {
       "symbol": "BTCUSDT",
       "interval": "1h",
       "days": 365
   }

8. 일일 스냅샷 수집:
   POST /api/v1/data/collect-daily-snapshot
"""
