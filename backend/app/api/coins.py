"""
코인 메타데이터 관리 API 엔드포인트
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from pydantic import BaseModel
import logging
from app.database import get_db
from app.services.coin_service import CoinService
from app.services.binance_service import BinanceService
from app.services.binance_futures_service import BinanceFuturesService
from app.config import get_settings
from app.models.coin import Coin

logger = logging.getLogger(__name__)


def get_binance_service() -> BinanceService:
    """Binance 현물 서비스 인스턴스 반환"""
    settings = get_settings()
    return BinanceService(
        api_key=settings.binance_api_key,
        secret_key=settings.binance_secret_key,
        testnet=settings.binance_testnet
    )


def get_futures_service() -> BinanceFuturesService:
    """Binance 선물 서비스 인스턴스 반환"""
    settings = get_settings()
    return BinanceFuturesService(
        api_key=settings.binance_api_key,
        secret_key=settings.binance_secret_key
    )

router = APIRouter(prefix="/api/v1/coins", tags=["coins"])


# ===== 헬스 체크 =====

@router.get("/check-db")
async def check_db_tables(db: AsyncSession = Depends(get_db)):
    """DB 테이블 존재 여부 확인"""
    try:
        # coins 테이블 확인
        result = await db.execute(text("SHOW TABLES LIKE 'coins'"))
        table_exists = result.fetchone() is not None
        
        if table_exists:
            # 테이블에 데이터가 있는지 확인
            count_result = await db.execute(text("SELECT COUNT(*) as count FROM coins"))
            count = count_result.fetchone()[0] if count_result.fetchone() else 0
            return {
                "success": True,
                "table_exists": True,
                "coin_count": count,
                "message": f"Coins table exists with {count} records"
            }
        else:
            return {
                "success": False,
                "table_exists": False,
                "message": "Coins table does not exist. Please run: alembic upgrade head"
            }
    except Exception as e:
        logger.error(f"Error checking DB: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Error checking database. Please check migration status."
        }


# ===== Pydantic 모델 =====

class CoinCreate(BaseModel):
    symbol: str
    base_asset: str
    quote_asset: str
    is_monitoring: bool = False
    market_type: str = 'spot'  # 'spot' 또는 'futures'
    full_name: Optional[str] = None
    description: Optional[str] = None


class CoinUpdate(BaseModel):
    is_monitoring: Optional[bool] = None
    priority: Optional[int] = None
    monitoring_timeframes: Optional[List[str]] = None


class CoinResponse(BaseModel):
    id: int
    symbol: str
    base_asset: str
    quote_asset: str
    market_type: str = 'spot'
    is_active: bool
    is_monitoring: bool
    current_price: Optional[float] = None
    price_change_24h: Optional[float] = None
    volume_24h: Optional[float] = None
    candle_count: int = 0
    monitoring_timeframes: Optional[List[str]] = None
    last_analysis_at: Optional[str] = None
    
    class Config:
        from_attributes = True


class CoinListResponse(BaseModel):
    total: int
    coins: List[CoinResponse]


# ===== API 엔드포인트 =====

@router.post("/add", response_model=CoinResponse)
async def add_coin(
    coin_data: CoinCreate,
    db: AsyncSession = Depends(get_db)
):
    """코인 추가"""
    try:
        coin = await CoinService.add_coin(
            db,
            symbol=coin_data.symbol,
            base_asset=coin_data.base_asset,
            quote_asset=coin_data.quote_asset,
            is_monitoring=coin_data.is_monitoring,
            full_name=coin_data.full_name,
            description=coin_data.description
        )
        return coin
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/add-monitoring/{symbol}")
async def add_monitoring_coin(
    symbol: str,
    timeframes: List[str] = None,
    market_type: str = 'spot',
    db: AsyncSession = Depends(get_db)
):
    """
    모니터링 코인 추가 및 자동 데이터 수집 시작
    
    Args:
        symbol: 심볼 (BTCUSDT)
        timeframes: 모니터링할 타임프레임 목록 (기본: ["1h"])
        market_type: 시장 유형 ('spot' 또는 'futures')
    """
    logger.info(f"🚀 Starting add_monitoring_coin for {symbol} ({market_type}) with timeframes: {timeframes}")
    try:
        if timeframes is None:
            timeframes = ["1h"]
        
        # 코인 추가 (DB에 저장)
        logger.info(f"📝 Calling CoinService.add_monitoring_coin for {symbol} ({market_type})")
        coin = await CoinService.add_monitoring_coin(db, symbol, timeframes, market_type)
        logger.info(f"📝 CoinService returned coin with ID: {coin.id if coin else None}")
        
        # get_db()가 자동으로 commit하므로, 여기서는 flush만 수행
        # commit은 get_db()의 finally 블록에서 자동으로 수행됨
        await db.flush()  # 변경사항을 DB에 반영 (아직 commit은 안됨)
        
        logger.info(f"✅ Coin {symbol} ({market_type}) added to session (ID: {coin.id}) - will be committed by get_db()")
        
        return {
            "success": True,
            "message": f"Coin {symbol} ({market_type}) added successfully. Data collection started in background.",
            "data": {
                "id": coin.id,
                "symbol": coin.symbol,
                "base_asset": coin.base_asset,
                "quote_asset": coin.quote_asset,
                "market_type": coin.market_type,
                "is_active": coin.is_active,
                "is_monitoring": coin.is_monitoring,
                "current_price": coin.current_price,
                "price_change_24h": coin.price_change_24h,
                "candle_count": coin.candle_count,
                "monitoring_timeframes": coin.monitoring_timeframes,
                "last_analysis_at": coin.last_analysis_at.isoformat() if coin.last_analysis_at else None,
            },
            "data_collection": {
                "status": "started",
                "timeframes": timeframes,
                "message": "Data collection is running in background"
            }
        }
    except Exception as e:
        logger.error(f"❌ Error adding monitoring coin {symbol}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/monitoring", response_model=CoinListResponse)
async def get_monitoring_coins(
    market_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    모니터링 중인 코인 목록 (실시간 가격 포함 - 병렬 조회)
    
    Args:
        market_type: 시장 유형 필터 ('spot', 'futures' 또는 None=전체)
    """
    import asyncio
    
    coins = await CoinService.get_monitoring_coins(db, market_type)
    
    if not coins:
        return CoinListResponse(total=0, coins=[])
    
    # 실시간 가격 정보 병렬 조회
    try:
        spot_binance = get_binance_service()
        futures_binance = get_futures_service()
        
        # 현물/선물 코인 분리
        spot_coins = [c for c in coins if c.market_type != 'futures']
        futures_coins = [c for c in coins if c.market_type == 'futures']
        
        # 한 번에 모든 티커 조회 (훨씬 빠름)
        spot_tickers = {}
        futures_tickers = {}
        
        async def fetch_all_spot_tickers():
            nonlocal spot_tickers
            try:
                all_tickers = await spot_binance.get_ticker_24h()
                if all_tickers:
                    spot_tickers = {t.get('symbol'): t for t in all_tickers}
            except Exception as e:
                logger.warning(f"Failed to fetch all spot tickers: {e}")
        
        async def fetch_all_futures_tickers():
            nonlocal futures_tickers
            try:
                all_tickers = await futures_binance.get_futures_ticker_24h()
                if all_tickers:
                    futures_tickers = {t.get('symbol'): t for t in all_tickers}
            except Exception as e:
                logger.warning(f"Failed to fetch all futures tickers: {e}")
        
        # 병렬로 모든 티커 조회
        tasks = []
        if spot_coins:
            tasks.append(fetch_all_spot_tickers())
        if futures_coins:
            tasks.append(fetch_all_futures_tickers())
        
        if tasks:
            await asyncio.gather(*tasks)
        
        # 각 코인에 가격 정보 매핑
        for coin in coins:
            try:
                if coin.market_type == 'futures':
                    ticker = futures_tickers.get(coin.symbol)
                    if ticker:
                        coin.current_price = float(ticker.get('lastPrice', 0) or ticker.get('price', 0))
                        coin.price_change_24h = float(ticker.get('priceChangePercent', 0))
                        coin.volume_24h = float(ticker.get('quoteVolume', 0))
                else:
                    ticker = spot_tickers.get(coin.symbol)
                    if ticker:
                        coin.current_price = float(ticker.get('lastPrice', 0))
                        coin.price_change_24h = float(ticker.get('priceChangePercent', 0))
                        coin.volume_24h = float(ticker.get('quoteVolume', 0))
            except Exception as e:
                logger.warning(f"Failed to map price for {coin.symbol}: {e}")
                
    except Exception as e:
        logger.error(f"Failed to fetch prices: {e}")
    
    return CoinListResponse(
        total=len(coins),
        coins=coins
    )


@router.get("/list", response_model=CoinListResponse)
async def get_all_coins(db: AsyncSession = Depends(get_db)):
    """모든 코인 목록"""
    summary = await CoinService.get_all_coins_summary(db)
    return CoinListResponse(
        total=len(summary),
        coins=summary
    )


@router.get("/{symbol}", response_model=CoinResponse)
async def get_coin(
    symbol: str,
    db: AsyncSession = Depends(get_db)
):
    """심볼로 코인 조회"""
    coin = await CoinService.get_coin_by_symbol(db, symbol)
    if not coin:
        raise HTTPException(status_code=404, detail=f"Coin {symbol} not found")
    return coin


@router.put("/{coin_id}", response_model=CoinResponse)
async def update_coin(
    coin_id: int,
    coin_update: CoinUpdate,
    db: AsyncSession = Depends(get_db)
):
    """코인 정보 업데이트"""
    try:
        # 기존 코인 조회
        from sqlalchemy import select
        stmt = select(Coin).where(Coin.id == coin_id)
        result = await db.execute(stmt)
        coin = result.scalar_one()
        
        # 업데이트
        if coin_update.is_monitoring is not None:
            coin.is_monitoring = coin_update.is_monitoring
        if coin_update.priority is not None:
            coin.priority = coin_update.priority
        if coin_update.monitoring_timeframes is not None:
            coin.monitoring_timeframes = coin_update.monitoring_timeframes
        
        await db.commit()
        return coin
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{coin_id}")
async def remove_coin(
    coin_id: int,
    db: AsyncSession = Depends(get_db)
):
    """모니터링 코인 제거"""
    success = await CoinService.remove_monitoring_coin(db, coin_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to remove coin")
    return {"message": "Coin removed successfully"}


@router.get("/{coin_id}/stats")
async def get_coin_stats(
    coin_id: int,
    db: AsyncSession = Depends(get_db)
):
    """코인 통계 조회"""
    stats = await CoinService.get_coin_stats(db, coin_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Stats not found")
    return stats


@router.get("/{coin_id}/config")
async def get_coin_config(
    coin_id: int,
    db: AsyncSession = Depends(get_db)
):
    """코인 분석 설정 조회"""
    config = await CoinService.get_coin_config(db, coin_id)
    if not config:
        raise HTTPException(status_code=404, detail="Config not found")
    return config


@router.put("/{coin_id}/config")
async def update_coin_config(
    coin_id: int,
    config_update: dict,
    db: AsyncSession = Depends(get_db)
):
    """코인 분석 설정 업데이트"""
    try:
        config = await CoinService.update_coin_config(db, coin_id, **config_update)
        return config
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ===== 선물 심볼 검색 API =====

@router.get("/search/spot")
async def search_spot_symbols(
    query: str = "",
    limit: int = 100
):
    """현물 심볼 검색"""
    from app.services.binance_service import BinanceService
    from app.config import get_settings
    
    config = get_settings()
    binance = BinanceService(config.binance_api_key, config.binance_secret_key)
    
    try:
        if query:
            symbols = await binance.search_symbols_advanced(query, quote_asset="USDT", limit=limit)
        else:
            symbols = await binance.get_top_symbols_by_volume(limit=limit, quote_asset="USDT")
        
        return {
            "success": True,
            "market_type": "spot",
            "total": len(symbols),
            "symbols": symbols
        }
    except Exception as e:
        logger.error(f"Error searching spot symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/futures")
async def search_futures_symbols(
    query: str = "",
    limit: int = 100
):
    """선물 심볼 검색"""
    from app.services.binance_futures_service import get_futures_service
    
    futures_service = get_futures_service()
    
    try:
        if query:
            symbols = await futures_service.search_futures_symbols(query, limit=limit)
        else:
            symbols = await futures_service.get_top_futures_by_volume(limit=limit)
        
        return {
            "success": True,
            "market_type": "futures",
            "total": len(symbols),
            "symbols": symbols
        }
    except Exception as e:
        logger.error(f"Error searching futures symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== 수동 데이터 수집 API =====

@router.post("/collect/{symbol}")
async def collect_coin_data(
    symbol: str,
    timeframe: str = "5m",
    limit: int = 500,
    db: AsyncSession = Depends(get_db)
):
    """
    특정 코인의 캔들 데이터 수동 수집
    
    Args:
        symbol: 심볼 (BTCUSDT)
        timeframe: 타임프레임 (5m, 1h, 4h 등)
        limit: 수집할 캔들 개수
    """
    logger.info(f"🚀 Manual data collection for {symbol} ({timeframe}), limit={limit}")
    
    try:
        # 코인 정보 조회
        coin = await CoinService.get_coin_by_symbol(db, symbol)
        
        if not coin:
            raise HTTPException(status_code=404, detail=f"Coin {symbol} not found")
        
        from app.services.market_data_service import MarketDataService
        market_service = MarketDataService(db)
        
        saved_count = 0
        
        if coin.market_type == 'futures':
            # 선물 데이터 수집
            from app.services.binance_futures_service import BinanceFuturesService
            settings = get_settings()
            futures_service = BinanceFuturesService(settings.binance_api_key, settings.binance_secret_key)
            
            klines = await futures_service.get_futures_klines(
                symbol=symbol,
                interval=timeframe,
                limit=limit
            )
            
            if klines:
                saved_count = await market_service.save_candles(
                    symbol=symbol,
                    timeframe=timeframe,
                    candles=klines
                )
                logger.info(f"✅ [Futures] Saved {saved_count} candles for {symbol}")
        else:
            # 현물 데이터 수집
            spot_binance = get_binance_service()
            
            klines = await spot_binance.get_klines(
                symbol=symbol,
                interval=timeframe,
                limit=limit
            )
            
            if klines:
                saved_count = await market_service.save_candles(
                    symbol=symbol,
                    timeframe=timeframe,
                    candles=klines
                )
                logger.info(f"✅ [Spot] Saved {saved_count} candles for {symbol}")
        
        # 코인 캔들 개수 업데이트
        if saved_count > 0:
            await CoinService.update_coin_candle_count(
                db,
                coin.id,
                (coin.candle_count or 0) + saved_count
            )
        
        return {
            "success": True,
            "symbol": symbol,
            "market_type": coin.market_type,
            "timeframe": timeframe,
            "saved_count": saved_count,
            "message": f"Collected {saved_count} candles for {symbol}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error collecting data for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collect-all")
async def collect_all_coins_data(
    timeframe: str = "5m",
    limit: int = 500,
    db: AsyncSession = Depends(get_db)
):
    """
    모든 모니터링 코인의 캔들 데이터 수동 수집
    """
    logger.info(f"🚀 Collecting data for all monitored coins ({timeframe})")
    
    try:
        # 모니터링 중인 코인 조회
        coins = await CoinService.get_monitoring_coins(db)
        
        results = []
        for coin in coins:
            try:
                # 개별 코인 데이터 수집 호출
                result = await collect_coin_data(
                    symbol=coin.symbol,
                    timeframe=timeframe,
                    limit=limit,
                    db=db
                )
                results.append(result)
            except Exception as e:
                results.append({
                    "success": False,
                    "symbol": coin.symbol,
                    "error": str(e)
                })
        
        success_count = sum(1 for r in results if r.get("success"))
        total_candles = sum(r.get("saved_count", 0) for r in results if r.get("success"))
        
        return {
            "success": True,
            "total_coins": len(coins),
            "success_count": success_count,
            "total_candles": total_candles,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"❌ Error collecting all coins data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
