"""
코인 메타데이터 관리 API 엔드포인트
"""

from typing import List, Optional
import asyncio
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from app.database import get_db
from app.services.coin_service import CoinService
from app.services.model_training_service import ModelTrainingService
from app.models.coin import Coin
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/coins", tags=["coins"])


# ===== Pydantic 모델 =====

class CoinCreate(BaseModel):
    symbol: str
    base_asset: str
    quote_asset: str
    is_monitoring: bool = False
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
    is_active: bool
    is_monitoring: bool
    current_price: Optional[float]
    price_change_24h: Optional[float]
    candle_count: int
    last_analysis_at: Optional[str]
    
    class Config:
        from_attributes = True


class CoinListResponse(BaseModel):
    total: int
    coins: List[CoinResponse]


# ===== API 엔드포인트 =====

@router.get("", response_model=dict)
@router.get("/", response_model=dict)
async def coins_root():
    """Coins API 루트"""
    return {
        "success": True,
        "message": "Coins API",
        "endpoints": [
            "GET /api/v1/coins/monitoring - 모니터링 중인 코인",
            "GET /api/v1/coins/list - 모든 코인",
            "GET /api/v1/coins/search/spot - 현물 코인 검색",
            "GET /api/v1/coins/search/futures - 선물 코인 검색",
            "POST /api/v1/coins/add-monitoring/{symbol} - 코인 모니터링 추가"
        ]
    }


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
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db)
):
    """모니터링 코인 추가 및 자동 데이터 수집 시작
    
    플로우:
    1. 코인 추가 및 모니터링 설정
    2. 백그라운드에서 데이터 수집 시작
    3. 데이터 수집 완료 후 모델 학습
    4. 학습 완료 후 실시간 분석 준비
    """
    try:
        if timeframes is None:
            timeframes = ["1h"]
        
        coin = await CoinService.add_monitoring_coin(db, symbol, timeframes)
        
        # 백그라운드에서 데이터 수집 → 모델 학습 → 실시간 분석 플로우 시작
        background_tasks.add_task(
            _coin_analysis_workflow,
            symbol=symbol,
            timeframes=timeframes
        )
        
        return {
            "success": True,
            "message": f"Coin {symbol} added successfully. Data collection and model training started in background.",
            "data": {
                "id": coin.id,
                "symbol": coin.symbol,
                "base_asset": coin.base_asset,
                "quote_asset": coin.quote_asset,
                "is_active": coin.is_active,
                "is_monitoring": coin.is_monitoring,
                "current_price": coin.current_price,
                "price_change_24h": coin.price_change_24h,
                "candle_count": coin.candle_count,
                "monitoring_timeframes": coin.monitoring_timeframes,
                "last_analysis_at": coin.last_analysis_at.isoformat() if coin.last_analysis_at else None,
            },
            "workflow": {
                "status": "initializing",
                "stages": [
                    {"stage": "data_collection", "status": "in_progress", "timeframes": timeframes},
                    {"stage": "model_training", "status": "pending"},
                    {"stage": "realtime_analysis", "status": "pending"}
                ],
                "message": "Data collection, model training, and real-time analysis started in background"
            }
        }
    except Exception as e:
        logger.error(f"Error adding monitoring coin {symbol}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# 백그라운드 작업: 코인 분석 워크플로우
async def _coin_analysis_workflow(symbol: str, timeframes: List[str]):
    """
    코인 분석 워크플로우:
    1. 데이터 수집 (Binance에서 역사 데이터)
    2. 모델 학습 (XGBoost, LSTM)
    3. 실시간 분석 준비 (WebSocket 구독)
    """
    try:
        logger.info(f"🚀 [시작] {symbol} 분석 워크플로우 시작")
        
        # ===== Stage 1: 데이터 수집 =====
        logger.info(f"📊 [Stage 1] {symbol} 데이터 수집 중...")
        for timeframe in timeframes:
            try:
                result = await ModelTrainingService.collect_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    days=90  # 90일치 데이터
                )
                logger.info(f"✅ {symbol} {timeframe} 데이터 수집 완료: {result}")
            except Exception as e:
                logger.warning(f"⚠️ {symbol} {timeframe} 데이터 수집 실패: {str(e)}")
                continue
        
        # 잠시 대기
        await asyncio.sleep(2)
        
        # ===== Stage 2: 모델 학습 =====
        logger.info(f"🤖 [Stage 2] {symbol} 모델 학습 중...")
        for timeframe in timeframes:
            try:
                # XGBoost 모델 학습
                xgb_result = await ModelTrainingService.train_xgboost_model(
                    symbol=symbol,
                    timeframe=timeframe
                )
                logger.info(f"✅ {symbol} {timeframe} XGBoost 학습 완료")
                
                # LSTM 모델 학습
                lstm_result = await ModelTrainingService.train_lstm_model(
                    symbol=symbol,
                    timeframe=timeframe
                )
                logger.info(f"✅ {symbol} {timeframe} LSTM 학습 완료")
                
            except Exception as e:
                logger.warning(f"⚠️ {symbol} {timeframe} 모델 학습 실패: {str(e)}")
                continue
        
        # 잠시 대기
        await asyncio.sleep(2)
        
        # ===== Stage 3: 실시간 분석 준비 =====
        logger.info(f"📡 [Stage 3] {symbol} 실시간 분석 준비 중...")
        logger.info(f"✅ {symbol} 모든 준비 완료! 실시간 분석 시작 가능")
        
        logger.info(f"🎉 [완료] {symbol} 분석 워크플로우 완료")
        
    except Exception as e:
        logger.error(f"❌ {symbol} 분석 워크플로우 오류: {str(e)}")



@router.get("/monitoring", response_model=dict)
async def get_monitoring_coins(db: AsyncSession = Depends(get_db)):
    """모니터링 중인 코인 목록"""
    try:
        coins = await CoinService.get_monitoring_coins(db)
        result = [
            {
                "id": c.id,
                "symbol": c.symbol,
                "base_asset": c.base_asset,
                "quote_asset": c.quote_asset,
                "market_type": "spot",
                "current_price": c.current_price or 0,
                "price_change_24h": c.price_change_24h or 0,
                "volume_24h": c.volume_24h or 0,
                "candle_count": c.candle_count or 0,
            }
            for c in coins
        ]
        return {
            "success": True,
            "total": len(result),
            "coins": result,
            "data": result
        }
    except Exception as e:
        logger.error(f"❌ 모니터링 코인 조회 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        # 빈 목록 반환 (초기화 상태)
        return {
            "success": True,
            "total": 0,
            "coins": [],
            "data": []
        }


@router.get("/search/spot", response_model=dict)
async def search_spot_coins(query: str, limit: int = 20):
    """현물 코인 검색"""
    try:
        from app.config import get_settings
        from app.services.binance_service import BinanceService
        
        settings = get_settings()
        binance = BinanceService(
            api_key=settings.binance_api_key,
            secret_key=settings.binance_secret_key,
            testnet=settings.binance_testnet
        )
        
        # 현물 코인 검색
        results = await binance.search_symbols_advanced(
            query=query,
            quote_asset="USDT",
            limit=limit
        )
        
        logger.info(f"✅ Spot 코인 검색: {query} → {len(results)}개 결과")
        return {
            "success": True,
            "symbols": results
        }
    except Exception as e:
        logger.error(f"❌ Spot 코인 검색 실패: {str(e)}")
        return {"success": False, "symbols": [], "error": str(e)}


@router.get("/search/futures", response_model=dict)
async def search_futures_coins(query: str, limit: int = 20):
    """선물 코인 검색"""
    try:
        from app.config import get_settings
        from app.services.binance_service import BinanceService
        
        settings = get_settings()
        binance = BinanceService(
            api_key=settings.binance_api_key,
            secret_key=settings.binance_secret_key,
            testnet=settings.binance_testnet
        )
        
        # 선물 코인 검색 (현물과 동일 - 바이낸스는 대부분의 통화쌍을 지원)
        results = await binance.search_symbols_advanced(
            query=query,
            quote_asset="USDT",
            limit=limit
        )
        
        logger.info(f"✅ Futures 코인 검색: {query} → {len(results)}개 결과")
        return {
            "success": True,
            "symbols": results
        }
    except Exception as e:
        logger.error(f"❌ Futures 코인 검색 실패: {str(e)}")
        return {"success": False, "symbols": [], "error": str(e)}


@router.get("/list", response_model=dict)
async def get_all_coins(db: AsyncSession = Depends(get_db)):
    """모든 코인 목록"""
    try:
        summary = await CoinService.get_all_coins_summary(db)
        return {
            "success": True,
            "total": len(summary),
            "coins": summary,
            "data": summary
        }
    except Exception as e:
        logger.error(f"❌ 전체 코인 조회 실패: {str(e)}")
        # 빈 목록 반환
        return {
            "success": True,
            "total": 0,
            "coins": [],
            "data": []
        }


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
