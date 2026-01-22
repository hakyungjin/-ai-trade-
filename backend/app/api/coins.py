"""
코인 메타데이터 관리 API 엔드포인트
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from app.database import get_db
from app.services.coin_service import CoinService
from app.models.coin import Coin

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


@router.post("/add-monitoring/{symbol}", response_model=CoinResponse)
async def add_monitoring_coin(
    symbol: str,
    timeframes: List[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """모니터링 코인 추가"""
    try:
        if timeframes is None:
            timeframes = ["1h"]
        
        coin = await CoinService.add_monitoring_coin(db, symbol, timeframes)
        return coin
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/monitoring", response_model=CoinListResponse)
async def get_monitoring_coins(db: AsyncSession = Depends(get_db)):
    """모니터링 중인 코인 목록"""
    coins = await CoinService.get_monitoring_coins(db)
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
