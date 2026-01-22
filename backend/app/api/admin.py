"""
Admin API 엔드포인트
- 가중치 관리
- 프리셋 관리
- 시스템 설정
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.services.weight_config import get_weight_config
from app.services.signal_service import RealTimeSignalService
from app.database import get_db
from app.models.strategy_weights import StrategyWeights

router = APIRouter()
logger = logging.getLogger(__name__)


class StrategyWeightUpdate(BaseModel):
    """전략 가중치 업데이트"""
    rsi_weight: float = Field(default=0.20, ge=0, le=1)
    macd_weight: float = Field(default=0.25, ge=0, le=1)
    bollinger_weight: float = Field(default=0.15, ge=0, le=1)
    ema_cross_weight: float = Field(default=0.20, ge=0, le=1)
    stochastic_weight: float = Field(default=0.10, ge=0, le=1)
    volume_weight: float = Field(default=0.10, ge=0, le=1)
    
    # 신호 임계값
    strong_buy_threshold: float = Field(default=0.6, ge=-1, le=1)
    buy_threshold: float = Field(default=0.3, ge=-1, le=1)
    sell_threshold: float = Field(default=-0.3, ge=-1, le=1)
    strong_sell_threshold: float = Field(default=-0.6, ge=-1, le=1)
    
    # 벡터 설정
    vector_boost_enabled: int = Field(default=1, ge=0, le=1)
    vector_similarity_threshold: float = Field(default=0.75, ge=0, le=1)
    vector_k_nearest: int = Field(default=5, ge=1, le=20)
    max_confidence_boost: float = Field(default=0.15, ge=0, le=0.5)


class WeightUpdate(BaseModel):
    """가중치 업데이트 요청 (하위호환성)"""
    rsi: float = Field(ge=0, le=1, description="RSI 가중치 (0-1)")
    macd: float = Field(ge=0, le=1, description="MACD 가중치 (0-1)")
    bollinger: float = Field(ge=0, le=1, description="Bollinger Bands 가중치 (0-1)")
    ema_cross: float = Field(ge=0, le=1, description="EMA Cross 가중치 (0-1)")
    stochastic: float = Field(ge=0, le=1, description="Stochastic 가중치 (0-1)")
    volume: float = Field(ge=0, le=1, description="Volume 가중치 (0-1)")


class CustomPreset(BaseModel):
    """커스텀 프리셋"""
    name: str = Field(min_length=1, max_length=50, description="프리셋 이름")
    description: str = Field(max_length=200, description="프리셋 설명")
    weights: WeightUpdate


@router.get("/weights")
async def get_weights():
    """
    현재 가중치 조회

    Returns:
        현재 가중치 설정
    """
    try:
        config = get_weight_config()
        weights = config.get_weights()

        return {
            'success': True,
            'weights': weights,
            'total': sum(weights.values())
        }

    except Exception as e:
        logger.error(f"Error getting weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weights/info")
async def get_weight_info():
    """
    가중치 상세 정보 조회

    Returns:
        가중치 정보 (현재, 기본값, 프리셋 등)
    """
    try:
        config = get_weight_config()
        info = config.get_weight_info()

        return {
            'success': True,
            'info': info
        }

    except Exception as e:
        logger.error(f"Error getting weight info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/weights")
async def update_weights(weights: WeightUpdate):
    """
    가중치 업데이트

    Args:
        weights: 새로운 가중치

    Returns:
        업데이트 결과
    """
    try:
        config = get_weight_config()

        # 딕셔너리로 변환
        weights_dict = weights.model_dump()

        # 업데이트
        result = config.update_weights(weights_dict)

        if not result['valid']:
            raise HTTPException(status_code=400, detail=result)

        return {
            'success': True,
            'message': result['message'],
            'weights': result['weights']
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/presets")
async def get_presets():
    """
    프리셋 목록 조회

    Returns:
        모든 프리셋
    """
    try:
        config = get_weight_config()
        presets = config.get_presets()

        return {
            'success': True,
            'presets': presets
        }

    except Exception as e:
        logger.error(f"Error getting presets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/presets/{preset_name}/load")
async def load_preset(preset_name: str):
    """
    프리셋 로드

    Args:
        preset_name: 프리셋 이름

    Returns:
        로드 결과
    """
    try:
        config = get_weight_config()
        result = config.load_preset(preset_name)

        if not result['valid']:
            raise HTTPException(status_code=404, detail=result)

        return {
            'success': True,
            'message': result['message'],
            'preset': result['preset'],
            'weights': result['weights']
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading preset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/presets/custom")
async def create_custom_preset(preset: CustomPreset):
    """
    커스텀 프리셋 생성

    Args:
        preset: 프리셋 정보

    Returns:
        생성 결과
    """
    try:
        config = get_weight_config()

        weights_dict = preset.weights.model_dump()

        result = config.create_custom_preset(
            name=preset.name,
            description=preset.description,
            weights=weights_dict
        )

        if not result['valid']:
            raise HTTPException(status_code=400, detail=result)

        return {
            'success': True,
            'message': result['message'],
            'preset': result['preset']
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating custom preset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/weights/reset")
async def reset_weights():
    """
    가중치를 기본값으로 리셋

    Returns:
        리셋 결과
    """
    try:
        config = get_weight_config()
        result = config.reset_to_default()

        if not result['valid']:
            raise HTTPException(status_code=500, detail=result)

        return {
            'success': True,
            'message': result['message'],
            'weights': result['weights']
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_system_stats():
    """
    시스템 통계 조회

    Returns:
        시스템 상태 및 통계
    """
    try:
        # 가중치 정보
        config = get_weight_config()
        weights = config.get_weights()

        # 시스템 정보 (추후 확장)
        stats = {
            'weights': weights,
            'active_features': {
                'real_time_signals': True,
                'ai_prediction': True,
                'backtesting': True,
                'risk_management': True
            }
        }

        return {
            'success': True,
            'stats': stats
        }

    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== DB 기반 가중치 관리 =====

@router.get("/strategy-weights")
async def get_active_weights(db: AsyncSession = Depends(get_db)):
    """활성 전략 가중치 조회"""
    try:
        result = await db.execute(
            select(StrategyWeights).where(StrategyWeights.active == 1)
        )
        weights = result.scalar_one_or_none()
        
        if not weights:
            raise HTTPException(status_code=404, detail="No active weights found")
        
        return {
            'id': weights.id,
            'name': weights.name,
            'description': weights.description,
            'rsi_weight': weights.rsi_weight,
            'macd_weight': weights.macd_weight,
            'bollinger_weight': weights.bollinger_weight,
            'ema_cross_weight': weights.ema_cross_weight,
            'stochastic_weight': weights.stochastic_weight,
            'volume_weight': weights.volume_weight,
            'strong_buy_threshold': weights.strong_buy_threshold,
            'buy_threshold': weights.buy_threshold,
            'sell_threshold': weights.sell_threshold,
            'strong_sell_threshold': weights.strong_sell_threshold,
            'vector_boost_enabled': weights.vector_boost_enabled,
            'vector_similarity_threshold': weights.vector_similarity_threshold,
            'vector_k_nearest': weights.vector_k_nearest,
            'max_confidence_boost': weights.max_confidence_boost,
        }
    except Exception as e:
        logger.error(f"Error getting active weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategy-weights/all")
async def get_all_weights(db: AsyncSession = Depends(get_db)):
    """모든 전략 가중치 조회"""
    try:
        result = await db.execute(select(StrategyWeights))
        all_weights = result.scalars().all()
        
        return [
            {
                'id': w.id,
                'name': w.name,
                'description': w.description,
                'active': w.active,
                'created_at': w.created_at.isoformat(),
            }
            for w in all_weights
        ]
    except Exception as e:
        logger.error(f"Error getting all weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategy-weights/{weight_id}")
async def update_weights(
    weight_id: int,
    update: StrategyWeightUpdate,
    db: AsyncSession = Depends(get_db)
):
    """전략 가중치 업데이트"""
    try:
        # 가중치 합 검증
        weight_sum = (
            update.rsi_weight + update.macd_weight + update.bollinger_weight +
            update.ema_cross_weight + update.stochastic_weight + update.volume_weight
        )
        
        if abs(weight_sum - 1.0) > 0.01:
            raise HTTPException(
                status_code=400,
                detail=f"Weights must sum to 1.0, got {weight_sum:.2f}"
            )
        
        # 임계값 검증
        if update.strong_buy_threshold <= update.buy_threshold:
            raise HTTPException(status_code=400, detail="strong_buy > buy threshold")
        if update.buy_threshold <= update.sell_threshold:
            raise HTTPException(status_code=400, detail="buy > sell threshold")
        if update.sell_threshold <= update.strong_sell_threshold:
            raise HTTPException(status_code=400, detail="sell > strong_sell threshold")
        
        # DB 업데이트
        result = await db.execute(select(StrategyWeights).where(StrategyWeights.id == weight_id))
        weights = result.scalar_one_or_none()
        
        if not weights:
            raise HTTPException(status_code=404, detail="Weights not found")
        
        # 모든 필드 업데이트
        weights.rsi_weight = update.rsi_weight
        weights.macd_weight = update.macd_weight
        weights.bollinger_weight = update.bollinger_weight
        weights.ema_cross_weight = update.ema_cross_weight
        weights.stochastic_weight = update.stochastic_weight
        weights.volume_weight = update.volume_weight
        weights.strong_buy_threshold = update.strong_buy_threshold
        weights.buy_threshold = update.buy_threshold
        weights.sell_threshold = update.sell_threshold
        weights.strong_sell_threshold = update.strong_sell_threshold
        weights.vector_boost_enabled = update.vector_boost_enabled
        weights.vector_similarity_threshold = update.vector_similarity_threshold
        weights.vector_k_nearest = update.vector_k_nearest
        weights.max_confidence_boost = update.max_confidence_boost
        
        await db.commit()
        logger.info(f"✅ Updated weights ID {weight_id}")
        
        return {
            'success': True,
            'message': f'Weights updated successfully',
            'weight_id': weight_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating weights: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategy-weights/{weight_id}/activate")
async def activate_weights(weight_id: int, db: AsyncSession = Depends(get_db)):
    """특정 가중치 설정 활성화"""
    try:
        # 기존 활성 가중치 비활성화
        await db.execute(
            select(StrategyWeights).where(StrategyWeights.active == 1)
        )
        existing = (await db.execute(select(StrategyWeights).where(StrategyWeights.active == 1))).scalar_one_or_none()
        if existing:
            existing.active = 0
        
        # 새 가중치 활성화
        result = await db.execute(select(StrategyWeights).where(StrategyWeights.id == weight_id))
        weights = result.scalar_one_or_none()
        
        if not weights:
            raise HTTPException(status_code=404, detail="Weights not found")
        
        weights.active = 1
        await db.commit()
        
        logger.info(f"✅ Activated weights ID {weight_id}: {weights.name}")
        
        return {
            'success': True,
            'message': f'Activated {weights.name}',
            'weight_id': weight_id
        }
    
    except Exception as e:
        logger.error(f"Error activating weights: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

