"""
Admin API 엔드포인트
- 가중치 관리
- 프리셋 관리
- 시스템 설정
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional
import logging

from app.services.weight_config import get_weight_config
from app.services.signal_service import RealTimeSignalService

router = APIRouter()
logger = logging.getLogger(__name__)


class WeightUpdate(BaseModel):
    """가중치 업데이트 요청"""
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
