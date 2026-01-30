"""
거래 피드백 API
- 거래 진입/종료 시 피드백 기록
- 모델 성과 분석
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.services.feedback_service import FeedbackService
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class EntryFeedbackRequest(BaseModel):
    """거래 진입 피드백"""
    symbol: str
    market_type: str = "futures"  # spot, futures
    position_type: str  # LONG, SHORT, BUY
    entry_price: float
    
    # AI 정보 (선택)
    ai_signal: Optional[str] = None
    ai_confidence: Optional[float] = None
    ai_probabilities: Optional[Dict[str, float]] = None
    model_used: Optional[str] = None
    
    # 지표 스냅샷 (선택)
    indicators: Optional[Dict[str, Any]] = None
    
    timeframe: str = "5m"
    leverage: int = 1
    is_paper: bool = True


class ExitFeedbackRequest(BaseModel):
    """거래 종료 피드백"""
    feedback_id: int
    exit_price: float
    pnl: float
    pnl_percent: float
    notes: Optional[str] = None


class FeedbackResponse(BaseModel):
    success: bool
    feedback_id: Optional[int] = None
    message: str


@router.post("/entry", response_model=FeedbackResponse)
async def record_entry(
    request: EntryFeedbackRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    🚀 거래 진입 시 피드백 기록 시작
    
    AI 예측 정보와 기술적 지표를 함께 저장하여
    나중에 모델 개선에 활용합니다.
    """
    try:
        service = FeedbackService(db)
        feedback_id = await service.record_entry(
            symbol=request.symbol,
            market_type=request.market_type,
            position_type=request.position_type,
            entry_price=request.entry_price,
            ai_signal=request.ai_signal,
            ai_confidence=request.ai_confidence,
            ai_probabilities=request.ai_probabilities,
            model_used=request.model_used,
            indicators=request.indicators,
            timeframe=request.timeframe,
            leverage=request.leverage,
            is_paper=request.is_paper
        )
        
        return FeedbackResponse(
            success=True,
            feedback_id=feedback_id,
            message=f"Entry feedback recorded: {request.symbol} {request.position_type}"
        )
    except Exception as e:
        logger.error(f"Entry feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/exit", response_model=FeedbackResponse)
async def record_exit(
    request: ExitFeedbackRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    📊 거래 종료 시 결과 기록
    
    실제 손익 결과를 기록하여 AI 예측의 정확도를 측정합니다.
    """
    try:
        service = FeedbackService(db)
        await service.record_exit(
            feedback_id=request.feedback_id,
            exit_price=request.exit_price,
            pnl=request.pnl,
            pnl_percent=request.pnl_percent,
            notes=request.notes
        )
        
        return FeedbackResponse(
            success=True,
            feedback_id=request.feedback_id,
            message=f"Exit feedback recorded: PnL {request.pnl_percent:.2f}%"
        )
    except Exception as e:
        logger.error(f"Exit feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/accuracy")
async def get_model_accuracy(
    symbol: Optional[str] = None,
    days: int = 30,
    db: AsyncSession = Depends(get_db)
):
    """
    🎯 모델 예측 정확도 조회
    
    실제 거래 결과 기반으로 AI 모델의 정확도를 계산합니다.
    """
    try:
        service = FeedbackService(db)
        stats = await service.get_model_accuracy(symbol, days)
        return stats
    except Exception as e:
        logger.error(f"Accuracy fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats-by-signal")
async def get_stats_by_signal(
    symbol: Optional[str] = None,
    days: int = 30,
    db: AsyncSession = Depends(get_db)
):
    """
    📈 AI 신호별 성과 분석
    
    BUY/SELL/HOLD 신호별로 실제 승률과 수익을 분석합니다.
    """
    try:
        service = FeedbackService(db)
        stats = await service.get_stats_by_signal(symbol, days)
        return stats
    except Exception as e:
        logger.error(f"Stats fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-data")
async def get_training_data(
    symbol: Optional[str] = None,
    timeframe: str = "5m",
    db: AsyncSession = Depends(get_db)
):
    """
    📚 모델 재학습용 데이터 조회
    
    피드백 데이터를 CSV 형태로 반환합니다.
    이 데이터로 모델을 fine-tuning할 수 있습니다.
    """
    try:
        service = FeedbackService(db)
        data = await service.get_feedback_for_training(
            symbol=symbol,
            timeframe=timeframe
        )
        
        return {
            "count": len(data),
            "symbol": symbol or "ALL",
            "timeframe": timeframe,
            "data": data
        }
    except Exception as e:
        logger.error(f"Training data fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

