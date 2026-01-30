"""
거래 피드백 서비스
- 거래 피드백 기록/조회
- 모델 개선용 데이터 생성
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
from app.models.trade_feedback import TradeFeedback

logger = logging.getLogger(__name__)


class FeedbackService:
    """거래 피드백 관리 서비스"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def record_entry(
        self,
        symbol: str,
        market_type: str,
        position_type: str,
        entry_price: float,
        ai_signal: str = None,
        ai_confidence: float = None,
        ai_probabilities: Dict = None,
        model_used: str = None,
        indicators: Dict = None,
        timeframe: str = '5m',
        leverage: int = 1,
        is_paper: bool = True
    ) -> int:
        """
        거래 진입 시 피드백 기록 시작
        Returns: feedback_id
        """
        feedback = TradeFeedback(
            symbol=symbol.upper(),
            market_type=market_type,
            position_type=position_type,
            entry_price=entry_price,
            ai_signal=ai_signal,
            ai_confidence=ai_confidence,
            ai_probabilities=ai_probabilities,
            model_used=model_used,
            indicators_snapshot=indicators,
            timeframe=timeframe,
            leverage=leverage,
            is_paper=is_paper,
            actual_label=-1  # 미정
        )
        
        self.db.add(feedback)
        await self.db.commit()
        await self.db.refresh(feedback)
        
        logger.info(f"📝 Feedback entry recorded: {symbol} {position_type} @ {entry_price}")
        return feedback.id
    
    async def record_exit(
        self,
        feedback_id: int,
        exit_price: float,
        pnl: float,
        pnl_percent: float,
        notes: str = None
    ) -> bool:
        """
        거래 종료 시 결과 기록
        """
        is_win = pnl > 0
        actual_label = 1 if is_win else 0
        
        stmt = (
            update(TradeFeedback)
            .where(TradeFeedback.id == feedback_id)
            .values(
                exit_price=exit_price,
                pnl=pnl,
                pnl_percent=pnl_percent,
                is_win=is_win,
                actual_label=actual_label,
                exit_time=datetime.utcnow(),
                notes=notes
            )
        )
        
        await self.db.execute(stmt)
        await self.db.commit()
        
        emoji = "✅" if is_win else "❌"
        logger.info(f"{emoji} Feedback exit recorded: ID={feedback_id}, PnL={pnl_percent:.2f}%")
        return True
    
    async def get_feedback_for_training(
        self,
        symbol: str = None,
        timeframe: str = '5m',
        min_trades: int = 50,
        only_closed: bool = True
    ) -> List[Dict[str, Any]]:
        """
        모델 학습용 피드백 데이터 조회
        """
        conditions = [TradeFeedback.actual_label >= 0]  # 종료된 거래만
        
        if symbol:
            conditions.append(TradeFeedback.symbol == symbol.upper())
        if timeframe:
            conditions.append(TradeFeedback.timeframe == timeframe)
        
        stmt = select(TradeFeedback).where(and_(*conditions))
        result = await self.db.execute(stmt)
        feedbacks = result.scalars().all()
        
        data = []
        for fb in feedbacks:
            if fb.indicators_snapshot:
                record = {
                    **fb.indicators_snapshot,
                    'ai_signal': fb.ai_signal,
                    'ai_confidence': fb.ai_confidence,
                    'position_type': fb.position_type,
                    'entry_price': fb.entry_price,
                    'exit_price': fb.exit_price,
                    'pnl_percent': fb.pnl_percent,
                    'actual_label': fb.actual_label,  # 학습 타겟
                    'is_win': fb.is_win
                }
                data.append(record)
        
        logger.info(f"📊 Retrieved {len(data)} feedback records for training")
        return data
    
    async def get_model_accuracy(
        self,
        symbol: str = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        모델 예측 정확도 분석
        """
        from datetime import timedelta
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        conditions = [
            TradeFeedback.actual_label >= 0,
            TradeFeedback.entry_time >= cutoff
        ]
        
        if symbol:
            conditions.append(TradeFeedback.symbol == symbol.upper())
        
        stmt = select(TradeFeedback).where(and_(*conditions))
        result = await self.db.execute(stmt)
        feedbacks = result.scalars().all()
        
        if not feedbacks:
            return {"error": "No data", "total_trades": 0}
        
        total = len(feedbacks)
        wins = sum(1 for f in feedbacks if f.is_win)
        
        # AI 예측 정확도 (AI가 BUY라고 했을 때 실제로 수익났는지)
        ai_correct = 0
        ai_total = 0
        
        for fb in feedbacks:
            if fb.ai_signal and fb.ai_confidence and fb.ai_confidence > 0.5:
                ai_total += 1
                # BUY 예측 + 수익 = 정확
                # SELL 예측 + 손실 = 정확 (SHORT의 경우)
                if fb.ai_signal == 'BUY' and fb.is_win:
                    ai_correct += 1
                elif fb.ai_signal == 'SELL' and not fb.is_win:
                    ai_correct += 1
        
        return {
            "period_days": days,
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": (wins / total * 100) if total > 0 else 0,
            "ai_predictions": ai_total,
            "ai_correct": ai_correct,
            "ai_accuracy": (ai_correct / ai_total * 100) if ai_total > 0 else 0,
            "total_pnl": sum(f.pnl or 0 for f in feedbacks),
            "avg_pnl_percent": sum(f.pnl_percent or 0 for f in feedbacks) / total if total > 0 else 0
        }
    
    async def get_stats_by_signal(
        self,
        symbol: str = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        AI 신호별 성과 분석
        """
        from datetime import timedelta
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        conditions = [
            TradeFeedback.actual_label >= 0,
            TradeFeedback.entry_time >= cutoff,
            TradeFeedback.ai_signal.isnot(None)
        ]
        
        if symbol:
            conditions.append(TradeFeedback.symbol == symbol.upper())
        
        stmt = select(TradeFeedback).where(and_(*conditions))
        result = await self.db.execute(stmt)
        feedbacks = result.scalars().all()
        
        # 신호별 분류
        stats = {
            'BUY': {'count': 0, 'wins': 0, 'total_pnl': 0},
            'SELL': {'count': 0, 'wins': 0, 'total_pnl': 0},
            'HOLD': {'count': 0, 'wins': 0, 'total_pnl': 0}
        }
        
        for fb in feedbacks:
            signal = fb.ai_signal or 'HOLD'
            if signal not in stats:
                stats[signal] = {'count': 0, 'wins': 0, 'total_pnl': 0}
            
            stats[signal]['count'] += 1
            if fb.is_win:
                stats[signal]['wins'] += 1
            stats[signal]['total_pnl'] += fb.pnl or 0
        
        # 승률 계산
        for signal in stats:
            count = stats[signal]['count']
            if count > 0:
                stats[signal]['win_rate'] = stats[signal]['wins'] / count * 100
                stats[signal]['avg_pnl'] = stats[signal]['total_pnl'] / count
            else:
                stats[signal]['win_rate'] = 0
                stats[signal]['avg_pnl'] = 0
        
        return stats

