"""
거래 피드백 모델
- AI 예측 시점의 지표와 실제 거래 결과를 기록
- 모델 개선에 활용
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.sql import func
from app.database import Base


class TradeFeedback(Base):
    """거래 피드백 테이블"""
    __tablename__ = "trade_feedback"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 거래 정보
    symbol = Column(String(20), nullable=False, index=True)
    market_type = Column(String(10), nullable=False)  # spot, futures
    position_type = Column(String(10), nullable=False)  # LONG, SHORT, BUY
    
    # AI 예측 정보 (거래 진입 시점)
    ai_signal = Column(String(20))  # BUY, SELL, HOLD
    ai_confidence = Column(Float)
    ai_probabilities = Column(JSON)  # {"BUY": 0.7, "SELL": 0.3}
    model_used = Column(String(100))  # XGBoost, Ensemble 등
    
    # 기술적 지표 스냅샷 (진입 시점)
    indicators_snapshot = Column(JSON)  # RSI, MACD, 볼린저 등
    
    # 가격 정보
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    
    # 거래 결과
    pnl = Column(Float)  # 손익 (USDT)
    pnl_percent = Column(Float)  # 손익률 (%)
    is_win = Column(Boolean)  # 수익 여부
    
    # 레이블 (모델 학습용)
    actual_label = Column(Integer)  # 1: 수익, 0: 손실, -1: 미정
    
    # 타임스탬프
    entry_time = Column(DateTime(timezone=True), server_default=func.now())
    exit_time = Column(DateTime(timezone=True))
    
    # 메타 정보
    timeframe = Column(String(10), default='5m')
    leverage = Column(Integer, default=1)
    is_paper = Column(Boolean, default=True)  # 모의투자 여부
    
    # 추가 분석용
    notes = Column(Text)  # 사용자 메모

