"""
전략 가중치 설정 모델
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Index
from sqlalchemy.sql import func
from app.database import Base


class StrategyWeights(Base):
    """매매 전략 가중치 설정"""
    __tablename__ = "strategy_weights"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False, unique=True)  # 'default', 'aggressive', etc.
    description = Column(String(255))
    
    # 기본 가중치 (합 = 1.0)
    rsi_weight = Column(Float, default=0.20)
    macd_weight = Column(Float, default=0.25)
    bollinger_weight = Column(Float, default=0.15)
    ema_cross_weight = Column(Float, default=0.20)
    stochastic_weight = Column(Float, default=0.10)
    volume_weight = Column(Float, default=0.10)
    
    # 신호 임계값
    strong_buy_threshold = Column(Float, default=0.6)
    buy_threshold = Column(Float, default=0.3)
    sell_threshold = Column(Float, default=-0.3)
    strong_sell_threshold = Column(Float, default=-0.6)
    
    # 벡터 패턴 설정
    vector_boost_enabled = Column(Integer, default=1)  # 0: 비활성, 1: 활성
    vector_similarity_threshold = Column(Float, default=0.75)  # 유사도 임계값
    vector_k_nearest = Column(Integer, default=5)  # 검색할 유사 패턴 개수
    max_confidence_boost = Column(Float, default=0.15)  # 최대 신뢰도 증가량
    
    # 활성화 여부
    active = Column(Integer, default=1)  # 0: 비활성, 1: 활성
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_weights_active', 'active'),
    )
