"""
벡터 패턴 분석 모델
- 과거 기술적 지표를 벡터로 저장
- 유사 패턴 검색용
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, LargeBinary, JSON, Index
from sqlalchemy.sql import func
from app.database import Base


class VectorPattern(Base):
    """벡터화된 기술적 지표 패턴"""
    __tablename__ = "vector_patterns"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    
    # 지표 벡터 (실제 저장은 FAISS에, DB에는 참조용)
    vector_id = Column(Integer, index=True)  # FAISS 인덱스
    
    # 지표 원본 데이터 (참고용)
    indicators = Column(JSON)  # {'rsi': 65.2, 'macd': 0.015, ...}
    
    # 신호 및 결과
    signal = Column(String(10))  # BUY, SELL, HOLD
    confidence = Column(Float)
    
    # 결과 추적 (1시간, 4시간, 24시간 후 수익률)
    return_1h = Column(Float)  # 1시간 후 수익률 (%)
    return_4h = Column(Float)  # 4시간 후 수익률
    return_24h = Column(Float)  # 24시간 후 수익률
    
    price_at_signal = Column(Float)  # 신호 발생 시 가격
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_pattern_symbol_time', 'symbol', 'timeframe', 'timestamp'),
        Index('idx_pattern_vector', 'vector_id'),
    )


class VectorSimilarity(Base):
    """벡터 유사도 검색 결과 캐시"""
    __tablename__ = "vector_similarities"

    id = Column(Integer, primary_key=True, index=True)
    query_pattern_id = Column(Integer, nullable=False)  # 검색한 패턴
    similar_pattern_id = Column(Integer, nullable=False)  # 유사한 과거 패턴
    
    similarity_score = Column(Float)  # 0~1 유사도
    
    # 과거 패턴의 수익률
    past_return_1h = Column(Float)
    past_return_4h = Column(Float)
    past_return_24h = Column(Float)
    
    # 신호 강화 여부
    signal_boosted = Column(Integer, default=0)  # 신호 강화 적용 여부
    boost_amount = Column(Float, default=0)  # 신뢰도 증가분
    
    created_at = Column(DateTime, server_default=func.now())
