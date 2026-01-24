"""
코인 메타데이터 및 모니터링 설정
- 모니터링 중인 코인 정보
- 코인별 설정 및 통계
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Index, ForeignKey
from sqlalchemy.sql import func
from app.database import Base


class Coin(Base):
    """모니터링 코인 기본 정보"""
    __tablename__ = "coins"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)  # BTCUSDT
    base_asset = Column(String(20), nullable=False)  # BTC
    quote_asset = Column(String(20), nullable=False)  # USDT
    market_type = Column(String(10), nullable=False, default='spot', index=True)  # spot 또는 futures
    
    # 상태 관리
    is_active = Column(Boolean, default=True, index=True)  # 활성 모니터링 중
    is_monitoring = Column(Boolean, default=False, index=True)  # 사용자가 추가한 코인
    
    # 코인 정보
    full_name = Column(String(100))  # Bitcoin
    description = Column(String(500))  # 설명
    
    # 최신 시세 정보 (캐시)
    current_price = Column(Float)
    price_change_24h = Column(Float)  # 24h 가격 변동률
    volume_24h = Column(Float)  # 24h 거래량
    market_cap = Column(Float)
    last_price_update = Column(DateTime)
    
    # 수집 통계
    candle_count = Column(Integer, default=0)  # 저장된 캔들 개수
    earliest_candle_time = Column(DateTime)  # 가장 오래된 캔들
    latest_candle_time = Column(DateTime)  # 가장 최신 캔들
    
    # 설정
    monitoring_timeframes = Column(JSON, default=lambda: ["1h"])  # ["1h", "4h", "1d"]
    priority = Column(Integer, default=0)  # 우선순위 (높을수록 먼저 분석)
    
    # 타임스탬프
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    last_analysis_at = Column(DateTime)  # 마지막 분석 시간

    __table_args__ = (
        Index('idx_coin_active', 'is_active'),
        Index('idx_coin_monitoring', 'is_monitoring'),
        Index('idx_coin_priority', 'priority'),
        Index('idx_coin_market_type', 'market_type'),
        Index('idx_coin_symbol_market', 'symbol', 'market_type', unique=True),  # 심볼+마켓타입 유니크
    )


class CoinStatistics(Base):
    """코인별 누적 통계"""
    __tablename__ = "coin_statistics"

    id = Column(Integer, primary_key=True, index=True)
    coin_id = Column(Integer, ForeignKey('coins.id'), nullable=False, unique=True)
    
    # 캔들 통계
    total_candles = Column(Integer, default=0)  # 총 캔들 개수
    candles_1h = Column(Integer, default=0)
    candles_4h = Column(Integer, default=0)
    candles_1d = Column(Integer, default=0)
    
    # 분석 통계
    total_signals = Column(Integer, default=0)  # 생성된 신호 개수
    buy_signals = Column(Integer, default=0)
    sell_signals = Column(Integer, default=0)
    neutral_signals = Column(Integer, default=0)
    
    # 성능 지표
    average_confidence = Column(Float)  # 평균 확신도
    win_rate = Column(Float)  # 수익률 (추후)
    total_returns = Column(Float)  # 총 수익 (추후)
    
    # 벡터 DB 패턴
    pattern_vectors_count = Column(Integer, default=0)  # 저장된 패턴 벡터 개수
    similar_patterns_found = Column(Integer, default=0)  # 발견된 유사 패턴 개수
    
    # 타임스탬프
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('idx_coin_stats_coin_id', 'coin_id'),
    )


class CoinAnalysisConfig(Base):
    """코인별 분석 설정"""
    __tablename__ = "coin_analysis_configs"

    id = Column(Integer, primary_key=True, index=True)
    coin_id = Column(Integer, ForeignKey('coins.id'), nullable=False, unique=True)
    
    # 기술적 지표 사용 여부
    use_rsi = Column(Boolean, default=True)
    use_macd = Column(Boolean, default=True)
    use_bollinger = Column(Boolean, default=True)
    use_moving_average = Column(Boolean, default=True)
    use_stochastic = Column(Boolean, default=True)
    
    # AI 분석 설정
    use_gemini_ai = Column(Boolean, default=True)
    use_local_model = Column(Boolean, default=True)
    use_vector_patterns = Column(Boolean, default=False)  # 벡터 패턴 유사성 활용
    
    # 신호 임계값
    buy_threshold = Column(Float, default=0.3)  # 매수 신호 임계값
    strong_buy_threshold = Column(Float, default=0.6)
    sell_threshold = Column(Float, default=-0.3)
    strong_sell_threshold = Column(Float, default=-0.6)
    
    # 알림 설정
    notify_on_strong_signals = Column(Boolean, default=True)
    notify_on_pattern_found = Column(Boolean, default=False)
    
    # 타임스탬프
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('idx_coin_config_coin_id', 'coin_id'),
    )


class CoinPriceHistory(Base):
    """코인 시세 이력 (캐시용)"""
    __tablename__ = "coin_price_history"

    id = Column(Integer, primary_key=True, index=True)
    coin_id = Column(Integer, ForeignKey('coins.id'), nullable=False, index=True)
    
    price = Column(Float, nullable=False)
    price_change_24h = Column(Float)
    volume_24h = Column(Float)
    market_cap = Column(Float)
    
    recorded_at = Column(DateTime, nullable=False, index=True)

    __table_args__ = (
        Index('idx_price_coin_time', 'coin_id', 'recorded_at'),
    )
