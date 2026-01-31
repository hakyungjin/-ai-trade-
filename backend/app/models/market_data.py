from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Index, UniqueConstraint
from sqlalchemy.sql import func
from app.database import Base


class MarketCandle(Base):
    """캔들 데이터 (OHLCV) - 최적화된 인덱싱"""
    __tablename__ = "market_candles"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)  # 1m, 5m, 15m, 1h, 4h, 1d
    open_time = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    close_time = Column(DateTime)
    quote_volume = Column(Float)  # 거래대금
    trades_count = Column(Integer)  # 거래 횟수

    # 통합 모델 지원 필드
    market_type = Column(String(20), server_default='crypto', index=True)  # crypto, nasdaq, forex
    is_market_open = Column(Integer, server_default='1')  # 시장 개장 여부 (1=open, 0=closed)

    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        # 기존 복합 인덱스
        Index('idx_candle_symbol_time', 'symbol', 'timeframe', 'open_time'),
        # ✅ 추가 최적화 인덱스 (MySQL/PostgreSQL 호환)
        Index('idx_candle_symbol_tf_time_desc', 'symbol', 'timeframe', 'open_time'),
        Index('idx_candle_symbol_time_desc', 'symbol', 'open_time'),
        Index('idx_candle_time_desc', 'open_time'),
        # 중복 방지 (같은 심볼/타임프레임/시간의 캔들 중복 저장 방지)
        UniqueConstraint('symbol', 'timeframe', 'open_time', name='uq_candle_symbol_tf_time'),
    )


class TechnicalIndicator(Base):
    """기술적 지표 데이터"""
    __tablename__ = "technical_indicators"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)

    # 이동평균선
    sma_5 = Column(Float)
    sma_10 = Column(Float)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)

    # RSI
    rsi_14 = Column(Float)

    # MACD
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)

    # 볼린저 밴드
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    bb_width = Column(Float)

    # 기타 지표
    atr_14 = Column(Float)
    volume_ratio = Column(Float)  # 평균 대비 거래량

    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index('idx_indicator_symbol_time', 'symbol', 'timeframe', 'timestamp'),
    )


class AIAnalysis(Base):
    """AI 분석 결과 저장"""
    __tablename__ = "ai_analyses"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)

    # 분석 결과
    signal = Column(String(10))  # BUY, SELL, HOLD
    confidence = Column(Float)
    direction = Column(String(10))  # UP, DOWN, NEUTRAL

    # AI 분석 내용
    analysis_text = Column(Text)
    short_term_analysis = Column(Text)
    mid_term_analysis = Column(Text)

    # 주요 레벨
    support_level = Column(Float)
    resistance_level = Column(Float)

    # 메타데이터
    source = Column(String(20))  # gemini, pytorch, rule_based
    model_version = Column(String(50))
    raw_response = Column(JSON)  # 전체 AI 응답 저장

    # 가격 정보 (분석 시점)
    price_at_analysis = Column(Float)

    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index('idx_analysis_symbol_time', 'symbol', 'timestamp'),
    )


class AITrainingData(Base):
    """AI 학습용 레이블 데이터"""
    __tablename__ = "ai_training_data"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)

    # 피처 데이터 (기술적 지표들)
    features = Column(JSON)  # 모든 지표를 JSON으로 저장

    # 레이블 (실제 결과)
    # future_price_1h: 1시간 후 가격
    # future_price_4h: 4시간 후 가격
    # future_price_24h: 24시간 후 가격
    future_price_1h = Column(Float)
    future_price_4h = Column(Float)
    future_price_24h = Column(Float)

    # 계산된 레이블
    # 0: 하락 (< -1%), 1: 횡보 (-1% ~ 1%), 2: 상승 (> 1%)
    label_1h = Column(Integer)
    label_4h = Column(Integer)
    label_24h = Column(Integer)

    # 현재 가격
    current_price = Column(Float, nullable=False)

    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index('idx_training_symbol_time', 'symbol', 'timeframe', 'timestamp'),
    )


class SignalHistory(Base):
    """생성된 매매 신호 히스토리"""
    __tablename__ = "signal_history"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)

    # 신호 정보
    signal = Column(String(10), nullable=False)  # BUY, SELL, HOLD
    confidence = Column(Float)
    source = Column(String(20))  # ai, rule_based, combined

    # 분석 시점 가격
    price_at_signal = Column(Float, nullable=False)

    # 신호 실행 여부
    executed = Column(Integer, default=0)  # 0: 미실행, 1: 실행됨

    # 결과 추적 (나중에 업데이트)
    price_after_1h = Column(Float)
    price_after_4h = Column(Float)
    price_after_24h = Column(Float)
    result = Column(String(20))  # profit, loss, neutral
    profit_percent = Column(Float)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
