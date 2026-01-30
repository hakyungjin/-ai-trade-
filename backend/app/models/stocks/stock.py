"""
Stock data models for AI Trader
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base
from datetime import datetime


class Stock(Base):
    """Stock metadata and monitoring configuration"""
    __tablename__ = "stocks"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)  # AAPL, MSFT, etc.
    name = Column(String(255), nullable=True)
    sector = Column(String(100), nullable=True)  # Technology, Finance, etc.
    
    # Monitoring
    is_active = Column(Boolean, default=True)
    is_monitoring = Column(Boolean, default=False, index=True)
    monitoring_timeframes = Column(JSON, default=["1h"])  # ["15m", "1h", "1d"]
    
    # Price cache
    current_price = Column(Float, nullable=True)
    price_change_24h = Column(Float, nullable=True)
    price_change_7d = Column(Float, nullable=True)
    
    # Statistics
    candle_count = Column(Integer, default=0)
    last_analysis_at = Column(DateTime, nullable=True)
    last_updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    statistics = relationship("StockStatistics", back_populates="stock", cascade="all, delete-orphan")
    config = relationship("StockAnalysisConfig", back_populates="stock", cascade="all, delete-orphan", uselist=False)
    price_history = relationship("StockPriceHistory", back_populates="stock", cascade="all, delete-orphan")


class StockStatistics(Base):
    """Stock analysis statistics"""
    __tablename__ = "stock_statistics"
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False, index=True)
    
    # Candle statistics
    candle_count_15m = Column(Integer, default=0)
    candle_count_1h = Column(Integer, default=0)
    candle_count_1d = Column(Integer, default=0)
    
    # Signal statistics
    total_signals = Column(Integer, default=0)
    buy_signals = Column(Integer, default=0)
    sell_signals = Column(Integer, default=0)
    hold_signals = Column(Integer, default=0)
    
    # Performance
    avg_confidence = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    profitable_trades = Column(Integer, default=0)
    total_trades = Column(Integer, default=0)
    
    # AI models
    xgboost_trained = Column(Boolean, default=False)
    lstm_trained = Column(Boolean, default=False)
    last_training_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    stock = relationship("Stock", back_populates="statistics")


class StockAnalysisConfig(Base):
    """Stock analysis settings"""
    __tablename__ = "stock_analysis_configs"
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False, unique=True, index=True)
    
    # Technical indicators (on/off)
    use_rsi = Column(Boolean, default=True)
    use_macd = Column(Boolean, default=True)
    use_bollinger = Column(Boolean, default=True)
    use_stochastic = Column(Boolean, default=True)
    use_atr = Column(Boolean, default=True)
    
    # AI models
    use_gemini = Column(Boolean, default=True)
    use_xgboost = Column(Boolean, default=True)
    use_lstm = Column(Boolean, default=True)
    
    # Signal thresholds
    buy_threshold = Column(Float, default=0.65)
    sell_threshold = Column(Float, default=0.35)
    
    # Risk parameters (stocks have different risk profile)
    max_position_size = Column(Float, default=10000.0)  # USD
    max_daily_loss = Column(Float, default=0.05)  # 5%
    stop_loss = Column(Float, default=0.03)  # 3%
    take_profit = Column(Float, default=0.05)  # 5%
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    stock = relationship("Stock", back_populates="config")


class StockPriceHistory(Base):
    """Stock price history (cache)"""
    __tablename__ = "stock_price_history"
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False, index=True)
    
    price = Column(Float, nullable=False)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    volume = Column(Integer, nullable=True)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationship
    stock = relationship("Stock", back_populates="price_history")


class StockCandles(Base):
    """Stock OHLCV candle data"""
    __tablename__ = "stock_candles"
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False, index=True)
    
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)  # 15m, 1h, 1d
    
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    
    timestamp = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    class Config:
        indexes = [
            ("symbol", "timeframe", "timestamp"),
        ]
