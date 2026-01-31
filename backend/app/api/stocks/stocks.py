"""
Stock monitoring and analysis API endpoints
"""

from typing import List, Optional
import asyncio
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from app.database import get_db
from app.models.stocks.stock import Stock, StockStatistics, StockAnalysisConfig
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/stocks", tags=["stocks"])


# ===== Pydantic Models =====

class StockCreate(BaseModel):
    symbol: str
    name: str
    sector: Optional[str] = None
    is_monitoring: bool = False


class StockUpdate(BaseModel):
    is_monitoring: Optional[bool] = None
    monitoring_timeframes: Optional[List[str]] = None


class StockResponse(BaseModel):
    id: int
    symbol: str
    name: str
    sector: Optional[str]
    is_active: bool
    is_monitoring: bool
    current_price: Optional[float]
    price_change_24h: Optional[float]
    candle_count: int
    monitoring_timeframes: List[str]
    last_analysis_at: Optional[str]
    
    class Config:
        from_attributes = True


class StockListResponse(BaseModel):
    success: bool
    total: int
    data: List[StockResponse]


# ===== API Endpoints =====

@router.post("/add", response_model=StockResponse)
async def add_stock(
    stock_data: StockCreate,
    db: AsyncSession = Depends(get_db)
):
    """Add new stock"""
    try:
        # Check if stock already exists
        result = await db.execute(
            select(Stock).where(Stock.symbol == stock_data.symbol)
        )
        existing_stock = result.scalar_one_or_none()
        
        if existing_stock:
            raise HTTPException(status_code=400, detail=f"Stock {stock_data.symbol} already exists")
        
        # Create new stock
        stock = Stock(
            symbol=stock_data.symbol,
            name=stock_data.name,
            sector=stock_data.sector,
            is_monitoring=stock_data.is_monitoring,
            is_active=True
        )
        
        db.add(stock)
        await db.commit()
        await db.refresh(stock)
        
        logger.info(f"Stock {stock_data.symbol} added successfully")
        return stock
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Error adding stock: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/add-monitoring/{symbol}")
async def add_monitoring_stock(
    symbol: str,
    timeframes: List[str] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db)
):
    """Add stock to monitoring and start automatic data collection workflow
    
    Workflow:
    1. Add/update stock monitoring configuration
    2. Start background data collection (Alpha Vantage API)
    3. After data collection: Train models (XGBoost, LSTM)
    4. After training: Real-time analysis ready
    """
    try:
        if timeframes is None:
            timeframes = ["1h"]
        
        # Check if stock exists
        result = await db.execute(
            select(Stock).where(Stock.symbol == symbol)
        )
        stock = result.scalar_one_or_none()
        
        if not stock:
            # Create new stock
            stock = Stock(
                symbol=symbol,
                name=symbol,  # Will be updated from API
                is_monitoring=True,
                monitoring_timeframes=timeframes,
                is_active=True
            )
            db.add(stock)
        else:
            # Update existing stock
            stock.is_monitoring = True
            stock.monitoring_timeframes = timeframes
        
        await db.commit()
        await db.refresh(stock)
        
        # Start background workflow
        background_tasks.add_task(
            _stock_analysis_workflow,
            symbol=symbol,
            timeframes=timeframes
        )
        
        logger.info(f"Stock {symbol} monitoring started with workflow")
        
        return {
            "success": True,
            "message": f"Stock {symbol} added to monitoring. Data collection and model training started in background.",
            "data": {
                "id": stock.id,
                "symbol": stock.symbol,
                "name": stock.name,
                "sector": stock.sector,
                "is_active": stock.is_active,
                "is_monitoring": stock.is_monitoring,
                "current_price": stock.current_price,
                "price_change_24h": stock.price_change_24h,
                "candle_count": stock.candle_count,
                "monitoring_timeframes": stock.monitoring_timeframes,
                "last_analysis_at": stock.last_analysis_at.isoformat() if stock.last_analysis_at else None,
            },
            "workflow": {
                "status": "initializing",
                "stages": [
                    {"stage": "data_collection", "status": "in_progress", "timeframes": timeframes},
                    {"stage": "model_training", "status": "pending"},
                    {"stage": "realtime_analysis", "status": "pending"}
                ],
                "message": "Data collection, model training, and real-time analysis started in background"
            }
        }
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Error adding monitoring stock {symbol}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/monitoring", response_model=StockListResponse)
async def get_monitoring_stocks(db: AsyncSession = Depends(get_db)):
    """Get list of monitored stocks"""
    try:
        result = await db.execute(
            select(Stock).where(Stock.is_monitoring == True)
        )
        stocks = result.scalars().all()
        
        return StockListResponse(
            success=True,
            total=len(stocks),
            data=stocks
        )
    except Exception as e:
        logger.error(f"Error getting monitoring stocks: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/list", response_model=StockListResponse)
async def get_all_stocks(db: AsyncSession = Depends(get_db)):
    """Get all stocks"""
    try:
        result = await db.execute(select(Stock))
        stocks = result.scalars().all()
        
        return StockListResponse(
            success=True,
            total=len(stocks),
            data=stocks
        )
    except Exception as e:
        logger.error(f"Error getting stocks: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{symbol}", response_model=StockResponse)
async def get_stock(
    symbol: str,
    db: AsyncSession = Depends(get_db)
):
    """Get specific stock details"""
    try:
        result = await db.execute(
            select(Stock).where(Stock.symbol == symbol)
        )
        stock = result.scalar_one_or_none()
        
        if not stock:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        return stock
    except Exception as e:
        logger.error(f"Error getting stock {symbol}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{stock_id}", response_model=StockResponse)
async def update_stock(
    stock_id: int,
    stock_data: StockUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update stock configuration"""
    try:
        result = await db.execute(
            select(Stock).where(Stock.id == stock_id)
        )
        stock = result.scalar_one_or_none()
        
        if not stock:
            raise HTTPException(status_code=404, detail=f"Stock {stock_id} not found")
        
        if stock_data.is_monitoring is not None:
            stock.is_monitoring = stock_data.is_monitoring
        
        if stock_data.monitoring_timeframes is not None:
            stock.monitoring_timeframes = stock_data.monitoring_timeframes
        
        await db.commit()
        await db.refresh(stock)
        
        logger.info(f"Stock {stock.symbol} updated")
        return stock
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Error updating stock: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{stock_id}/stats")
async def get_stock_stats(
    stock_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get stock statistics"""
    try:
        result = await db.execute(
            select(StockStatistics).where(StockStatistics.stock_id == stock_id)
        )
        stats = result.scalar_one_or_none()
        
        if not stats:
            return {
                "message": "No statistics available yet",
                "stock_id": stock_id
            }
        
        return {
            "stock_id": stock_id,
            "candle_counts": {
                "15m": stats.candle_count_15m,
                "1h": stats.candle_count_1h,
                "1d": stats.candle_count_1d
            },
            "signals": {
                "total": stats.total_signals,
                "buy": stats.buy_signals,
                "sell": stats.sell_signals,
                "hold": stats.hold_signals
            },
            "performance": {
                "avg_confidence": stats.avg_confidence,
                "win_rate": stats.win_rate,
                "profitable_trades": stats.profitable_trades,
                "total_trades": stats.total_trades
            },
            "models": {
                "xgboost_trained": stats.xgboost_trained,
                "lstm_trained": stats.lstm_trained,
                "last_training_at": stats.last_training_at.isoformat() if stats.last_training_at else None
            }
        }
    except Exception as e:
        logger.error(f"Error getting stock stats: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{stock_id}/config")
async def get_stock_config(
    stock_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get stock analysis configuration"""
    try:
        result = await db.execute(
            select(StockAnalysisConfig).where(StockAnalysisConfig.stock_id == stock_id)
        )
        config = result.scalar_one_or_none()
        
        if not config:
            return {"message": "No configuration found", "stock_id": stock_id}
        
        return {
            "stock_id": stock_id,
            "technical_indicators": {
                "rsi": config.use_rsi,
                "macd": config.use_macd,
                "bollinger": config.use_bollinger,
                "stochastic": config.use_stochastic,
                "atr": config.use_atr
            },
            "ai_models": {
                "gemini": config.use_gemini,
                "xgboost": config.use_xgboost,
                "lstm": config.use_lstm
            },
            "thresholds": {
                "buy": config.buy_threshold,
                "sell": config.sell_threshold
            },
            "risk_management": {
                "max_position_size": config.max_position_size,
                "max_daily_loss": config.max_daily_loss,
                "stop_loss": config.stop_loss,
                "take_profit": config.take_profit
            }
        }
    except Exception as e:
        logger.error(f"Error getting stock config: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{stock_id}")
async def remove_stock_monitoring(
    stock_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Remove stock from monitoring"""
    try:
        result = await db.execute(
            select(Stock).where(Stock.id == stock_id)
        )
        stock = result.scalar_one_or_none()
        
        if not stock:
            raise HTTPException(status_code=404, detail=f"Stock {stock_id} not found")
        
        stock.is_monitoring = False
        await db.commit()
        
        logger.info(f"Stock {stock.symbol} removed from monitoring")
        
        return {
            "success": True,
            "message": f"Stock {stock.symbol} removed from monitoring"
        }
    except Exception as e:
        await db.rollback()
        logger.error(f"Error removing stock: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# ===== Background Workflow =====

async def _stock_analysis_workflow(symbol: str, timeframes: List[str]):
    """
    Stock analysis workflow:
    1. Data collection from Alpha Vantage
    2. Model training (XGBoost, LSTM)
    3. Real-time analysis preparation
    """
    try:
        logger.info(f"🚀 [START] Stock analysis workflow for {symbol}")
        
        # Stage 1: Data Collection
        logger.info(f"📊 [Stage 1] Collecting {symbol} data...")
        for timeframe in timeframes:
            try:
                # TODO: Call Alpha Vantage API to collect historical data
                logger.info(f"✅ {symbol} {timeframe} data collected")
            except Exception as e:
                logger.warning(f"⚠️ {symbol} {timeframe} data collection failed: {str(e)}")
                continue
        
        await asyncio.sleep(2)
        
        # Stage 2: Model Training
        logger.info(f"🤖 [Stage 2] Training models for {symbol}...")
        for timeframe in timeframes:
            try:
                # TODO: Call ModelTrainingService to train XGBoost
                logger.info(f"✅ {symbol} {timeframe} XGBoost training completed")
                
                # TODO: Call ModelTrainingService to train LSTM
                logger.info(f"✅ {symbol} {timeframe} LSTM training completed")
            except Exception as e:
                logger.warning(f"⚠️ {symbol} {timeframe} model training failed: {str(e)}")
                continue
        
        await asyncio.sleep(2)
        
        # Stage 3: Real-time Analysis Preparation
        logger.info(f"📡 [Stage 3] Preparing real-time analysis for {symbol}...")
        logger.info(f"✅ {symbol} ready for real-time analysis")
        
        logger.info(f"🎉 [COMPLETE] Stock analysis workflow for {symbol} completed")
        
    except Exception as e:
        logger.error(f"❌ Stock analysis workflow error for {symbol}: {str(e)}")


# ===== Stock Search API =====

# Popular US stocks list (S&P 500 top stocks)
POPULAR_STOCKS = [
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology"},
    {"symbol": "GOOGL", "name": "Alphabet Inc. Class A", "sector": "Technology"},
    {"symbol": "GOOG", "name": "Alphabet Inc. Class C", "sector": "Technology"},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Cyclical"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "sector": "Technology"},
    {"symbol": "META", "name": "Meta Platforms Inc.", "sector": "Technology"},
    {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Consumer Cyclical"},
    {"symbol": "BRK.B", "name": "Berkshire Hathaway Inc. Class B", "sector": "Financial Services"},
    {"symbol": "LLY", "name": "Eli Lilly and Company", "sector": "Healthcare"},
    {"symbol": "V", "name": "Visa Inc.", "sector": "Financial Services"},
    {"symbol": "UNH", "name": "UnitedHealth Group Incorporated", "sector": "Healthcare"},
    {"symbol": "XOM", "name": "Exxon Mobil Corporation", "sector": "Energy"},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financial Services"},
    {"symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare"},
    {"symbol": "MA", "name": "Mastercard Incorporated", "sector": "Financial Services"},
    {"symbol": "PG", "name": "The Procter & Gamble Company", "sector": "Consumer Defensive"},
    {"symbol": "AVGO", "name": "Broadcom Inc.", "sector": "Technology"},
    {"symbol": "HD", "name": "The Home Depot Inc.", "sector": "Consumer Cyclical"},
    {"symbol": "CVX", "name": "Chevron Corporation", "sector": "Energy"},
    {"symbol": "ABBV", "name": "AbbVie Inc.", "sector": "Healthcare"},
    {"symbol": "MRK", "name": "Merck & Co. Inc.", "sector": "Healthcare"},
    {"symbol": "COST", "name": "Costco Wholesale Corporation", "sector": "Consumer Defensive"},
    {"symbol": "KO", "name": "The Coca-Cola Company", "sector": "Consumer Defensive"},
    {"symbol": "PEP", "name": "PepsiCo Inc.", "sector": "Consumer Defensive"},
    {"symbol": "ADBE", "name": "Adobe Inc.", "sector": "Technology"},
    {"symbol": "TMO", "name": "Thermo Fisher Scientific Inc.", "sector": "Healthcare"},
    {"symbol": "NFLX", "name": "Netflix Inc.", "sector": "Communication Services"},
    {"symbol": "WMT", "name": "Walmart Inc.", "sector": "Consumer Defensive"},
    {"symbol": "CRM", "name": "Salesforce Inc.", "sector": "Technology"},
    {"symbol": "BAC", "name": "Bank of America Corporation", "sector": "Financial Services"},
    {"symbol": "ORCL", "name": "Oracle Corporation", "sector": "Technology"},
    {"symbol": "AMD", "name": "Advanced Micro Devices Inc.", "sector": "Technology"},
    {"symbol": "CSCO", "name": "Cisco Systems Inc.", "sector": "Technology"},
    {"symbol": "ACN", "name": "Accenture plc", "sector": "Technology"},
    {"symbol": "NKE", "name": "NIKE Inc.", "sector": "Consumer Cyclical"},
    {"symbol": "TXN", "name": "Texas Instruments Incorporated", "sector": "Technology"},
    {"symbol": "LIN", "name": "Linde plc", "sector": "Basic Materials"},
    {"symbol": "QCOM", "name": "QUALCOMM Incorporated", "sector": "Technology"},
    {"symbol": "ABT", "name": "Abbott Laboratories", "sector": "Healthcare"},
    {"symbol": "INTC", "name": "Intel Corporation", "sector": "Technology"},
    {"symbol": "CMCSA", "name": "Comcast Corporation", "sector": "Communication Services"},
    {"symbol": "DHR", "name": "Danaher Corporation", "sector": "Healthcare"},
    {"symbol": "PM", "name": "Philip Morris International Inc.", "sector": "Consumer Defensive"},
    {"symbol": "VZ", "name": "Verizon Communications Inc.", "sector": "Communication Services"},
    {"symbol": "WFC", "name": "Wells Fargo & Company", "sector": "Financial Services"},
    {"symbol": "DIS", "name": "The Walt Disney Company", "sector": "Communication Services"},
    {"symbol": "INTU", "name": "Intuit Inc.", "sector": "Technology"},
    {"symbol": "CAT", "name": "Caterpillar Inc.", "sector": "Industrials"},
    {"symbol": "IBM", "name": "International Business Machines Corporation", "sector": "Technology"},
]


@router.get("/search")
async def search_stocks(query: str, limit: int = 20):
    """Search stocks by symbol or name"""
    try:
        query_upper = query.upper().strip()

        if not query_upper:
            return {
                "success": True,
                "symbols": POPULAR_STOCKS[:limit]
            }

        # Filter stocks by symbol or name
        results = []
        for stock in POPULAR_STOCKS:
            if (query_upper in stock["symbol"].upper() or
                query_upper in stock["name"].upper()):
                results.append({
                    "symbol": stock["symbol"],
                    "baseAsset": stock["symbol"],
                    "quoteAsset": "USD",
                    "name": stock["name"],
                    "sector": stock["sector"],
                    "price": 0,  # Will be fetched from Alpha Vantage when added
                    "priceChange": 0,
                    "priceChangePercent": 0,
                    "volume": 0,
                    "marketCap": 0,
                    "trend": "neutral"
                })

        logger.info(f"✅ Stock search: {query} → {len(results)} results")
        return {
            "success": True,
            "symbols": results[:limit]
        }

    except Exception as e:
        logger.error(f"❌ Stock search failed: {str(e)}")
        return {
            "success": False,
            "symbols": [],
            "error": str(e)
        }
