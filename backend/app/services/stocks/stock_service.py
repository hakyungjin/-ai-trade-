"""
Stock market data service (Alpha Vantage integration)
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import aiohttp

logger = logging.getLogger(__name__)


class AlphaVantageService:
    """Alpha Vantage API service for US stock data"""
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: str = None):
        """Initialize Alpha Vantage service
        
        Args:
            api_key: Alpha Vantage API key (can be set via environment variable API_KEY_ALPHA_VANTAGE)
        """
        from app.config import settings
        self.api_key = api_key or settings.alpha_vantage_api_key
        
        if not self.api_key:
            logger.warning("Alpha Vantage API key not configured")
        
        self.rate_limit_delay = 0.25  # 5 requests/min for free tier = 1 req per 12 seconds
        self.last_request_time = None
    
    async def _rate_limit_wait(self):
        """Implement rate limiting for API requests"""
        if self.last_request_time:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            if elapsed < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = datetime.now()
    
    async def get_intraday(
        self,
        symbol: str,
        interval: str = "60min",
        outputsize: str = "full"
    ) -> Dict[str, Any]:
        """
        Get intraday (minute) data
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            interval: 1min, 5min, 15min, 30min, 60min
            outputsize: "compact" (last 100) or "full" (up to 20 years)
        
        Returns:
            Dictionary with intraday data
        """
        try:
            await self._rate_limit_wait()
            
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "interval": interval,
                "outputsize": outputsize,
                "apikey": self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.BASE_URL, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Intraday data fetched for {symbol}")
                        return data
                    else:
                        logger.error(f"❌ Alpha Vantage API error: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching intraday data: {str(e)}")
            return None
    
    async def get_daily(
        self,
        symbol: str,
        outputsize: str = "full"
    ) -> Dict[str, Any]:
        """
        Get daily data (full market history)
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            outputsize: "compact" (last 100) or "full" (up to 20 years)
        
        Returns:
            Dictionary with daily data
        """
        try:
            await self._rate_limit_wait()
            
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": outputsize,
                "apikey": self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.BASE_URL, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Daily data fetched for {symbol}")
                        return data
                    else:
                        logger.error(f"❌ Alpha Vantage API error: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching daily data: {str(e)}")
            return None
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time stock quote
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
        
        Returns:
            Dictionary with current price and metadata
        """
        try:
            await self._rate_limit_wait()
            
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.BASE_URL, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "Global Quote" in data:
                            quote = data["Global Quote"]
                            logger.info(f"✅ Quote fetched for {symbol}: ${quote.get('05. price', 'N/A')}")
                            return quote
                        else:
                            logger.warning(f"No quote data for {symbol}")
                            return None
                    else:
                        logger.error(f"❌ Alpha Vantage API error: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching quote: {str(e)}")
            return None
    
    async def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """
        Get company information
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
        
        Returns:
            Dictionary with company details (sector, industry, market cap, etc.)
        """
        try:
            await self._rate_limit_wait()
            
            params = {
                "function": "OVERVIEW",
                "symbol": symbol,
                "apikey": self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.BASE_URL, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Company overview fetched for {symbol}")
                        return data
                    else:
                        logger.error(f"❌ Alpha Vantage API error: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching company overview: {str(e)}")
            return None


class StockDataService:
    """Stock data collection and management service"""
    
    def __init__(self):
        """Initialize stock data service"""
        self.alpha_vantage = AlphaVantageService()
        self.logger = logging.getLogger(__name__)
    
    async def collect_historical_data(
        self,
        symbol: str,
        days: int = 90,
        interval: str = "60min"
    ) -> List[Dict[str, Any]]:
        """
        Collect historical stock data for training
        
        Args:
            symbol: Stock symbol
            days: Number of days to collect (approximately)
            interval: Time interval (1min, 5min, 15min, 30min, 60min, 1d)
        
        Returns:
            List of OHLCV data points
        """
        try:
            self.logger.info(f"📊 Collecting historical data for {symbol}...")
            
            # For intraday: fetch full data
            if interval != "1d":
                data = await self.alpha_vantage.get_intraday(
                    symbol,
                    interval=interval,
                    outputsize="full"
                )
                
                if not data:
                    return []
                
                # Parse time series data
                time_series_key = f"Time Series ({interval})"
                if time_series_key not in data:
                    self.logger.warning(f"No time series data for {symbol}")
                    return []
                
                candles = []
                for timestamp, values in data[time_series_key].items():
                    candle = {
                        "timestamp": timestamp,
                        "open": float(values["1. open"]),
                        "high": float(values["2. high"]),
                        "low": float(values["3. low"]),
                        "close": float(values["4. close"]),
                        "volume": int(values["5. volume"])
                    }
                    candles.append(candle)
                
                self.logger.info(f"✅ Collected {len(candles)} {interval} candles for {symbol}")
                return candles
            
            # For daily: fetch daily data
            else:
                data = await self.alpha_vantage.get_daily(symbol, outputsize="full")
                
                if not data:
                    return []
                
                time_series_key = "Time Series (Daily)"
                if time_series_key not in data:
                    self.logger.warning(f"No daily data for {symbol}")
                    return []
                
                candles = []
                for date, values in data[time_series_key].items():
                    candle = {
                        "timestamp": date,
                        "open": float(values["1. open"]),
                        "high": float(values["2. high"]),
                        "low": float(values["3. low"]),
                        "close": float(values["4. close"]),
                        "volume": int(values["5. volume"])
                    }
                    candles.append(candle)
                
                self.logger.info(f"✅ Collected {len(candles)} daily candles for {symbol}")
                return candles
        
        except Exception as e:
            self.logger.error(f"Error collecting historical data: {str(e)}")
            return []
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current stock price"""
        try:
            quote = await self.alpha_vantage.get_quote(symbol)
            if quote:
                return float(quote.get("05. price", 0))
            return None
        except Exception as e:
            self.logger.error(f"Error getting current price: {str(e)}")
            return None
    
    async def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get stock company information"""
        try:
            overview = await self.alpha_vantage.get_company_overview(symbol)
            if overview:
                return {
                    "symbol": symbol,
                    "name": overview.get("Name", symbol),
                    "sector": overview.get("Sector"),
                    "industry": overview.get("Industry"),
                    "market_cap": overview.get("MarketCapitalization"),
                    "pe_ratio": overview.get("PERatio"),
                    "dividend_yield": overview.get("DividendYield"),
                    "description": overview.get("Description")
                }
            return None
        except Exception as e:
            self.logger.error(f"Error getting stock info: {str(e)}")
            return None
