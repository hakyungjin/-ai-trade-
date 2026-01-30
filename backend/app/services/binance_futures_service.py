"""
Binance Futures API 서비스
- 선물 시장 데이터 조회
- 선물 심볼 검색
"""

from binance.client import Client
from binance.enums import *
from typing import Optional, List, Dict, Any
import asyncio
from functools import partial
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BinanceFuturesService:
    """Binance 선물 시장 전용 서비스"""
    
    def __init__(self, api_key: str = "", secret_key: str = ""):
        self.api_key = api_key
        self.secret_key = secret_key
        
        # 선물 마켓 데이터용 클라이언트 (인증 불필요)
        self.client = Client("", "")
        # Futures API URL 설정
        self.futures_base_url = "https://fapi.binance.com"
    
    async def _run_sync(self, func, *args, **kwargs):
        """동기 함수를 비동기로 실행"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(func, *args, **kwargs))
    
    async def get_futures_exchange_info(self) -> Dict[str, Any]:
        """선물 거래소 정보 조회"""
        try:
            info = await self._run_sync(self.client.futures_exchange_info)
            return info
        except Exception as e:
            logger.error(f"Failed to get futures exchange info: {e}")
            raise
    
    async def get_futures_ticker_24h(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """선물 24시간 티커 데이터 조회"""
        try:
            if symbol:
                ticker = await self._run_sync(self.client.futures_ticker, symbol=symbol)
                tickers = [ticker] if isinstance(ticker, dict) else ticker
            else:
                tickers = await self._run_sync(self.client.futures_ticker)
            
            return [
                {
                    "symbol": t["symbol"],
                    "price": float(t.get("lastPrice", 0)),
                    "priceChange": float(t.get("priceChange", 0)),
                    "priceChangePercent": float(t.get("priceChangePercent", 0)),
                    "highPrice": float(t.get("highPrice", 0)),
                    "lowPrice": float(t.get("lowPrice", 0)),
                    "volume": float(t.get("volume", 0)),
                    "quoteVolume": float(t.get("quoteVolume", 0)),
                    "openPrice": float(t.get("openPrice", 0)),
                    "weightedAvgPrice": float(t.get("weightedAvgPrice", 0)),
                }
                for t in tickers
            ]
        except Exception as e:
            logger.error(f"Failed to get futures ticker: {e}")
            raise
    
    async def get_futures_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 100,
        startTime: Optional[int] = None,
        endTime: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """선물 캔들 데이터 조회"""
        interval_map = {
            "1m": KLINE_INTERVAL_1MINUTE,
            "3m": KLINE_INTERVAL_3MINUTE,
            "5m": KLINE_INTERVAL_5MINUTE,
            "15m": KLINE_INTERVAL_15MINUTE,
            "30m": KLINE_INTERVAL_30MINUTE,
            "1h": KLINE_INTERVAL_1HOUR,
            "2h": KLINE_INTERVAL_2HOUR,
            "4h": KLINE_INTERVAL_4HOUR,
            "6h": KLINE_INTERVAL_6HOUR,
            "8h": KLINE_INTERVAL_8HOUR,
            "12h": KLINE_INTERVAL_12HOUR,
            "1d": KLINE_INTERVAL_1DAY,
            "1w": KLINE_INTERVAL_1WEEK,
        }
        
        kline_interval = interval_map.get(interval, KLINE_INTERVAL_1HOUR)
        
        kwargs = {
            "symbol": symbol,
            "interval": kline_interval,
            "limit": limit
        }
        if startTime:
            kwargs["startTime"] = startTime
        if endTime:
            kwargs["endTime"] = endTime
        
        try:
            klines = await self._run_sync(self.client.futures_klines, **kwargs)
            
            return [
                {
                    "timestamp": k[0],
                    "open_time": datetime.fromtimestamp(k[0] / 1000).isoformat(),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_time": datetime.fromtimestamp(k[6] / 1000).isoformat(),
                    "quote_volume": float(k[7]),
                    "trades_count": k[8]
                }
                for k in klines
            ]
        except Exception as e:
            logger.error(f"Failed to get futures klines: {e}")
            raise
    
    async def get_all_futures_symbols(self) -> List[Dict[str, Any]]:
        """모든 USDT 선물 거래쌍 조회"""
        try:
            info = await self.get_futures_exchange_info()
            
            symbols = []
            for s in info.get("symbols", []):
                if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT":
                    symbols.append({
                        "symbol": s["symbol"],
                        "baseAsset": s["baseAsset"],
                        "quoteAsset": s["quoteAsset"],
                        "status": s["status"],
                        "contractType": s.get("contractType", "PERPETUAL"),
                    })
            
            return symbols
        except Exception as e:
            logger.error(f"Failed to get all futures symbols: {e}")
            raise
    
    async def search_futures_symbols(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """선물 심볼 검색"""
        try:
            info = await self.get_futures_exchange_info()
            tickers = await self.get_futures_ticker_24h()
            ticker_map = {t["symbol"]: t for t in tickers}
            
            query_upper = query.upper()
            symbols = []
            
            for s in info.get("symbols", []):
                if (s.get("status") == "TRADING" and 
                    s.get("quoteAsset") == "USDT" and
                    (query_upper in s["symbol"] or query_upper in s["baseAsset"])):
                    
                    ticker = ticker_map.get(s["symbol"])
                    if ticker:
                        symbols.append({
                            "symbol": s["symbol"],
                            "baseAsset": s["baseAsset"],
                            "quoteAsset": s["quoteAsset"],
                            "contractType": s.get("contractType", "PERPETUAL"),
                            "price": ticker.get("price", 0),
                            "priceChange": ticker.get("priceChange", 0),
                            "priceChangePercent": ticker.get("priceChangePercent", 0),
                            "volume": ticker.get("quoteVolume", 0),
                            "trend": "up" if ticker.get("priceChangePercent", 0) > 0 else "down" if ticker.get("priceChangePercent", 0) < 0 else "neutral"
                        })
            
            return symbols[:limit]
        except Exception as e:
            logger.error(f"Failed to search futures symbols: {e}")
            raise
    
    async def get_futures_symbols_with_ticker(self) -> List[Dict[str, Any]]:
        """모든 선물 거래쌍과 시세 정보 조회"""
        try:
            info = await self.get_futures_exchange_info()
            tickers = await self.get_futures_ticker_24h()
            ticker_map = {t["symbol"]: t for t in tickers}
            
            symbols = []
            for s in info.get("symbols", []):
                if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT":
                    ticker = ticker_map.get(s["symbol"])
                    if ticker:
                        symbols.append({
                            "symbol": s["symbol"],
                            "baseAsset": s["baseAsset"],
                            "quoteAsset": s["quoteAsset"],
                            "contractType": s.get("contractType", "PERPETUAL"),
                            "price": ticker.get("price", 0),
                            "priceChange": ticker.get("priceChange", 0),
                            "priceChangePercent": ticker.get("priceChangePercent", 0),
                            "volume": ticker.get("quoteVolume", 0),
                            "trend": "up" if ticker.get("priceChangePercent", 0) > 0 else "down" if ticker.get("priceChangePercent", 0) < 0 else "neutral"
                        })
            
            return symbols
        except Exception as e:
            logger.error(f"Failed to get futures symbols with ticker: {e}")
            raise
    
    async def get_top_futures_by_volume(self, limit: int = 50) -> List[Dict[str, Any]]:
        """거래량 기준 상위 선물 심볼 조회"""
        try:
            symbols = await self.get_futures_symbols_with_ticker()
            symbols.sort(key=lambda x: x.get("volume", 0), reverse=True)
            return symbols[:limit]
        except Exception as e:
            logger.error(f"Failed to get top futures by volume: {e}")
            raise


# 싱글톤 인스턴스
_futures_service: Optional[BinanceFuturesService] = None


def get_futures_service() -> BinanceFuturesService:
    """선물 서비스 인스턴스 반환"""
    global _futures_service
    if _futures_service is None:
        _futures_service = BinanceFuturesService()
    return _futures_service





