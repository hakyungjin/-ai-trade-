"""
Binance WebSocket 스트림 서비스
바이낸스에서 실시간 캔들 데이터를 직접 수신하고 가공
"""
import asyncio
import json
import logging
from typing import Callable, Dict, Optional
from datetime import datetime
import aiohttp

logger = logging.getLogger(__name__)


class BinanceStreamManager:
    """Binance WebSocket 스트림 관리 및 캔들 데이터 처리"""
    
    def __init__(self):
        self.streams: Dict[str, asyncio.Task] = {}
        self.subscribed_callbacks: Dict[str, list] = {}
        self.base_url = "wss://stream.binance.com:9443/ws"

    async def subscribe_kline(
        self,
        symbol: str,
        interval: str,
        callback: Callable
    ) -> str:
        """
        바이낸스 캔들 스트림 구독
        
        Args:
            symbol: 거래쌍 (e.g., BTCUSDT)
            interval: 캔들 간격 (1m, 5m, 15m, 1h, 4h, 1d)
            callback: 데이터 수신 콜백 함수 (async)
            
        Returns:
            stream_id: 스트림 고유 ID
        """
        stream_id = f"{symbol.lower()}_{interval}"
        
        if stream_id not in self.subscribed_callbacks:
            self.subscribed_callbacks[stream_id] = []
        
        self.subscribed_callbacks[stream_id].append(callback)
        
        # 이미 스트림이 실행 중이면 콜백만 추가
        if stream_id in self.streams and not self.streams[stream_id].done():
            logger.info(f"Callback added to existing stream: {stream_id}")
            return stream_id
        
        # 새로운 스트림 시작
        logger.info(f"Starting Binance stream: {stream_id}")
        task = asyncio.create_task(
            self._stream_kline(symbol, interval, stream_id)
        )
        self.streams[stream_id] = task
        
        return stream_id

    async def _stream_kline(self, symbol: str, interval: str, stream_id: str):
        """바이낸스 캔들 WebSocket 스트림 실행"""
        ws_url = f"{self.base_url}/{symbol.lower()}@kline_{interval}"
        
        retry_count = 0
        max_retries = 5
        
        while retry_count < max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(ws_url) as ws:
                        logger.info(f"Connected to Binance stream: {stream_id}")
                        retry_count = 0  # 연결 성공 시 재설정
                        
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    data = json.loads(msg.data)
                                    await self._process_kline(data, stream_id)
                                except Exception as e:
                                    logger.error(f"Error processing Binance data: {e}")
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                logger.error(f"WebSocket error: {ws.exception()}")
                                break
                        
            except asyncio.CancelledError:
                logger.info(f"Stream cancelled: {stream_id}")
                break
            except Exception as e:
                retry_count += 1
                logger.warning(
                    f"Stream error (retry {retry_count}/{max_retries}): {stream_id} - {e}"
                )
                if retry_count < max_retries:
                    await asyncio.sleep(min(2 ** retry_count, 30))  # Exponential backoff

    async def _process_kline(self, data: dict, stream_id: str):
        """
        바이낸스 캔들 데이터 처리 및 콜백 호출
        
        Binance 형식:
        {
            "e": "kline",
            "E": 1672531200000,  # Event time
            "s": "BTCUSDT",
            "k": {
                "t": 1672531200000,  # Kline start time
                "T": 1672531260000,  # Kline close time
                "s": "BTCUSDT",
                "i": "1m",
                "f": 123456,  # First trade ID
                "L": 123460,  # Last trade ID
                "o": "16500.00",  # Open
                "c": "16510.00",  # Close
                "h": "16550.00",  # High
                "l": "16480.00",  # Low
                "v": "100.5",  # Volume
                "n": 5,  # Number of trades
                "x": False,  # Is this kline closed?
                "q": "1650000",  # Quote asset volume
                "V": "50.2",  # Taker buy base asset volume
                "Q": "825000",  # Taker buy quote asset volume
            }
        }
        """
        try:
            kline = data.get("k", {})
            
            # 리액트 차트용 데이터 포맷
            chart_data = {
                "type": "kline",
                "symbol": data.get("s", ""),
                "interval": kline.get("i", ""),
                "timestamp": kline.get("t", 0),
                "timestamp_ms": kline.get("T", 0),
                "open": float(kline.get("o", 0)),
                "close": float(kline.get("c", 0)),
                "high": float(kline.get("h", 0)),
                "low": float(kline.get("l", 0)),
                "volume": float(kline.get("v", 0)),
                "quoteVolume": float(kline.get("q", 0)),
                "trades": int(kline.get("n", 0)),
                "isClosed": kline.get("x", False),
                "takerBuyVolume": float(kline.get("V", 0)),
                "takerBuyQuoteVolume": float(kline.get("Q", 0)),
                "receivedAt": datetime.now().isoformat(),
            }
            
            # 등록된 모든 콜백 호출
            if stream_id in self.subscribed_callbacks:
                for callback in self.subscribed_callbacks[stream_id]:
                    try:
                        await callback(chart_data)
                    except Exception as e:
                        logger.error(f"Error in callback for {stream_id}: {e}")
        
        except Exception as e:
            logger.error(f"Error processing kline data: {e}")

    async def unsubscribe(self, stream_id: str, callback: Optional[Callable] = None):
        """
        스트림 구독 해제
        
        Args:
            stream_id: 스트림 ID
            callback: 특정 콜백만 제거 (None이면 전체 제거)
        """
        if stream_id not in self.subscribed_callbacks:
            return
        
        if callback:
            if callback in self.subscribed_callbacks[stream_id]:
                self.subscribed_callbacks[stream_id].remove(callback)
        else:
            self.subscribed_callbacks[stream_id] = []
        
        # 콜백이 모두 제거되면 스트림 종료
        if not self.subscribed_callbacks[stream_id]:
            if stream_id in self.streams:
                self.streams[stream_id].cancel()
                del self.streams[stream_id]
            del self.subscribed_callbacks[stream_id]
            logger.info(f"Stream unsubscribed: {stream_id}")

    async def shutdown(self):
        """모든 스트림 종료"""
        for stream_id, task in list(self.streams.items()):
            if not task.done():
                task.cancel()
        
        # 모든 작업 완료 대기
        if self.streams:
            await asyncio.gather(*self.streams.values(), return_exceptions=True)
        
        logger.info("All Binance streams shut down")


# 글로벌 스트림 매니저
binance_stream_manager = BinanceStreamManager()
