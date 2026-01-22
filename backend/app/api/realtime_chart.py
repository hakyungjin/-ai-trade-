"""
실시간 차트 API - Binance WebSocket 직접 연결
지연시간 최소화를 위해 바이낸스 스트림을 직접 구독
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Optional, Dict, List
import asyncio
import logging
from datetime import datetime

from app.services.binance_stream import binance_stream_manager
from app.services.binance_service import BinanceService
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


class RealtimeConnectionManager:
    """실시간 차트 WebSocket 연결 관리"""
    
    def __init__(self):
        # {client_id: {stream_id: websocket}}
        self.connections: Dict[str, Dict[str, WebSocket]] = {}
        self.lock = asyncio.Lock()

    async def connect(self, client_id: str, stream_id: str, websocket: WebSocket):
        """클라이언트 연결 등록"""
        await websocket.accept()
        
        async with self.lock:
            if client_id not in self.connections:
                self.connections[client_id] = {}
            
            # 이전 같은 스트림 연결이 있으면 종료
            if stream_id in self.connections[client_id]:
                old_ws = self.connections[client_id][stream_id]
                try:
                    await old_ws.send_json({
                        "type": "close",
                        "reason": "New connection established"
                    })
                except:
                    pass
            
            self.connections[client_id][stream_id] = websocket
            logger.info(f"Connected: {client_id} - {stream_id}")

    async def disconnect(self, client_id: str, stream_id: str):
        """클라이언트 연결 해제"""
        async with self.lock:
            if client_id in self.connections and stream_id in self.connections[client_id]:
                del self.connections[client_id][stream_id]
                
                # 클라이언트의 모든 연결 제거 시 클라이언트 제거
                if not self.connections[client_id]:
                    del self.connections[client_id]
                
                logger.info(f"Disconnected: {client_id} - {stream_id}")

    async def send_to_client(self, client_id: str, stream_id: str, message: dict):
        """특정 클라이언트에 메시지 전송"""
        async with self.lock:
            if (client_id in self.connections and 
                stream_id in self.connections[client_id]):
                ws = self.connections[client_id][stream_id]
                try:
                    await ws.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send message: {e}")
                    return False
        return True


manager = RealtimeConnectionManager()


@router.websocket("/ws/realtime/{symbol}")
async def websocket_realtime_chart(websocket: WebSocket, symbol: str, interval: str = "1m"):
    """
    🚀 실시간 차트 WebSocket (Binance 직접 연결)
    
    - Binance 스트림을 직접 구독하여 지연시간 최소화
    - 초기 과거 데이터 로드 후 실시간 업데이트
    - 다중 클라이언트 지원
    
    Parameters:
    - symbol: 거래쌍 (BTCUSDT, ETHUSDT)
    - interval: 캔들 간격 (1m, 5m, 15m, 1h, 4h, 1d)
    """
    
    symbol = symbol.upper()
    stream_id = f"{symbol}_{interval}"
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    
    try:
        # 클라이언트 연결 등록
        await manager.connect(client_id, stream_id, websocket)
        
        logger.info(f"🔌 Real-time chart opened: {client_id} - {symbol} {interval}")
        
        # 초기 데이터 로드 (REST API)
        binance = BinanceService(
            api_key=get_settings().binance_api_key,
            secret_key=get_settings().binance_secret_key,
            testnet=get_settings().binance_testnet
        )
        
        try:
            logger.info(f"📊 Loading initial klines: {symbol} {interval}")
            initial_klines = await binance.get_klines(
                symbol=symbol, 
                interval=interval, 
                limit=200
            )
            
            # 초기 데이터 전송
            await websocket.send_json({
                "type": "initial",
                "symbol": symbol,
                "interval": interval,
                "data": initial_klines,
                "count": len(initial_klines),
                "timestamp": initial_klines[-1]["timestamp"] if initial_klines else None,
                "receivedAt": None
            })
            
            logger.info(f"📤 Sent {len(initial_klines)} initial candles to {client_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load initial klines: {e}")
            await websocket.send_json({
                "type": "error",
                "message": f"Failed to load initial data: {str(e)}"
            })
            return
        
        # Binance 스트림 콜백 정의
        async def stream_callback(chart_data: dict):
            """바이낸스 스트림에서 받은 데이터를 클라이언트에 전송"""
            await manager.send_to_client(client_id, stream_id, chart_data)
        
        # Binance 스트림 구독
        logger.info(f"🔗 Subscribing to Binance stream: {stream_id}")
        await binance_stream_manager.subscribe_kline(
            symbol=symbol,
            interval=interval,
            callback=stream_callback
        )
        
        # 웹소켓 연결 유지 (클라이언트와 서버 모두 대기)
        try:
            while True:
                # 타임아웃 30초로 클라이언트 메시지 수신 대기
                # (타임아웃되면 다시 대기 - 연결 유지)
                try:
                    message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    # 클라이언트에서 종료 신호 받음
                    if message == "close":
                        break
                except asyncio.TimeoutError:
                    # 타임아웃되어도 계속 연결 유지
                    continue
        except WebSocketDisconnect:
            logger.info(f"🔌 Client disconnected: {client_id} - {stream_id}")
        except Exception as e:
            logger.error(f"⚠️ WebSocket receive error: {e}")
        
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    
    finally:
        # 정리 작업
        await manager.disconnect(client_id, stream_id)
        
        # Binance 스트림 구독 해제
        # (다른 클라이언트가 같은 스트림을 구독하지 않으면 자동 종료)
        logger.info(f"✅ Connection closed: {client_id} - {stream_id}")
