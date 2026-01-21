"""
실시간 데이터 스트리밍 및 AI 분석 WebSocket
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List, Set
import asyncio
import json
from datetime import datetime

from app.services.binance_service import BinanceService
from app.services.gemini_service import GeminiService
from app.config import get_settings

router = APIRouter()

# 연결된 클라이언트 관리
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[WebSocket, Set[str]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[websocket] = set()

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]

    def subscribe(self, websocket: WebSocket, symbol: str):
        if websocket in self.subscriptions:
            self.subscriptions[websocket].add(symbol)

    def unsubscribe(self, websocket: WebSocket, symbol: str):
        if websocket in self.subscriptions:
            self.subscriptions[websocket].discard(symbol)

    async def broadcast(self, message: dict, symbol: str = None):
        """특정 심볼 구독자 또는 전체에게 메시지 전송"""
        for connection in self.active_connections:
            try:
                if symbol is None or symbol in self.subscriptions.get(connection, set()):
                    await connection.send_json(message)
            except:
                pass

    async def send_personal(self, websocket: WebSocket, message: dict):
        try:
            await websocket.send_json(message)
        except:
            pass


manager = ConnectionManager()


@router.websocket("/ws/market/{symbol}")
async def websocket_market_stream(websocket: WebSocket, symbol: str):
    """
    실시간 시장 데이터 + AI 분석 스트림

    클라이언트에서:
    ws = new WebSocket('ws://localhost:8000/api/ai/ws/market/BTCUSDT')
    ws.onmessage = (event) => { console.log(JSON.parse(event.data)) }
    """
    await manager.connect(websocket)
    manager.subscribe(websocket, symbol)

    settings = get_settings()
    binance = BinanceService(
        api_key=settings.binance_api_key,
        secret_key=settings.binance_secret_key,
        testnet=settings.binance_testnet
    )
    gemini = GeminiService(api_key=settings.gemini_api_key)

    # 분석 간격 (초)
    PRICE_UPDATE_INTERVAL = 5  # 가격은 5초마다
    AI_ANALYSIS_INTERVAL = 60  # AI 분석은 60초마다

    last_ai_analysis = 0

    try:
        while True:
            current_time = asyncio.get_event_loop().time()

            try:
                # 현재가 조회
                price_data = await binance.get_current_price(symbol)
                current_price = price_data.get("price", 0)

                # 가격 업데이트 전송
                await manager.send_personal(websocket, {
                    "type": "price",
                    "symbol": symbol,
                    "price": current_price,
                    "timestamp": datetime.now().isoformat()
                })

                # AI 분석 (60초마다)
                if current_time - last_ai_analysis >= AI_ANALYSIS_INTERVAL:
                    last_ai_analysis = current_time

                    # 캔들 데이터 수집
                    candles = await binance.get_klines(
                        symbol=symbol,
                        interval="1h",
                        limit=100
                    )

                    # Gemini 분석 (API 키가 있을 때만)
                    if settings.gemini_api_key:
                        analysis = await gemini.analyze_chart(
                            symbol=symbol,
                            candles=candles,
                            current_price=current_price,
                            timeframe="1h"
                        )

                        await manager.send_personal(websocket, {
                            "type": "analysis",
                            "symbol": symbol,
                            "data": analysis,
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        # 기본 기술적 분석
                        await manager.send_personal(websocket, {
                            "type": "analysis",
                            "symbol": symbol,
                            "data": {
                                "signal": "HOLD",
                                "confidence": 0.0,
                                "analysis": "Gemini API 키를 설정하면 AI 분석을 받을 수 있습니다"
                            },
                            "timestamp": datetime.now().isoformat()
                        })

            except Exception as e:
                await manager.send_personal(websocket, {
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                })

            await asyncio.sleep(PRICE_UPDATE_INTERVAL)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        manager.disconnect(websocket)


@router.websocket("/ws/multi")
async def websocket_multi_stream(websocket: WebSocket):
    """
    여러 심볼 동시 구독

    클라이언트에서:
    ws.send(JSON.stringify({action: 'subscribe', symbols: ['BTCUSDT', 'ETHUSDT']}))
    ws.send(JSON.stringify({action: 'unsubscribe', symbols: ['ETHUSDT']}))
    """
    await manager.connect(websocket)

    settings = get_settings()
    binance = BinanceService(
        api_key=settings.binance_api_key,
        secret_key=settings.binance_secret_key,
        testnet=settings.binance_testnet
    )

    subscribed_symbols: Set[str] = set()

    async def price_updater():
        """가격 업데이트 태스크"""
        while True:
            for symbol in list(subscribed_symbols):
                try:
                    price_data = await binance.get_current_price(symbol)
                    await manager.send_personal(websocket, {
                        "type": "price",
                        "symbol": symbol,
                        "price": price_data.get("price", 0),
                        "timestamp": datetime.now().isoformat()
                    })
                except:
                    pass
            await asyncio.sleep(3)

    # 가격 업데이트 태스크 시작
    price_task = asyncio.create_task(price_updater())

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            action = message.get("action")
            symbols = message.get("symbols", [])

            if action == "subscribe":
                for s in symbols:
                    subscribed_symbols.add(s.upper())
                    manager.subscribe(websocket, s.upper())
                await manager.send_personal(websocket, {
                    "type": "subscribed",
                    "symbols": list(subscribed_symbols)
                })

            elif action == "unsubscribe":
                for s in symbols:
                    subscribed_symbols.discard(s.upper())
                    manager.unsubscribe(websocket, s.upper())
                await manager.send_personal(websocket, {
                    "type": "unsubscribed",
                    "symbols": symbols
                })

            elif action == "analyze":
                # 특정 심볼 즉시 분석 요청
                symbol = message.get("symbol", "").upper()
                if symbol and settings.gemini_api_key:
                    gemini = GeminiService(api_key=settings.gemini_api_key)
                    candles = await binance.get_klines(symbol=symbol, interval="1h", limit=100)
                    price_data = await binance.get_current_price(symbol)

                    analysis = await gemini.analyze_chart(
                        symbol=symbol,
                        candles=candles,
                        current_price=price_data.get("price", 0),
                        timeframe="1h"
                    )

                    await manager.send_personal(websocket, {
                        "type": "analysis",
                        "symbol": symbol,
                        "data": analysis,
                        "timestamp": datetime.now().isoformat()
                    })

    except WebSocketDisconnect:
        price_task.cancel()
        manager.disconnect(websocket)
    except Exception as e:
        price_task.cancel()
        manager.disconnect(websocket)
