"""
실시간 신호 API 엔드포인트
- 심볼 검색 및 등록
- 실시간 신호 조회
- WebSocket을 통한 실시간 업데이트
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from typing import List, Dict, Any, Optional
import asyncio
import json
import logging
from datetime import datetime

from app.services.binance_service import BinanceService
from app.services.signal_service import RealTimeSignalService
from app.config import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)

# 전역 서비스 인스턴스
signal_service: Optional[RealTimeSignalService] = None

# WebSocket 연결 관리
active_connections: List[WebSocket] = []


def get_signal_service():
    """신호 서비스 의존성"""
    global signal_service
    if signal_service is None:
        config = get_settings()
        binance = BinanceService(
            api_key=config.binance_api_key,
            secret_key=config.binance_secret_key,
            testnet=config.binance_testnet
        )
        signal_service = RealTimeSignalService(binance, use_ai=True)
    return signal_service


@router.get("/symbols/search")
async def search_symbols(
    query: str = "",
    limit: int = 50,
    service: RealTimeSignalService = Depends(get_signal_service)
):
    """
    심볼 검색

    Args:
        query: 검색어 (예: BTC, ETH)
        limit: 최대 결과 개수

    Returns:
        심볼 목록
    """
    try:
        all_symbols = await service.get_available_symbols()

        # 검색어로 필터링
        if query:
            query = query.upper()
            filtered = [
                s for s in all_symbols
                if query in s['symbol'] or query in s['baseAsset']
            ]
        else:
            filtered = all_symbols

        # 인기 코인 우선 정렬
        popular_coins = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOT', 'DOGE', 'AVAX', 'MATIC']

        def sort_key(symbol):
            base = symbol['baseAsset']
            if base in popular_coins:
                return (0, popular_coins.index(base))
            return (1, base)

        filtered.sort(key=sort_key)

        return {
            'success': True,
            'symbols': filtered[:limit],
            'total': len(filtered)
        }

    except Exception as e:
        logger.error(f"Error searching symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/symbols/add")
async def add_symbol(
    symbol: str,
    service: RealTimeSignalService = Depends(get_signal_service)
):
    """
    심볼 추가 및 모니터링 시작

    Args:
        symbol: 추가할 심볼 (예: BTCUSDT)

    Returns:
        추가 결과 및 초기 신호
    """
    try:
        # 심볼 추가
        result = await service.add_symbol(symbol)

        if not result['success']:
            raise HTTPException(status_code=400, detail=result['message'])

        # 초기 신호 생성
        signal = await service.generate_signal(symbol)

        return {
            'success': True,
            'message': f'{symbol} added and monitoring started',
            'symbol': symbol,
            'signal': signal
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding symbol: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/symbols/{symbol}")
async def remove_symbol(
    symbol: str,
    service: RealTimeSignalService = Depends(get_signal_service)
):
    """
    심볼 제거

    Args:
        symbol: 제거할 심볼

    Returns:
        제거 결과
    """
    try:
        result = await service.remove_symbol(symbol)

        if not result['success']:
            raise HTTPException(status_code=404, detail=result['message'])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing symbol: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/symbols/active")
async def get_active_symbols(
    service: RealTimeSignalService = Depends(get_signal_service)
):
    """
    활성 심볼 목록 조회

    Returns:
        활성 심볼 목록
    """
    try:
        symbols = service.get_active_symbols()
        return {
            'success': True,
            'symbols': symbols,
            'count': len(symbols)
        }

    except Exception as e:
        logger.error(f"Error getting active symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signal/{symbol}")
async def get_signal(
    symbol: str,
    service: RealTimeSignalService = Depends(get_signal_service)
):
    """
    특정 심볼의 신호 조회

    Args:
        symbol: 조회할 심볼

    Returns:
        신호 정보
    """
    try:
        signal = await service.generate_signal(symbol)
        return {
            'success': True,
            'signal': signal
        }

    except Exception as e:
        logger.error(f"Error getting signal for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals/all")
async def get_all_signals(
    service: RealTimeSignalService = Depends(get_signal_service)
):
    """
    모든 활성 심볼의 신호 조회

    Returns:
        모든 신호 정보
    """
    try:
        signals = service.get_all_cached_signals()
        return {
            'success': True,
            'signals': signals,
            'count': len(signals)
        }

    except Exception as e:
        logger.error(f"Error getting all signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/signals/update")
async def update_signals(
    service: RealTimeSignalService = Depends(get_signal_service)
):
    """
    모든 활성 심볼의 신호 업데이트

    Returns:
        업데이트 결과
    """
    try:
        await service.update_all_signals()
        signals = service.get_all_cached_signals()

        return {
            'success': True,
            'message': 'Signals updated',
            'signals': signals,
            'count': len(signals)
        }

    except Exception as e:
        logger.error(f"Error updating signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket 엔드포인트
@router.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    """
    실시간 신호 WebSocket

    주기적으로 모든 활성 심볼의 신호를 전송
    """
    await websocket.accept()
    active_connections.append(websocket)

    logger.info("WebSocket connection established")

    try:
        service = get_signal_service()

        while True:
            try:
                # 모든 신호 업데이트
                await service.update_all_signals()

                # 신호 가져오기
                signals = service.get_all_cached_signals()

                # 전송
                await websocket.send_json({
                    'type': 'signals_update',
                    'timestamp': datetime.now().isoformat(),
                    'signals': signals,
                    'count': len(signals)
                })

                # 30초마다 업데이트
                await asyncio.sleep(30)

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in WebSocket loop: {e}")
                await asyncio.sleep(5)

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)


@router.websocket("/ws/signal/{symbol}")
async def websocket_single_signal(websocket: WebSocket, symbol: str):
    """
    특정 심볼의 실시간 신호 WebSocket

    Args:
        symbol: 모니터링할 심볼
    """
    await websocket.accept()

    logger.info(f"WebSocket connection established for {symbol}")

    try:
        service = get_signal_service()

        # 심볼 추가 (아직 없는 경우)
        await service.add_symbol(symbol)

        while True:
            try:
                # 신호 생성
                signal = await service.generate_signal(symbol)

                # 전송
                await websocket.send_json({
                    'type': 'signal_update',
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'signal': signal
                })

                # 10초마다 업데이트
                await asyncio.sleep(10)

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in WebSocket loop for {symbol}: {e}")
                await asyncio.sleep(5)

    except WebSocketDisconnect:
        logger.info(f"WebSocket connection closed for {symbol}")
    except Exception as e:
        logger.error(f"WebSocket error for {symbol}: {e}")


async def broadcast_signal(signal: Dict[str, Any]):
    """
    모든 연결된 클라이언트에 신호 브로드캐스트

    Args:
        signal: 브로드캐스트할 신호
    """
    if not active_connections:
        return

    message = json.dumps({
        'type': 'signal_broadcast',
        'timestamp': datetime.now().isoformat(),
        'signal': signal
    })

    disconnected = []

    for connection in active_connections:
        try:
            await connection.send_text(message)
        except Exception as e:
            logger.error(f"Error broadcasting to connection: {e}")
            disconnected.append(connection)

    # 연결 끊긴 클라이언트 제거
    for connection in disconnected:
        if connection in active_connections:
            active_connections.remove(connection)
