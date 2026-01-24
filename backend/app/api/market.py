"""
실시간 마켓 데이터 API - 바이낸스 심볼 상승/하락 조회
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from typing import List, Optional
import asyncio
from datetime import datetime
import logging
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.binance_service import BinanceService
from app.services.binance_stream import binance_stream_manager
from app.services.market_data_service import MarketDataService
from app.services.unified_data_service import UnifiedDataService
from app.config import get_settings
from app.database import get_db, AsyncSessionLocal

logger = logging.getLogger(__name__)
router = APIRouter()


def get_binance_service() -> BinanceService:
    """바이낸스 서비스 인스턴스 생성"""
    settings = get_settings()
    return BinanceService(
        api_key=settings.binance_api_key,
        secret_key=settings.binance_secret_key,
        testnet=settings.binance_testnet
    )


@router.get("/tickers")
async def get_all_tickers():
    """
    모든 USDT 페어의 24시간 티커 데이터 조회
    """
    binance = get_binance_service()
    tickers = await binance.get_ticker_24h()

    # USDT 페어만 필터링
    usdt_tickers = [t for t in tickers if t["symbol"].endswith("USDT")]

    # 거래량 순으로 정렬
    sorted_tickers = sorted(usdt_tickers, key=lambda x: x["quoteVolume"], reverse=True)

    return {
        "success": True,
        "data": sorted_tickers[:100],  # 상위 100개
        "total": len(sorted_tickers),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/ticker/{symbol}")
async def get_ticker(symbol: str, market_type: str = Query(default="spot")):
    """
    특정 심볼의 24시간 티커 데이터 조회 (현물/선물 지원)
    
    - **symbol**: 심볼 (예: BTCUSDT)
    - **market_type**: 마켓 타입 (spot 또는 futures)
    """
    binance = get_binance_service()
    market_type = market_type.lower()
    try:
        tickers = await binance.get_ticker_24h(symbol.upper(), market_type=market_type)
        if tickers:
            ticker = tickers[0]
            return {
                "success": True,
                "data": ticker,
                "trend": "up" if ticker["priceChangePercent"] > 0 else "down" if ticker["priceChangePercent"] < 0 else "neutral",
                "timestamp": datetime.now().isoformat()
            }
        return {"success": False, "error": "Ticker not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/gainers-losers")
async def get_gainers_losers(limit: int = Query(default=10, ge=1, le=50)):
    """
    상승/하락 상위 코인 조회

    - **limit**: 조회할 코인 수 (기본 10, 최대 50)
    """
    binance = get_binance_service()
    data = await binance.get_top_gainers_losers(limit=limit)

    return {
        "success": True,
        "gainers": data["gainers"],
        "losers": data["losers"],
        "timestamp": datetime.now().isoformat()
    }


@router.get("/trending")
async def get_trending_coins(limit: int = Query(default=20, ge=1, le=100)):
    """
    트렌딩 코인 (거래량 + 변동성 기준)
    """
    binance = get_binance_service()
    tickers = await binance.get_ticker_24h()

    # USDT 페어만
    usdt_tickers = [t for t in tickers if t["symbol"].endswith("USDT")]

    # 거래량 최소 기준
    filtered = [t for t in usdt_tickers if t["quoteVolume"] > 1000000]

    # 변동성(절대값) + 거래량으로 트렌딩 점수 계산
    for t in filtered:
        volatility = abs(t["priceChangePercent"])
        volume_score = min(t["quoteVolume"] / 1000000000, 10)  # 최대 10점
        t["trendScore"] = volatility * 0.7 + volume_score * 0.3

    # 트렌딩 점수 순 정렬
    trending = sorted(filtered, key=lambda x: x["trendScore"], reverse=True)[:limit]

    return {
        "success": True,
        "data": trending,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/search")
async def search_symbols(query: str = Query(..., min_length=1)):
    """
    심볼 검색
    """
    binance = get_binance_service()
    symbols = await binance.search_symbols(query)

    # 검색된 심볼들의 티커 정보도 함께 가져오기
    if symbols:
        tickers = await binance.get_ticker_24h()
        ticker_map = {t["symbol"]: t for t in tickers}

        results = []
        for s in symbols:
            ticker = ticker_map.get(s["symbol"])
            if ticker:
                results.append({
                    **s,
                    "price": ticker["price"],
                    "priceChange": ticker["priceChange"],
                    "priceChangePercent": ticker["priceChangePercent"],
                    "volume": ticker["quoteVolume"],
                    "trend": "up" if ticker["priceChangePercent"] > 0 else "down" if ticker["priceChangePercent"] < 0 else "neutral"
                })

        return {
            "success": True,
            "data": results,
            "total": len(results),
            "timestamp": datetime.now().isoformat()
        }

    return {
        "success": True,
        "data": [],
        "total": 0,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/search/all")
async def search_all_symbols(
    query: str = Query(..., min_length=1),
    limit: int = Query(100, ge=1, le=500),
    quote_asset: str = Query("USDT", pattern="^[A-Z]+$")
):
    """
    모든 심볼 검색 (알트코인 포함)
    
    Args:
        query: 검색어 (예: BTC, ETH, DOGE)
        limit: 최대 결과 개수 (1-500)
        quote_asset: 쿼트 자산 (기본값: USDT)
    
    Returns:
        검색된 심볼 목록 (시세 정보 포함)
    """
    binance = get_binance_service()
    symbols = await binance.search_symbols_advanced(query, quote_asset=quote_asset, limit=limit)

    return {
        "success": True,
        "data": symbols,
        "total": len(symbols),
        "query": query,
        "quoteAsset": quote_asset,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/search/trending")
async def get_trending_symbols(
    limit: int = Query(50, ge=1, le=500),
    quote_asset: str = Query("USDT", pattern="^[A-Z]+$")
):
    """
    거래량 기준 상위 심볼 조회 (트렌딩 코인)
    
    Args:
        limit: 최대 결과 개수 (1-500)
        quote_asset: 쿼트 자산 (기본값: USDT)
    
    Returns:
        거래량 상위 심볼 목록
    """
    binance = get_binance_service()
    symbols = await binance.get_top_symbols_by_volume(limit=limit, quote_asset=quote_asset)

    return {
        "success": True,
        "data": symbols,
        "total": len(symbols),
        "quoteAsset": quote_asset,
        "sortBy": "volume",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/search/altcoins")
async def search_altcoins(
    query: str = Query("", min_length=0),
    limit: int = Query(100, ge=1, le=500),
    exclude_major: bool = Query(True)
):
    """
    알트코인 검색 (BTC, ETH 제외 옵션)
    
    Args:
        query: 검색어 (빈 문자열 = 모든 알트코인)
        limit: 최대 결과 개수
        exclude_major: 메이저 코인 제외 여부
    
    Returns:
        알트코인 목록
    """
    binance = get_binance_service()
    symbols = await binance.search_symbols_advanced(query or "", limit=limit)
    
    # 메이저 코인 제외 필터
    if exclude_major:
        major_coins = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        symbols = [s for s in symbols if s['symbol'] not in major_coins]
    
    # 가격 변동률로 정렬 (상위부터 내림차순)
    symbols.sort(key=lambda x: abs(x.get('priceChangePercent', 0)), reverse=True)

    return {
        "success": True,
        "data": symbols,
        "total": len(symbols),
        "query": query or "all",
        "excludeMajor": exclude_major,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/symbols/list")
async def get_all_symbols(
    quote_asset: str = Query("USDT", pattern="^[A-Z]+$"),
    limit: int = Query(500, ge=1, le=2000)
):
    """
    모든 심볼 목록 조회 (페이징 포함)
    
    Args:
        quote_asset: 쿼트 자산 (기본값: USDT)
        limit: 최대 결과 개수
    
    Returns:
        전체 심볼 목록
    """
    binance = get_binance_service()
    all_symbols = await binance.get_all_symbols_with_ticker()
    
    # 필터링
    filtered = [s for s in all_symbols if s['quoteAsset'] == quote_asset]
    
    return {
        "success": True,
        "data": filtered[:limit],
        "total": len(filtered),
        "quoteAsset": quote_asset,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/overview")
async def get_market_overview():
    """
    마켓 전체 개요 (BTC, ETH 등 주요 코인 + 전체 통계)
    """
    binance = get_binance_service()
    tickers = await binance.get_ticker_24h()

    # USDT 페어만
    usdt_tickers = [t for t in tickers if t["symbol"].endswith("USDT")]

    # 주요 코인
    major_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT", "DOTUSDT"]
    major_coins = [t for t in usdt_tickers if t["symbol"] in major_symbols]

    # 전체 통계
    gainers_count = len([t for t in usdt_tickers if t["priceChangePercent"] > 0])
    losers_count = len([t for t in usdt_tickers if t["priceChangePercent"] < 0])
    total_volume = sum(t["quoteVolume"] for t in usdt_tickers)

    # 상위 상승/하락
    sorted_by_change = sorted(usdt_tickers, key=lambda x: x["priceChangePercent"], reverse=True)
    top_gainer = sorted_by_change[0] if sorted_by_change else None
    top_loser = sorted_by_change[-1] if sorted_by_change else None

    return {
        "success": True,
        "majorCoins": major_coins,
        "stats": {
            "totalCoins": len(usdt_tickers),
            "gainersCount": gainers_count,
            "losersCount": losers_count,
            "totalVolume": total_volume,
            "topGainer": top_gainer,
            "topLoser": top_loser
        },
        "timestamp": datetime.now().isoformat()
    }


# WebSocket 연결 관리
class KlinesConnectionManager:
    """
    심볼별/인터벌별 WebSocket 연결을 클라이언트별로 관리
    같은 클라이언트가 새로운 연결을 하면 이전 연결을 자동 종료
    """
    def __init__(self):
        # {client_id: {symbol_interval: websocket}}
        self.active_connections: dict = {}
        self.connection_lock = asyncio.Lock()

    async def connect(self, client_id: str, symbol: str, interval: str, websocket: WebSocket):
        """
        클라이언트의 새로운 연결 등록 (이전 연결은 자동 종료)
        """
        await websocket.accept()
        
        async with self.connection_lock:
            symbol_interval = f"{symbol}_{interval}"
            
            # 클라이언트가 없으면 생성
            if client_id not in self.active_connections:
                self.active_connections[client_id] = {}
            
            # 같은 심볼/인터벌의 이전 연결 종료 (async 작업이므로 spawn으로 처리)
            if symbol_interval in self.active_connections[client_id]:
                old_ws = self.active_connections[client_id][symbol_interval]
                # 비동기로 close 작업 실행 (메인 루프를 블로킹하지 않음)
                try:
                    # 이미 accepted 상태인 연결은 close 대신 send_json으로 종료 신호 전송
                    await old_ws.send_json({
                        "type": "close",
                        "reason": "New connection established for same symbol/interval"
                    })
                except:
                    pass
                
                print(f"Closed previous connection: {client_id} {symbol_interval}")
            
            # 새 연결 저장
            self.active_connections[client_id][symbol_interval] = websocket
            print(f"Connected: {client_id} {symbol_interval}")

    async def disconnect(self, client_id: str, symbol: str, interval: str):
        """연결 해제"""
        async with self.connection_lock:
            symbol_interval = f"{symbol}_{interval}"
            if client_id in self.active_connections:
                if symbol_interval in self.active_connections[client_id]:
                    del self.active_connections[client_id][symbol_interval]
                    print(f"Disconnected: {client_id} {symbol_interval}")
                
                # 클라이언트의 모든 연결이 종료되면 클라이언트 제거
                if not self.active_connections[client_id]:
                    del self.active_connections[client_id]

    def get_connection_count(self, client_id: str) -> int:
        """클라이언트의 활성 연결 수"""
        return len(self.active_connections.get(client_id, {}))


klines_manager = KlinesConnectionManager()


@router.websocket("/ws/tickers")
async def websocket_all_tickers(websocket: WebSocket):
    """
    전체 마켓 티커 실시간 스트림 (5초마다 업데이트)

    주요 코인 + 상승/하락 상위 코인 전송
    """
    await market_manager.connect(websocket)
    binance = get_binance_service()

    UPDATE_INTERVAL = 5  # 5초마다 업데이트

    try:
        while True:
            try:
                tickers = await binance.get_ticker_24h()
                usdt_tickers = [t for t in tickers if t["symbol"].endswith("USDT")]

                # 거래량 필터
                filtered = [t for t in usdt_tickers if t["quoteVolume"] > 100000]

                # 정렬
                sorted_by_change = sorted(filtered, key=lambda x: x["priceChangePercent"], reverse=True)

                # 주요 코인
                major_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"]
                major_coins = [t for t in usdt_tickers if t["symbol"] in major_symbols]

                message = {
                    "type": "market_update",
                    "majorCoins": major_coins,
                    "gainers": sorted_by_change[:10],
                    "losers": sorted_by_change[-10:][::-1],
                    "stats": {
                        "gainersCount": len([t for t in filtered if t["priceChangePercent"] > 0]),
                        "losersCount": len([t for t in filtered if t["priceChangePercent"] < 0]),
                    },
                    "timestamp": datetime.now().isoformat()
                }

                await websocket.send_json(message)

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                })

            await asyncio.sleep(UPDATE_INTERVAL)

    except WebSocketDisconnect:
        market_manager.disconnect(websocket)
    except Exception:
        market_manager.disconnect(websocket)


@router.websocket("/ws/ticker/{symbol}")
async def websocket_single_ticker(websocket: WebSocket, symbol: str):
    """
    단일 심볼 실시간 티커 스트림 (3초마다 업데이트)
    """
    await websocket.accept()
    binance = get_binance_service()
    symbol = symbol.upper()

    UPDATE_INTERVAL = 3
    previous_price = None

    try:
        while True:
            try:
                tickers = await binance.get_ticker_24h(symbol)
                if tickers:
                    ticker = tickers[0]
                    current_price = ticker["price"]

                    # 이전 가격 대비 변화 방향
                    instant_trend = "neutral"
                    if previous_price is not None:
                        if current_price > previous_price:
                            instant_trend = "up"
                        elif current_price < previous_price:
                            instant_trend = "down"

                    previous_price = current_price

                    message = {
                        "type": "ticker_update",
                        "symbol": symbol,
                        "data": ticker,
                        "trend24h": "up" if ticker["priceChangePercent"] > 0 else "down" if ticker["priceChangePercent"] < 0 else "neutral",
                        "instantTrend": instant_trend,
                        "timestamp": datetime.now().isoformat()
                    }

                    await websocket.send_json(message)

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                })

            await asyncio.sleep(UPDATE_INTERVAL)

    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ===== 차트 데이터 (OHLCV) API =====

@router.get("/klines/{symbol}")
async def get_klines(
    symbol: str,
    interval: str = Query(default="1h", pattern="^(1m|5m|15m|30m|1h|4h|1d|1w)$"),
    limit: int = Query(default=100, ge=1, le=1000),
    market_type: str = Query(default="spot"),
    db: AsyncSession = Depends(get_db)
):
    """
    캔들스틱 차트 데이터 조회 (OHLCV) - 현물/선물 지원
    - DB에서 먼저 데이터 조회 (현물만)
    - 선물은 바이낸스 API에서 직접 조회
    - 새로 가져온 데이터는 자동으로 DB에 저장

    - **symbol**: 심볼 (예: BTCUSDT)
    - **interval**: 캔들 간격 (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)
    - **limit**: 조회할 캔들 수 (최대 1000)
    - **market_type**: 마켓 타입 (spot 또는 futures)
    """
    binance = get_binance_service()
    symbol = symbol.upper()
    market_type = market_type.lower()
    logger.info(f"📊 Fetching klines for {symbol} {interval} (limit: {limit}, market: {market_type})")

    try:
        if market_type == 'futures':
            # 선물은 바이낸스 API에서 직접 조회
            klines = await binance.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit,
                market_type='futures'
            )
            source = "binance_futures"
        else:
            # 현물은 DB 캐시 + 증분 수집
            unified_service = UnifiedDataService(db, binance)
            klines = await unified_service.get_klines_with_cache(
                symbol=symbol,
                timeframe=interval,
                limit=limit
            )
            source = "db_cache"
        
        logger.info(f"✅ Retrieved {len(klines)} candles for {symbol} {interval} ({source})")

        return {
            "success": True,
            "symbol": symbol,
            "interval": interval,
            "data": klines,
            "count": len(klines),
            "timestamp": datetime.now().isoformat(),
            "source": source
        }
    except Exception as e:
        logger.error(f"❌ Error fetching klines: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.websocket("/ws/klines/{symbol}")
async def websocket_klines(websocket: WebSocket, symbol: str, interval: str = "1m"):
    """
    실시간 캔들스틱 차트 스트림
    """
    binance = get_binance_service()
    symbol = symbol.upper()
    
    # 클라이언트 ID (IP + port로 고유성 확보)
    client_id = f"{websocket.client.host}:{websocket.client.port}"

    try:
        # 연결 관리자에 등록 (이전 연결 자동 종료)
        await klines_manager.connect(client_id, symbol, interval, websocket)
        
        print(f"WebSocket opened: {client_id} - {symbol} {interval}")

        # 간격에 따른 업데이트 주기
        update_intervals = {
            "1m": 10,
            "5m": 30,
            "15m": 30,
            "30m": 60,
            "1h": 60,
            "4h": 120,
            "1d": 300,
            "1w": 300
        }
        UPDATE_INTERVAL = update_intervals.get(interval, 60)

        # 초기 데이터 전송 (최근 200개 캔들) - DB 캐시 우선 조회
        initial_klines = None
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries and initial_klines is None:
            try:
                print(f"Loading initial klines: {symbol} {interval} (attempt {retry_count + 1}) - DB first")
                
                # DB 세션 생성하여 UnifiedDataService 사용
                async with AsyncSessionLocal() as db_session:
                    unified_service = UnifiedDataService(db_session, binance)
                    initial_klines = await unified_service.get_klines_with_cache(
                        symbol=symbol,
                        timeframe=interval,
                        limit=200
                    )
                
                print(f"Initial klines loaded: {len(initial_klines)} candles for {symbol} {interval} (from DB cache)")
                break
            except Exception as e:
                retry_count += 1
                print(f"Failed to load initial klines (attempt {retry_count}): {e}")
                if retry_count < max_retries:
                    await asyncio.sleep(1)  # 1초 대기 후 재시도
        
        if initial_klines is None:
            print(f"Failed to load initial klines after {max_retries} attempts: {symbol} {interval}")
            await websocket.send_json({
                "type": "error",
                "message": f"Failed to load initial chart data",
                "timestamp": datetime.now().isoformat()
            })
            return
        
        # 초기 데이터 전송
        await websocket.send_json({
            "type": "initial",
            "symbol": symbol,
            "interval": interval,
            "data": initial_klines,
            "count": len(initial_klines),
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"Initial data sent to {client_id}: {symbol} {interval}")

        last_candle_timestamp = None
        if initial_klines:
            last_candle_timestamp = initial_klines[-1]["timestamp"]

        # 메인 업데이트 루프
        update_error_count = 0
        max_consecutive_errors = 5
        
        while True:
            try:
                # 최신 캔들 10개 가져오기
                klines = await binance.get_klines(symbol=symbol, interval=interval, limit=10)

                if klines:
                    latest_candle = klines[-1]
                    
                    # 중복 업데이트 방지
                    if last_candle_timestamp != latest_candle["timestamp"]:
                        last_candle_timestamp = latest_candle["timestamp"]
                        previous_candle = klines[-2] if len(klines) > 1 else None

                        # 현재 티커 정보도 함께
                        try:
                            ticker_data = await binance.get_ticker_24h(symbol)
                            ticker = ticker_data[0] if ticker_data else {}
                        except Exception as ticker_error:
                            print(f"Warning: Failed to get ticker for {symbol}: {ticker_error}")
                            ticker = {}

                        await websocket.send_json({
                            "type": "update",
                            "symbol": symbol,
                            "interval": interval,
                            "latestCandle": latest_candle,
                            "previousCandle": previous_candle,
                            "recentCandles": klines[-5:],
                            "ticker": {
                                "price": ticker.get("price", 0),
                                "priceChangePercent": ticker.get("priceChangePercent", 0),
                                "volume": ticker.get("quoteVolume", 0)
                            },
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # 에러 카운트 리셋
                        update_error_count = 0

            except asyncio.CancelledError:
                print(f"WebSocket cancelled: {client_id} {symbol} {interval}")
                break
            except Exception as e:
                update_error_count += 1
                import traceback
                error_detail = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                print(f"Error in update loop ({update_error_count}/{max_consecutive_errors}): {client_id} {symbol} {interval} - {error_detail}")
                
                # 연속 에러 초과 시 연결 종료
                if update_error_count >= max_consecutive_errors:
                    print(f"Too many consecutive errors - closing connection: {client_id} {symbol} {interval}")
                    try:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Too many consecutive errors: {str(e)}",
                            "timestamp": datetime.now().isoformat()
                        })
                    except:
                        pass
                    break

            await asyncio.sleep(UPDATE_INTERVAL)

    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {client_id} {symbol} {interval}")
    except Exception as e:
        print(f"WebSocket error: {client_id} {symbol} {interval} - {e}")
    finally:
        await klines_manager.disconnect(client_id, symbol, interval)
        print(f"WebSocket closed: {client_id} {symbol} {interval} (Active: {klines_manager.get_connection_count(client_id)})")


@router.get("/mini-chart/{symbol}")
async def get_mini_chart(
    symbol: str,
    interval: str = Query(default="1h"),
    limit: int = Query(default=24),
    market_type: str = Query(default="spot")
):
    """
    미니 차트용 간단한 가격 데이터 (스파크라인) - 현물/선물 지원

    프론트엔드에서 간단한 가격 추세 그래프를 그리기 위한 데이터
    """
    binance = get_binance_service()
    symbol = symbol.upper()
    market_type = market_type.lower()

    try:
        klines = await binance.get_klines(symbol=symbol, interval=interval, limit=limit, market_type=market_type)

        # 간단한 형태로 변환 (close 가격만)
        prices = [k["close"] for k in klines]
        timestamps = [k["timestamp"] for k in klines]

        # 추세 계산
        if len(prices) >= 2:
            first_price = prices[0]
            last_price = prices[-1]
            change_percent = ((last_price - first_price) / first_price) * 100
            trend = "up" if change_percent > 0 else "down" if change_percent < 0 else "neutral"
        else:
            change_percent = 0
            trend = "neutral"

        return {
            "success": True,
            "symbol": symbol,
            "prices": prices,
            "timestamps": timestamps,
            "trend": trend,
            "changePercent": round(change_percent, 2),
            "high": max(prices) if prices else 0,
            "low": min(prices) if prices else 0,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
