from binance.client import Client
from binance.enums import *
from typing import Optional, List, Dict, Any
import asyncio
from functools import partial
from datetime import datetime


class BinanceService:
    def __init__(self, api_key: str, secret_key: str, testnet: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet

        # 클라이언트 초기화
        self.client = Client(api_key, secret_key, testnet=testnet)

        if testnet:
            # 테스트넷 URL 설정
            self.client.API_URL = "https://testnet.binance.vision/api"

        # 마켓 데이터용 클라이언트 (실제 바이낸스 API - 인증 불필요)
        self.market_client = Client("", "", testnet=False)

    async def _run_sync(self, func, *args, **kwargs):
        """동기 함수를 비동기로 실행"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(func, *args, **kwargs))

    async def get_current_price(self, symbol: str) -> Dict[str, Any]:
        """현재가 조회"""
        ticker = await self._run_sync(self.client.get_symbol_ticker, symbol=symbol)
        return {
            "symbol": ticker["symbol"],
            "price": float(ticker["price"])
        }

    async def get_balance(self) -> List[Dict[str, Any]]:
        """잔고 조회"""
        account = await self._run_sync(self.client.get_account)
        balances = []
        for balance in account["balances"]:
            free = float(balance["free"])
            locked = float(balance["locked"])
            if free > 0 or locked > 0:
                balances.append({
                    "asset": balance["asset"],
                    "free": free,
                    "locked": locked,
                    "total": free + locked
                })
        return balances

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """주문 생성"""
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity
        }

        if order_type == "LIMIT" and price:
            params["price"] = price
            params["timeInForce"] = TIME_IN_FORCE_GTC

        order = await self._run_sync(self.client.create_order, **params)
        return order

    async def create_stop_loss(
        self,
        symbol: str,
        quantity: float,
        stop_price: float
    ) -> Dict[str, Any]:
        """스탑로스 주문 생성"""
        order = await self._run_sync(
            self.client.create_order,
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_STOP_LOSS_LIMIT,
            quantity=quantity,
            stopPrice=stop_price,
            price=stop_price * 0.99,  # 스탑가보다 약간 낮게
            timeInForce=TIME_IN_FORCE_GTC
        )
        return order

    async def create_take_profit(
        self,
        symbol: str,
        quantity: float,
        take_profit_price: float
    ) -> Dict[str, Any]:
        """익절 주문 생성"""
        order = await self._run_sync(
            self.client.create_order,
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_TAKE_PROFIT_LIMIT,
            quantity=quantity,
            stopPrice=take_profit_price,
            price=take_profit_price * 1.01,
            timeInForce=TIME_IN_FORCE_GTC
        )
        return order

    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """주문 취소"""
        result = await self._run_sync(
            self.client.cancel_order,
            symbol=symbol,
            orderId=int(order_id)
        )
        return result

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """미체결 주문 조회"""
        if symbol:
            orders = await self._run_sync(self.client.get_open_orders, symbol=symbol)
        else:
            orders = await self._run_sync(self.client.get_open_orders)
        return orders

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """보유 포지션 조회 (스팟 기준)"""
        balances = await self.get_balance()
        positions = []
        for balance in balances:
            if balance["asset"] != "USDT" and balance["total"] > 0:
                symbol = f"{balance['asset']}USDT"
                try:
                    price_data = await self.get_current_price(symbol)
                    positions.append({
                        "symbol": symbol,
                        "quantity": balance["total"],
                        "current_price": price_data["price"],
                        "value_usdt": balance["total"] * price_data["price"]
                    })
                except Exception:
                    # 해당 페어가 없는 경우 무시
                    pass
        return positions

    async def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """캔들 데이터 조회 (실제 바이낸스 API 사용)"""
        interval_map = {
            "1m": KLINE_INTERVAL_1MINUTE,
            "5m": KLINE_INTERVAL_5MINUTE,
            "15m": KLINE_INTERVAL_15MINUTE,
            "30m": KLINE_INTERVAL_30MINUTE,
            "1h": KLINE_INTERVAL_1HOUR,
            "4h": KLINE_INTERVAL_4HOUR,
            "1d": KLINE_INTERVAL_1DAY,
            "1w": KLINE_INTERVAL_1WEEK,
        }

        kline_interval = interval_map.get(interval, KLINE_INTERVAL_1HOUR)
        # 마켓 데이터는 실제 바이낸스 API 사용
        klines = await self._run_sync(
            self.market_client.get_klines,
            symbol=symbol,
            interval=kline_interval,
            limit=limit
        )

        return [
            {
                "timestamp": k[0],  # 밀리초 단위 타임스탬프
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

    async def get_trade_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """거래 내역 조회"""
        if symbol:
            trades = await self._run_sync(
                self.client.get_my_trades,
                symbol=symbol,
                limit=limit
            )
        else:
            # 주요 코인 거래 내역 조회
            trades = []
            for sym in ["BTCUSDT", "ETHUSDT"]:
                try:
                    sym_trades = await self._run_sync(
                        self.client.get_my_trades,
                        symbol=sym,
                        limit=limit // 2
                    )
                    trades.extend(sym_trades)
                except Exception:
                    pass

        return [
            {
                "symbol": t["symbol"],
                "id": t["id"],
                "order_id": t["orderId"],
                "price": float(t["price"]),
                "quantity": float(t["qty"]),
                "commission": float(t["commission"]),
                "time": t["time"],
                "is_buyer": t["isBuyer"]
            }
            for t in trades
        ]

    async def get_ticker_24h(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """24시간 티커 데이터 조회 (실제 바이낸스 API 사용)"""
        if symbol:
            ticker = await self._run_sync(self.market_client.get_ticker, symbol=symbol)
            tickers = [ticker]
        else:
            tickers = await self._run_sync(self.market_client.get_ticker)

        return [
            {
                "symbol": t["symbol"],
                "price": float(t["lastPrice"]),
                "priceChange": float(t["priceChange"]),
                "priceChangePercent": float(t["priceChangePercent"]),
                "highPrice": float(t["highPrice"]),
                "lowPrice": float(t["lowPrice"]),
                "volume": float(t["volume"]),
                "quoteVolume": float(t["quoteVolume"]),
                "openPrice": float(t["openPrice"]),
                "prevClosePrice": float(t["prevClosePrice"]),
                "bidPrice": float(t["bidPrice"]),
                "askPrice": float(t["askPrice"]),
                "weightedAvgPrice": float(t["weightedAvgPrice"]),
                "openTime": t["openTime"],
                "closeTime": t["closeTime"],
                "count": t["count"]
            }
            for t in tickers
        ]

    async def get_top_gainers_losers(self, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """상승/하락 상위 코인 조회 (USDT 페어만)"""
        all_tickers = await self.get_ticker_24h()

        # USDT 페어만 필터링
        usdt_tickers = [t for t in all_tickers if t["symbol"].endswith("USDT")]

        # 거래량이 일정 이상인 것만 필터 (너무 작은 코인 제외)
        filtered = [t for t in usdt_tickers if t["quoteVolume"] > 100000]

        # 상승률 기준 정렬
        sorted_by_change = sorted(filtered, key=lambda x: x["priceChangePercent"], reverse=True)

        gainers = sorted_by_change[:limit]
        losers = sorted_by_change[-limit:][::-1]  # 하락 상위는 역순

        return {
            "gainers": gainers,
            "losers": losers,
            "timestamp": asyncio.get_event_loop().time()
        }

    async def get_exchange_info(self) -> Dict[str, Any]:
        """거래소 정보 조회 (실제 바이낸스 API 사용)"""
        info = await self._run_sync(self.market_client.get_exchange_info)
        return info

    async def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """심볼 검색"""
        info = await self.get_exchange_info()
        query_upper = query.upper()

        symbols = []
        for s in info["symbols"]:
            if s["status"] == "TRADING" and query_upper in s["symbol"]:
                symbols.append({
                    "symbol": s["symbol"],
                    "baseAsset": s["baseAsset"],
                    "quoteAsset": s["quoteAsset"],
                    "status": s["status"]
                })

        return symbols[:50]  # 최대 50개
