from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Literal
from app.services.binance_service import BinanceService
from app.config import get_settings, Settings

router = APIRouter()


class OrderRequest(BaseModel):
    symbol: str  # e.g., "BTCUSDT"
    side: Literal["BUY", "SELL"]
    quantity: float
    order_type: Literal["MARKET", "LIMIT"] = "MARKET"
    price: Optional[float] = None  # LIMIT 주문시 필요
    stop_loss: Optional[float] = None  # 스탑로스 가격
    take_profit: Optional[float] = None  # 익절 가격


class OrderResponse(BaseModel):
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    status: str


def get_binance_service(settings: Settings = Depends(get_settings)) -> BinanceService:
    return BinanceService(
        api_key=settings.binance_api_key,
        secret_key=settings.binance_secret_key,
        testnet=settings.binance_testnet
    )


@router.post("/order", response_model=OrderResponse)
async def create_order(
    order: OrderRequest,
    binance: BinanceService = Depends(get_binance_service)
):
    """주문 생성 (매수/매도)"""
    try:
        result = await binance.create_order(
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price
        )

        # 스탑로스 설정시 OCO 주문 생성
        if order.stop_loss and order.side == "BUY":
            await binance.create_stop_loss(
                symbol=order.symbol,
                quantity=order.quantity,
                stop_price=order.stop_loss
            )

        return OrderResponse(
            order_id=str(result.get("orderId", "")),
            symbol=result.get("symbol", order.symbol),
            side=result.get("side", order.side),
            quantity=float(result.get("executedQty", order.quantity)),
            price=float(result.get("price", 0)),
            status=result.get("status", "UNKNOWN")
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/balance")
async def get_balance(
    binance: BinanceService = Depends(get_binance_service)
):
    """계좌 잔고 조회"""
    try:
        return await binance.get_balance()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/positions")
async def get_positions(
    binance: BinanceService = Depends(get_binance_service)
):
    """현재 포지션 조회"""
    try:
        return await binance.get_open_positions()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/price/{symbol}")
async def get_price(
    symbol: str,
    binance: BinanceService = Depends(get_binance_service)
):
    """현재가 조회"""
    try:
        return await binance.get_current_price(symbol)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/history")
async def get_trade_history(
    symbol: Optional[str] = None,
    limit: int = 50,
    binance: BinanceService = Depends(get_binance_service)
):
    """거래 내역 조회"""
    try:
        return await binance.get_trade_history(symbol=symbol, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/order/{symbol}/{order_id}")
async def cancel_order(
    symbol: str,
    order_id: str,
    binance: BinanceService = Depends(get_binance_service)
):
    """주문 취소"""
    try:
        return await binance.cancel_order(symbol=symbol, order_id=order_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
