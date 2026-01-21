from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from app.services.ai_service import AIService
from app.services.gemini_service import GeminiService
from app.services.binance_service import BinanceService
from app.config import get_settings

router = APIRouter()

# 설정 로드
settings = get_settings()

# Gemini 서비스 초기화
gemini_service = GeminiService(api_key=settings.gemini_api_key)

# 기존 AI 서비스 (fallback용)
model_path = settings.model_path if hasattr(settings, 'model_path') else None
if model_path and not os.path.isabs(model_path):
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', model_path)
ai_service = AIService(model_path=model_path)


class PredictionRequest(BaseModel):
    symbol: str
    timeframe: str = "1h"  # 1m, 5m, 15m, 1h, 4h, 1d


class PredictionResponse(BaseModel):
    symbol: str
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 ~ 1.0
    predicted_direction: str  # "UP", "DOWN", "NEUTRAL"
    current_price: float
    analysis: str


class PromptSettingRequest(BaseModel):
    prompt: str  # 사용자 프롬프트 (예: "스탑로스 3%, 익절 5% 설정해줘")


class TradingRule(BaseModel):
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_position: Optional[float] = None
    trailing_stop: Optional[float] = None
    description: str


@router.post("/predict", response_model=PredictionResponse)
async def get_prediction(request: PredictionRequest):
    """AI 예측 신호 조회 (Gemini 우선, fallback으로 기존 모델)"""
    try:
        config = get_settings()
        binance = BinanceService(
            api_key=config.binance_api_key,
            secret_key=config.binance_secret_key,
            testnet=config.binance_testnet
        )

        # 현재가 조회
        price_data = await binance.get_current_price(request.symbol)
        current_price = price_data.get("price", 0)

        # 캔들 데이터 조회 (AI 분석용)
        candles = await binance.get_klines(
            symbol=request.symbol,
            interval=request.timeframe,
            limit=100
        )

        # Gemini API 키가 있으면 Gemini 사용
        if config.gemini_api_key:
            prediction = await gemini_service.analyze_chart(
                symbol=request.symbol,
                candles=candles,
                current_price=current_price,
                timeframe=request.timeframe
            )
        else:
            # Gemini 없으면 기존 AI 서비스 사용
            prediction = await ai_service.predict_signal(
                symbol=request.symbol,
                candles=candles,
                current_price=current_price
            )

        return PredictionResponse(
            symbol=request.symbol,
            signal=prediction["signal"],
            confidence=prediction["confidence"],
            predicted_direction=prediction["direction"],
            current_price=current_price,
            analysis=prediction["analysis"]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/parse-prompt", response_model=TradingRule)
async def parse_trading_prompt(request: PromptSettingRequest):
    """
    자연어 프롬프트를 거래 규칙으로 변환
    예: "스탑로스 3%로 설정하고 익절은 5%로 해줘"
    """
    try:
        rule = await ai_service.parse_trading_prompt(request.prompt)
        return TradingRule(**rule)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/market-analysis/{symbol}")
async def get_market_analysis(symbol: str, timeframe: str = "1h"):
    """종합 시장 분석"""
    try:
        settings = get_settings()
        binance = BinanceService(
            api_key=settings.binance_api_key,
            secret_key=settings.binance_secret_key,
            testnet=settings.binance_testnet
        )

        # 데이터 수집
        candles = await binance.get_klines(symbol=symbol, interval=timeframe, limit=200)
        price_data = await binance.get_current_price(symbol)

        # AI 분석 수행
        analysis = await ai_service.analyze_market(
            symbol=symbol,
            candles=candles,
            current_price=price_data.get("price", 0)
        )

        return analysis
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/signals")
async def get_active_signals():
    """현재 활성화된 AI 신호 목록"""
    try:
        return await ai_service.get_active_signals()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========== Gemini 전용 엔드포인트 ==========

@router.post("/gemini/analyze")
async def gemini_analyze(request: PredictionRequest):
    """Gemini AI 상세 차트 분석"""
    try:
        config = get_settings()
        if not config.gemini_api_key:
            raise HTTPException(status_code=400, detail="Gemini API 키가 설정되지 않았습니다")

        binance = BinanceService(
            api_key=config.binance_api_key,
            secret_key=config.binance_secret_key,
            testnet=config.binance_testnet
        )

        # 데이터 수집
        price_data = await binance.get_current_price(request.symbol)
        current_price = price_data.get("price", 0)

        candles = await binance.get_klines(
            symbol=request.symbol,
            interval=request.timeframe,
            limit=100
        )

        # Gemini 분석
        analysis = await gemini_service.analyze_chart(
            symbol=request.symbol,
            candles=candles,
            current_price=current_price,
            timeframe=request.timeframe
        )

        return analysis

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/gemini/history")
async def get_gemini_history(limit: int = 20):
    """Gemini 분석 히스토리 조회"""
    return gemini_service.get_recent_analyses(limit)


@router.get("/gemini/market-sentiment")
async def get_market_sentiment():
    """전체 시장 심리 분석 (주요 코인 종합)"""
    try:
        config = get_settings()
        if not config.gemini_api_key:
            raise HTTPException(status_code=400, detail="Gemini API 키가 설정되지 않았습니다")

        binance = BinanceService(
            api_key=config.binance_api_key,
            secret_key=config.binance_secret_key,
            testnet=config.binance_testnet
        )

        # 주요 코인 데이터 수집
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
        candles_data = {}

        for symbol in symbols:
            try:
                candles = await binance.get_klines(symbol=symbol, interval="1h", limit=24)
                candles_data[symbol] = candles
            except:
                pass

        # Gemini 시장 심리 분석
        sentiment = await gemini_service.get_market_sentiment(symbols, candles_data)
        return sentiment

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
