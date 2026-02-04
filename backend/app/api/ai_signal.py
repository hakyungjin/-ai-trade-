from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.ai_service import AIService
from app.services.gemini_service import GeminiService
from app.services.binance_service import BinanceService
from app.services.weighted_strategy import WeightedStrategy
from app.services.technical_indicators import TechnicalIndicators
from app.services.vector_pattern_service import VectorPatternService
from app.services.unified_model_service import get_unified_service
from app.services.unified_data_service import UnifiedDataService
from app.config import get_settings
from app.database import get_db
from app.models.vector_pattern import VectorPattern
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)

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
    use_unified_model: bool = True  # 통합 모델 사용 여부 (기본값: True)


class PredictionResponse(BaseModel):
    symbol: str
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 ~ 1.0
    predicted_direction: str  # "UP", "DOWN", "NEUTRAL"
    current_price: float
    analysis: str


class VolumeStats(BaseModel):
    """거래량 종합 통계"""
    current_volume: float = 0
    ma_20_ratio: float = 0      # 20기간 MA 대비 배수
    ma_50_ratio: float = 0      # 50기간 MA 대비 배수
    percentile: float = 0       # 전체 대비 백분위 (0~100)
    prev_5_ratio: float = 0     # 직전 5캔들 평균 대비 배수
    is_spike: bool = False      # 급등 여부 (2σ 초과)
    avg: float = 0              # 전체 평균
    max: float = 0              # 전체 최대
    min: float = 0              # 전체 최소
    std: float = 0              # 표준편차
    z_score: float = 0          # Z-score


class WeightedSignalResponse(BaseModel):
    signal: str  # "STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"
    score: float  # -1 ~ 1
    confidence: float  # 0.0 ~ 1.0
    indicators: Dict[str, Any]  # 각 지표별 점수
    recommendation: str
    volume_stats: Optional[VolumeStats] = None  # 거래량 종합 통계


class CombinedPredictionResponse(BaseModel):
    """AI 예측 + 가중치 전략 응답"""
    symbol: str
    current_price: float
    timeframe: str
    
    # AI 예측
    ai_prediction: PredictionResponse
    
    # 가중치 기반 전략
    weighted_signal: WeightedSignalResponse
    
    # 최종 종합 신호
    final_signal: str  # "BUY", "SELL", "HOLD"
    final_confidence: float


class PromptSettingRequest(BaseModel):
    prompt: str  # 사용자 프롬프트 (예: "스탑로스 3%, 익절 5% 설정해줘")


class TradingRule(BaseModel):
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_position: Optional[float] = None
    trailing_stop: Optional[float] = None
    description: str


@router.post("/predict", response_model=PredictionResponse)
async def get_prediction(
    request: PredictionRequest,
    db: AsyncSession = Depends(get_db)
):
    """AI 예측 신호 조회 (DB 캐시 활용 + Gemini 우선, fallback으로 기존 모델)"""
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

        # 통합 데이터 서비스로 캐시 + 증분 수집 활용
        unified_service = UnifiedDataService(db, binance)
        candles = await unified_service.get_klines_with_cache(
            symbol=request.symbol,
            timeframe=request.timeframe,
            limit=100
        )

        prediction = None
        
        # Gemini API 키가 있으면 Gemini 사용
        if config.gemini_api_key:
            try:
                prediction = await gemini_service.analyze_chart(
                    symbol=request.symbol,
                    candles=candles,
                    current_price=current_price,
                    timeframe=request.timeframe
                )
            except Exception as gemini_error:
                # Gemini 에러 (할당량 초과 등) - 로깅만 하고 fallback
                print(f"⚠️  Gemini API error (fallback to local model): {gemini_error}")
                prediction = None
        
        # Gemini 없거나 실패 시 로컬 모델 사용
        if prediction is None:
            # 통합 모델 우선 시도
            if request.use_unified_model:
                try:
                    unified_service = get_unified_service()
                    if unified_service:
                        logger.info(f"🧠 Using Unified LSTM Model for {request.symbol}")
                        prediction = await unified_service.predict(
                            symbol=request.symbol,
                            candles=candles,
                            db=db
                        )
                    else:
                        logger.warning("⚠️  Unified model not available, falling back to XGBoost")
                        prediction = None
                except Exception as unified_error:
                    logger.error(f"⚠️  Unified model error: {unified_error}")
                    prediction = None

            # 통합 모델 실패 또는 미사용 시 기존 XGBoost 모델
            if prediction is None:
                try:
                    prediction = await ai_service.predict_signal(
                        symbol=request.symbol,
                        candles=candles,
                        current_price=current_price
                    )
                except Exception as local_error:
                    logger.error(f"⚠️  Local AI model error: {local_error}")
                    # Fallback: 중립 신호 반환
                    prediction = {
                        "signal": "HOLD",
                        "confidence": 0.5,
                        "direction": "NEUTRAL",
                        "analysis": "Unable to analyze - using default neutral signal"
                    }

        return PredictionResponse(
            symbol=request.symbol,
            signal=prediction["signal"],
            confidence=prediction["confidence"],
            predicted_direction=prediction["direction"],
            current_price=current_price,
            analysis=prediction["analysis"]
        )
    except Exception as e:
        print(f"❌ Prediction error: {e}")
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


@router.post("/combined-analysis", response_model=CombinedPredictionResponse)
async def get_combined_analysis(
    request: PredictionRequest,
    db: AsyncSession = Depends(get_db)
):
    """🚀 AI 예측 + 가중치 기반 전략 통합 분석"""
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

        # 데이터 수집 (Binance에서 직접)
        try:
            print(f"🔄 Fetching candles for {request.symbol} {request.timeframe}...")
            candles = await binance.get_klines(
                symbol=request.symbol,
                interval=request.timeframe,
                limit=100
            )
            print(f"✅ Fetched {len(candles)} candles from Binance")
        except Exception as e:
            print(f"❌ Error fetching klines: {e}")
            import traceback
            traceback.print_exc()
            return CombinedPredictionResponse(
                symbol=request.symbol,
                current_price=0,
                timeframe=request.timeframe,
                ai_prediction=PredictionResponse(
                    symbol=request.symbol,
                    signal="HOLD",
                    confidence=0.0,
                    predicted_direction="NEUTRAL",
                    current_price=0,
                    analysis="Failed to fetch candles"
                ),
                weighted_signal=WeightedSignalResponse(
                    signal="neutral",
                    score=0,
                    confidence=0,
                    indicators={},
                    recommendation="캔들 데이터 조회 실패"
                ),
                final_signal="HOLD",
                final_confidence=0.0
            )

        # ===== AI 예측 =====
        ai_prediction = None
        if config.gemini_api_key:
            try:
                ai_prediction = await gemini_service.analyze_chart(
                    symbol=request.symbol,
                    candles=candles,
                    current_price=current_price,
                    timeframe=request.timeframe
                )
            except Exception as e:
                print(f"⚠️ Gemini API error: {e}")
                ai_prediction = None
        
        if ai_prediction is None:
            try:
                ai_prediction = await ai_service.predict_signal(
                    symbol=request.symbol,
                    candles=candles,
                    current_price=current_price
                )
            except Exception as e:
                print(f"⚠️ Local AI model error: {e}")
                ai_prediction = {
                    "signal": "HOLD",
                    "confidence": 0.5,
                    "direction": "NEUTRAL",
                    "analysis": "Unable to analyze - using default neutral signal"
                }

        # ===== 가중치 기반 전략 =====
        try:
            import pandas as pd
            import logging
            
            logger = logging.getLogger(__name__)
            logger.info(f"📊 Starting weighted analysis for {request.symbol}")
            
            # 캔들 데이터를 DataFrame으로 변환
            df = pd.DataFrame(candles)
            print(f"📋 Initial candle DataFrame columns: {df.columns.tolist()}")
            print(f"📋 First row: {df.iloc[0].to_dict() if len(df) > 0 else 'EMPTY'}")
            
            # datetime 필드 제거 (OHLCV만 사용)
            datetime_cols = ['open_time', 'close_time', 'timestamp']
            df = df.drop(columns=[col for col in datetime_cols if col in df.columns])
            
            # 필요한 컬럼만 선택 및 숫자형 변환
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"⚠️ Missing required columns. Available: {df.columns.tolist()}")
                raise ValueError(f"Required columns missing: {required_cols}")
            
            # 모든 OHLCV 컬럼을 float로 변환
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # NaN 제거
            df = df.dropna(subset=required_cols)
            
            if len(df) < 5:
                raise ValueError(f"Not enough candle data: {len(df)} < 5")
            
            logger.info(f"✅ Cleaned candle data shape: {df.shape}, columns: {df.columns.tolist()}")
            
            # 기술 지표 계산 (static 메서드)
            tech_data = TechnicalIndicators.calculate_all_indicators(df)
            
            logger.info(f"Tech data shape after indicators: {tech_data.shape}")
            
            # 가중치 전략 적용
            strategy = WeightedStrategy()
            analysis_result = strategy.analyze(tech_data)
            
            logger.info(f"✅ Weighted analysis result: {analysis_result.get('signal')}")
            logger.info(f"Analysis result keys: {analysis_result.keys()}")
            
            # 응답 객체 생성
            recommendation_text = '기술적 분석 중립'
            if isinstance(analysis_result.get('recommendation'), dict):
                recommendation_text = analysis_result['recommendation'].get('description', '기술적 분석 중립')
            
            # 거래량 통계 변환
            raw_volume_stats = analysis_result.get('volume_stats', {})
            volume_stats_obj = VolumeStats(**raw_volume_stats) if raw_volume_stats else None

            weighted_signal = WeightedSignalResponse(
                signal=analysis_result.get('signal', 'neutral'),
                score=float(analysis_result.get('combined_score', 0)),
                confidence=float(analysis_result.get('confidence', 0)),
                indicators=analysis_result.get('indicator_scores', {}),
                recommendation=recommendation_text,
                volume_stats=volume_stats_obj
            )
            logger.info(f"📈 Weighted signal response created: {weighted_signal.signal}")
            
        except Exception as e:
            import traceback
            logger.error(f"❌ Weighted strategy error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            print(f"⚠️ Weighted strategy error: {e}")
            print(traceback.format_exc())
            
            weighted_signal = WeightedSignalResponse(
                signal="neutral",
                score=0,
                confidence=0.5,
                indicators={},
                recommendation="기술적 분석 불가"
            )

        # ===== 최종 종합 신호 =====
        # AI 신호와 가중치 신호를 종합
        ai_signal_value = 1 if ai_prediction["signal"] == "BUY" else -1 if ai_prediction["signal"] == "SELL" else 0
        final_score = (ai_signal_value * ai_prediction["confidence"] + weighted_signal.score) / 2
        
        final_signal = "BUY" if final_score > 0.3 else "SELL" if final_score < -0.3 else "HOLD"
        final_confidence = min(abs(final_score), 1.0)

        logger.info(f"🎯 Final signal: {final_signal} (confidence: {final_confidence})")

        # ===== VectorPattern 자동 저장 =====
        try:
            # 최신 캔들 데이터에서 지표 추출
            latest_candle = candles[-1] if candles else {}
            indicators_dict = {
                'rsi_14': analysis_result.get('indicators', {}).get('rsi'),
                'macd': analysis_result.get('indicators', {}).get('macd'),
                'macd_signal': analysis_result.get('indicators', {}).get('macd_signal'),
                'macd_histogram': analysis_result.get('indicators', {}).get('macd_histogram'),
                'bb_upper': analysis_result.get('indicators', {}).get('bb_upper'),
                'bb_middle': analysis_result.get('indicators', {}).get('bb_middle'),
                'bb_lower': analysis_result.get('indicators', {}).get('bb_lower'),
                'ema_12': analysis_result.get('indicators', {}).get('ema_12'),
                'ema_26': analysis_result.get('indicators', {}).get('ema_26'),
                'stoch_k': analysis_result.get('indicators', {}).get('stoch_k'),
                'stoch_d': analysis_result.get('indicators', {}).get('stoch_d'),
                'atr_14': analysis_result.get('indicators', {}).get('atr_14'),
                'volume': latest_candle.get('volume', 0),
                'close': current_price,
            }
            
            # VectorPattern 레코드 생성 및 저장
            vector_pattern = VectorPattern(
                symbol=request.symbol,
                timeframe=request.timeframe,
                timestamp=datetime.now(),
                vector_id=None,  # FAISS 인덱싱은 나중에
                indicators=indicators_dict,
                signal=final_signal,
                confidence=final_confidence,
                price_at_signal=current_price,
                return_1h=None,      # 1시간 후 업데이트
                return_4h=None,      # 4시간 후 업데이트
                return_24h=None,     # 24시간 후 업데이트
            )
            
            db.add(vector_pattern)
            await db.commit()
            
            logger.info(f"💾 Saved VectorPattern: {request.symbol} {final_signal} @ {current_price}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save VectorPattern: {e}")
            await db.rollback()
            # 저장 실패해도 응답은 진행

        return CombinedPredictionResponse(
            symbol=request.symbol,
            current_price=current_price,
            timeframe=request.timeframe,
            ai_prediction=PredictionResponse(
                symbol=request.symbol,
                signal=ai_prediction["signal"],
                confidence=ai_prediction["confidence"],
                predicted_direction=ai_prediction["direction"],
                current_price=current_price,
                analysis=ai_prediction["analysis"]
            ),
            weighted_signal=weighted_signal,
            final_signal=final_signal,
            final_confidence=final_confidence
        )
    except Exception as e:
        import traceback
        print(f"❌ Combined analysis error: {e}")
        print(traceback.format_exc())
        logger.error(f"❌ Combined analysis error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
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
