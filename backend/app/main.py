from contextlib import asynccontextmanager
import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import trading, ai_signal, settings, realtime, signals, admin, market, realtime_chart, data, coins
from app.api.stocks import stocks
from app.config import get_settings
from app.database import init_db, close_db
from app.services.binance_stream import binance_stream_manager
from app.services.binance_service import BinanceService
from app.services.batch_candle_collector import BatchCandleCollector

logger = logging.getLogger(__name__)

# 배치 수집기 인스턴스
batch_collector: BatchCandleCollector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 DB 연결 및 스트림 관리"""
    global batch_collector

    # DB 초기화 (실패해도 앱은 시작)
    try:
        await init_db()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.warning(f"⚠️ Database initialization failed (will retry): {e}")

    # 배치 캔들 수집기 초기화 및 시작 (실패해도 앱은 시작)
    try:
        config = get_settings()
        binance = BinanceService(
            api_key=config.binance_api_key,
            secret_key=config.binance_secret_key,
            testnet=config.binance_testnet
        )
        batch_collector = BatchCandleCollector(binance)

        # 백그라운드 작업으로 주기적 수집 시작 (1시간마다)
        # 앱이 시작되면 첫 수집은 지연시키기 (30초 후 - Cloud Run 타임아웃 방지)
        asyncio.create_task(delayed_collection_start(batch_collector, delay=30))
        logger.info("✅ Batch candle collector initialized")
    except Exception as e:
        logger.warning(f"⚠️ Batch collector initialization failed: {e}")

    yield

    # 종료 시
    try:
        if batch_collector:
            batch_collector.stop_periodic_collection()
        await binance_stream_manager.shutdown()
        await close_db()
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")


async def delayed_collection_start(collector: BatchCandleCollector, delay: int = 5):
    """지연 후 주기적 수집 시작"""
    await asyncio.sleep(delay)
    logger.info("🚀 Starting background candle collection...")
    await collector.start_periodic_collection(
        interval="1h",
        limit=500,
        collect_interval_hours=1
    )

app = FastAPI(
    title="Crypto AI Trader",
    description="AI 기반 암호화폐 자동매매 시스템",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정 (React 프론트엔드 연동)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "https://ai-trader-e69ae.web.app",
        "https://attract--web.web.app",
        # Railway & Vercel 배포용
        "https://*.up.railway.app",
        "https://*.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.(railway\.app|vercel\.app)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(trading.router, prefix="/api/trading", tags=["Trading"])
app.include_router(ai_signal.router, prefix="/api/ai", tags=["AI Signal"])
app.include_router(settings.router, prefix="/api/settings", tags=["Settings"])
app.include_router(realtime.router, prefix="/api/realtime", tags=["Realtime"])
app.include_router(signals.router, prefix="/api/signals", tags=["Signals"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
app.include_router(market.router, prefix="/api/market", tags=["Market"])
app.include_router(realtime_chart.router, prefix="/api/chart", tags=["Realtime Chart"])
app.include_router(coins.router)  # coins API는 /api/v1/coins 접두사 포함
app.include_router(stocks.router)  # stocks API는 /api/v1/stocks 접두사 포함
app.include_router(data.router)


@app.get("/")
async def root():
    return {
        "message": "Crypto AI Trader API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트 - Cloud Run 시작 확인용"""
    try:
        config = get_settings()
        return {
            "status": "healthy",
            "testnet": config.binance_testnet,
            "binance_configured": bool(config.binance_api_key),
            "gemini_configured": bool(config.gemini_api_key),
            "database_configured": "localhost" not in config.database_url
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "degraded",
            "error": str(e)
        }
