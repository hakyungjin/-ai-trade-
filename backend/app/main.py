from contextlib import asynccontextmanager
import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import trading, ai_signal, settings, realtime, signals, admin, market, realtime_chart, data, coins
from app.config import get_settings
from app.database import init_db, close_db
from app.services.binance_stream import binance_stream_manager
from app.services.binance_service import BinanceService
from app.services.batch_candle_collector import BatchCandleCollector
from app.services.smart_candle_scheduler import SmartCandleScheduler, init_scheduler

logger = logging.getLogger(__name__)

# 배치 수집기 인스턴스 (레거시 - 향후 제거 예정)
batch_collector: BatchCandleCollector = None
# 스마트 스케줄러 인스턴스
smart_scheduler: SmartCandleScheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 DB 연결 및 스트림 관리"""
    global batch_collector, smart_scheduler
    
    await init_db()
    
    # 바이낸스 서비스 초기화
    try:
        config = get_settings()
        binance = BinanceService(
            api_key=config.binance_api_key,
            secret_key=config.binance_secret_key,
            testnet=config.binance_testnet
        )
        
        # 🚀 스마트 스케줄러 초기화 및 시작 (권장)
        smart_scheduler = init_scheduler(binance)
        asyncio.create_task(delayed_scheduler_start(smart_scheduler, delay=5))
        logger.info("✅ Smart candle scheduler initialized")
        
        # 레거시 배치 수집기 (비활성화 - 스마트 스케줄러로 대체)
        # batch_collector = BatchCandleCollector(binance)
        # asyncio.create_task(delayed_collection_start(batch_collector, delay=5))
        # logger.info("✅ Batch candle collector initialized")
        
    except Exception as e:
        logger.error(f"❌ Error initializing scheduler: {e}")
    
    yield
    
    # 종료 시
    if smart_scheduler:
        await smart_scheduler.stop()
    if batch_collector:
        batch_collector.stop_periodic_collection()
    await binance_stream_manager.shutdown()
    await close_db()


async def delayed_collection_start(collector: BatchCandleCollector, delay: int = 5):
    """지연 후 주기적 수집 시작 (레거시)"""
    await asyncio.sleep(delay)
    logger.info("🚀 Starting background candle collection...")
    await collector.start_periodic_collection(
        interval="1h",
        limit=500,
        collect_interval_hours=1
    )


async def delayed_scheduler_start(scheduler: SmartCandleScheduler, delay: int = 5):
    """지연 후 스마트 스케줄러 시작"""
    await asyncio.sleep(delay)
    logger.info("🚀 Starting smart candle scheduler...")
    # 주요 타임프레임만 수집 (1h, 4h)
    # 필요 시 timeframes 리스트에 추가 (예: ["1m", "5m", "15m", "1h", "4h"])
    await scheduler.start(timeframes=["1h", "4h"])

app = FastAPI(
    title="Crypto AI Trader",
    description="AI 기반 암호화폐 자동매매 시스템",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정 (React 프론트엔드 연동)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
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
    config = get_settings()
    return {
        "status": "healthy",
        "testnet": config.binance_testnet,
        "binance_configured": bool(config.binance_api_key),
        "gemini_configured": bool(config.gemini_api_key)
    }
