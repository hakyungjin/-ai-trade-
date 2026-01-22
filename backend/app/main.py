from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import trading, ai_signal, settings, realtime, signals, admin, market, realtime_chart, data
from app.config import get_settings
from app.database import init_db, close_db
from app.services.binance_stream import binance_stream_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 DB 연결 및 스트림 관리"""
    await init_db()
    yield
    await binance_stream_manager.shutdown()
    await close_db()

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
