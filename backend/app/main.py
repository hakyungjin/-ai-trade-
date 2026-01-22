from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import trading, ai_signal, settings, realtime, signals, admin
from app.config import get_settings

app = FastAPI(
    title="Crypto AI Trader",
    description="AI 기반 암호화폐 자동매매 시스템",
    version="1.0.0"
)

# CORS 설정 (React 프론트엔드 연동)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
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
