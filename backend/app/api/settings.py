from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
import os

router = APIRouter()

SETTINGS_FILE = "trading_settings.json"


class TradingSettings(BaseModel):
    # 기본 거래 설정
    default_stop_loss: float = 0.02  # 2%
    default_take_profit: float = 0.05  # 5%
    max_position_size: float = 100  # USDT

    # AI 설정
    auto_trading_enabled: bool = False
    prediction_threshold: float = 0.6  # 60% 이상 확신시 거래

    # 리스크 관리
    max_daily_trades: int = 10
    max_daily_loss: float = 0.1  # 10%
    trailing_stop_enabled: bool = False
    trailing_stop_percent: float = 0.01  # 1%

    # 알림 설정
    telegram_enabled: bool = False
    telegram_chat_id: Optional[str] = None


def load_settings() -> TradingSettings:
    """설정 파일 로드"""
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)
            return TradingSettings(**data)
    return TradingSettings()


def save_settings(settings: TradingSettings):
    """설정 파일 저장"""
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings.model_dump(), f, indent=2)


@router.get("/", response_model=TradingSettings)
async def get_settings():
    """현재 설정 조회"""
    return load_settings()


@router.put("/", response_model=TradingSettings)
async def update_settings(settings: TradingSettings):
    """설정 업데이트"""
    try:
        save_settings(settings)
        return settings
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch("/")
async def patch_settings(updates: dict):
    """설정 부분 업데이트"""
    try:
        current = load_settings()
        current_dict = current.model_dump()
        current_dict.update(updates)
        new_settings = TradingSettings(**current_dict)
        save_settings(new_settings)
        return new_settings
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/reset")
async def reset_settings():
    """설정 초기화"""
    default_settings = TradingSettings()
    save_settings(default_settings)
    return {"message": "Settings reset to default", "settings": default_settings}
