from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Binance API
    binance_api_key: str = ""
    binance_secret_key: str = ""
    binance_testnet: bool = True

    # Database (MySQL)
    database_url: str = "mysql+aiomysql://root:password@localhost:3306/crypto_trader"
    sqlalchemy_url: str = "mysql+aiomysql://root:password@localhost:3306/crypto_trader"

    # Gemini AI
    gemini_api_key: str = ""

    # AI Model (기존 PyTorch 모델 - fallback용)
    model_path: str = "../ai-model/models/price_predictor.pt"
    prediction_threshold: float = 0.6

    # Trading Settings
    default_stop_loss: float = 0.02  # 2%
    default_take_profit: float = 0.05  # 5%
    max_position_size: float = 100  # USDT

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
