"""
바이낸스에서 과거 가격 데이터를 수집하는 모듈
"""
import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import time

load_dotenv()


class DataCollector:
    def __init__(self, api_key: str = None, secret_key: str = None):
        self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self.secret_key = secret_key or os.getenv("BINANCE_SECRET_KEY", "")
        self.client = Client(self.api_key, self.secret_key)

    def fetch_historical_data(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        start_date: str = None,
        end_date: str = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        바이낸스에서 과거 캔들 데이터 수집

        Args:
            symbol: 거래 페어 (예: BTCUSDT)
            interval: 캔들 간격 (1m, 5m, 15m, 1h, 4h, 1d)
            start_date: 시작 날짜 (예: "2024-01-01")
            end_date: 종료 날짜
            limit: 최대 캔들 수

        Returns:
            DataFrame with OHLCV data
        """
        interval_map = {
            "1m": Client.KLINE_INTERVAL_1MINUTE,
            "5m": Client.KLINE_INTERVAL_5MINUTE,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "4h": Client.KLINE_INTERVAL_4HOUR,
            "1d": Client.KLINE_INTERVAL_1DAY,
        }

        kline_interval = interval_map.get(interval, Client.KLINE_INTERVAL_1HOUR)

        klines = self.client.get_historical_klines(
            symbol=symbol,
            interval=kline_interval,
            start_str=start_date,
            end_str=end_date,
            limit=limit
        )

        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])

        # 데이터 타입 변환
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)

        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df.set_index("timestamp", inplace=True)

        return df

    def fetch_multiple_symbols(
        self,
        symbols: list = ["BTCUSDT", "ETHUSDT"],
        interval: str = "1h",
        start_date: str = None,
        days: int = 365
    ) -> dict:
        """
        여러 심볼의 데이터를 수집

        Returns:
            Dict of DataFrames for each symbol
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        data = {}
        for symbol in symbols:
            print(f"Fetching {symbol}...")
            try:
                df = self.fetch_historical_data(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date
                )
                data[symbol] = df
                time.sleep(0.5)  # Rate limit 방지
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")

        return data

    def save_data(self, df: pd.DataFrame, filepath: str):
        """데이터를 CSV로 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath)
        print(f"Data saved to {filepath}")

    def load_data(self, filepath: str) -> pd.DataFrame:
        """CSV에서 데이터 로드"""
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df


if __name__ == "__main__":
    collector = DataCollector()

    # BTC 1년치 1시간봉 데이터 수집
    btc_data = collector.fetch_historical_data(
        symbol="BTCUSDT",
        interval="1h",
        start_date="2024-01-01"
    )

    print(f"Collected {len(btc_data)} candles")
    print(btc_data.head())

    # 저장
    collector.save_data(btc_data, "../data/btcusdt_1h.csv")
