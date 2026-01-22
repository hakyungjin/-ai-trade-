"""
데이터 수집 및 저장 서비스
- 바이낸스에서 대량 과거 데이터 수집
- SQLite 데이터베이스에 저장
- 캐싱 및 업데이트 기능
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import aiosqlite
from pathlib import Path
import logging

from .binance_service import BinanceService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    """데이터 수집 및 저장 클래스"""

    def __init__(self, binance_service: BinanceService, db_path: str = "./data/market_data.db"):
        self.binance = binance_service
        self.db_path = db_path

        # 데이터 디렉토리 생성
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async def init_database(self):
        """데이터베이스 초기화"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS klines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    close_time INTEGER,
                    quote_volume REAL,
                    trades INTEGER,
                    UNIQUE(symbol, interval, timestamp)
                )
            """)

            # 인덱스 생성
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_interval_timestamp
                ON klines(symbol, interval, timestamp)
            """)

            await db.commit()
            logger.info("Database initialized")

    async def collect_historical_data(
        self,
        symbol: str,
        interval: str = "1h",
        days: int = 30,
        save_to_db: bool = True
    ) -> pd.DataFrame:
        """
        과거 데이터 수집

        Args:
            symbol: 심볼 (예: BTCUSDT)
            interval: 시간 간격 (1m, 5m, 15m, 1h, 4h, 1d)
            days: 수집할 일수
            save_to_db: 데이터베이스 저장 여부

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Collecting {days} days of {interval} data for {symbol}")

        # 시간당 캔들 개수 계산
        interval_minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "4h": 240,
            "1d": 1440
        }

        minutes = interval_minutes.get(interval, 60)
        candles_per_day = 1440 // minutes
        total_candles = candles_per_day * days

        # 바이낸스 API는 한 번에 최대 1000개까지 조회 가능
        limit = 1000
        all_klines = []

        # 여러 번 요청하여 데이터 수집
        for i in range(0, total_candles, limit):
            try:
                klines = await self.binance.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=min(limit, total_candles - i)
                )

                if not klines:
                    break

                all_klines.extend(klines)
                logger.info(f"Collected {len(all_klines)}/{total_candles} candles")

                # API 레이트 리밋 방지
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error collecting data: {e}")
                break

        # DataFrame 생성
        df = pd.DataFrame(all_klines)

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # 데이터베이스 저장
            if save_to_db:
                await self._save_to_db(symbol, interval, all_klines)

            logger.info(f"Successfully collected {len(df)} candles for {symbol}")

        return df

    async def _save_to_db(self, symbol: str, interval: str, klines: List[Dict]):
        """데이터베이스에 캔들 데이터 저장"""
        async with aiosqlite.connect(self.db_path) as db:
            for kline in klines:
                try:
                    await db.execute("""
                        INSERT OR REPLACE INTO klines
                        (symbol, interval, timestamp, open, high, low, close, volume,
                         close_time, quote_volume, trades)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        interval,
                        kline['timestamp'],
                        kline['open'],
                        kline['high'],
                        kline['low'],
                        kline['close'],
                        kline['volume'],
                        kline.get('close_time'),
                        kline.get('quote_volume'),
                        kline.get('trades')
                    ))
                except Exception as e:
                    logger.error(f"Error saving kline: {e}")

            await db.commit()
            logger.info(f"Saved {len(klines)} candles to database")

    async def load_from_db(
        self,
        symbol: str,
        interval: str = "1h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        데이터베이스에서 데이터 로드

        Args:
            symbol: 심볼
            interval: 시간 간격
            start_time: 시작 시간
            end_time: 종료 시간
            limit: 최대 레코드 수

        Returns:
            DataFrame with OHLCV data
        """
        query = """
            SELECT timestamp, open, high, low, close, volume,
                   quote_volume, trades
            FROM klines
            WHERE symbol = ? AND interval = ?
        """
        params = [symbol, interval]

        if start_time:
            query += " AND timestamp >= ?"
            params.append(int(start_time.timestamp() * 1000))

        if end_time:
            query += " AND timestamp <= ?"
            params.append(int(end_time.timestamp() * 1000))

        query += " ORDER BY timestamp ASC"

        if limit:
            query += f" LIMIT {limit}"

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=[
            'timestamp', 'open', 'high', 'low', 'close',
            'volume', 'quote_volume', 'trades'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        logger.info(f"Loaded {len(df)} candles from database")
        return df

    async def update_latest_data(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 100
    ) -> pd.DataFrame:
        """
        최신 데이터로 업데이트

        Args:
            symbol: 심볼
            interval: 시간 간격
            limit: 가져올 캔들 개수

        Returns:
            Updated DataFrame
        """
        # 최신 데이터 가져오기
        klines = await self.binance.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )

        # 데이터베이스에 저장
        if klines:
            await self._save_to_db(symbol, interval, klines)

        # 데이터베이스에서 로드
        return await self.load_from_db(symbol, interval, limit=limit)

    async def collect_multiple_symbols(
        self,
        symbols: List[str],
        interval: str = "1h",
        days: int = 30
    ) -> Dict[str, pd.DataFrame]:
        """
        여러 심볼의 데이터 동시 수집

        Args:
            symbols: 심볼 리스트
            interval: 시간 간격
            days: 수집할 일수

        Returns:
            Dict mapping symbol to DataFrame
        """
        logger.info(f"Collecting data for {len(symbols)} symbols")

        # 동시 실행을 위한 태스크 생성
        tasks = [
            self.collect_historical_data(symbol, interval, days)
            for symbol in symbols
        ]

        # 모든 태스크 실행
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과를 딕셔너리로 변환
        data_dict = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error collecting {symbol}: {result}")
            else:
                data_dict[symbol] = result

        return data_dict

    async def get_latest_price(self, symbol: str) -> float:
        """최신 가격 조회"""
        price_data = await self.binance.get_current_price(symbol)
        return price_data['price']

    async def get_data_stats(self, symbol: str, interval: str) -> Dict[str, Any]:
        """데이터 통계 조회"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT
                    COUNT(*) as count,
                    MIN(timestamp) as first_timestamp,
                    MAX(timestamp) as last_timestamp
                FROM klines
                WHERE symbol = ? AND interval = ?
            """, (symbol, interval)) as cursor:
                row = await cursor.fetchone()

        if row and row[0] > 0:
            return {
                'symbol': symbol,
                'interval': interval,
                'count': row[0],
                'first_date': datetime.fromtimestamp(row[1] / 1000).strftime('%Y-%m-%d %H:%M'),
                'last_date': datetime.fromtimestamp(row[2] / 1000).strftime('%Y-%m-%d %H:%M')
            }

        return {
            'symbol': symbol,
            'interval': interval,
            'count': 0,
            'first_date': None,
            'last_date': None
        }
