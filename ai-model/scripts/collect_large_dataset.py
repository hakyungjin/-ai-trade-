"""
대량 데이터 수집 스크립트
- Binance API limit (1000개) 우회
- 여러 번 요청으로 과거 데이터 수집
- 1분봉 기준 최대 몇 달치 데이터 수집 가능
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime, timedelta

# backend 디렉토리를 경로에 추가 (app 모듈 import용)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
backend_path = os.path.join(project_root, 'backend')
sys.path.insert(0, backend_path)

from app.database import AsyncSessionLocal, init_db
from app.services.binance_service import BinanceService
from app.models.market_data import MarketCandle
from app.config import get_settings
from sqlalchemy import select, func


async def get_candle_count(db, symbol: str, timeframe: str) -> int:
    """현재 저장된 캔들 개수 조회"""
    stmt = select(func.count()).where(
        MarketCandle.symbol == symbol,
        MarketCandle.timeframe == timeframe
    )
    result = await db.execute(stmt)
    return result.scalar() or 0


async def get_oldest_candle_time(db, symbol: str, timeframe: str):
    """가장 오래된 캔들 시간 조회"""
    stmt = select(MarketCandle.open_time).where(
        MarketCandle.symbol == symbol,
        MarketCandle.timeframe == timeframe
    ).order_by(MarketCandle.open_time.asc()).limit(1)
    result = await db.execute(stmt)
    return result.scalar()


async def collect_historical_batch(
    binance: BinanceService,
    db,
    symbol: str,
    timeframe: str,
    end_time: datetime,
    limit: int = 1000
) -> int:
    """특정 시점 이전 데이터 배치 수집"""
    try:
        # end_time을 밀리초로 변환
        end_ms = int(end_time.timestamp() * 1000)
        
        # Binance API 호출 (endTime 파라미터 사용)
        klines = await binance.get_klines(
            symbol=symbol,
            interval=timeframe,
            limit=limit,
            endTime=end_ms
        )
        
        if not klines:
            return 0
        
        saved_count = 0
        for kline in klines:
            try:
                # 타임스탬프를 datetime으로 변환
                open_time_raw = kline.get("timestamp")
                if isinstance(open_time_raw, (int, float)):
                    open_time_dt = datetime.fromtimestamp(open_time_raw / 1000)
                else:
                    open_time_dt = open_time_raw
                
                # 중복 체크 (datetime으로 비교)
                stmt = select(MarketCandle).where(
                    MarketCandle.symbol == symbol,
                    MarketCandle.timeframe == timeframe,
                    MarketCandle.open_time == open_time_dt
                )
                existing = await db.execute(stmt)
                if existing.scalar():
                    continue
                
                # 새 캔들 저장
                candle = MarketCandle(
                    symbol=symbol,
                    timeframe=timeframe,
                    open_time=open_time_dt,
                    open=float(kline.get("open", 0)),
                    high=float(kline.get("high", 0)),
                    low=float(kline.get("low", 0)),
                    close=float(kline.get("close", 0)),
                    volume=float(kline.get("volume", 0)),
                    quote_volume=float(kline.get("quote_volume", 0)),
                    trades_count=int(kline.get("trades_count", 0))
                )
                db.add(candle)
                saved_count += 1
                
            except Exception as e:
                print(f"  [WARN] Error saving candle: {e}")
                continue
        
        if saved_count > 0:
            await db.commit()
        
        return saved_count
        
    except Exception as e:
        print(f"  [ERROR] Batch collection error: {e}")
        await db.rollback()
        return 0


async def collect_large_dataset(
    symbol: str,
    timeframe: str,
    target_count: int = 50000,
    batch_size: int = 1000
):
    """대량 데이터 수집 메인 함수"""
    
    print("=" * 60)
    print(f"  Large Dataset Collection")
    print(f"  Symbol: {symbol}, Timeframe: {timeframe}")
    print(f"  Target: {target_count:,} candles")
    print("=" * 60)
    
    # DB 초기화
    await init_db()
    
    # Binance 서비스 초기화
    settings = get_settings()
    binance = BinanceService(
        api_key=settings.binance_api_key,
        secret_key=settings.binance_secret_key,
        testnet=False
    )
    
    # 타임프레임별 분 단위
    timeframe_minutes = {
        "1m": 1,
        "3m": 3,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "6h": 360,
        "8h": 480,
        "12h": 720,
        "1d": 1440,
    }
    
    minutes = timeframe_minutes.get(timeframe, 60)
    batch_duration = timedelta(minutes=minutes * batch_size)
    
    async with AsyncSessionLocal() as db:
        # 현재 캔들 개수 확인
        current_count = await get_candle_count(db, symbol, timeframe)
        print(f"\n[INFO] Current candles in DB: {current_count:,}")
        
        if current_count >= target_count:
            print(f"[OK] Already have enough data ({current_count:,} >= {target_count:,})")
            return
        
        # 가장 오래된 캔들 시간 확인
        oldest_time = await get_oldest_candle_time(db, symbol, timeframe)
        
        if oldest_time:
            print(f"[INFO] Oldest candle: {oldest_time}")
            # 가장 오래된 시간 이전부터 수집
            end_time = oldest_time
        else:
            print("[INFO] No existing data, starting from now")
            end_time = datetime.utcnow()
        
        # 배치 수집 시작
        total_collected = 0
        batch_num = 0
        max_batches = (target_count - current_count) // batch_size + 10
        
        print(f"\n[START] Collecting up to {max_batches} batches...")
        print("-" * 60)
        
        while current_count + total_collected < target_count and batch_num < max_batches:
            batch_num += 1
            
            print(f"  Batch {batch_num}: Collecting before {end_time}...", end=" ", flush=True)
            
            saved = await collect_historical_batch(
                binance=binance,
                db=db,
                symbol=symbol,
                timeframe=timeframe,
                end_time=end_time,
                limit=batch_size
            )
            
            if saved == 0:
                print("No more data available")
                break
            
            total_collected += saved
            print(f"Saved {saved} candles (Total: {current_count + total_collected:,})")
            
            # 다음 배치를 위해 end_time 업데이트
            end_time = end_time - batch_duration
            
            # API 레이트 리밋 방지
            await asyncio.sleep(0.5)
        
        print("-" * 60)
        print(f"\n[DONE] Collection Complete!")
        print(f"  Total collected: {total_collected:,}")
        print(f"  Final count: {current_count + total_collected:,}")
        print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Collect large dataset from Binance")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair")
    parser.add_argument("--timeframe", type=str, default="1m", help="Candle timeframe (1m, 5m, 15m, 1h, etc)")
    parser.add_argument("--target", type=int, default=50000, help="Target number of candles to collect")
    parser.add_argument("--batch", type=int, default=1000, help="Batch size per API call (max 1000)")
    
    args = parser.parse_args()
    
    await collect_large_dataset(
        symbol=args.symbol,
        timeframe=args.timeframe,
        target_count=args.target,
        batch_size=min(args.batch, 1000)
    )


if __name__ == "__main__":
    asyncio.run(main())

