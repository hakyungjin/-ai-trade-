"""
모니터링 중인 코인들의 5분봉 데이터 수집 스크립트
- DB에서 모니터링 중인 코인 목록 조회
- 각 코인의 5분봉 데이터 수집 (현물/선물 자동 구분)
- AI 학습용 데이터 준비
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime, timedelta, timezone
from typing import Union

# backend 디렉토리를 경로에 추가 (app 모듈 import용)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
backend_path = os.path.join(project_root, 'backend')
sys.path.insert(0, backend_path)

from app.database import AsyncSessionLocal, init_db
from app.services.binance_service import BinanceService
from app.services.binance_futures_service import BinanceFuturesService
from app.models.market_data import MarketCandle
from app.models.coin import Coin
from app.config import get_settings
from sqlalchemy import select, func


# 서비스 타입 힌트용 Union
ServiceType = Union[BinanceService, BinanceFuturesService]


async def get_monitored_coins(db) -> list:
    """모니터링 중인 코인 목록 조회 (symbol + market_type)"""
    stmt = select(Coin).where(Coin.is_active == True)
    result = await db.execute(stmt)
    coins = result.scalars().all()
    return [{"symbol": coin.symbol, "market_type": coin.market_type or "spot"} for coin in coins]


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
    service: ServiceType,
    db,
    symbol: str,
    timeframe: str,
    end_time: datetime,
    market_type: str = "spot",
    limit: int = 1000
) -> int:
    """특정 시점 이전 데이터 배치 수집 (현물/선물 자동 분기)"""
    try:
        # end_time을 밀리초로 변환
        end_ms = int(end_time.timestamp() * 1000)
        
        # 마켓 타입에 따라 다른 API 호출
        if market_type == "futures":
            klines = await service.get_futures_klines(
                symbol=symbol,
                interval=timeframe,
                limit=limit,
                endTime=end_ms
            )
        else:
            klines = await service.get_klines(
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
                print(f"    [WARN] Error saving candle: {e}")
                continue
        
        if saved_count > 0:
            await db.commit()
        
        return saved_count
        
    except Exception as e:
        print(f"    [ERROR] Batch collection error: {e}")
        await db.rollback()
        return 0


async def collect_coin_data(
    service: ServiceType,
    db,
    symbol: str,
    timeframe: str = "5m",
    market_type: str = "spot",
    target_count: int = 10000,
    batch_size: int = 1000
) -> int:
    """단일 코인 데이터 수집 (현물/선물 자동 분기)"""
    
    # 타임프레임별 분 단위
    timeframe_minutes = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "1d": 1440,
    }
    
    minutes = timeframe_minutes.get(timeframe, 5)
    batch_duration = timedelta(minutes=minutes * batch_size)
    
    # 현재 캔들 개수 확인
    current_count = await get_candle_count(db, symbol, timeframe)
    
    if current_count >= target_count:
        print(f"  [SKIP] {symbol}: Already have {current_count:,} candles")
        return 0
    
    # 가장 오래된 캔들 시간 확인
    oldest_time = await get_oldest_candle_time(db, symbol, timeframe)
    
    if oldest_time:
        end_time = oldest_time
    else:
        end_time = datetime.now(timezone.utc).replace(tzinfo=None)
    
    # 배치 수집
    total_collected = 0
    batch_num = 0
    max_batches = (target_count - current_count) // batch_size + 5
    
    while current_count + total_collected < target_count and batch_num < max_batches:
        batch_num += 1
        
        saved = await collect_historical_batch(
            service=service,
            db=db,
            symbol=symbol,
            timeframe=timeframe,
            end_time=end_time,
            market_type=market_type,
            limit=batch_size
        )
        
        if saved == 0:
            break
        
        total_collected += saved
        
        # 다음 배치를 위해 end_time 업데이트
        end_time = end_time - batch_duration
        
        # API 레이트 리밋 방지
        await asyncio.sleep(0.3)
    
    return total_collected


async def main():
    parser = argparse.ArgumentParser(description="Collect candles for monitored coins (spot/futures)")
    parser.add_argument("--timeframe", type=str, default="5m", help="Candle timeframe (default: 5m)")
    parser.add_argument("--target", type=int, default=10000, help="Target candles per coin (default: 10000)")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols (empty = all monitored)")
    parser.add_argument("--market", type=str, default="", help="Force market type: spot or futures (empty = auto from DB)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  Monitored Coins Candle Data Collection (Spot/Futures)")
    print(f"  Timeframe: {args.timeframe}, Target per coin: {args.target:,}")
    print("=" * 70)
    
    # DB 초기화
    await init_db()
    
    # Binance 서비스 초기화 (현물 + 선물)
    settings = get_settings()
    spot_service = BinanceService(
        api_key=settings.binance_api_key,
        secret_key=settings.binance_secret_key,
        testnet=False
    )
    futures_service = BinanceFuturesService(
        api_key=settings.binance_api_key,
        secret_key=settings.binance_secret_key
    )
    
    async with AsyncSessionLocal() as db:
        # 코인 목록 결정
        if args.symbols:
            # 명시적 심볼 지정 시
            symbol_list = [s.strip().upper() for s in args.symbols.split(",")]
            market_type = args.market if args.market else "spot"
            coins = [{"symbol": s, "market_type": market_type} for s in symbol_list]
            print(f"\n[INFO] Using specified symbols ({market_type}): {symbol_list}")
        else:
            coins = await get_monitored_coins(db)
            if not coins:
                # 기본 코인 목록 (현물)
                coins = [
                    {"symbol": "BTCUSDT", "market_type": "spot"},
                    {"symbol": "ETHUSDT", "market_type": "spot"},
                    {"symbol": "BNBUSDT", "market_type": "spot"},
                    {"symbol": "SOLUSDT", "market_type": "spot"},
                    {"symbol": "XRPUSDT", "market_type": "spot"},
                ]
                print(f"\n[INFO] No monitored coins found, using defaults")
            else:
                symbol_names = [c["symbol"] for c in coins]
                print(f"\n[INFO] Found {len(coins)} monitored coins: {symbol_names}")
        
        # 현물/선물 구분 표시
        spot_coins = [c for c in coins if c["market_type"] == "spot"]
        futures_coins = [c for c in coins if c["market_type"] == "futures"]
        print(f"  - Spot: {len(spot_coins)} coins")
        print(f"  - Futures: {len(futures_coins)} coins")
        print("-" * 70)
        
        # 각 코인별 수집
        results = {}
        for i, coin_info in enumerate(coins, 1):
            symbol = coin_info["symbol"]
            market_type = coin_info["market_type"]
            
            # 적절한 서비스 선택
            service = futures_service if market_type == "futures" else spot_service
            market_label = "[F]" if market_type == "futures" else "[S]"
            
            print(f"\n[{i}/{len(coins)}] {market_label} Collecting {symbol} {args.timeframe}...")
            
            try:
                collected = await collect_coin_data(
                    service=service,
                    db=db,
                    symbol=symbol,
                    timeframe=args.timeframe,
                    market_type=market_type,
                    target_count=args.target
                )
                
                final_count = await get_candle_count(db, symbol, args.timeframe)
                results[symbol] = {"collected": collected, "total": final_count, "market": market_type}
                
                print(f"  [OK] {symbol}: +{collected:,} candles, Total: {final_count:,}")
                
            except Exception as e:
                print(f"  [ERROR] {symbol}: {e}")
                results[symbol] = {"collected": 0, "total": 0, "error": str(e), "market": market_type}
        
        # 결과 요약
        print("\n" + "=" * 70)
        print("  Collection Summary")
        print("=" * 70)
        
        total_collected = 0
        for symbol, data in results.items():
            status = "OK" if data.get("collected", 0) > 0 or data.get("total", 0) > 0 else "FAIL"
            market_label = "[F]" if data.get("market") == "futures" else "[S]"
            print(f"  [{status}] {market_label} {symbol}: {data.get('total', 0):,} candles")
            total_collected += data.get("collected", 0)
        
        print("-" * 70)
        print(f"  Total new candles: {total_collected:,}")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())


