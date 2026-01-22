#!/usr/bin/env python3
"""캔들 데이터 확인"""
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, func, text
from app.models.market_data import MarketCandle
from app.config import get_settings
import sys

async def check_candles():
    settings = get_settings()
    
    # 데이터베이스 엔진 생성
    engine = create_async_engine(settings.database_url, echo=False)
    
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    try:
        async with async_session() as session:
            # 전체 캔들 개수
            result = await session.execute(select(func.count(MarketCandle.id)))
            total = result.scalar()
            print(f"📊 Total candles in DB: {total}")
            
            # 심볼별 캔들 개수
            print("\n📈 Candles by symbol and timeframe:")
            result = await session.execute(
                text("""
                    SELECT symbol, timeframe, COUNT(*) as count 
                    FROM market_candles 
                    GROUP BY symbol, timeframe 
                    ORDER BY count DESC
                """)
            )
            for row in result:
                print(f"   {row[0]} {row[1]}: {row[2]} candles")
            
            # 최근 저장된 캔들 5개
            print("\n🕐 Most recent candles:")
            result = await session.execute(
                select(MarketCandle)
                .order_by(MarketCandle.created_at.desc())
                .limit(5)
            )
            for candle in result:
                print(f"   {candle.symbol} {candle.timeframe} @ {candle.open_time}: {candle.open} -> {candle.close} (vol: {candle.volume})")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(check_candles())
