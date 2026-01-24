"""DB 중복 데이터 정리 스크립트"""
import asyncio
from sqlalchemy import text
from app.database import engine

async def cleanup_duplicates():
    async with engine.begin() as conn:
        # 1. 중복 개수 확인
        result = await conn.execute(text('''
            SELECT symbol, timeframe, open_time, COUNT(*) as cnt 
            FROM market_candles 
            GROUP BY symbol, timeframe, open_time 
            HAVING COUNT(*) > 1
            LIMIT 10
        '''))
        duplicates = result.fetchall()
        print(f'Duplicate samples: {len(duplicates)}')
        for d in duplicates[:5]:
            print(f'  {d}')
        
        if duplicates:
            # 2. MySQL DELETE with JOIN
            try:
                delete_result = await conn.execute(text('''
                    DELETE t1 FROM market_candles t1
                    INNER JOIN market_candles t2 
                    WHERE t1.id > t2.id 
                    AND t1.symbol = t2.symbol 
                    AND t1.timeframe = t2.timeframe 
                    AND t1.open_time = t2.open_time
                '''))
                print(f'Deleted duplicate rows: {delete_result.rowcount}')
            except Exception as e:
                print(f'MySQL delete failed, trying SQLite approach: {e}')
                # SQLite approach
                await conn.execute(text('''
                    DELETE FROM market_candles 
                    WHERE id NOT IN (
                        SELECT MIN(id) 
                        FROM market_candles 
                        GROUP BY symbol, timeframe, open_time
                    )
                '''))
                print('Deleted duplicates using SQLite approach')
        
        # 3. 정리 후 확인
        result2 = await conn.execute(text('''
            SELECT symbol, timeframe, COUNT(*) as cnt, 
                   MIN(open_time) as oldest, MAX(open_time) as newest
            FROM market_candles 
            GROUP BY symbol, timeframe
        '''))
        stats = result2.fetchall()
        print(f'\nData after cleanup:')
        for s in stats:
            print(f'  {s[0]} {s[1]}: {s[2]} candles ({s[3]} ~ {s[4]})')

if __name__ == "__main__":
    asyncio.run(cleanup_duplicates())

