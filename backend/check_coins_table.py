"""coins 테이블 존재 여부 및 데이터 확인 스크립트"""
import asyncio
from sqlalchemy import text, inspect
from app.database import engine, AsyncSessionLocal
from app.models.coin import Coin
from sqlalchemy.ext.asyncio import AsyncSession

async def check_coins_table():
    """coins 테이블 확인"""
    async with AsyncSessionLocal() as session:
        try:
            # 1. 테이블 존재 여부 확인
            inspector = inspect(engine.sync_engine)
            tables = inspector.get_table_names()
            print(f"📋 Available tables: {tables}")
            
            if 'coins' not in tables:
                print("❌ 'coins' table does NOT exist!")
                print("💡 Please run: alembic upgrade head")
                return
            
            print("✅ 'coins' table exists")
            
            # 2. 테이블 구조 확인
            columns = inspector.get_columns('coins')
            print(f"\n📊 Table structure:")
            for col in columns:
                print(f"  - {col['name']}: {col['type']}")
            
            # 3. 데이터 개수 확인
            result = await session.execute(text("SELECT COUNT(*) as count FROM coins"))
            count = result.scalar()
            print(f"\n📈 Total coins in DB: {count}")
            
            # 4. 모니터링 코인 확인
            result = await session.execute(text("SELECT COUNT(*) as count FROM coins WHERE is_monitoring = 1"))
            monitoring_count = result.scalar()
            print(f"📈 Monitoring coins: {monitoring_count}")
            
            # 5. 최근 5개 코인 조회
            if count > 0:
                result = await session.execute(
                    text("SELECT id, symbol, base_asset, quote_asset, is_monitoring FROM coins ORDER BY id DESC LIMIT 5")
                )
                coins = result.fetchall()
                print(f"\n📋 Recent 5 coins:")
                for coin in coins:
                    print(f"  - ID: {coin[0]}, Symbol: {coin[1]}, Base: {coin[2]}, Quote: {coin[3]}, Monitoring: {coin[4]}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await engine.dispose()

if __name__ == "__main__":
    asyncio.run(check_coins_table())


