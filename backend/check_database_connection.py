"""실제 연결된 데이터베이스 확인"""
import asyncio
from sqlalchemy import text, inspect
from app.database import engine, AsyncSessionLocal
from app.config import get_settings

async def check_database():
    """실제 연결된 데이터베이스 확인"""
    settings = get_settings()
    
    print("=" * 60)
    print("🔍 데이터베이스 연결 정보 확인")
    print("=" * 60)
    print(f"📝 Config database_url: {settings.database_url}")
    print(f"📝 Config sqlalchemy_url: {settings.sqlalchemy_url}")
    print()
    
    try:
        async with AsyncSessionLocal() as session:
            # 1. 실제 연결된 DB 정보 확인
            result = await session.execute(text("SELECT DATABASE()"))
            db_name = result.scalar()
            print(f"✅ 현재 연결된 데이터베이스: {db_name}")
            
            # 2. DB 버전 확인
            result = await session.execute(text("SELECT VERSION()"))
            version = result.scalar()
            print(f"✅ 데이터베이스 버전: {version}")
            
            # 3. 테이블 목록 확인
            inspector = inspect(engine.sync_engine)
            tables = inspector.get_table_names()
            print(f"\n📋 데이터베이스의 테이블 목록 ({len(tables)}개):")
            for table in sorted(tables):
                print(f"  - {table}")
            
            # 4. coins 테이블 확인
            if 'coins' in tables:
                print(f"\n✅ 'coins' 테이블이 존재합니다")
                
                # coins 테이블 데이터 개수
                result = await session.execute(text("SELECT COUNT(*) FROM coins"))
                count = result.scalar()
                print(f"📊 coins 테이블의 데이터 개수: {count}")
                
                # 모니터링 코인 개수
                result = await session.execute(
                    text("SELECT COUNT(*) FROM coins WHERE is_monitoring = 1")
                )
                monitoring_count = result.scalar()
                print(f"📊 모니터링 중인 코인 개수: {monitoring_count}")
                
                # 최근 5개 코인
                if count > 0:
                    result = await session.execute(
                        text("""
                            SELECT id, symbol, base_asset, quote_asset, is_monitoring, created_at 
                            FROM coins 
                            ORDER BY id DESC 
                            LIMIT 5
                        """)
                    )
                    coins = result.fetchall()
                    print(f"\n📋 최근 추가된 5개 코인:")
                    for coin in coins:
                        print(f"  - ID: {coin[0]}, Symbol: {coin[1]}, Base: {coin[2]}, Quote: {coin[3]}, Monitoring: {coin[4]}, Created: {coin[5]}")
            else:
                print(f"\n❌ 'coins' 테이블이 존재하지 않습니다!")
                print("💡 마이그레이션을 실행하세요: alembic upgrade head")
            
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        
        # 에러 타입 확인
        error_str = str(e).lower()
        if 'sqlite' in error_str or 'trading.db' in error_str:
            print("\n⚠️ SQLite 관련 에러가 감지되었습니다!")
            print("💡 MySQL이 아닌 SQLite에 연결하려고 시도한 것 같습니다.")
        elif 'mysql' in error_str or 'mariadb' in error_str:
            print("\n⚠️ MySQL 연결 에러가 감지되었습니다!")
            print("💡 MySQL 서버가 실행 중인지 확인하세요.")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(check_database())


