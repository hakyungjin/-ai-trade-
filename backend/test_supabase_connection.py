import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine
from dotenv import load_dotenv

load_dotenv(".env.production")

async def test_connection():
    # Supabase connection pooler (PgBouncer) - port 6543
    database_url = "postgresql+asyncpg://postgres:ha89498912%40%40@db.vmiinfjxpnoevsehhzey.supabase.co:6543/postgres"
    
    print(f"🔗 Connecting to Supabase PgBouncer: {database_url.split('@')[1] if '@' in database_url else 'hidden'}")
    
    try:
        engine = create_async_engine(
            database_url,
            echo=False,
            connect_args={
                "command_timeout": 10,
            },
        )
        
        async with engine.begin() as conn:
            result = await conn.execute("SELECT 1;")
            print("✅ Connection successful!")
            print(f"Result: {result.fetchone()}")
        
        await engine.dispose()
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_connection())
