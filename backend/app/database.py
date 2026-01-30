import asyncio
import subprocess
import os
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    echo=True,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()


async def get_db():
    """의존성 주입용 DB 세션 생성기"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """앱 시작 시 자동으로 Alembic 마이그레이션을 백그라운드에서 실행"""
    # 마이그레이션을 백그라운드에서 실행 (앱 시작을 차단하지 않음)
    asyncio.create_task(run_migrations_async())


async def run_migrations_async():
    """백그라운드에서 마이그레이션 실행"""
    try:
        # 5초 대기 (앱이 시작되도록)
        await asyncio.sleep(5)
        
        # Alembic 디렉토리 경로
        alembic_dir = os.path.join(os.path.dirname(__file__), '..', 'alembic')
        
        # alembic upgrade head 명령 실행
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            subprocess.run,
            ['alembic', 'upgrade', 'head'],
            alembic_dir,  # cwd
            subprocess.PIPE,  # stdout
            subprocess.PIPE,  # stderr
            False  # text
        )
        
        if result.returncode == 0:
            logger.info("✅ Database migrations applied successfully")
        else:
            logger.error(f"❌ Migration failed: {result.stderr}")
    except Exception as e:
        logger.error(f"❌ Error running migrations: {e}")


async def close_db():
    """데이터베이스 연결 종료"""
    await engine.dispose()
