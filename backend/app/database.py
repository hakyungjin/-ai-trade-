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
    """앱 시작 시 자동으로 Alembic 마이그레이션 실행"""
    try:
        # Alembic 디렉토리 경로
        alembic_dir = os.path.join(os.path.dirname(__file__), '..', 'alembic')
        
        # alembic upgrade head 명령 실행
        result = subprocess.run(
            ['alembic', 'upgrade', 'head'],
            cwd=alembic_dir,
            capture_output=True,
            text=True,
            env={**os.environ, 'DATABASE_URL': settings.database_url}
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
