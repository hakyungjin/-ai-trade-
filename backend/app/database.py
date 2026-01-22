from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.config import get_settings

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
    """마이그레이션으로 테이블이 생성됩니다. 이 함수는 더 이상 사용되지 않습니다."""
    # Alembic을 통한 마이그레이션으로 테이블 관리
    pass


async def close_db():
    """데이터베이스 연결 종료"""
    await engine.dispose()
