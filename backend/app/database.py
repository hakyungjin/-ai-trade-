import logging
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.config import get_settings

logger = logging.getLogger(__name__)

# Cloud Run에서는 echo=False로 설정하여 로그 과다 출력 방지
is_cloud_run = os.getenv('K_SERVICE') is not None

try:
    settings = get_settings()
    database_url = settings.database_url

    # DATABASE_URL이 기본값이면 경고
    if "localhost" in database_url and is_cloud_run:
        logger.warning("⚠️ Using localhost DATABASE_URL in Cloud Run - this will not work!")
        logger.warning("⚠️ Please set DATABASE_URL environment variable")

    # PostgreSQL과 MySQL에 따라 다른 connect_args 사용
    connect_args = {}
    if "mysql" in database_url or "aiomysql" in database_url:
        connect_args = {"timeout": 10}  # MySQL/aiomysql 전용
    elif "postgresql" in database_url or "asyncpg" in database_url:
        connect_args = {"timeout": 10.0, "command_timeout": 10.0}  # PostgreSQL/asyncpg 전용

    engine = create_async_engine(
        database_url,
        echo=not is_cloud_run,  # Cloud Run에서는 echo 비활성화
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        connect_args=connect_args
    )
    logger.info(f"✅ Database engine created (Cloud Run: {is_cloud_run})")
except Exception as e:
    logger.error(f"❌ Failed to create database engine: {e}")
    # 기본 엔진 생성 (사용하지 않음)
    engine = None

if engine:
    AsyncSessionLocal = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
else:
    AsyncSessionLocal = None

Base = declarative_base()


async def get_db():
    """의존성 주입용 DB 세션 생성기"""
    if AsyncSessionLocal is None:
        raise RuntimeError("Database not initialized")

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
    """앱 시작 - DB 연결 테스트"""
    if engine is None:
        logger.warning("⚠️ Database engine not initialized - skipping connection test")
        return

    try:
        # 간단한 연결 테스트
        async with engine.begin() as conn:
            from sqlalchemy import text
            await conn.execute(text("SELECT 1"))
        logger.info("✅ Database connection verified")
    except Exception as e:
        logger.warning(f"⚠️ Database connection test failed: {e}")


async def close_db():
    """데이터베이스 연결 종료"""
    if engine:
        await engine.dispose()
