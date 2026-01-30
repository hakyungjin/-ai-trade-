from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.config import get_settings
import logging

logger = logging.getLogger(__name__)

settings = get_settings()

# Database URL이 없으면 기본값 사용 (로컬 개발용)
# Supports both MySQL (mysql+aiomysql://) and PostgreSQL (postgresql+asyncpg://)
database_url = settings.database_url or "mysql+aiomysql://root:password@localhost:3306/crypto_trader"

try:
    engine = create_async_engine(
        database_url,
        echo=False,  # Cloud Run에서는 echo=False로 설정 (로그 최소화)
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        connect_args={"connect_timeout": 10}  # 연결 타임아웃 설정
    )
    logger.info(f"Database engine created with URL: {database_url.split('@')[1] if '@' in database_url else 'configured'}")
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    engine = None

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()


async def get_db():
    """의존성 주입용 DB 세션 생성기"""
    if engine is None:
        raise Exception("Database engine not initialized. Please check DATABASE_URL environment variable.")
    
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
    """테이블 자동 생성 (개발용)"""
    if engine is None:
        logger.warning("Database engine not initialized, skipping table creation")
        return
    
    try:
        # 모델 import (테이블 생성을 위해)
        from app.models.market_data import MarketCandle
        from app.models.vector_pattern import VectorPattern
        from app.models.trade_feedback import TradeFeedback
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}")
        raise


async def close_db():
    """데이터베이스 연결 종료"""
    if engine:
        await engine.dispose()
