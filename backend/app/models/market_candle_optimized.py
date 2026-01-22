"""
캔들 데이터 저장 최적화 구현
- 인덱싱 최적화
- 복합 인덱스 추가
- 쿼리 성능 향상
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Index, BigInteger, UniqueConstraint
from sqlalchemy.sql import func
from app.database import Base


class MarketCandle(Base):
    """캔들 데이터 (OHLCV) - 최적화된 버전"""
    __tablename__ = "market_candles"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)  # ✅ 심볼 인덱스 추가
    timeframe = Column(String(10), nullable=False, index=True)  # ✅ 타임프레임 인덱스
    open_time = Column(DateTime, nullable=False, index=True)  # ✅ 시간 인덱스 (내림차순으로 검색)
    
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    close_time = Column(DateTime)
    quote_volume = Column(Float)
    trades_count = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())

    # ✅ 복합 인덱스들 (쿼리 성능 중요)
    __table_args__ = (
        # 기존 인덱스
        Index('idx_candle_symbol_time', 'symbol', 'timeframe', 'open_time'),
        
        # ✅ 추가 복합 인덱스 (최적화)
        Index('idx_candle_symbol_timeframe', 'symbol', 'timeframe'),
        Index('idx_candle_time_desc', 'open_time'),  # 최근 캔들 빠른 조회
        Index('idx_candle_symbol_time_desc', 'symbol', 'open_time'),  # 심볼별 최신 캔들
        
        # ✅ UNIQUE 제약 (중복 방지)
        UniqueConstraint('symbol', 'timeframe', 'open_time', name='uq_candle_symbol_tf_time'),
    )


# ===== 아래는 마이그레이션 파일에 포함될 최적화 SQL =====

OPTIMIZATION_SQL = """
-- ===== 1️⃣ 인덱싱 최적화 =====

-- 단일 컬럼 인덱스 추가
CREATE INDEX IF NOT EXISTS idx_candle_symbol ON market_candles(symbol);
CREATE INDEX IF NOT EXISTS idx_candle_timeframe ON market_candles(timeframe);
CREATE INDEX IF NOT EXISTS idx_candle_open_time ON market_candles(open_time DESC);

-- 복합 인덱스 추가 (특정 심볼과 타임프레임의 최신 캔들 빠르게 조회)
CREATE INDEX IF NOT EXISTS idx_candle_symbol_tf_time 
ON market_candles(symbol, timeframe, open_time DESC);

-- 심볼별 최신 캔들 빠르게 조회
CREATE INDEX IF NOT EXISTS idx_candle_symbol_time_desc 
ON market_candles(symbol, open_time DESC);

-- 전체 조회 성능 개선
CREATE INDEX IF NOT EXISTS idx_candle_all_symbols_time 
ON market_candles(open_time DESC, symbol, timeframe);

-- ===== 2️⃣ 중복 방지 =====

-- 중복 삽입 방지 (같은 심볼/타임프레임/시간의 캔들)
ALTER TABLE market_candles 
ADD CONSTRAINT uq_candle_symbol_tf_time 
UNIQUE (symbol, timeframe, open_time);

-- ===== 3️⃣ 테이블 통계 갱신 =====
ANALYZE TABLE market_candles;

-- ===== 4️⃣ 확인 쿼리 =====

-- 현재 인덱스 확인
SHOW INDEXES FROM market_candles;

-- 테이블 용량 확인
SELECT 
    table_name,
    ROUND(((data_length + index_length) / 1024 / 1024), 2) as size_mb,
    TABLE_ROWS as row_count
FROM information_schema.TABLES 
WHERE table_schema = 'crypto_trader' 
AND table_name = 'market_candles';

-- 심볼별 캔들 개수 확인
SELECT 
    symbol,
    COUNT(*) as candle_count,
    COUNT(DISTINCT timeframe) as timeframe_count,
    MIN(open_time) as earliest,
    MAX(open_time) as latest
FROM market_candles
GROUP BY symbol
ORDER BY candle_count DESC;

-- ===== 5️⃣ 쿼리 성능 테스트 =====

-- 테스트: 특정 심볼의 1h 봉 최신 100개 조회
EXPLAIN SELECT * FROM market_candles 
WHERE symbol = 'BTCUSDT' 
AND timeframe = '1h'
ORDER BY open_time DESC
LIMIT 100;

-- ===== 6️⃣ 데이터 아카이빙 (선택사항) =====

-- 30일 이상 오래된 데이터 확인
SELECT COUNT(*) as old_candle_count
FROM market_candles
WHERE open_time < DATE_SUB(NOW(), INTERVAL 30 DAY);

-- 오래된 데이터 별도 테이블로 이동 (옵션)
-- CREATE TABLE market_candles_archive LIKE market_candles;
-- INSERT INTO market_candles_archive 
-- SELECT * FROM market_candles 
-- WHERE open_time < DATE_SUB(NOW(), INTERVAL 30 DAY);
-- DELETE FROM market_candles 
-- WHERE open_time < DATE_SUB(NOW(), INTERVAL 30 DAY);
"""


# ===== 자동 정리 서비스 =====

class CandleMaintenanceService:
    """캔들 데이터 정기 유지보수"""
    
    @staticmethod
    async def cleanup_old_candles(db_session, days_threshold: int = 90):
        """
        오래된 캔들 데이터 정리 (아카이빙)
        
        Args:
            db_session: DB 세션
            days_threshold: 이 일수 이상 오래된 데이터 이동
        """
        from datetime import timedelta, datetime
        from sqlalchemy import delete
        
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        
        # 오래된 데이터 개수 확인
        result = db_session.execute(
            f"SELECT COUNT(*) FROM market_candles WHERE open_time < '{cutoff_date}'"
        )
        count = result.scalar()
        
        if count > 0:
            print(f"🗑️ Found {count} old candles (older than {days_threshold} days)")
            # 여기서 아카이빙 처리
            # 구현: INSERT INTO archive 후 DELETE
        
        return count
    
    @staticmethod
    async def optimize_indexes(db_session):
        """인덱스 최적화"""
        print("🔧 Optimizing indexes...")
        db_session.execute("ANALYZE TABLE market_candles")
        print("✅ Index optimization complete")
    
    @staticmethod
    async def get_table_stats(db_session):
        """테이블 통계 조회"""
        result = db_session.execute("""
            SELECT 
                symbol,
                timeframe,
                COUNT(*) as count,
                MIN(open_time) as earliest,
                MAX(open_time) as latest
            FROM market_candles
            GROUP BY symbol, timeframe
            ORDER BY count DESC
        """)
        return result.fetchall()
