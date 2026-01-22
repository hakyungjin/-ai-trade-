"""optimize_candle_indexes

Revision ID: optimize_candle_v1
Revises: 
Create Date: 2025-01-22 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'optimize_candle_v1'
down_revision = 'add_vector_patterns_weights'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """캔들 데이터 저장 최적화: 인덱싱 개선"""
    
    # ===== 1️⃣ 인덱스 추가 (MySQL 호환성) =====
    
    try:
        # symbol 단일 인덱스
        op.create_index('idx_candle_symbol', 'market_candles', ['symbol'])
    except:
        pass
    
    try:
        # timeframe 단일 인덱스
        op.create_index('idx_candle_timeframe', 'market_candles', ['timeframe'])
    except:
        pass
    
    try:
        # open_time 단일 인덱스
        op.create_index('idx_candle_time_desc', 'market_candles', ['open_time'])
    except:
        pass
    
    try:
        # 복합 인덱스: symbol, timeframe, open_time
        op.create_index(
            'idx_candle_symbol_tf_time_desc',
            'market_candles',
            ['symbol', 'timeframe', 'open_time']
        )
    except:
        pass
    
    try:
        # 복합 인덱스: symbol, open_time
        op.create_index(
            'idx_candle_symbol_time_desc',
            'market_candles',
            ['symbol', 'open_time']
        )
    except:
        pass
    
    try:
        # 복합 인덱스: open_time, symbol, timeframe
        op.create_index(
            'idx_candle_all_symbols_time',
            'market_candles',
            ['open_time', 'symbol', 'timeframe']
        )
    except:
        pass
    
    # ===== 2️⃣ UNIQUE 제약 추가 (중복 방지) =====
    try:
        op.create_unique_constraint(
            'uq_candle_symbol_tf_time',
            'market_candles',
            ['symbol', 'timeframe', 'open_time']
        )
        print("✅ Created unique constraint: uq_candle_symbol_tf_time")
    except Exception as e:
        print(f"⚠️ Unique constraint already exists or duplicate data: {e}")
    
    # ===== 3️⃣ 테이블 통계 갱신 =====
    try:
        op.execute("ANALYZE TABLE market_candles")
        print("✅ Analyzed table: market_candles")
    except:
        pass
    
    print("✅ Migration complete: Candle indexes optimized")


def downgrade() -> None:
    """인덱스 최적화 롤백"""
    
    # 인덱스 제거 (존재하지 않으면 무시)
    try:
        op.drop_index('idx_candle_symbol', 'market_candles')
    except:
        pass
    
    try:
        op.drop_index('idx_candle_timeframe', 'market_candles')
    except:
        pass
    
    try:
        op.drop_index('idx_candle_time_desc', 'market_candles')
    except:
        pass
    
    try:
        op.drop_index('idx_candle_symbol_tf_time_desc', 'market_candles')
    except:
        pass
    
    try:
        op.drop_index('idx_candle_symbol_time_desc', 'market_candles')
    except:
        pass
    
    try:
        op.drop_index('idx_candle_all_symbols_time', 'market_candles')
    except:
        pass
    
    # UNIQUE 제약 제거
    try:
        op.drop_constraint('uq_candle_symbol_tf_time', 'market_candles', type_='unique')
    except:
        pass
    
    print("✅ Downgrade complete: Candle indexes optimization reverted")
