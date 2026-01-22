"""add_coin_metadata_tables

Revision ID: add_coin_metadata_v1
Revises: optimize_candle_v1
Create Date: 2025-01-22 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_coin_metadata_v1'
down_revision = 'optimize_candle_v1'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """코인 메타데이터 테이블 생성"""
    
    # ===== 1️⃣ Coin 테이블 생성 =====
    op.create_table(
        'coins',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('base_asset', sa.String(20), nullable=False),
        sa.Column('quote_asset', sa.String(20), nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('is_monitoring', sa.Boolean(), default=False),
        sa.Column('full_name', sa.String(100)),
        sa.Column('description', sa.String(500)),
        sa.Column('current_price', sa.Float()),
        sa.Column('price_change_24h', sa.Float()),
        sa.Column('volume_24h', sa.Float()),
        sa.Column('market_cap', sa.Float()),
        sa.Column('last_price_update', sa.DateTime()),
        sa.Column('candle_count', sa.Integer(), default=0),
        sa.Column('earliest_candle_time', sa.DateTime()),
        sa.Column('latest_candle_time', sa.DateTime()),
        sa.Column('monitoring_timeframes', sa.JSON()),
        sa.Column('priority', sa.Integer(), default=0),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('last_analysis_at', sa.DateTime()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', name='uq_coin_symbol')
    )
    op.create_index('idx_coin_active', 'coins', ['is_active'])
    op.create_index('idx_coin_monitoring', 'coins', ['is_monitoring'])
    op.create_index('idx_coin_priority', 'coins', ['priority'])
    op.create_index('idx_coin_symbol', 'coins', ['symbol'])
    
    # ===== 2️⃣ CoinStatistics 테이블 생성 =====
    op.create_table(
        'coin_statistics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('coin_id', sa.Integer(), nullable=False),
        sa.Column('total_candles', sa.Integer(), default=0),
        sa.Column('candles_1h', sa.Integer(), default=0),
        sa.Column('candles_4h', sa.Integer(), default=0),
        sa.Column('candles_1d', sa.Integer(), default=0),
        sa.Column('total_signals', sa.Integer(), default=0),
        sa.Column('buy_signals', sa.Integer(), default=0),
        sa.Column('sell_signals', sa.Integer(), default=0),
        sa.Column('neutral_signals', sa.Integer(), default=0),
        sa.Column('average_confidence', sa.Float()),
        sa.Column('win_rate', sa.Float()),
        sa.Column('total_returns', sa.Float()),
        sa.Column('pattern_vectors_count', sa.Integer(), default=0),
        sa.Column('similar_patterns_found', sa.Integer(), default=0),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['coin_id'], ['coins.id']),
        sa.UniqueConstraint('coin_id', name='uq_coin_stats_coin_id')
    )
    op.create_index('idx_coin_stats_coin_id', 'coin_statistics', ['coin_id'])
    
    # ===== 3️⃣ CoinAnalysisConfig 테이블 생성 =====
    op.create_table(
        'coin_analysis_configs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('coin_id', sa.Integer(), nullable=False),
        sa.Column('use_rsi', sa.Boolean(), default=True),
        sa.Column('use_macd', sa.Boolean(), default=True),
        sa.Column('use_bollinger', sa.Boolean(), default=True),
        sa.Column('use_moving_average', sa.Boolean(), default=True),
        sa.Column('use_stochastic', sa.Boolean(), default=True),
        sa.Column('use_gemini_ai', sa.Boolean(), default=True),
        sa.Column('use_local_model', sa.Boolean(), default=True),
        sa.Column('use_vector_patterns', sa.Boolean(), default=False),
        sa.Column('buy_threshold', sa.Float(), default=0.3),
        sa.Column('strong_buy_threshold', sa.Float(), default=0.6),
        sa.Column('sell_threshold', sa.Float(), default=-0.3),
        sa.Column('strong_sell_threshold', sa.Float(), default=-0.6),
        sa.Column('notify_on_strong_signals', sa.Boolean(), default=True),
        sa.Column('notify_on_pattern_found', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['coin_id'], ['coins.id']),
        sa.UniqueConstraint('coin_id', name='uq_coin_config_coin_id')
    )
    op.create_index('idx_coin_config_coin_id', 'coin_analysis_configs', ['coin_id'])
    
    # ===== 4️⃣ CoinPriceHistory 테이블 생성 =====
    op.create_table(
        'coin_price_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('coin_id', sa.Integer(), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('price_change_24h', sa.Float()),
        sa.Column('volume_24h', sa.Float()),
        sa.Column('market_cap', sa.Float()),
        sa.Column('recorded_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['coin_id'], ['coins.id'])
    )
    op.create_index('idx_price_coin', 'coin_price_history', ['coin_id'])
    op.create_index('idx_price_recorded_at', 'coin_price_history', ['recorded_at'])
    op.create_index('idx_price_coin_time', 'coin_price_history', ['coin_id', 'recorded_at'])
    
    print("✅ Migration complete: Coin metadata tables created")


def downgrade() -> None:
    """코인 메타데이터 테이블 제거"""
    
    op.drop_index('idx_price_coin_time', 'coin_price_history')
    op.drop_index('idx_price_recorded_at', 'coin_price_history')
    op.drop_index('idx_price_coin', 'coin_price_history')
    op.drop_table('coin_price_history')
    
    op.drop_index('idx_coin_config_coin_id', 'coin_analysis_configs')
    op.drop_table('coin_analysis_configs')
    
    op.drop_index('idx_coin_stats_coin_id', 'coin_statistics')
    op.drop_table('coin_statistics')
    
    op.drop_index('idx_coin_symbol', 'coins')
    op.drop_index('idx_coin_priority', 'coins')
    op.drop_index('idx_coin_monitoring', 'coins')
    op.drop_index('idx_coin_active', 'coins')
    op.drop_table('coins')
    
    print("✅ Downgrade complete: Coin metadata tables removed")
