"""Add vector patterns and weights tables

Revision ID: add_vector_patterns_weights
Revises: 7d936f1f64d7
Create Date: 2026-01-22 17:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_vector_patterns_weights'
down_revision = '7d936f1f64d7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # VectorPattern 테이블
    op.create_table(
        'vector_patterns',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('vector_id', sa.Integer()),
        sa.Column('indicators', sa.JSON()),
        sa.Column('signal', sa.String(10)),
        sa.Column('confidence', sa.Float()),
        sa.Column('return_1h', sa.Float()),
        sa.Column('return_4h', sa.Float()),
        sa.Column('return_24h', sa.Float()),
        sa.Column('price_at_signal', sa.Float()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_pattern_symbol_time', 'vector_patterns', ['symbol', 'timeframe', 'timestamp'])
    op.create_index('idx_pattern_vector', 'vector_patterns', ['vector_id'])

    # VectorSimilarity 테이블
    op.create_table(
        'vector_similarities',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('query_pattern_id', sa.Integer(), nullable=False),
        sa.Column('similar_pattern_id', sa.Integer(), nullable=False),
        sa.Column('similarity_score', sa.Float()),
        sa.Column('past_return_1h', sa.Float()),
        sa.Column('past_return_4h', sa.Float()),
        sa.Column('past_return_24h', sa.Float()),
        sa.Column('signal_boosted', sa.Integer(), default=0),
        sa.Column('boost_amount', sa.Float(), default=0),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id')
    )

    # StrategyWeights 테이블 (가중치 설정)
    op.create_table(
        'strategy_weights',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(50), nullable=False, unique=True),
        sa.Column('description', sa.String(255)),
        
        # 기본 가중치
        sa.Column('rsi_weight', sa.Float(), default=0.20),
        sa.Column('macd_weight', sa.Float(), default=0.25),
        sa.Column('bollinger_weight', sa.Float(), default=0.15),
        sa.Column('ema_cross_weight', sa.Float(), default=0.20),
        sa.Column('stochastic_weight', sa.Float(), default=0.10),
        sa.Column('volume_weight', sa.Float(), default=0.10),
        
        # 신호 임계값
        sa.Column('strong_buy_threshold', sa.Float(), default=0.6),
        sa.Column('buy_threshold', sa.Float(), default=0.3),
        sa.Column('sell_threshold', sa.Float(), default=-0.3),
        sa.Column('strong_sell_threshold', sa.Float(), default=-0.6),
        
        # 벡터 패턴 설정
        sa.Column('vector_boost_enabled', sa.Integer(), default=1),
        sa.Column('vector_similarity_threshold', sa.Float(), default=0.75),
        sa.Column('vector_k_nearest', sa.Integer(), default=5),
        sa.Column('max_confidence_boost', sa.Float(), default=0.15),
        
        # 활성화 여부
        sa.Column('active', sa.Integer(), default=1),
        
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_weights_active', 'strategy_weights', ['active'])


def downgrade() -> None:
    op.drop_index('idx_weights_active', table_name='strategy_weights')
    op.drop_table('strategy_weights')
    op.drop_index('idx_pattern_vector', table_name='vector_patterns')
    op.drop_index('idx_pattern_symbol_time', table_name='vector_patterns')
    op.drop_table('vector_similarities')
    op.drop_table('vector_patterns')
