"""Add unified model support: AssetMapping table and market_type fields

Revision ID: add_unified_model_support
Revises: add_trained_models
Create Date: 2026-01-31

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_unified_model_support'
down_revision = 'add_trained_models'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ===== 1. AssetMapping 테이블 생성 =====
    op.create_table(
        'asset_mappings',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('symbol', sa.String(50), nullable=False),
        sa.Column('asset_id', sa.Integer(), nullable=False),
        sa.Column('market_type', sa.String(20), nullable=False),
        sa.Column('category', sa.String(50), nullable=True),
        sa.Column('full_name', sa.String(200), nullable=True),
        sa.Column('description', sa.String(500), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', name='uq_asset_mappings_symbol'),
        sa.UniqueConstraint('asset_id', name='uq_asset_mappings_asset_id'),
    )

    # 인덱스 생성
    op.create_index('ix_asset_mappings_symbol', 'asset_mappings', ['symbol'])
    op.create_index('ix_asset_mappings_asset_id', 'asset_mappings', ['asset_id'])
    op.create_index('ix_asset_mappings_market_type', 'asset_mappings', ['market_type'])

    # ===== 2. MarketCandle 테이블에 market_type, is_market_open 추가 =====
    op.add_column('market_candles', sa.Column('market_type', sa.String(20), nullable=True, server_default='crypto'))
    op.add_column('market_candles', sa.Column('is_market_open', sa.Boolean(), nullable=True, server_default=sa.true()))

    # 기존 데이터에 대해 기본값 설정 (NULL → 'crypto', TRUE)
    op.execute("UPDATE market_candles SET market_type = 'crypto' WHERE market_type IS NULL")
    op.execute("UPDATE market_candles SET is_market_open = TRUE WHERE is_market_open IS NULL")

    # 인덱스 추가
    op.create_index('ix_market_candles_market_type', 'market_candles', ['market_type'])

    # ===== 3. TrainedModel 테이블에 통합 모델 관련 필드 추가 =====
    op.add_column('trained_models', sa.Column('is_unified', sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column('trained_models', sa.Column('supported_assets', sa.JSON(), nullable=True))
    op.add_column('trained_models', sa.Column('asset_count', sa.Integer(), nullable=True))
    op.add_column('trained_models', sa.Column('embedding_dim', sa.Integer(), nullable=True))

    # 인덱스 추가
    op.create_index('ix_trained_models_is_unified', 'trained_models', ['is_unified'])


def downgrade() -> None:
    # TrainedModel 필드 제거
    op.drop_index('ix_trained_models_is_unified', table_name='trained_models')
    op.drop_column('trained_models', 'embedding_dim')
    op.drop_column('trained_models', 'asset_count')
    op.drop_column('trained_models', 'supported_assets')
    op.drop_column('trained_models', 'is_unified')

    # MarketCandle 필드 제거
    op.drop_index('ix_market_candles_market_type', table_name='market_candles')
    op.drop_column('market_candles', 'is_market_open')
    op.drop_column('market_candles', 'market_type')

    # AssetMapping 테이블 제거
    op.drop_index('ix_asset_mappings_market_type', table_name='asset_mappings')
    op.drop_index('ix_asset_mappings_asset_id', table_name='asset_mappings')
    op.drop_index('ix_asset_mappings_symbol', table_name='asset_mappings')
    op.drop_table('asset_mappings')
