"""add market_type to coins

Revision ID: 3923aa842786
Revises: add_coin_metadata_v1
Create Date: 2026-01-24 00:27:46.821810

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3923aa842786'
down_revision: Union[str, Sequence[str], None] = 'add_coin_metadata_v1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    conn = op.get_bind()
    
    # 1. market_type 컬럼이 이미 있는지 확인
    result = conn.execute(sa.text(
        "SELECT COUNT(*) FROM information_schema.columns "
        "WHERE table_schema = DATABASE() AND table_name = 'coins' AND column_name = 'market_type'"
    ))
    column_exists = result.scalar() > 0
    
    if not column_exists:
        # coins 테이블에 market_type 컬럼 추가 (기본값 'spot')
        op.add_column('coins', sa.Column('market_type', sa.String(length=10), nullable=True, server_default='spot'))
    
    # 기존 데이터 업데이트
    op.execute("UPDATE coins SET market_type = 'spot' WHERE market_type IS NULL")
    
    # NOT NULL로 변경 (이미 NOT NULL이면 무시)
    if not column_exists:
        op.alter_column('coins', 'market_type', nullable=False)
    
    # 2. 기존 unique 인덱스 제거 (존재하는 경우만)
    result = conn.execute(sa.text(
        "SELECT COUNT(*) FROM information_schema.statistics "
        "WHERE table_schema = DATABASE() AND table_name = 'coins' AND index_name = 'uq_coin_symbol'"
    ))
    if result.scalar() > 0:
        op.drop_index('uq_coin_symbol', table_name='coins')
    
    # 3. 새 인덱스 추가 (존재하지 않는 경우만)
    result = conn.execute(sa.text(
        "SELECT COUNT(*) FROM information_schema.statistics "
        "WHERE table_schema = DATABASE() AND table_name = 'coins' AND index_name = 'idx_coin_market_type'"
    ))
    if result.scalar() == 0:
        op.create_index('idx_coin_market_type', 'coins', ['market_type'], unique=False)
    
    result = conn.execute(sa.text(
        "SELECT COUNT(*) FROM information_schema.statistics "
        "WHERE table_schema = DATABASE() AND table_name = 'coins' AND index_name = 'idx_coin_symbol_market'"
    ))
    if result.scalar() == 0:
        op.create_index('idx_coin_symbol_market', 'coins', ['symbol', 'market_type'], unique=True)


def downgrade() -> None:
    """Downgrade schema."""
    # 새 인덱스 제거
    op.drop_index('idx_coin_symbol_market', table_name='coins')
    op.drop_index('idx_coin_market_type', table_name='coins')
    
    # 기존 unique 인덱스 복원
    op.create_index('uq_coin_symbol', 'coins', ['symbol'], unique=True)
    
    # market_type 컬럼 제거
    op.drop_column('coins', 'market_type')
