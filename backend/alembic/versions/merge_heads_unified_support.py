"""Merge heads: unified model support + market type

Revision ID: merge_heads_unified
Revises: add_unified_model_support, 3923aa842786
Create Date: 2026-01-31

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'merge_heads_unified'
down_revision = ('add_unified_model_support', '3923aa842786')
branch_labels = None
depends_on = None


def upgrade() -> None:
    # This is a merge migration - no changes needed
    pass


def downgrade() -> None:
    # This is a merge migration - no changes needed
    pass
