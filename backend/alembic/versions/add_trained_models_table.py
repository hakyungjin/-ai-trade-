"""Add trained_models table for model metadata

Revision ID: add_trained_models
Revises: add_vector_patterns_weights
Create Date: 2026-01-30

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_trained_models'
down_revision = 'add_vector_patterns_weights'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'trained_models',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('symbol', sa.String(50), nullable=False, index=True),
        sa.Column('timeframe', sa.String(10), nullable=False, index=True),
        sa.Column('model_type', sa.String(50), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False, default=1),

        # 파일 경로
        sa.Column('model_path', sa.String(500), nullable=True),
        sa.Column('scaler_path', sa.String(500), nullable=True),
        sa.Column('features_path', sa.String(500), nullable=True),

        # 학습 설정
        sa.Column('num_classes', sa.Integer(), nullable=False, default=3),
        sa.Column('num_features', sa.Integer(), nullable=True),
        sa.Column('feature_names', sa.JSON(), nullable=True),
        sa.Column('threshold', sa.Float(), nullable=True, default=0.02),
        sa.Column('lookahead', sa.Integer(), nullable=True, default=5),
        sa.Column('sequence_length', sa.Integer(), nullable=True, default=20),

        # 학습 데이터 정보
        sa.Column('training_samples', sa.Integer(), nullable=True),
        sa.Column('test_samples', sa.Integer(), nullable=True),
        sa.Column('data_start_date', sa.String(50), nullable=True),
        sa.Column('data_end_date', sa.String(50), nullable=True),

        # 학습 결과 메트릭
        sa.Column('accuracy', sa.Float(), nullable=True),
        sa.Column('precision_score', sa.Float(), nullable=True),
        sa.Column('recall_score', sa.Float(), nullable=True),
        sa.Column('f1_score', sa.Float(), nullable=True),
        sa.Column('train_loss', sa.Float(), nullable=True),
        sa.Column('val_loss', sa.Float(), nullable=True),
        sa.Column('class_metrics', sa.JSON(), nullable=True),

        # 실거래 성과
        sa.Column('live_accuracy', sa.Float(), nullable=True),
        sa.Column('live_total_predictions', sa.Integer(), nullable=True, default=0),
        sa.Column('live_correct_predictions', sa.Integer(), nullable=True, default=0),
        sa.Column('live_total_pnl', sa.Float(), nullable=True, default=0.0),

        # 상태
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('is_best', sa.Boolean(), nullable=False, default=False),

        # 타임스탬프
        sa.Column('trained_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),

        sa.PrimaryKeyConstraint('id'),
    )

    # 인덱스
    op.create_index('ix_trained_models_symbol_timeframe', 'trained_models', ['symbol', 'timeframe'])
    op.create_index('ix_trained_models_active', 'trained_models', ['is_active'])


def downgrade() -> None:
    op.drop_index('ix_trained_models_active', table_name='trained_models')
    op.drop_index('ix_trained_models_symbol_timeframe', table_name='trained_models')
    op.drop_table('trained_models')
