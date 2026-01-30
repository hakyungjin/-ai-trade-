"""
학습된 모델 메타데이터 DB 모델

파일 시스템에만 의존하지 않고, 모델 정보를 DB에서 관리:
- 모델 학습 이력 추적
- 피처 목록, 학습 메트릭 저장
- 심볼별/타임프레임별 최적 모델 조회
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON
from sqlalchemy.sql import func
from app.database import Base


class TrainedModel(Base):
    """학습된 AI 모델 메타데이터"""
    __tablename__ = 'trained_models'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 모델 식별
    symbol = Column(String(50), nullable=False, index=True)          # BTCUSDT
    timeframe = Column(String(10), nullable=False, index=True)       # 5m, 1h, 4h
    model_type = Column(String(50), nullable=False)                   # xgboost, lstm, ensemble
    version = Column(Integer, nullable=False, default=1)

    # 파일 경로 (파일 기반 모델 호환)
    model_path = Column(String(500), nullable=True)                   # ai-model/models/xgboost_btcusdt_5m_v3.joblib
    scaler_path = Column(String(500), nullable=True)
    features_path = Column(String(500), nullable=True)

    # 학습 설정
    num_classes = Column(Integer, nullable=False, default=3)          # 2, 3, 5
    num_features = Column(Integer, nullable=True)                     # 피처 수
    feature_names = Column(JSON, nullable=True)                       # ["rsi_normalized", "macd_normalized", ...]
    threshold = Column(Float, nullable=True, default=0.02)            # 레이블 임계값
    lookahead = Column(Integer, nullable=True, default=5)             # 미래 캔들 수
    sequence_length = Column(Integer, nullable=True, default=20)      # LSTM 시퀀스 길이

    # 학습 데이터 정보
    training_samples = Column(Integer, nullable=True)                 # 학습 샘플 수
    test_samples = Column(Integer, nullable=True)                     # 테스트 샘플 수
    data_start_date = Column(String(50), nullable=True)               # 학습 데이터 시작일
    data_end_date = Column(String(50), nullable=True)                 # 학습 데이터 종료일

    # 학습 결과 메트릭
    accuracy = Column(Float, nullable=True)                           # 테스트 정확도
    precision_score = Column(Float, nullable=True)
    recall_score = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)

    # 클래스별 정확도 (JSON)
    class_metrics = Column(JSON, nullable=True)                       # {"BUY": {"precision": 0.65, ...}, ...}

    # 실거래 성과 (업데이트)
    live_accuracy = Column(Float, nullable=True)                      # 실거래 기반 정확도
    live_total_predictions = Column(Integer, nullable=True, default=0)
    live_correct_predictions = Column(Integer, nullable=True, default=0)
    live_total_pnl = Column(Float, nullable=True, default=0.0)

    # 상태
    is_active = Column(Boolean, nullable=False, default=True)         # 현재 사용 중인 모델
    is_best = Column(Boolean, nullable=False, default=False)          # 해당 심볼/타임프레임 최고 성능 모델

    # 타임스탬프
    trained_at = Column(DateTime, nullable=True, server_default=func.now())
    last_used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<TrainedModel {self.model_type}_{self.symbol}_{self.timeframe}_v{self.version}>"

    @property
    def model_name(self) -> str:
        return f"{self.model_type}_{self.symbol.lower()}_{self.timeframe}_v{self.version}"

    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'model_type': self.model_type,
            'version': self.version,
            'model_path': self.model_path,
            'num_classes': self.num_classes,
            'num_features': self.num_features,
            'feature_names': self.feature_names,
            'accuracy': self.accuracy,
            'f1_score': self.f1_score,
            'training_samples': self.training_samples,
            'live_accuracy': self.live_accuracy,
            'live_total_predictions': self.live_total_predictions,
            'is_active': self.is_active,
            'is_best': self.is_best,
            'trained_at': self.trained_at.isoformat() if self.trained_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
