"""
Asset Mapping Model
통합 모델을 위한 심볼 → Asset ID 매핑 테이블
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from app.database import Base


class AssetMapping(Base):
    """자산 매핑 테이블 (통합 모델용)"""
    __tablename__ = "asset_mappings"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(50), unique=True, nullable=False, index=True)
    asset_id = Column(Integer, unique=True, nullable=False, index=True)

    # 자산 분류
    market_type = Column(String(20), nullable=False, index=True)  # crypto, nasdaq, forex
    category = Column(String(50), nullable=True)  # major_coin, altcoin, stock, etf

    # 메타데이터
    full_name = Column(String(200), nullable=True)
    description = Column(String(500), nullable=True)

    # 상태
    is_active = Column(Boolean, default=True, nullable=False)

    # 타임스탬프
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self):
        return f"<AssetMapping(symbol={self.symbol}, asset_id={self.asset_id}, market_type={self.market_type})>"

    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'asset_id': self.asset_id,
            'market_type': self.market_type,
            'category': self.category,
            'full_name': self.full_name,
            'description': self.description,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
