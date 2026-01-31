"""
Asset Mapping Service
통합 모델을 위한 심볼 → Asset ID 매핑 관리
"""
from typing import Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.asset_mapping import AssetMapping
import logging

logger = logging.getLogger(__name__)


class AssetMappingService:
    """자산 ID 매핑 관리 서비스"""

    # 인메모리 캐시 (빠른 조회용)
    _cache: Dict[str, int] = {}
    _reverse_cache: Dict[int, str] = {}

    # ID 범위 정의
    ID_RANGES = {
        'major_coin': (0, 49),      # BTC, ETH, BNB 등
        'altcoin': (50, 99),         # 알트코인
        'stock': (100, 199),         # 나스닥 주식
        'etf': (200, 299),           # ETF
        'forex': (300, 399),         # 외환
        'commodity': (400, 499),     # 원자재
    }

    @classmethod
    async def get_asset_id(
        cls,
        db: AsyncSession,
        symbol: str,
        create_if_missing: bool = True
    ) -> Optional[int]:
        """심볼로 Asset ID 조회 (캐시 우선)"""
        # 캐시 확인
        if symbol in cls._cache:
            return cls._cache[symbol]

        # DB 조회
        stmt = select(AssetMapping).where(AssetMapping.symbol == symbol)
        result = await db.execute(stmt)
        mapping = result.scalar_one_or_none()

        if mapping:
            cls._cache[symbol] = mapping.asset_id
            cls._reverse_cache[mapping.asset_id] = symbol
            return mapping.asset_id

        # 자동 생성
        if create_if_missing:
            logger.info(f"Asset ID not found for {symbol}, creating new mapping...")
            new_mapping = await cls.create_asset_mapping(db, symbol)
            if new_mapping:
                return new_mapping.asset_id

        return None

    @classmethod
    async def get_symbol(cls, db: AsyncSession, asset_id: int) -> Optional[str]:
        """Asset ID로 심볼 조회"""
        # 캐시 확인
        if asset_id in cls._reverse_cache:
            return cls._reverse_cache[asset_id]

        # DB 조회
        stmt = select(AssetMapping).where(AssetMapping.asset_id == asset_id)
        result = await db.execute(stmt)
        mapping = result.scalar_one_or_none()

        if mapping:
            cls._reverse_cache[asset_id] = mapping.symbol
            cls._cache[mapping.symbol] = asset_id
            return mapping.symbol

        return None

    @classmethod
    async def create_asset_mapping(
        cls,
        db: AsyncSession,
        symbol: str,
        market_type: Optional[str] = None,
        category: Optional[str] = None,
        full_name: Optional[str] = None
    ) -> Optional[AssetMapping]:
        """새 자산 매핑 생성"""
        try:
            # market_type 자동 감지
            if market_type is None:
                market_type = cls._detect_market_type(symbol)

            # category 자동 감지
            if category is None:
                category = cls._detect_category(symbol, market_type)

            # 사용 가능한 asset_id 찾기
            asset_id = await cls._get_next_available_id(db, category)

            if asset_id is None:
                logger.error(f"No available asset_id for category {category}")
                return None

            # 매핑 생성
            mapping = AssetMapping(
                symbol=symbol,
                asset_id=asset_id,
                market_type=market_type,
                category=category,
                full_name=full_name,
                is_active=True
            )

            db.add(mapping)
            await db.commit()
            await db.refresh(mapping)

            # 캐시 업데이트
            cls._cache[symbol] = asset_id
            cls._reverse_cache[asset_id] = symbol

            logger.info(f"✅ Created asset mapping: {symbol} → {asset_id} ({market_type}/{category})")
            return mapping

        except Exception as e:
            logger.error(f"❌ Error creating asset mapping for {symbol}: {e}")
            await db.rollback()
            return None

    @classmethod
    async def _get_next_available_id(cls, db: AsyncSession, category: str) -> Optional[int]:
        """카테고리별 사용 가능한 다음 ID 찾기"""
        if category not in cls.ID_RANGES:
            logger.warning(f"Unknown category: {category}, defaulting to altcoin")
            category = 'altcoin'

        start_id, end_id = cls.ID_RANGES[category]

        # 해당 범위 내 기존 ID 조회
        stmt = select(AssetMapping.asset_id).where(
            AssetMapping.asset_id >= start_id,
            AssetMapping.asset_id <= end_id
        ).order_by(AssetMapping.asset_id)

        result = await db.execute(stmt)
        used_ids = {row[0] for row in result.fetchall()}

        # 빈 ID 찾기
        for asset_id in range(start_id, end_id + 1):
            if asset_id not in used_ids:
                return asset_id

        logger.error(f"❌ No available IDs in range {start_id}-{end_id} for category {category}")
        return None

    @classmethod
    def _detect_market_type(cls, symbol: str) -> str:
        """심볼로 market_type 자동 감지"""
        symbol_upper = symbol.upper()

        # 암호화폐 (USDT, BTC, ETH 기준)
        if any(quote in symbol_upper for quote in ['USDT', 'BUSD', 'USDC', 'BTC', 'ETH']):
            return 'crypto'

        # 주식 (일반적으로 짧은 심볼)
        if len(symbol) <= 5 and symbol.isalpha():
            return 'nasdaq'

        # 외환 (쌍 표기: EUR/USD, EURUSD 등)
        if '/' in symbol or (len(symbol) == 6 and symbol.isalpha()):
            return 'forex'

        # 기본값
        return 'crypto'

    @classmethod
    def _detect_category(cls, symbol: str, market_type: str) -> str:
        """심볼로 category 자동 감지"""
        symbol_upper = symbol.upper()

        if market_type == 'crypto':
            # 메이저 코인
            major_coins = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT']
            if symbol_upper in major_coins:
                return 'major_coin'
            return 'altcoin'

        elif market_type == 'nasdaq':
            # FAANG + 주요 기업
            major_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA']
            if symbol_upper in major_stocks:
                return 'stock'

            # ETF (보통 3-4글자)
            if symbol_upper.endswith('ETF') or len(symbol) == 3:
                return 'etf'

            return 'stock'

        elif market_type == 'forex':
            return 'forex'

        return 'altcoin'  # 기본값

    @classmethod
    async def get_all_mappings(cls, db: AsyncSession) -> List[AssetMapping]:
        """모든 자산 매핑 조회"""
        stmt = select(AssetMapping).where(AssetMapping.is_active == True).order_by(AssetMapping.asset_id)
        result = await db.execute(stmt)
        return result.scalars().all()

    @classmethod
    async def get_mappings_by_market_type(cls, db: AsyncSession, market_type: str) -> List[AssetMapping]:
        """마켓 타입별 매핑 조회"""
        stmt = select(AssetMapping).where(
            AssetMapping.market_type == market_type,
            AssetMapping.is_active == True
        ).order_by(AssetMapping.asset_id)
        result = await db.execute(stmt)
        return result.scalars().all()

    @classmethod
    async def update_mapping(
        cls,
        db: AsyncSession,
        symbol: str,
        **kwargs
    ) -> Optional[AssetMapping]:
        """자산 매핑 업데이트"""
        try:
            stmt = select(AssetMapping).where(AssetMapping.symbol == symbol)
            result = await db.execute(stmt)
            mapping = result.scalar_one_or_none()

            if not mapping:
                logger.warning(f"Mapping not found for {symbol}")
                return None

            # 업데이트
            for key, value in kwargs.items():
                if hasattr(mapping, key):
                    setattr(mapping, key, value)

            await db.commit()
            await db.refresh(mapping)

            # 캐시 무효화
            if symbol in cls._cache:
                del cls._cache[symbol]
            if mapping.asset_id in cls._reverse_cache:
                del cls._reverse_cache[mapping.asset_id]

            logger.info(f"✅ Updated asset mapping: {symbol}")
            return mapping

        except Exception as e:
            logger.error(f"❌ Error updating asset mapping for {symbol}: {e}")
            await db.rollback()
            return None

    @classmethod
    async def initialize_default_mappings(cls, db: AsyncSession) -> int:
        """기본 자산 매핑 초기화 (메이저 코인 + 주요 주식)"""
        default_mappings = [
            # 메이저 코인 (0-9)
            ('BTCUSDT', 0, 'crypto', 'major_coin', 'Bitcoin'),
            ('ETHUSDT', 1, 'crypto', 'major_coin', 'Ethereum'),
            ('BNBUSDT', 2, 'crypto', 'major_coin', 'Binance Coin'),
            ('XRPUSDT', 3, 'crypto', 'major_coin', 'Ripple'),
            ('ADAUSDT', 4, 'crypto', 'major_coin', 'Cardano'),
            ('SOLUSDT', 5, 'crypto', 'major_coin', 'Solana'),
            ('DOTUSDT', 6, 'crypto', 'major_coin', 'Polkadot'),
            ('MATICUSDT', 7, 'crypto', 'major_coin', 'Polygon'),
            ('AVAXUSDT', 8, 'crypto', 'major_coin', 'Avalanche'),
            ('LINKUSDT', 9, 'crypto', 'major_coin', 'Chainlink'),

            # 주요 주식 (100-109)
            ('AAPL', 100, 'nasdaq', 'stock', 'Apple Inc.'),
            ('MSFT', 101, 'nasdaq', 'stock', 'Microsoft Corporation'),
            ('GOOGL', 102, 'nasdaq', 'stock', 'Alphabet Inc.'),
            ('AMZN', 103, 'nasdaq', 'stock', 'Amazon.com Inc.'),
            ('META', 104, 'nasdaq', 'stock', 'Meta Platforms Inc.'),
            ('TSLA', 105, 'nasdaq', 'stock', 'Tesla Inc.'),
            ('NVDA', 106, 'nasdaq', 'stock', 'NVIDIA Corporation'),
            ('NFLX', 107, 'nasdaq', 'stock', 'Netflix Inc.'),
            ('PYPL', 108, 'nasdaq', 'stock', 'PayPal Holdings Inc.'),
            ('INTC', 109, 'nasdaq', 'stock', 'Intel Corporation'),
        ]

        created_count = 0
        for symbol, asset_id, market_type, category, full_name in default_mappings:
            # 기존 확인
            stmt = select(AssetMapping).where(AssetMapping.symbol == symbol)
            result = await db.execute(stmt)
            existing = result.scalar_one_or_none()

            if not existing:
                mapping = AssetMapping(
                    symbol=symbol,
                    asset_id=asset_id,
                    market_type=market_type,
                    category=category,
                    full_name=full_name,
                    is_active=True
                )
                db.add(mapping)
                created_count += 1

        if created_count > 0:
            await db.commit()
            logger.info(f"✅ Initialized {created_count} default asset mappings")

        # 캐시 초기화
        cls._cache.clear()
        cls._reverse_cache.clear()

        return created_count
