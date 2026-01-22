"""
코인 메타데이터 관리 서비스
- 모니터링 코인 추가/제거
- 코인 통계 업데이트
- 코인 설정 관리
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.coin import Coin, CoinStatistics, CoinAnalysisConfig, CoinPriceHistory

logger = logging.getLogger(__name__)


class CoinService:
    """코인 정보 관리"""
    
    @staticmethod
    async def add_coin(
        db_session: AsyncSession,
        symbol: str,
        base_asset: str,
        quote_asset: str,
        is_monitoring: bool = False,
        **kwargs
    ) -> Coin:
        """
        새로운 코인 추가
        
        Args:
            db_session: DB 세션
            symbol: 심볼 (BTCUSDT)
            base_asset: 기초 자산 (BTC)
            quote_asset: 인용 자산 (USDT)
            is_monitoring: 모니터링 여부
            **kwargs: full_name, description 등
        
        Returns:
            생성된 Coin 객체
        """
        try:
            # 기존 코인 확인
            stmt = select(Coin).where(Coin.symbol == symbol)
            result = await db_session.execute(stmt)
            existing = result.scalar_one_or_none()
            
            if existing:
                logger.info(f"✅ Coin {symbol} already exists")
                return existing
            
            # 새 코인 생성
            coin = Coin(
                symbol=symbol,
                base_asset=base_asset,
                quote_asset=quote_asset,
                is_monitoring=is_monitoring,
                **kwargs
            )
            db_session.add(coin)
            await db_session.flush()
            
            # 통계 및 설정 생성
            stats = CoinStatistics(coin_id=coin.id)
            config = CoinAnalysisConfig(coin_id=coin.id)
            db_session.add(stats)
            db_session.add(config)
            
            await db_session.commit()
            logger.info(f"✅ Added coin {symbol}")
            return coin
            
        except Exception as e:
            await db_session.rollback()
            logger.error(f"❌ Error adding coin: {e}")
            raise
    
    @staticmethod
    async def add_monitoring_coin(
        db_session: AsyncSession,
        symbol: str,
        timeframes: List[str] = None
    ) -> Coin:
        """모니터링할 코인 추가"""
        if timeframes is None:
            timeframes = ["1h"]
        
        # 바이낸스에서 코인 정보 조회
        from app.services.binance_service import BinanceService
        from app.config import get_settings
        
        config = get_settings()
        binance = BinanceService(config.binance_api_key, config.binance_secret_key)
        
        # 심볼 검색
        info = await binance.get_exchange_info()
        coin_info = None
        for s in info['symbols']:
            if s['symbol'] == symbol:
                coin_info = s
                break
        
        if not coin_info:
            raise ValueError(f"Symbol {symbol} not found")
        
        # 코인 추가
        coin = await CoinService.add_coin(
            db_session,
            symbol=symbol,
            base_asset=coin_info['baseAsset'],
            quote_asset=coin_info['quoteAsset'],
            is_monitoring=True,
            monitoring_timeframes=timeframes
        )
        
        return coin
    
    @staticmethod
    async def get_monitoring_coins(db_session: AsyncSession) -> List[Coin]:
        """모니터링 중인 모든 코인 조회"""
        stmt = select(Coin).where(
            and_(
                Coin.is_active == True,
                Coin.is_monitoring == True
            )
        ).order_by(Coin.priority.desc())
        
        result = await db_session.execute(stmt)
        return result.scalars().all()
    
    @staticmethod
    async def get_coin_by_symbol(db_session: AsyncSession, symbol: str) -> Optional[Coin]:
        """심볼로 코인 조회"""
        stmt = select(Coin).where(Coin.symbol == symbol)
        result = await db_session.execute(stmt)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def update_coin_price(
        db_session: AsyncSession,
        coin_id: int,
        price: float,
        price_change_24h: float = None,
        volume_24h: float = None,
        market_cap: float = None
    ) -> Coin:
        """코인 가격 정보 업데이트"""
        try:
            stmt = select(Coin).where(Coin.id == coin_id)
            result = await db_session.execute(stmt)
            coin = result.scalar_one()
            
            coin.current_price = price
            if price_change_24h is not None:
                coin.price_change_24h = price_change_24h
            if volume_24h is not None:
                coin.volume_24h = volume_24h
            if market_cap is not None:
                coin.market_cap = market_cap
            coin.last_price_update = datetime.now()
            
            # 가격 이력 저장
            price_history = CoinPriceHistory(
                coin_id=coin_id,
                price=price,
                price_change_24h=price_change_24h,
                volume_24h=volume_24h,
                market_cap=market_cap,
                recorded_at=datetime.now()
            )
            db_session.add(price_history)
            
            await db_session.commit()
            logger.info(f"✅ Updated price for coin {coin.symbol}: ${price}")
            return coin
            
        except Exception as e:
            await db_session.rollback()
            logger.error(f"❌ Error updating coin price: {e}")
            raise
    
    @staticmethod
    async def update_coin_candle_count(
        db_session: AsyncSession,
        coin_id: int,
        new_count: int,
        earliest: datetime = None,
        latest: datetime = None
    ) -> Coin:
        """코인 캔들 개수 및 시간 범위 업데이트"""
        try:
            stmt = select(Coin).where(Coin.id == coin_id)
            result = await db_session.execute(stmt)
            coin = result.scalar_one()
            
            coin.candle_count = new_count
            if earliest:
                coin.earliest_candle_time = earliest
            if latest:
                coin.latest_candle_time = latest
            
            await db_session.commit()
            logger.info(f"✅ Updated candle count for {coin.symbol}: {new_count}")
            return coin
            
        except Exception as e:
            await db_session.rollback()
            logger.error(f"❌ Error updating candle count: {e}")
            raise
    
    @staticmethod
    async def get_coin_stats(db_session: AsyncSession, coin_id: int) -> CoinStatistics:
        """코인 통계 조회"""
        stmt = select(CoinStatistics).where(CoinStatistics.coin_id == coin_id)
        result = await db_session.execute(stmt)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def update_coin_stats(
        db_session: AsyncSession,
        coin_id: int,
        **kwargs
    ) -> CoinStatistics:
        """코인 통계 업데이트"""
        try:
            stmt = select(CoinStatistics).where(CoinStatistics.coin_id == coin_id)
            result = await db_session.execute(stmt)
            stats = result.scalar_one()
            
            for key, value in kwargs.items():
                if hasattr(stats, key):
                    setattr(stats, key, value)
            
            stats.updated_at = datetime.now()
            await db_session.commit()
            return stats
            
        except Exception as e:
            await db_session.rollback()
            logger.error(f"❌ Error updating coin stats: {e}")
            raise
    
    @staticmethod
    async def get_coin_config(db_session: AsyncSession, coin_id: int) -> CoinAnalysisConfig:
        """코인 분석 설정 조회"""
        stmt = select(CoinAnalysisConfig).where(CoinAnalysisConfig.coin_id == coin_id)
        result = await db_session.execute(stmt)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def update_coin_config(
        db_session: AsyncSession,
        coin_id: int,
        **kwargs
    ) -> CoinAnalysisConfig:
        """코인 분석 설정 업데이트"""
        try:
            stmt = select(CoinAnalysisConfig).where(CoinAnalysisConfig.coin_id == coin_id)
            result = await db_session.execute(stmt)
            config = result.scalar_one()
            
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            config.updated_at = datetime.now()
            await db_session.commit()
            return config
            
        except Exception as e:
            await db_session.rollback()
            logger.error(f"❌ Error updating coin config: {e}")
            raise
    
    @staticmethod
    async def get_all_coins_summary(db_session: AsyncSession) -> List[Dict[str, Any]]:
        """모든 코인의 요약 정보 조회"""
        try:
            stmt = select(Coin).order_by(Coin.priority.desc(), Coin.created_at.asc())
            result = await db_session.execute(stmt)
            coins = result.scalars().all()
            
            summary = []
            for coin in coins:
                stats = await CoinService.get_coin_stats(db_session, coin.id)
                summary.append({
                    'id': coin.id,
                    'symbol': coin.symbol,
                    'base_asset': coin.base_asset,
                    'is_monitoring': coin.is_monitoring,
                    'current_price': coin.current_price,
                    'price_change_24h': coin.price_change_24h,
                    'candle_count': coin.candle_count,
                    'earliest_candle': coin.earliest_candle_time,
                    'latest_candle': coin.latest_candle_time,
                    'total_signals': stats.total_signals if stats else 0,
                    'pattern_vectors': stats.pattern_vectors_count if stats else 0,
                    'last_analysis': coin.last_analysis_at,
                })
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error getting coins summary: {e}")
            return []
    
    @staticmethod
    async def remove_monitoring_coin(db_session: AsyncSession, coin_id: int) -> bool:
        """모니터링 코인 제거 (비활성화)"""
        try:
            stmt = select(Coin).where(Coin.id == coin_id)
            result = await db_session.execute(stmt)
            coin = result.scalar_one()
            
            coin.is_monitoring = False
            coin.is_active = False
            
            await db_session.commit()
            logger.info(f"✅ Removed monitoring for coin {coin.symbol}")
            return True
            
        except Exception as e:
            await db_session.rollback()
            logger.error(f"❌ Error removing coin: {e}")
            return False
