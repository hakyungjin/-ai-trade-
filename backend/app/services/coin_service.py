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
        market_type: str = 'spot',
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
            market_type: 시장 유형 ('spot' 또는 'futures')
            **kwargs: full_name, description 등
        
        Returns:
            생성된 Coin 객체
        """
        try:
            # 기존 코인 확인 (심볼 + 마켓 타입으로 확인)
            stmt = select(Coin).where(
                and_(
                    Coin.symbol == symbol,
                    Coin.market_type == market_type
                )
            )
            result = await db_session.execute(stmt)
            existing = result.scalar_one_or_none()
            
            if existing:
                logger.info(f"✅ Coin {symbol} ({market_type}) already exists")
                return existing
            
            # 새 코인 생성
            coin = Coin(
                symbol=symbol,
                base_asset=base_asset,
                quote_asset=quote_asset,
                is_monitoring=is_monitoring,
                market_type=market_type,
                **kwargs
            )
            db_session.add(coin)
            await db_session.flush()  # ID를 얻기 위해 flush
            
            # 통계 및 설정 생성
            stats = CoinStatistics(coin_id=coin.id)
            config = CoinAnalysisConfig(coin_id=coin.id)
            db_session.add(stats)
            db_session.add(config)
            
            # flush만 하고 commit은 get_db()에서 자동으로 처리
            # 이렇게 하면 FastAPI의 표준 패턴을 따름
            await db_session.flush()  # 모든 변경사항을 DB에 반영 (아직 commit은 안됨)
            
            logger.info(f"✅ Added coin {symbol} ({market_type}) (ID: {coin.id}) - ready for commit by get_db()")
            return coin
            
        except Exception as e:
            # rollback은 get_db()에서 자동으로 처리됨
            logger.error(f"❌ Error adding coin {symbol}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    @staticmethod
    async def add_monitoring_coin(
        db_session: AsyncSession,
        symbol: str,
        timeframes: List[str] = None,
        market_type: str = 'spot'
    ) -> Coin:
        """
        모니터링할 코인 추가 및 자동 데이터 수집 시작
        
        Args:
            db_session: DB 세션
            symbol: 심볼 (BTCUSDT)
            timeframes: 모니터링할 타임프레임 목록
            market_type: 시장 유형 ('spot' 또는 'futures')
        """
        import asyncio
        from app.services.incremental_collector import IncrementalDataCollector
        
        if timeframes is None:
            timeframes = ["1h"]
        
        # 바이낸스에서 코인 정보 조회
        from app.services.binance_service import BinanceService
        from app.services.binance_futures_service import BinanceFuturesService, get_futures_service
        from app.config import get_settings
        
        config = get_settings()
        coin_info = None
        
        if market_type == 'futures':
            # 선물 시장에서 심볼 검색
            futures_service = get_futures_service()
            info = await futures_service.get_futures_exchange_info()
            for s in info.get('symbols', []):
                if s['symbol'] == symbol:
                    coin_info = s
                    break
        else:
            # 현물 시장에서 심볼 검색
            binance = BinanceService(config.binance_api_key, config.binance_secret_key)
            info = await binance.get_exchange_info()
            for s in info['symbols']:
                if s['symbol'] == symbol:
                    coin_info = s
                    break
        
        if not coin_info:
            raise ValueError(f"Symbol {symbol} not found in {market_type} market")
        
        # 코인 추가
        coin = await CoinService.add_coin(
            db_session,
            symbol=symbol,
            base_asset=coin_info['baseAsset'],
            quote_asset=coin_info['quoteAsset'],
            is_monitoring=True,
            market_type=market_type,
            monitoring_timeframes=timeframes
        )
        
        # 백그라운드에서 데이터 수집 시작 (현물/선물 모두 지원)
        async def start_data_collection():
            """백그라운드에서 데이터 수집 시작"""
            try:
                # 새로운 DB 세션 생성 (백그라운드 작업용)
                from app.database import AsyncSessionLocal
                async with AsyncSessionLocal() as bg_db:
                    
                    logger.info(f"🚀 Starting data collection for {symbol} ({market_type}) with timeframes: {timeframes}")
                    
                    if market_type == 'futures':
                        # 선물 데이터 수집
                        from app.services.binance_futures_service import BinanceFuturesService
                        futures_service = BinanceFuturesService(config.binance_api_key, config.binance_secret_key)
                        
                        for timeframe in timeframes:
                            try:
                                klines = await futures_service.get_futures_klines(
                                    symbol=symbol,
                                    interval=timeframe,
                                    limit=500
                                )
                                
                                if klines:
                                    from app.services.market_data_service import MarketDataService
                                    market_service = MarketDataService(bg_db)
                                    saved_count = await market_service.save_candles(
                                        symbol=symbol,
                                        timeframe=timeframe,
                                        candles=klines
                                    )
                                    
                                    logger.info(f"✅ [Futures] Collected {saved_count} candles for {symbol} ({timeframe})")
                                    
                                    # 코인 캔들 개수 업데이트
                                    await CoinService.update_coin_candle_count(
                                        bg_db,
                                        coin.id,
                                        (coin.candle_count or 0) + saved_count
                                    )
                                else:
                                    logger.warning(f"⚠️ No futures data for {symbol} ({timeframe})")
                            except Exception as e:
                                logger.error(f"❌ Error collecting futures data for {symbol} ({timeframe}): {e}")
                    else:
                        # 현물 데이터 수집
                        binance = BinanceService(config.binance_api_key, config.binance_secret_key)
                        collector = IncrementalDataCollector(bg_db, binance)
                        
                        for timeframe in timeframes:
                            try:
                                success, saved_count = await collector.collect_incremental_data(
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    force_full=False  # 증분 수집
                                )
                                if success:
                                    logger.info(f"✅ [Spot] Collected {saved_count} candles for {symbol} ({timeframe})")
                                    
                                    # 코인 캔들 개수 업데이트
                                    await CoinService.update_coin_candle_count(
                                        bg_db,
                                        coin.id,
                                        (coin.candle_count or 0) + saved_count
                                    )
                                else:
                                    logger.warning(f"⚠️ Failed to collect data for {symbol} ({timeframe})")
                            except Exception as e:
                                logger.error(f"❌ Error collecting data for {symbol} ({timeframe}): {e}")
                    
                    logger.info(f"✅ Data collection completed for {symbol}")
            except Exception as e:
                logger.error(f"❌ Error in background data collection for {symbol}: {e}")
                import traceback
                traceback.print_exc()
        
        # 백그라운드 태스크로 실행 (응답을 빠르게 반환)
        asyncio.create_task(start_data_collection())
        
        logger.info(f"✅ Coin {symbol} ({market_type}) added, data collection started in background")
        
        return coin
    
    @staticmethod
    async def get_monitoring_coins(
        db_session: AsyncSession,
        market_type: Optional[str] = None
    ) -> List[Coin]:
        """
        모니터링 중인 모든 코인 조회
        
        Args:
            db_session: DB 세션
            market_type: 시장 유형 필터 ('spot', 'futures' 또는 None=전체)
        """
        conditions = [
            Coin.is_active == True,
            Coin.is_monitoring == True
        ]
        
        if market_type:
            conditions.append(Coin.market_type == market_type)
        
        stmt = select(Coin).where(and_(*conditions)).order_by(Coin.priority.desc())
        
        result = await db_session.execute(stmt)
        return result.scalars().all()
    
    @staticmethod
    async def get_coin_by_symbol(
        db_session: AsyncSession,
        symbol: str,
        market_type: Optional[str] = None
    ) -> Optional[Coin]:
        """
        심볼로 코인 조회
        
        Args:
            db_session: DB 세션
            symbol: 심볼 (BTCUSDT)
            market_type: 시장 유형 ('spot' 또는 'futures'), None이면 심볼만으로 검색
        """
        if market_type:
            stmt = select(Coin).where(
                and_(
                    Coin.symbol == symbol,
                    Coin.market_type == market_type
                )
            )
        else:
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
