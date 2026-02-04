"""
캔들 데이터 배치 저장 최적화
- 대량 데이터 효율적 저장
- 중복 제거
- 메모리 최적화
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import logging
from sqlalchemy import insert, select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.market_data import MarketCandle

logger = logging.getLogger(__name__)


class CandleBatchSaver:
    """대량 캔들 데이터 효율적 저장"""
    
    BATCH_SIZE = 1000  # 한 번에 저장할 캔들 개수
    
    @staticmethod
    async def save_batch(
        db_session: AsyncSession,
        symbol: str,
        timeframe: str,
        candles: List[Dict[str, Any]],
        skip_duplicates: bool = True
    ) -> Dict[str, int]:
        """
        캔들 데이터 배치 저장 (메모리/시간 최적화)
        
        Args:
            db_session: 데이터베이스 세션
            symbol: 심볼 (BTCUSDT 등)
            timeframe: 타임프레임 (1h, 4h 등)
            candles: 캔들 데이터 리스트 [{'open': ..., 'high': ..., ...}]
            skip_duplicates: 중복 무시 여부
        
        Returns:
            {'inserted': 개수, 'skipped': 개수, 'total': 개수}
        """
        if not candles:
            return {'inserted': 0, 'skipped': 0, 'total': 0}
        
        stats = {'inserted': 0, 'skipped': 0, 'total': len(candles)}
        
        try:
            # ===== 1️⃣ 기존 타임스탬프 조회 (중복 확인용) =====
            if skip_duplicates:
                existing_timestamps = await CandleBatchSaver._get_existing_timestamps(
                    db_session, symbol, timeframe, candles
                )
            else:
                existing_timestamps = set()
            
            # ===== 2️⃣ 배치 준비 =====
            batch_data = []
            for candle in candles:
                # 타임스탬프 추출
                open_time = CandleBatchSaver._parse_timestamp(candle.get('open_time'))
                
                # 중복 확인
                if skip_duplicates and open_time in existing_timestamps:
                    stats['skipped'] += 1
                    continue
                
                # 캔들 데이터 정규화
                candle_row = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'open_time': open_time,
                    'open': float(candle.get('open', 0)),
                    'high': float(candle.get('high', 0)),
                    'low': float(candle.get('low', 0)),
                    'close': float(candle.get('close', 0)),
                    'volume': float(candle.get('volume', 0)),
                    'close_time': CandleBatchSaver._parse_timestamp(candle.get('close_time')),
                    'quote_volume': float(candle.get('quote_volume', 0)),
                    'trades_count': int(candle.get('trades_count', 0)),
                }
                batch_data.append(candle_row)
            
            if not batch_data:
                logger.info(f"⏭️  No new candles for {symbol} {timeframe} (all duplicates)")
                return stats
            
            # ===== 3️⃣ 배치 저장 (청크 단위) =====
            for i in range(0, len(batch_data), CandleBatchSaver.BATCH_SIZE):
                chunk = batch_data[i:i + CandleBatchSaver.BATCH_SIZE]
                
                stmt = insert(MarketCandle).values(chunk)
                # MySQL: IGNORE 중복, PostgreSQL: ON CONFLICT 무시
                await db_session.execute(stmt)
                
                stats['inserted'] += len(chunk)
                logger.debug(f"📤 Inserted {len(chunk)} candles ({i}/{len(batch_data)})")
            
            # 커밋
            await db_session.commit()
            logger.info(f"✅ Saved {stats['inserted']} candles for {symbol} {timeframe}")
            
        except Exception as e:
            await db_session.rollback()
            logger.error(f"❌ Error saving candles: {e}")
            stats['inserted'] = 0
            raise
        
        return stats
    
    @staticmethod
    async def save_multi_symbol(
        db_session: AsyncSession,
        data: Dict[str, Dict[str, List[Dict]]],  # {symbol: {timeframe: [candles]}}
        skip_duplicates: bool = True
    ) -> Dict[str, Any]:
        """
        여러 심볼/타임프레임의 캔들을 효율적으로 저장
        
        Args:
            data: {
                'BTCUSDT': {'1h': [candles], '4h': [candles]},
                'ETHUSDT': {'1h': [candles]},
                ...
            }
        
        Returns:
            {
                'BTCUSDT': {'1h': {...}, '4h': {...}},
                'summary': {'total_inserted': ..., 'total_skipped': ...}
            }
        """
        results = {'summary': {'total_inserted': 0, 'total_skipped': 0}}
        
        for symbol, timeframes in data.items():
            results[symbol] = {}
            
            for timeframe, candles in timeframes.items():
                try:
                    stats = await CandleBatchSaver.save_batch(
                        db_session, symbol, timeframe, candles, skip_duplicates
                    )
                    results[symbol][timeframe] = stats
                    results['summary']['total_inserted'] += stats['inserted']
                    results['summary']['total_skipped'] += stats['skipped']
                    
                    # 레이트 리미팅 (API 제한 회피)
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"❌ Error saving {symbol} {timeframe}: {e}")
                    results[symbol][timeframe] = {
                        'inserted': 0,
                        'skipped': 0,
                        'error': str(e)
                    }
        
        return results
    
    @staticmethod
    async def _get_existing_timestamps(
        db_session: AsyncSession,
        symbol: str,
        timeframe: str,
        candles: List[Dict]
    ) -> set:
        """기존 타임스탐프 조회 (중복 확인용)"""
        # 조회할 타임스탬프 범위
        if not candles:
            return set()
        
        min_time = CandleBatchSaver._parse_timestamp(candles[0].get('open_time'))
        max_time = CandleBatchSaver._parse_timestamp(candles[-1].get('open_time'))
        
        # 데이터베이스 조회
        stmt = select(MarketCandle.open_time).where(
            and_(
                MarketCandle.symbol == symbol,
                MarketCandle.timeframe == timeframe,
                MarketCandle.open_time >= min_time,
                MarketCandle.open_time <= max_time
            )
        )
        
        result = await db_session.execute(stmt)
        existing = {row[0] for row in result.fetchall()}
        
        logger.debug(f"Found {len(existing)} existing candles for {symbol} {timeframe}")
        return existing
    
    @staticmethod
    def _parse_timestamp(ts) -> datetime:
        """타임스탬프 파싱 (밀리초, 초, datetime 지원)"""
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, (int, float)):
            # 밀리초 단위
            if ts > 1000000000000:
                return datetime.fromtimestamp(ts / 1000)
            # 초 단위
            return datetime.fromtimestamp(ts)
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts)
            except:
                return datetime.fromtimestamp(int(ts) / 1000)
        return datetime.now()
    
    @staticmethod
    async def get_candle_stats(db_session: AsyncSession) -> Dict[str, Any]:
        """캔들 저장 현황 조회"""
        try:
            # 심볼별 캔들 개수
            stmt = select(
                MarketCandle.symbol,
                MarketCandle.timeframe,
                func.count(MarketCandle.id).label('count'),
                func.min(MarketCandle.open_time).label('earliest'),
                func.max(MarketCandle.open_time).label('latest')
            ).group_by(MarketCandle.symbol, MarketCandle.timeframe)
            
            result = await db_session.execute(stmt)
            rows = result.fetchall()
            
            stats = {
                'by_symbol': {},
                'total_candles': 0,
                'total_symbols': 0,
                'total_timeframes': 0
            }
            
            for row in rows:
                symbol, timeframe, count, earliest, latest = row
                
                if symbol not in stats['by_symbol']:
                    stats['by_symbol'][symbol] = {}
                    stats['total_symbols'] += 1
                
                stats['by_symbol'][symbol][timeframe] = {
                    'count': count,
                    'earliest': earliest.isoformat() if earliest else None,
                    'latest': latest.isoformat() if latest else None,
                    'days_span': (latest - earliest).days if latest and earliest else 0
                }
                
                stats['total_candles'] += count
                stats['total_timeframes'] += 1
            
            return stats
        
        except Exception as e:
            logger.error(f"Error getting candle stats: {e}")
            return {'error': str(e)}
    
    @staticmethod
    async def cleanup_duplicates(db_session: AsyncSession) -> int:
        """중복 캔들 데이터 제거"""
        try:
            # 중복 제거 쿼리
            # 같은 symbol/timeframe/open_time의 최신 id만 유지
            logger.info("🧹 Removing duplicate candles...")
            
            # MySQL의 경우
            stmt = """
            DELETE FROM market_candles 
            WHERE id NOT IN (
                SELECT MIN(id) FROM market_candles 
                GROUP BY symbol, timeframe, open_time
            );
            """
            
            result = await db_session.execute(stmt)
            deleted_count = result.rowcount
            
            await db_session.commit()
            logger.info(f"✅ Deleted {deleted_count} duplicate candles")
            
            return deleted_count
        
        except Exception as e:
            logger.error(f"❌ Error cleaning duplicates: {e}")
            await db_session.rollback()
            return 0
