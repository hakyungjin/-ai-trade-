"""
벡터 패턴 검색 서비스
- FAISS를 사용한 과거 유사 패턴 검색
- 1년치 데이터로 학습
"""

import numpy as np
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

# FAISS는 optional (메모리 절약)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

from app.models.vector_pattern import VectorPattern, VectorSimilarity
from app.services.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class VectorPatternService:
    """벡터 기반 패턴 검색"""

    def __init__(self, db: AsyncSession, symbol: str = "BTCUSDT"):
        self.db = db
        self.symbol = symbol
        self.vector_dim = 16  # 16차원 벡터: MACD(4) + BB(3) + EMA/SMA(4) + Stoch(2) + ATR/Vol(2) + 1
        self.index = None
        self.index_path = Path(f"data/faiss_index_{symbol}.idx")
        self.metadata_path = Path(f"data/faiss_metadata_{symbol}.pkl")
        
        # 초기화
        self._load_or_create_index()

    def _load_or_create_index(self):
        """기존 인덱스 로드 또는 새로 생성"""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available - vector pattern search disabled")
            return

        try:
            if self.index_path.exists() and self.metadata_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                logger.info(f"✅ Loaded FAISS index for {self.symbol}")
                return
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")

        # 새 인덱스 생성
        self.index = faiss.IndexFlatL2(self.vector_dim)
        logger.info(f"Created new FAISS index for {self.symbol}")

    async def build_index_from_history(self):
        """과거 1년 데이터로 FAISS 인덱스 구축"""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available - skipping index build")
            return 0

        try:
            logger.info(f"🔨 Building vector index for {self.symbol} from 1-year history...")
            
            # 1년 전 데이터 조회
            one_year_ago = datetime.now() - timedelta(days=365)
            result = await self.db.execute(
                select(VectorPattern).where(
                    VectorPattern.symbol == self.symbol,
                    VectorPattern.timestamp >= one_year_ago
                ).order_by(VectorPattern.timestamp)
            )
            patterns = result.scalars().all()
            
            if not patterns:
                logger.warning(f"No historical patterns found for {self.symbol}")
                return 0
            
            vectors = []
            metadata = []
            
            for idx, pattern in enumerate(patterns):
                try:
                    # 지표를 벡터로 변환
                    vector = self._indicators_to_vector(pattern.indicators)
                    if vector is not None:
                        vectors.append(vector)
                        metadata.append({
                            'pattern_id': pattern.id,
                            'timestamp': pattern.timestamp.isoformat(),
                            'signal': pattern.signal,
                            'confidence': pattern.confidence,
                            'return_1h': pattern.return_1h,
                            'return_4h': pattern.return_4h,
                            'return_24h': pattern.return_24h,
                        })
                        
                        # FAISS 인덱스 업데이트
                        pattern.vector_id = len(vectors) - 1
                        self.db.add(pattern)
                
                except Exception as e:
                    logger.warning(f"Error processing pattern {pattern.id}: {e}")
                    continue
            
            if vectors:
                # FAISS 인덱스에 벡터 추가
                vectors_array = np.array(vectors, dtype=np.float32)
                self.index.add(vectors_array)
                
                # 인덱스와 메타데이터 저장
                self.index_path.parent.mkdir(exist_ok=True)
                faiss.write_index(self.index, str(self.index_path))
                with open(self.metadata_path, 'wb') as f:
                    pickle.dump(metadata, f)
                
                await self.db.commit()
                
                logger.info(f"✅ Built index with {len(vectors)} vectors for {self.symbol}")
                return len(vectors)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            await self.db.rollback()
            return 0

    def _indicators_to_vector(self, indicators: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        기술적 지표를 16차원 벡터로 변환 (정규화됨)
        
        벡터 구성:
        [0-3]: MACD 관련 (MACD, Signal, Histogram, Momentum)
        [4-6]: 볼린저밴드 (Upper, Middle, Lower)
        [7-10]: 이동평균선 (EMA_12, EMA_26, SMA_20, SMA_50)
        [11-12]: 스토캐스틱 (K, D)
        [13-14]: ATR, Volume
        [15]: RSI (메인 지표)
        """
        if not indicators:
            return None
        
        try:
            close = indicators.get('close', 1)
            
            # MACD 관련
            macd = indicators.get('macd', 0) or 0
            macd_signal = indicators.get('macd_signal', 0) or 0
            macd_histogram = indicators.get('macd_histogram', 0) or 0
            macd_momentum = macd - macd_signal
            
            # 볼린저 밴드
            bb_upper = indicators.get('bb_upper', close) or close
            bb_middle = indicators.get('bb_middle', close) or close
            bb_lower = indicators.get('bb_lower', close) or close
            
            # 이동평균선
            ema_12 = indicators.get('ema_12', close) or close
            ema_26 = indicators.get('ema_26', close) or close
            sma_20 = indicators.get('sma_20', close) or close
            sma_50 = indicators.get('sma_50', close) or close
            
            # 스토캐스틱
            stoch_k = indicators.get('stoch_k', 50) or 50
            stoch_d = indicators.get('stoch_d', 50) or 50
            
            # ATR과 Volume
            atr = indicators.get('atr_14', 0) or 0
            volume = indicators.get('volume', 0) or 0
            
            # RSI
            rsi = indicators.get('rsi_14', 50) or 50
            
            # 특징 벡터 생성 및 정규화
            features = [
                # MACD 관련 (0-3)
                np.clip(macd / 0.1, -2, 2) / 2,  # MACD
                np.clip(macd_signal / 0.1, -2, 2) / 2,  # Signal
                np.clip(macd_histogram / 0.05, -2, 2) / 2,  # Histogram
                np.clip(macd_momentum / 0.05, -2, 2) / 2,  # Momentum
                
                # 볼린저밴드 (4-6) - 가격 위치
                (close - bb_lower) / (bb_upper - bb_lower + 1e-6),  # BB 내에서의 위치
                (bb_upper - close) / (bb_upper - bb_lower + 1e-6),  # 상단까지 거리 비율
                (close - bb_lower) / (bb_upper - bb_lower + 1e-6),  # 하단부터 거리 비율
                
                # 이동평균선 (7-10) - 비율로 정규화
                (close - ema_12) / (ema_12 + 1e-6),  # Close vs EMA12
                (ema_12 - ema_26) / (ema_26 + 1e-6),  # EMA 크로스
                (close - sma_20) / (sma_20 + 1e-6),  # Close vs SMA20
                (sma_20 - sma_50) / (sma_50 + 1e-6),  # MA 크로스
                
                # 스토캐스틱 (11-12)
                stoch_k / 100,  # 0-1 정규화
                stoch_d / 100,  # 0-1 정규화
                
                # ATR과 Volume (13-14)
                np.clip(atr / (close * 0.02), -1, 1),  # 변동성
                np.clip(volume / 1e8, 0, 2) / 2,  # 거래량
                
                # RSI (15) - 메인 신호
                rsi / 100,  # 0-1 정규화
            ]
            
            # NaN/Inf 체크 및 범위 제한
            features = [
                0 if (f != f or f is None or np.isinf(f)) else np.clip(float(f), -1, 1)
                for f in features
            ]
            
            if len(features) != 16:
                logger.warning(f"Feature count mismatch: {len(features)} != 16")
                return None
            
            return np.array(features, dtype=np.float32)
        
        except Exception as e:
            logger.warning(f"Error converting indicators to vector: {e}")
            return None

    async def find_similar_patterns(
        self,
        current_indicators: Dict[str, Any],
        k: int = 5,
        similarity_threshold: float = 0.75
    ) -> List[Dict[str, Any]]:
        """현재 지표와 유사한 과거 패턴 검색"""
        if not FAISS_AVAILABLE or self.index is None:
            return []

        try:
            if self.index.ntotal == 0:
                logger.warning("FAISS index is empty")
                return []
            
            # 현재 지표를 벡터로 변환
            query_vector = self._indicators_to_vector(current_indicators)
            if query_vector is None:
                return []
            
            # FAISS에서 검색
            query_array = np.array([query_vector], dtype=np.float32)
            distances, indices = self.index.search(query_array, min(k, self.index.ntotal))
            
            # 메타데이터 로드
            metadata = self._load_metadata()
            if not metadata:
                return []
            
            similar_patterns = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(metadata):
                    continue
                
                # 거리를 유사도로 변환 (낮을수록 유사)
                similarity = 1 / (1 + dist)
                
                if similarity >= similarity_threshold:
                    pattern_data = metadata[idx].copy()
                    pattern_data['similarity'] = float(similarity)
                    similar_patterns.append(pattern_data)
            
            logger.info(f"Found {len(similar_patterns)} similar patterns (threshold: {similarity_threshold})")
            return similar_patterns
        
        except Exception as e:
            logger.error(f"Error finding similar patterns: {e}")
            return []

    def _load_metadata(self) -> Optional[List[Dict]]:
        """저장된 메타데이터 로드"""
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading metadata: {e}")
        return None

    async def boost_signal_with_patterns(
        self,
        current_signal: str,
        current_confidence: float,
        current_indicators: Dict[str, Any],
        k: int = 5
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        과거 유사 패턴으로 신호 강화
        
        Returns:
            (boosted_signal, boosted_confidence, pattern_analysis)
        """
        try:
            # 유사 패턴 검색
            similar_patterns = await self.find_similar_patterns(
                current_indicators,
                k=k,
                similarity_threshold=0.75
            )
            
            if not similar_patterns:
                return current_signal, current_confidence, {'similar_count': 0}
            
            # 유사 패턴의 수익률 분석
            avg_return_1h = np.mean([p.get('return_1h', 0) or 0 for p in similar_patterns])
            avg_return_4h = np.mean([p.get('return_4h', 0) or 0 for p in similar_patterns])
            
            # 수익률이 양수면 신호 강화
            boost_amount = 0
            if avg_return_1h > 0:
                boost_amount += min(avg_return_1h / 100, 0.15)  # 최대 +15%
            if avg_return_4h > 0:
                boost_amount += min(avg_return_4h / 100, 0.15)  # 최대 +15%
            
            boosted_confidence = min(current_confidence + boost_amount, 0.95)
            
            # 신호 변경 여부 판단
            boosted_signal = current_signal
            if boosted_confidence > 0.7:
                if current_signal == "SELL" and avg_return_1h > 1:
                    boosted_signal = "HOLD"  # SELL → HOLD로 완화
                elif current_signal == "HOLD" and (avg_return_1h > 0.5 or avg_return_4h > 1):
                    boosted_signal = "BUY"  # HOLD → BUY로 강화
            
            logger.info(
                f"✅ Signal boosted: {current_signal}({current_confidence:.2f}) "
                f"→ {boosted_signal}({boosted_confidence:.2f})"
            )
            
            return boosted_signal, boosted_confidence, {
                'similar_count': len(similar_patterns),
                'avg_return_1h': float(avg_return_1h),
                'avg_return_4h': float(avg_return_4h),
                'boost_amount': float(boost_amount),
                'similar_patterns': similar_patterns[:3]  # 상위 3개만 반환
            }
        
        except Exception as e:
            logger.error(f"Error boosting signal: {e}")
            return current_signal, current_confidence, {'error': str(e)}
