"""
실시간 신호 생성 서비스
- 심볼별 실시간 신호 생성
- 가중치 기반 + AI 기반 종합 신호
- 신호 강도 계산
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from enum import Enum

from .binance_service import BinanceService
from .data_collector import DataCollector
from .weighted_strategy import WeightedStrategy, Signal as StrategySignal
from .ai_strategy import AIStrategy
from .technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    """신호 타입"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class SignalStrength(str, Enum):
    """신호 강도"""
    VERY_STRONG = "very_strong"  # 90-100%
    STRONG = "strong"            # 70-90%
    MODERATE = "moderate"        # 50-70%
    WEAK = "weak"                # 30-50%
    VERY_WEAK = "very_weak"      # 0-30%


class RealTimeSignalService:
    """실시간 신호 생성 서비스"""

    def __init__(
        self,
        binance_service: BinanceService,
        use_ai: bool = True
    ):
        """
        Args:
            binance_service: 바이낸스 서비스
            use_ai: AI 전략 사용 여부
        """
        self.binance = binance_service
        self.collector = DataCollector(binance_service)
        self.weighted_strategy = WeightedStrategy()
        self.ai_strategy = AIStrategy() if use_ai else None
        self.use_ai = use_ai

        # 활성 심볼 추적
        self.active_symbols: List[str] = []

        # 신호 캐시 (심볼별)
        self.signal_cache: Dict[str, Dict[str, Any]] = {}

        logger.info("RealTimeSignalService initialized")

    async def add_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        심볼 추가

        Args:
            symbol: 추가할 심볼 (예: BTCUSDT)

        Returns:
            상태 정보
        """
        if symbol in self.active_symbols:
            return {
                'success': True,
                'message': f'{symbol} already active'
            }

        # 심볼 유효성 검사
        try:
            await self.binance.get_current_price(symbol)
        except Exception as e:
            return {
                'success': False,
                'message': f'Invalid symbol: {symbol}',
                'error': str(e)
            }

        self.active_symbols.append(symbol)
        logger.info(f"Symbol added: {symbol}")

        return {
            'success': True,
            'message': f'{symbol} added successfully',
            'symbol': symbol
        }

    async def remove_symbol(self, symbol: str) -> Dict[str, Any]:
        """심볼 제거"""
        if symbol in self.active_symbols:
            self.active_symbols.remove(symbol)
            if symbol in self.signal_cache:
                del self.signal_cache[symbol]

            logger.info(f"Symbol removed: {symbol}")
            return {
                'success': True,
                'message': f'{symbol} removed'
            }

        return {
            'success': False,
            'message': f'{symbol} not found'
        }

    async def get_available_symbols(self) -> List[Dict[str, str]]:
        """
        거래 가능한 심볼 목록 조회

        Returns:
            심볼 목록
        """
        try:
            # 바이낸스에서 모든 심볼 조회
            exchange_info = await self.binance._run_sync(
                self.binance.client.get_exchange_info
            )

            symbols = []
            for symbol_info in exchange_info['symbols']:
                if symbol_info['status'] == 'TRADING' and symbol_info['quoteAsset'] == 'USDT':
                    symbols.append({
                        'symbol': symbol_info['symbol'],
                        'baseAsset': symbol_info['baseAsset'],
                        'quoteAsset': symbol_info['quoteAsset']
                    })

            logger.info(f"Retrieved {len(symbols)} available symbols")
            return symbols

        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            return []

    async def generate_signal(
        self,
        symbol: str,
        interval: str = "1h",
        lookback: int = 200
    ) -> Dict[str, Any]:
        """
        심볼에 대한 신호 생성

        Args:
            symbol: 심볼
            interval: 시간 간격
            lookback: 조회할 캔들 개수

        Returns:
            신호 정보
        """
        try:
            # 최신 데이터 가져오기
            klines = await self.binance.get_klines(
                symbol=symbol,
                interval=interval,
                limit=lookback
            )

            if not klines:
                return {
                    'symbol': symbol,
                    'signal': SignalType.NEUTRAL,
                    'strength': SignalStrength.VERY_WEAK,
                    'confidence': 0.0,
                    'error': 'No data available'
                }

            # DataFrame 생성
            import pandas as pd
            df = pd.DataFrame(klines)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # 가중치 전략 신호
            weighted_result = self.weighted_strategy.analyze(df)

            # AI 전략 신호 (사용 가능한 경우)
            ai_result = None
            if self.use_ai and self.ai_strategy:
                try:
                    ai_result = self.ai_strategy.generate_signal(
                        df,
                        combine_with_indicators=False
                    )
                except Exception as e:
                    logger.warning(f"AI signal generation failed: {e}")

            # 종합 신호 계산
            final_signal = self._combine_signals(weighted_result, ai_result)

            # 현재 가격
            current_price = df.iloc[-1]['close']

            # 기술적 지표 요약
            indicators = TechnicalIndicators.get_signal_summary(
                TechnicalIndicators.calculate_all_indicators(df)
            )

            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'price': float(current_price),
                'signal': final_signal['signal'],
                'strength': final_signal['strength'],
                'confidence': final_signal['confidence'],
                'score': final_signal['score'],
                'weighted_signal': weighted_result['signal'],
                'weighted_confidence': weighted_result['confidence'],
                'ai_signal': ai_result['signal'] if ai_result else None,
                'ai_confidence': ai_result['confidence'] if ai_result else None,
                'indicators': {
                    'rsi': indicators.get('rsi', {}),
                    'macd': indicators.get('macd', {}),
                    'bollinger': indicators.get('bollinger_bands', {}),
                    'ema_cross': indicators.get('ema_cross', {})
                },
                'recommendation': final_signal['recommendation']
            }

            # 캐시 업데이트
            self.signal_cache[symbol] = result

            logger.info(f"Signal generated for {symbol}: {final_signal['signal']} ({final_signal['confidence']:.2f})")

            return result

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return {
                'symbol': symbol,
                'signal': SignalType.NEUTRAL,
                'strength': SignalStrength.VERY_WEAK,
                'confidence': 0.0,
                'error': str(e)
            }

    def _combine_signals(
        self,
        weighted_result: Dict[str, Any],
        ai_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        가중치 전략과 AI 전략 신호 결합

        Args:
            weighted_result: 가중치 전략 결과
            ai_result: AI 전략 결과

        Returns:
            종합 신호
        """
        # 가중치 전략 점수
        weighted_score = weighted_result['combined_score']
        weighted_confidence = weighted_result['confidence']

        # AI가 없거나 실패한 경우
        if not ai_result or ai_result.get('confidence', 0) < 0.3:
            signal_type, strength = self._determine_signal(
                weighted_score,
                weighted_confidence
            )

            return {
                'signal': signal_type,
                'strength': strength,
                'confidence': weighted_confidence,
                'score': weighted_score,
                'recommendation': self._get_recommendation(
                    signal_type,
                    strength,
                    weighted_confidence
                )
            }

        # AI와 가중치 전략 결합 (가중치: 60% weighted, 40% AI)
        ai_confidence = ai_result.get('confidence', 0)
        ai_score = ai_result.get('combined_score', 0) if 'combined_score' in ai_result else 0

        # AI 신호를 점수로 변환
        if 'direction' in ai_result:
            direction = ai_result['direction']
            if direction == 'bullish':
                ai_score = ai_confidence
            elif direction == 'bearish':
                ai_score = -ai_confidence
            else:
                ai_score = 0

        # 가중 평균
        combined_score = weighted_score * 0.6 + ai_score * 0.4
        combined_confidence = weighted_confidence * 0.6 + ai_confidence * 0.4

        signal_type, strength = self._determine_signal(
            combined_score,
            combined_confidence
        )

        return {
            'signal': signal_type,
            'strength': strength,
            'confidence': combined_confidence,
            'score': combined_score,
            'recommendation': self._get_recommendation(
                signal_type,
                strength,
                combined_confidence
            )
        }

    def _determine_signal(
        self,
        score: float,
        confidence: float
    ) -> tuple[SignalType, SignalStrength]:
        """
        점수와 신뢰도로부터 신호 타입과 강도 결정

        Args:
            score: 종합 점수 (-1 ~ 1)
            confidence: 신뢰도 (0 ~ 1)

        Returns:
            (신호 타입, 신호 강도)
        """
        # 신호 타입 결정
        if score >= 0.6:
            signal = SignalType.STRONG_BUY
        elif score >= 0.3:
            signal = SignalType.BUY
        elif score <= -0.6:
            signal = SignalType.STRONG_SELL
        elif score <= -0.3:
            signal = SignalType.SELL
        else:
            signal = SignalType.NEUTRAL

        # 신호 강도 결정
        if confidence >= 0.9:
            strength = SignalStrength.VERY_STRONG
        elif confidence >= 0.7:
            strength = SignalStrength.STRONG
        elif confidence >= 0.5:
            strength = SignalStrength.MODERATE
        elif confidence >= 0.3:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.VERY_WEAK

        return signal, strength

    def _get_recommendation(
        self,
        signal: SignalType,
        strength: SignalStrength,
        confidence: float
    ) -> Dict[str, Any]:
        """
        매매 권장사항 생성

        Args:
            signal: 신호 타입
            strength: 신호 강도
            confidence: 신뢰도

        Returns:
            권장사항
        """
        recommendation = {
            'action': signal.value,
            'strength': strength.value,
            'confidence_level': confidence
        }

        # 신호별 메시지
        if signal == SignalType.STRONG_BUY:
            recommendation['message'] = '🚀 강한 매수 신호! 진입을 고려하세요.'
            recommendation['action_text'] = '매수'
        elif signal == SignalType.BUY:
            recommendation['message'] = '📈 매수 신호. 신중한 진입을 권장합니다.'
            recommendation['action_text'] = '매수'
        elif signal == SignalType.NEUTRAL:
            recommendation['message'] = '⏸️ 횡보 구간. 관망을 권장합니다.'
            recommendation['action_text'] = '관망'
        elif signal == SignalType.SELL:
            recommendation['message'] = '📉 매도 신호. 포지션 정리를 고려하세요.'
            recommendation['action_text'] = '매도'
        elif signal == SignalType.STRONG_SELL:
            recommendation['message'] = '🔴 강한 매도 신호! 즉시 청산을 권장합니다.'
            recommendation['action_text'] = '매도'

        # 강도별 추가 메시지
        if strength == SignalStrength.VERY_STRONG:
            recommendation['strength_message'] = '매우 강한 신호입니다.'
        elif strength == SignalStrength.STRONG:
            recommendation['strength_message'] = '강한 신호입니다.'
        elif strength == SignalStrength.MODERATE:
            recommendation['strength_message'] = '중간 강도의 신호입니다.'
        elif strength == SignalStrength.WEAK:
            recommendation['strength_message'] = '약한 신호입니다. 추가 확인이 필요합니다.'
        else:
            recommendation['strength_message'] = '매우 약한 신호입니다. 신중하게 판단하세요.'

        return recommendation

    async def get_signal(self, symbol: str) -> Dict[str, Any]:
        """
        캐시된 신호 조회

        Args:
            symbol: 심볼

        Returns:
            신호 정보
        """
        if symbol in self.signal_cache:
            return self.signal_cache[symbol]

        # 캐시에 없으면 새로 생성
        return await self.generate_signal(symbol)

    async def update_all_signals(self):
        """활성 심볼의 모든 신호 업데이트"""
        if not self.active_symbols:
            return

        logger.info(f"Updating signals for {len(self.active_symbols)} symbols")

        tasks = [
            self.generate_signal(symbol)
            for symbol in self.active_symbols
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

    def get_active_symbols(self) -> List[str]:
        """활성 심볼 목록 조회"""
        return self.active_symbols.copy()

    def get_all_cached_signals(self) -> Dict[str, Dict[str, Any]]:
        """모든 캐시된 신호 조회"""
        return self.signal_cache.copy()
