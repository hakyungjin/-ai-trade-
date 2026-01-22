"""
AI 학습용 데이터 수집 및 저장 서비스
- 실시간 차트 데이터를 학습 데이터로 변환
- 피처 엔지니어링 및 레이블 생성
- DB에 저장
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
import json
import numpy as np

from app.models.market_data import (
    AITrainingData,
    MarketCandle,
    TechnicalIndicator,
    AIAnalysis,
    SignalHistory
)
from app.services.technical_indicators import TechnicalIndicators


class TrainingDataService:
    """AI 학습용 데이터 관리 서비스"""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.ti = TechnicalIndicators()

    async def save_training_data(
        self,
        symbol: str,
        timeframe: str,
        candles: List[Dict[str, Any]],
        analysis: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        차트 데이터를 학습 데이터로 변환 및 저장

        Args:
            symbol: 거래쌍
            timeframe: 시간프레임
            candles: OHLCV 캔들 데이터
            analysis: AI 분석 결과 (선택사항)

        Returns:
            저장 성공 여부
        """
        if len(candles) < 50:
            return False

        try:
            timestamp = datetime.fromisoformat(candles[-1]["open_time"])
            current_price = candles[-1]["close"]

            # 피처 추출
            features = self._extract_features(candles)

            # 미래 레이블 계산
            labels = await self._calculate_future_labels(
                symbol, timeframe, timestamp
            )

            # 학습 데이터 생성
            training_data = AITrainingData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                features=features,
                current_price=current_price,
                future_price_1h=labels.get("price_1h"),
                future_price_4h=labels.get("price_4h"),
                future_price_24h=labels.get("price_24h"),
                label_1h=labels.get("label_1h"),
                label_4h=labels.get("label_4h"),
                label_24h=labels.get("label_24h"),
            )

            self.db.add(training_data)

            # AI 분석 저장
            if analysis:
                await self._save_analysis(symbol, timeframe, timestamp, analysis)

            await self.db.commit()
            return True

        except Exception as e:
            print(f"Error saving training data: {e}")
            await self.db.rollback()
            return False

    def _extract_features(self, candles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """차트 데이터에서 피처 추출"""
        if len(candles) < 50:
            return {}

        recent = candles[-50:]
        closes = [c["close"] for c in recent]
        highs = [c["high"] for c in recent]
        lows = [c["low"] for c in recent]
        volumes = [c["volume"] for c in recent]

        # 기본 기술적 지표
        features = {
            "price": closes[-1],
            "sma_5": sum(closes[-5:]) / 5 if len(closes) >= 5 else closes[-1],
            "sma_10": sum(closes[-10:]) / 10 if len(closes) >= 10 else closes[-1],
            "sma_20": sum(closes[-20:]) / 20 if len(closes) >= 20 else closes[-1],
            "sma_50": sum(closes) / len(closes),
            "high_50": max(highs),
            "low_50": min(lows),
            "volume_avg": sum(volumes) / len(volumes),
            "volatility": self._calculate_volatility(closes),
            "rsi": self._calculate_rsi(closes),
            "macd": self._calculate_macd(closes),
        }

        # Price change
        features["price_change_5"] = (closes[-1] - closes[-5]) / closes[-5] * 100
        features["price_change_20"] = (closes[-1] - closes[-20]) / closes[-20] * 100
        features["price_change_50"] = (closes[-1] - closes[-50]) / closes[-50] * 100

        # Volume change
        features["volume_change"] = (volumes[-1] - np.mean(volumes[:-1])) / np.mean(volumes[:-1]) * 100

        return features

    def _calculate_volatility(self, prices: List[float]) -> float:
        """변동성 계산 (표준편차)"""
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        return float(np.std(returns)) * 100

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """RSI 계산"""
        if len(prices) < period + 1:
            return 50.0

        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        seed = deltas[:period]
        up = sum(max(d, 0) for d in seed) / period
        down = sum(max(-d, 0) for d in seed) / period

        rs = up / down if down != 0 else 0
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def _calculate_macd(self, prices: List[float], fast=12, slow=26) -> float:
        """MACD 계산"""
        if len(prices) < slow:
            return 0.0

        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)

        return float(ema_fast - ema_slow)

    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """EMA 계산"""
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period

        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

    async def _calculate_future_labels(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime
    ) -> Dict[str, Any]:
        """미래 가격 데이터에서 레이블 계산"""
        labels = {
            "price_1h": None,
            "price_4h": None,
            "price_24h": None,
            "label_1h": None,
            "label_4h": None,
            "label_24h": None,
        }

        try:
            # 현재 가격 조회
            current_candle = await self.db.execute(
                select(MarketCandle)
                .where(
                    MarketCandle.symbol == symbol,
                    MarketCandle.timeframe == timeframe,
                    MarketCandle.open_time == timestamp
                )
                .order_by(desc(MarketCandle.id))
                .limit(1)
            )
            current = current_candle.scalar_one_or_none()

            if not current:
                return labels

            current_price = current.close

            # 1시간 후
            future_1h = timestamp + timedelta(hours=1)
            candle_1h = await self.db.execute(
                select(MarketCandle)
                .where(
                    MarketCandle.symbol == symbol,
                    MarketCandle.timeframe == timeframe,
                    MarketCandle.open_time == future_1h
                )
                .limit(1)
            )
            c1h = candle_1h.scalar_one_or_none()
            if c1h:
                labels["price_1h"] = c1h.close
                change = (c1h.close - current_price) / current_price * 100
                labels["label_1h"] = self._price_to_label(change)

            # 4시간 후
            future_4h = timestamp + timedelta(hours=4)
            candle_4h = await self.db.execute(
                select(MarketCandle)
                .where(
                    MarketCandle.symbol == symbol,
                    MarketCandle.timeframe == timeframe,
                    MarketCandle.open_time == future_4h
                )
                .limit(1)
            )
            c4h = candle_4h.scalar_one_or_none()
            if c4h:
                labels["price_4h"] = c4h.close
                change = (c4h.close - current_price) / current_price * 100
                labels["label_4h"] = self._price_to_label(change)

            # 24시간 후
            future_24h = timestamp + timedelta(hours=24)
            candle_24h = await self.db.execute(
                select(MarketCandle)
                .where(
                    MarketCandle.symbol == symbol,
                    MarketCandle.timeframe == timeframe,
                    MarketCandle.open_time == future_24h
                )
                .limit(1)
            )
            c24h = candle_24h.scalar_one_or_none()
            if c24h:
                labels["price_24h"] = c24h.close
                change = (c24h.close - current_price) / current_price * 100
                labels["label_24h"] = self._price_to_label(change)

            return labels

        except Exception as e:
            print(f"Error calculating future labels: {e}")
            return labels

    @staticmethod
    def _price_to_label(change_percent: float) -> int:
        """
        가격 변화를 레이블로 변환
        0: 하락 (< -1%)
        1: 횡보 (-1% ~ 1%)
        2: 상승 (> 1%)
        """
        if change_percent < -1.0:
            return 0
        elif change_percent > 1.0:
            return 2
        else:
            return 1

    async def _save_analysis(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
        analysis: Dict[str, Any]
    ) -> bool:
        """AI 분석 결과 저장"""
        try:
            ai_analysis = AIAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                signal=analysis.get("signal"),
                confidence=analysis.get("confidence"),
                direction=analysis.get("direction"),
                analysis_text=analysis.get("analysis"),
                source=analysis.get("source", "unknown"),
                model_version=analysis.get("model_version"),
                raw_response=analysis,
                price_at_analysis=analysis.get("price_at_analysis"),
            )

            self.db.add(ai_analysis)
            await self.db.commit()
            return True

        except Exception as e:
            print(f"Error saving analysis: {e}")
            await self.db.rollback()
            return False

    async def save_signal_history(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        price: float,
        source: str = "combined"
    ) -> bool:
        """매매 신호 히스토리 저장"""
        try:
            signal_record = SignalHistory(
                symbol=symbol,
                timestamp=datetime.now(),
                signal=signal,
                confidence=confidence,
                source=source,
                price_at_signal=price,
            )

            self.db.add(signal_record)
            await self.db.commit()
            return True

        except Exception as e:
            print(f"Error saving signal history: {e}")
            await self.db.rollback()
            return False

    async def get_training_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """학습 데이터 조회"""
        try:
            result = await self.db.execute(
                select(AITrainingData)
                .where(
                    AITrainingData.symbol == symbol,
                    AITrainingData.timeframe == timeframe,
                    AITrainingData.label_1h.isnot(None)  # 레이블이 있는 것만
                )
                .order_by(desc(AITrainingData.timestamp))
                .limit(limit)
            )

            data = result.scalars().all()
            return [
                {
                    "timestamp": d.timestamp,
                    "features": d.features,
                    "price": d.current_price,
                    "label_1h": d.label_1h,
                    "label_4h": d.label_4h,
                    "label_24h": d.label_24h,
                }
                for d in data
            ]

        except Exception as e:
            print(f"Error retrieving training data: {e}")
            return []

    async def get_signal_statistics(
        self,
        symbol: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """신호 통계 조회"""
        try:
            since = datetime.now() - timedelta(days=days)

            result = await self.db.execute(
                select(SignalHistory)
                .where(
                    SignalHistory.symbol == symbol,
                    SignalHistory.timestamp >= since
                )
                .order_by(desc(SignalHistory.timestamp))
            )

            signals = result.scalars().all()

            if not signals:
                return {"total": 0, "buy_count": 0, "sell_count": 0, "hold_count": 0}

            total = len(signals)
            buy_count = sum(1 for s in signals if s.signal == "BUY")
            sell_count = sum(1 for s in signals if s.signal == "SELL")
            hold_count = sum(1 for s in signals if s.signal == "HOLD")

            # 수익성 있는 신호 계산
            profitable = sum(1 for s in signals if s.result == "profit")
            loss = sum(1 for s in signals if s.result == "loss")

            win_rate = (profitable / (profitable + loss)) * 100 if (profitable + loss) > 0 else 0

            return {
                "total": total,
                "buy_count": buy_count,
                "sell_count": sell_count,
                "hold_count": hold_count,
                "profitable": profitable,
                "loss": loss,
                "win_rate": win_rate,
                "avg_confidence": sum(s.confidence for s in signals if s.confidence) / len([s for s in signals if s.confidence]) if signals else 0,
            }

        except Exception as e:
            print(f"Error getting signal statistics: {e}")
            return {}
