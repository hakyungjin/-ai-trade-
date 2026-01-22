"""
시장 데이터 저장 서비스
- 실시간 캔들 데이터 DB 저장
- 기술적 지표 계산 및 저장
- 데이터 조회 및 통계
"""
from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
import numpy as np

from app.models.market_data import MarketCandle, TechnicalIndicator
from app.services.technical_indicators import TechnicalIndicators


class MarketDataService:
    """시장 데이터 관리 서비스"""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.ti = TechnicalIndicators()

    async def save_candles(
        self,
        symbol: str,
        timeframe: str,
        candles: List[Dict[str, Any]]
    ) -> int:
        """
        캔들 데이터 저장

        Args:
            symbol: 거래쌍
            timeframe: 시간프레임
            candles: OHLCV 캔들 리스트

        Returns:
            저장된 캔들 개수
        """
        saved_count = 0

        try:
            print(f"🔍 save_candles called: symbol={symbol}, timeframe={timeframe}, candles={len(candles)}")
            print(f"📋 First candle sample: {candles[0] if candles else 'EMPTY'}")
            
            for idx, candle in enumerate(candles):
                try:
                    # 필드명 확인
                    if "open_time" not in candle:
                        print(f"❌ Candle {idx} missing open_time field. Keys: {candle.keys()}")
                        continue
                        
                    open_time = datetime.fromisoformat(candle["open_time"])

                    # 기존 데이터 확인
                    existing = await self.db.execute(
                        select(MarketCandle).where(
                            MarketCandle.symbol == symbol,
                            MarketCandle.timeframe == timeframe,
                            MarketCandle.open_time == open_time
                        )
                    )

                    if existing.scalar_one_or_none():
                        # print(f"⏭️  Candle {idx} already exists, skipping")
                        continue  # 이미 존재하면 스킵

                    market_candle = MarketCandle(
                        symbol=symbol,
                        timeframe=timeframe,
                        open_time=open_time,
                        open=float(candle["open"]),
                        high=float(candle["high"]),
                        low=float(candle["low"]),
                        close=float(candle["close"]),
                        volume=float(candle["volume"]),
                        close_time=datetime.fromisoformat(candle.get("close_time", candle["open_time"])),
                        quote_volume=float(candle.get("quote_volume", 0)),
                        trades_count=int(candle.get("trades_count", 0)),
                    )

                    self.db.add(market_candle)
                    saved_count += 1

                except Exception as e:
                    print(f"❌ Error processing candle {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            if saved_count > 0:
                await self.db.commit()
                print(f"✅ Committed {saved_count} new candles to DB")
            else:
                print(f"⏭️  No new candles to save (all duplicates or errors)")

            return saved_count

        except Exception as e:
            print(f"❌ Error saving candles: {e}")
            import traceback
            traceback.print_exc()
            await self.db.rollback()
            return 0

    async def calculate_and_save_indicators(
        self,
        symbol: str,
        timeframe: str,
        candles: List[Dict[str, Any]]
    ) -> bool:
        """
        기술적 지표 계산 및 저장

        Args:
            symbol: 거래쌍
            timeframe: 시간프레임
            candles: OHLCV 캔들 리스트

        Returns:
            저장 성공 여부
        """
        if len(candles) < 50:
            return False

        try:
            closes = [float(c["close"]) for c in candles]
            highs = [float(c["high"]) for c in candles]
            lows = [float(c["low"]) for c in candles]
            volumes = [float(c["volume"]) for c in candles]

            # 최근 캔들의 지표만 계산
            latest_candle = candles[-1]
            timestamp = datetime.fromisoformat(latest_candle["open_time"])

            # 기술적 지표 계산
            indicators = TechnicalIndicators()

            # 이동평균
            sma_5 = indicators.calculate_sma(closes, 5)
            sma_10 = indicators.calculate_sma(closes, 10)
            sma_20 = indicators.calculate_sma(closes, 20)
            sma_50 = indicators.calculate_sma(closes, 50)

            ema_12 = indicators.calculate_ema(closes, 12)
            ema_26 = indicators.calculate_ema(closes, 26)

            # RSI
            rsi_14 = indicators.calculate_rsi(closes, 14)

            # MACD
            macd_data = indicators.calculate_macd(closes, 12, 26, 9)

            # 볼린저 밴드
            bb_data = indicators.calculate_bollinger_bands(closes, 20)

            # ATR
            atr_14 = indicators.calculate_atr(highs, lows, closes, 14)

            # 거래량 비율
            volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 else 1.0

            # DB에 저장
            tech_indicator = TechnicalIndicator(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                sma_5=sma_5,
                sma_10=sma_10,
                sma_20=sma_20,
                sma_50=sma_50,
                ema_12=ema_12,
                ema_26=ema_26,
                rsi_14=rsi_14,
                macd=macd_data.get("macd") if macd_data else None,
                macd_signal=macd_data.get("signal") if macd_data else None,
                macd_histogram=macd_data.get("histogram") if macd_data else None,
                bb_upper=bb_data.get("upper") if bb_data else None,
                bb_middle=bb_data.get("middle") if bb_data else None,
                bb_lower=bb_data.get("lower") if bb_data else None,
                bb_width=bb_data.get("width") if bb_data else None,
                atr_14=atr_14,
                volume_ratio=volume_ratio,
            )

            self.db.add(tech_indicator)
            await self.db.commit()

            return True

        except Exception as e:
            print(f"Error calculating indicators: {e}")
            await self.db.rollback()
            return False

    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """
        캔들 데이터 조회

        Args:
            symbol: 거래쌍
            timeframe: 시간프레임
            limit: 조회 개수

        Returns:
            캔들 데이터 리스트
        """
        try:
            result = await self.db.execute(
                select(MarketCandle)
                .where(
                    MarketCandle.symbol == symbol,
                    MarketCandle.timeframe == timeframe
                )
                .order_by(desc(MarketCandle.open_time))
                .limit(limit)
            )

            candles = result.scalars().all()
            candles.reverse()  # 시간순으로 정렬

            return [
                {
                    "open_time": c.open_time.isoformat(),
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                    "quote_volume": c.quote_volume,
                    "trades_count": c.trades_count,
                }
                for c in candles
            ]

        except Exception as e:
            print(f"Error retrieving candles: {e}")
            return []

    async def get_latest_indicator(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """최근 기술적 지표 조회"""
        try:
            result = await self.db.execute(
                select(TechnicalIndicator)
                .where(
                    TechnicalIndicator.symbol == symbol,
                    TechnicalIndicator.timeframe == timeframe
                )
                .order_by(desc(TechnicalIndicator.timestamp))
                .limit(1)
            )

            indicator = result.scalar_one_or_none()

            if not indicator:
                return None

            return {
                "timestamp": indicator.timestamp.isoformat(),
                "sma_5": indicator.sma_5,
                "sma_10": indicator.sma_10,
                "sma_20": indicator.sma_20,
                "sma_50": indicator.sma_50,
                "ema_12": indicator.ema_12,
                "ema_26": indicator.ema_26,
                "rsi_14": indicator.rsi_14,
                "macd": indicator.macd,
                "macd_signal": indicator.macd_signal,
                "macd_histogram": indicator.macd_histogram,
                "bb_upper": indicator.bb_upper,
                "bb_middle": indicator.bb_middle,
                "bb_lower": indicator.bb_lower,
                "bb_width": indicator.bb_width,
                "atr_14": indicator.atr_14,
                "volume_ratio": indicator.volume_ratio,
            }

        except Exception as e:
            print(f"Error retrieving indicator: {e}")
            return None

    async def get_market_statistics(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """시장 통계 계산"""
        candles = await self.get_candles(symbol, timeframe, limit)

        if not candles:
            return {}

        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        volumes = [c["volume"] for c in candles]

        # 가격 통계
        current_price = closes[-1]
        high = max(highs)
        low = min(lows)
        open_price = closes[0]

        # 수익률
        price_change = current_price - open_price
        price_change_percent = (price_change / open_price * 100) if open_price > 0 else 0

        # 변동성
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        volatility = np.std(returns) * 100

        # 거래량
        avg_volume = np.mean(volumes)
        max_volume = max(volumes)

        return {
            "current_price": current_price,
            "open_price": open_price,
            "high": high,
            "low": low,
            "price_change": price_change,
            "price_change_percent": price_change_percent,
            "volatility": volatility,
            "avg_volume": avg_volume,
            "max_volume": max_volume,
            "candle_count": len(candles),
        }
