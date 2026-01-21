"""
Gemini AI를 활용한 차트 분석 서비스
- 실시간 차트 데이터 분석
- 매매 신호 생성
- 시장 상황 해석
"""
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import aiohttp


class GeminiService:
    """Google Gemini API를 활용한 암호화폐 차트 분석"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.model = "gemini-2.5-flash"  # 빠른 응답용
        self.analysis_history: List[Dict] = []  # 분석 히스토리 저장

    async def analyze_chart(
        self,
        symbol: str,
        candles: List[Dict[str, Any]],
        current_price: float,
        timeframe: str = "1h"
    ) -> Dict[str, Any]:
        """
        차트 데이터를 Gemini에게 분석 요청

        Args:
            symbol: 거래쌍 (예: BTCUSDT)
            candles: OHLCV 캔들 데이터 리스트
            current_price: 현재가
            timeframe: 시간 프레임

        Returns:
            분석 결과 (signal, confidence, analysis 등)
        """
        if not self.api_key:
            return self._fallback_response("Gemini API 키가 설정되지 않음")

        # 차트 데이터를 분석용 텍스트로 변환
        chart_summary = self._prepare_chart_data(candles, current_price)

        # 프롬프트 구성
        prompt = self._build_analysis_prompt(symbol, chart_summary, timeframe)

        try:
            # Gemini API 호출
            response = await self._call_gemini_api(prompt)

            # 응답 파싱
            result = self._parse_analysis_response(response, symbol, current_price)

            # 히스토리에 저장
            self.analysis_history.append({
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "result": result
            })

            # 히스토리 100개로 제한
            if len(self.analysis_history) > 100:
                self.analysis_history = self.analysis_history[-100:]

            return result

        except Exception as e:
            print(f"Gemini analysis error: {e}")
            return self._fallback_response(str(e))

    def _prepare_chart_data(
        self,
        candles: List[Dict[str, Any]],
        current_price: float
    ) -> str:
        """차트 데이터를 분석용 텍스트로 변환"""
        if len(candles) < 20:
            return "데이터 부족"

        # 최근 캔들 데이터 요약
        recent = candles[-50:] if len(candles) >= 50 else candles

        # 기본 통계
        closes = [c["close"] for c in recent]
        highs = [c["high"] for c in recent]
        lows = [c["low"] for c in recent]
        volumes = [c["volume"] for c in recent]

        # 이동평균
        ma5 = sum(closes[-5:]) / 5
        ma20 = sum(closes[-20:]) / 20
        ma50 = sum(closes) / len(closes) if len(closes) >= 50 else ma20

        # 가격 변화
        price_change_24h = ((closes[-1] - closes[0]) / closes[0]) * 100 if closes[0] > 0 else 0

        # RSI 계산 (간단 버전)
        gains = []
        losses = []
        for i in range(1, min(15, len(closes))):
            diff = closes[i] - closes[i-1]
            if diff > 0:
                gains.append(diff)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(diff))

        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0.0001
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # 볼린저 밴드
        import statistics
        if len(closes) >= 20:
            bb_sma = sum(closes[-20:]) / 20
            bb_std = statistics.stdev(closes[-20:])
            bb_upper = bb_sma + (2 * bb_std)
            bb_lower = bb_sma - (2 * bb_std)
        else:
            bb_upper = bb_lower = current_price

        # 거래량 분석
        avg_volume = sum(volumes) / len(volumes)
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # 최근 캔들 패턴 (마지막 5개)
        recent_candles = []
        for c in candles[-5:]:
            candle_type = "양봉" if c["close"] > c["open"] else "음봉"
            body_size = abs(c["close"] - c["open"])
            wick_upper = c["high"] - max(c["open"], c["close"])
            wick_lower = min(c["open"], c["close"]) - c["low"]
            recent_candles.append(f"{candle_type}(몸통:{body_size:.2f}, 윗꼬리:{wick_upper:.2f}, 아래꼬리:{wick_lower:.2f})")

        summary = f"""
=== 차트 데이터 요약 ===
현재가: {current_price:,.2f}
24시간 변화율: {price_change_24h:+.2f}%

이동평균선:
- MA5: {ma5:,.2f} (현재가 {'위' if current_price > ma5 else '아래'})
- MA20: {ma20:,.2f} (현재가 {'위' if current_price > ma20 else '아래'})
- MA50: {ma50:,.2f} (현재가 {'위' if current_price > ma50 else '아래'})

기술적 지표:
- RSI(14): {rsi:.1f} ({'과매수' if rsi > 70 else '과매도' if rsi < 30 else '중립'})
- 볼린저밴드 상단: {bb_upper:,.2f}
- 볼린저밴드 하단: {bb_lower:,.2f}
- 현재가 볼린저 위치: {'상단 근접' if current_price > bb_upper * 0.98 else '하단 근접' if current_price < bb_lower * 1.02 else '중간'}

거래량:
- 현재 거래량 비율: {volume_ratio:.2f}x (평균 대비)
- {'거래량 급증' if volume_ratio > 2 else '거래량 증가' if volume_ratio > 1.5 else '보통'}

최근 5개 캔들:
{chr(10).join(recent_candles)}

최근 고가: {max(highs):,.2f}
최근 저가: {min(lows):,.2f}
"""
        return summary

    def _build_analysis_prompt(
        self,
        symbol: str,
        chart_summary: str,
        timeframe: str
    ) -> str:
        """분석 요청 프롬프트 생성"""
        return f"""당신은 전문 암호화폐 트레이더입니다. 다음 {symbol} {timeframe} 차트 데이터를 분석하고 매매 신호를 제공해주세요.

{chart_summary}

다음 JSON 형식으로만 응답해주세요 (다른 텍스트 없이):
{{
    "signal": "BUY" 또는 "SELL" 또는 "HOLD",
    "confidence": 0.0에서 1.0 사이의 신뢰도,
    "direction": "UP" 또는 "DOWN" 또는 "NEUTRAL",
    "short_term": "1-4시간 내 예상 방향과 이유",
    "mid_term": "1-3일 내 예상 방향과 이유",
    "key_levels": {{
        "support": 주요 지지선 가격,
        "resistance": 주요 저항선 가격
    }},
    "risk_reward": "리스크 대비 보상 비율 평가",
    "analysis": "종합 분석 (2-3문장)"
}}

주의사항:
1. 현재 시장 상황을 객관적으로 분석하세요
2. 확신이 낮으면 HOLD를 권장하세요
3. 리스크 관리를 항상 고려하세요
4. JSON 형식을 정확히 지켜주세요"""

    async def _call_gemini_api(self, prompt: str) -> str:
        """Gemini API 호출"""
        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.3,  # 낮은 온도로 일관된 응답
                "maxOutputTokens": 1024,
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Gemini API error: {response.status} - {error_text}")

                data = await response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]

    def _parse_analysis_response(
        self,
        response: str,
        symbol: str,
        current_price: float
    ) -> Dict[str, Any]:
        """Gemini 응답 파싱"""
        try:
            # JSON 부분 추출 (```json ... ``` 형태일 수 있음)
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            data = json.loads(json_str.strip())

            return {
                "signal": data.get("signal", "HOLD"),
                "confidence": float(data.get("confidence", 0.5)),
                "direction": data.get("direction", "NEUTRAL"),
                "analysis": data.get("analysis", "분석 결과 없음"),
                "short_term": data.get("short_term", ""),
                "mid_term": data.get("mid_term", ""),
                "key_levels": data.get("key_levels", {}),
                "risk_reward": data.get("risk_reward", ""),
                "source": "gemini",
                "model": self.model,
                "symbol": symbol,
                "current_price": current_price
            }
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Failed to parse Gemini response: {e}")
            print(f"Raw response: {response}")
            return self._fallback_response(f"응답 파싱 실패: {str(e)}")

    def _fallback_response(self, reason: str) -> Dict[str, Any]:
        """폴백 응답"""
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "direction": "NEUTRAL",
            "analysis": f"AI 분석 불가: {reason}",
            "source": "fallback",
            "error": reason
        }

    async def get_market_sentiment(
        self,
        symbols: List[str],
        candles_data: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """여러 코인의 전체 시장 심리 분석"""
        if not self.api_key:
            return {"sentiment": "neutral", "analysis": "API 키 없음"}

        # 각 심볼의 간단한 요약 생성
        summaries = []
        for symbol in symbols:
            if symbol in candles_data and candles_data[symbol]:
                candles = candles_data[symbol]
                closes = [c["close"] for c in candles[-20:]]
                if closes:
                    change = ((closes[-1] - closes[0]) / closes[0]) * 100
                    summaries.append(f"{symbol}: {change:+.2f}%")

        prompt = f"""다음은 주요 암호화폐들의 최근 가격 변화입니다:
{chr(10).join(summaries)}

현재 전체 시장 심리를 분석해주세요. JSON 형식으로만 응답:
{{
    "sentiment": "bullish" 또는 "bearish" 또는 "neutral",
    "strength": 0.0에서 1.0 사이,
    "analysis": "간단한 시장 분석"
}}"""

        try:
            response = await self._call_gemini_api(prompt)
            json_str = response
            if "```" in response:
                json_str = response.split("```")[1].split("```")[0]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
            return json.loads(json_str.strip())
        except Exception as e:
            return {"sentiment": "neutral", "analysis": f"분석 실패: {e}"}

    def get_recent_analyses(self, limit: int = 10) -> List[Dict]:
        """최근 분석 히스토리 조회"""
        return self.analysis_history[-limit:]