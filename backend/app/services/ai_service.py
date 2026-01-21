import numpy as np
import re
from typing import Dict, Any, List, Optional
import asyncio
import os

# PyTorch 관련 (선택적 임포트)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class AIService:
    """AI 예측 및 프롬프트 파싱 서비스"""

    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.feature_columns = None
        self.sequence_length = 20
        self.num_classes = 3
        self.device = None
        self.model_loaded = False
        self.active_signals = []

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        """학습된 PyTorch 모델 로드"""
        if not TORCH_AVAILABLE:
            print("PyTorch not available, using rule-based prediction")
            return False

        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(model_path, map_location=self.device)

            # 모델 설정 복원
            model_type = checkpoint.get("model_type", "lstm")
            self.sequence_length = checkpoint.get("sequence_length", 20)
            self.num_classes = checkpoint.get("num_classes", 3)
            self.feature_columns = checkpoint.get("feature_columns", [])

            # Scaler 복원
            self.scaler_mean = np.array(checkpoint.get("scaler_mean", []))
            self.scaler_scale = np.array(checkpoint.get("scaler_scale", []))

            # 모델 생성 및 가중치 로드
            input_size = len(self.feature_columns)
            self.model = self._create_model(model_type, input_size)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            self.model_loaded = True
            print(f"Model loaded successfully: {model_type} with {input_size} features")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def _create_model(self, model_type: str, input_size: int):
        """모델 타입에 따라 모델 생성"""
        if model_type == "lstm":
            return LSTMClassifier(input_size=input_size, num_classes=self.num_classes)
        elif model_type == "transformer":
            return TransformerClassifier(input_size=input_size, num_classes=self.num_classes)
        else:
            return SimpleMLP(input_size=input_size * self.sequence_length, num_classes=self.num_classes)

    async def predict_signal(
        self,
        symbol: str,
        candles: List[Dict[str, Any]],
        current_price: float
    ) -> Dict[str, Any]:
        """
        가격 데이터 기반 매매 신호 예측

        학습된 딥러닝 모델이 있으면 사용하고,
        없으면 기술적 지표 기반 규칙 사용
        """
        if len(candles) < self.sequence_length + 50:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "direction": "NEUTRAL",
                "analysis": "데이터 부족"
            }

        # 기술적 지표 계산 (모델/규칙 모두에 필요)
        closes = np.array([c["close"] for c in candles])
        highs = np.array([c["high"] for c in candles])
        lows = np.array([c["low"] for c in candles])
        volumes = np.array([c["volume"] for c in candles])

        # 학습된 모델이 있으면 딥러닝 예측 사용
        if self.model_loaded and TORCH_AVAILABLE:
            return await self._predict_with_model(closes, highs, lows, volumes, current_price)

        # 모델이 없으면 규칙 기반 예측
        return await self._predict_with_rules(closes, volumes, current_price)

    async def _predict_with_model(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        current_price: float
    ) -> Dict[str, Any]:
        """학습된 딥러닝 모델로 예측"""
        try:
            # 피처 계산
            features = self._compute_features(closes, highs, lows, volumes)

            # 스케일링
            features_scaled = (features - self.scaler_mean) / (self.scaler_scale + 1e-8)

            # 시퀀스 생성 (마지막 sequence_length개)
            seq = features_scaled[-self.sequence_length:]
            seq = seq.reshape(1, self.sequence_length, -1)

            # 텐서 변환 및 예측
            x_tensor = torch.FloatTensor(seq).to(self.device)
            with torch.no_grad():
                outputs = self.model(x_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                pred_class = np.argmax(probs)

            # 클래스 해석 (0: 하락, 1: 횡보, 2: 상승)
            class_map = {0: ("SELL", "DOWN"), 1: ("HOLD", "NEUTRAL"), 2: ("BUY", "UP")}
            signal, direction = class_map[pred_class]
            confidence = float(probs[pred_class])

            # RSI 등 추가 지표로 분석 텍스트 생성
            rsi = self._calculate_rsi(closes)
            analysis_parts = [f"AI 모델 예측 (신뢰도 {confidence*100:.1f}%)"]
            if rsi < 30:
                analysis_parts.append(f"RSI 과매도 ({rsi:.1f})")
            elif rsi > 70:
                analysis_parts.append(f"RSI 과매수 ({rsi:.1f})")

            return {
                "signal": signal,
                "confidence": confidence,
                "direction": direction,
                "analysis": " | ".join(analysis_parts),
                "model_probs": {"down": float(probs[0]), "neutral": float(probs[1]), "up": float(probs[2])},
                "indicators": {
                    "rsi": rsi,
                    "ma_5": float(np.mean(closes[-5:])),
                    "ma_20": float(np.mean(closes[-20:]))
                }
            }
        except Exception as e:
            print(f"Model prediction error: {e}")
            return await self._predict_with_rules(closes, volumes, current_price)

    def _compute_features(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray
    ) -> np.ndarray:
        """기술적 지표 피처 계산"""
        n = len(closes)
        features = np.zeros((n, 19))

        # 이동평균선
        features[:, 0] = self._rolling_mean(closes, 5)   # sma_5
        features[:, 1] = self._rolling_mean(closes, 10)  # sma_10
        features[:, 2] = self._rolling_mean(closes, 20)  # sma_20
        features[:, 3] = self._rolling_mean(closes, 50)  # sma_50

        # EMA
        features[:, 4] = self._ema(closes, 12)  # ema_12
        features[:, 5] = self._ema(closes, 26)  # ema_26

        # RSI
        features[:, 6] = self._rolling_rsi(closes, 14)  # rsi

        # MACD
        macd = features[:, 4] - features[:, 5]
        features[:, 7] = macd  # macd
        features[:, 8] = self._ema(macd, 9)  # macd_signal
        features[:, 9] = macd - features[:, 8]  # macd_hist

        # 볼린저 밴드
        bb_middle = features[:, 2]  # sma_20
        bb_std = self._rolling_std(closes, 20)
        features[:, 10] = bb_middle + 2 * bb_std  # bb_upper
        features[:, 11] = bb_middle - 2 * bb_std  # bb_lower
        features[:, 12] = (features[:, 10] - features[:, 11]) / (bb_middle + 1e-8)  # bb_width

        # ATR
        features[:, 13] = self._calculate_atr(highs, lows, closes, 14)  # atr

        # 거래량
        volume_sma = self._rolling_mean(volumes, 20)
        features[:, 14] = volumes / (volume_sma + 1e-8)  # volume_ratio

        # 가격 변화율
        features[:, 15] = self._pct_change(closes, 1)   # price_change_1
        features[:, 16] = self._pct_change(closes, 5)   # price_change_5
        features[:, 17] = self._pct_change(closes, 10)  # price_change_10

        # 고가/저가 비율
        features[:, 18] = (closes - lows) / (highs - lows + 1e-8)  # high_low_ratio

        return features

    def _rolling_mean(self, arr: np.ndarray, window: int) -> np.ndarray:
        """롤링 평균"""
        result = np.full(len(arr), np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result

    def _rolling_std(self, arr: np.ndarray, window: int) -> np.ndarray:
        """롤링 표준편차"""
        result = np.full(len(arr), np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.std(arr[i - window + 1:i + 1])
        return result

    def _ema(self, arr: np.ndarray, span: int) -> np.ndarray:
        """지수이동평균"""
        alpha = 2 / (span + 1)
        result = np.zeros(len(arr))
        result[0] = arr[0]
        for i in range(1, len(arr)):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
        return result

    def _rolling_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """롤링 RSI"""
        result = np.full(len(prices), 50.0)
        deltas = np.diff(prices, prepend=prices[0])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        for i in range(period, len(prices)):
            avg_gain = np.mean(gains[i - period + 1:i + 1])
            avg_loss = np.mean(losses[i - period + 1:i + 1])
            if avg_loss == 0:
                result[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                result[i] = 100 - (100 / (1 + rs))
        return result

    def _calculate_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """ATR 계산"""
        tr = np.maximum(
            highs - lows,
            np.maximum(
                np.abs(highs - np.roll(closes, 1)),
                np.abs(lows - np.roll(closes, 1))
            )
        )
        tr[0] = highs[0] - lows[0]
        return self._rolling_mean(tr, period)

    def _pct_change(self, arr: np.ndarray, periods: int) -> np.ndarray:
        """변화율 계산"""
        result = np.zeros(len(arr))
        result[periods:] = (arr[periods:] - arr[:-periods]) / (arr[:-periods] + 1e-8)
        return result

    async def _predict_with_rules(
        self,
        closes: np.ndarray,
        volumes: np.ndarray,
        current_price: float
    ) -> Dict[str, Any]:
        """규칙 기반 예측 (기존 로직)"""

        # 이동평균
        ma_5 = np.mean(closes[-5:])
        ma_20 = np.mean(closes[-20:])
        ma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else ma_20

        # RSI 계산
        rsi = self._calculate_rsi(closes)

        # 볼린저 밴드
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(closes)

        # 거래량 분석
        avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # 신호 판단 로직
        signal = "HOLD"
        confidence = 0.5
        direction = "NEUTRAL"
        analysis_parts = []

        # 골든 크로스 / 데드 크로스
        if ma_5 > ma_20 and closes[-2] < np.mean(closes[-7:-2]):
            signal = "BUY"
            confidence += 0.15
            direction = "UP"
            analysis_parts.append("단기 이평선이 장기 이평선을 상향 돌파")

        elif ma_5 < ma_20 and closes[-2] > np.mean(closes[-7:-2]):
            signal = "SELL"
            confidence += 0.15
            direction = "DOWN"
            analysis_parts.append("단기 이평선이 장기 이평선을 하향 돌파")

        # RSI 기반 판단
        if rsi < 30:
            if signal != "SELL":
                signal = "BUY"
                confidence += 0.1
            direction = "UP"
            analysis_parts.append(f"RSI 과매도 구간 ({rsi:.1f})")

        elif rsi > 70:
            if signal != "BUY":
                signal = "SELL"
                confidence += 0.1
            direction = "DOWN"
            analysis_parts.append(f"RSI 과매수 구간 ({rsi:.1f})")

        # 볼린저 밴드 기반 판단
        if current_price < bb_lower:
            confidence += 0.1
            if signal != "SELL":
                signal = "BUY"
            analysis_parts.append("하단 볼린저 밴드 터치")

        elif current_price > bb_upper:
            confidence += 0.1
            if signal != "BUY":
                signal = "SELL"
            analysis_parts.append("상단 볼린저 밴드 터치")

        # 거래량 확인
        if volume_ratio > 1.5:
            confidence += 0.05
            analysis_parts.append(f"거래량 증가 ({volume_ratio:.1f}x)")

        # 신뢰도 정규화
        confidence = min(confidence, 0.95)

        analysis = " | ".join(analysis_parts) if analysis_parts else "뚜렷한 신호 없음"

        return {
            "signal": signal,
            "confidence": confidence,
            "direction": direction,
            "analysis": analysis,
            "indicators": {
                "rsi": rsi,
                "ma_5": ma_5,
                "ma_20": ma_20,
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
                "volume_ratio": volume_ratio
            }
        }

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """RSI 계산"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_bollinger_bands(
        self,
        prices: np.ndarray,
        period: int = 20,
        std_dev: float = 2.0
    ) -> tuple:
        """볼린저 밴드 계산"""
        if len(prices) < period:
            return prices[-1], prices[-1], prices[-1]

        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])

        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)

        return upper, sma, lower

    async def parse_trading_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        자연어 프롬프트를 거래 규칙으로 변환

        예시 입력:
        - "스탑로스 3%로 설정해줘"
        - "익절 5%, 손절 2%"
        - "최대 포지션 50 USDT로 제한"
        """
        result = {
            "stop_loss": None,
            "take_profit": None,
            "max_position": None,
            "trailing_stop": None,
            "description": ""
        }

        prompt_lower = prompt.lower()

        # 스탑로스 파싱
        stop_loss_patterns = [
            r"스탑로스\s*(\d+(?:\.\d+)?)\s*%",
            r"손절\s*(\d+(?:\.\d+)?)\s*%",
            r"stop\s*loss\s*(\d+(?:\.\d+)?)\s*%",
            r"sl\s*(\d+(?:\.\d+)?)\s*%"
        ]
        for pattern in stop_loss_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                result["stop_loss"] = float(match.group(1)) / 100
                break

        # 익절 파싱
        take_profit_patterns = [
            r"익절\s*(\d+(?:\.\d+)?)\s*%",
            r"take\s*profit\s*(\d+(?:\.\d+)?)\s*%",
            r"tp\s*(\d+(?:\.\d+)?)\s*%",
            r"이익실현\s*(\d+(?:\.\d+)?)\s*%"
        ]
        for pattern in take_profit_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                result["take_profit"] = float(match.group(1)) / 100
                break

        # 최대 포지션 파싱
        position_patterns = [
            r"최대\s*포지션\s*(\d+(?:\.\d+)?)\s*(?:usdt|달러)?",
            r"max\s*position\s*(\d+(?:\.\d+)?)",
            r"포지션\s*(\d+(?:\.\d+)?)\s*(?:usdt|달러)?\s*(?:제한|이하)"
        ]
        for pattern in position_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                result["max_position"] = float(match.group(1))
                break

        # 트레일링 스탑 파싱
        trailing_patterns = [
            r"트레일링\s*(?:스탑)?\s*(\d+(?:\.\d+)?)\s*%",
            r"trailing\s*stop\s*(\d+(?:\.\d+)?)\s*%"
        ]
        for pattern in trailing_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                result["trailing_stop"] = float(match.group(1)) / 100
                break

        # 설명 생성
        descriptions = []
        if result["stop_loss"]:
            descriptions.append(f"스탑로스 {result['stop_loss']*100}%")
        if result["take_profit"]:
            descriptions.append(f"익절 {result['take_profit']*100}%")
        if result["max_position"]:
            descriptions.append(f"최대 포지션 {result['max_position']} USDT")
        if result["trailing_stop"]:
            descriptions.append(f"트레일링 스탑 {result['trailing_stop']*100}%")

        result["description"] = ", ".join(descriptions) if descriptions else "설정 변경 없음"

        return result

    async def analyze_market(
        self,
        symbol: str,
        candles: List[Dict[str, Any]],
        current_price: float
    ) -> Dict[str, Any]:
        """종합 시장 분석"""
        prediction = await self.predict_signal(symbol, candles, current_price)

        # 추가 분석
        closes = np.array([c["close"] for c in candles])
        highs = np.array([c["high"] for c in candles])
        lows = np.array([c["low"] for c in candles])

        # 변동성 계산
        volatility = np.std(closes[-20:]) / np.mean(closes[-20:]) * 100

        # 추세 강도
        trend_strength = abs(closes[-1] - closes[-20]) / closes[-20] * 100

        # 지지/저항 레벨 (간단한 버전)
        support = np.min(lows[-20:])
        resistance = np.max(highs[-20:])

        return {
            "symbol": symbol,
            "current_price": current_price,
            "prediction": prediction,
            "volatility": f"{volatility:.2f}%",
            "trend_strength": f"{trend_strength:.2f}%",
            "support_level": support,
            "resistance_level": resistance,
            "recommendation": self._generate_recommendation(prediction, volatility)
        }

    def _generate_recommendation(
        self,
        prediction: Dict[str, Any],
        volatility: float
    ) -> str:
        """거래 추천 생성"""
        signal = prediction["signal"]
        confidence = prediction["confidence"]

        if confidence < 0.5:
            return "관망 추천 - 신호 불확실"

        if volatility > 5:
            return f"주의: 높은 변동성 ({volatility:.1f}%) - 포지션 크기 축소 권장"

        if signal == "BUY" and confidence > 0.6:
            return "매수 고려 - 상승 신호 감지"
        elif signal == "SELL" and confidence > 0.6:
            return "매도 고려 - 하락 신호 감지"

        return "관망 추천"

    async def get_active_signals(self) -> List[Dict[str, Any]]:
        """현재 활성화된 신호 목록"""
        return self.active_signals


# PyTorch 모델 클래스 정의 (모델 로드용)
if TORCH_AVAILABLE:
    import torch.nn as nn

    class LSTMClassifier(nn.Module):
        """LSTM 기반 가격 방향 예측 모델"""
        def __init__(
            self,
            input_size: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            num_classes: int = 3,
            dropout: float = 0.2
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )

            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1)
            )

            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_classes)
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            attn_weights = torch.softmax(
                self.attention(lstm_out).squeeze(-1), dim=1
            )
            context = torch.bmm(
                attn_weights.unsqueeze(1), lstm_out
            ).squeeze(1)
            return self.classifier(context)

    class TransformerClassifier(nn.Module):
        """Transformer 기반 가격 예측 모델"""
        def __init__(
            self,
            input_size: int,
            d_model: int = 128,
            nhead: int = 4,
            num_layers: int = 2,
            num_classes: int = 3,
            dropout: float = 0.1
        ):
            super().__init__()
            self.input_projection = nn.Linear(input_size, d_model)

            # Positional Encoding
            self.register_buffer(
                "pe",
                self._generate_positional_encoding(d_model, 500)
            )
            self.dropout = nn.Dropout(dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes)
            )

        def _generate_positional_encoding(self, d_model: int, max_len: int):
            import math
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe.unsqueeze(0)

        def forward(self, x):
            x = self.input_projection(x)
            x = x + self.pe[:, :x.size(1), :]
            x = self.dropout(x)
            x = self.transformer(x)
            x = x.mean(dim=1)
            return self.classifier(x)

    class SimpleMLP(nn.Module):
        """간단한 MLP 모델"""
        def __init__(
            self,
            input_size: int,
            hidden_sizes: list = None,
            num_classes: int = 3,
            dropout: float = 0.2
        ):
            super().__init__()
            if hidden_sizes is None:
                hidden_sizes = [512, 256, 128]

            layers = []
            prev_size = input_size
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.BatchNorm1d(hidden_size)
                ])
                prev_size = hidden_size
            layers.append(nn.Linear(prev_size, num_classes))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            if len(x.shape) == 3:
                x = x.reshape(x.size(0), -1)
            return self.network(x)
