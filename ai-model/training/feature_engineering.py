"""
기술적 지표 기반 피처 엔지니어링
"""
import pandas as pd
import numpy as np
from typing import Tuple


class FeatureEngineer:
    def __init__(self):
        pass

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        기술적 지표 추가

        - 이동평균선 (SMA, EMA)
        - RSI
        - MACD
        - 볼린저 밴드
        - ATR
        - 거래량 지표
        """
        df = df.copy()

        # 이동평균선
        df["sma_5"] = df["close"].rolling(window=5).mean()
        df["sma_10"] = df["close"].rolling(window=10).mean()
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()

        df["ema_12"] = df["close"].ewm(span=12).mean()
        df["ema_26"] = df["close"].ewm(span=26).mean()

        # RSI
        df["rsi"] = self._calculate_rsi(df["close"], period=14)

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # 볼린저 밴드
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (2 * bb_std)
        df["bb_lower"] = df["bb_middle"] - (2 * bb_std)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # ATR (Average True Range)
        df["atr"] = self._calculate_atr(df, period=14)

        # 거래량 지표
        df["volume_sma"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # 가격 변화율
        df["price_change_1"] = df["close"].pct_change(1)
        df["price_change_5"] = df["close"].pct_change(5)
        df["price_change_10"] = df["close"].pct_change(10)

        # 고가/저가 대비 현재가 위치
        df["high_low_ratio"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR 계산"""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def create_labels(
        self,
        df: pd.DataFrame,
        threshold: float = 0.02,
        lookahead: int = 5
    ) -> pd.DataFrame:
        """
        학습용 레이블 생성

        Args:
            df: 가격 데이터
            threshold: 상승/하락 판단 기준 (2% = 0.02)
            lookahead: 미래 몇 개 캔들을 볼 것인지

        Returns:
            DataFrame with labels (0: 하락, 1: 횡보, 2: 상승)
        """
        df = df.copy()

        # 미래 가격 대비 변화율
        future_return = df["close"].shift(-lookahead) / df["close"] - 1

        # 레이블 생성
        df["label"] = 1  # 기본값: 횡보
        df.loc[future_return > threshold, "label"] = 2  # 상승
        df.loc[future_return < -threshold, "label"] = 0  # 하락

        # 이진 분류용 레이블 (상승 여부)
        df["label_binary"] = (future_return > 0).astype(int)

        # 미래 수익률 (회귀용)
        df["future_return"] = future_return

        return df

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        feature_columns: list = None,
        target_column: str = "label"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        학습용 데이터셋 준비

        Returns:
            X: feature array
            y: label array
        """
        df = df.dropna()

        if feature_columns is None:
            # 기본 피처 컬럼
            feature_columns = [
                "sma_5", "sma_10", "sma_20", "sma_50",
                "ema_12", "ema_26",
                "rsi", "macd", "macd_signal", "macd_hist",
                "bb_upper", "bb_lower", "bb_width",
                "atr", "volume_ratio",
                "price_change_1", "price_change_5", "price_change_10",
                "high_low_ratio"
            ]

        # 존재하는 컬럼만 선택
        feature_columns = [col for col in feature_columns if col in df.columns]

        X = df[feature_columns].values
        y = df[target_column].values

        return X, y

    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        LSTM용 시퀀스 데이터 생성

        Args:
            X: feature array (samples, features)
            y: label array
            sequence_length: 시퀀스 길이

        Returns:
            X_seq: (samples, sequence_length, features)
            y_seq: (samples,)
        """
        X_seq = []
        y_seq = []

        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])

        return np.array(X_seq), np.array(y_seq)


if __name__ == "__main__":
    # 테스트
    from data_collector import DataCollector

    collector = DataCollector()
    df = collector.fetch_historical_data("BTCUSDT", "1h", start_date="2024-01-01")

    engineer = FeatureEngineer()
    df = engineer.add_technical_indicators(df)
    df = engineer.create_labels(df, threshold=0.02, lookahead=5)

    print("Features added:")
    print(df.columns.tolist())

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    X, y = engineer.prepare_dataset(df)
    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
