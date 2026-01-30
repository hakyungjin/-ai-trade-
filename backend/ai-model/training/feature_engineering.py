"""
기술적 지표 기반 피처 엔지니어링

통합 피처 엔지니어링 모듈(unified_feature_engineering)을 사용하여
학습 시에도 추론과 동일한 피처를 생성
"""
import sys
import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

# 통합 피처 모듈 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class FeatureEngineer:
    def __init__(self, use_unified: bool = True):
        """
        Args:
            use_unified: True면 통합 피처 모듈 사용 (추론과 동일한 피처)
        """
        self.use_unified = use_unified
        self._unified_module = None

        if use_unified:
            try:
                from app.services.unified_feature_engineering import (
                    compute_all_features,
                    create_labels as unified_create_labels,
                    get_all_feature_names,
                    get_core_feature_names,
                    prepare_feature_matrix,
                )
                self._unified_module = {
                    'compute_all_features': compute_all_features,
                    'create_labels': unified_create_labels,
                    'get_all_feature_names': get_all_feature_names,
                    'get_core_feature_names': get_core_feature_names,
                    'prepare_feature_matrix': prepare_feature_matrix,
                }
                print(f"[FeatureEngineer] Unified feature module loaded ({len(get_all_feature_names())} features)")
            except ImportError as e:
                print(f"[FeatureEngineer] Unified module not available, using basic features: {e}")
                self.use_unified = False

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        기술적 지표 추가

        기본 지표:
        - 이동평균선 (SMA, EMA)
        - RSI, MACD
        - 볼린저 밴드, ATR
        - 거래량 지표

        통합 모듈 사용 시 추가:
        - OBV, MFI, Williams %R
        - 캔들 패턴 (도지, 망치, 잉걸핑 등)
        - 펌프 앤 덤프 감지
        - 연속 패턴 (모멘텀)
        - 복합 신호
        """
        df = df.copy()

        # 기본 기술적 지표 (항상 계산)
        df = self._add_base_indicators(df)

        # 통합 모듈 사용 시 풍부한 피처 추가
        if self.use_unified and self._unified_module:
            df = self._unified_module['compute_all_features'](df)
            print(f"[FeatureEngineer] Total columns after unified features: {len(df.columns)}")
        else:
            # 기본 피처만 추가 (레거시 호환)
            df = self._add_basic_derived_features(df)

        return df

    def _add_base_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 기술적 지표 계산"""
        # 이동평균선
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()

        df["ema_12"] = df["close"].ewm(span=12).mean()
        df["ema_26"] = df["close"].ewm(span=26).mean()

        # RSI
        df["rsi_14"] = self._calculate_rsi(df["close"], period=14)

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # 볼린저 밴드
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (2 * bb_std)
        df["bb_lower"] = df["bb_middle"] - (2 * bb_std)

        # Stochastic
        highest_high = df['high'].rolling(14).max()
        lowest_low = df['low'].rolling(14).min()
        df['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low + 1e-8)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # ATR (Average True Range)
        df["atr_14"] = self._calculate_atr(df, period=14)

        # OBV (기본 계산 - 통합 모듈이 없을 때 사용)
        if 'obv' not in df.columns:
            df["obv"] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

        return df

    def _add_basic_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 파생 피처 (통합 모듈 미사용 시)"""
        # 거래량 지표
        df["volume_sma"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_sma"] + 1e-8)

        # 가격 변화율
        df["price_change_1"] = df["close"].pct_change(1)
        df["price_change_5"] = df["close"].pct_change(5)
        df["price_change_10"] = df["close"].pct_change(10)
        df["price_change_20"] = df["close"].pct_change(20)

        # 고가/저가 대비 현재가 위치
        df["price_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)

        # 볼린저 밴드 위치
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['close'] + 1e-8)

        # RSI 정규화
        if 'rsi_14' in df.columns:
            df['rsi_normalized'] = df['rsi_14'] / 100

        # MACD 정규화
        if 'macd' in df.columns:
            df['macd_normalized'] = df['macd'] / (df['close'] + 1e-8) * 100

        # EMA 크로스
        if 'ema_12' in df.columns and 'ema_26' in df.columns:
            df['ema_cross'] = (df['ema_12'] - df['ema_26']) / (df['close'] + 1e-8) * 100

        # 변동성
        df['volatility_5'] = df['close'].rolling(5).std() / (df['close'].rolling(5).mean() + 1e-8)
        df['volatility_20'] = df['close'].rolling(20).std() / (df['close'].rolling(20).mean() + 1e-8)
        df['atr_ratio'] = df['atr_14'] / (df['close'] + 1e-8) * 100

        # 캔들 비율
        df['candle_body'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        df['is_bullish'] = (df['close'] > df['open']).astype(int)

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
        lookahead: int = 5,
        num_classes: int = 3
    ) -> pd.DataFrame:
        """
        학습용 레이블 생성

        Args:
            df: 가격 데이터
            threshold: 상승/하락 판단 기준
            lookahead: 미래 캔들 수
            num_classes: 분류 클래스 수 (2, 3, 5)

        Returns:
            DataFrame with labels
        """
        if self.use_unified and self._unified_module:
            return self._unified_module['create_labels'](df, threshold, lookahead, num_classes)

        # 레거시 방식
        df = df.copy()
        future_return = df["close"].shift(-lookahead) / df["close"] - 1

        if num_classes == 2:
            df["label"] = (future_return > 0).astype(int)
        elif num_classes == 3:
            df["label"] = 1  # 횡보
            df.loc[future_return > threshold, "label"] = 2  # 상승
            df.loc[future_return < -threshold, "label"] = 0  # 하락
        elif num_classes == 5:
            strong_threshold = threshold * 2
            df["label"] = 2  # HOLD
            df.loc[future_return > threshold, "label"] = 3
            df.loc[future_return > strong_threshold, "label"] = 4
            df.loc[future_return < -threshold, "label"] = 1
            df.loc[future_return < -strong_threshold, "label"] = 0

        df["label_binary"] = (future_return > 0).astype(int)
        df["future_return"] = future_return

        return df

    def get_feature_columns(self, use_core: bool = False) -> Optional[List[str]]:
        """통합 피처 이름 목록 반환"""
        if self.use_unified and self._unified_module:
            if use_core:
                return self._unified_module['get_core_feature_names']()
            return self._unified_module['get_all_feature_names']()
        return None

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        feature_columns: list = None,
        target_column: str = "label"
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        학습용 데이터셋 준비

        Returns:
            X: feature array
            y: label array
            feature_names: 실제 사용된 피처 이름 목록
        """
        if self.use_unified and self._unified_module and feature_columns is None:
            return self._unified_module['prepare_feature_matrix'](
                df, feature_names=None, target_column=target_column
            )

        df = df.dropna()

        if feature_columns is None:
            feature_columns = self.get_feature_columns() or [
                "sma_20", "sma_50",
                "ema_12", "ema_26",
                "rsi_14", "macd", "macd_signal", "macd_histogram",
                "bb_upper", "bb_lower",
                "atr_14", "obv",
                "price_change_1", "price_change_5", "price_change_10",
                "price_position",
                "bb_position", "bb_width",
                "rsi_normalized", "macd_normalized", "ema_cross",
                "volatility_5", "volatility_20", "atr_ratio",
                "candle_body", "is_bullish",
            ]

        # 존재하는 컬럼만 선택
        available = [col for col in feature_columns if col in df.columns]
        missing = [col for col in feature_columns if col not in df.columns]
        if missing:
            print(f"[FeatureEngineer] Missing features ({len(missing)}): {missing[:10]}...")

        X = df[available].values
        y = df[target_column].values if target_column in df.columns else np.zeros(len(df))

        return X, y, available

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

    engineer = FeatureEngineer(use_unified=True)
    df = engineer.add_technical_indicators(df)
    df = engineer.create_labels(df, threshold=0.02, lookahead=5)

    print("Features added:")
    print(f"Total columns: {len(df.columns)}")
    print(df.columns.tolist())

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    X, y, feature_names = engineer.prepare_dataset(df)
    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Feature count: {len(feature_names)}")
