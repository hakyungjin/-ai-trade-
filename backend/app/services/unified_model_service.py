"""
Unified Model Service
통합 LSTM 모델을 사용한 예측 서비스

Features:
- PyTorch 통합 모델 로드
- Asset ID 기반 예측
- 실시간 추론
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from pathlib import Path
import joblib

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent
AI_MODEL_PATH = PROJECT_ROOT / "ai-model"
sys.path.insert(0, str(AI_MODEL_PATH))

from app.services.technical_indicators import TechnicalIndicators
from app.services.unified_feature_engineering import compute_all_features
from app.services.asset_mapping_service import AssetMappingService
from models.unified_lstm_model import UnifiedLSTMModel

logger = logging.getLogger(__name__)


class UnifiedModelService:
    """통합 모델 예측 서비스"""

    # 레이블 매핑
    LABEL_MAP_3 = {
        0: ('SELL', -1),
        1: ('HOLD', 0),
        2: ('BUY', 1)
    }

    LABEL_MAP_5 = {
        0: ('STRONG_SELL', -2),
        1: ('SELL', -1),
        2: ('HOLD', 0),
        3: ('BUY', 1),
        4: ('STRONG_BUY', 2)
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        features_path: Optional[str] = None,
        device: str = None
    ):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_loaded = False
        self.num_classes = 3
        self.sequence_length = 60
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # 레이블 매핑
        self.label_map = self.LABEL_MAP_3

        if model_path and scaler_path and features_path:
            self.load_model(model_path, scaler_path, features_path)

    def load_model(self, model_path: str, scaler_path: str, features_path: str) -> bool:
        """통합 모델 로드"""
        try:
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                return False

            # Checkpoint 로드
            checkpoint = torch.load(model_path, map_location=self.device)

            # 모델 구조 재구성 (checkpoint에서 파라미터 추출)
            # 실제로는 모델 config도 함께 저장해야 함
            self.model = UnifiedLSTMModel(
                num_assets=500,  # 설정에서 읽어야 함
                embedding_dim=16,
                num_features=100,  # features_path에서 계산
                hidden_size=64,
                num_classes=3,
                num_layers=2,
                dropout=0.3
            )

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            # Scaler & Features 로드
            self.scaler = joblib.load(scaler_path)
            self.feature_names = joblib.load(features_path)

            self.num_classes = checkpoint.get('num_classes', 3)
            self.label_map = self.LABEL_MAP_3 if self.num_classes == 3 else self.LABEL_MAP_5

            self.is_loaded = True

            logger.info(f"✅ Loaded unified model from {model_path}")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   Features: {len(self.feature_names)}, Classes: {self.num_classes}")
            logger.info(f"   Sequence length: {self.sequence_length}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to load unified model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.is_loaded = False
            return False

    async def predict(
        self,
        symbol: str,
        candles: List[Dict[str, Any]],
        db
    ) -> Dict[str, Any]:
        """
        통합 모델로 예측 수행

        Args:
            symbol: 자산 심볼 (BTCUSDT, AAPL 등)
            candles: 캔들 데이터 리스트 (최소 60개)
            db: DB 세션 (Asset ID 조회용)

        Returns:
            {
                'signal': 'BUY' | 'SELL' | 'HOLD',
                'detailed_signal': 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL',
                'confidence': 0.0 ~ 1.0,
                'probabilities': {...},
                'asset_id': int
            }
        """
        if not self.is_loaded:
            return self._default_response("Unified model not loaded")

        if len(candles) < self.sequence_length:
            return self._default_response(
                f"Insufficient data: {len(candles)} candles (min {self.sequence_length})"
            )

        try:
            # 1. Asset ID 조회
            asset_id = await AssetMappingService.get_asset_id(db, symbol, create_if_missing=True)
            if asset_id is None:
                return self._default_response(f"Failed to get asset_id for {symbol}")

            logger.info(f"🔍 Predicting for {symbol} (asset_id={asset_id})...")

            # 2. DataFrame 생성 및 피처 계산
            df = pd.DataFrame(candles)

            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')

            # 기술적 지표 계산
            df = TechnicalIndicators.calculate_all_indicators(df)

            # 통합 피처 엔지니어링
            df = compute_all_features(df)

            # NaN 처리
            df[self.feature_names] = df[self.feature_names].fillna(0)
            df[self.feature_names] = df[self.feature_names].replace([np.inf, -np.inf], 0)

            # 3. 마지막 60개 캔들 추출
            if len(df) < self.sequence_length:
                return self._default_response(
                    f"Not enough data after processing: {len(df)} < {self.sequence_length}"
                )

            last_60 = df[self.feature_names].iloc[-self.sequence_length:].values

            # 4. 정규화
            last_60_scaled = self.scaler.transform(last_60)

            # 5. Tensor 변환
            time_series = torch.FloatTensor(last_60_scaled).unsqueeze(0).to(self.device)  # (1, 60, features)
            asset_id_tensor = torch.LongTensor([asset_id]).to(self.device)  # (1,)

            # 6. 예측
            with torch.no_grad():
                logits = self.model(time_series, asset_id_tensor)
                probabilities = F.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            # 7. 결과 매핑
            detailed_signal, signal_value = self.label_map[predicted_class]
            simple_signal = self._to_simple_signal(detailed_signal)

            # 확률 딕셔너리
            prob_dict = {}
            for class_id, (label, _) in self.label_map.items():
                if class_id < probabilities.size(1):
                    prob_dict[label] = float(probabilities[0][class_id].item())

            result = {
                'signal': simple_signal,
                'detailed_signal': detailed_signal,
                'signal_value': signal_value,
                'confidence': confidence,
                'direction': 'UP' if signal_value > 0 else ('DOWN' if signal_value < 0 else 'NEUTRAL'),
                'probabilities': prob_dict,
                'asset_id': asset_id,
                'analysis': self._generate_analysis(detailed_signal, confidence, prob_dict)
            }

            logger.info(f"✅ Prediction: {simple_signal} ({detailed_signal}) with {confidence:.2%} confidence")

            return result

        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._default_response(f"Prediction error: {str(e)}")

    def _to_simple_signal(self, detailed_signal: str) -> str:
        """상세 신호 → 간단한 신호 변환"""
        mapping = {
            'STRONG_SELL': 'SELL',
            'SELL': 'SELL',
            'HOLD': 'HOLD',
            'BUY': 'BUY',
            'STRONG_BUY': 'BUY'
        }
        return mapping.get(detailed_signal, 'HOLD')

    def _generate_analysis(
        self,
        detailed_signal: str,
        confidence: float,
        probabilities: Dict[str, float]
    ) -> str:
        """분석 텍스트 생성"""
        confidence_text = "높은" if confidence > 0.7 else ("중간" if confidence > 0.5 else "낮은")

        analysis = f"{detailed_signal} 신호가 {confidence_text} 확신도({confidence:.1%})로 감지되었습니다.\n\n"
        analysis += "확률 분포:\n"

        for label, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 20)
            analysis += f"  {label:15s}: {prob:5.1%} {bar}\n"

        return analysis

    def _default_response(self, message: str) -> Dict[str, Any]:
        """기본 응답 (에러 시)"""
        logger.warning(f"⚠️ Default response: {message}")
        return {
            'signal': 'HOLD',
            'detailed_signal': 'HOLD',
            'signal_value': 0,
            'confidence': 0.0,
            'direction': 'NEUTRAL',
            'probabilities': {'HOLD': 1.0},
            'analysis': f"예측 실패: {message}",
            'error': message
        }


# ===== 글로벌 인스턴스 (싱글톤) =====
_unified_service_instance: Optional[UnifiedModelService] = None


def get_unified_service(force_reload: bool = False) -> Optional[UnifiedModelService]:
    """통합 모델 서비스 싱글톤 인스턴스 반환"""
    global _unified_service_instance

    if _unified_service_instance is None or force_reload:
        # 최신 모델 파일 찾기
        models_dir = AI_MODEL_PATH / "models"

        # unified_YYYYMMDD_HHMMSS 디렉토리 찾기
        unified_dirs = sorted(models_dir.glob("unified_*"), reverse=True)

        if not unified_dirs:
            logger.warning("⚠️ No unified model directories found")
            return None

        model_dir = unified_dirs[0]
        model_path = model_dir / "unified_model_best.pt"
        scaler_path = model_dir / "unified_scaler.joblib"
        features_path = model_dir / "unified_features.joblib"

        if not model_path.exists():
            logger.warning(f"⚠️ Model file not found: {model_path}")
            return None

        logger.info(f"📂 Loading unified model from {model_dir}")

        _unified_service_instance = UnifiedModelService(
            model_path=str(model_path),
            scaler_path=str(scaler_path),
            features_path=str(features_path)
        )

    return _unified_service_instance if _unified_service_instance and _unified_service_instance.is_loaded else None
