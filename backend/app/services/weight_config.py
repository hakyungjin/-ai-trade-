"""
가중치 설정 관리 서비스
- 전략 가중치 저장/로드
- 프리셋 관리
- 실시간 가중치 업데이트
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class WeightConfig:
    """가중치 설정 클래스"""

    # 기본 가중치
    DEFAULT_WEIGHTS = {
        'rsi': 0.20,
        'macd': 0.25,
        'bollinger': 0.15,
        'ema_cross': 0.20,
        'stochastic': 0.10,
        'volume': 0.10
    }

    # 프리셋
    PRESETS = {
        'balanced': {
            'name': '균형',
            'description': '모든 지표를 균형있게 사용',
            'weights': {
                'rsi': 0.20,
                'macd': 0.25,
                'bollinger': 0.15,
                'ema_cross': 0.20,
                'stochastic': 0.10,
                'volume': 0.10
            }
        },
        'trend_following': {
            'name': '추세 추종',
            'description': 'MACD와 EMA 크로스에 집중',
            'weights': {
                'rsi': 0.10,
                'macd': 0.35,
                'bollinger': 0.10,
                'ema_cross': 0.35,
                'stochastic': 0.05,
                'volume': 0.05
            }
        },
        'momentum': {
            'name': '모멘텀',
            'description': 'RSI와 Stochastic에 집중',
            'weights': {
                'rsi': 0.35,
                'macd': 0.15,
                'bollinger': 0.10,
                'ema_cross': 0.10,
                'stochastic': 0.25,
                'volume': 0.05
            }
        },
        'volatility': {
            'name': '변동성',
            'description': 'Bollinger Bands와 ATR 중심',
            'weights': {
                'rsi': 0.15,
                'macd': 0.15,
                'bollinger': 0.40,
                'ema_cross': 0.10,
                'stochastic': 0.10,
                'volume': 0.10
            }
        },
        'volume_based': {
            'name': '거래량 기반',
            'description': '거래량과 추세를 중시',
            'weights': {
                'rsi': 0.15,
                'macd': 0.25,
                'bollinger': 0.10,
                'ema_cross': 0.20,
                'stochastic': 0.05,
                'volume': 0.25
            }
        }
    }

    def __init__(self, config_path: str = "./data/weight_config.json"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # 현재 가중치
        self.current_weights = self.DEFAULT_WEIGHTS.copy()

        # 설정 로드
        self.load()

        logger.info("WeightConfig initialized")

    def load(self) -> Dict[str, float]:
        """
        저장된 가중치 로드

        Returns:
            가중치 딕셔너리
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    self.current_weights = data.get('weights', self.DEFAULT_WEIGHTS)
                    logger.info(f"Loaded weights from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load weights: {e}")
                self.current_weights = self.DEFAULT_WEIGHTS.copy()
        else:
            logger.info("No saved weights found, using defaults")
            self.save()  # 기본값 저장

        return self.current_weights

    def save(self) -> bool:
        """
        현재 가중치 저장

        Returns:
            성공 여부
        """
        try:
            data = {
                'weights': self.current_weights,
                'updated_at': datetime.now().isoformat()
            }

            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved weights to {self.config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save weights: {e}")
            return False

    def get_weights(self) -> Dict[str, float]:
        """현재 가중치 조회"""
        return self.current_weights.copy()

    def update_weights(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """
        가중치 업데이트

        Args:
            weights: 새로운 가중치

        Returns:
            업데이트 결과
        """
        # 검증
        validation_result = self.validate_weights(weights)
        if not validation_result['valid']:
            return validation_result

        # 업데이트
        self.current_weights.update(weights)

        # 저장
        if self.save():
            return {
                'valid': True,
                'message': 'Weights updated successfully',
                'weights': self.current_weights
            }
        else:
            return {
                'valid': False,
                'message': 'Failed to save weights',
                'errors': ['Save operation failed']
            }

    def validate_weights(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """
        가중치 검증

        Args:
            weights: 검증할 가중치

        Returns:
            검증 결과
        """
        errors = []

        # 모든 키가 존재하는지 확인
        required_keys = set(self.DEFAULT_WEIGHTS.keys())
        provided_keys = set(weights.keys())

        missing_keys = required_keys - provided_keys
        if missing_keys:
            errors.append(f"Missing keys: {missing_keys}")

        # 값 범위 확인 (0 ~ 1)
        for key, value in weights.items():
            if not isinstance(value, (int, float)):
                errors.append(f"{key}: value must be a number")
            elif value < 0 or value > 1:
                errors.append(f"{key}: value must be between 0 and 1")

        # 합계 확인 (0.95 ~ 1.05 허용)
        total = sum(weights.values())
        if not (0.95 <= total <= 1.05):
            errors.append(f"Sum of weights must be close to 1.0 (got {total:.3f})")

        if errors:
            return {
                'valid': False,
                'message': 'Validation failed',
                'errors': errors
            }

        return {
            'valid': True,
            'message': 'Validation passed'
        }

    def load_preset(self, preset_name: str) -> Dict[str, Any]:
        """
        프리셋 로드

        Args:
            preset_name: 프리셋 이름

        Returns:
            로드 결과
        """
        if preset_name not in self.PRESETS:
            return {
                'valid': False,
                'message': f'Preset "{preset_name}" not found',
                'available_presets': list(self.PRESETS.keys())
            }

        preset = self.PRESETS[preset_name]
        self.current_weights = preset['weights'].copy()

        if self.save():
            return {
                'valid': True,
                'message': f'Loaded preset: {preset["name"]}',
                'preset': preset,
                'weights': self.current_weights
            }
        else:
            return {
                'valid': False,
                'message': 'Failed to save preset'
            }

    def get_presets(self) -> Dict[str, Any]:
        """모든 프리셋 조회"""
        return self.PRESETS.copy()

    def reset_to_default(self) -> Dict[str, Any]:
        """기본 가중치로 리셋"""
        self.current_weights = self.DEFAULT_WEIGHTS.copy()

        if self.save():
            return {
                'valid': True,
                'message': 'Reset to default weights',
                'weights': self.current_weights
            }
        else:
            return {
                'valid': False,
                'message': 'Failed to save default weights'
            }

    def get_weight_info(self) -> Dict[str, Any]:
        """
        가중치 정보 조회

        Returns:
            가중치 정보
        """
        return {
            'current_weights': self.current_weights,
            'default_weights': self.DEFAULT_WEIGHTS,
            'total': sum(self.current_weights.values()),
            'presets': self.PRESETS,
            'updated_at': datetime.now().isoformat()
        }

    def create_custom_preset(
        self,
        name: str,
        description: str,
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        커스텀 프리셋 생성

        Args:
            name: 프리셋 이름
            description: 설명
            weights: 가중치

        Returns:
            생성 결과
        """
        # 검증
        validation_result = self.validate_weights(weights)
        if not validation_result['valid']:
            return validation_result

        # 커스텀 프리셋 저장 경로
        custom_preset_path = self.config_path.parent / "custom_presets.json"

        # 기존 커스텀 프리셋 로드
        custom_presets = {}
        if custom_preset_path.exists():
            try:
                with open(custom_preset_path, 'r') as f:
                    custom_presets = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load custom presets: {e}")

        # 새 프리셋 추가
        preset_key = name.lower().replace(' ', '_')
        custom_presets[preset_key] = {
            'name': name,
            'description': description,
            'weights': weights,
            'created_at': datetime.now().isoformat()
        }

        # 저장
        try:
            with open(custom_preset_path, 'w') as f:
                json.dump(custom_presets, f, indent=2)

            return {
                'valid': True,
                'message': f'Custom preset "{name}" created',
                'preset_key': preset_key,
                'preset': custom_presets[preset_key]
            }

        except Exception as e:
            logger.error(f"Failed to save custom preset: {e}")
            return {
                'valid': False,
                'message': 'Failed to save custom preset',
                'errors': [str(e)]
            }


# 전역 인스턴스
_weight_config: Optional[WeightConfig] = None


def get_weight_config() -> WeightConfig:
    """가중치 설정 싱글톤 인스턴스 조회"""
    global _weight_config
    if _weight_config is None:
        _weight_config = WeightConfig()
    return _weight_config
