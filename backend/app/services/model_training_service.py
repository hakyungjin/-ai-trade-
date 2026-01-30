"""
AI 모델 학습 서비스
- DB 기반 모델 메타데이터 관리
- 통합 피처 엔지니어링 사용
- 데이터 수집 → 피처 생성 → 학습 → DB 저장 파이프라인
"""

import asyncio
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from sqlalchemy import select, update, and_
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent
AI_MODEL_PATH = PROJECT_ROOT / "ai-model"
MODELS_PATH = AI_MODEL_PATH / "models"


class ModelTrainingService:
    """모델 학습 관리 서비스 (DB 기반)"""

    # 학습 상태 추적 (인메모리)
    _training_status: Dict[str, Dict[str, Any]] = {}

    # ===== DB 기반 모델 조회 =====

    @classmethod
    async def get_model_from_db(
        cls,
        db: AsyncSession,
        symbol: str,
        timeframe: str = "5m",
        model_type: str = "xgboost"
    ) -> Optional[Dict[str, Any]]:
        """DB에서 활성 모델 메타데이터 조회"""
        from app.models.trained_model import TrainedModel

        stmt = (
            select(TrainedModel)
            .where(
                and_(
                    TrainedModel.symbol == symbol.upper(),
                    TrainedModel.timeframe == timeframe,
                    TrainedModel.model_type == model_type,
                    TrainedModel.is_active == True
                )
            )
            .order_by(TrainedModel.version.desc())
            .limit(1)
        )

        result = await db.execute(stmt)
        model = result.scalar_one_or_none()

        if model:
            return model.to_dict()
        return None

    @classmethod
    async def get_best_model_from_db(
        cls,
        db: AsyncSession,
        symbol: str,
        timeframe: str = "5m"
    ) -> Optional[Dict[str, Any]]:
        """DB에서 해당 심볼/타임프레임의 best 모델 조회"""
        from app.models.trained_model import TrainedModel

        stmt = (
            select(TrainedModel)
            .where(
                and_(
                    TrainedModel.symbol == symbol.upper(),
                    TrainedModel.timeframe == timeframe,
                    TrainedModel.is_active == True
                )
            )
            .order_by(TrainedModel.is_best.desc(), TrainedModel.accuracy.desc(), TrainedModel.version.desc())
            .limit(1)
        )

        result = await db.execute(stmt)
        model = result.scalar_one_or_none()

        if model:
            return model.to_dict()
        return None

    @classmethod
    async def save_model_metadata(
        cls,
        db: AsyncSession,
        symbol: str,
        timeframe: str,
        model_type: str,
        version: int,
        model_path: str,
        scaler_path: Optional[str] = None,
        features_path: Optional[str] = None,
        num_classes: int = 3,
        num_features: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
        accuracy: Optional[float] = None,
        f1_score: Optional[float] = None,
        training_samples: Optional[int] = None,
        test_samples: Optional[int] = None,
        threshold: float = 0.02,
        lookahead: int = 5,
        class_metrics: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """모델 메타데이터를 DB에 저장"""
        from app.models.trained_model import TrainedModel

        # 이전 best 플래그 해제
        await db.execute(
            update(TrainedModel)
            .where(
                and_(
                    TrainedModel.symbol == symbol.upper(),
                    TrainedModel.timeframe == timeframe,
                    TrainedModel.model_type == model_type,
                )
            )
            .values(is_best=False)
        )

        model_record = TrainedModel(
            symbol=symbol.upper(),
            timeframe=timeframe,
            model_type=model_type,
            version=version,
            model_path=model_path,
            scaler_path=scaler_path,
            features_path=features_path,
            num_classes=num_classes,
            num_features=num_features,
            feature_names=feature_names,
            threshold=threshold,
            lookahead=lookahead,
            accuracy=accuracy,
            f1_score=f1_score,
            training_samples=training_samples,
            test_samples=test_samples,
            class_metrics=class_metrics,
            is_active=True,
            is_best=True,
            trained_at=datetime.now(),
        )

        db.add(model_record)
        await db.commit()
        await db.refresh(model_record)

        logger.info(f"DB saved: {model_record}")
        return model_record.to_dict()

    @classmethod
    async def list_models_from_db(
        cls,
        db: AsyncSession,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """DB에서 모델 목록 조회"""
        from app.models.trained_model import TrainedModel

        stmt = select(TrainedModel).where(TrainedModel.is_active == True)
        if symbol:
            stmt = stmt.where(TrainedModel.symbol == symbol.upper())
        stmt = stmt.order_by(TrainedModel.symbol, TrainedModel.timeframe, TrainedModel.version.desc())

        result = await db.execute(stmt)
        models = result.scalars().all()

        return [m.to_dict() for m in models]

    @classmethod
    async def update_live_metrics(
        cls,
        db: AsyncSession,
        model_id: int,
        is_correct: bool,
        pnl: float = 0.0
    ):
        """실거래 성과 업데이트"""
        from app.models.trained_model import TrainedModel

        stmt = select(TrainedModel).where(TrainedModel.id == model_id)
        result = await db.execute(stmt)
        model = result.scalar_one_or_none()

        if model:
            model.live_total_predictions = (model.live_total_predictions or 0) + 1
            if is_correct:
                model.live_correct_predictions = (model.live_correct_predictions or 0) + 1
            model.live_total_pnl = (model.live_total_pnl or 0.0) + pnl
            if model.live_total_predictions > 0:
                model.live_accuracy = model.live_correct_predictions / model.live_total_predictions
            model.last_used_at = datetime.now()
            await db.commit()

    # ===== 파일 시스템 기반 모델 조회 (하위 호환) =====

    @classmethod
    def get_model_path(cls, symbol: str, timeframe: str = "5m") -> Optional[Path]:
        """모델 파일 경로 확인 (파일 시스템 기반, 하위 호환)"""
        symbol_lower = symbol.lower()

        logger.info(f"Looking for model: {symbol_lower}, timeframe: {timeframe}")

        # 버전별 검색
        patterns = []
        for version in range(10, 0, -1):
            patterns.append(f"xgboost_{symbol_lower}_{timeframe}_v{version}.joblib")
        patterns.append(f"xgboost_{symbol_lower}_{timeframe}.joblib")
        patterns.append(f"lstm_{symbol_lower}_{timeframe}.pt")

        for pattern in patterns:
            model_file = MODELS_PATH / pattern
            if model_file.exists():
                logger.info(f"Found model: {model_file}")
                return model_file

        # glob 패턴 매칭
        if MODELS_PATH.exists():
            for f in MODELS_PATH.glob(f"*{symbol_lower}*{timeframe}*.joblib"):
                logger.info(f"Found model via glob: {f}")
                return f

        logger.warning(f"No model found for {symbol_lower} {timeframe}")
        return None

    @classmethod
    def check_model_exists(cls, symbol: str, timeframe: str = "5m") -> Dict[str, Any]:
        """모델 존재 여부 확인"""
        symbol_lower = symbol.lower()

        xgb_path = cls.get_model_path(symbol, timeframe)
        lstm_path = MODELS_PATH / f"lstm_{symbol_lower}_{timeframe}.pt"
        lstm_exists = lstm_path.exists()

        if xgb_path and xgb_path.exists():
            stat = xgb_path.stat()
            return {
                "exists": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "model_type": "XGBoost",
                "model_path": str(xgb_path),
                "model_name": xgb_path.name,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "lstm_available": lstm_exists
            }

        if lstm_exists:
            stat = lstm_path.stat()
            return {
                "exists": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "model_type": "LSTM",
                "model_path": str(lstm_path),
                "model_name": lstm_path.name,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
            }

        return {
            "exists": False,
            "symbol": symbol,
            "timeframe": timeframe,
            "models_path": str(MODELS_PATH),
            "message": f"{symbol} 모델이 아직 없습니다. 데이터 수집 후 학습이 필요합니다."
        }

    # ===== 학습 상태 관리 =====

    @classmethod
    def get_training_status(cls, symbol: str) -> Dict[str, Any]:
        """학습 상태 확인"""
        return cls._training_status.get(symbol, {
            "status": "idle",
            "message": "대기 중"
        })

    # ===== 학습 파이프라인 =====

    @classmethod
    async def collect_data_for_training(
        cls,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 10000,
        market_type: str = "spot"
    ) -> Dict[str, Any]:
        """모델 학습용 데이터 수집"""
        logger.info(f"Starting data collection for {symbol} ({timeframe}), limit={limit}")

        cls._training_status[symbol] = {
            "status": "collecting",
            "step": "데이터 수집 중",
            "progress": 0,
            "started_at": datetime.now().isoformat()
        }

        try:
            script_path = AI_MODEL_PATH / "scripts" / "collect_large_dataset.py"

            cmd = [
                "python", str(script_path),
                "--symbol", symbol,
                "--timeframe", timeframe,
                "--limit", str(limit),
                "--market", market_type
            ]

            logger.info(f"Running: {' '.join(cmd)}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(AI_MODEL_PATH)
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                cls._training_status[symbol] = {
                    "status": "collected",
                    "step": "데이터 수집 완료",
                    "progress": 50,
                    "message": stdout.decode() if stdout else "수집 완료"
                }
                return {
                    "success": True,
                    "symbol": symbol,
                    "message": f"데이터 수집 완료 ({limit}개 요청)"
                }
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                cls._training_status[symbol] = {
                    "status": "error",
                    "step": "데이터 수집 실패",
                    "error": error_msg
                }
                return {
                    "success": False,
                    "error": error_msg
                }

        except Exception as e:
            logger.error(f"Data collection error: {e}")
            cls._training_status[symbol] = {
                "status": "error",
                "error": str(e)
            }
            return {"success": False, "error": str(e)}

    @classmethod
    async def prepare_training_data(
        cls,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 10000,
        classes: int = 3,
        threshold: float = 0.005
    ) -> Dict[str, Any]:
        """학습 데이터 준비 (통합 피처 엔지니어링 사용)"""
        logger.info(f"Preparing training data for {symbol}")

        cls._training_status[symbol] = {
            "status": "preparing",
            "step": "학습 데이터 준비 중 (통합 피처 생성)",
            "progress": 60
        }

        try:
            script_path = AI_MODEL_PATH / "scripts" / "prepare_training_data.py"

            cmd = [
                "python", str(script_path),
                "--symbol", symbol,
                "--timeframe", timeframe,
                "--limit", str(limit),
                "--classes", str(classes),
                "--threshold", str(threshold),
                "--use-unified-features",  # 통합 피처 엔지니어링 사용 플래그
            ]

            logger.info(f"Running: {' '.join(cmd)}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(AI_MODEL_PATH)
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                cls._training_status[symbol] = {
                    "status": "prepared",
                    "step": "학습 데이터 준비 완료",
                    "progress": 70
                }
                return {"success": True, "message": "학습 데이터 준비 완료"}
            else:
                return {"success": False, "error": stderr.decode()}

        except Exception as e:
            logger.error(f"Data preparation error: {e}")
            return {"success": False, "error": str(e)}

    @classmethod
    async def train_model(
        cls,
        symbol: str,
        timeframe: str = "5m",
        db: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """XGBoost 모델 학습 + DB 메타데이터 저장"""
        logger.info(f"Training model for {symbol}")

        cls._training_status[symbol] = {
            "status": "training",
            "step": "모델 학습 중",
            "progress": 80
        }

        try:
            script_path = AI_MODEL_PATH / "scripts" / "train_model.py"
            data_path = AI_MODEL_PATH / "data" / f"{symbol}_{timeframe}_training_data.csv"

            if not data_path.exists():
                return {
                    "success": False,
                    "error": f"학습 데이터가 없습니다: {data_path}"
                }

            # 출력 모델 버전 결정
            existing = cls.get_model_path(symbol, timeframe)
            if existing:
                import re
                match = re.search(r'_v(\d+)\.joblib', existing.name)
                new_version = int(match.group(1)) + 1 if match else 1
            else:
                new_version = 1

            output_model = MODELS_PATH / f"xgboost_{symbol.lower()}_{timeframe}_v{new_version}.joblib"

            cmd = [
                "python", str(script_path),
                "--data", str(data_path),
                "--output", str(output_model)
            ]

            logger.info(f"Running: {' '.join(cmd)}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(AI_MODEL_PATH)
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                # DB에 모델 메타데이터 저장
                if db:
                    try:
                        from app.services.unified_feature_engineering import get_all_feature_names
                        feature_names = get_all_feature_names()

                        await cls.save_model_metadata(
                            db=db,
                            symbol=symbol,
                            timeframe=timeframe,
                            model_type="xgboost",
                            version=new_version,
                            model_path=str(output_model),
                            scaler_path=str(output_model).replace('.joblib', '_scaler.joblib'),
                            features_path=str(output_model).replace('.joblib', '_features.joblib'),
                            num_features=len(feature_names),
                            feature_names=feature_names,
                        )
                        logger.info(f"Model metadata saved to DB: v{new_version}")
                    except Exception as db_err:
                        logger.warning(f"Failed to save model metadata to DB: {db_err}")

                cls._training_status[symbol] = {
                    "status": "completed",
                    "step": "학습 완료",
                    "progress": 100,
                    "model_path": str(output_model),
                    "completed_at": datetime.now().isoformat()
                }
                return {
                    "success": True,
                    "model_path": str(output_model),
                    "message": f"모델 학습 완료: v{new_version}"
                }
            else:
                return {"success": False, "error": stderr.decode()}

        except Exception as e:
            logger.error(f"Training error: {e}")
            cls._training_status[symbol] = {
                "status": "error",
                "error": str(e)
            }
            return {"success": False, "error": str(e)}

    @classmethod
    async def auto_train_pipeline(
        cls,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 10000,
        market_type: str = "spot",
        classes: int = 3,
        threshold: float = 0.005,
        db: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """자동 학습 파이프라인 (데이터 수집 -> 준비 -> 학습 -> DB 저장)"""
        logger.info(f"Starting auto-train pipeline for {symbol}")

        # 1. 데이터 수집
        result = await cls.collect_data_for_training(symbol, timeframe, limit, market_type)
        if not result.get("success"):
            return result

        # 2. 학습 데이터 준비
        result = await cls.prepare_training_data(symbol, timeframe, limit, classes, threshold)
        if not result.get("success"):
            return result

        # 3. 모델 학습 + DB 저장
        result = await cls.train_model(symbol, timeframe, db=db)

        return result

    @classmethod
    def list_available_models(cls) -> list:
        """사용 가능한 모델 목록 (파일 시스템)"""
        models = []

        if MODELS_PATH.exists():
            for f in MODELS_PATH.glob("*.joblib"):
                if '_scaler' in f.name or '_features' in f.name:
                    continue
                stat = f.stat()
                models.append({
                    "name": f.name,
                    "path": str(f),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

        return sorted(models, key=lambda x: x["modified_at"], reverse=True)

    @classmethod
    async def collect_historical_data(
        cls,
        symbol: str,
        timeframe: str = "1h",
        days: int = 90
    ) -> Dict[str, Any]:
        """비동기 데이터 수집 (코인 모니터링 워크플로우용)"""
        logger.info(f"[데이터 수집] {symbol} {timeframe} {days}일 데이터 수집 시작")

        try:
            result = await cls.collect_data_for_training(
                symbol=symbol,
                timeframe=timeframe,
                limit=int(days * 24 * 60 / max(int(''.join(filter(str.isdigit, timeframe)) or '60'), 1)),
            )
            logger.info(f"{symbol} {timeframe} 데이터 수집 완료")
            return result
        except Exception as e:
            logger.error(f"{symbol} {timeframe} 데이터 수집 실패: {str(e)}")
            return {"status": "error", "error": str(e)}

    @classmethod
    async def train_xgboost_model(
        cls,
        symbol: str,
        timeframe: str = "1h",
        db: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """비동기 XGBoost 모델 학습"""
        logger.info(f"[XGBoost 학습] {symbol} {timeframe} 모델 학습 시작")

        try:
            result = await cls.train_model(
                symbol=symbol,
                timeframe=timeframe,
                db=db
            )
            logger.info(f"{symbol} {timeframe} XGBoost 학습 완료")
            return result
        except Exception as e:
            logger.error(f"{symbol} {timeframe} XGBoost 학습 실패: {str(e)}")
            return {"status": "error", "error": str(e)}

    @classmethod
    async def train_lstm_model(
        cls,
        symbol: str,
        timeframe: str = "1h",
        db: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """비동기 LSTM 모델 학습"""
        logger.info(f"[LSTM 학습] {symbol} {timeframe} 모델 학습 시작")

        try:
            result = await cls.train_model(
                symbol=symbol,
                timeframe=timeframe,
                db=db
            )
            logger.info(f"{symbol} {timeframe} LSTM 학습 완료")
            return result
        except Exception as e:
            logger.error(f"{symbol} {timeframe} LSTM 학습 실패: {str(e)}")
            return {"status": "error", "error": str(e)}
