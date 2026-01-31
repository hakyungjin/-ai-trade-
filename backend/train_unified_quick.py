"""
Quick Unified Model Training Script
빠른 통합모델 학습 (소량 데이터로 테스트)
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

async def main():
    from app.database import AsyncSessionLocal
    from sqlalchemy import select, func
    from app.models.market_data import MarketCandle
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("🚀 통합모델 학습 시작...")

    # 1. 데이터 현황 확인
    logger.info("\n📊 Step 1: 데이터 현황 확인...")
    async with AsyncSessionLocal() as db:
        stmt = select(
            MarketCandle.symbol,
            MarketCandle.timeframe,
            func.count(MarketCandle.id).label('count')
        ).group_by(
            MarketCandle.symbol,
            MarketCandle.timeframe
        ).having(
            func.count(MarketCandle.id) > 1000
        ).order_by(
            func.count(MarketCandle.id).desc()
        )

        result = await db.execute(stmt)
        rows = result.all()

        logger.info(f"   충분한 데이터가 있는 심볼: {len(rows)}개")

        # 1h 타임프레임만 선택
        symbols_1h = [row.symbol for row in rows if row.timeframe == '1h'][:10]

        if not symbols_1h:
            logger.error("❌ 1h 타임프레임 데이터가 없습니다!")
            return

        logger.info(f"   학습에 사용할 심볼 (1h): {', '.join(symbols_1h)}")

    # 2. 데이터 준비
    logger.info("\n📦 Step 2: 통합 데이터셋 준비...")

    import subprocess
    data_dir = PROJECT_ROOT / "ai-model" / "data"
    data_dir.mkdir(exist_ok=True, parents=True)

    output_csv = data_dir / "unified_dataset.csv"

    # prepare_unified_dataset.py 실행
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "ai-model" / "scripts" / "prepare_unified_dataset.py"),
        "--symbols", *symbols_1h,
        "--timeframe", "1h",
        "--limit", "5000",
        "--output", str(output_csv)
    ]

    logger.info(f"   실행: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 데이터 준비 실패: {e}")
        logger.error(e.stdout)
        logger.error(e.stderr)
        return

    # 3. 모델 학습
    logger.info("\n🧠 Step 3: 통합 LSTM 모델 학습...")

    models_dir = PROJECT_ROOT / "ai-model" / "models"
    models_dir.mkdir(exist_ok=True, parents=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "ai-model" / "scripts" / "train_unified_model.py"),
        "--data", str(output_csv),
        "--output", str(models_dir / "unified"),
        "--num-assets", str(len(symbols_1h) + 100),  # 여유분
        "--embedding-dim", "16",
        "--hidden-size", "64",
        "--num-layers", "2",
        "--dropout", "0.3",
        "--sequence-length", "60",
        "--batch-size", "32",
        "--epochs", "30",
        "--lr", "0.001",
        "--patience", "5"
    ]

    logger.info(f"   실행: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 모델 학습 실패: {e}")
        logger.error(e.stdout)
        logger.error(e.stderr)
        return

    logger.info("\n✅ 통합모델 학습 완료!")

if __name__ == "__main__":
    asyncio.run(main())
