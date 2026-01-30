"""
간단한 학습 파이프라인 CLI

사용법 예:
    cd ai-model
    python pipeline.py --symbol BTCUSDT --start_date 2023-01-01 --epochs 50

이 스크립트는 `training.Trainer`를 사용해 데이터 수집, 전처리, 학습, 평가까지 실행합니다.
"""
import argparse
import os
import json

from training.train import Trainer


def main():
    parser = argparse.ArgumentParser(description="AI 학습 파이프라인")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--start_date", default="2023-01-01")
    parser.add_argument("--lookahead", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--model_type", default="lstm", choices=["lstm", "transformer", "mlp"])
    parser.add_argument("--sequence_length", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--save_dir", default="./models")
    parser.add_argument("--save_history", default="./history.json")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    trainer = Trainer(
        model_type=args.model_type,
        sequence_length=args.sequence_length,
        num_classes=3
    )

    # 데이터 준비 (수집 + 지표 추가 + 레이블)
    X, y = trainer.prepare_data(
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start_date,
        threshold=args.threshold,
        lookahead=args.lookahead
    )

    # 데이터로더 생성
    train_loader, test_loader = trainer.create_dataloaders(
        X, y, test_size=args.test_size, batch_size=args.batch_size
    )

    # 학습
    history = trainer.train(
        train_loader,
        test_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        early_stopping_patience=10
    )

    # 모델은 Trainer.save_model에서 이미 저장됨 (../models/best_model.pt)
    # 히스토리/메트릭 저장
    history_path = os.path.abspath(args.save_history)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"Training finished. History saved to {history_path}")


if __name__ == "__main__":
    main()
