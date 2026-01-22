"""
전략 비교 테스트 스크립트

가중치 기반 전략 vs AI 기반 전략 백테스팅 및 성능 비교

사용법:
    python test_strategies.py --symbol BTCUSDT --days 30
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
import argparse
import pandas as pd
from dotenv import load_dotenv

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent))

from app.services.binance_service import BinanceService
from app.services.data_collector import DataCollector
from app.services.weighted_strategy import WeightedStrategy
from app.services.ai_strategy import AIStrategy
from app.services.backtesting import Backtester, BacktestResult, StrategyComparator
from app.services.risk_manager import RiskManager

# 환경 변수 로드
load_dotenv()


async def collect_data(symbol: str, days: int) -> pd.DataFrame:
    """
    데이터 수집

    Args:
        symbol: 심볼 (예: BTCUSDT)
        days: 수집할 일수

    Returns:
        OHLCV 데이터프레임
    """
    print(f"\n{'='*60}")
    print(f"📊 데이터 수집 시작: {symbol} ({days}일)")
    print(f"{'='*60}\n")

    # 바이낸스 서비스 초기화
    api_key = os.getenv('BINANCE_API_KEY', 'test')
    secret_key = os.getenv('BINANCE_SECRET_KEY', 'test')
    testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'

    binance = BinanceService(api_key, secret_key, testnet)
    collector = DataCollector(binance)

    # 데이터베이스 초기화
    await collector.init_database()

    # 데이터 수집
    df = await collector.collect_historical_data(
        symbol=symbol,
        interval='1h',
        days=days,
        save_to_db=True
    )

    print(f"✅ 데이터 수집 완료: {len(df)}개 캔들")
    print(f"   기간: {df.index[0]} ~ {df.index[-1]}")
    print(f"   시작가: ${df.iloc[0]['close']:.2f}")
    print(f"   종가: ${df.iloc[-1]['close']:.2f}")
    print(f"   변동: {((df.iloc[-1]['close'] - df.iloc[0]['close']) / df.iloc[0]['close'] * 100):.2f}%\n")

    return df


def test_weighted_strategy(df: pd.DataFrame, initial_capital: float = 10000) -> BacktestResult:
    """
    가중치 기반 전략 테스트

    Args:
        df: OHLCV 데이터프레임
        initial_capital: 초기 자본

    Returns:
        백테스트 결과
    """
    print(f"\n{'='*60}")
    print("🎯 가중치 기반 전략 백테스팅")
    print(f"{'='*60}\n")

    # 전략 초기화
    strategy = WeightedStrategy()

    # 백테스터 초기화
    backtester = Backtester(initial_capital=initial_capital)

    # 전략 함수 정의
    def strategy_func(data: pd.DataFrame, idx: int) -> dict:
        try:
            result = strategy.analyze(data)
            return {
                'signal': result['signal'],
                'confidence': result['confidence']
            }
        except Exception as e:
            return {'signal': 'neutral', 'confidence': 0.0}

    # 백테스트 실행
    result = backtester.run(df, strategy_func)

    # 결과 출력
    metrics = result.get_metrics()
    print(f"📈 백테스트 결과:")
    print(f"   초기 자본: ${metrics['initial_capital']:,.2f}")
    print(f"   최종 자본: ${metrics['final_capital']:,.2f}")
    print(f"   수익률: {metrics['total_return_pct']:.2f}%")
    print(f"   총 거래: {metrics['total_trades']}회")
    print(f"   승률: {metrics['win_rate_pct']:.2f}%")
    print(f"   샤프 비율: {metrics['sharpe_ratio']:.2f}")
    print(f"   최대 낙폭: {metrics['max_drawdown_pct']:.2f}%")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}\n")

    return result


def test_ai_strategy(df: pd.DataFrame, initial_capital: float = 10000) -> BacktestResult:
    """
    AI 기반 전략 테스트

    Args:
        df: OHLCV 데이터프레임
        initial_capital: 초기 자본

    Returns:
        백테스트 결과
    """
    print(f"\n{'='*60}")
    print("🤖 AI 기반 전략 백테스팅")
    print(f"{'='*60}\n")

    # 전략 초기화 (모델 없이 기본 LSTM 사용)
    strategy = AIStrategy()

    # 백테스터 초기화
    backtester = Backtester(initial_capital=initial_capital)

    # 전략 함수 정의
    def strategy_func(data: pd.DataFrame, idx: int) -> dict:
        try:
            result = strategy.generate_signal(data, combine_with_indicators=True)
            return {
                'signal': result['signal'],
                'confidence': result['confidence']
            }
        except Exception as e:
            return {'signal': 'neutral', 'confidence': 0.0}

    # 백테스트 실행
    result = backtester.run(df, strategy_func)

    # 결과 출력
    metrics = result.get_metrics()
    print(f"📈 백테스트 결과:")
    print(f"   초기 자본: ${metrics['initial_capital']:,.2f}")
    print(f"   최종 자본: ${metrics['final_capital']:,.2f}")
    print(f"   수익률: {metrics['total_return_pct']:.2f}%")
    print(f"   총 거래: {metrics['total_trades']}회")
    print(f"   승률: {metrics['win_rate_pct']:.2f}%")
    print(f"   샤프 비율: {metrics['sharpe_ratio']:.2f}")
    print(f"   최대 낙폭: {metrics['max_drawdown_pct']:.2f}%")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}\n")

    return result


def compare_strategies(results: dict):
    """
    전략 비교

    Args:
        results: {전략명: BacktestResult} 딕셔너리
    """
    print(f"\n{'='*60}")
    print("📊 전략 비교 분석")
    print(f"{'='*60}\n")

    # 비교 DataFrame 생성
    comparison = StrategyComparator.compare(results)

    print("성능 지표 비교:")
    print(comparison.to_string())
    print()

    # 순위 매기기
    rankings = StrategyComparator.rank_strategies(results, 'sharpe_ratio')

    print("\n🏆 전략 순위 (샤프 비율 기준):")
    for i, (strategy_name, score) in enumerate(rankings, 1):
        print(f"   {i}. {strategy_name}: {score:.2f}")

    # Buy & Hold 전략과 비교
    print("\n💡 인사이트:")
    for name, result in results.items():
        metrics = result.get_metrics()
        if metrics['total_return_pct'] > 0:
            print(f"   ✅ {name}: 수익 발생 (+{metrics['total_return_pct']:.2f}%)")
        else:
            print(f"   ❌ {name}: 손실 발생 ({metrics['total_return_pct']:.2f}%)")


async def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='트레이딩 전략 비교 테스트')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='심볼 (예: BTCUSDT)')
    parser.add_argument('--days', type=int, default=30, help='수집할 일수')
    parser.add_argument('--capital', type=float, default=10000, help='초기 자본')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("🤖 암호화폐 트레이딩 전략 비교 시스템")
    print("="*60)

    try:
        # 1. 데이터 수집
        df = await collect_data(args.symbol, args.days)

        if df.empty:
            print("❌ 데이터 수집 실패")
            return

        # 2. 가중치 기반 전략 테스트
        weighted_result = test_weighted_strategy(df, args.capital)

        # 3. AI 기반 전략 테스트
        ai_result = test_ai_strategy(df, args.capital)

        # 4. 전략 비교
        results = {
            'Weighted Strategy': weighted_result,
            'AI Strategy': ai_result
        }

        compare_strategies(results)

        # 5. 리스크 관리 정보
        print(f"\n{'='*60}")
        print("⚠️  리스크 관리 권장사항")
        print(f"{'='*60}\n")

        risk_manager = RiskManager()
        print(f"   리스크 레벨: {risk_manager.risk_level.value}")
        print(f"   최대 포지션 크기: {risk_manager.max_position_size_pct * 100:.0f}%")
        print(f"   손절: {risk_manager.stop_loss_pct}%")
        print(f"   익절: {risk_manager.take_profit_pct}%")
        print(f"   일일 최대 손실: {risk_manager.max_daily_loss_pct}%")
        print(f"   주간 최대 손실: {risk_manager.max_weekly_loss_pct}%")

        print(f"\n{'='*60}")
        print("✅ 테스트 완료!")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
