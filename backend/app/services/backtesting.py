"""
백테스팅 시스템
- 과거 데이터로 전략 테스트
- 성능 지표 계산 (수익률, 샤프 비율, MDD 등)
- 거래 내역 추적
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BacktestResult:
    """백테스트 결과"""

    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []

    def add_trade(self, trade: Dict[str, Any]):
        """거래 추가"""
        self.trades.append(trade)

    def add_equity_point(self, timestamp: datetime, equity: float):
        """자산 곡선 포인트 추가"""
        self.timestamps.append(timestamp)
        self.equity_curve.append(equity)

    def get_metrics(self) -> Dict[str, Any]:
        """성능 지표 계산"""
        if not self.trades or not self.equity_curve:
            return {}

        # DataFrame 생성
        df_trades = pd.DataFrame(self.trades)
        equity_series = pd.Series(self.equity_curve, index=self.timestamps)

        # 기본 통계
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in self.trades if t.get('pnl', 0) < 0])

        # 수익률 계산
        initial_capital = self.equity_curve[0]
        final_capital = self.equity_curve[-1]
        total_return = (final_capital - initial_capital) / initial_capital * 100

        # 일별 수익률
        daily_returns = equity_series.pct_change().dropna()

        # 샤프 비율 (연율화, 무위험 이자율 0 가정)
        if len(daily_returns) > 0 and daily_returns.std() != 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)
        else:
            sharpe_ratio = 0

        # Maximum Drawdown (MDD)
        cumulative_max = equity_series.cummax()
        drawdown = (equity_series - cumulative_max) / cumulative_max * 100
        max_drawdown = drawdown.min()

        # 평균 수익/손실
        if 'pnl' in df_trades.columns:
            avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

            # Profit Factor
            total_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum() if winning_trades > 0 else 0
            total_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else 0
            profit_factor = total_profit / total_loss if total_loss != 0 else 0
        else:
            avg_win = avg_loss = win_rate = profit_factor = 0

        return {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'trades': self.trades,
            'equity_curve': {
                'timestamps': [ts.isoformat() for ts in self.timestamps],
                'values': self.equity_curve
            }
        }


class Backtester:
    """백테스팅 엔진"""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.0005  # 0.05%
    ):
        """
        Args:
            initial_capital: 초기 자본
            commission: 수수료율
            slippage: 슬리피지율
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

        # 상태 변수
        self.capital = initial_capital
        self.position = 0  # 보유 수량
        self.position_value = 0  # 포지션 가치
        self.entry_price = 0  # 진입가
        self.trades = []

        logger.info(f"Backtester initialized with capital: ${initial_capital}")

    def run(
        self,
        df: pd.DataFrame,
        strategy_func: Callable[[pd.DataFrame, int], Dict[str, Any]]
    ) -> BacktestResult:
        """
        백테스트 실행

        Args:
            df: OHLCV 데이터프레임
            strategy_func: 전략 함수 (df, index) -> {'signal': str, 'confidence': float}

        Returns:
            BacktestResult
        """
        logger.info(f"Starting backtest with {len(df)} candles")

        result = BacktestResult()
        self._reset()

        # 초기 자산 기록
        result.add_equity_point(df.index[0], self.capital)

        # 각 시점마다 전략 실행
        for i in range(1, len(df)):
            current_data = df.iloc[:i+1]
            current_price = df.iloc[i]['close']
            timestamp = df.index[i]

            # 전략 신호 생성
            signal_data = strategy_func(current_data, i)
            signal = signal_data.get('signal', 'neutral')
            confidence = signal_data.get('confidence', 0.5)

            # 포지션 관리
            if self.position == 0:
                # 진입 신호 확인
                if signal in ['buy', 'strong_buy']:
                    self._enter_long(current_price, timestamp, confidence)

            else:
                # 청산 신호 확인
                if signal in ['sell', 'strong_sell']:
                    trade = self._exit_position(current_price, timestamp)
                    if trade:
                        result.add_trade(trade)

            # 현재 자산 계산
            current_equity = self._calculate_equity(current_price)
            result.add_equity_point(timestamp, current_equity)

        # 마지막 포지션 청산
        if self.position != 0:
            final_price = df.iloc[-1]['close']
            final_timestamp = df.index[-1]
            trade = self._exit_position(final_price, final_timestamp)
            if trade:
                result.add_trade(trade)

        logger.info(f"Backtest completed: {len(result.trades)} trades")

        return result

    def _reset(self):
        """상태 초기화"""
        self.capital = self.initial_capital
        self.position = 0
        self.position_value = 0
        self.entry_price = 0
        self.trades = []

    def _enter_long(self, price: float, timestamp: datetime, confidence: float):
        """
        롱 포지션 진입

        Args:
            price: 진입 가격
            timestamp: 진입 시각
            confidence: 신뢰도 (0-1)
        """
        # 신뢰도에 따른 포지션 크기 결정 (최대 자본의 90%)
        position_size_pct = min(confidence * 0.9, 0.9)
        position_value = self.capital * position_size_pct

        # 실제 진입 가격 (슬리피지 고려)
        actual_entry_price = price * (1 + self.slippage)

        # 수수료 계산
        commission_cost = position_value * self.commission

        # 포지션 수량 계산
        quantity = (position_value - commission_cost) / actual_entry_price

        # 상태 업데이트
        self.position = quantity
        self.entry_price = actual_entry_price
        self.position_value = quantity * actual_entry_price
        self.capital -= (self.position_value + commission_cost)

        logger.debug(f"Enter LONG: price={actual_entry_price:.2f}, qty={quantity:.6f}")

    def _exit_position(self, price: float, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """
        포지션 청산

        Args:
            price: 청산 가격
            timestamp: 청산 시각

        Returns:
            거래 정보
        """
        if self.position == 0:
            return None

        # 실제 청산 가격 (슬리피지 고려)
        actual_exit_price = price * (1 - self.slippage)

        # 청산 금액
        exit_value = self.position * actual_exit_price

        # 수수료
        commission_cost = exit_value * self.commission

        # 순 수익
        net_exit_value = exit_value - commission_cost

        # 손익 계산
        pnl = net_exit_value - self.position_value
        pnl_pct = (pnl / self.position_value) * 100

        # 자본 업데이트
        self.capital += net_exit_value

        # 거래 기록
        trade = {
            'entry_price': self.entry_price,
            'exit_price': actual_exit_price,
            'quantity': self.position,
            'entry_value': self.position_value,
            'exit_value': net_exit_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_timestamp': timestamp,
            'commission': commission_cost * 2  # 진입 + 청산
        }

        logger.debug(f"Exit position: price={actual_exit_price:.2f}, pnl={pnl:.2f} ({pnl_pct:.2f}%)")

        # 포지션 초기화
        self.position = 0
        self.position_value = 0
        self.entry_price = 0

        return trade

    def _calculate_equity(self, current_price: float) -> float:
        """
        현재 총 자산 계산

        Args:
            current_price: 현재 가격

        Returns:
            총 자산
        """
        if self.position == 0:
            return self.capital
        else:
            # 현재 포지션 가치
            current_position_value = self.position * current_price
            return self.capital + current_position_value


class StrategyComparator:
    """전략 비교 도구"""

    @staticmethod
    def compare(
        results: Dict[str, BacktestResult],
        metrics_to_compare: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        여러 전략의 백테스트 결과 비교

        Args:
            results: {전략명: BacktestResult} 딕셔너리
            metrics_to_compare: 비교할 지표 리스트

        Returns:
            비교 결과 DataFrame
        """
        if not metrics_to_compare:
            metrics_to_compare = [
                'total_return_pct',
                'sharpe_ratio',
                'max_drawdown_pct',
                'win_rate_pct',
                'profit_factor',
                'total_trades'
            ]

        comparison_data = {}

        for strategy_name, result in results.items():
            metrics = result.get_metrics()
            comparison_data[strategy_name] = {
                metric: metrics.get(metric, 0)
                for metric in metrics_to_compare
            }

        df_comparison = pd.DataFrame(comparison_data).T

        logger.info(f"Strategy comparison completed for {len(results)} strategies")

        return df_comparison

    @staticmethod
    def rank_strategies(
        results: Dict[str, BacktestResult],
        ranking_metric: str = 'sharpe_ratio'
    ) -> List[tuple]:
        """
        전략 순위 매기기

        Args:
            results: {전략명: BacktestResult} 딕셔너리
            ranking_metric: 순위 기준 지표

        Returns:
            [(전략명, 점수), ...] 리스트 (내림차순)
        """
        rankings = []

        for strategy_name, result in results.items():
            metrics = result.get_metrics()
            score = metrics.get(ranking_metric, 0)
            rankings.append((strategy_name, score))

        # 내림차순 정렬
        rankings.sort(key=lambda x: x[1], reverse=True)

        return rankings
