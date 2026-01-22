"""
리스크 관리 시스템
- 포지션 크기 관리
- 손절/익절 자동 설정
- 최대 손실 한도
- 일일/주간 손실 제한
- 레버리지 관리
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """리스크 레벨"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class PositionSide(Enum):
    """포지션 방향"""
    LONG = "long"
    SHORT = "short"


class RiskManager:
    """리스크 관리 클래스"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 리스크 관리 설정
        """
        self.config = config or {}

        # 기본 설정
        self.risk_level = RiskLevel(self.config.get('risk_level', 'moderate'))

        # 포지션 크기 설정 (계좌 잔고의 비율)
        self.max_position_size_pct = self.config.get('max_position_size_pct', {
            RiskLevel.CONSERVATIVE: 0.3,
            RiskLevel.MODERATE: 0.5,
            RiskLevel.AGGRESSIVE: 0.8
        })[self.risk_level]

        # 손절/익절 설정
        self.stop_loss_pct = self.config.get('stop_loss_pct', 2.0)  # 2%
        self.take_profit_pct = self.config.get('take_profit_pct', 4.0)  # 4%

        # 트레일링 스탑
        self.use_trailing_stop = self.config.get('use_trailing_stop', True)
        self.trailing_stop_pct = self.config.get('trailing_stop_pct', 1.0)  # 1%

        # 일일/주간 손실 한도
        self.max_daily_loss_pct = self.config.get('max_daily_loss_pct', 5.0)  # 5%
        self.max_weekly_loss_pct = self.config.get('max_weekly_loss_pct', 10.0)  # 10%

        # 최대 동시 포지션 수
        self.max_concurrent_positions = self.config.get('max_concurrent_positions', 3)

        # 레버리지 설정
        self.max_leverage = self.config.get('max_leverage', {
            RiskLevel.CONSERVATIVE: 2,
            RiskLevel.MODERATE: 5,
            RiskLevel.AGGRESSIVE: 10
        })[self.risk_level]

        # 거래 이력
        self.trade_history: List[Dict[str, Any]] = []
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        self.last_weekly_reset = datetime.now() - timedelta(days=datetime.now().weekday())

        # 현재 포지션
        self.current_positions: List[Dict[str, Any]] = []

        logger.info(f"RiskManager initialized with {self.risk_level.value} risk level")

    def calculate_position_size(
        self,
        account_balance: float,
        current_price: float,
        confidence: float = 0.5,
        volatility: float = 0.02
    ) -> Dict[str, Any]:
        """
        포지션 크기 계산

        Args:
            account_balance: 계좌 잔고
            current_price: 현재 가격
            confidence: 신뢰도 (0-1)
            volatility: 변동성

        Returns:
            포지션 크기 정보
        """
        # 기본 포지션 크기 (잔고의 비율)
        base_position_value = account_balance * self.max_position_size_pct

        # 신뢰도에 따른 조정
        confidence_adjusted = base_position_value * confidence

        # 변동성에 따른 조정 (변동성이 높으면 포지션 축소)
        volatility_factor = max(0.5, min(1.5, 1.0 / (1 + volatility * 10)))
        volatility_adjusted = confidence_adjusted * volatility_factor

        # 최종 포지션 크기
        position_value = min(volatility_adjusted, account_balance * 0.9)  # 최대 90%

        # 수량 계산
        quantity = position_value / current_price

        # 손절/익절 가격 계산
        stop_loss_price = current_price * (1 - self.stop_loss_pct / 100)
        take_profit_price = current_price * (1 + self.take_profit_pct / 100)

        # 리스크/리워드 비율
        risk_amount = position_value * (self.stop_loss_pct / 100)
        reward_amount = position_value * (self.take_profit_pct / 100)
        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0

        result = {
            'position_value': position_value,
            'quantity': quantity,
            'position_size_pct': (position_value / account_balance) * 100,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'risk_amount': risk_amount,
            'reward_amount': reward_amount,
            'risk_reward_ratio': risk_reward_ratio,
            'confidence': confidence,
            'volatility_factor': volatility_factor
        }

        logger.debug(f"Position size calculated: {quantity:.6f} units (${position_value:.2f})")

        return result

    def validate_trade(
        self,
        symbol: str,
        side: PositionSide,
        quantity: float,
        price: float,
        account_balance: float
    ) -> Dict[str, Any]:
        """
        거래 검증

        Args:
            symbol: 심볼
            side: 포지션 방향
            quantity: 수량
            price: 가격
            account_balance: 계좌 잔고

        Returns:
            검증 결과 {'allowed': bool, 'reason': str}
        """
        # 일일/주간 손실 체크
        self._update_pnl_tracking()

        if abs(self.daily_pnl) >= account_balance * (self.max_daily_loss_pct / 100):
            return {
                'allowed': False,
                'reason': f'Daily loss limit reached ({self.max_daily_loss_pct}%)'
            }

        if abs(self.weekly_pnl) >= account_balance * (self.max_weekly_loss_pct / 100):
            return {
                'allowed': False,
                'reason': f'Weekly loss limit reached ({self.max_weekly_loss_pct}%)'
            }

        # 동시 포지션 수 체크
        if len(self.current_positions) >= self.max_concurrent_positions:
            return {
                'allowed': False,
                'reason': f'Maximum concurrent positions reached ({self.max_concurrent_positions})'
            }

        # 포지션 크기 체크
        position_value = quantity * price
        max_allowed_value = account_balance * self.max_position_size_pct

        if position_value > max_allowed_value:
            return {
                'allowed': False,
                'reason': f'Position size exceeds limit (max: ${max_allowed_value:.2f})'
            }

        # 잔고 체크
        if position_value > account_balance * 0.95:  # 5% 여유 자금 확보
            return {
                'allowed': False,
                'reason': 'Insufficient balance (need 5% buffer)'
            }

        return {
            'allowed': True,
            'reason': 'Trade validated successfully'
        }

    def calculate_stop_loss_take_profit(
        self,
        entry_price: float,
        side: PositionSide,
        atr: Optional[float] = None
    ) -> Dict[str, float]:
        """
        손절/익절 가격 계산

        Args:
            entry_price: 진입 가격
            side: 포지션 방향
            atr: Average True Range (변동성 지표)

        Returns:
            {'stop_loss': float, 'take_profit': float}
        """
        # ATR 기반 조정
        if atr:
            # ATR이 크면 더 넓은 손절/익절
            atr_multiplier = max(1.0, min(2.0, atr / entry_price * 100))
            stop_loss_pct = self.stop_loss_pct * atr_multiplier
            take_profit_pct = self.take_profit_pct * atr_multiplier
        else:
            stop_loss_pct = self.stop_loss_pct
            take_profit_pct = self.take_profit_pct

        if side == PositionSide.LONG:
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            take_profit = entry_price * (1 + take_profit_pct / 100)
        else:  # SHORT
            stop_loss = entry_price * (1 + stop_loss_pct / 100)
            take_profit = entry_price * (1 - take_profit_pct / 100)

        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }

    def update_trailing_stop(
        self,
        position: Dict[str, Any],
        current_price: float
    ) -> Optional[float]:
        """
        트레일링 스탑 업데이트

        Args:
            position: 포지션 정보
            current_price: 현재 가격

        Returns:
            새로운 손절 가격 (업데이트된 경우)
        """
        if not self.use_trailing_stop:
            return None

        entry_price = position['entry_price']
        current_stop_loss = position['stop_loss']
        side = PositionSide(position['side'])

        if side == PositionSide.LONG:
            # 롱 포지션: 가격이 상승하면 손절가도 상승
            profit_pct = (current_price - entry_price) / entry_price * 100

            if profit_pct > self.trailing_stop_pct:
                # 현재가에서 trailing_stop_pct 만큼 아래로 손절가 설정
                new_stop_loss = current_price * (1 - self.trailing_stop_pct / 100)

                # 손절가가 상승한 경우에만 업데이트
                if new_stop_loss > current_stop_loss:
                    logger.info(f"Trailing stop updated: {current_stop_loss:.2f} -> {new_stop_loss:.2f}")
                    return new_stop_loss

        else:  # SHORT
            # 숏 포지션: 가격이 하락하면 손절가도 하락
            profit_pct = (entry_price - current_price) / entry_price * 100

            if profit_pct > self.trailing_stop_pct:
                new_stop_loss = current_price * (1 + self.trailing_stop_pct / 100)

                if new_stop_loss < current_stop_loss:
                    logger.info(f"Trailing stop updated: {current_stop_loss:.2f} -> {new_stop_loss:.2f}")
                    return new_stop_loss

        return None

    def check_exit_conditions(
        self,
        position: Dict[str, Any],
        current_price: float
    ) -> Dict[str, Any]:
        """
        청산 조건 확인

        Args:
            position: 포지션 정보
            current_price: 현재 가격

        Returns:
            {'should_exit': bool, 'reason': str}
        """
        side = PositionSide(position['side'])
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']

        if side == PositionSide.LONG:
            # 손절 체크
            if current_price <= stop_loss:
                return {
                    'should_exit': True,
                    'reason': 'stop_loss',
                    'price': current_price
                }

            # 익절 체크
            if current_price >= take_profit:
                return {
                    'should_exit': True,
                    'reason': 'take_profit',
                    'price': current_price
                }

        else:  # SHORT
            if current_price >= stop_loss:
                return {
                    'should_exit': True,
                    'reason': 'stop_loss',
                    'price': current_price
                }

            if current_price <= take_profit:
                return {
                    'should_exit': True,
                    'reason': 'take_profit',
                    'price': current_price
                }

        return {
            'should_exit': False,
            'reason': 'no_exit_condition'
        }

    def add_position(self, position: Dict[str, Any]):
        """포지션 추가"""
        self.current_positions.append(position)
        logger.info(f"Position added: {position['symbol']} {position['side']}")

    def remove_position(self, symbol: str):
        """포지션 제거"""
        self.current_positions = [p for p in self.current_positions if p['symbol'] != symbol]
        logger.info(f"Position removed: {symbol}")

    def record_trade(self, trade: Dict[str, Any]):
        """거래 기록"""
        self.trade_history.append(trade)

        # PnL 업데이트
        pnl = trade.get('pnl', 0)
        self.daily_pnl += pnl
        self.weekly_pnl += pnl

        logger.info(f"Trade recorded: PnL=${pnl:.2f}, Daily PnL=${self.daily_pnl:.2f}")

    def _update_pnl_tracking(self):
        """일일/주간 손익 추적 업데이트"""
        today = datetime.now().date()
        current_week_start = datetime.now() - timedelta(days=datetime.now().weekday())

        # 일일 리셋
        if today > self.last_reset_date:
            logger.info(f"Daily PnL reset: ${self.daily_pnl:.2f}")
            self.daily_pnl = 0.0
            self.last_reset_date = today

        # 주간 리셋
        if current_week_start > self.last_weekly_reset:
            logger.info(f"Weekly PnL reset: ${self.weekly_pnl:.2f}")
            self.weekly_pnl = 0.0
            self.last_weekly_reset = current_week_start

    def get_risk_metrics(self, account_balance: float) -> Dict[str, Any]:
        """
        리스크 지표 조회

        Returns:
            리스크 지표
        """
        total_position_value = sum(p['value'] for p in self.current_positions)
        position_exposure_pct = (total_position_value / account_balance * 100) if account_balance > 0 else 0

        return {
            'risk_level': self.risk_level.value,
            'current_positions': len(self.current_positions),
            'max_concurrent_positions': self.max_concurrent_positions,
            'position_exposure_pct': position_exposure_pct,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': (self.daily_pnl / account_balance * 100) if account_balance > 0 else 0,
            'max_daily_loss_pct': self.max_daily_loss_pct,
            'daily_loss_remaining_pct': self.max_daily_loss_pct - abs(self.daily_pnl / account_balance * 100) if account_balance > 0 else 0,
            'weekly_pnl': self.weekly_pnl,
            'weekly_pnl_pct': (self.weekly_pnl / account_balance * 100) if account_balance > 0 else 0,
            'max_weekly_loss_pct': self.max_weekly_loss_pct,
            'total_trades_today': len([t for t in self.trade_history if t.get('timestamp', datetime.min).date() == datetime.now().date()])
        }
