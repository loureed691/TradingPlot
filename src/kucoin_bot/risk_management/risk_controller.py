"""Risk controller for the trading bot."""

import logging
from dataclasses import dataclass

from kucoin_bot.config import RiskConfig
from kucoin_bot.risk_management.position_manager import PortfolioState
from kucoin_bot.strategies.base import Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class RiskAssessment:
    """Risk assessment result."""

    approved: bool
    adjusted_signal: Signal | None
    risk_score: float  # 0-1, higher is riskier
    warnings: list[str]
    reason: str


class RiskController:
    """Controls and manages trading risk."""

    def __init__(self, config: RiskConfig):
        """Initialize risk controller."""
        self.config = config
        self._consecutive_losses = 0
        self._max_consecutive_losses = 5
        self._drawdown_threshold = 0.15  # 15% drawdown
        self._peak_balance = 0.0

    def assess_signal(
        self, signal: Signal, portfolio: PortfolioState
    ) -> RiskAssessment:
        """Assess risk of a trading signal and potentially adjust it."""
        warnings = []
        risk_score = 0.0

        # Update peak balance
        if portfolio.total_balance > self._peak_balance:
            self._peak_balance = portfolio.total_balance

        # Check drawdown
        if self._peak_balance > 0:
            drawdown = (self._peak_balance - portfolio.total_balance) / self._peak_balance
            if drawdown > self._drawdown_threshold:
                return RiskAssessment(
                    approved=False,
                    adjusted_signal=None,
                    risk_score=1.0,
                    warnings=["Maximum drawdown exceeded"],
                    reason=f"Drawdown of {drawdown:.1%} exceeds threshold",
                )
            risk_score += drawdown / self._drawdown_threshold * 0.3

        # Check consecutive losses
        if self._consecutive_losses >= self._max_consecutive_losses:
            return RiskAssessment(
                approved=False,
                adjusted_signal=None,
                risk_score=1.0,
                warnings=["Too many consecutive losses"],
                reason="Trading paused due to consecutive losses",
            )
        risk_score += self._consecutive_losses / self._max_consecutive_losses * 0.2

        # Check daily loss limit
        daily_loss_limit = (
            portfolio.total_balance * self.config.max_daily_loss_percent / 100
        )
        if portfolio.daily_pnl < -daily_loss_limit:
            return RiskAssessment(
                approved=False,
                adjusted_signal=None,
                risk_score=1.0,
                warnings=["Daily loss limit reached"],
                reason="Maximum daily loss exceeded",
            )

        # Check signal confidence
        if signal.confidence < 0.5:
            warnings.append(f"Low confidence signal: {signal.confidence:.1%}")
            risk_score += 0.2

        # Adjust leverage based on risk
        adjusted_leverage = signal.leverage

        # Reduce leverage if we have unrealized losses
        if portfolio.unrealized_pnl < 0:
            pnl_ratio = abs(portfolio.unrealized_pnl) / portfolio.total_balance
            if pnl_ratio > 0.05:  # More than 5% unrealized loss
                adjusted_leverage = max(1, signal.leverage // 2)
                warnings.append("Leverage reduced due to unrealized losses")

        # Reduce leverage based on consecutive losses
        if self._consecutive_losses > 2:
            adjusted_leverage = max(1, adjusted_leverage - self._consecutive_losses)
            warnings.append("Leverage reduced due to consecutive losses")

        # Cap leverage at maximum
        adjusted_leverage = min(adjusted_leverage, self.config.max_leverage)

        # Adjust stop loss if too tight
        if signal.stop_loss and signal.price:
            stop_distance = abs(signal.price - signal.stop_loss) / signal.price
            min_stop_distance = self.config.stop_loss_percent / 100

            if stop_distance < min_stop_distance:
                if signal.signal_type == SignalType.LONG:
                    adjusted_stop_loss = signal.price * (1 - min_stop_distance)
                else:
                    adjusted_stop_loss = signal.price * (1 + min_stop_distance)
                warnings.append("Stop loss adjusted to minimum distance")
            else:
                adjusted_stop_loss = signal.stop_loss
        else:
            adjusted_stop_loss = signal.stop_loss

        # Create adjusted signal
        adjusted_signal = Signal(
            signal_type=signal.signal_type,
            symbol=signal.symbol,
            confidence=signal.confidence,
            price=signal.price,
            stop_loss=adjusted_stop_loss,
            take_profit=signal.take_profit,
            leverage=adjusted_leverage,
            reason=signal.reason,
            strategy_name=signal.strategy_name,
        )

        return RiskAssessment(
            approved=True,
            adjusted_signal=adjusted_signal,
            risk_score=min(risk_score, 1.0),
            warnings=warnings,
            reason="Signal approved" + (" with adjustments" if warnings else ""),
        )

    def on_trade_result(self, pnl: float) -> None:
        """Update risk state based on trade result."""
        if pnl < 0:
            self._consecutive_losses += 1
            logger.warning(
                f"Trade loss recorded. Consecutive losses: {self._consecutive_losses}"
            )
        else:
            self._consecutive_losses = 0

    def should_pause_trading(self, portfolio: PortfolioState) -> tuple[bool, str]:
        """Determine if trading should be paused."""
        # Check drawdown
        if self._peak_balance > 0:
            drawdown = (self._peak_balance - portfolio.total_balance) / self._peak_balance
            if drawdown > self._drawdown_threshold:
                return True, f"Drawdown {drawdown:.1%} exceeds threshold"

        # Check consecutive losses
        if self._consecutive_losses >= self._max_consecutive_losses:
            return True, f"Too many consecutive losses: {self._consecutive_losses}"

        # Check daily loss
        daily_loss_limit = (
            portfolio.total_balance * self.config.max_daily_loss_percent / 100
        )
        if portfolio.daily_pnl < -daily_loss_limit:
            return True, "Daily loss limit exceeded"

        return False, ""

    def calculate_max_position_value(self, portfolio: PortfolioState) -> float:
        """Calculate maximum position value based on current risk state."""
        base_max = portfolio.total_balance * self.config.max_position_size_percent / 100

        # Reduce based on consecutive losses
        if self._consecutive_losses > 0:
            reduction = 1 - (self._consecutive_losses * 0.1)
            base_max *= max(reduction, 0.5)

        # Reduce based on drawdown
        if self._peak_balance > 0:
            drawdown = (self._peak_balance - portfolio.total_balance) / self._peak_balance
            if drawdown > 0.05:
                base_max *= (1 - drawdown)

        return base_max

    def reset_state(self) -> None:
        """Reset risk controller state."""
        self._consecutive_losses = 0
        self._peak_balance = 0.0
