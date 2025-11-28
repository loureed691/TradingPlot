"""Adaptive risk settings that automatically determine optimal parameters."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MarketConditions:
    """Current market conditions for risk assessment."""

    volatility: float  # Current market volatility (0.0 to 1.0)
    trend_strength: float  # Strength of current trend (-1.0 to 1.0)
    volume_ratio: float  # Current volume vs average (1.0 = average)


@dataclass
class StrategyPerformance:
    """Performance metrics from strategies."""

    win_rate: float  # Win rate (0.0 to 1.0)
    avg_profit: float  # Average profit per trade
    avg_loss: float  # Average loss per trade
    sharpe_ratio: float  # Risk-adjusted return
    total_trades: int  # Total number of trades


@dataclass
class AdaptiveRiskParameters:
    """Dynamically calculated risk parameters."""

    max_leverage: int
    max_position_size_percent: float
    stop_loss_percent: float
    take_profit_percent: float


class AdaptiveRiskSettings:
    """Automatically determines optimal risk management settings based on market
    conditions and strategy performance.
    """

    # Default bounds for risk parameters
    MIN_LEVERAGE = 1
    MAX_LEVERAGE_CAP = 20
    MIN_POSITION_SIZE = 1.0
    MAX_POSITION_SIZE = 10.0
    MIN_STOP_LOSS = 0.5
    MAX_STOP_LOSS = 5.0
    MIN_TAKE_PROFIT = 1.0
    MAX_TAKE_PROFIT = 10.0

    # Base values (used when no performance data available)
    BASE_LEVERAGE = 5
    BASE_POSITION_SIZE = 3.0
    BASE_STOP_LOSS = 2.0
    BASE_TAKE_PROFIT = 4.0

    def __init__(self) -> None:
        """Initialize adaptive risk settings."""
        self._trade_results: list[float] = []
        self._recent_volatility: list[float] = []
        self._current_params = AdaptiveRiskParameters(
            max_leverage=self.BASE_LEVERAGE,
            max_position_size_percent=self.BASE_POSITION_SIZE,
            stop_loss_percent=self.BASE_STOP_LOSS,
            take_profit_percent=self.BASE_TAKE_PROFIT,
        )

    def calculate_optimal_leverage(
        self,
        market_conditions: MarketConditions,
        performance: StrategyPerformance,
    ) -> int:
        """Calculate optimal leverage based on conditions and performance.

        Lower leverage when:
        - High volatility
        - Low win rate
        - Recent losses

        Higher leverage when:
        - Low volatility with strong trend
        - High win rate
        - Consistent profits
        """
        base_leverage = self.BASE_LEVERAGE

        # Volatility adjustment: reduce leverage in high volatility
        if market_conditions.volatility > 0.10:
            volatility_factor = max(0.3, 1.0 - market_conditions.volatility * 3)
        elif market_conditions.volatility < 0.03:
            volatility_factor = 1.2  # Slightly higher leverage in low volatility
        else:
            volatility_factor = 1.0

        # Performance adjustment
        if performance.total_trades >= 10:
            if performance.win_rate >= 0.6:
                performance_factor = 1.0 + (performance.win_rate - 0.5) * 0.5
            elif performance.win_rate < 0.4:
                performance_factor = max(0.5, performance.win_rate * 1.5)
            else:
                performance_factor = 1.0
        else:
            performance_factor = 0.8  # Conservative when limited data

        # Trend strength adjustment
        trend_factor = 1.0 + abs(market_conditions.trend_strength) * 0.2

        optimal = base_leverage * volatility_factor * performance_factor * trend_factor

        return max(
            self.MIN_LEVERAGE, min(int(round(optimal)), self.MAX_LEVERAGE_CAP)
        )

    def calculate_optimal_position_size(
        self,
        market_conditions: MarketConditions,
        performance: StrategyPerformance,
    ) -> float:
        """Calculate optimal position size percentage.

        Smaller positions when:
        - High volatility
        - Low win rate
        - Recent consecutive losses

        Larger positions when:
        - Good risk/reward ratio (Sharpe)
        - High win rate
        - Consistent performance
        """
        base_size = self.BASE_POSITION_SIZE

        # Volatility adjustment
        if market_conditions.volatility > 0.08:
            volatility_factor = max(0.4, 1.0 - market_conditions.volatility * 4)
        else:
            volatility_factor = 1.0 + (0.08 - market_conditions.volatility) * 2

        # Performance adjustment
        if performance.total_trades >= 10:
            # Based on Sharpe ratio
            if performance.sharpe_ratio > 1.0:
                sharpe_factor = min(1.5, 1.0 + (performance.sharpe_ratio - 1.0) * 0.25)
            elif performance.sharpe_ratio < 0:
                sharpe_factor = max(0.5, 0.8 + performance.sharpe_ratio * 0.1)
            else:
                sharpe_factor = 1.0
        else:
            sharpe_factor = 0.8  # Conservative when limited data

        optimal = base_size * volatility_factor * sharpe_factor

        return max(
            self.MIN_POSITION_SIZE, min(round(optimal, 1), self.MAX_POSITION_SIZE)
        )

    def calculate_optimal_stop_loss(
        self,
        market_conditions: MarketConditions,
        performance: StrategyPerformance,
    ) -> float:
        """Calculate optimal stop loss percentage.

        Tighter stops when:
        - Low volatility
        - High win rate (less room needed)
        - Good trend alignment

        Wider stops when:
        - High volatility (avoid noise triggers)
        - Lower win rate (give trades more room)
        """
        base_stop = self.BASE_STOP_LOSS

        # Volatility is the primary driver for stop loss
        # Scale stop loss with volatility, but within bounds
        volatility_factor = 1.0 + (market_conditions.volatility - 0.05) * 10
        volatility_factor = max(0.5, min(volatility_factor, 2.0))

        # Win rate adjustment: higher win rate allows tighter stops
        if performance.total_trades >= 10:
            if performance.win_rate >= 0.6:
                win_rate_factor = 0.9  # Tighter stops with high win rate
            elif performance.win_rate < 0.4:
                win_rate_factor = 1.2  # Wider stops with low win rate
            else:
                win_rate_factor = 1.0
        else:
            win_rate_factor = 1.1  # Slightly wider when no data

        optimal = base_stop * volatility_factor * win_rate_factor

        return max(self.MIN_STOP_LOSS, min(round(optimal, 1), self.MAX_STOP_LOSS))

    def calculate_optimal_take_profit(
        self,
        market_conditions: MarketConditions,
        performance: StrategyPerformance,
        stop_loss: float,
    ) -> float:
        """Calculate optimal take profit percentage.

        Based on:
        - Minimum risk/reward ratio of 1.5:1
        - Trend strength (stronger trends = higher targets)
        - Historical average profit
        """
        # Minimum take profit based on stop loss (1.5:1 risk/reward)
        min_target = stop_loss * 1.5

        base_target = self.BASE_TAKE_PROFIT

        # Trend adjustment: stronger trends allow higher targets
        trend_factor = 1.0 + abs(market_conditions.trend_strength) * 0.5

        # Performance adjustment: if avg profit is high, aim higher
        if performance.total_trades >= 10 and performance.avg_profit > 0:
            profit_factor = min(1.5, 1.0 + performance.avg_profit / 100)
        else:
            profit_factor = 1.0

        # Volatility adjustment: higher volatility = higher targets possible
        if market_conditions.volatility > 0.05:
            volatility_factor = 1.0 + market_conditions.volatility * 3
        else:
            volatility_factor = 1.0

        optimal = base_target * trend_factor * profit_factor * volatility_factor

        # Ensure minimum risk/reward ratio is met
        optimal = max(optimal, min_target)

        return max(self.MIN_TAKE_PROFIT, min(round(optimal, 1), self.MAX_TAKE_PROFIT))

    def calculate_adaptive_parameters(
        self,
        market_conditions: MarketConditions | None = None,
        performance: StrategyPerformance | None = None,
    ) -> AdaptiveRiskParameters:
        """Calculate all adaptive risk parameters.

        Args:
            market_conditions: Current market conditions. If None, uses defaults.
            performance: Strategy performance metrics. If None, uses conservative defaults.

        Returns:
            AdaptiveRiskParameters with optimized values.
        """
        # Use defaults if not provided
        if market_conditions is None:
            market_conditions = MarketConditions(
                volatility=0.05,
                trend_strength=0.0,
                volume_ratio=1.0,
            )

        if performance is None:
            performance = StrategyPerformance(
                win_rate=0.5,
                avg_profit=0.0,
                avg_loss=0.0,
                sharpe_ratio=0.0,
                total_trades=0,
            )

        max_leverage = self.calculate_optimal_leverage(market_conditions, performance)
        max_position_size = self.calculate_optimal_position_size(
            market_conditions, performance
        )
        stop_loss = self.calculate_optimal_stop_loss(market_conditions, performance)
        take_profit = self.calculate_optimal_take_profit(
            market_conditions, performance, stop_loss
        )

        self._current_params = AdaptiveRiskParameters(
            max_leverage=max_leverage,
            max_position_size_percent=max_position_size,
            stop_loss_percent=stop_loss,
            take_profit_percent=take_profit,
        )

        logger.info(
            f"Adaptive risk parameters calculated: "
            f"leverage={max_leverage}, position_size={max_position_size}%, "
            f"stop_loss={stop_loss}%, take_profit={take_profit}%"
        )

        return self._current_params

    def record_trade_result(self, pnl: float) -> None:
        """Record a trade result for performance tracking."""
        self._trade_results.append(pnl)
        # Keep only recent results
        if len(self._trade_results) > 100:
            self._trade_results = self._trade_results[-100:]

    def record_volatility(self, volatility: float) -> None:
        """Record current volatility for tracking."""
        self._recent_volatility.append(volatility)
        if len(self._recent_volatility) > 50:
            self._recent_volatility = self._recent_volatility[-50:]

    def get_current_parameters(self) -> AdaptiveRiskParameters:
        """Get the current adaptive parameters."""
        return self._current_params

    def get_performance_from_history(self) -> StrategyPerformance:
        """Calculate performance metrics from recorded trade history."""
        if not self._trade_results:
            return StrategyPerformance(
                win_rate=0.5,
                avg_profit=0.0,
                avg_loss=0.0,
                sharpe_ratio=0.0,
                total_trades=0,
            )

        total = len(self._trade_results)
        wins = [r for r in self._trade_results if r > 0]
        losses = [r for r in self._trade_results if r < 0]

        win_rate = len(wins) / total if total > 0 else 0.5
        avg_profit = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0

        # Calculate Sharpe ratio (simplified)
        if len(self._trade_results) >= 2:
            mean_return = sum(self._trade_results) / len(self._trade_results)
            variance = sum(
                (r - mean_return) ** 2 for r in self._trade_results
            ) / len(self._trade_results)
            std_dev = variance**0.5
            sharpe_ratio = mean_return / std_dev if std_dev > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        return StrategyPerformance(
            win_rate=win_rate,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            sharpe_ratio=sharpe_ratio,
            total_trades=total,
        )
