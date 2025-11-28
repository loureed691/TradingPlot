"""Arbitrage strategy for cross-pair opportunities."""

from dataclasses import dataclass

from kucoin_bot.strategies.base import BaseStrategy, Signal, SignalType


@dataclass
class SpreadData:
    """Data for spread analysis."""

    symbol1: str
    symbol2: str
    ratio: float
    z_score: float


class ArbitrageStrategy(BaseStrategy):
    """Statistical arbitrage strategy for correlated pairs."""

    def __init__(
        self,
        lookback_period: int = 100,
        z_score_entry: float = 2.0,
        z_score_exit: float = 0.5,
    ):
        """Initialize arbitrage strategy."""
        super().__init__("Arbitrage")
        self.lookback_period = lookback_period
        self.z_score_entry = z_score_entry
        self.z_score_exit = z_score_exit
        self._spread_history: dict[str, list[float]] = {}
        self._ratio_history: dict[str, list[float]] = {}

    def get_required_history_length(self) -> int:
        """Return required history length."""
        return self.lookback_period

    def _calculate_z_score(self, values: list[float]) -> float:
        """Calculate z-score of the latest value."""
        if len(values) < 20:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = variance**0.5

        if std == 0:
            return 0.0

        return (values[-1] - mean) / std

    async def analyze(
        self, symbol: str, prices: list[float], volumes: list[float]
    ) -> Signal | None:
        """Analyze for arbitrage opportunities.

        For this single-symbol version, we look for mean reversion
        in the price ratio compared to a rolling mean.
        """
        if len(prices) < self.get_required_history_length():
            return None

        # Calculate price ratio to its rolling mean (synthetic pair with itself)
        rolling_mean = sum(prices[-self.lookback_period :]) / self.lookback_period
        current_ratio = prices[-1] / rolling_mean

        # Update ratio history
        if symbol not in self._ratio_history:
            self._ratio_history[symbol] = []

        self._ratio_history[symbol].append(current_ratio)

        # Keep only recent history
        if len(self._ratio_history[symbol]) > self.lookback_period:
            self._ratio_history[symbol] = self._ratio_history[symbol][
                -self.lookback_period :
            ]

        # Calculate z-score
        z_score = self._calculate_z_score(self._ratio_history[symbol])

        signal_type = SignalType.HOLD
        confidence = 0.0
        reason = ""
        current_price = prices[-1]

        # Entry signals
        if z_score < -self.z_score_entry:
            signal_type = SignalType.LONG
            confidence = min(abs(z_score) / 4, 0.8)
            reason = f"Statistical arbitrage: z-score={z_score:.2f} (mean reversion long)"

        elif z_score > self.z_score_entry:
            signal_type = SignalType.SHORT
            confidence = min(abs(z_score) / 4, 0.8)
            reason = f"Statistical arbitrage: z-score={z_score:.2f} (mean reversion short)"

        # Exit signals
        elif abs(z_score) < self.z_score_exit:
            signal_type = SignalType.CLOSE
            confidence = 0.7
            reason = f"Mean reversion complete: z-score={z_score:.2f}"

        if signal_type == SignalType.HOLD:
            return None

        # Calculate stops based on statistical levels
        std_price = (
            sum((p - rolling_mean) ** 2 for p in prices[-self.lookback_period :])
            / self.lookback_period
        ) ** 0.5

        if signal_type == SignalType.LONG:
            stop_loss = current_price - 2.5 * std_price
            take_profit = rolling_mean
        elif signal_type == SignalType.SHORT:
            stop_loss = current_price + 2.5 * std_price
            take_profit = rolling_mean
        else:
            stop_loss = None
            take_profit = None

        return Signal(
            signal_type=signal_type,
            symbol=symbol,
            confidence=confidence,
            price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=3,  # Lower leverage for statistical strategies
            reason=reason,
        )
