"""Trend following strategy using moving averages and momentum."""


from kucoin_bot.strategies.base import BaseStrategy, Signal, SignalType
from kucoin_bot.utils.indicators import TechnicalIndicators


class TrendFollowingStrategy(BaseStrategy):
    """Trend following strategy using EMA crossovers and RSI."""

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        rsi_period: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
    ):
        """Initialize trend following strategy."""
        super().__init__("TrendFollowing")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

    def get_required_history_length(self) -> int:
        """Return required history length."""
        return max(self.slow_period, self.rsi_period) + 10

    async def analyze(
        self, symbol: str, prices: list[float], volumes: list[float]
    ) -> Signal | None:
        """Analyze trend and generate signal."""
        if len(prices) < self.get_required_history_length():
            return None

        # Calculate indicators
        fast_ema = TechnicalIndicators.ema(prices, self.fast_period)
        slow_ema = TechnicalIndicators.ema(prices, self.slow_period)
        rsi = TechnicalIndicators.rsi(prices, self.rsi_period)

        if not fast_ema or not slow_ema or not rsi:
            return None

        current_price = prices[-1]
        current_rsi = rsi[-1]

        # Align EMAs for comparison
        fast_last = fast_ema[-1]
        slow_last = slow_ema[-1]

        # Previous values for crossover detection
        if len(fast_ema) > 1 and len(slow_ema) > 1:
            fast_prev = fast_ema[-2]
            slow_prev = slow_ema[-2]
        else:
            return None

        signal_type = SignalType.HOLD
        confidence = 0.0
        reason = ""

        # Bullish crossover (fast crosses above slow) + RSI not overbought
        if fast_prev <= slow_prev and fast_last > slow_last:
            if current_rsi < self.rsi_overbought:
                signal_type = SignalType.LONG
                confidence = min(0.5 + (self.rsi_overbought - current_rsi) / 100, 0.9)
                reason = f"Bullish EMA crossover, RSI: {current_rsi:.1f}"

        # Bearish crossover (fast crosses below slow) + RSI not oversold
        elif fast_prev >= slow_prev and fast_last < slow_last:
            if current_rsi > self.rsi_oversold:
                signal_type = SignalType.SHORT
                confidence = min(0.5 + (current_rsi - self.rsi_oversold) / 100, 0.9)
                reason = f"Bearish EMA crossover, RSI: {current_rsi:.1f}"

        # Strong trend continuation signals
        elif fast_last > slow_last and current_rsi < 40:
            signal_type = SignalType.LONG
            confidence = 0.6
            reason = f"Uptrend with oversold RSI: {current_rsi:.1f}"

        elif fast_last < slow_last and current_rsi > 60:
            signal_type = SignalType.SHORT
            confidence = 0.6
            reason = f"Downtrend with overbought RSI: {current_rsi:.1f}"

        if signal_type == SignalType.HOLD:
            return None

        # Calculate stop loss and take profit
        atr = TechnicalIndicators.atr(
            [p * 1.01 for p in prices],  # Approximate highs
            [p * 0.99 for p in prices],  # Approximate lows
            prices,
            period=14,
        )
        atr_value = atr[-1] if atr else current_price * 0.02

        if signal_type == SignalType.LONG:
            stop_loss = current_price - 2 * atr_value
            take_profit = current_price + 3 * atr_value
        else:
            stop_loss = current_price + 2 * atr_value
            take_profit = current_price - 3 * atr_value

        return Signal(
            signal_type=signal_type,
            symbol=symbol,
            confidence=confidence,
            price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=5,  # Moderate leverage for trend following
            reason=reason,
        )
