"""Scalping strategy for short-term trades."""


from kucoin_bot.strategies.base import BaseStrategy, Signal, SignalType
from kucoin_bot.utils.indicators import TechnicalIndicators


class ScalpingStrategy(BaseStrategy):
    """Scalping strategy using Bollinger Bands and RSI for quick trades."""

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 7,
        profit_target_percent: float = 0.5,
        stop_loss_percent: float = 0.3,
    ):
        """Initialize scalping strategy."""
        super().__init__("Scalping")
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.profit_target_percent = profit_target_percent
        self.stop_loss_percent = stop_loss_percent

    def get_required_history_length(self) -> int:
        """Return required history length."""
        return self.bb_period + 5

    async def analyze(
        self, symbol: str, prices: list[float], volumes: list[float]
    ) -> Signal | None:
        """Analyze for scalping opportunities."""
        if len(prices) < self.get_required_history_length():
            return None

        # Calculate Bollinger Bands
        upper, middle, lower = TechnicalIndicators.bollinger_bands(
            prices, self.bb_period, self.bb_std
        )

        if not upper or not middle or not lower:
            return None

        # Calculate RSI
        rsi = TechnicalIndicators.rsi(prices, self.rsi_period)
        if not rsi:
            return None

        current_price = prices[-1]
        current_rsi = rsi[-1]
        upper_band = upper[-1]
        lower_band = lower[-1]
        middle_band = middle[-1]

        signal_type = SignalType.HOLD
        confidence = 0.0
        reason = ""

        # Calculate volume surge (current vs average)
        avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else volumes[-1]
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # Long signal: Price near lower band + oversold RSI + volume confirmation
        if current_price <= lower_band * 1.01 and current_rsi < 30:
            if volume_ratio > 1.2:  # Volume surge
                signal_type = SignalType.LONG
                confidence = min(0.5 + (30 - current_rsi) / 60 + (volume_ratio - 1) / 5, 0.85)
                reason = f"Price at lower BB, RSI: {current_rsi:.1f}, Volume: {volume_ratio:.1f}x"

        # Short signal: Price near upper band + overbought RSI + volume confirmation
        elif current_price >= upper_band * 0.99 and current_rsi > 70:
            if volume_ratio > 1.2:
                signal_type = SignalType.SHORT
                confidence = min(0.5 + (current_rsi - 70) / 60 + (volume_ratio - 1) / 5, 0.85)
                reason = f"Price at upper BB, RSI: {current_rsi:.1f}, Volume: {volume_ratio:.1f}x"

        # Mean reversion opportunities
        elif current_price < middle_band * 0.98 and current_rsi < 40:
            signal_type = SignalType.LONG
            confidence = 0.55
            reason = f"Mean reversion long, RSI: {current_rsi:.1f}"

        elif current_price > middle_band * 1.02 and current_rsi > 60:
            signal_type = SignalType.SHORT
            confidence = 0.55
            reason = f"Mean reversion short, RSI: {current_rsi:.1f}"

        if signal_type == SignalType.HOLD:
            return None

        # Tight stop loss and take profit for scalping
        if signal_type == SignalType.LONG:
            stop_loss = current_price * (1 - self.stop_loss_percent / 100)
            take_profit = current_price * (1 + self.profit_target_percent / 100)
        else:
            stop_loss = current_price * (1 + self.stop_loss_percent / 100)
            take_profit = current_price * (1 - self.profit_target_percent / 100)

        return Signal(
            signal_type=signal_type,
            symbol=symbol,
            confidence=confidence,
            price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=10,  # Higher leverage for scalping with tight stops
            reason=reason,
        )
