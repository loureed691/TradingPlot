"""Technical indicators for trading strategies."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators from price data."""

    @staticmethod
    def sma(prices: list[float], period: int) -> list[float]:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return []
        sma_values = []
        for i in range(period - 1, len(prices)):
            avg = sum(prices[i - period + 1 : i + 1]) / period
            sma_values.append(avg)
        return sma_values

    @staticmethod
    def ema(prices: list[float], period: int) -> list[float]:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return []

        multiplier = 2 / (period + 1)
        ema_values = [sum(prices[:period]) / period]  # Start with SMA

        for i in range(period, len(prices)):
            ema_val = (prices[i] - ema_values[-1]) * multiplier + ema_values[-1]
            ema_values.append(ema_val)

        return ema_values

    @staticmethod
    def rsi(prices: list[float], period: int = 14) -> list[float]:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return []

        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        rsi_values = []

        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                rsi_values.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))

        return rsi_values

    @staticmethod
    def macd(
        prices: list[float],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> tuple[list[float], list[float], list[float]]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        if len(prices) < slow_period:
            return [], [], []

        fast_ema = TechnicalIndicators.ema(prices, fast_period)
        slow_ema = TechnicalIndicators.ema(prices, slow_period)

        # Align EMAs
        diff = len(fast_ema) - len(slow_ema)
        fast_ema = fast_ema[diff:]

        macd_line = [f - s for f, s in zip(fast_ema, slow_ema, strict=False)]
        signal_line = TechnicalIndicators.ema(macd_line, signal_period)

        # Align MACD and signal
        diff = len(macd_line) - len(signal_line)
        macd_line_aligned = macd_line[diff:]

        histogram = [m - s for m, s in zip(macd_line_aligned, signal_line, strict=False)]

        return macd_line_aligned, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        prices: list[float], period: int = 20, std_dev: float = 2.0
    ) -> tuple[list[float], list[float], list[float]]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            return [], [], []

        sma_values = TechnicalIndicators.sma(prices, period)
        upper_band = []
        lower_band = []

        for i, sma in enumerate(sma_values):
            start_idx = i
            window = prices[start_idx : start_idx + period]
            std = np.std(window)
            upper_band.append(sma + std_dev * std)
            lower_band.append(sma - std_dev * std)

        return upper_band, sma_values, lower_band

    @staticmethod
    def atr(
        highs: list[float], lows: list[float], closes: list[float], period: int = 14
    ) -> list[float]:
        """Calculate Average True Range."""
        if len(highs) < period + 1:
            return []

        true_ranges = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            true_ranges.append(tr)

        atr_values = [sum(true_ranges[:period]) / period]

        for i in range(period, len(true_ranges)):
            atr_val = (atr_values[-1] * (period - 1) + true_ranges[i]) / period
            atr_values.append(atr_val)

        return atr_values

    @staticmethod
    def calculate_trend(prices: list[float], short_period: int = 10, long_period: int = 50) -> str:
        """Determine market trend based on moving averages."""
        if len(prices) < long_period:
            return "neutral"

        short_ma = TechnicalIndicators.sma(prices, short_period)
        long_ma = TechnicalIndicators.sma(prices, long_period)

        if not short_ma or not long_ma:
            return "neutral"

        # Align the moving averages
        short_ma_last = short_ma[-1]
        long_ma_last = long_ma[-1]

        if short_ma_last > long_ma_last * 1.01:  # 1% threshold
            return "bullish"
        elif short_ma_last < long_ma_last * 0.99:
            return "bearish"
        return "neutral"
