"""Tests for technical indicators."""


from kucoin_bot.utils.indicators import TechnicalIndicators


class TestTechnicalIndicators:
    """Tests for TechnicalIndicators class."""

    def test_sma_calculation(self):
        """Test Simple Moving Average calculation."""
        prices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        sma = TechnicalIndicators.sma(prices, 5)

        assert len(sma) == 7  # len(prices) - period + 1
        assert sma[0] == 12.0  # (10+11+12+13+14)/5
        assert sma[-1] == 18.0  # (16+17+18+19+20)/5

    def test_sma_insufficient_data(self):
        """Test SMA with insufficient data."""
        prices = [10, 11, 12]
        sma = TechnicalIndicators.sma(prices, 5)
        assert sma == []

    def test_ema_calculation(self):
        """Test Exponential Moving Average calculation."""
        prices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        ema = TechnicalIndicators.ema(prices, 5)

        assert len(ema) == 7
        assert ema[0] == 12.0  # First EMA equals SMA

    def test_rsi_calculation(self):
        """Test RSI calculation."""
        # Simulated uptrend prices
        prices = list(range(50, 100))
        rsi = TechnicalIndicators.rsi(prices, 14)

        assert len(rsi) > 0
        # In an uptrend, RSI should be high
        assert rsi[-1] > 50

    def test_rsi_downtrend(self):
        """Test RSI in downtrend."""
        # Simulated downtrend prices
        prices = list(range(100, 50, -1))
        rsi = TechnicalIndicators.rsi(prices, 14)

        assert len(rsi) > 0
        # In a downtrend, RSI should be low
        assert rsi[-1] < 50

    def test_macd_calculation(self):
        """Test MACD calculation."""
        prices = [float(i) for i in range(50, 100)]
        macd_line, signal_line, histogram = TechnicalIndicators.macd(prices)

        assert len(macd_line) > 0
        assert len(signal_line) > 0
        assert len(histogram) > 0
        assert len(macd_line) == len(signal_line) == len(histogram)

    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        prices = [float(i) for i in range(50, 100)]
        upper, middle, lower = TechnicalIndicators.bollinger_bands(prices, 20)

        assert len(upper) > 0
        assert len(middle) > 0
        assert len(lower) > 0

        # Upper band should be above middle, middle above lower
        for upper_val, mid_val, lower_val in zip(upper, middle, lower, strict=False):
            assert upper_val >= mid_val >= lower_val

    def test_atr_calculation(self):
        """Test ATR calculation."""
        highs = [float(i + 1) for i in range(50, 100)]
        lows = [float(i - 1) for i in range(50, 100)]
        closes = [float(i) for i in range(50, 100)]

        atr = TechnicalIndicators.atr(highs, lows, closes, 14)

        assert len(atr) > 0
        assert all(a > 0 for a in atr)

    def test_calculate_trend_bullish(self):
        """Test trend calculation for bullish market."""
        # Strong uptrend
        prices = [float(i) for i in range(1, 100)]
        trend = TechnicalIndicators.calculate_trend(prices, 10, 50)

        assert trend == "bullish"

    def test_calculate_trend_bearish(self):
        """Test trend calculation for bearish market."""
        # Strong downtrend
        prices = [float(i) for i in range(100, 1, -1)]
        trend = TechnicalIndicators.calculate_trend(prices, 10, 50)

        assert trend == "bearish"
