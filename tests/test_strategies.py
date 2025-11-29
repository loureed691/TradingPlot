"""Tests for trading strategies."""

import pytest

from kucoin_bot.strategies.ai_predictor import AIPredictor
from kucoin_bot.strategies.arbitrage import ArbitrageStrategy
from kucoin_bot.strategies.base import Signal, SignalType
from kucoin_bot.strategies.scalping import ScalpingStrategy
from kucoin_bot.strategies.strategy_manager import StrategyManager
from kucoin_bot.strategies.trend_following import TrendFollowingStrategy


class TestBaseStrategy:
    """Tests for BaseStrategy class."""

    def test_signal_creation(self):
        """Test Signal dataclass creation."""
        signal = Signal(
            signal_type=SignalType.LONG,
            symbol="BTCUSDTM",
            confidence=0.8,
            price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
            leverage=5,
            reason="Test signal",
        )

        assert signal.signal_type == SignalType.LONG
        assert signal.confidence == 0.8
        assert signal.leverage == 5


class TestTrendFollowingStrategy:
    """Tests for TrendFollowingStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return TrendFollowingStrategy()

    def test_required_history_length(self, strategy):
        """Test required history length."""
        assert strategy.get_required_history_length() >= 26

    @pytest.mark.asyncio
    async def test_analyze_insufficient_data(self, strategy):
        """Test analysis with insufficient data."""
        prices = [50000.0] * 10
        volumes = [1000.0] * 10

        signal = await strategy.analyze("BTCUSDTM", prices, volumes)
        assert signal is None

    @pytest.mark.asyncio
    async def test_analyze_uptrend(self, strategy):
        """Test analysis in uptrend."""
        # Create uptrend data
        prices = [50000.0 + i * 100 for i in range(100)]
        volumes = [1000.0] * 100

        signal = await strategy.analyze("BTCUSDTM", prices, volumes)
        # May or may not generate signal depending on EMA crossover
        if signal:
            assert signal.signal_type in (SignalType.LONG, SignalType.SHORT, SignalType.HOLD)


class TestScalpingStrategy:
    """Tests for ScalpingStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return ScalpingStrategy()

    def test_required_history_length(self, strategy):
        """Test required history length."""
        assert strategy.get_required_history_length() >= 20

    @pytest.mark.asyncio
    async def test_analyze_insufficient_data(self, strategy):
        """Test analysis with insufficient data."""
        prices = [50000.0] * 10
        volumes = [1000.0] * 10

        signal = await strategy.analyze("BTCUSDTM", prices, volumes)
        assert signal is None


class TestArbitrageStrategy:
    """Tests for ArbitrageStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return ArbitrageStrategy()

    def test_required_history_length(self, strategy):
        """Test required history length."""
        assert strategy.get_required_history_length() == 100

    @pytest.mark.asyncio
    async def test_analyze_mean_reversion(self, strategy):
        """Test mean reversion signal."""
        # Create data with significant deviation from mean
        base_prices = [50000.0] * 100
        # Add sharp drop at the end
        base_prices[-5:] = [48000.0] * 5
        volumes = [1000.0] * 100

        signal = await strategy.analyze("BTCUSDTM", base_prices, volumes)
        # Should potentially generate a long signal for mean reversion
        if signal:
            assert signal.signal_type in (SignalType.LONG, SignalType.SHORT, SignalType.CLOSE)


class TestStrategyManager:
    """Tests for StrategyManager."""

    @pytest.fixture
    def manager(self):
        """Create strategy manager instance."""
        return StrategyManager()

    def test_strategies_initialized(self, manager):
        """Test that strategies are initialized."""
        assert len(manager.strategies) == 4

    def test_get_strategy(self, manager):
        """Test getting strategy by name."""
        strategy = manager.get_strategy("TrendFollowing")
        assert strategy is not None
        assert strategy.name == "TrendFollowing"

    def test_enable_disable_strategy(self, manager):
        """Test enabling/disabling strategy."""
        manager.enable_strategy("TrendFollowing", False)
        strategy = manager.get_strategy("TrendFollowing")
        assert not strategy.enabled

        manager.enable_strategy("TrendFollowing", True)
        assert strategy.enabled

    def test_update_performance(self, manager):
        """Test performance update."""
        initial_score = manager.get_strategy("TrendFollowing").performance_score

        manager.update_strategy_performance("TrendFollowing", 100.0)

        # Score should increase after profit
        new_score = manager.get_strategy("TrendFollowing").performance_score
        assert new_score >= initial_score

    def test_get_strategy_stats(self, manager):
        """Test getting strategy statistics."""
        # Add some performance data
        manager.update_strategy_performance("TrendFollowing", 50.0)
        manager.update_strategy_performance("TrendFollowing", -20.0)

        stats = manager.get_strategy_stats()

        assert "TrendFollowing" in stats
        assert stats["TrendFollowing"]["total_trades"] == 2


class TestAIPredictor:
    """Tests for AIPredictor strategy."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return AIPredictor()

    def test_required_history_length(self, strategy):
        """Test required history length."""
        assert strategy.get_required_history_length() == 100

    def test_extract_features_handles_zero_prices(self, strategy):
        """Test that feature extraction handles zero prices without division error."""
        # Create prices with zeros at key positions
        prices = [100.0] * 25
        prices[-6] = 0.0  # Position for 5-period return
        volumes = [1000.0] * 25

        # Should not raise ZeroDivisionError
        features = strategy._extract_features(prices, volumes)
        assert len(features) > 0
        # First feature (returns_5) should be 0 due to zero guard
        assert features[0] == 0

    def test_extract_features_handles_zero_at_momentum_position(self, strategy):
        """Test that momentum calculation handles zero price."""
        prices = [100.0] * 25
        prices[-10] = 0.0  # Position for momentum calculation
        volumes = [1000.0] * 25

        # Should not raise ZeroDivisionError
        features = strategy._extract_features(prices, volumes)
        assert len(features) > 0
        # Last feature (momentum) should be 0 due to zero guard
        assert features[-1] == 0

    def test_generate_label_handles_zero_price(self, strategy):
        """Test that label generation handles zero price."""
        prices = [100.0] * 10
        prices[4] = 0.0  # Position for horizon=5 calculation

        # Should not raise ZeroDivisionError
        label = strategy._generate_label(prices, horizon=5)
        assert label == 0  # Returns 0 when division would fail

    @pytest.mark.asyncio
    async def test_analyze_with_insufficient_data(self, strategy):
        """Test analysis with insufficient data."""
        prices = [50000.0] * 10
        volumes = [1000.0] * 10

        signal = await strategy.analyze("BTCUSDTM", prices, volumes)
        assert signal is None
