"""Tests for adaptive risk settings."""

import pytest

from kucoin_bot.config import RiskConfig
from kucoin_bot.risk_management.adaptive_settings import (
    AdaptiveRiskParameters,
    AdaptiveRiskSettings,
    MarketConditions,
    StrategyPerformance,
)
from kucoin_bot.risk_management.risk_controller import RiskController


class TestAdaptiveRiskSettings:
    """Tests for AdaptiveRiskSettings class."""

    @pytest.fixture
    def settings(self):
        """Create adaptive risk settings instance."""
        return AdaptiveRiskSettings()

    @pytest.fixture
    def low_volatility_conditions(self):
        """Create low volatility market conditions."""
        return MarketConditions(
            volatility=0.02,
            trend_strength=0.3,
            volume_ratio=1.2,
        )

    @pytest.fixture
    def high_volatility_conditions(self):
        """Create high volatility market conditions."""
        return MarketConditions(
            volatility=0.15,
            trend_strength=-0.2,
            volume_ratio=2.0,
        )

    @pytest.fixture
    def good_performance(self):
        """Create good strategy performance."""
        return StrategyPerformance(
            win_rate=0.65,
            avg_profit=50.0,
            avg_loss=25.0,
            sharpe_ratio=1.5,
            total_trades=50,
        )

    @pytest.fixture
    def poor_performance(self):
        """Create poor strategy performance."""
        return StrategyPerformance(
            win_rate=0.35,
            avg_profit=20.0,
            avg_loss=40.0,
            sharpe_ratio=-0.5,
            total_trades=50,
        )

    @pytest.fixture
    def no_performance(self):
        """Create no trading history performance."""
        return StrategyPerformance(
            win_rate=0.5,
            avg_profit=0.0,
            avg_loss=0.0,
            sharpe_ratio=0.0,
            total_trades=0,
        )

    def test_calculate_optimal_leverage_low_volatility_good_performance(
        self, settings, low_volatility_conditions, good_performance
    ):
        """Test leverage calculation with low volatility and good performance."""
        leverage = settings.calculate_optimal_leverage(
            low_volatility_conditions, good_performance
        )
        # Low volatility + good performance = higher leverage
        assert leverage >= 5
        assert leverage <= 20

    def test_calculate_optimal_leverage_high_volatility_poor_performance(
        self, settings, high_volatility_conditions, poor_performance
    ):
        """Test leverage calculation with high volatility and poor performance."""
        leverage = settings.calculate_optimal_leverage(
            high_volatility_conditions, poor_performance
        )
        # High volatility + poor performance = lower leverage
        assert leverage >= 1
        assert leverage <= 5

    def test_calculate_optimal_leverage_no_history(
        self, settings, low_volatility_conditions, no_performance
    ):
        """Test leverage calculation with no trading history."""
        leverage = settings.calculate_optimal_leverage(
            low_volatility_conditions, no_performance
        )
        # Should be conservative with no data
        assert leverage >= 1
        assert leverage <= 10

    def test_calculate_optimal_position_size_good_conditions(
        self, settings, low_volatility_conditions, good_performance
    ):
        """Test position size with good conditions."""
        size = settings.calculate_optimal_position_size(
            low_volatility_conditions, good_performance
        )
        assert size >= 3.0
        assert size <= 10.0

    def test_calculate_optimal_position_size_poor_conditions(
        self, settings, high_volatility_conditions, poor_performance
    ):
        """Test position size with poor conditions."""
        size = settings.calculate_optimal_position_size(
            high_volatility_conditions, poor_performance
        )
        # Should be smaller with high volatility and poor performance
        assert size >= 1.0
        assert size <= 5.0

    def test_calculate_optimal_stop_loss_high_volatility(
        self, settings, high_volatility_conditions, good_performance
    ):
        """Test stop loss with high volatility."""
        stop_loss = settings.calculate_optimal_stop_loss(
            high_volatility_conditions, good_performance
        )
        # Wider stop loss for high volatility
        assert stop_loss >= 2.0
        assert stop_loss <= 5.0

    def test_calculate_optimal_stop_loss_low_volatility(
        self, settings, low_volatility_conditions, good_performance
    ):
        """Test stop loss with low volatility."""
        stop_loss = settings.calculate_optimal_stop_loss(
            low_volatility_conditions, good_performance
        )
        # Tighter stop loss for low volatility
        assert stop_loss >= 0.5
        assert stop_loss <= 3.0

    def test_calculate_optimal_take_profit_minimum_ratio(
        self, settings, low_volatility_conditions, good_performance
    ):
        """Test take profit maintains minimum risk/reward ratio."""
        stop_loss = settings.calculate_optimal_stop_loss(
            low_volatility_conditions, good_performance
        )
        take_profit = settings.calculate_optimal_take_profit(
            low_volatility_conditions, good_performance, stop_loss
        )
        # Take profit should be at least 1.5x stop loss
        assert take_profit >= stop_loss * 1.5

    def test_calculate_adaptive_parameters_returns_all_fields(
        self, settings, low_volatility_conditions, good_performance
    ):
        """Test that calculate_adaptive_parameters returns all fields."""
        params = settings.calculate_adaptive_parameters(
            low_volatility_conditions, good_performance
        )
        assert isinstance(params, AdaptiveRiskParameters)
        assert params.max_leverage >= 1
        assert params.max_position_size_percent >= 1.0
        assert params.stop_loss_percent >= 0.5
        assert params.take_profit_percent >= 1.0

    def test_calculate_adaptive_parameters_with_defaults(self, settings):
        """Test calculate_adaptive_parameters works with no arguments."""
        params = settings.calculate_adaptive_parameters()
        assert isinstance(params, AdaptiveRiskParameters)
        # Should use conservative defaults
        assert params.max_leverage >= 1
        assert params.max_position_size_percent >= 1.0

    def test_record_trade_result(self, settings):
        """Test recording trade results."""
        settings.record_trade_result(100.0)
        settings.record_trade_result(-50.0)
        settings.record_trade_result(75.0)

        performance = settings.get_performance_from_history()
        assert performance.total_trades == 3
        assert performance.win_rate == 2 / 3

    def test_get_current_parameters(self, settings):
        """Test getting current parameters."""
        params = settings.get_current_parameters()
        # Should return default parameters before any calculation
        assert isinstance(params, AdaptiveRiskParameters)

    def test_performance_from_empty_history(self, settings):
        """Test performance calculation with no history."""
        performance = settings.get_performance_from_history()
        assert performance.total_trades == 0
        assert performance.win_rate == 0.5  # Default


class TestRiskControllerAdaptiveMode:
    """Tests for RiskController with adaptive mode."""

    @pytest.fixture
    def adaptive_config(self):
        """Create config with adaptive mode enabled."""
        return RiskConfig(
            max_leverage=10,
            max_position_size_percent=5.0,
            stop_loss_percent=2.0,
            take_profit_percent=4.0,
            adaptive_mode=True,
        )

    @pytest.fixture
    def non_adaptive_config(self):
        """Create config with adaptive mode disabled."""
        return RiskConfig(
            max_leverage=10,
            max_position_size_percent=5.0,
            stop_loss_percent=2.0,
            take_profit_percent=4.0,
            adaptive_mode=False,
        )

    @pytest.fixture
    def market_conditions(self):
        """Create test market conditions."""
        return MarketConditions(
            volatility=0.05,
            trend_strength=0.2,
            volume_ratio=1.0,
        )

    @pytest.fixture
    def strategy_performance(self):
        """Create test strategy performance."""
        return StrategyPerformance(
            win_rate=0.55,
            avg_profit=30.0,
            avg_loss=20.0,
            sharpe_ratio=0.8,
            total_trades=30,
        )

    def test_adaptive_mode_enabled(self, adaptive_config):
        """Test that adaptive mode is correctly detected."""
        controller = RiskController(adaptive_config)
        assert controller.is_adaptive_mode()

    def test_adaptive_mode_disabled(self, non_adaptive_config):
        """Test that non-adaptive mode is correctly detected."""
        controller = RiskController(non_adaptive_config)
        assert not controller.is_adaptive_mode()

    def test_update_adaptive_parameters(
        self, adaptive_config, market_conditions, strategy_performance
    ):
        """Test updating adaptive parameters."""
        controller = RiskController(adaptive_config)

        params = controller.update_adaptive_parameters(
            market_conditions=market_conditions,
            strategy_performance=strategy_performance,
        )

        assert params is not None
        assert params.max_leverage >= 1
        assert params.max_position_size_percent >= 1.0

        # Config should be updated
        assert controller.config.max_leverage == params.max_leverage
        assert controller.config.max_position_size_percent == params.max_position_size_percent

    def test_update_adaptive_parameters_non_adaptive(
        self, non_adaptive_config, market_conditions, strategy_performance
    ):
        """Test that update returns None in non-adaptive mode."""
        controller = RiskController(non_adaptive_config)

        params = controller.update_adaptive_parameters(
            market_conditions=market_conditions,
            strategy_performance=strategy_performance,
        )

        assert params is None

    def test_get_adaptive_parameters(self, adaptive_config):
        """Test getting adaptive parameters."""
        controller = RiskController(adaptive_config)
        params = controller.get_adaptive_parameters()
        assert params is not None

    def test_get_adaptive_parameters_non_adaptive(self, non_adaptive_config):
        """Test getting parameters in non-adaptive mode returns None."""
        controller = RiskController(non_adaptive_config)
        params = controller.get_adaptive_parameters()
        assert params is None

    def test_on_trade_result_records_for_adaptive(self, adaptive_config):
        """Test that trade results are recorded for adaptive settings."""
        controller = RiskController(adaptive_config)

        # Record some trades
        controller.on_trade_result(50.0)
        controller.on_trade_result(-20.0)

        # Should not raise any errors
        assert controller._consecutive_losses == 1

    def test_reset_state_with_adaptive(self, adaptive_config):
        """Test reset state resets adaptive settings."""
        controller = RiskController(adaptive_config)

        # Make some changes
        controller.on_trade_result(-50.0)
        controller._peak_balance = 10000.0

        # Reset
        controller.reset_state()

        assert controller._consecutive_losses == 0
        assert controller._peak_balance == 0.0
        # Adaptive settings should be reset too
        assert controller.is_adaptive_mode()
