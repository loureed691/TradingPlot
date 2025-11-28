"""Tests for risk management."""

from unittest.mock import MagicMock

import pytest

from kucoin_bot.config import RiskConfig
from kucoin_bot.risk_management.position_manager import PortfolioState, PositionManager
from kucoin_bot.risk_management.risk_controller import RiskController
from kucoin_bot.strategies.base import Signal, SignalType


class TestPositionManagerCurrency:
    """Tests for PositionManager currency configuration."""

    def test_position_manager_default_currency(self):
        """Test PositionManager uses default USDT currency."""
        mock_client = MagicMock()
        config = RiskConfig()
        manager = PositionManager(mock_client, config)
        assert manager.currency == "USDT"

    def test_position_manager_custom_currency(self):
        """Test PositionManager with custom currency."""
        mock_client = MagicMock()
        config = RiskConfig()
        manager = PositionManager(mock_client, config, currency="XBT")
        assert manager.currency == "XBT"


class TestRiskController:
    """Tests for RiskController."""

    @pytest.fixture
    def config(self):
        """Create risk configuration."""
        return RiskConfig(
            max_leverage=10,
            max_position_size_percent=5.0,
            stop_loss_percent=2.0,
            take_profit_percent=4.0,
            max_open_positions=5,
            max_daily_loss_percent=10.0,
        )

    @pytest.fixture
    def controller(self, config):
        """Create risk controller."""
        return RiskController(config)

    @pytest.fixture
    def portfolio(self):
        """Create test portfolio."""
        return PortfolioState(
            total_balance=10000.0,
            available_balance=8000.0,
            unrealized_pnl=0.0,
            positions=[],
            daily_pnl=0.0,
            trade_count=0,
        )

    @pytest.fixture
    def signal(self):
        """Create test signal."""
        return Signal(
            signal_type=SignalType.LONG,
            symbol="BTCUSDTM",
            confidence=0.75,
            price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
            leverage=5,
            reason="Test signal",
        )

    def test_assess_signal_approved(self, controller, signal, portfolio):
        """Test signal assessment approval."""
        assessment = controller.assess_signal(signal, portfolio)

        assert assessment.approved
        assert assessment.adjusted_signal is not None
        assert assessment.risk_score >= 0

    def test_assess_signal_low_confidence(self, controller, signal, portfolio):
        """Test signal with low confidence."""
        signal.confidence = 0.4
        assessment = controller.assess_signal(signal, portfolio)

        assert assessment.approved
        assert "Low confidence" in str(assessment.warnings)

    def test_consecutive_losses_tracking(self, controller):
        """Test consecutive losses tracking."""
        for _ in range(4):
            controller.on_trade_result(-100.0)

        assert controller._consecutive_losses == 4

        controller.on_trade_result(50.0)
        assert controller._consecutive_losses == 0

    def test_trading_paused_after_max_losses(self, controller, signal, portfolio):
        """Test trading paused after max consecutive losses."""
        for _ in range(5):
            controller.on_trade_result(-100.0)

        assessment = controller.assess_signal(signal, portfolio)
        assert not assessment.approved
        assert "consecutive losses" in assessment.reason.lower()

    def test_leverage_reduction_on_losses(self, controller, signal, portfolio):
        """Test leverage reduction after losses."""
        controller._consecutive_losses = 3

        assessment = controller.assess_signal(signal, portfolio)

        assert assessment.approved
        assert assessment.adjusted_signal.leverage < signal.leverage

    def test_daily_loss_limit(self, controller, signal, portfolio):
        """Test daily loss limit check."""
        portfolio.daily_pnl = -1500.0  # 15% loss

        assessment = controller.assess_signal(signal, portfolio)

        assert not assessment.approved
        assert "daily loss" in assessment.reason.lower()

    def test_drawdown_limit(self, controller, signal, portfolio):
        """Test drawdown limit check."""
        controller._peak_balance = 15000.0  # Higher peak
        portfolio.total_balance = 10000.0  # Current balance (33% drawdown)

        assessment = controller.assess_signal(signal, portfolio)

        assert not assessment.approved
        assert "drawdown" in assessment.reason.lower()

    def test_calculate_max_position_value(self, controller, portfolio):
        """Test max position value calculation."""
        max_value = controller.calculate_max_position_value(portfolio)

        expected = portfolio.total_balance * 5 / 100  # 5% max
        assert max_value == expected

    def test_max_position_reduced_on_losses(self, controller, portfolio):
        """Test max position reduction after losses."""
        controller._consecutive_losses = 3

        max_value = controller.calculate_max_position_value(portfolio)
        base_max = portfolio.total_balance * 5 / 100

        assert max_value < base_max

    def test_should_pause_trading(self, controller, portfolio):
        """Test should_pause_trading function."""
        # Normal conditions
        should_pause, reason = controller.should_pause_trading(portfolio)
        assert not should_pause

        # After max consecutive losses
        controller._consecutive_losses = 5
        should_pause, reason = controller.should_pause_trading(portfolio)
        assert should_pause
        assert "consecutive" in reason.lower()

    def test_reset_state(self, controller):
        """Test state reset."""
        controller._consecutive_losses = 5
        controller._peak_balance = 15000.0

        controller.reset_state()

        assert controller._consecutive_losses == 0
        assert controller._peak_balance == 0.0
