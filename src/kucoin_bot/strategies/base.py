"""Base strategy class for all trading strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Trading signal type."""

    LONG = "long"
    SHORT = "short"
    CLOSE = "close"
    HOLD = "hold"


@dataclass
class Signal:
    """Trading signal from a strategy."""

    signal_type: SignalType
    symbol: str
    confidence: float  # 0.0 to 1.0
    price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    leverage: int = 1
    reason: str = ""


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, name: str):
        """Initialize strategy."""
        self.name = name
        self._enabled = True
        self._performance_score = 0.5  # Initial score

    @abstractmethod
    async def analyze(
        self, symbol: str, prices: list[float], volumes: list[float]
    ) -> Signal | None:
        """Analyze market data and generate a trading signal.

        Args:
            symbol: Trading pair symbol
            prices: Historical price data
            volumes: Historical volume data

        Returns:
            Signal if conditions are met, None otherwise
        """
        pass

    @abstractmethod
    def get_required_history_length(self) -> int:
        """Return the number of historical data points required."""
        pass

    @property
    def enabled(self) -> bool:
        """Check if strategy is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable strategy."""
        self._enabled = value

    @property
    def performance_score(self) -> float:
        """Get strategy performance score."""
        return self._performance_score

    def update_performance(self, profit_loss: float) -> None:
        """Update strategy performance based on trade outcome."""
        # Simple exponential moving average for performance
        alpha = 0.1
        normalized_pnl = max(min(profit_loss / 100, 1), -1)  # Normalize to -1 to 1
        self._performance_score = alpha * (0.5 + normalized_pnl * 0.5) + (
            1 - alpha
        ) * self._performance_score
