"""Trading strategies module."""

from kucoin_bot.strategies.ai_predictor import AIPredictor
from kucoin_bot.strategies.arbitrage import ArbitrageStrategy
from kucoin_bot.strategies.base import BaseStrategy, Signal, SignalType
from kucoin_bot.strategies.scalping import ScalpingStrategy
from kucoin_bot.strategies.strategy_manager import StrategyManager
from kucoin_bot.strategies.trend_following import TrendFollowingStrategy

__all__ = [
    "BaseStrategy",
    "Signal",
    "SignalType",
    "TrendFollowingStrategy",
    "ScalpingStrategy",
    "ArbitrageStrategy",
    "AIPredictor",
    "StrategyManager",
]
