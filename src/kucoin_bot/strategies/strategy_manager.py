"""Strategy manager for dynamic strategy selection and execution."""

import logging

from kucoin_bot.strategies.ai_predictor import AIPredictor
from kucoin_bot.strategies.arbitrage import ArbitrageStrategy
from kucoin_bot.strategies.base import BaseStrategy, Signal, SignalType
from kucoin_bot.strategies.scalping import ScalpingStrategy
from kucoin_bot.strategies.trend_following import TrendFollowingStrategy

logger = logging.getLogger(__name__)


class StrategyManager:
    """Manages multiple strategies and selects the best one dynamically."""

    def __init__(self):
        """Initialize strategy manager with all available strategies."""
        self.strategies: list[BaseStrategy] = [
            TrendFollowingStrategy(),
            ScalpingStrategy(),
            ArbitrageStrategy(),
            AIPredictor(),
        ]
        self._strategy_performance: dict[str, list[float]] = {}
        self._last_signals: dict[str, Signal] = {}

    def get_strategy(self, name: str) -> BaseStrategy | None:
        """Get a strategy by name."""
        for strategy in self.strategies:
            if strategy.name == name:
                return strategy
        return None

    def enable_strategy(self, name: str, enabled: bool = True) -> bool:
        """Enable or disable a strategy."""
        strategy = self.get_strategy(name)
        if strategy:
            strategy.enabled = enabled
            return True
        return False

    async def get_signals(
        self, symbol: str, prices: list[float], volumes: list[float]
    ) -> list[Signal]:
        """Get signals from all enabled strategies."""
        signals = []

        for strategy in self.strategies:
            if not strategy.enabled:
                continue

            try:
                signal = await strategy.analyze(symbol, prices, volumes)
                if signal and signal.signal_type != SignalType.HOLD:
                    signals.append(signal)
                    self._last_signals[f"{strategy.name}_{symbol}"] = signal
            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed: {e}")

        return signals

    async def get_best_signal(
        self, symbol: str, prices: list[float], volumes: list[float]
    ) -> Signal | None:
        """Get the best signal from all strategies.

        Selection is based on:
        1. Signal confidence
        2. Strategy performance history
        3. Market condition suitability
        """
        signals = await self.get_signals(symbol, prices, volumes)

        if not signals:
            return None

        # Score each signal
        scored_signals: list[tuple[float, Signal, str]] = []

        for signal in signals:
            # Find the strategy that generated this signal
            for strategy in self.strategies:
                if strategy.name in signal.reason or self._last_signals.get(
                    f"{strategy.name}_{symbol}"
                ) == signal:
                    # Combined score: confidence * performance
                    combined_score = signal.confidence * strategy.performance_score
                    scored_signals.append((combined_score, signal, strategy.name))
                    break

        if not scored_signals:
            return None

        # Sort by combined score
        scored_signals.sort(key=lambda x: x[0], reverse=True)

        best_score, best_signal, strategy_name = scored_signals[0]
        logger.info(
            f"Best signal from {strategy_name}: {best_signal.signal_type.value} "
            f"with score {best_score:.2f}"
        )

        return best_signal

    def update_strategy_performance(self, strategy_name: str, profit_loss: float) -> None:
        """Update strategy performance after a trade."""
        strategy = self.get_strategy(strategy_name)
        if strategy:
            strategy.update_performance(profit_loss)

            if strategy_name not in self._strategy_performance:
                self._strategy_performance[strategy_name] = []

            self._strategy_performance[strategy_name].append(profit_loss)

            # Keep only recent history
            if len(self._strategy_performance[strategy_name]) > 100:
                self._strategy_performance[strategy_name] = self._strategy_performance[
                    strategy_name
                ][-100:]

    def get_strategy_stats(self) -> dict[str, dict]:
        """Get statistics for all strategies."""
        stats = {}

        for strategy in self.strategies:
            history = self._strategy_performance.get(strategy.name, [])

            if history:
                total_trades = len(history)
                winning_trades = len([p for p in history if p > 0])
                total_pnl = sum(history)
                avg_pnl = total_pnl / total_trades
                win_rate = winning_trades / total_trades
            else:
                total_trades = 0
                winning_trades = 0
                total_pnl = 0
                avg_pnl = 0
                win_rate = 0

            stats[strategy.name] = {
                "enabled": strategy.enabled,
                "performance_score": strategy.performance_score,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "total_pnl": total_pnl,
                "average_pnl": avg_pnl,
                "win_rate": win_rate,
            }

        return stats

    def auto_adjust_strategies(self) -> None:
        """Automatically enable/disable strategies based on performance."""
        for strategy in self.strategies:
            history = self._strategy_performance.get(strategy.name, [])

            if len(history) < 20:
                continue  # Not enough data

            recent_history = history[-20:]
            recent_pnl = sum(recent_history)
            recent_win_rate = len([p for p in recent_history if p > 0]) / 20

            # Disable strategies with poor recent performance
            if recent_pnl < -10 and recent_win_rate < 0.3:
                if strategy.enabled:
                    logger.warning(
                        f"Disabling {strategy.name} due to poor performance"
                    )
                    strategy.enabled = False

            # Re-enable strategies that might have recovered
            elif recent_pnl > 5 and recent_win_rate > 0.5 and not strategy.enabled:
                logger.info(f"Re-enabling {strategy.name} after recovery")
                strategy.enabled = True
