"""Main trading bot implementation."""

import asyncio
import logging
from datetime import datetime, timezone

from kucoin_bot.api.client import KuCoinFuturesClient
from kucoin_bot.config import BotConfig
from kucoin_bot.risk_management.position_manager import PositionManager
from kucoin_bot.risk_management.risk_controller import RiskController
from kucoin_bot.strategies.base import SignalType
from kucoin_bot.strategies.strategy_manager import StrategyManager
from kucoin_bot.utils.market_analyzer import MarketAnalyzer, PairScore

logger = logging.getLogger(__name__)


class KuCoinFuturesBot:
    """KuCoin Futures Trading Bot with advanced strategies and risk management."""

    def __init__(self, config: BotConfig | None = None):
        """Initialize the trading bot."""
        self.config = config or BotConfig.from_env()
        self._setup_logging()

        # Initialize components
        self.client = KuCoinFuturesClient(self.config.api)
        self.market_analyzer = MarketAnalyzer(self.client, self.config.trading)
        self.strategy_manager = StrategyManager()
        self.position_manager = PositionManager(self.client, self.config.risk)
        self.risk_controller = RiskController(self.config.risk)

        # Trading state
        self._running = False
        self._active_pairs: list[PairScore] = []
        self._price_cache: dict[str, list[float]] = {}
        self._volume_cache: dict[str, list[float]] = {}
        self._pending_signals: dict[str, tuple] = {}  # symbol -> (signal, strategy_name)

    def _setup_logging(self) -> None:
        """Configure logging."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    async def start(self) -> None:
        """Start the trading bot."""
        logger.info("Starting KuCoin Futures Trading Bot...")
        self._running = True

        try:
            # Initial setup
            await self._initialize()

            # Main trading loop
            while self._running:
                await self._trading_cycle()
                await asyncio.sleep(self.config.trading.update_interval_seconds)

        except Exception as e:
            logger.error(f"Bot error: {e}")
            raise
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the trading bot."""
        logger.info("Stopping trading bot...")
        self._running = False
        await self.client.close()

    async def _initialize(self) -> None:
        """Initialize bot components."""
        logger.info("Initializing bot...")

        # Get account info
        portfolio = await self.position_manager.get_portfolio_state()
        logger.info(f"Account balance: ${portfolio.total_balance:.2f}")

        # Select initial trading pairs
        await self._update_trading_pairs()

        logger.info(f"Selected {len(self._active_pairs)} trading pairs")

    async def _update_trading_pairs(self) -> None:
        """Update the list of active trading pairs."""
        self._active_pairs = await self.market_analyzer.select_best_pairs(
            max_pairs=self.config.risk.max_open_positions
        )

        for pair in self._active_pairs:
            logger.info(
                f"Active pair: {pair.symbol} (score: {pair.total_score:.2f})"
            )

    async def _update_market_data(self, symbol: str) -> None:
        """Update market data for a symbol."""
        try:
            klines = await self.client.get_klines(symbol, granularity=60)

            if not klines:
                return

            # Extract prices and volumes
            prices = [float(k[2]) for k in klines]  # Close prices
            volumes = [float(k[5]) for k in klines]  # Volumes

            self._price_cache[symbol] = prices
            self._volume_cache[symbol] = volumes

        except Exception as e:
            logger.error(f"Failed to update market data for {symbol}: {e}")

    async def _trading_cycle(self) -> None:
        """Execute one trading cycle."""
        try:
            # Check if trading should be paused
            portfolio = await self.position_manager.get_portfolio_state()
            should_pause, reason = self.risk_controller.should_pause_trading(portfolio)

            if should_pause:
                logger.warning(f"Trading paused: {reason}")
                return

            # Update trading pairs periodically
            if datetime.now(timezone.utc).minute % 15 == 0:
                await self._update_trading_pairs()

            # Monitor positions and check for exits
            await self._check_positions()

            # Analyze each active pair for new opportunities
            for pair_score in self._active_pairs:
                symbol = pair_score.symbol

                # Update market data
                await self._update_market_data(symbol)

                prices = self._price_cache.get(symbol, [])
                volumes = self._volume_cache.get(symbol, [])

                if len(prices) < 50:
                    continue

                # Get best signal from strategies
                signal = await self.strategy_manager.get_best_signal(
                    symbol, prices, volumes
                )

                if signal and signal.signal_type in (SignalType.LONG, SignalType.SHORT):
                    await self._process_signal(signal, portfolio)

            # Auto-adjust strategies based on performance
            self.strategy_manager.auto_adjust_strategies()

        except Exception as e:
            logger.error(f"Trading cycle error: {e}")

    async def _process_signal(self, signal, portfolio) -> None:
        """Process a trading signal."""
        # Assess risk
        assessment = self.risk_controller.assess_signal(signal, portfolio)

        if not assessment.approved:
            logger.info(f"Signal rejected: {assessment.reason}")
            return

        for warning in assessment.warnings:
            logger.warning(f"Risk warning: {warning}")

        # Use adjusted signal
        adjusted_signal = assessment.adjusted_signal

        # Check if we can open position
        can_open, reason = await self.position_manager.can_open_position(
            adjusted_signal, portfolio
        )

        if not can_open:
            logger.info(f"Cannot open position: {reason}")
            return

        # Use strategy name from signal (set by strategy_manager)
        strategy_name = adjusted_signal.strategy_name or signal.strategy_name

        # Open position
        order_id = await self.position_manager.open_position(
            adjusted_signal, strategy_name
        )

        if order_id:
            logger.info(
                f"Opened {adjusted_signal.signal_type.value} position: "
                f"{adjusted_signal.symbol} @ ${adjusted_signal.price:.2f}"
            )
            self._pending_signals[adjusted_signal.symbol] = (
                adjusted_signal,
                strategy_name,
            )

    async def _check_positions(self) -> None:
        """Check existing positions for exit conditions."""
        portfolio = await self.position_manager.get_portfolio_state()

        for position in portfolio.positions:
            symbol = position.symbol

            # Get current price
            market_data = await self.market_analyzer.get_market_data(symbol)
            if market_data is None:
                continue

            current_price = market_data.price

            # Check if we have stop loss/take profit for this position
            if symbol in self._pending_signals:
                signal, strategy_name = self._pending_signals[symbol]

                if signal.stop_loss and signal.take_profit:
                    exit_reason = await self.position_manager.check_stop_loss_take_profit(
                        symbol, current_price, signal.stop_loss, signal.take_profit
                    )

                    if exit_reason:
                        trade = await self.position_manager.close_position(
                            symbol, strategy_name, current_price
                        )

                        if trade:
                            logger.info(
                                f"Closed position ({exit_reason}): "
                                f"{symbol} PnL: ${trade.pnl:.2f}"
                            )

                            # Update risk controller and strategy performance
                            self.risk_controller.on_trade_result(trade.pnl)
                            self.strategy_manager.update_strategy_performance(
                                strategy_name, trade.pnl
                            )

                            del self._pending_signals[symbol]

    async def get_status(self) -> dict:
        """Get current bot status."""
        portfolio = await self.position_manager.get_portfolio_state()
        strategy_stats = self.strategy_manager.get_strategy_stats()
        performance = self.position_manager.get_performance_stats()

        return {
            "running": self._running,
            "portfolio": {
                "total_balance": portfolio.total_balance,
                "available_balance": portfolio.available_balance,
                "unrealized_pnl": portfolio.unrealized_pnl,
                "open_positions": len(portfolio.positions),
                "daily_pnl": portfolio.daily_pnl,
            },
            "active_pairs": [p.symbol for p in self._active_pairs],
            "strategies": strategy_stats,
            "performance": performance,
        }


async def main():
    """Main entry point."""
    bot = KuCoinFuturesBot()

    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
