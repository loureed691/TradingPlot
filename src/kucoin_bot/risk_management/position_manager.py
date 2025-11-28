"""Position management for trading bot."""

import logging
import time
from dataclasses import dataclass, field

from kucoin_bot.api.client import KuCoinFuturesClient, Position
from kucoin_bot.config import RiskConfig
from kucoin_bot.strategies.base import Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a completed trade."""

    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    leverage: int
    pnl: float
    strategy_name: str
    timestamp: int


@dataclass
class PortfolioState:
    """Current portfolio state."""

    total_balance: float
    available_balance: float
    unrealized_pnl: float
    positions: list[Position] = field(default_factory=list)
    daily_pnl: float = 0.0
    trade_count: int = 0


class PositionManager:
    """Manages trading positions and portfolio."""

    def __init__(self, client: KuCoinFuturesClient, config: RiskConfig):
        """Initialize position manager."""
        self.client = client
        self.config = config
        self._trade_history: list[TradeRecord] = []
        self._daily_pnl = 0.0
        self._daily_trades = 0

    async def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state."""
        try:
            account = await self.client.get_account_overview()
            data = account.get("data", {})

            positions = await self.client.get_positions()
            unrealized_pnl = sum(p.unrealized_pnl for p in positions)

            return PortfolioState(
                total_balance=float(data.get("accountEquity", 0)),
                available_balance=float(data.get("availableBalance", 0)),
                unrealized_pnl=unrealized_pnl,
                positions=positions,
                daily_pnl=self._daily_pnl,
                trade_count=self._daily_trades,
            )
        except Exception as e:
            logger.error(f"Failed to get portfolio state: {e}")
            return PortfolioState(
                total_balance=0, available_balance=0, unrealized_pnl=0
            )

    def calculate_position_size(
        self, signal: Signal, portfolio: PortfolioState
    ) -> int:
        """Calculate optimal position size based on risk parameters."""
        if portfolio.available_balance <= 0:
            return 0

        # Maximum position value based on portfolio percentage
        max_position_value = (
            portfolio.total_balance * self.config.max_position_size_percent / 100
        )

        # Adjust for leverage
        effective_leverage = min(signal.leverage, self.config.max_leverage)
        margin_required = max_position_value / effective_leverage

        # Ensure we don't exceed available balance
        margin_required = min(margin_required, portfolio.available_balance * 0.9)

        # Calculate position size (number of contracts)
        # Assuming each contract is worth 1 USD
        position_size = int(margin_required * effective_leverage / signal.price)

        return max(position_size, 1)

    async def can_open_position(
        self, signal: Signal, portfolio: PortfolioState
    ) -> tuple[bool, str]:
        """Check if a new position can be opened."""
        # Check max open positions
        if len(portfolio.positions) >= self.config.max_open_positions:
            return False, "Maximum open positions reached"

        # Check if already in position for this symbol
        for pos in portfolio.positions:
            if pos.symbol == signal.symbol:
                return False, f"Already in position for {signal.symbol}"

        # Check daily loss limit
        daily_loss_limit = (
            portfolio.total_balance * self.config.max_daily_loss_percent / 100
        )
        if self._daily_pnl < -daily_loss_limit:
            return False, "Daily loss limit reached"

        # Check available balance
        position_size = self.calculate_position_size(signal, portfolio)
        if position_size <= 0:
            return False, "Insufficient balance for position"

        return True, "OK"

    async def open_position(
        self, signal: Signal, strategy_name: str
    ) -> str | None:
        """Open a new position based on signal."""
        portfolio = await self.get_portfolio_state()

        can_open, reason = await self.can_open_position(signal, portfolio)
        if not can_open:
            logger.warning(f"Cannot open position: {reason}")
            return None

        position_size = self.calculate_position_size(signal, portfolio)
        effective_leverage = min(signal.leverage, self.config.max_leverage)

        side = "buy" if signal.signal_type == SignalType.LONG else "sell"

        try:
            order = await self.client.place_order(
                symbol=signal.symbol,
                side=side,
                size=position_size,
                leverage=effective_leverage,
                order_type="market",
            )

            logger.info(
                f"Opened {side} position for {signal.symbol}: "
                f"size={position_size}, leverage={effective_leverage}"
            )

            self._daily_trades += 1

            return order.order_id

        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            return None

    async def close_position(
        self, symbol: str, strategy_name: str, exit_price: float
    ) -> TradeRecord | None:
        """Close an existing position."""
        portfolio = await self.get_portfolio_state()

        position = None
        for pos in portfolio.positions:
            if pos.symbol == symbol:
                position = pos
                break

        if position is None:
            logger.warning(f"No position found for {symbol}")
            return None

        try:
            success = await self.client.close_position(symbol)

            if success:
                # Calculate PnL
                if position.side == "long":
                    pnl = (exit_price - position.entry_price) / position.entry_price
                else:
                    pnl = (position.entry_price - exit_price) / position.entry_price

                pnl_amount = pnl * position.size * position.leverage

                trade_record = TradeRecord(
                    symbol=symbol,
                    side=position.side,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    size=position.size,
                    leverage=position.leverage,
                    pnl=pnl_amount,
                    strategy_name=strategy_name,
                    timestamp=int(time.time()),
                )

                self._trade_history.append(trade_record)
                self._daily_pnl += pnl_amount

                logger.info(
                    f"Closed position for {symbol}: PnL={pnl_amount:.2f}"
                )

                return trade_record

        except Exception as e:
            logger.error(f"Failed to close position: {e}")

        return None

    async def check_stop_loss_take_profit(
        self, symbol: str, current_price: float, stop_loss: float, take_profit: float
    ) -> str | None:
        """Check if stop loss or take profit is hit."""
        portfolio = await self.get_portfolio_state()

        for pos in portfolio.positions:
            if pos.symbol != symbol:
                continue

            if pos.side == "long":
                if current_price <= stop_loss:
                    return "stop_loss"
                if current_price >= take_profit:
                    return "take_profit"
            else:  # short
                if current_price >= stop_loss:
                    return "stop_loss"
                if current_price <= take_profit:
                    return "take_profit"

        return None

    def get_trade_history(self, limit: int = 100) -> list[TradeRecord]:
        """Get recent trade history."""
        return self._trade_history[-limit:]

    def get_performance_stats(self) -> dict:
        """Get trading performance statistics."""
        if not self._trade_history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl": 0,
                "win_rate": 0,
                "average_pnl": 0,
            }

        total_trades = len(self._trade_history)
        winning_trades = len([t for t in self._trade_history if t.pnl > 0])
        losing_trades = total_trades - winning_trades
        total_pnl = sum(t.pnl for t in self._trade_history)
        average_pnl = total_pnl / total_trades

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "total_pnl": total_pnl,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "average_pnl": average_pnl,
        }

    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at start of new trading day)."""
        self._daily_pnl = 0.0
        self._daily_trades = 0
