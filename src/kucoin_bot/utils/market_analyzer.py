"""Market analysis utilities for pair selection and monitoring."""

import logging
from dataclasses import dataclass

from kucoin_bot.api.client import KuCoinFuturesClient, MarketData
from kucoin_bot.config import TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class PairScore:
    """Score for a trading pair based on various metrics."""

    symbol: str
    volume_score: float
    volatility_score: float
    trend_score: float
    total_score: float


class MarketAnalyzer:
    """Analyzes market data to select optimal trading pairs."""

    def __init__(self, client: KuCoinFuturesClient, config: TradingConfig):
        """Initialize market analyzer."""
        self.client = client
        self.config = config
        self._market_cache: dict[str, MarketData] = {}

    async def get_market_data(self, symbol: str) -> MarketData | None:
        """Fetch and calculate market data for a symbol."""
        try:
            ticker = await self.client.get_ticker(symbol)
            if not ticker:
                return None

            price = float(ticker.get("price", 0))
            volume = float(ticker.get("volumeOf24h", 0))

            # Get 24h high/low for volatility calculation
            klines = await self.client.get_klines(symbol, granularity=60)
            if klines and len(klines) > 0:
                highs = [float(k[3]) for k in klines[-24:]]  # High prices
                lows = [float(k[4]) for k in klines[-24:]]  # Low prices
                high_24h = max(highs) if highs else price
                low_24h = min(lows) if lows else price
            else:
                high_24h = price
                low_24h = price

            # Calculate volatility as (high - low) / price
            volatility = (high_24h - low_24h) / price if price > 0 else 0

            market_data = MarketData(
                symbol=symbol,
                price=price,
                volume_24h=volume,
                high_24h=high_24h,
                low_24h=low_24h,
                volatility=volatility,
                timestamp=int(ticker.get("ts", 0)),
            )

            self._market_cache[symbol] = market_data
            return market_data

        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None

    async def get_tradeable_pairs(self) -> list[str]:
        """Get list of all tradeable futures pairs."""
        try:
            contracts = await self.client.get_contracts()
            return [c["symbol"] for c in contracts if c.get("status") == "Open"]
        except Exception as e:
            logger.error(f"Failed to get tradeable pairs: {e}")
            return []

    def calculate_pair_score(self, market_data: MarketData) -> PairScore:
        """Calculate a composite score for a trading pair."""
        # Volume score (normalized, higher is better)
        volume_score = min(market_data.volume_24h / self.config.min_volume_usd, 10.0)

        # Volatility score (prefer medium volatility)
        if market_data.volatility < self.config.min_volatility:
            volatility_score = market_data.volatility / self.config.min_volatility
        elif market_data.volatility > self.config.max_volatility:
            volatility_score = self.config.max_volatility / market_data.volatility
        else:
            # Optimal range
            volatility_score = 1.0 + (
                market_data.volatility - self.config.min_volatility
            ) / (self.config.max_volatility - self.config.min_volatility)

        # Trend score (placeholder - will be enhanced with strategy data)
        trend_score = 1.0

        # Weighted total score
        total_score = volume_score * 0.4 + volatility_score * 0.4 + trend_score * 0.2

        return PairScore(
            symbol=market_data.symbol,
            volume_score=volume_score,
            volatility_score=volatility_score,
            trend_score=trend_score,
            total_score=total_score,
        )

    async def select_best_pairs(self, max_pairs: int = 5) -> list[PairScore]:
        """Select the best trading pairs based on volume and volatility."""
        pairs = await self.get_tradeable_pairs()
        scored_pairs: list[PairScore] = []

        for symbol in pairs:
            market_data = await self.get_market_data(symbol)
            if market_data is None:
                continue

            # Filter by minimum volume
            if market_data.volume_24h < self.config.min_volume_usd:
                continue

            # Filter by volatility range
            if (
                market_data.volatility < self.config.min_volatility
                or market_data.volatility > self.config.max_volatility
            ):
                continue

            score = self.calculate_pair_score(market_data)
            scored_pairs.append(score)

        # Sort by total score descending
        scored_pairs.sort(key=lambda x: x.total_score, reverse=True)

        return scored_pairs[:max_pairs]

    def get_cached_data(self, symbol: str) -> MarketData | None:
        """Get cached market data for a symbol."""
        return self._market_cache.get(symbol)
