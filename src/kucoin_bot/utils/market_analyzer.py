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
            # Use turnover/turnoverOf24h which is the 24h volume in quote currency (USDT)
            # This is needed for proper comparison with min_volume_usd threshold
            # The API may return either "turnover" or "turnoverOf24h" depending on version
            # Use explicit None checks to handle zero values correctly
            turnover = ticker.get("turnover")
            if turnover is not None:
                volume = float(turnover)
            else:
                volume = float(ticker.get("turnoverOf24h", 0))

            # Get 24h high/low for volatility calculation
            klines = await self.client.get_klines(symbol, granularity=60)
            high_24h = price
            low_24h = price

            if klines and len(klines) > 0:
                # Safely extract high and low prices from klines
                # Each kline is expected to be [time, open, close, high, low, volume, ...]
                highs: list[float] = []
                lows: list[float] = []
                for k in klines[-24:]:
                    if isinstance(k, list) and len(k) >= 5:
                        try:
                            highs.append(float(k[3]))
                            lows.append(float(k[4]))
                        except (ValueError, TypeError):
                            continue

                if highs:
                    high_24h = max(highs)
                if lows:
                    low_24h = min(lows)

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
        """Select the best trading pairs based on volume and volatility.

        First attempts to find pairs that meet strict volume and volatility criteria.
        If no pairs pass the strict filters, falls back to selecting the highest
        volume pairs regardless of volatility to ensure trading can proceed.
        """
        pairs = await self.get_tradeable_pairs()
        scored_pairs: list[PairScore] = []
        all_pairs_with_data: list[tuple[MarketData, PairScore]] = []

        # Track filter statistics for debugging
        volume_filtered = 0
        volatility_filtered = 0

        for symbol in pairs:
            market_data = await self.get_market_data(symbol)
            if market_data is None:
                continue

            score = self.calculate_pair_score(market_data)
            all_pairs_with_data.append((market_data, score))

            # Filter by minimum volume
            if market_data.volume_24h < self.config.min_volume_usd:
                volume_filtered += 1
                logger.debug(
                    f"Pair {symbol} filtered: volume ${market_data.volume_24h:,.0f} "
                    f"< ${self.config.min_volume_usd:,.0f}"
                )
                continue

            # Filter by volatility range
            if (
                market_data.volatility < self.config.min_volatility
                or market_data.volatility > self.config.max_volatility
            ):
                volatility_filtered += 1
                logger.debug(
                    f"Pair {symbol} filtered: volatility {market_data.volatility:.4f} "
                    f"outside range [{self.config.min_volatility}, {self.config.max_volatility}]"
                )
                continue

            scored_pairs.append(score)

        # Sort by total score descending
        scored_pairs.sort(key=lambda x: x.total_score, reverse=True)

        # If no pairs pass strict filters, fall back to best available pairs
        if not scored_pairs and all_pairs_with_data:
            logger.warning(
                f"No pairs met strict criteria (volume_filtered={volume_filtered}, "
                f"volatility_filtered={volatility_filtered}). "
                f"Falling back to top {max_pairs} pairs by volume."
            )
            # Sort all pairs by volume and return top ones
            all_pairs_with_data.sort(key=lambda x: x[0].volume_24h, reverse=True)
            scored_pairs = [score for _, score in all_pairs_with_data[:max_pairs]]

        return scored_pairs[:max_pairs]

    def get_cached_data(self, symbol: str) -> MarketData | None:
        """Get cached market data for a symbol."""
        return self._market_cache.get(symbol)
