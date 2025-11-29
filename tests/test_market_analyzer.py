"""Tests for market analyzer."""

import pytest

from kucoin_bot.api.client import KuCoinFuturesClient, MarketData
from kucoin_bot.config import APIConfig, TradingConfig
from kucoin_bot.utils.market_analyzer import MarketAnalyzer


class TestMarketAnalyzer:
    """Tests for MarketAnalyzer class."""

    @pytest.fixture
    def api_config(self):
        """Create API configuration."""
        return APIConfig(
            api_key="test_key",
            api_secret="test_secret",
            api_passphrase="test_pass",
            sandbox=True,
        )

    @pytest.fixture
    def trading_config(self):
        """Create trading configuration."""
        return TradingConfig(
            min_volume_usd=1000000.0,
            min_volatility=0.02,
            max_volatility=0.15,
        )

    @pytest.fixture
    def client(self, api_config):
        """Create KuCoin Futures client."""
        return KuCoinFuturesClient(api_config)

    @pytest.fixture
    def analyzer(self, client, trading_config):
        """Create market analyzer."""
        return MarketAnalyzer(client, trading_config)

    @pytest.mark.asyncio
    async def test_get_market_data_uses_turnover_for_volume(self, analyzer, mocker):
        """Test that get_market_data uses turnoverOf24h for volume comparison.

        This is critical because the volume needs to be in USD/USDT units to
        properly compare against the min_volume_usd threshold. The API returns:
        - volumeOf24h: volume in base asset (e.g., BTC)
        - turnoverOf24h: volume in quote currency (e.g., USDT)

        We need turnoverOf24h for USD comparison.
        """
        # Mock ticker response with both volume fields
        mock_ticker = {
            "price": "50000.0",
            "volumeOf24h": "100.0",  # 100 BTC - should NOT be used
            "turnoverOf24h": "5000000.0",  # 5M USDT - should be used
            "ts": 1234567890000,
        }

        # Mock klines for volatility calculation
        # Format: [time, open, close, high, low, volume]
        # high >= max(open, close) and low <= min(open, close)
        mock_klines = [
            [1234567890000, 49500, 50500, 51000, 49000, 1000]
            for _ in range(30)
        ]

        # Patch the client methods
        mocker.patch.object(
            analyzer.client, "get_ticker", return_value=mock_ticker
        )
        mocker.patch.object(
            analyzer.client, "get_klines", return_value=mock_klines
        )

        market_data = await analyzer.get_market_data("BTCUSDTM")

        assert market_data is not None
        # Volume should be 5,000,000 (turnoverOf24h), not 100 (volumeOf24h)
        assert market_data.volume_24h == 5000000.0
        assert market_data.price == 50000.0

    @pytest.mark.asyncio
    async def test_get_market_data_handles_missing_turnover(self, analyzer, mocker):
        """Test that missing turnoverOf24h defaults to 0."""
        mock_ticker = {
            "price": "50000.0",
            "volumeOf24h": "100.0",  # Only base volume provided
            "ts": 1234567890000,
        }

        # Format: [time, open, close, high, low, volume]
        mock_klines = [
            [1234567890000, 49500, 50500, 51000, 49000, 1000]
            for _ in range(30)
        ]

        mocker.patch.object(
            analyzer.client, "get_ticker", return_value=mock_ticker
        )
        mocker.patch.object(
            analyzer.client, "get_klines", return_value=mock_klines
        )

        market_data = await analyzer.get_market_data("BTCUSDTM")

        assert market_data is not None
        # Should default to 0, not use volumeOf24h
        assert market_data.volume_24h == 0.0

    @pytest.mark.asyncio
    async def test_select_best_pairs_filters_by_volume_in_usd(
        self, analyzer, mocker
    ):
        """Test that pair selection properly filters by volume in USD."""
        # Mock contracts
        mock_contracts = [
            {"symbol": "BTCUSDTM", "status": "Open"},
            {"symbol": "LOWVOLUSDTM", "status": "Open"},
        ]

        # BTC has high turnover (5M USD), LOWVOL has low turnover (100K USD)
        async def mock_get_ticker(symbol):
            if symbol == "BTCUSDTM":
                return {
                    "price": "50000.0",
                    "turnoverOf24h": "5000000.0",  # 5M USD
                    "ts": 1234567890000,
                }
            else:
                return {
                    "price": "1.0",
                    "turnoverOf24h": "100000.0",  # 100K USD - below threshold
                    "ts": 1234567890000,
                }

        # Format: [time, open, close, high, low, volume]
        mock_klines = [
            [1234567890000, 49500, 50500, 51000, 49000, 1000]
            for _ in range(30)
        ]

        mocker.patch.object(
            analyzer.client, "get_contracts", return_value=mock_contracts
        )
        mocker.patch.object(analyzer.client, "get_ticker", side_effect=mock_get_ticker)
        mocker.patch.object(
            analyzer.client, "get_klines", return_value=mock_klines
        )

        pairs = await analyzer.select_best_pairs(max_pairs=5)

        # Only BTCUSDTM should pass the volume filter (5M > 1M threshold)
        assert len(pairs) == 1
        assert pairs[0].symbol == "BTCUSDTM"

    def test_calculate_pair_score(self, analyzer):
        """Test pair score calculation."""
        market_data = MarketData(
            symbol="BTCUSDTM",
            price=50000.0,
            volume_24h=5000000.0,
            high_24h=52000.0,
            low_24h=48000.0,
            volatility=0.08,  # Within acceptable range
            timestamp=1234567890000,
        )

        score = analyzer.calculate_pair_score(market_data)

        assert score.symbol == "BTCUSDTM"
        assert score.volume_score > 0
        assert score.volatility_score > 0
        assert score.total_score > 0

    @pytest.mark.asyncio
    async def test_get_tradeable_pairs_filters_by_status(self, analyzer, mocker):
        """Test that only Open contracts are returned."""
        mock_contracts = [
            {"symbol": "BTCUSDTM", "status": "Open"},
            {"symbol": "ETHUSDTM", "status": "Open"},
            {"symbol": "OLDUSDTM", "status": "Closed"},
        ]

        mocker.patch.object(
            analyzer.client, "get_contracts", return_value=mock_contracts
        )

        pairs = await analyzer.get_tradeable_pairs()

        assert len(pairs) == 2
        assert "BTCUSDTM" in pairs
        assert "ETHUSDTM" in pairs
        assert "OLDUSDTM" not in pairs
