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
    async def test_get_market_data_uses_turnover24h_for_volume(self, analyzer, mocker):
        """Test that get_market_data uses turnover24h for volume comparison.

        This is critical because the volume needs to be in USD/USDT units to
        properly compare against the min_volume_usd threshold. The ticker API returns:
        - vol24h: volume in base asset (e.g., BTC)
        - turnover24h: volume in quote currency (e.g., USDT)

        We need turnover24h for USD comparison.
        """
        # Mock ticker response with turnover24h field (actual KuCoin Futures API format)
        mock_ticker = {
            "price": "50000.0",
            "vol24h": "100.0",  # 100 BTC - should NOT be used
            "turnover24h": "5000000.0",  # 5M USDT - should be used
            "ts": 1234567890000,
        }

        # Mock klines for volatility calculation
        # Format: [time, open, close, high, low, volume]
        # high >= max(open, close) and low <= min(open, close)
        mock_klines = [
            [1234567890000, 49500, 50500, 51000, 49000, 1000] for _ in range(30)
        ]

        # Patch the client methods
        mocker.patch.object(analyzer.client, "get_ticker", return_value=mock_ticker)
        mocker.patch.object(analyzer.client, "get_klines", return_value=mock_klines)

        market_data = await analyzer.get_market_data("BTCUSDTM")

        assert market_data is not None
        # Volume should be 5,000,000 (turnover24h), not 100 (vol24h)
        assert market_data.volume_24h == 5000000.0
        assert market_data.price == 50000.0

    @pytest.mark.asyncio
    async def test_get_market_data_fallback_to_turnoverOf24h(self, analyzer, mocker):
        """Test that get_market_data falls back to turnoverOf24h when turnover24h is missing.

        Some versions of the contracts API may use turnoverOf24h instead of turnover24h.
        """
        # Mock ticker response with turnoverOf24h field (contracts API format)
        mock_ticker = {
            "price": "50000.0",
            "volumeOf24h": "100.0",  # 100 BTC - should NOT be used
            "turnoverOf24h": "5000000.0",  # 5M USDT - should be used as fallback
            "ts": 1234567890000,
        }

        # Mock klines for volatility calculation
        mock_klines = [
            [1234567890000, 49500, 50500, 51000, 49000, 1000] for _ in range(30)
        ]

        mocker.patch.object(analyzer.client, "get_ticker", return_value=mock_ticker)
        mocker.patch.object(analyzer.client, "get_klines", return_value=mock_klines)

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
            [1234567890000, 49500, 50500, 51000, 49000, 1000] for _ in range(30)
        ]

        mocker.patch.object(analyzer.client, "get_ticker", return_value=mock_ticker)
        mocker.patch.object(analyzer.client, "get_klines", return_value=mock_klines)

        market_data = await analyzer.get_market_data("BTCUSDTM")

        assert market_data is not None
        # Should default to 0, not use volumeOf24h
        assert market_data.volume_24h == 0.0

    @pytest.mark.asyncio
    async def test_get_market_data_uses_turnover_field(self, analyzer, mocker):
        """Test that get_market_data uses 'turnover' field when 'turnoverOf24h' is missing.

        Some versions of the KuCoin API return 'turnover' instead of 'turnoverOf24h'.
        The code should handle both field names.
        """
        # Mock ticker response with 'turnover' field (without Of24h suffix)
        mock_ticker = {
            "price": "50000.0",
            "vol": "100.0",  # 100 BTC - should NOT be used
            "turnover": "5000000.0",  # 5M USDT - should be used
            "ts": 1234567890000,
        }

        # Mock klines for volatility calculation
        mock_klines = [
            [1234567890000, 49500, 50500, 51000, 49000, 1000] for _ in range(30)
        ]

        mocker.patch.object(analyzer.client, "get_ticker", return_value=mock_ticker)
        mocker.patch.object(analyzer.client, "get_klines", return_value=mock_klines)

        market_data = await analyzer.get_market_data("BTCUSDTM")

        assert market_data is not None
        # Volume should be 5,000,000 (turnover), not 100 (vol)
        assert market_data.volume_24h == 5000000.0
        assert market_data.price == 50000.0

    @pytest.mark.asyncio
    async def test_get_market_data_prioritizes_turnover24h(self, analyzer, mocker):
        """Test that 'turnover24h' takes priority over other field names."""
        mock_ticker = {
            "price": "50000.0",
            "turnover24h": "6000000.0",  # Should use this (highest priority)
            "turnover": "3000000.0",  # Should ignore this
            "turnoverOf24h": "5000000.0",  # Should ignore this
            "ts": 1234567890000,
        }
        mock_klines = [
            [1234567890000, 49500, 50500, 51000, 49000, 1000] for _ in range(30)
        ]
        mocker.patch.object(analyzer.client, "get_ticker", return_value=mock_ticker)
        mocker.patch.object(analyzer.client, "get_klines", return_value=mock_klines)

        market_data = await analyzer.get_market_data("BTCUSDTM")

        assert market_data is not None
        assert market_data.volume_24h == 6000000.0  # Should use 'turnover24h'

    @pytest.mark.asyncio
    async def test_get_market_data_handles_zero_turnover(self, analyzer, mocker):
        """Test that zero turnover value is correctly handled."""
        mock_ticker = {
            "price": "50000.0",
            "turnover24h": "0",  # Zero but present - should use this, not fall through
            "ts": 1234567890000,
        }
        mock_klines = [
            [1234567890000, 49500, 50500, 51000, 49000, 1000] for _ in range(30)
        ]
        mocker.patch.object(analyzer.client, "get_ticker", return_value=mock_ticker)
        mocker.patch.object(analyzer.client, "get_klines", return_value=mock_klines)

        market_data = await analyzer.get_market_data("BTCUSDTM")

        assert market_data is not None
        assert market_data.volume_24h == 0.0

    @pytest.mark.asyncio
    async def test_select_best_pairs_filters_by_volume_in_usd(self, analyzer, mocker):
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
                    "turnover24h": "5000000.0",  # 5M USD
                    "ts": 1234567890000,
                }
            else:
                return {
                    "price": "1.0",
                    "turnover24h": "100000.0",  # 100K USD - below threshold
                    "ts": 1234567890000,
                }

        def mock_get_klines(symbol, granularity):
            if symbol == "BTCUSDTM":
                # BTCUSDTM: realistic high volatility, high price
                return [
                    [1234567890000, 49500, 50500, 51000, 49000, 1000] for _ in range(30)
                ]
            else:  # LOWVOLUSDTM
                # LOWVOLUSDTM: low volatility, low price
                return [
                    [1234567890000, 0.95, 1.05, 1.05, 0.95, 1000] for _ in range(30)
                ]

        mocker.patch.object(
            analyzer.client, "get_contracts", return_value=mock_contracts
        )
        mocker.patch.object(analyzer.client, "get_ticker", side_effect=mock_get_ticker)
        mocker.patch.object(analyzer.client, "get_klines", side_effect=mock_get_klines)

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

    @pytest.mark.asyncio
    async def test_select_best_pairs_fallback_when_no_pairs_pass_filters(
        self, analyzer, mocker
    ):
        """Test that fallback returns pairs by volume when none meet strict criteria.

        When all pairs fail volume or volatility filters, the method should
        fall back to returning the highest volume pairs to ensure trading can proceed.
        """
        # Mock contracts
        mock_contracts = [
            {"symbol": "BTCUSDTM", "status": "Open"},
            {"symbol": "ETHUSDTM", "status": "Open"},
        ]

        # Both pairs have low volume (below threshold) but valid volatility
        async def mock_get_ticker(symbol):
            if symbol == "BTCUSDTM":
                return {
                    "price": "50000.0",
                    "turnover24h": "500000.0",  # 500K USD - below 1M threshold
                    "ts": 1234567890000,
                }
            else:
                return {
                    "price": "3000.0",
                    "turnover24h": "200000.0",  # 200K USD - below 1M threshold
                    "ts": 1234567890000,
                }

        def mock_get_klines(symbol, granularity):
            if symbol == "BTCUSDTM":
                # BTCUSDTM: price=50000, volatility ~4%
                return [
                    [1234567890000, 49500, 50500, 51000, 49000, 1000] for _ in range(30)
                ]
            elif symbol == "ETHUSDTM":
                # ETHUSDTM: price=3000, volatility ~4%
                return [
                    [1234567890000, 2970, 3030, 3060, 2940, 1000] for _ in range(30)
                ]

        mocker.patch.object(
            analyzer.client, "get_contracts", return_value=mock_contracts
        )
        mocker.patch.object(analyzer.client, "get_ticker", side_effect=mock_get_ticker)
        mocker.patch.object(analyzer.client, "get_klines", side_effect=mock_get_klines)

        pairs = await analyzer.select_best_pairs(max_pairs=5)

        # Fallback should return both pairs sorted by volume
        # BTCUSDTM (500K) should be first, ETHUSDTM (200K) second
        assert len(pairs) == 2
        assert pairs[0].symbol == "BTCUSDTM"
        assert pairs[1].symbol == "ETHUSDTM"

    @pytest.mark.asyncio
    async def test_select_best_pairs_fallback_respects_max_pairs(
        self, analyzer, mocker
    ):
        """Test that fallback respects max_pairs limit."""
        # Mock contracts
        mock_contracts = [
            {"symbol": "BTCUSDTM", "status": "Open"},
            {"symbol": "ETHUSDTM", "status": "Open"},
            {"symbol": "XRPUSDTM", "status": "Open"},
        ]

        # All pairs have low volume (below threshold)
        async def mock_get_ticker(symbol):
            volumes = {
                "BTCUSDTM": "500000.0",  # highest
                "ETHUSDTM": "300000.0",  # middle
                "XRPUSDTM": "100000.0",  # lowest
            }
            return {
                "price": "100.0",
                "turnover24h": volumes.get(symbol, "0"),
                "ts": 1234567890000,
            }

        def mock_get_klines(symbol, granularity):
            # Use high=105, low=97, price=100.0 for ~8% volatility (within 0.02-0.15)
            return [[1234567890000, 99, 105, 105, 97, 1000] for _ in range(30)]

        mocker.patch.object(
            analyzer.client, "get_contracts", return_value=mock_contracts
        )
        mocker.patch.object(analyzer.client, "get_ticker", side_effect=mock_get_ticker)
        mocker.patch.object(analyzer.client, "get_klines", side_effect=mock_get_klines)

        pairs = await analyzer.select_best_pairs(max_pairs=2)

        # Fallback should respect max_pairs limit
        assert len(pairs) == 2
        assert pairs[0].symbol == "BTCUSDTM"  # highest volume
        assert pairs[1].symbol == "ETHUSDTM"  # second highest volume
