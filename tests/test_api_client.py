"""Tests for KuCoin API client."""

import base64
import hashlib
import hmac

import pytest

from kucoin_bot.api.client import KuCoinFuturesClient, Position
from kucoin_bot.config import APIConfig


class TestKuCoinFuturesClient:
    """Tests for KuCoinFuturesClient."""

    @pytest.fixture
    def config(self):
        """Create API configuration."""
        return APIConfig(
            api_key="test_api_key",
            api_secret="test_api_secret",
            api_passphrase="test_passphrase",
            sandbox=True,
        )

    @pytest.fixture
    def client(self, config):
        """Create KuCoin Futures client."""
        return KuCoinFuturesClient(config)

    def test_encrypt_passphrase(self, client, config):
        """Test passphrase encryption for API v2."""
        encrypted = client._encrypt_passphrase()

        # Verify the encryption is correct
        expected = hmac.new(
            config.api_secret.encode("utf-8"),
            config.api_passphrase.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        expected_b64 = base64.b64encode(expected).decode("utf-8")

        assert encrypted == expected_b64

    def test_encrypt_passphrase_different_secrets(self):
        """Test that different secrets produce different encrypted passphrases."""
        config1 = APIConfig(
            api_key="key1",
            api_secret="secret1",
            api_passphrase="passphrase",
            sandbox=True,
        )
        config2 = APIConfig(
            api_key="key2",
            api_secret="secret2",
            api_passphrase="passphrase",
            sandbox=True,
        )

        client1 = KuCoinFuturesClient(config1)
        client2 = KuCoinFuturesClient(config2)

        assert client1._encrypt_passphrase() != client2._encrypt_passphrase()

    def test_get_headers_includes_encrypted_passphrase(self, client, config):
        """Test that headers include encrypted passphrase."""
        headers = client._get_headers("GET", "/api/v1/test")

        # Verify the passphrase is encrypted, not plain text
        assert headers["KC-API-PASSPHRASE"] != config.api_passphrase

        # Verify it's using the cached encrypted passphrase
        assert headers["KC-API-PASSPHRASE"] == client._encrypted_passphrase

    def test_encrypted_passphrase_cached_on_init(self, client, config):
        """Test that encrypted passphrase is cached during initialization."""
        # Verify the encrypted passphrase is cached
        assert hasattr(client, "_encrypted_passphrase")

        # Verify the cached value matches the encryption result
        expected = client._encrypt_passphrase()
        assert client._encrypted_passphrase == expected

    def test_get_headers_includes_api_version_2(self, client):
        """Test that headers include API version 2."""
        headers = client._get_headers("GET", "/api/v1/test")

        assert headers["KC-API-KEY-VERSION"] == "2"

    def test_generate_signature(self, client, config):
        """Test signature generation."""
        timestamp = "1234567890000"
        method = "GET"
        endpoint = "/api/v1/test"
        body = ""

        signature = client._generate_signature(timestamp, method, endpoint, body)

        # Verify the signature is correct
        message = timestamp + method + endpoint + body
        expected = hmac.new(
            config.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        expected_b64 = base64.b64encode(expected).decode("utf-8")

        assert signature == expected_b64

    def test_sandbox_url_configuration(self, config):
        """Test sandbox URL is used when sandbox is True."""
        client = KuCoinFuturesClient(config)
        assert client.base_url == KuCoinFuturesClient.SANDBOX_URL

    def test_production_url_configuration(self):
        """Test production URL is used when sandbox is False."""
        config = APIConfig(
            api_key="test",
            api_secret="test",
            api_passphrase="test",
            sandbox=False,
        )
        client = KuCoinFuturesClient(config)
        assert client.base_url == KuCoinFuturesClient.PRODUCTION_URL

    def test_default_currency_configuration(self):
        """Test default currency is set in APIConfig."""
        config = APIConfig(
            api_key="test",
            api_secret="test",
            api_passphrase="test",
            sandbox=True,
        )
        # Default should be USDT
        assert config.default_currency == "USDT"

    def test_custom_currency_configuration(self):
        """Test custom currency can be set in APIConfig."""
        config = APIConfig(
            api_key="test",
            api_secret="test",
            api_passphrase="test",
            sandbox=True,
            default_currency="XBT",
        )
        assert config.default_currency == "XBT"

    def test_invalid_currency_configuration_raises_error(self):
        """Test that invalid currency in APIConfig raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            APIConfig(
                api_key="test",
                api_secret="test",
                api_passphrase="test",
                sandbox=True,
                default_currency="INVALID",
            )
        assert "Invalid default_currency" in str(exc_info.value)
        assert "Must be 'USDT' or 'XBT'" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_account_overview_with_default_currency(self, client, mocker):
        """Test get_account_overview uses default USDT currency."""
        mock_request = mocker.patch.object(
            client, "_request", return_value={"code": "200000", "data": {}}
        )

        await client.get_account_overview()

        mock_request.assert_called_once_with(
            "GET", "/api/v1/account-overview?currency=USDT"
        )

    @pytest.mark.asyncio
    async def test_get_account_overview_with_xbt_currency(self, client, mocker):
        """Test get_account_overview passes XBT currency parameter."""
        mock_request = mocker.patch.object(
            client, "_request", return_value={"code": "200000", "data": {}}
        )

        await client.get_account_overview(currency="XBT")

        mock_request.assert_called_once_with(
            "GET", "/api/v1/account-overview?currency=XBT"
        )

    @pytest.mark.asyncio
    async def test_get_account_overview_invalid_currency_raises_error(self, client):
        """Test get_account_overview raises ValueError for invalid currency."""
        with pytest.raises(ValueError) as exc_info:
            await client.get_account_overview(currency="INVALID")

        assert "Invalid currency" in str(exc_info.value)
        assert "Must be 'USDT' or 'XBT'" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_24h_stats_uses_ticker_endpoint(self, client, mocker):
        """Test get_24h_stats uses ticker endpoint (deprecated trade-statistics)."""
        mock_ticker = mocker.patch.object(
            client,
            "get_ticker",
            return_value={"price": "50000", "volumeOf24h": "1000000"},
        )

        result = await client.get_24h_stats("XBTUSDTM")

        mock_ticker.assert_called_once_with("XBTUSDTM")
        assert result["price"] == "50000"
        assert result["volumeOf24h"] == "1000000"

    @pytest.mark.asyncio
    async def test_get_klines_with_default_time_range(self, client, mocker):
        """Test get_klines provides default from/to parameters."""
        mock_request = mocker.patch.object(
            client, "_request", return_value={"code": "200000", "data": []}
        )

        await client.get_klines("XBTUSDTM", granularity=60)

        # Verify the call includes from and to parameters
        call_args = mock_request.call_args
        assert call_args[0][0] == "GET"
        assert call_args[0][1] == "/api/v1/kline/query"
        params = call_args[1]["params"]
        assert params["symbol"] == "XBTUSDTM"
        assert params["granularity"] == 60
        assert "from" in params
        assert "to" in params
        # Default is 24 hours of data
        assert params["to"] - params["from"] == 24 * 60 * 60 * 1000

    @pytest.mark.asyncio
    async def test_get_klines_with_custom_time_range(self, client, mocker):
        """Test get_klines with custom from/to parameters."""
        mock_request = mocker.patch.object(
            client, "_request", return_value={"code": "200000", "data": []}
        )

        start = 1704067200000
        end = 1704153600000

        await client.get_klines("XBTUSDTM", granularity=3600, start=start, end=end)

        call_args = mock_request.call_args
        params = call_args[1]["params"]
        assert params["from"] == start
        assert params["to"] == end

    @pytest.mark.asyncio
    async def test_close_position_uses_close_order_param(self, client, mocker):
        """Test close_position uses closeOrder=true via orders endpoint."""
        # Mock get_positions to return a position
        mock_positions = [
            Position(
                symbol="XBTUSDTM",
                side="long",
                size=100,
                entry_price=50000.0,
                leverage=10,
                unrealized_pnl=50.0,
                margin=500.0,
            )
        ]
        mocker.patch.object(client, "get_positions", return_value=mock_positions)

        mock_request = mocker.patch.object(
            client, "_request", return_value={"code": "200000", "data": {}}
        )

        result = await client.close_position("XBTUSDTM")

        assert result is True
        call_args = mock_request.call_args
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "/api/v1/orders"
        data = call_args[1]["data"]
        assert data["symbol"] == "XBTUSDTM"
        assert data["side"] == "sell"  # Opposite of long
        assert data["type"] == "market"
        assert data["closeOrder"] is True

    @pytest.mark.asyncio
    async def test_close_position_short_uses_buy_side(self, client, mocker):
        """Test close_position for short position uses buy side."""
        mock_positions = [
            Position(
                symbol="ETHUSDTM",
                side="short",
                size=50,
                entry_price=3000.0,
                leverage=5,
                unrealized_pnl=-20.0,
                margin=300.0,
            )
        ]
        mocker.patch.object(client, "get_positions", return_value=mock_positions)

        mock_request = mocker.patch.object(
            client, "_request", return_value={"code": "200000", "data": {}}
        )

        await client.close_position("ETHUSDTM")

        data = mock_request.call_args[1]["data"]
        assert data["side"] == "buy"  # Opposite of short

    @pytest.mark.asyncio
    async def test_close_position_no_position_returns_false(self, client, mocker):
        """Test close_position returns False when no position found."""
        mocker.patch.object(client, "get_positions", return_value=[])

        result = await client.close_position("XBTUSDTM")

        assert result is False

    @pytest.mark.asyncio
    async def test_set_leverage_uses_v2_endpoint(self, client, mocker):
        """Test set_leverage uses correct API v2 endpoint."""
        mock_request = mocker.patch.object(
            client, "_request", return_value={"code": "200000", "data": {}}
        )

        result = await client.set_leverage("XBTUSDTM", 10)

        assert result is True
        mock_request.assert_called_once_with(
            "POST",
            "/api/v2/changeCrossUserLeverage",
            data={"symbol": "XBTUSDTM", "leverage": "10"},
        )
