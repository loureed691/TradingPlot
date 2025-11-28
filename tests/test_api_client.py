"""Tests for KuCoin API client."""

import base64
import hashlib
import hmac

import pytest

from kucoin_bot.api.client import KuCoinFuturesClient
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
