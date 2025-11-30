"""KuCoin Futures API client wrapper."""

import base64
import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import aiohttp

from kucoin_bot.config import APIConfig

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Market data for a trading pair."""

    symbol: str
    price: float
    volume_24h: float
    high_24h: float
    low_24h: float
    volatility: float
    timestamp: int


@dataclass
class Position:
    """Trading position information."""

    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    leverage: int
    unrealized_pnl: float
    margin: float


@dataclass
class Order:
    """Order information."""

    order_id: str
    symbol: str
    side: str
    order_type: str
    size: float
    price: float | None
    status: str


class KuCoinFuturesClient:
    """Async client for KuCoin Futures API."""

    SANDBOX_URL = "https://api-sandbox-futures.kucoin.com"
    PRODUCTION_URL = "https://api-futures.kucoin.com"

    def __init__(self, config: APIConfig):
        """Initialize the KuCoin Futures client."""
        self.config = config
        self.base_url = self.SANDBOX_URL if config.sandbox else self.PRODUCTION_URL
        self._session: aiohttp.ClientSession | None = None
        self._encrypted_passphrase = self._encrypt_passphrase()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _generate_signature(
        self, timestamp: str, method: str, endpoint: str, body: str = ""
    ) -> str:
        """Generate API signature for authentication."""
        message = timestamp + method + endpoint + body
        signature = hmac.new(
            self.config.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(signature).decode("utf-8")

    def _encrypt_passphrase(self) -> str:
        """Encrypt passphrase for API v2 authentication.

        For KuCoin API v2, the passphrase must be encrypted using HMAC-SHA256
        with the API secret as the key, then base64 encoded.
        """
        encrypted_passphrase = hmac.new(
            self.config.api_secret.encode("utf-8"),
            self.config.api_passphrase.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(encrypted_passphrase).decode("utf-8")

    def _get_headers(
        self, method: str, endpoint: str, body: str = ""
    ) -> dict[str, str]:
        """Generate authenticated headers."""
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, method, endpoint, body)

        return {
            "KC-API-KEY": self.config.api_key,
            "KC-API-SIGN": signature,
            "KC-API-TIMESTAMP": timestamp,
            "KC-API-PASSPHRASE": self._encrypted_passphrase,
            "KC-API-KEY-VERSION": "2",
            "Content-Type": "application/json",
        }

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        data: dict | None = None,
    ) -> dict[str, Any]:
        """Make authenticated API request.

        For GET/DELETE requests with query parameters, the signature includes
        the query string as part of the request path per KuCoin API docs:
        signature = HMAC-SHA256(timestamp + method + requestPath + queryString, secret)

        For POST/PUT requests with a body, the signature includes the
        minified JSON body:
        signature = HMAC-SHA256(timestamp + method + requestPath + body, secret)
        """
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"

        # Build the request path for signature generation
        # For GET/DELETE: include query string in the path
        # For POST/PUT: use endpoint as-is (body included separately)
        if params and method in ("GET", "DELETE"):
            query_string = urlencode(params)
            request_path = f"{endpoint}?{query_string}"
        else:
            request_path = endpoint

        # Use minified JSON for POST body (separators without spaces)
        body = json.dumps(data, separators=(",", ":")) if data else ""
        headers = self._get_headers(method, request_path, body)

        try:
            async with session.request(
                method, url, headers=headers, params=params, data=body or None
            ) as response:
                result: dict[str, Any] = await response.json()
                if result.get("code") != "200000":
                    logger.error(f"API error: {result}")
                return result
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

    async def get_account_overview(self, currency: str = "USDT") -> dict[str, Any]:
        """Get account overview including balance.

        Args:
            currency: The settlement currency for the futures account.
                      Use 'USDT' for USDT-margined futures or 'XBT' for BTC-margined futures.
                      Defaults to 'USDT'.

        Returns:
            Account overview data including accountEquity and availableBalance.

        Raises:
            ValueError: If currency is not 'USDT' or 'XBT'.
        """
        if currency not in ("USDT", "XBT"):
            raise ValueError(f"Invalid currency '{currency}'. Must be 'USDT' or 'XBT'.")

        result = await self._request(
            "GET", f"/api/v1/account-overview?currency={currency}"
        )
        return result

    async def get_contracts(self) -> list[dict[str, Any]]:
        """Get all available futures contracts."""
        result = await self._request("GET", "/api/v1/contracts/active")
        contracts: list[dict[str, Any]] = result.get("data", [])
        return contracts

    async def get_ticker(self, symbol: str) -> dict[str, Any]:
        """Get ticker information for a symbol."""
        result = await self._request("GET", f"/api/v1/ticker?symbol={symbol}")
        data: dict[str, Any] = result.get("data", {})
        return data

    async def get_24h_stats(self, symbol: str) -> dict[str, Any]:
        """Get 24-hour statistics for a symbol.

        Note: The /api/v1/trade-statistics endpoint is deprecated in API v3.
        This method now uses ticker data which provides volume and turnover info.
        """
        # Use ticker endpoint which includes 24h volume data
        return await self.get_ticker(symbol)

    async def get_klines(
        self,
        symbol: str,
        granularity: int = 60,
        start: int | None = None,
        end: int | None = None,
    ) -> list[list[Any]]:
        """Get kline/candlestick data.

        Args:
            symbol: Trading pair symbol (e.g., 'XBTUSDTM').
            granularity: Kline interval in seconds. Supported values:
                60 (1min), 300 (5min), 900 (15min), 1800 (30min),
                3600 (1hr), 7200 (2hr), 14400 (4hr), 21600 (6hr),
                28800 (8hr), 43200 (12hr), 86400 (1day), 604800 (1week).
            start: Start time in milliseconds (from). If not provided,
                   defaults to 24 hours ago.
            end: End time in milliseconds (to). If not provided,
                 defaults to current time.

        Returns:
            List of kline data. Each kline is a list:
            [time, open, close, high, low, volume, turnover]
        """
        # Calculate default time range if not provided
        current_time = int(time.time() * 1000)
        if end is None:
            end = current_time
        if start is None:
            # Default to 24 hours of data
            start = end - (24 * 60 * 60 * 1000)

        params: dict[str, Any] = {
            "symbol": symbol,
            "granularity": granularity,
            "from": start,
            "to": end,
        }
        result = await self._request("GET", "/api/v1/kline/query", params=params)
        klines: list[list[Any]] = result.get("data", [])
        return klines

    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        result = await self._request("GET", "/api/v1/positions")
        positions: list[Position] = []
        data = result.get("data")
        if not isinstance(data, list):
            logger.warning(f"Unexpected positions response format: {type(data)}")
            return positions

        for pos in data:
            if not isinstance(pos, dict):
                continue
            if not pos.get("isOpen"):
                continue

            # Safely extract required fields with defaults
            symbol = pos.get("symbol")
            current_qty = pos.get("currentQty", 0)

            if not symbol:
                continue

            positions.append(
                Position(
                    symbol=symbol,
                    side="long" if current_qty > 0 else "short",
                    size=abs(current_qty),
                    entry_price=float(pos.get("avgEntryPrice", 0)),
                    leverage=int(pos.get("realLeverage", 1)),
                    unrealized_pnl=float(pos.get("unrealisedPnl", 0)),
                    margin=float(pos.get("posMargin", 0)),
                )
            )
        return positions

    async def place_order(
        self,
        symbol: str,
        side: str,  # 'buy' or 'sell'
        size: int,
        leverage: int = 1,
        order_type: str = "market",
        price: float | None = None,
        stop_price: float | None = None,
        stop_type: str | None = None,  # 'down' or 'up'
    ) -> Order:
        """Place a new order."""
        data = {
            "clientOid": str(uuid.uuid4()),
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "size": size,
            "leverage": leverage,
        }

        if order_type == "limit" and price:
            data["price"] = str(price)

        if stop_price and stop_type:
            data["stop"] = stop_type
            data["stopPrice"] = str(stop_price)
            data["stopPriceType"] = "TP"

        result = await self._request("POST", "/api/v1/orders", data=data)

        return Order(
            order_id=result.get("data", {}).get("orderId", ""),
            symbol=symbol,
            side=side,
            order_type=order_type,
            size=size,
            price=price,
            status="pending",
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        result = await self._request("DELETE", f"/api/v1/orders/{order_id}")
        return result.get("code") == "200000"

    async def close_position(self, symbol: str) -> bool:
        """Close a position by placing a market order with closeOrder=true.

        Note: The /api/v1/position/close endpoint does not exist in API v3.
        Instead, we place a market order with closeOrder=true to close the
        entire position.

        Args:
            symbol: Trading pair symbol to close.

        Returns:
            True if the close order was placed successfully, False otherwise.
        """
        # First get the current position to determine the side
        positions = await self.get_positions()
        position = None
        for pos in positions:
            if pos.symbol == symbol:
                position = pos
                break

        if position is None:
            logger.warning(f"No open position found for {symbol}")
            return False

        # Determine the opposite side to close the position
        close_side = "sell" if position.side == "long" else "buy"

        data = {
            "clientOid": str(uuid.uuid4()),
            "symbol": symbol,
            "side": close_side,
            "type": "market",
            "closeOrder": True,  # Close entire position
        }

        result = await self._request("POST", "/api/v1/orders", data=data)
        return result.get("code") == "200000"

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol.

        Note: Uses the /api/v2/changeCrossUserLeverage endpoint for API v3
        (not /api/v1/position/risk-limit-level/change which is for changing
        risk limit level, not leverage).

        Args:
            symbol: Trading pair symbol.
            leverage: Desired leverage value.

        Returns:
            True if leverage was set successfully, False otherwise.
        """
        result = await self._request(
            "POST",
            "/api/v2/changeCrossUserLeverage",
            data={"symbol": symbol, "leverage": str(leverage)},
        )
        return result.get("code") == "200000"

    async def close(self) -> None:
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
