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
            "KC-API-PASSPHRASE": self.config.api_passphrase,
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
        """Make authenticated API request."""
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"

        body = json.dumps(data) if data else ""
        headers = self._get_headers(method, endpoint, body)

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

    async def get_account_overview(self) -> dict[str, Any]:
        """Get account overview including balance."""
        result = await self._request("GET", "/api/v1/account-overview")
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
        """Get 24-hour statistics for a symbol."""
        result = await self._request("GET", f"/api/v1/trade-statistics?symbol={symbol}")
        data: dict[str, Any] = result.get("data", {})
        return data

    async def get_klines(
        self, symbol: str, granularity: int = 60, start: int | None = None
    ) -> list[list[Any]]:
        """Get kline/candlestick data."""
        params: dict[str, Any] = {"symbol": symbol, "granularity": granularity}
        if start:
            params["from"] = start
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
        """Close a position."""
        result = await self._request(
            "POST", "/api/v1/position/close", data={"symbol": symbol}
        )
        return result.get("code") == "200000"

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        result = await self._request(
            "POST",
            "/api/v1/position/risk-limit-level/change",
            data={"symbol": symbol, "level": leverage},
        )
        return result.get("code") == "200000"

    async def close(self) -> None:
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
