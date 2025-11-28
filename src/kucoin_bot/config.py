"""Configuration settings for the KuCoin Futures Trading Bot."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class APIConfig:
    """KuCoin API configuration."""

    api_key: str = field(default_factory=lambda: os.getenv("KUCOIN_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("KUCOIN_API_SECRET", ""))
    api_passphrase: str = field(
        default_factory=lambda: os.getenv("KUCOIN_API_PASSPHRASE", "")
    )
    sandbox: bool = field(
        default_factory=lambda: os.getenv("KUCOIN_SANDBOX", "true").lower() == "true"
    )
    default_currency: str = field(
        default_factory=lambda: os.getenv("KUCOIN_DEFAULT_CURRENCY", "USDT")
    )


@dataclass
class RiskConfig:
    """Risk management configuration.

    When adaptive_mode is True, max_leverage, max_position_size_percent,
    stop_loss_percent, and take_profit_percent will be dynamically calculated
    by the AdaptiveRiskSettings based on market conditions and strategy performance.
    The values provided here serve as initial defaults until adaptive calculations
    are available.
    """

    max_leverage: int = 10
    max_position_size_percent: float = 5.0  # Max % of portfolio per position
    stop_loss_percent: float = 2.0  # Default stop-loss %
    take_profit_percent: float = 4.0  # Default take-profit %
    max_open_positions: int = 5
    max_daily_loss_percent: float = 10.0  # Max daily loss before stopping
    adaptive_mode: bool = True  # When True, risk parameters are auto-calculated


@dataclass
class TradingConfig:
    """Trading configuration."""

    min_volume_usd: float = 1000000.0  # Minimum 24h volume for pair selection
    min_volatility: float = 0.02  # Minimum volatility threshold
    max_volatility: float = 0.15  # Maximum volatility threshold
    update_interval_seconds: int = 60  # Market data update interval
    strategy_switch_interval: int = 300  # Strategy evaluation interval


@dataclass
class BotConfig:
    """Main bot configuration."""

    api: APIConfig = field(default_factory=APIConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "BotConfig":
        """Create configuration from environment variables."""
        # Check if adaptive mode is enabled (default: True)
        adaptive_mode = os.getenv("ADAPTIVE_RISK_MODE", "true").lower() == "true"

        return cls(
            api=APIConfig(),
            risk=RiskConfig(
                max_leverage=int(os.getenv("MAX_LEVERAGE", "10")),
                max_position_size_percent=float(
                    os.getenv("MAX_POSITION_SIZE_PERCENT", "5.0")
                ),
                stop_loss_percent=float(os.getenv("STOP_LOSS_PERCENT", "2.0")),
                take_profit_percent=float(os.getenv("TAKE_PROFIT_PERCENT", "4.0")),
                adaptive_mode=adaptive_mode,
            ),
            trading=TradingConfig(
                min_volume_usd=float(os.getenv("MIN_VOLUME_USD", "1000000.0")),
                update_interval_seconds=int(os.getenv("UPDATE_INTERVAL", "60")),
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
