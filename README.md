# KuCoin Futures Trading Bot

An advanced, fully automated trading bot for KuCoin Futures with leverage support, AI-powered predictions, and comprehensive risk management.

## Features

### 🔄 24/7 Continuous Market Monitoring
- Real-time market data analysis
- Automatic trading pair selection based on volume and volatility
- Continuous position monitoring and management

### 📈 Multi-Strategy Trading
- **Trend Following**: EMA crossovers combined with RSI for trend-based entries
- **Scalping**: Bollinger Bands and RSI for quick mean-reversion trades
- **Statistical Arbitrage**: Z-score based mean reversion strategy
- **AI Predictions**: Machine learning (Random Forest) for price direction forecasting

### 🎯 Auto Position Management
- Automatic long/short position selection
- Dynamic leverage adjustment based on market conditions
- Stop-loss and take-profit automation
- Position sizing based on portfolio percentage

### ⚡ Advanced Risk Management
- Maximum leverage limits
- Position size controls
- Daily loss limits
- Drawdown protection
- Consecutive loss tracking
- Automatic strategy adjustment based on performance

### 🧠 Dynamic Strategy Switching
- Performance-based strategy selection
- Automatic enabling/disabling of underperforming strategies
- Weighted signal aggregation from multiple strategies

## Installation

### Prerequisites
- Python 3.10 or higher
- KuCoin Futures API credentials

### Setup

1. Clone the repository:
```bash
git clone https://github.com/loureed691/TradingPlot.git
cd TradingPlot
```

2. Install dependencies:
```bash
pip install -e ".[dev]"
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your KuCoin API credentials
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KUCOIN_API_KEY` | Your KuCoin API key | Required |
| `KUCOIN_API_SECRET` | Your KuCoin API secret | Required |
| `KUCOIN_API_PASSPHRASE` | Your KuCoin API passphrase | Required |
| `KUCOIN_SANDBOX` | Use sandbox/testnet | `true` |
| `MAX_LEVERAGE` | Maximum leverage allowed | `10` |
| `MAX_POSITION_SIZE_PERCENT` | Max % of portfolio per position | `5.0` |
| `STOP_LOSS_PERCENT` | Default stop-loss percentage | `2.0` |
| `TAKE_PROFIT_PERCENT` | Default take-profit percentage | `4.0` |
| `MIN_VOLUME_USD` | Minimum 24h volume for pair selection | `1000000.0` |
| `UPDATE_INTERVAL` | Market data update interval (seconds) | `60` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Usage

### Running the Bot

```bash
python run_bot.py
```

Or using the module:

```bash
python -m kucoin_bot.bot
```

### Programmatic Usage

```python
import asyncio
from kucoin_bot.bot import KuCoinFuturesBot
from kucoin_bot.config import BotConfig

# Create custom configuration
config = BotConfig.from_env()
config.risk.max_leverage = 5
config.risk.max_position_size_percent = 3.0

# Initialize and run bot
bot = KuCoinFuturesBot(config)
asyncio.run(bot.start())
```

## Architecture

```
src/kucoin_bot/
├── __init__.py
├── bot.py                 # Main bot orchestration
├── config.py              # Configuration management
├── api/
│   ├── __init__.py
│   └── client.py          # KuCoin Futures API client
├── strategies/
│   ├── __init__.py
│   ├── base.py            # Base strategy class
│   ├── trend_following.py # EMA/RSI trend strategy
│   ├── scalping.py        # Bollinger Bands scalping
│   ├── arbitrage.py       # Statistical arbitrage
│   ├── ai_predictor.py    # ML-based predictions
│   └── strategy_manager.py # Strategy orchestration
├── risk_management/
│   ├── __init__.py
│   ├── position_manager.py # Position and portfolio management
│   └── risk_controller.py  # Risk assessment and control
└── utils/
    ├── __init__.py
    ├── market_analyzer.py  # Market data analysis
    └── indicators.py       # Technical indicators
```

## Trading Strategies

### Trend Following
Uses EMA crossovers (12/26) combined with RSI (14) to identify trend reversals:
- Long: Fast EMA crosses above slow EMA + RSI not overbought
- Short: Fast EMA crosses below slow EMA + RSI not oversold

### Scalping
Uses Bollinger Bands (20, 2σ) and short-term RSI (7):
- Long: Price at lower band + RSI oversold + volume surge
- Short: Price at upper band + RSI overbought + volume surge

### Statistical Arbitrage
Mean reversion based on Z-score:
- Entry: Z-score exceeds ±2.0 standard deviations
- Exit: Z-score returns to ±0.5 standard deviations

### AI Predictor
Machine learning model (Random Forest) trained on:
- Price returns at multiple timeframes
- Technical indicators (RSI, MACD, Bollinger Bands)
- Volume patterns
- Price momentum

## Risk Management

### Position Sizing
- Maximum 5% of portfolio per position (configurable)
- Reduced sizing after consecutive losses
- Dynamic adjustment based on drawdown

### Leverage Control
- Maximum leverage cap (default: 10x)
- Automatic reduction during unfavorable conditions
- Strategy-specific leverage recommendations

### Loss Protection
- Daily loss limit (default: 10% of portfolio)
- Maximum consecutive losses before pause (5)
- Drawdown threshold (15%)

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ --cov=kucoin_bot --cov-report=html
```

## ⚠️ Disclaimer

**This trading bot is provided for educational and research purposes only.**

- Trading futures with leverage carries significant risk
- Past performance does not guarantee future results
- Always start with the sandbox/testnet environment
- Never trade with funds you cannot afford to lose
- The authors are not responsible for any financial losses

## License

MIT License - See LICENSE file for details