#!/usr/bin/env python3
"""Entry point for the KuCoin Futures Trading Bot."""

import asyncio
import sys

from kucoin_bot.bot import KuCoinFuturesBot


def main():
    """Main entry point."""
    bot = KuCoinFuturesBot()

    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\nShutdown requested...")
        sys.exit(0)


if __name__ == "__main__":
    main()
