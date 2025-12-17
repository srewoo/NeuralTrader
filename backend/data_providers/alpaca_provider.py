"""
Alpaca Data Provider
Market data with up to 200 API calls/min on free tier
Best for US stocks, also supports crypto
API Docs: https://alpaca.markets/docs/api-references/market-data-api/
"""

from typing import Optional
import pandas as pd
import aiohttp
from datetime import datetime, timedelta
from .base_provider import BaseDataProvider, StockData
import logging
import os
import base64

logger = logging.getLogger(__name__)


class AlpacaProvider(BaseDataProvider):
    """
    Alpaca market data provider
    Pros: 200 calls/min free, good for US stocks, broker integration available
    Cons: Requires API key, limited Indian stock support (mainly US markets)
    """

    # Alpaca uses separate data API endpoint
    BASE_URL = "https://data.alpaca.markets/v2"

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        # Try to get credentials from environment if not provided
        if not api_key:
            api_key = os.getenv("ALPACA_API_KEY")
        if not api_secret:
            api_secret = os.getenv("ALPACA_API_SECRET")

        self.api_secret = api_secret
        super().__init__("Alpaca", api_key=api_key)

    def _check_availability(self) -> bool:
        """Check if Alpaca API credentials are configured"""
        return (
            self.api_key is not None and
            len(self.api_key) > 0 and
            self.api_secret is not None and
            len(self.api_secret) > 0
        )

    def _get_headers(self) -> dict:
        """Get authentication headers for Alpaca API"""
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret
        }

    async def get_quote(self, symbol: str) -> Optional[StockData]:
        """Get current quote from Alpaca"""
        if not self.is_available:
            logger.warning("Alpaca API credentials not configured")
            return None

        try:
            # Alpaca primarily supports US stocks
            # For Indian stocks, this will likely fail - it's better as fallback for US symbols
            alpaca_symbol = self.normalize_symbol(symbol)

            async with aiohttp.ClientSession() as session:
                # Get latest quote
                quote_url = f"{self.BASE_URL}/stocks/{alpaca_symbol}/quotes/latest"
                headers = self._get_headers()

                async with session.get(quote_url, headers=headers) as response:
                    if response.status != 200:
                        logger.warning(f"Alpaca quote failed with status {response.status}")
                        return None

                    data = await response.json()
                    quote = data.get("quote", {})

                    if not quote:
                        return None

                    # Get snapshot for more complete data
                    snapshot_url = f"{self.BASE_URL}/stocks/{alpaca_symbol}/snapshot"
                    async with session.get(snapshot_url, headers=headers) as snap_response:
                        snapshot = {}
                        if snap_response.status == 200:
                            snapshot = await snap_response.json()

                    latest_trade = snapshot.get("latestTrade", {})
                    prev_daily = snapshot.get("prevDailyBar", {})

                    current_price = latest_trade.get("p", quote.get("ap", 0))  # Ask price as fallback
                    previous_close = prev_daily.get("c", current_price)

                    return StockData(
                        symbol=alpaca_symbol,
                        name=alpaca_symbol,  # Alpaca doesn't provide company names in quotes
                        current_price=current_price,
                        previous_close=previous_close,
                        volume=latest_trade.get("s", 0),  # Size (volume)
                        market_cap=None,  # Not provided
                        pe_ratio=None,  # Not provided
                        week_52_high=None,  # Would need separate API call
                        week_52_low=None,  # Would need separate API call
                        sector=None,  # Not provided
                        industry=None,  # Not provided
                        provider=self.name
                    )

        except Exception as e:
            logger.error(f"Alpaca quote failed for {symbol}: {e}")
            return None

    async def get_historical_data(
        self,
        symbol: str,
        period: str = "6mo",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Get historical bars from Alpaca"""
        if not self.is_available:
            logger.warning("Alpaca API credentials not configured")
            return None

        try:
            alpaca_symbol = self.normalize_symbol(symbol)

            # Convert period to date range
            end_date = datetime.now()
            period_map = {
                "1d": timedelta(days=1),
                "5d": timedelta(days=5),
                "1mo": timedelta(days=30),
                "3mo": timedelta(days=90),
                "6mo": timedelta(days=180),
                "1y": timedelta(days=365)
            }
            delta = period_map.get(period, timedelta(days=180))
            start_date = end_date - delta

            # Determine timeframe (Alpaca uses different format)
            timeframe_map = {
                "1m": "1Min",
                "5m": "5Min",
                "15m": "15Min",
                "1h": "1Hour",
                "1d": "1Day"
            }
            timeframe = timeframe_map.get(interval, "1Day")

            async with aiohttp.ClientSession() as session:
                bars_url = f"{self.BASE_URL}/stocks/{alpaca_symbol}/bars"
                headers = self._get_headers()
                params = {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "timeframe": timeframe,
                    "limit": 10000  # Max limit
                }

                async with session.get(bars_url, headers=headers, params=params) as response:
                    if response.status != 200:
                        logger.warning(f"Alpaca bars failed with status {response.status}")
                        return None

                    data = await response.json()
                    bars = data.get("bars", [])

                    if not bars or len(bars) < 10:
                        return None

                    # Convert to DataFrame
                    df = pd.DataFrame([
                        {
                            'Open': bar['o'],
                            'High': bar['h'],
                            'Low': bar['l'],
                            'Close': bar['c'],
                            'Volume': bar['v']
                        }
                        for bar in bars
                    ])

                    # Add timestamp as index
                    df.index = pd.to_datetime([bar['t'] for bar in bars])

                    return df

        except Exception as e:
            logger.error(f"Alpaca historical data failed for {symbol}: {e}")
            return None

    def get_rate_limit_info(self) -> dict:
        """Alpaca rate limits"""
        return {
            "provider": self.name,
            "calls_per_minute": 200,
            "calls_per_day": "Unlimited on free tier",
            "requires_api_key": True,
            "cost": "Free tier available",
            "websocket_available": True,
            "notes": "Best for US stocks, limited international coverage"
        }

    def normalize_symbol(self, symbol: str, exchange: str = "NSE") -> str:
        """
        Normalize symbol for Alpaca (US format)
        Note: Alpaca primarily supports US stocks, so Indian stocks may not work
        """
        # Remove any exchange suffixes
        base = symbol.upper().strip().replace('.NS', '').replace('.BO', '')

        # For US stocks, Alpaca uses plain ticker symbols
        # Indian stocks would need special handling (likely won't work)
        return base
