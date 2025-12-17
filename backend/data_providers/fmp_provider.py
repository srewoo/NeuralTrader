"""
Financial Modeling Prep (FMP) Data Provider
Strong for fundamentals and US stocks (250 requests/day free tier)
API Docs: https://site.financialmodelingprep.com/developer/docs
"""

from typing import Optional
import pandas as pd
import aiohttp
from datetime import datetime, timedelta
from .base_provider import BaseDataProvider, StockData
import logging
import os

logger = logging.getLogger(__name__)


class FMPProvider(BaseDataProvider):
    """
    Financial Modeling Prep data provider
    Pros: 250 requests/day free, excellent fundamentals data, good docs
    Cons: Requires API key, rate limits on free tier, primarily US-focused
    """

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: Optional[str] = None):
        # Try to get API key from environment if not provided
        if not api_key:
            api_key = os.getenv("FMP_API_KEY")
        super().__init__("Financial Modeling Prep", api_key=api_key)

    def _check_availability(self) -> bool:
        """Check if FMP API key is configured"""
        return self.api_key is not None and len(self.api_key) > 0

    async def get_quote(self, symbol: str) -> Optional[StockData]:
        """Get current quote from FMP"""
        if not self.is_available:
            logger.warning("FMP API key not configured")
            return None

        try:
            fmp_symbol = self.normalize_symbol(symbol)

            async with aiohttp.ClientSession() as session:
                # Get real-time quote
                quote_url = f"{self.BASE_URL}/quote/{fmp_symbol}"
                params = {"apikey": self.api_key}

                async with session.get(quote_url, params=params) as response:
                    if response.status != 200:
                        logger.warning(f"FMP quote failed with status {response.status}")
                        return None

                    data = await response.json()

                    if not data or len(data) == 0:
                        return None

                    quote = data[0]  # FMP returns array with single item

                    # Get company profile for additional info
                    profile_url = f"{self.BASE_URL}/profile/{fmp_symbol}"
                    profile_data = {}
                    async with session.get(profile_url, params=params) as profile_response:
                        if profile_response.status == 200:
                            profile_list = await profile_response.json()
                            if profile_list and len(profile_list) > 0:
                                profile_data = profile_list[0]

                    return StockData(
                        symbol=fmp_symbol,
                        name=quote.get("name", profile_data.get("companyName", symbol)),
                        current_price=quote.get("price", 0),
                        previous_close=quote.get("previousClose", 0),
                        volume=quote.get("volume", 0),
                        market_cap=quote.get("marketCap", profile_data.get("mktCap")),
                        pe_ratio=quote.get("pe", profile_data.get("pe")),
                        week_52_high=quote.get("yearHigh"),
                        week_52_low=quote.get("yearLow"),
                        sector=profile_data.get("sector"),
                        industry=profile_data.get("industry"),
                        provider=self.name
                    )

        except Exception as e:
            logger.error(f"FMP quote failed for {symbol}: {e}")
            return None

    async def get_historical_data(
        self,
        symbol: str,
        period: str = "6mo",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Get historical data from FMP"""
        if not self.is_available:
            logger.warning("FMP API key not configured")
            return None

        try:
            fmp_symbol = self.normalize_symbol(symbol)

            # Determine endpoint based on interval
            if interval in ["1m", "5m", "15m", "30m", "1h"]:
                # Intraday data (limited to last 5 days on free tier)
                endpoint = f"{self.BASE_URL}/historical-chart/{interval}/{fmp_symbol}"

                async with aiohttp.ClientSession() as session:
                    params = {"apikey": self.api_key}
                    async with session.get(endpoint, params=params) as response:
                        if response.status != 200:
                            return None

                        data = await response.json()

                        if not data or len(data) < 10:
                            return None

                        # Convert to DataFrame (FMP returns newest first, reverse it)
                        df = pd.DataFrame([
                            {
                                'Open': bar['open'],
                                'High': bar['high'],
                                'Low': bar['low'],
                                'Close': bar['close'],
                                'Volume': bar['volume']
                            }
                            for bar in reversed(data)
                        ])

                        df.index = pd.to_datetime([bar['date'] for bar in reversed(data)])
                        return df

            else:
                # Daily data
                # Calculate date range
                end_date = datetime.now()
                period_map = {
                    "1d": timedelta(days=1),
                    "5d": timedelta(days=5),
                    "1mo": timedelta(days=30),
                    "3mo": timedelta(days=90),
                    "6mo": timedelta(days=180),
                    "1y": timedelta(days=365),
                    "5y": timedelta(days=1825)
                }
                delta = period_map.get(period, timedelta(days=180))
                start_date = end_date - delta

                endpoint = f"{self.BASE_URL}/historical-price-full/{fmp_symbol}"

                async with aiohttp.ClientSession() as session:
                    params = {
                        "apikey": self.api_key,
                        "from": start_date.strftime("%Y-%m-%d"),
                        "to": end_date.strftime("%Y-%m-%d")
                    }

                    async with session.get(endpoint, params=params) as response:
                        if response.status != 200:
                            return None

                        data = await response.json()
                        historical = data.get("historical", [])

                        if not historical or len(historical) < 10:
                            return None

                        # Convert to DataFrame (FMP returns newest first, reverse it)
                        df = pd.DataFrame([
                            {
                                'Open': bar['open'],
                                'High': bar['high'],
                                'Low': bar['low'],
                                'Close': bar['close'],
                                'Volume': bar['volume']
                            }
                            for bar in reversed(historical)
                        ])

                        df.index = pd.to_datetime([bar['date'] for bar in reversed(historical)])
                        return df

        except Exception as e:
            logger.error(f"FMP historical data failed for {symbol}: {e}")
            return None

    def get_rate_limit_info(self) -> dict:
        """FMP rate limits"""
        return {
            "provider": self.name,
            "calls_per_minute": "~5-10 (estimated)",
            "calls_per_day": 250,
            "requires_api_key": True,
            "cost": "Free tier: 250/day, Premium: $14-299/mo",
            "strong_for": "Fundamentals, financial statements, US stocks"
        }

    def normalize_symbol(self, symbol: str, exchange: str = "NSE") -> str:
        """
        Normalize symbol for FMP
        US stocks: plain ticker (e.g., "AAPL")
        Indian stocks: likely won't work on free tier
        """
        # Remove any exchange suffixes
        base = symbol.upper().strip().replace('.NS', '').replace('.BO', '')
        return base
