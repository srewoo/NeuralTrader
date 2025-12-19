"""
Polygon.io (Massive.com) Data Provider
Real-time and historical stock market data
Free tier: 5 API calls/minute, end-of-day data, 2 years history
"""

from typing import Optional
import pandas as pd
import aiohttp
from datetime import datetime, timedelta
from .base_provider import BaseDataProvider, StockData
import logging

logger = logging.getLogger(__name__)


class PolygonProvider(BaseDataProvider):
    """
    Polygon.io (now Massive.com) data provider
    Free tier: 5 calls/min, EOD data, 2 years historical
    Paid: Real-time data, unlimited calls
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("Polygon.io", api_key)

    def _check_availability(self) -> bool:
        """Polygon requires API key"""
        return self.api_key is not None

    async def get_quote(self, symbol: str) -> Optional[StockData]:
        """Get latest quote from Polygon.io"""
        if not self.is_available:
            return None

        try:
            # Get previous close (available on free tier)
            url = f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/prev"
            params = {"apiKey": self.api_key}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.warning(f"Polygon API error {response.status} for {symbol}")
                        return None

                    data = await response.json()

                    if data.get("status") != "OK" or not data.get("results"):
                        return None

                    result = data["results"][0]

                    # Get ticker details for company info
                    details_url = f"{self.BASE_URL}/v3/reference/tickers/{symbol}"
                    async with session.get(details_url, params=params) as details_response:
                        details_data = {}
                        if details_response.status == 200:
                            details_data = await details_response.json()

                    company_name = details_data.get("results", {}).get("name", symbol)
                    market_cap = details_data.get("results", {}).get("market_cap")

                    return StockData(
                        symbol=symbol,
                        name=company_name,
                        current_price=float(result["c"]),  # Close price
                        previous_close=float(result["o"]),  # Open price
                        volume=int(result["v"]),
                        market_cap=market_cap,
                        pe_ratio=None,  # Not available in basic response
                        week_52_high=result.get("h"),  # Daily high
                        week_52_low=result.get("l"),  # Daily low
                        sector=None,
                        industry=None,
                        provider=self.name
                    )

        except Exception as e:
            logger.error(f"Polygon quote failed for {symbol}: {e}")
            return None

    async def get_historical_data(
        self,
        symbol: str,
        period: str = "6mo",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Get historical data from Polygon.io"""
        if not self.is_available:
            return None

        try:
            # Convert period to date range
            end_date = datetime.now()
            period_map = {
                "1d": 1, "5d": 5, "1mo": 30, "3mo": 90,
                "6mo": 180, "1y": 365, "2y": 730
            }
            days = period_map.get(period, 180)
            start_date = end_date - timedelta(days=days)

            # Convert interval to Polygon format
            interval_map = {
                "1m": ("1", "minute"),
                "5m": ("5", "minute"),
                "15m": ("15", "minute"),
                "1h": ("1", "hour"),
                "1d": ("1", "day")
            }
            multiplier, timespan = interval_map.get(interval, ("1", "day"))

            url = f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {
                "apiKey": self.api_key,
                "limit": 50000  # Max results
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.warning(f"Polygon historical data error {response.status} for {symbol}")
                        return None

                    data = await response.json()

                    if data.get("status") != "OK" or not data.get("results"):
                        return None

                    # Convert to DataFrame
                    results = data["results"]
                    df = pd.DataFrame(results)

                    # Rename columns to match yfinance format
                    df.rename(columns={
                        "t": "Date",
                        "o": "Open",
                        "h": "High",
                        "l": "Low",
                        "c": "Close",
                        "v": "Volume"
                    }, inplace=True)

                    # Convert timestamp to datetime
                    df["Date"] = pd.to_datetime(df["Date"], unit="ms")
                    df.set_index("Date", inplace=True)

                    return df

        except Exception as e:
            logger.error(f"Polygon historical data failed for {symbol}: {e}")
            return None

    def get_rate_limit_info(self) -> dict:
        """Polygon.io rate limits"""
        return {
            "provider": self.name,
            "calls_per_minute": "5 (free) / unlimited (paid)",
            "calls_per_day": "Unlimited",
            "requires_api_key": True,
            "cost": "Free tier available, paid from $29/month",
            "note": "Free tier: EOD data only, 2 years history. Paid: Real-time <20ms"
        }


def get_polygon_provider(api_key: str) -> PolygonProvider:
    """Factory function to create Polygon provider"""
    return PolygonProvider(api_key=api_key)
