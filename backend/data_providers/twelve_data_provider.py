"""
Twelve Data API Provider
Real-time and historical stock market data
Free tier: 8 calls/min, 800 calls/day, ~170ms latency
"""

from typing import Optional
import pandas as pd
import aiohttp
from datetime import datetime
from .base_provider import BaseDataProvider, StockData
import logging

logger = logging.getLogger(__name__)


class TwelveDataProvider(BaseDataProvider):
    """
    Twelve Data API provider
    Free tier: 8 calls/min, 800/day, real-time data
    Paid: Starting $29/month for 55 calls/min
    """

    BASE_URL = "https://api.twelvedata.com"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("Twelve Data", api_key)

    def _check_availability(self) -> bool:
        """Twelve Data requires API key"""
        return self.api_key is not None

    async def get_quote(self, symbol: str) -> Optional[StockData]:
        """Get real-time quote from Twelve Data"""
        if not self.is_available:
            return None

        try:
            # Get real-time quote
            url = f"{self.BASE_URL}/quote"
            params = {
                "symbol": symbol,
                "apikey": self.api_key
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.warning(f"Twelve Data API error {response.status} for {symbol}")
                        return None

                    data = await response.json()

                    # Check for API errors
                    if "code" in data or "status" in data:
                        logger.warning(f"Twelve Data error for {symbol}: {data.get('message', 'Unknown error')}")
                        return None

                    # Get additional company info
                    profile_url = f"{self.BASE_URL}/profile"
                    profile_params = {
                        "symbol": symbol,
                        "apikey": self.api_key
                    }

                    profile_data = {}
                    async with session.get(profile_url, params=profile_params) as profile_response:
                        if profile_response.status == 200:
                            profile_data = await profile_response.json()

                    return StockData(
                        symbol=data.get("symbol", symbol),
                        name=data.get("name", profile_data.get("name", symbol)),
                        current_price=float(data.get("close", 0)),
                        previous_close=float(data.get("previous_close", data.get("close", 0))),
                        volume=int(data.get("volume", 0)),
                        market_cap=profile_data.get("market_cap"),
                        pe_ratio=profile_data.get("pe_ratio"),
                        week_52_high=float(data.get("fifty_two_week", {}).get("high", 0)) if isinstance(data.get("fifty_two_week"), dict) else None,
                        week_52_low=float(data.get("fifty_two_week", {}).get("low", 0)) if isinstance(data.get("fifty_two_week"), dict) else None,
                        sector=profile_data.get("sector"),
                        industry=profile_data.get("industry"),
                        provider=self.name
                    )

        except Exception as e:
            logger.error(f"Twelve Data quote failed for {symbol}: {e}")
            return None

    async def get_historical_data(
        self,
        symbol: str,
        period: str = "6mo",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Get historical time series from Twelve Data"""
        if not self.is_available:
            return None

        try:
            # Convert period to outputsize
            period_map = {
                "1d": 1, "5d": 5, "1mo": 30, "3mo": 90,
                "6mo": 180, "1y": 365, "2y": 730, "5y": 1825
            }
            outputsize = period_map.get(period, 180)

            # Convert interval to Twelve Data format
            interval_map = {
                "1m": "1min",
                "5m": "5min",
                "15m": "15min",
                "30m": "30min",
                "1h": "1h",
                "1d": "1day",
                "1wk": "1week",
                "1mo": "1month"
            }
            td_interval = interval_map.get(interval, "1day")

            url = f"{self.BASE_URL}/time_series"
            params = {
                "symbol": symbol,
                "interval": td_interval,
                "outputsize": min(outputsize, 5000),  # Max 5000
                "apikey": self.api_key,
                "format": "JSON"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.warning(f"Twelve Data historical error {response.status} for {symbol}")
                        return None

                    data = await response.json()

                    # Check for API errors
                    if "code" in data or "status" in data:
                        logger.warning(f"Twelve Data error for {symbol}: {data.get('message', 'Unknown error')}")
                        return None

                    if "values" not in data or not data["values"]:
                        return None

                    # Convert to DataFrame
                    df = pd.DataFrame(data["values"])

                    # Rename columns to match yfinance format
                    df.rename(columns={
                        "datetime": "Date",
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume"
                    }, inplace=True)

                    # Convert datetime and set as index
                    df["Date"] = pd.to_datetime(df["Date"])
                    df.set_index("Date", inplace=True)

                    # Convert price columns to float
                    for col in ["Open", "High", "Low", "Close"]:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                    # Convert volume to int
                    if "Volume" in df.columns:
                        df["Volume"] = pd.to_numeric(df["Volume"], errors='coerce').fillna(0).astype(int)

                    # Sort by date (oldest first)
                    df.sort_index(inplace=True)

                    return df

        except Exception as e:
            logger.error(f"Twelve Data historical data failed for {symbol}: {e}")
            return None

    def get_rate_limit_info(self) -> dict:
        """Twelve Data rate limits"""
        return {
            "provider": self.name,
            "calls_per_minute": "8 (free) / 55+ (paid)",
            "calls_per_day": "800 (free) / unlimited (paid)",
            "requires_api_key": True,
            "cost": "Free tier available, paid from $29/month",
            "note": "Free: 8/min, 800/day, real-time ~170ms. Paid: 55+/min, unlimited/day"
        }


def get_twelve_data_provider(api_key: str) -> TwelveDataProvider:
    """Factory function to create Twelve Data provider"""
    return TwelveDataProvider(api_key=api_key)
