"""
Finnhub Data Provider
Real-time quotes, international coverage, generous free tier (60 calls/min)
API Docs: https://finnhub.io/docs/api
"""

from typing import Optional
import pandas as pd
import aiohttp
from datetime import datetime, timedelta
from .base_provider import BaseDataProvider, StockData
import logging
import os

logger = logging.getLogger(__name__)


class FinnhubProvider(BaseDataProvider):
    """
    Finnhub data provider
    Pros: 60 calls/min free, real-time (15-min delayed), international coverage, WebSocket
    Cons: Requires API key, Indian stocks need special symbol format
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        # Try to get API key from environment if not provided
        if not api_key:
            api_key = os.getenv("FINNHUB_API_KEY")
        super().__init__("Finnhub", api_key=api_key)

    def _check_availability(self) -> bool:
        """Check if Finnhub API key is configured"""
        return self.api_key is not None and len(self.api_key) > 0

    async def get_quote(self, symbol: str) -> Optional[StockData]:
        """Get current quote from Finnhub"""
        if not self.is_available:
            logger.warning("Finnhub API key not configured")
            return None

        try:
            # For Indian stocks, Finnhub uses format like "RELIANCE.NSE" or "RELIANCE.BSE"
            for exchange in ["NSE", "BSE"]:
                finnhub_symbol = self.normalize_symbol(symbol, exchange)

                async with aiohttp.ClientSession() as session:
                    # Get quote
                    quote_url = f"{self.BASE_URL}/quote"
                    params = {"symbol": finnhub_symbol, "token": self.api_key}

                    async with session.get(quote_url, params=params) as response:
                        if response.status != 200:
                            continue

                        quote_data = await response.json()

                        # Check if we got valid data
                        if quote_data.get("c") == 0:  # Current price is 0 means no data
                            continue

                        # Get company profile for additional info
                        profile_url = f"{self.BASE_URL}/stock/profile2"
                        async with session.get(profile_url, params=params) as profile_response:
                            profile_data = {}
                            if profile_response.status == 200:
                                profile_data = await profile_response.json()

                        return StockData(
                            symbol=finnhub_symbol,
                            name=profile_data.get("name", symbol),
                            current_price=quote_data.get("c", 0),  # Current price
                            previous_close=quote_data.get("pc", 0),  # Previous close
                            volume=0,  # Finnhub quote doesn't include volume
                            market_cap=profile_data.get("marketCapitalization"),
                            pe_ratio=None,  # Not in basic quote
                            week_52_high=quote_data.get("h", None),  # High of the day (approximation)
                            week_52_low=quote_data.get("l", None),  # Low of the day (approximation)
                            sector=profile_data.get("finnhubIndustry"),
                            industry=profile_data.get("finnhubIndustry"),
                            provider=self.name
                        )

            logger.warning(f"No Finnhub data for {symbol} on NSE or BSE")
            return None

        except Exception as e:
            logger.error(f"Finnhub quote failed for {symbol}: {e}")
            return None

    async def get_historical_data(
        self,
        symbol: str,
        period: str = "6mo",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Get historical candles from Finnhub"""
        if not self.is_available:
            logger.warning("Finnhub API key not configured")
            return None

        try:
            # Convert period to timestamps
            end_time = int(datetime.now().timestamp())
            period_map = {
                "1d": timedelta(days=1),
                "5d": timedelta(days=5),
                "1mo": timedelta(days=30),
                "3mo": timedelta(days=90),
                "6mo": timedelta(days=180),
                "1y": timedelta(days=365)
            }
            delta = period_map.get(period, timedelta(days=180))
            start_time = int((datetime.now() - delta).timestamp())

            # Interval conversion
            resolution_map = {
                "1m": "1",
                "5m": "5",
                "15m": "15",
                "30m": "30",
                "1h": "60",
                "1d": "D",
                "1w": "W",
                "1mo": "M"
            }
            resolution = resolution_map.get(interval, "D")

            # Try NSE first, then BSE
            for exchange in ["NSE", "BSE"]:
                finnhub_symbol = self.normalize_symbol(symbol, exchange)

                async with aiohttp.ClientSession() as session:
                    candles_url = f"{self.BASE_URL}/stock/candle"
                    params = {
                        "symbol": finnhub_symbol,
                        "resolution": resolution,
                        "from": start_time,
                        "to": end_time,
                        "token": self.api_key
                    }

                    async with session.get(candles_url, params=params) as response:
                        if response.status != 200:
                            continue

                        data = await response.json()

                        # Check if we got valid data
                        if data.get("s") != "ok" or not data.get("c"):
                            continue

                        # Convert to DataFrame
                        df = pd.DataFrame({
                            'Open': data['o'],
                            'High': data['h'],
                            'Low': data['l'],
                            'Close': data['c'],
                            'Volume': data['v']
                        })

                        # Add timestamp as index
                        df.index = pd.to_datetime(data['t'], unit='s')

                        if len(df) >= 10:  # Minimum data check
                            return df

            logger.warning(f"Insufficient Finnhub historical data for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Finnhub historical data failed for {symbol}: {e}")
            return None

    def get_rate_limit_info(self) -> dict:
        """Finnhub rate limits"""
        return {
            "provider": self.name,
            "calls_per_minute": 60,
            "calls_per_day": "Unlimited on free tier",
            "requires_api_key": True,
            "cost": "Free tier available",
            "websocket_available": True
        }

    def normalize_symbol(self, symbol: str, exchange: str = "NSE") -> str:
        """Normalize symbol for Finnhub (format: SYMBOL.EXCHANGE)"""
        base = symbol.upper().strip().replace('.NS', '').replace('.BO', '').replace('.NSE', '').replace('.BSE', '')
        return f"{base}.{exchange}"
