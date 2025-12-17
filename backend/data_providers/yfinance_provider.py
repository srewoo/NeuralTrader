"""
Yahoo Finance Data Provider
Uses yfinance library (free, no API key needed)
"""

from typing import Optional
import pandas as pd
import yfinance as yf
from .base_provider import BaseDataProvider, StockData
import logging

logger = logging.getLogger(__name__)


class YFinanceProvider(BaseDataProvider):
    """
    Yahoo Finance data provider using yfinance library
    Pros: Free, no API key, good coverage
    Cons: Rate limits unclear, can be unstable, 15-min delayed
    """

    def __init__(self):
        super().__init__("Yahoo Finance", api_key=None)

    def _check_availability(self) -> bool:
        """Yahoo Finance is always available (no API key required)"""
        return True

    async def get_quote(self, symbol: str) -> Optional[StockData]:
        """Get current quote from Yahoo Finance"""
        try:
            # Try NSE first, then BSE
            for suffix in ['.NS', '.BO']:
                ticker_symbol = self._get_indian_symbol(symbol, suffix)
                ticker = yf.Ticker(ticker_symbol)
                info = ticker.info
                hist = ticker.history(period="1d")

                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    previous_close = float(info.get('previousClose', hist['Close'].iloc[-1]))

                    return StockData(
                        symbol=ticker_symbol,
                        name=info.get('longName', symbol),
                        current_price=current_price,
                        previous_close=previous_close,
                        volume=int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0,
                        market_cap=info.get('marketCap'),
                        pe_ratio=info.get('trailingPE'),
                        week_52_high=info.get('fiftyTwoWeekHigh'),
                        week_52_low=info.get('fiftyTwoWeekLow'),
                        sector=info.get('sector'),
                        industry=info.get('industry'),
                        provider=self.name
                    )

            logger.warning(f"No data found for {symbol} on NSE or BSE")
            return None

        except Exception as e:
            logger.error(f"YFinance quote failed for {symbol}: {e}")
            return None

    async def get_historical_data(
        self,
        symbol: str,
        period: str = "6mo",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Get historical data from Yahoo Finance"""
        try:
            # Try NSE first, then BSE
            for suffix in ['.NS', '.BO']:
                ticker_symbol = self._get_indian_symbol(symbol, suffix)
                ticker = yf.Ticker(ticker_symbol)
                hist = ticker.history(period=period, interval=interval)

                if not hist.empty and len(hist) >= 10:  # Minimum data check
                    return hist

            logger.warning(f"Insufficient historical data for {symbol}")
            return None

        except Exception as e:
            logger.error(f"YFinance historical data failed for {symbol}: {e}")
            return None

    def get_rate_limit_info(self) -> dict:
        """Yahoo Finance rate limits are unclear/undocumented"""
        return {
            "provider": self.name,
            "calls_per_minute": "Unknown (estimated ~60)",
            "calls_per_day": "Unknown (estimated ~2000)",
            "requires_api_key": False,
            "cost": "Free"
        }

    def _get_indian_symbol(self, symbol: str, suffix: str) -> str:
        """Add NSE/BSE suffix to Indian stock symbols"""
        base = symbol.upper().strip().replace('.NS', '').replace('.BO', '')
        return f"{base}{suffix}"

    def normalize_symbol(self, symbol: str, exchange: str = "NSE") -> str:
        """Normalize symbol for Yahoo Finance (add .NS or .BO)"""
        suffix = ".NS" if exchange == "NSE" else ".BO"
        return self._get_indian_symbol(symbol, suffix)
