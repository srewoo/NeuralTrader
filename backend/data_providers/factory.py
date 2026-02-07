"""
Data Provider Factory (Simplified)
Uses yfinance as the primary provider (free, no API key)
TVScreener is handled separately via tvscreener_provider.py
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class DataProviderFactory:
    """
    Factory for data providers — simplified to yfinance only.
    TVScreener (for Indian stocks) is used directly via tvscreener_provider.py
    """

    def __init__(self, provider_keys: Dict[str, Any] = None):
        self.provider_keys = provider_keys or {}
        self.providers = {}
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize yfinance as the sole provider"""
        self.providers["yfinance"] = "yfinance"
        logger.info("yfinance provider available (free, no API key)")

    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote via yfinance"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if info and "regularMarketPrice" in info:
                current_price = info.get("regularMarketPrice")
                previous_close = info.get("previousClose", current_price)
                return {
                    "symbol": symbol.replace('.NS', '').replace('.BO', ''),
                    "name": info.get("longName", info.get("shortName", symbol)),
                    "current_price": round(current_price, 2) if current_price else None,
                    "previous_close": round(previous_close, 2) if previous_close else None,
                    "change": round(info.get("regularMarketChange", 0), 2),
                    "change_percent": round(info.get("regularMarketChangePercent", 0), 2),
                    "volume": info.get("volume"),
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "week_52_high": info.get("fiftyTwoWeekHigh"),
                    "week_52_low": info.get("fiftyTwoWeekLow"),
                    "sector": info.get("sector", "N/A"),
                    "industry": info.get("industry", "N/A"),
                    "source": "yfinance"
                }
        except Exception as e:
            logger.warning(f"yfinance quote failed for {symbol}: {e}")

        return None

    async def get_historical_data(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Get historical data via yfinance"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            if not hist.empty:
                logger.info(f"Historical data from yfinance: {symbol}")
                return hist
        except Exception as e:
            logger.warning(f"yfinance historical data failed for {symbol}: {e}")

        return None

    async def get_company_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company information via yfinance"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if info:
                return {
                    "symbol": symbol,
                    "name": info.get("longName"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "description": info.get("longBusinessSummary"),
                    "website": info.get("website"),
                    "employees": info.get("fullTimeEmployees"),
                    "source": "yfinance"
                }
        except Exception as e:
            logger.warning(f"yfinance company info failed for {symbol}: {e}")

        return None

    async def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get fundamental data via yfinance"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if info:
                return {
                    "symbol": symbol,
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "peg_ratio": info.get("pegRatio"),
                    "price_to_book": info.get("priceToBook"),
                    "dividend_yield": info.get("dividendYield"),
                    "profit_margin": info.get("profitMargins"),
                    "roe": info.get("returnOnEquity"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "source": "yfinance"
                }
        except Exception as e:
            logger.warning(f"yfinance fundamentals failed for {symbol}: {e}")

        return None


# Singleton instance
import threading
_factory_instance: Optional[DataProviderFactory] = None
_factory_lock = threading.Lock()


def get_data_provider_factory(provider_keys: Dict[str, Any] = None) -> DataProviderFactory:
    """Get or create data provider factory instance"""
    global _factory_instance
    with _factory_lock:
        if _factory_instance is None:
            _factory_instance = DataProviderFactory(provider_keys)
    return _factory_instance
