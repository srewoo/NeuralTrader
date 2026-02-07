"""
Data Provider Manager (Simplified)
Uses TVScreener (free, primary) + yfinance (free, fallback)
"""

from typing import Optional, List, Dict, Any
import pandas as pd
from .base_provider import BaseDataProvider, StockData
from .yfinance_provider import YFinanceProvider
import logging

logger = logging.getLogger(__name__)


class DataProviderManager:
    """
    Manages data providers: TVScreener (primary) + yfinance (fallback)
    Both are free and require no API keys.
    """

    def __init__(self, api_keys: Optional[Dict[str, Any]] = None):
        self.providers: List[BaseDataProvider] = []

        # TVScreener (free, primary for Indian stocks)
        try:
            from .tvscreener_provider import TVScreenerProvider
            tv = TVScreenerProvider()
            self.providers.append(tv)
            logger.info("TVScreener added as primary provider")
        except Exception as e:
            logger.warning(f"TVScreener unavailable: {e}")

        # YFinance (always available, no API key needed)
        yfinance = YFinanceProvider()
        self.providers.append(yfinance)
        provider_names = ', '.join(p.name for p in self.providers)
        logger.info(f"Initialized {len(self.providers)} data providers ({provider_names})")

    def _validate_quote(self, quote) -> bool:
        """Validate quote data for quality issues."""
        if not quote:
            return False
        price = getattr(quote, 'current_price', None)
        if price is None or price <= 0:
            return False
        # NaN check
        if price != price:
            return False
        # Extreme value check
        prev = getattr(quote, 'previous_close', None)
        if prev and prev > 0:
            ratio = price / prev
            if ratio > 5 or ratio < 0.2:
                logger.warning(f"Extreme price change for {getattr(quote, 'symbol', '?')}: {ratio:.2f}x")
                return False
        return True

    async def get_quote(self, symbol: str) -> Optional[StockData]:
        """Get quote with automatic fallback across providers"""
        for provider in self.providers:
            try:
                logger.info(f"Trying {provider.name} for quote: {symbol}")
                quote = await provider.get_quote(symbol)
                if quote and self._validate_quote(quote):
                    logger.info(f"Successfully fetched quote from {provider.name}")
                    return quote
                elif quote:
                    logger.warning(f"{provider.name} returned invalid data, trying next")
                    continue
            except Exception as e:
                logger.warning(f"{provider.name} failed for {symbol}: {e}")
                continue

        logger.error(f"All providers failed to fetch quote for {symbol}")
        return None

    async def get_historical_data(
        self,
        symbol: str,
        period: str = "6mo",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Get historical data with automatic fallback"""
        for provider in self.providers:
            try:
                logger.info(f"Trying {provider.name} for historical data: {symbol}")
                data = await provider.get_historical_data(symbol, period, interval)
                if data is not None and len(data) >= 10:
                    logger.info(f"Successfully fetched {len(data)} bars from {provider.name}")
                    return data
            except Exception as e:
                logger.warning(f"{provider.name} failed for {symbol}: {e}")
                continue

        logger.error(f"All providers failed to fetch historical data for {symbol}")
        return None

    def get_provider_status(self) -> List[Dict[str, Any]]:
        """Get status of all providers"""
        return [
            {
                "name": provider.name,
                "available": provider.is_available,
                "has_api_key": provider.api_key is not None,
                "rate_limits": provider.get_rate_limit_info()
            }
            for provider in self.providers
        ]

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return [p.name for p in self.providers if p.is_available]


# Global instance
import threading
_manager_instance: Optional[DataProviderManager] = None
_manager_lock = threading.Lock()


def get_provider_manager(api_keys: Optional[Dict[str, Any]] = None) -> DataProviderManager:
    """Get or create global provider manager instance"""
    global _manager_instance
    with _manager_lock:
        if _manager_instance is None:
            _manager_instance = DataProviderManager(api_keys)
    return _manager_instance
