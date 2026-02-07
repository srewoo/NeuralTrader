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

        # YFinance (always available, no API key needed)
        yfinance = YFinanceProvider()
        self.providers.append(yfinance)
        logger.info(f"Initialized {len(self.providers)} data providers (yfinance)")

    async def get_quote(self, symbol: str) -> Optional[StockData]:
        """Get quote with automatic fallback across providers"""
        for provider in self.providers:
            try:
                logger.info(f"Trying {provider.name} for quote: {symbol}")
                quote = await provider.get_quote(symbol)
                if quote:
                    logger.info(f"Successfully fetched quote from {provider.name}")
                    return quote
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
