"""
Data Provider Manager
Orchestrates data fetching across multiple providers with automatic fallback
"""

from typing import Optional, List, Dict, Any
import pandas as pd
from .base_provider import BaseDataProvider, StockData
from .yfinance_provider import YFinanceProvider
from .finnhub_provider import FinnhubProvider
from .alpaca_provider import AlpacaProvider
from .fmp_provider import FMPProvider
import logging

logger = logging.getLogger(__name__)


class DataProviderManager:
    """
    Manages multiple data providers with automatic fallback
    Tries providers in priority order until successful
    """

    def __init__(self, api_keys: Optional[Dict[str, Any]] = None):
        """
        Initialize provider manager with API keys

        Args:
            api_keys: Dictionary with API keys for different providers:
                {
                    "finnhub": "your_key",
                    "alpaca": {"key": "...", "secret": "..."},
                    "fmp": "your_key"
                }
        """
        api_keys = api_keys or {}

        # Initialize all providers
        self.providers: List[BaseDataProvider] = []

        # Finnhub (best for Indian stocks if API key available)
        finnhub_key = api_keys.get("finnhub")
        if finnhub_key:
            finnhub = FinnhubProvider(api_key=finnhub_key)
            if finnhub.is_available:
                self.providers.append(finnhub)
                logger.info("Finnhub provider initialized (60 calls/min)")

        # YFinance (always available, no API key needed)
        yfinance = YFinanceProvider()
        self.providers.append(yfinance)
        logger.info("Yahoo Finance provider initialized (free, no API key)")

        # Alpaca (good for US stocks)
        alpaca_creds = api_keys.get("alpaca", {})
        if isinstance(alpaca_creds, dict):
            alpaca = AlpacaProvider(
                api_key=alpaca_creds.get("key"),
                api_secret=alpaca_creds.get("secret")
            )
            if alpaca.is_available:
                self.providers.append(alpaca)
                logger.info("Alpaca provider initialized (200 calls/min)")

        # FMP (good for fundamentals)
        fmp_key = api_keys.get("fmp")
        if fmp_key:
            fmp = FMPProvider(api_key=fmp_key)
            if fmp.is_available:
                self.providers.append(fmp)
                logger.info("FMP provider initialized (250 calls/day)")

        logger.info(f"Initialized {len(self.providers)} data providers")

    async def get_quote(self, symbol: str) -> Optional[StockData]:
        """
        Get quote with automatic fallback across providers

        Args:
            symbol: Stock symbol

        Returns:
            StockData or None if all providers failed
        """
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
        """
        Get historical data with automatic fallback

        Args:
            symbol: Stock symbol
            period: Time period
            interval: Data interval

        Returns:
            DataFrame or None if all providers failed
        """
        for provider in self.providers:
            try:
                logger.info(f"Trying {provider.name} for historical data: {symbol}")
                data = await provider.get_historical_data(symbol, period, interval)

                if data is not None and len(data) >= 10:
                    logger.info(
                        f"Successfully fetched {len(data)} bars from {provider.name}"
                    )
                    return data

            except Exception as e:
                logger.warning(f"{provider.name} failed for {symbol}: {e}")
                continue

        logger.error(f"All providers failed to fetch historical data for {symbol}")
        return None

    def get_provider_status(self) -> List[Dict[str, Any]]:
        """
        Get status of all providers

        Returns:
            List of provider status dictionaries
        """
        status = []
        for provider in self.providers:
            rate_limits = provider.get_rate_limit_info()
            status.append({
                "name": provider.name,
                "available": provider.is_available,
                "has_api_key": provider.api_key is not None,
                "rate_limits": rate_limits
            })

        return status

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return [p.name for p in self.providers if p.is_available]

    def update_api_keys(self, api_keys: Dict[str, Any]):
        """
        Update API keys and reinitialize providers

        Args:
            api_keys: New API keys dictionary
        """
        logger.info("Updating API keys and reinitializing providers")
        self.__init__(api_keys)


# Global instance (will be initialized by server with user's API keys)
_manager_instance: Optional[DataProviderManager] = None


def get_provider_manager(api_keys: Optional[Dict[str, Any]] = None) -> DataProviderManager:
    """
    Get or create global provider manager instance

    Args:
        api_keys: Optional API keys to initialize/update manager

    Returns:
        DataProviderManager instance
    """
    global _manager_instance

    if _manager_instance is None or api_keys is not None:
        _manager_instance = DataProviderManager(api_keys)

    return _manager_instance
