"""
Base Data Provider Interface
Defines the contract that all data providers must implement
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime


class StockData:
    """Standardized stock data structure"""

    def __init__(
        self,
        symbol: str,
        name: str,
        current_price: float,
        previous_close: float,
        volume: int,
        market_cap: Optional[float] = None,
        pe_ratio: Optional[float] = None,
        week_52_high: Optional[float] = None,
        week_52_low: Optional[float] = None,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        provider: str = "unknown"
    ):
        self.symbol = symbol
        self.name = name
        self.current_price = current_price
        self.previous_close = previous_close
        self.volume = volume
        self.market_cap = market_cap
        self.pe_ratio = pe_ratio
        self.week_52_high = week_52_high
        self.week_52_low = week_52_low
        self.sector = sector
        self.industry = industry
        self.provider = provider

        # Calculated fields
        self.change = current_price - previous_close
        self.change_percent = (self.change / previous_close * 100) if previous_close else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "symbol": self.symbol,
            "ticker_symbol": self.symbol,
            "data_source": f"{self.provider} API",
            "name": self.name,
            "current_price": self.current_price,
            "previous_close": self.previous_close,
            "change": self.change,
            "change_percent": self.change_percent,
            "volume": self.volume,
            "market_cap": self.market_cap,
            "pe_ratio": self.pe_ratio,
            "week_52_high": self.week_52_high,
            "week_52_low": self.week_52_low,
            "sector": self.sector,
            "industry": self.industry
        }


class BaseDataProvider(ABC):
    """
    Abstract base class for all data providers
    Each provider must implement these methods
    """

    def __init__(self, name: str, api_key: Optional[str] = None):
        self.name = name
        self.api_key = api_key
        self.is_available = self._check_availability()

    @abstractmethod
    def _check_availability(self) -> bool:
        """
        Check if the provider is available (API key configured, service reachable, etc.)

        Returns:
            True if provider can be used, False otherwise
        """
        pass

    @abstractmethod
    async def get_quote(self, symbol: str) -> Optional[StockData]:
        """
        Get current quote/snapshot for a symbol

        Args:
            symbol: Stock symbol (without exchange suffix for Indian stocks)

        Returns:
            StockData object or None if failed
        """
        pass

    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        period: str = "6mo",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data

        Args:
            symbol: Stock symbol
            period: Time period (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y")
            interval: Data interval (e.g., "1m", "5m", "1h", "1d")

        Returns:
            DataFrame with columns [Open, High, Low, Close, Volume] or None if failed
        """
        pass

    @abstractmethod
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """
        Get information about rate limits for this provider

        Returns:
            Dictionary with rate limit info (calls_per_minute, calls_per_day, etc.)
        """
        pass

    def normalize_symbol(self, symbol: str, exchange: str = "NSE") -> str:
        """
        Normalize symbol for this provider's API
        Some APIs need specific formats (e.g., "RELIANCE.NS" vs "RELIANCE:NSE" vs "RELIANCE")

        Args:
            symbol: Base symbol
            exchange: Exchange code (NSE, BSE, etc.)

        Returns:
            Normalized symbol for this provider
        """
        # Default implementation - override in subclasses
        return symbol.upper().strip()
