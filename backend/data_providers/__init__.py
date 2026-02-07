"""
Data Providers Package
TVScreener (free) + yfinance fallback for Indian stock data
"""

from .base_provider import BaseDataProvider
from .yfinance_provider import YFinanceProvider
from .provider_manager import DataProviderManager

__all__ = [
    'BaseDataProvider',
    'YFinanceProvider',
    'DataProviderManager'
]
