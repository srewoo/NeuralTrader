"""
Market Data Module
Provides access to Indian market indices data (Nifty, Sensex, etc.)
"""

from .indian_indices import get_indian_indices_data

__all__ = [
    'get_indian_indices_data'
]
