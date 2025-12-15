"""
Backtesting System
Allows testing trading strategies against historical data
"""

from .engine import BacktestEngine
from .strategies import Strategy, StrategyRegistry
from .metrics import PerformanceMetrics
from .price_cache import PriceCache

__all__ = [
    'BacktestEngine',
    'Strategy',
    'StrategyRegistry',
    'PerformanceMetrics',
    'PriceCache'
]

