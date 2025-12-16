"""
Backtesting System
Allows testing trading strategies against historical data
"""

from .engine import BacktestEngine
from .strategies import Strategy, StrategyRegistry
from .metrics import PerformanceMetrics
from .price_cache import PriceCache
from .walk_forward import WalkForwardEngine, IndianMarketCosts, get_walk_forward_engine
from .monte_carlo import MonteCarloSimulator, MonteCarloConfig, ParameterOptimizer, get_monte_carlo_simulator
from .portfolio import PortfolioBacktester, CorrelationAnalyzer, get_portfolio_backtester, get_correlation_analyzer
from .regime_tester import RegimeStrategyTester, RegimeDetector, MarketRegime, get_regime_tester
from .custom_indicators import CustomIndicatorBuilder, TechnicalIndicators, get_indicator_builder

__all__ = [
    'BacktestEngine',
    'Strategy',
    'StrategyRegistry',
    'PerformanceMetrics',
    'PriceCache',
    # Walk-forward testing
    'WalkForwardEngine',
    'IndianMarketCosts',
    'get_walk_forward_engine',
    # Monte Carlo
    'MonteCarloSimulator',
    'MonteCarloConfig',
    'ParameterOptimizer',
    'get_monte_carlo_simulator',
    # Portfolio
    'PortfolioBacktester',
    'CorrelationAnalyzer',
    'get_portfolio_backtester',
    'get_correlation_analyzer',
    # Regime testing
    'RegimeStrategyTester',
    'RegimeDetector',
    'MarketRegime',
    'get_regime_tester',
    # Custom indicators
    'CustomIndicatorBuilder',
    'TechnicalIndicators',
    'get_indicator_builder'
]

