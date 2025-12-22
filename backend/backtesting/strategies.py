"""
Trading Strategy Framework
Defines base strategy class and built-in strategies
"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import pandas as pd
import ta
from enum import Enum


class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Strategy(ABC):
    """
    Abstract base class for trading strategies
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize strategy

        Args:
            name: Strategy name
            params: Strategy parameters
        """
        self.name = name
        self.params = params or {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from price data

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Series with signals (BUY/SELL/HOLD)
        """
        pass

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit strategy to training data (optional for strategies that need training)

        Args:
            data: Training data DataFrame

        Default implementation does nothing - override for adaptive strategies
        """
        pass

    def get_description(self) -> str:
        """Get strategy description"""
        return f"{self.name} strategy"


class MeanReversionStrategy(Strategy):
    """
    Mean Reversion Strategy
    Buy when RSI < oversold_threshold, Sell when RSI > overbought_threshold
    """
    
    def __init__(self, oversold: int = 30, overbought: int = 70, rsi_period: int = 14):
        super().__init__("Mean Reversion", {
            "oversold": oversold,
            "overbought": overbought,
            "rsi_period": rsi_period
        })
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on RSI"""
        # Calculate RSI
        rsi = ta.momentum.RSIIndicator(
            data['Close'],
            window=self.params['rsi_period']
        ).rsi()
        
        # Generate signals
        signals = pd.Series(SignalType.HOLD.value, index=data.index)
        signals[rsi < self.params['oversold']] = SignalType.BUY.value
        signals[rsi > self.params['overbought']] = SignalType.SELL.value
        
        return signals
    
    def get_description(self) -> str:
        return (f"Mean Reversion: Buy RSI<{self.params['oversold']}, "
                f"Sell RSI>{self.params['overbought']}")


class TrendFollowingStrategy(Strategy):
    """
    Trend Following Strategy
    Buy when SMA fast > SMA slow, Sell when SMA fast < SMA slow
    """
    
    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__("Trend Following", {
            "fast_period": fast_period,
            "slow_period": slow_period
        })
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on moving average crossover"""
        # Calculate moving averages
        sma_fast = ta.trend.SMAIndicator(
            data['Close'],
            window=self.params['fast_period']
        ).sma_indicator()
        
        sma_slow = ta.trend.SMAIndicator(
            data['Close'],
            window=self.params['slow_period']
        ).sma_indicator()
        
        # Generate signals
        signals = pd.Series(SignalType.HOLD.value, index=data.index)
        signals[sma_fast > sma_slow] = SignalType.BUY.value
        signals[sma_fast < sma_slow] = SignalType.SELL.value
        
        return signals
    
    def get_description(self) -> str:
        return (f"Trend Following: SMA{self.params['fast_period']} vs "
                f"SMA{self.params['slow_period']}")


class MACDStrategy(Strategy):
    """
    MACD Strategy
    Buy when MACD crosses above signal, Sell when MACD crosses below signal
    """
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__("MACD Crossover", {
            "fast": fast,
            "slow": slow,
            "signal": signal
        })
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on MACD crossover"""
        # Calculate MACD
        macd_indicator = ta.trend.MACD(
            data['Close'],
            window_slow=self.params['slow'],
            window_fast=self.params['fast'],
            window_sign=self.params['signal']
        )
        
        macd = macd_indicator.macd()
        signal_line = macd_indicator.macd_signal()
        
        # Generate signals
        signals = pd.Series(SignalType.HOLD.value, index=data.index)
        signals[macd > signal_line] = SignalType.BUY.value
        signals[macd < signal_line] = SignalType.SELL.value
        
        return signals
    
    def get_description(self) -> str:
        return f"MACD({self.params['fast']},{self.params['slow']},{self.params['signal']})"


class BollingerBandsStrategy(Strategy):
    """
    Bollinger Bands Strategy
    Buy when price touches lower band, Sell when price touches upper band
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__("Bollinger Bands", {
            "period": period,
            "std_dev": std_dev
        })
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on Bollinger Bands"""
        # Calculate Bollinger Bands
        bb = ta.volatility.BollingerBands(
            data['Close'],
            window=self.params['period'],
            window_dev=self.params['std_dev']
        )
        
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        
        # Generate signals
        signals = pd.Series(SignalType.HOLD.value, index=data.index)
        signals[data['Close'] <= bb_lower] = SignalType.BUY.value
        signals[data['Close'] >= bb_upper] = SignalType.SELL.value
        
        return signals
    
    def get_description(self) -> str:
        return f"Bollinger Bands({self.params['period']}, {self.params['std_dev']}σ)"


class MomentumStrategy(Strategy):
    """
    Momentum Strategy
    Buy when price momentum is positive and strong
    """
    
    def __init__(self, lookback: int = 10, threshold: float = 0.02):
        super().__init__("Momentum", {
            "lookback": lookback,
            "threshold": threshold
        })
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on price momentum"""
        # Calculate momentum (rate of change)
        momentum = data['Close'].pct_change(periods=self.params['lookback'])
        
        # Generate signals
        signals = pd.Series(SignalType.HOLD.value, index=data.index)
        signals[momentum > self.params['threshold']] = SignalType.BUY.value
        signals[momentum < -self.params['threshold']] = SignalType.SELL.value
        
        return signals
    
    def get_description(self) -> str:
        return f"Momentum({self.params['lookback']} days, {self.params['threshold']*100}%)"


class StrategyRegistry:
    """
    Registry of available strategies
    """
    
    _strategies = {
        "mean_reversion": MeanReversionStrategy,
        "trend_following": TrendFollowingStrategy,
        "macd": MACDStrategy,
        "bollinger_bands": BollingerBandsStrategy,
        "momentum": MomentumStrategy
    }
    
    @classmethod
    def get_strategy(cls, name: str, params: Optional[Dict[str, Any]] = None) -> Strategy:
        """
        Get strategy instance by name
        
        Args:
            name: Strategy name
            params: Strategy parameters
            
        Returns:
            Strategy instance
        """
        strategy_class = cls._strategies.get(name)
        if not strategy_class:
            raise ValueError(f"Unknown strategy: {name}")
        
        if params:
            return strategy_class(**params)
        return strategy_class()
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """Get list of available strategies"""
        return list(cls._strategies.keys())
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type):
        """
        Register a custom strategy
        
        Args:
            name: Strategy name
            strategy_class: Strategy class
        """
        cls._strategies[name] = strategy_class

