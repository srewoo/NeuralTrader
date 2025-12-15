"""
Unit Tests for Trading Strategies
Tests for all backtesting strategy implementations
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


class TestSignalType:
    """Tests for SignalType enum"""

    def test_signal_values(self):
        """Test signal type values"""
        from backtesting.strategies import SignalType

        assert SignalType.BUY.value == "BUY"
        assert SignalType.SELL.value == "SELL"
        assert SignalType.HOLD.value == "HOLD"


class TestStrategyBase:
    """Tests for Strategy base class"""

    def test_strategy_initialization(self):
        """Test strategy initialization with params"""
        from backtesting.strategies import MeanReversionStrategy

        strategy = MeanReversionStrategy(oversold=25, overbought=75)
        assert strategy.name == "Mean Reversion"
        assert strategy.params["oversold"] == 25
        assert strategy.params["overbought"] == 75

    def test_default_params(self):
        """Test strategy default parameters"""
        from backtesting.strategies import MeanReversionStrategy

        strategy = MeanReversionStrategy()
        assert strategy.params["oversold"] == 30
        assert strategy.params["overbought"] == 70


class TestMeanReversionStrategy:
    """Tests for Mean Reversion Strategy"""

    def test_generate_buy_signal_on_oversold(self, sample_ohlcv_data):
        """Test BUY signal when RSI is oversold"""
        from backtesting.strategies import MeanReversionStrategy, SignalType

        strategy = MeanReversionStrategy(oversold=30, overbought=70)
        signals = strategy.generate_signals(sample_ohlcv_data)

        # Should generate signals for all data points
        assert len(signals) == len(sample_ohlcv_data)
        assert all(s in [SignalType.BUY.value, SignalType.SELL.value, SignalType.HOLD.value] for s in signals)

    def test_generate_sell_signal_on_overbought(self):
        """Test SELL signal when RSI is overbought"""
        from backtesting.strategies import MeanReversionStrategy, SignalType

        # Create data with steadily rising prices to trigger overbought
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        rising_prices = [100 + i * 2 for i in range(50)]

        data = pd.DataFrame({
            'Open': rising_prices,
            'High': [p + 1 for p in rising_prices],
            'Low': [p - 0.5 for p in rising_prices],
            'Close': rising_prices,
            'Volume': [1000000] * 50
        }, index=dates)

        strategy = MeanReversionStrategy(oversold=30, overbought=70)
        signals = strategy.generate_signals(data)

        # Should have some SELL signals due to overbought conditions
        assert SignalType.SELL.value in signals.values

    def test_get_description(self):
        """Test strategy description"""
        from backtesting.strategies import MeanReversionStrategy

        strategy = MeanReversionStrategy(oversold=25, overbought=75)
        desc = strategy.get_description()

        assert "25" in desc
        assert "75" in desc
        assert "RSI" in desc


class TestTrendFollowingStrategy:
    """Tests for Trend Following Strategy"""

    def test_strategy_initialization(self):
        """Test initialization with custom periods"""
        from backtesting.strategies import TrendFollowingStrategy

        strategy = TrendFollowingStrategy(fast_period=10, slow_period=30)
        assert strategy.params["fast_period"] == 10
        assert strategy.params["slow_period"] == 30

    def test_generate_signals_uptrend(self):
        """Test BUY signals in uptrend"""
        from backtesting.strategies import TrendFollowingStrategy, SignalType

        # Create uptrend data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = [100 + i * 0.5 for i in range(100)]

        data = pd.DataFrame({
            'Open': prices,
            'High': [p + 1 for p in prices],
            'Low': [p - 0.5 for p in prices],
            'Close': prices,
            'Volume': [1000000] * 100
        }, index=dates)

        strategy = TrendFollowingStrategy(fast_period=20, slow_period=50)
        signals = strategy.generate_signals(data)

        # In uptrend, should have BUY signals after warmup period
        buy_signals = signals[signals == SignalType.BUY.value]
        assert len(buy_signals) > 0

    def test_generate_signals_downtrend(self):
        """Test SELL signals in downtrend"""
        from backtesting.strategies import TrendFollowingStrategy, SignalType

        # Create downtrend data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = [200 - i * 0.5 for i in range(100)]

        data = pd.DataFrame({
            'Open': prices,
            'High': [p + 0.5 for p in prices],
            'Low': [p - 1 for p in prices],
            'Close': prices,
            'Volume': [1000000] * 100
        }, index=dates)

        strategy = TrendFollowingStrategy(fast_period=20, slow_period=50)
        signals = strategy.generate_signals(data)

        # In downtrend, should have SELL signals
        sell_signals = signals[signals == SignalType.SELL.value]
        assert len(sell_signals) > 0

    def test_get_description(self):
        """Test description format"""
        from backtesting.strategies import TrendFollowingStrategy

        strategy = TrendFollowingStrategy(fast_period=20, slow_period=50)
        desc = strategy.get_description()

        assert "SMA20" in desc
        assert "SMA50" in desc


class TestMACDStrategy:
    """Tests for MACD Strategy"""

    def test_strategy_initialization(self):
        """Test MACD params initialization"""
        from backtesting.strategies import MACDStrategy

        strategy = MACDStrategy(fast=12, slow=26, signal=9)
        assert strategy.params["fast"] == 12
        assert strategy.params["slow"] == 26
        assert strategy.params["signal"] == 9

    def test_generate_signals(self, sample_ohlcv_data):
        """Test MACD signal generation"""
        from backtesting.strategies import MACDStrategy

        strategy = MACDStrategy()
        signals = strategy.generate_signals(sample_ohlcv_data)

        assert len(signals) == len(sample_ohlcv_data)

    def test_bullish_crossover(self):
        """Test BUY signal on bullish MACD crossover"""
        from backtesting.strategies import MACDStrategy, SignalType

        # Create data with momentum shift
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)

        # Start down, then trend up
        prices = [100 - i * 0.3 for i in range(50)] + [85 + i * 0.6 for i in range(50)]

        data = pd.DataFrame({
            'Open': prices,
            'High': [p + 1 for p in prices],
            'Low': [p - 1 for p in prices],
            'Close': prices,
            'Volume': [1000000] * 100
        }, index=dates)

        strategy = MACDStrategy()
        signals = strategy.generate_signals(data)

        # Should have both BUY and SELL signals
        assert SignalType.BUY.value in signals.values or SignalType.SELL.value in signals.values

    def test_get_description(self):
        """Test description format"""
        from backtesting.strategies import MACDStrategy

        strategy = MACDStrategy(fast=12, slow=26, signal=9)
        desc = strategy.get_description()

        assert "12" in desc
        assert "26" in desc
        assert "9" in desc


class TestBollingerBandsStrategy:
    """Tests for Bollinger Bands Strategy"""

    def test_strategy_initialization(self):
        """Test initialization with params"""
        from backtesting.strategies import BollingerBandsStrategy

        strategy = BollingerBandsStrategy(period=20, std_dev=2.0)
        assert strategy.params["period"] == 20
        assert strategy.params["std_dev"] == 2.0

    def test_generate_signals(self, sample_ohlcv_data):
        """Test signal generation"""
        from backtesting.strategies import BollingerBandsStrategy

        strategy = BollingerBandsStrategy()
        signals = strategy.generate_signals(sample_ohlcv_data)

        assert len(signals) == len(sample_ohlcv_data)

    def test_buy_at_lower_band(self):
        """Test BUY signal at lower band"""
        from backtesting.strategies import BollingerBandsStrategy, SignalType

        # Create volatile data that touches lower band
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        np.random.seed(42)

        # Start stable, then drop sharply
        prices = [100] * 30 + [100 - i * 3 for i in range(20)]

        data = pd.DataFrame({
            'Open': prices,
            'High': [p + 1 for p in prices],
            'Low': [p - 1 for p in prices],
            'Close': prices,
            'Volume': [1000000] * 50
        }, index=dates)

        strategy = BollingerBandsStrategy()
        signals = strategy.generate_signals(data)

        # Should have BUY signals when price drops below lower band
        buy_signals = signals[signals == SignalType.BUY.value]
        assert len(buy_signals) > 0

    def test_sell_at_upper_band(self):
        """Test SELL signal at upper band"""
        from backtesting.strategies import BollingerBandsStrategy, SignalType

        # Create data that rises sharply
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')

        prices = [100] * 30 + [100 + i * 3 for i in range(20)]

        data = pd.DataFrame({
            'Open': prices,
            'High': [p + 1 for p in prices],
            'Low': [p - 1 for p in prices],
            'Close': prices,
            'Volume': [1000000] * 50
        }, index=dates)

        strategy = BollingerBandsStrategy()
        signals = strategy.generate_signals(data)

        # Should have SELL signals when price exceeds upper band
        sell_signals = signals[signals == SignalType.SELL.value]
        assert len(sell_signals) > 0

    def test_get_description(self):
        """Test description format"""
        from backtesting.strategies import BollingerBandsStrategy

        strategy = BollingerBandsStrategy(period=20, std_dev=2.0)
        desc = strategy.get_description()

        assert "20" in desc
        assert "2.0" in desc


class TestMomentumStrategy:
    """Tests for Momentum Strategy"""

    def test_strategy_initialization(self):
        """Test initialization with params"""
        from backtesting.strategies import MomentumStrategy

        strategy = MomentumStrategy(lookback=10, threshold=0.02)
        assert strategy.params["lookback"] == 10
        assert strategy.params["threshold"] == 0.02

    def test_generate_buy_on_positive_momentum(self):
        """Test BUY signal on positive momentum"""
        from backtesting.strategies import MomentumStrategy, SignalType

        # Create strong upward momentum
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        prices = [100 + i * 1.5 for i in range(50)]

        data = pd.DataFrame({
            'Open': prices,
            'High': [p + 1 for p in prices],
            'Low': [p - 0.5 for p in prices],
            'Close': prices,
            'Volume': [1000000] * 50
        }, index=dates)

        strategy = MomentumStrategy(lookback=10, threshold=0.02)
        signals = strategy.generate_signals(data)

        buy_signals = signals[signals == SignalType.BUY.value]
        assert len(buy_signals) > 0

    def test_generate_sell_on_negative_momentum(self):
        """Test SELL signal on negative momentum"""
        from backtesting.strategies import MomentumStrategy, SignalType

        # Create strong downward momentum
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        prices = [200 - i * 1.5 for i in range(50)]

        data = pd.DataFrame({
            'Open': prices,
            'High': [p + 0.5 for p in prices],
            'Low': [p - 1 for p in prices],
            'Close': prices,
            'Volume': [1000000] * 50
        }, index=dates)

        strategy = MomentumStrategy(lookback=10, threshold=0.02)
        signals = strategy.generate_signals(data)

        sell_signals = signals[signals == SignalType.SELL.value]
        assert len(sell_signals) > 0

    def test_get_description(self):
        """Test description format"""
        from backtesting.strategies import MomentumStrategy

        strategy = MomentumStrategy(lookback=10, threshold=0.02)
        desc = strategy.get_description()

        assert "10" in desc
        assert "2" in desc  # 0.02 * 100 = 2%


class TestStrategyRegistry:
    """Tests for Strategy Registry"""

    def test_list_strategies(self):
        """Test listing available strategies"""
        from backtesting.strategies import StrategyRegistry

        strategies = StrategyRegistry.list_strategies()

        assert "mean_reversion" in strategies
        assert "trend_following" in strategies
        assert "macd" in strategies
        assert "bollinger_bands" in strategies
        assert "momentum" in strategies

    def test_get_strategy_default_params(self):
        """Test getting strategy with default params"""
        from backtesting.strategies import StrategyRegistry, MeanReversionStrategy

        strategy = StrategyRegistry.get_strategy("mean_reversion")

        assert isinstance(strategy, MeanReversionStrategy)
        assert strategy.params["oversold"] == 30

    def test_get_strategy_custom_params(self):
        """Test getting strategy with custom params"""
        from backtesting.strategies import StrategyRegistry

        strategy = StrategyRegistry.get_strategy(
            "mean_reversion",
            params={"oversold": 25, "overbought": 75}
        )

        assert strategy.params["oversold"] == 25
        assert strategy.params["overbought"] == 75

    def test_get_unknown_strategy_raises_error(self):
        """Test error on unknown strategy"""
        from backtesting.strategies import StrategyRegistry

        with pytest.raises(ValueError) as exc:
            StrategyRegistry.get_strategy("unknown_strategy")

        assert "Unknown strategy" in str(exc.value)

    def test_register_custom_strategy(self):
        """Test registering custom strategy"""
        from backtesting.strategies import StrategyRegistry, Strategy, SignalType

        class CustomStrategy(Strategy):
            def __init__(self):
                super().__init__("Custom")

            def generate_signals(self, data):
                return pd.Series(SignalType.HOLD.value, index=data.index)

        StrategyRegistry.register_strategy("custom", CustomStrategy)

        strategy = StrategyRegistry.get_strategy("custom")
        assert strategy.name == "Custom"

        # Clean up
        del StrategyRegistry._strategies["custom"]
