"""
Custom Indicator Builder
Build and combine technical indicators for custom strategies.
Supports:
- Standard indicators (SMA, EMA, RSI, MACD, etc.)
- Custom formulas using existing indicators
- Indicator combinations and conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class IndicatorType(Enum):
    """Types of indicators"""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    CUSTOM = "custom"


@dataclass
class IndicatorDefinition:
    """Definition of a technical indicator"""
    name: str
    indicator_type: IndicatorType
    parameters: Dict[str, Any]
    description: str


class TechnicalIndicators:
    """Library of standard technical indicators"""

    @staticmethod
    def sma(data: pd.Series, period: int = 20) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data: pd.Series, period: int = 20) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def wma(data: pd.Series, period: int = 20) -> pd.Series:
        """Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        return data.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD - Moving Average Convergence Divergence"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return {
            'upper': upper,
            'middle': sma,
            'lower': lower,
            'bandwidth': (upper - lower) / sma * 100,
            'percent_b': (data - lower) / (upper - lower)
        }

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """Average Directional Index"""
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        return {'k': k, 'd': d}

    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        return (typical_price - sma) / (0.015 * mad)

    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low)

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        direction = np.where(close > close.shift(1), 1,
                             np.where(close < close.shift(1), -1, 0))
        return (volume * direction).cumsum()

    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()

    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series,
            volume: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi

    @staticmethod
    def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                   period: int = 10, multiplier: float = 3.0) -> Dict[str, pd.Series]:
        """Supertrend Indicator"""
        hl2 = (high + low) / 2

        # ATR
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Upper and Lower bands
        upper_basic = hl2 + (multiplier * atr)
        lower_basic = hl2 - (multiplier * atr)

        upper_band = upper_basic.copy()
        lower_band = lower_basic.copy()
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)

        for i in range(period, len(close)):
            if close.iloc[i - 1] <= upper_band.iloc[i - 1]:
                upper_band.iloc[i] = min(upper_basic.iloc[i], upper_band.iloc[i - 1])
            else:
                upper_band.iloc[i] = upper_basic.iloc[i]

            if close.iloc[i - 1] >= lower_band.iloc[i - 1]:
                lower_band.iloc[i] = max(lower_basic.iloc[i], lower_band.iloc[i - 1])
            else:
                lower_band.iloc[i] = lower_basic.iloc[i]

            if close.iloc[i] > upper_band.iloc[i]:
                direction.iloc[i] = 1
                supertrend.iloc[i] = lower_band.iloc[i]
            elif close.iloc[i] < lower_band.iloc[i]:
                direction.iloc[i] = -1
                supertrend.iloc[i] = upper_band.iloc[i]
            else:
                direction.iloc[i] = direction.iloc[i - 1] if i > period else 1
                supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]

        return {
            'supertrend': supertrend,
            'direction': direction,
            'upper_band': upper_band,
            'lower_band': lower_band
        }

    @staticmethod
    def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """Pivot Points (Standard)"""
        pivot = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
        r1 = 2 * pivot - low.shift(1)
        s1 = 2 * pivot - high.shift(1)
        r2 = pivot + (high.shift(1) - low.shift(1))
        s2 = pivot - (high.shift(1) - low.shift(1))
        r3 = high.shift(1) + 2 * (pivot - low.shift(1))
        s3 = low.shift(1) - 2 * (high.shift(1) - pivot)

        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }


class CustomIndicatorBuilder:
    """
    Build custom indicators from combinations of standard indicators.
    Supports:
    - Arithmetic operations (+, -, *, /)
    - Comparisons (>, <, ==, >=, <=)
    - Logical operations (and, or)
    - Custom formulas
    """

    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.cached_indicators: Dict[str, pd.Series] = {}

    def calculate_indicator(
        self,
        data: pd.DataFrame,
        indicator_name: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Union[pd.Series, Dict[str, pd.Series]]:
        """
        Calculate a standard indicator.

        Args:
            data: OHLCV DataFrame
            indicator_name: Name of the indicator
            params: Optional parameters

        Returns:
            Indicator values as Series or Dict of Series
        """
        params = params or {}

        indicator_map = {
            'sma': lambda: self.indicators.sma(data['Close'], params.get('period', 20)),
            'ema': lambda: self.indicators.ema(data['Close'], params.get('period', 20)),
            'wma': lambda: self.indicators.wma(data['Close'], params.get('period', 20)),
            'rsi': lambda: self.indicators.rsi(data['Close'], params.get('period', 14)),
            'macd': lambda: self.indicators.macd(
                data['Close'],
                params.get('fast', 12),
                params.get('slow', 26),
                params.get('signal', 9)
            ),
            'bollinger': lambda: self.indicators.bollinger_bands(
                data['Close'],
                params.get('period', 20),
                params.get('std_dev', 2.0)
            ),
            'atr': lambda: self.indicators.atr(
                data['High'], data['Low'], data['Close'],
                params.get('period', 14)
            ),
            'adx': lambda: self.indicators.adx(
                data['High'], data['Low'], data['Close'],
                params.get('period', 14)
            ),
            'stochastic': lambda: self.indicators.stochastic(
                data['High'], data['Low'], data['Close'],
                params.get('k_period', 14),
                params.get('d_period', 3)
            ),
            'cci': lambda: self.indicators.cci(
                data['High'], data['Low'], data['Close'],
                params.get('period', 20)
            ),
            'williams_r': lambda: self.indicators.williams_r(
                data['High'], data['Low'], data['Close'],
                params.get('period', 14)
            ),
            'obv': lambda: self.indicators.obv(data['Close'], data['Volume']),
            'vwap': lambda: self.indicators.vwap(
                data['High'], data['Low'], data['Close'], data['Volume']
            ),
            'mfi': lambda: self.indicators.mfi(
                data['High'], data['Low'], data['Close'], data['Volume'],
                params.get('period', 14)
            ),
            'supertrend': lambda: self.indicators.supertrend(
                data['High'], data['Low'], data['Close'],
                params.get('period', 10),
                params.get('multiplier', 3.0)
            ),
            'pivot_points': lambda: self.indicators.pivot_points(
                data['High'], data['Low'], data['Close']
            )
        }

        if indicator_name.lower() not in indicator_map:
            raise ValueError(f"Unknown indicator: {indicator_name}")

        return indicator_map[indicator_name.lower()]()

    def build_custom_indicator(
        self,
        data: pd.DataFrame,
        formula: str,
        name: str = "custom"
    ) -> pd.Series:
        """
        Build a custom indicator from a formula.

        Formula syntax:
        - Indicators: sma(20), ema(50), rsi(14), etc.
        - Price data: close, open, high, low, volume
        - Operations: +, -, *, /, >, <, ==, and, or
        - Functions: abs(), min(), max()

        Examples:
        - "sma(20) - sma(50)" - MA crossover
        - "rsi(14) > 70" - Overbought condition
        - "(close > sma(20)) and (rsi(14) < 30)" - Combined condition

        Args:
            data: OHLCV DataFrame
            formula: Custom formula string
            name: Name for the indicator

        Returns:
            Calculated indicator values
        """
        try:
            # Parse and calculate
            result = self._evaluate_formula(data, formula)

            # Cache result
            self.cached_indicators[name] = result

            return result

        except Exception as e:
            logger.error(f"Error building custom indicator: {e}")
            raise ValueError(f"Invalid formula: {formula}. Error: {e}")

    def _evaluate_formula(self, data: pd.DataFrame, formula: str) -> pd.Series:
        """Evaluate a custom formula"""

        # Create context with price data
        context = {
            'close': data['Close'],
            'open': data['Open'],
            'high': data['High'],
            'low': data['Low'],
            'volume': data['Volume'],
            'abs': np.abs,
            'min': np.minimum,
            'max': np.maximum
        }

        # Parse indicator calls like sma(20), rsi(14)
        indicator_pattern = r'(\w+)\(([^)]+)\)'
        matches = re.findall(indicator_pattern, formula)

        for match in matches:
            indicator_name = match[0].lower()
            param_str = match[1]

            # Parse parameters
            params = self._parse_params(param_str)

            # Calculate indicator
            try:
                result = self.calculate_indicator(data, indicator_name, params)

                # Handle multi-output indicators
                if isinstance(result, dict):
                    # Use first output by default
                    result = list(result.values())[0]

                # Add to context
                key = f"{indicator_name}_{param_str.replace(',', '_').replace(' ', '')}"
                context[key] = result

                # Replace in formula
                formula = formula.replace(f"{match[0]}({match[1]})", key)

            except ValueError:
                # Not a recognized indicator, might be a function call
                continue

        # Replace logical operators
        formula = formula.replace(' and ', ' & ')
        formula = formula.replace(' or ', ' | ')

        # Evaluate
        try:
            result = pd.eval(formula, local_dict=context)
            if isinstance(result, (bool, np.bool_)):
                result = pd.Series([result] * len(data), index=data.index)
            return result
        except Exception as e:
            raise ValueError(f"Cannot evaluate formula: {e}")

    def _parse_params(self, param_str: str) -> Dict[str, Any]:
        """Parse parameter string into dict"""
        params = {}
        parts = param_str.split(',')

        if len(parts) == 1 and '=' not in parts[0]:
            # Single numeric parameter
            try:
                params['period'] = int(parts[0].strip())
            except ValueError:
                try:
                    params['period'] = float(parts[0].strip())
                except ValueError:
                    pass
        else:
            # Named parameters
            for part in parts:
                if '=' in part:
                    key, value = part.split('=')
                    try:
                        params[key.strip()] = int(value.strip())
                    except ValueError:
                        try:
                            params[key.strip()] = float(value.strip())
                        except ValueError:
                            params[key.strip()] = value.strip()

        return params

    def create_signal_generator(
        self,
        buy_condition: str,
        sell_condition: str
    ) -> Callable[[pd.DataFrame], pd.Series]:
        """
        Create a signal generator from buy/sell conditions.

        Args:
            buy_condition: Formula that evaluates to True for buy signals
            sell_condition: Formula that evaluates to True for sell signals

        Returns:
            Function that generates signals from price data
        """
        def generate_signals(data: pd.DataFrame) -> pd.Series:
            buy_signals = self.build_custom_indicator(data, buy_condition, "buy_condition")
            sell_signals = self.build_custom_indicator(data, sell_condition, "sell_condition")

            signals = pd.Series(index=data.index, data=0)
            signals[buy_signals == True] = 1  # Buy
            signals[sell_signals == True] = -1  # Sell

            return signals

        return generate_signals

    def get_available_indicators(self) -> List[Dict[str, Any]]:
        """Get list of available indicators with descriptions"""
        return [
            {
                "name": "sma",
                "type": "trend",
                "params": ["period"],
                "description": "Simple Moving Average",
                "example": "sma(20)"
            },
            {
                "name": "ema",
                "type": "trend",
                "params": ["period"],
                "description": "Exponential Moving Average",
                "example": "ema(50)"
            },
            {
                "name": "wma",
                "type": "trend",
                "params": ["period"],
                "description": "Weighted Moving Average",
                "example": "wma(20)"
            },
            {
                "name": "rsi",
                "type": "momentum",
                "params": ["period"],
                "description": "Relative Strength Index",
                "example": "rsi(14)"
            },
            {
                "name": "macd",
                "type": "momentum",
                "params": ["fast", "slow", "signal"],
                "description": "Moving Average Convergence Divergence",
                "example": "macd(12, 26, 9)"
            },
            {
                "name": "bollinger",
                "type": "volatility",
                "params": ["period", "std_dev"],
                "description": "Bollinger Bands",
                "example": "bollinger(20, 2.0)"
            },
            {
                "name": "atr",
                "type": "volatility",
                "params": ["period"],
                "description": "Average True Range",
                "example": "atr(14)"
            },
            {
                "name": "adx",
                "type": "trend",
                "params": ["period"],
                "description": "Average Directional Index",
                "example": "adx(14)"
            },
            {
                "name": "stochastic",
                "type": "momentum",
                "params": ["k_period", "d_period"],
                "description": "Stochastic Oscillator",
                "example": "stochastic(14, 3)"
            },
            {
                "name": "cci",
                "type": "momentum",
                "params": ["period"],
                "description": "Commodity Channel Index",
                "example": "cci(20)"
            },
            {
                "name": "williams_r",
                "type": "momentum",
                "params": ["period"],
                "description": "Williams %R",
                "example": "williams_r(14)"
            },
            {
                "name": "obv",
                "type": "volume",
                "params": [],
                "description": "On Balance Volume",
                "example": "obv()"
            },
            {
                "name": "vwap",
                "type": "volume",
                "params": [],
                "description": "Volume Weighted Average Price",
                "example": "vwap()"
            },
            {
                "name": "mfi",
                "type": "volume",
                "params": ["period"],
                "description": "Money Flow Index",
                "example": "mfi(14)"
            },
            {
                "name": "supertrend",
                "type": "trend",
                "params": ["period", "multiplier"],
                "description": "Supertrend Indicator",
                "example": "supertrend(10, 3.0)"
            },
            {
                "name": "pivot_points",
                "type": "volatility",
                "params": [],
                "description": "Pivot Points (Support/Resistance)",
                "example": "pivot_points()"
            }
        ]

    def get_formula_examples(self) -> List[Dict[str, str]]:
        """Get example formulas for custom indicators"""
        return [
            {
                "name": "MA Crossover",
                "buy": "sma(20) > sma(50)",
                "sell": "sma(20) < sma(50)",
                "description": "Buy when short MA crosses above long MA"
            },
            {
                "name": "RSI Oversold/Overbought",
                "buy": "rsi(14) < 30",
                "sell": "rsi(14) > 70",
                "description": "Buy oversold, sell overbought"
            },
            {
                "name": "Bollinger Bounce",
                "buy": "close < bollinger(20, 2.0)",
                "sell": "close > bollinger(20, 2.0)",
                "description": "Buy at lower band, sell at upper band"
            },
            {
                "name": "MACD Signal Cross",
                "buy": "macd(12, 26, 9) > 0",
                "sell": "macd(12, 26, 9) < 0",
                "description": "Buy when MACD crosses above signal"
            },
            {
                "name": "Supertrend",
                "buy": "close > supertrend(10, 3.0)",
                "sell": "close < supertrend(10, 3.0)",
                "description": "Follow supertrend direction"
            },
            {
                "name": "Combined RSI + MA",
                "buy": "(rsi(14) < 30) and (close > sma(200))",
                "sell": "(rsi(14) > 70) or (close < sma(200))",
                "description": "RSI oversold in uptrend, exit on overbought or trend break"
            },
            {
                "name": "ADX Trend Filter",
                "buy": "(adx(14) > 25) and (close > ema(20))",
                "sell": "(adx(14) < 20) or (close < ema(20))",
                "description": "Enter strong trends, exit weak trends"
            },
            {
                "name": "Stochastic + RSI",
                "buy": "(stochastic(14, 3) < 20) and (rsi(14) < 30)",
                "sell": "(stochastic(14, 3) > 80) and (rsi(14) > 70)",
                "description": "Double confirmation on oversold/overbought"
            }
        ]


# Global instance
_indicator_builder = None


def get_indicator_builder() -> CustomIndicatorBuilder:
    """Get or create indicator builder instance"""
    global _indicator_builder
    if _indicator_builder is None:
        _indicator_builder = CustomIndicatorBuilder()
    return _indicator_builder
