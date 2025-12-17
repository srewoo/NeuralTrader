"""
Percentile-based scoring system for technical indicators
Provides context by comparing current values to historical distribution
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class PercentileScorer:
    """
    Calculate percentile ranks for technical indicators to provide historical context
    Based on best practices from ai-stock-dashboard repository
    """

    def __init__(self):
        self.lookback_period = "6mo"  # 6-month window for percentile calculation

    def calculate_percentiles(
        self,
        hist: pd.DataFrame,
        current_indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate percentile scores for key metrics

        Args:
            hist: Historical price data (pandas DataFrame with OHLCV)
            current_indicators: Current technical indicators

        Returns:
            Dictionary with percentile scores and insights
        """
        try:
            close = hist['Close']
            high = hist['High']
            low = hist['Low']
            volume = hist['Volume']

            # Calculate historical RSI for percentile
            import ta
            rsi_series = ta.momentum.RSIIndicator(close, window=14).rsi()
            current_rsi = current_indicators.get('rsi')

            # Calculate historical ATR for volatility percentile
            atr_series = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
            current_atr = current_indicators.get('atr')

            # Volume percentile (rolling 20-day average)
            volume_ma = volume.rolling(20).mean()
            current_volume = volume.iloc[-1]

            # Calculate 10-day volatility series
            volatility_10d_series = close.pct_change().rolling(10).std() * 100
            current_vol_10d = current_indicators.get('volatility_10d')

            # Calculate percentiles
            percentiles = {}

            # RSI Percentile
            if current_rsi is not None and len(rsi_series.dropna()) > 0:
                rsi_percentile = self._calculate_percentile(rsi_series.dropna(), current_rsi)
                percentiles['rsi_percentile'] = round(rsi_percentile, 1)
                percentiles['rsi_interpretation'] = self._interpret_rsi_percentile(
                    current_rsi, rsi_percentile
                )
            else:
                percentiles['rsi_percentile'] = None
                percentiles['rsi_interpretation'] = "Insufficient data"

            # Volatility Percentile (using 10D volatility)
            if current_vol_10d is not None and len(volatility_10d_series.dropna()) > 0:
                vol_percentile = self._calculate_percentile(
                    volatility_10d_series.dropna(), current_vol_10d
                )
                percentiles['volatility_percentile'] = round(vol_percentile, 1)
                percentiles['volatility_interpretation'] = self._interpret_volatility_percentile(
                    current_vol_10d, vol_percentile
                )
            else:
                percentiles['volatility_percentile'] = None
                percentiles['volatility_interpretation'] = "Insufficient data"

            # Volume Percentile
            if len(volume_ma.dropna()) > 0:
                volume_percentile = self._calculate_percentile(volume_ma.dropna(), current_volume)
                percentiles['volume_percentile'] = round(volume_percentile, 1)
                percentiles['volume_interpretation'] = self._interpret_volume_percentile(
                    current_volume, volume_ma.iloc[-1], volume_percentile
                )
            else:
                percentiles['volume_percentile'] = None
                percentiles['volume_interpretation'] = "Insufficient data"

            # Price Position Percentile (relative to range)
            current_price = close.iloc[-1]
            price_range = close.max() - close.min()
            if price_range > 0:
                price_position = ((current_price - close.min()) / price_range) * 100
                percentiles['price_position_percentile'] = round(price_position, 1)
                percentiles['price_position_interpretation'] = self._interpret_price_position(
                    price_position
                )
            else:
                percentiles['price_position_percentile'] = None
                percentiles['price_position_interpretation'] = "Insufficient data"

            # Calculate composite score (0-100)
            composite_score = self._calculate_composite_score(percentiles, current_indicators)
            percentiles['composite_score'] = round(composite_score, 1)
            percentiles['composite_interpretation'] = self._interpret_composite_score(composite_score)

            return percentiles

        except Exception as e:
            # Return empty percentiles on error
            return {
                'rsi_percentile': None,
                'volatility_percentile': None,
                'volume_percentile': None,
                'price_position_percentile': None,
                'composite_score': 50,
                'error': str(e)
            }

    def _calculate_percentile(self, series: pd.Series, current_value: float) -> float:
        """
        Calculate percentile rank of current value in historical series

        Args:
            series: Historical series
            current_value: Current value to rank

        Returns:
            Percentile (0-100)
        """
        if len(series) == 0:
            return 50.0

        # Count how many values are less than current
        count_below = (series < current_value).sum()
        percentile = (count_below / len(series)) * 100

        return percentile

    def _interpret_rsi_percentile(self, rsi: float, percentile: float) -> str:
        """Generate natural language interpretation of RSI percentile"""
        if percentile < 20:
            return f"RSI at {rsi:.1f} is in the {percentile:.0f}th percentile - extremely oversold historically"
        elif percentile < 40:
            return f"RSI at {rsi:.1f} is in the {percentile:.0f}th percentile - below average, oversold territory"
        elif percentile < 60:
            return f"RSI at {rsi:.1f} is in the {percentile:.0f}th percentile - neutral zone"
        elif percentile < 80:
            return f"RSI at {rsi:.1f} is in the {percentile:.0f}th percentile - above average, overbought territory"
        else:
            return f"RSI at {rsi:.1f} is in the {percentile:.0f}th percentile - extremely overbought historically"

    def _interpret_volatility_percentile(self, volatility: float, percentile: float) -> str:
        """Generate natural language interpretation of volatility percentile"""
        if percentile < 20:
            return f"10D volatility at {volatility:.2f}% is in the {percentile:.0f}th percentile - extremely low volatility, tight range"
        elif percentile < 40:
            return f"10D volatility at {volatility:.2f}% is in the {percentile:.0f}th percentile - below average volatility"
        elif percentile < 60:
            return f"10D volatility at {volatility:.2f}% is in the {percentile:.0f}th percentile - normal volatility range"
        elif percentile < 80:
            return f"10D volatility at {volatility:.2f}% is in the {percentile:.0f}th percentile - elevated volatility"
        else:
            return f"10D volatility at {volatility:.2f}% is in the {percentile:.0f}th percentile - extreme volatility, high risk"

    def _interpret_volume_percentile(
        self, current_volume: float, avg_volume: float, percentile: float
    ) -> str:
        """Generate natural language interpretation of volume percentile"""
        ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        if percentile < 20:
            return f"Volume at {ratio:.2f}x average is in the {percentile:.0f}th percentile - very low participation"
        elif percentile < 40:
            return f"Volume at {ratio:.2f}x average is in the {percentile:.0f}th percentile - below average participation"
        elif percentile < 60:
            return f"Volume at {ratio:.2f}x average is in the {percentile:.0f}th percentile - normal participation"
        elif percentile < 80:
            return f"Volume at {ratio:.2f}x average is in the {percentile:.0f}th percentile - strong participation, accumulation"
        else:
            return f"Volume at {ratio:.2f}x average is in the {percentile:.0f}th percentile - exceptional volume, institutional activity"

    def _interpret_price_position(self, position: float) -> str:
        """Generate natural language interpretation of price position in 6-month range"""
        if position < 20:
            return f"Price at {position:.0f}th percentile of 6-month range - near lows, potential support"
        elif position < 40:
            return f"Price at {position:.0f}th percentile of 6-month range - lower third, consolidation zone"
        elif position < 60:
            return f"Price at {position:.0f}th percentile of 6-month range - mid-range, neutral"
        elif position < 80:
            return f"Price at {position:.0f}th percentile of 6-month range - upper third, strength"
        else:
            return f"Price at {position:.0f}th percentile of 6-month range - near highs, potential resistance"

    def _calculate_composite_score(
        self, percentiles: Dict[str, Any], indicators: Dict[str, Any]
    ) -> float:
        """
        Calculate composite score (0-100) based on multiple factors
        Higher score = more bullish setup

        Args:
            percentiles: Percentile data
            indicators: Technical indicators

        Returns:
            Composite score 0-100
        """
        score = 50.0  # Start neutral

        # RSI component (30% weight)
        rsi_percentile = percentiles.get('rsi_percentile')
        if rsi_percentile is not None:
            # Favor oversold (low percentile) for buying opportunities
            if rsi_percentile < 30:
                score += 15  # Oversold bonus
            elif rsi_percentile > 70:
                score -= 15  # Overbought penalty

        # Volume component (25% weight)
        volume_percentile = percentiles.get('volume_percentile')
        if volume_percentile is not None:
            # Higher volume is bullish if price is rising
            return_1d = indicators.get('return_1d', 0)
            if volume_percentile > 60 and return_1d > 0:
                score += 12  # Strong volume + positive return
            elif volume_percentile < 40:
                score -= 8  # Weak volume

        # Trend component (25% weight)
        price_vs_sma20 = indicators.get('price_vs_sma20_pct', 0)
        price_vs_sma50 = indicators.get('price_vs_sma50_pct', 0)
        if price_vs_sma20 > 0 and price_vs_sma50 > 0:
            score += 12  # Above both SMAs
        elif price_vs_sma20 < 0 and price_vs_sma50 < 0:
            score -= 12  # Below both SMAs

        # Momentum component (20% weight)
        macd_hist = indicators.get('macd_histogram', 0)
        if macd_hist > 0:
            score += 10  # Positive momentum
        else:
            score -= 10  # Negative momentum

        # Volatility adjustment
        vol_percentile = percentiles.get('volatility_percentile')
        if vol_percentile is not None and vol_percentile > 80:
            score -= 5  # High volatility risk penalty

        # Clamp to 0-100 range
        return max(0, min(100, score))

    def _interpret_composite_score(self, score: float) -> str:
        """Generate interpretation of composite score"""
        if score < 20:
            return "Extremely bearish setup - Strong sell signals across multiple indicators"
        elif score < 40:
            return "Bearish setup - Multiple negative factors, consider reducing exposure"
        elif score < 60:
            return "Neutral setup - Mixed signals, wait for clearer direction"
        elif score < 80:
            return "Bullish setup - Multiple positive factors align, favorable for entry"
        else:
            return "Extremely bullish setup - Strong buy signals across multiple indicators"
