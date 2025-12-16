"""
Market Regime Detection Module
Identifies bull, bear, or sideways market conditions and adapts analysis accordingly.
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, Any, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    SIDEWAYS = "sideways"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class MarketRegimeDetector:
    """
    Detects market regime using multiple indicators:
    - Trend direction (SMA slopes)
    - Volatility regime (ATR, Bollinger Band width)
    - Momentum regime (ADX, RSI distribution)
    - Volume regime
    """

    def __init__(self):
        self.regime_cache = {}

    def detect_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect current market regime from price data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dict with regime classification and metrics
        """
        if df.empty or len(df) < 50:
            return {
                "primary_regime": MarketRegime.SIDEWAYS.value,
                "volatility_regime": "normal",
                "trend_strength": 0,
                "confidence": 0,
                "metrics": {}
            }

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # Calculate regime indicators
        trend_analysis = self._analyze_trend(close)
        volatility_analysis = self._analyze_volatility(high, low, close)
        momentum_analysis = self._analyze_momentum(high, low, close)
        volume_analysis = self._analyze_volume(volume)

        # Determine primary regime
        primary_regime = self._determine_primary_regime(
            trend_analysis, volatility_analysis, momentum_analysis
        )

        # Calculate confidence
        confidence = self._calculate_regime_confidence(
            trend_analysis, volatility_analysis, momentum_analysis
        )

        return {
            "primary_regime": primary_regime.value,
            "volatility_regime": volatility_analysis["regime"],
            "trend_strength": trend_analysis["strength"],
            "momentum_state": momentum_analysis["state"],
            "volume_regime": volume_analysis["regime"],
            "confidence": confidence,
            "metrics": {
                "sma_20_slope": trend_analysis["sma_20_slope"],
                "sma_50_slope": trend_analysis["sma_50_slope"],
                "adx": momentum_analysis["adx"],
                "atr_percentile": volatility_analysis["atr_percentile"],
                "bb_width": volatility_analysis["bb_width"],
                "volume_trend": volume_analysis["trend"]
            }
        }

    def _analyze_trend(self, close: pd.Series) -> Dict[str, Any]:
        """Analyze trend direction and strength"""
        # Calculate SMAs
        sma_20 = ta.trend.SMAIndicator(close, window=20).sma_indicator()
        sma_50 = ta.trend.SMAIndicator(close, window=50).sma_indicator()
        sma_200 = ta.trend.SMAIndicator(close, window=200).sma_indicator() if len(close) >= 200 else None

        # Calculate slopes (rate of change over 5 periods)
        sma_20_slope = ((sma_20.iloc[-1] / sma_20.iloc[-5]) - 1) * 100 if len(sma_20) >= 5 else 0
        sma_50_slope = ((sma_50.iloc[-1] / sma_50.iloc[-5]) - 1) * 100 if len(sma_50) >= 5 else 0

        # Price position relative to SMAs
        current_price = close.iloc[-1]
        above_20 = current_price > sma_20.iloc[-1]
        above_50 = current_price > sma_50.iloc[-1]
        above_200 = current_price > sma_200.iloc[-1] if sma_200 is not None else None

        # Trend alignment
        sma_20_above_50 = sma_20.iloc[-1] > sma_50.iloc[-1]

        # Calculate trend strength (0-100)
        strength = 0
        if above_20:
            strength += 20
        if above_50:
            strength += 20
        if above_200:
            strength += 20
        if sma_20_above_50:
            strength += 20
        if sma_20_slope > 0.5:
            strength += 10
        if sma_50_slope > 0.3:
            strength += 10

        # Direction
        if sma_20_slope > 1 and sma_50_slope > 0.5:
            direction = "strong_up"
        elif sma_20_slope > 0.3 and sma_50_slope > 0:
            direction = "up"
        elif sma_20_slope < -1 and sma_50_slope < -0.5:
            direction = "strong_down"
        elif sma_20_slope < -0.3 and sma_50_slope < 0:
            direction = "down"
        else:
            direction = "sideways"

        return {
            "direction": direction,
            "strength": min(100, strength),
            "sma_20_slope": round(float(sma_20_slope), 4),
            "sma_50_slope": round(float(sma_50_slope), 4),
            "above_sma_20": above_20,
            "above_sma_50": above_50,
            "above_sma_200": above_200,
            "sma_aligned": sma_20_above_50
        }

    def _analyze_volatility(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, Any]:
        """Analyze volatility regime"""
        # ATR
        atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
        atr_current = atr.iloc[-1]

        # ATR as percentage of price
        atr_pct = (atr_current / close.iloc[-1]) * 100

        # ATR percentile over last 100 periods
        atr_percentile = (atr.iloc[-100:] < atr_current).mean() * 100 if len(atr) >= 100 else 50

        # Bollinger Band width
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        bb_width = (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1]) / bb.bollinger_mavg().iloc[-1] * 100

        # Historical BB width percentile
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_mid = bb.bollinger_mavg()
        historical_width = ((bb_upper - bb_lower) / bb_mid * 100).dropna()
        bb_width_percentile = (historical_width.iloc[-50:] < bb_width).mean() * 100 if len(historical_width) >= 50 else 50

        # Determine volatility regime
        if atr_percentile > 80 or bb_width_percentile > 80:
            regime = "high"
        elif atr_percentile < 20 or bb_width_percentile < 20:
            regime = "low"
        else:
            regime = "normal"

        return {
            "regime": regime,
            "atr": round(float(atr_current), 2),
            "atr_pct": round(float(atr_pct), 2),
            "atr_percentile": round(float(atr_percentile), 1),
            "bb_width": round(float(bb_width), 2),
            "bb_width_percentile": round(float(bb_width_percentile), 1)
        }

    def _analyze_momentum(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, Any]:
        """Analyze momentum regime"""
        # ADX for trend strength
        adx_indicator = ta.trend.ADXIndicator(high, low, close, window=14)
        adx = adx_indicator.adx().iloc[-1]
        plus_di = adx_indicator.adx_pos().iloc[-1]
        minus_di = adx_indicator.adx_neg().iloc[-1]

        # RSI
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
        rsi_current = rsi.iloc[-1]
        rsi_avg = rsi.iloc[-20:].mean()

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        stoch_k = stoch.stoch().iloc[-1]

        # Determine momentum state
        if adx > 25:
            if plus_di > minus_di:
                state = "strong_bullish"
            else:
                state = "strong_bearish"
        elif adx > 20:
            if plus_di > minus_di:
                state = "bullish"
            else:
                state = "bearish"
        else:
            state = "weak"

        return {
            "state": state,
            "adx": round(float(adx), 2),
            "plus_di": round(float(plus_di), 2),
            "minus_di": round(float(minus_di), 2),
            "rsi": round(float(rsi_current), 2),
            "rsi_avg": round(float(rsi_avg), 2),
            "stochastic": round(float(stoch_k), 2)
        }

    def _analyze_volume(self, volume: pd.Series) -> Dict[str, Any]:
        """Analyze volume regime"""
        avg_20 = volume.rolling(20).mean().iloc[-1]
        avg_50 = volume.rolling(50).mean().iloc[-1]
        current = volume.iloc[-1]

        # Volume ratio
        ratio = current / avg_20 if avg_20 > 0 else 1

        # Volume trend
        if avg_20 > avg_50 * 1.2:
            trend = "increasing"
        elif avg_20 < avg_50 * 0.8:
            trend = "decreasing"
        else:
            trend = "stable"

        # Volume regime
        if ratio > 2:
            regime = "very_high"
        elif ratio > 1.5:
            regime = "high"
        elif ratio < 0.5:
            regime = "low"
        else:
            regime = "normal"

        return {
            "regime": regime,
            "ratio": round(float(ratio), 2),
            "trend": trend,
            "avg_20": int(avg_20),
            "avg_50": int(avg_50)
        }

    def _determine_primary_regime(
        self,
        trend: Dict,
        volatility: Dict,
        momentum: Dict
    ) -> MarketRegime:
        """Determine primary market regime from component analyses"""

        # High volatility overrides other regimes
        if volatility["regime"] == "high" and volatility["atr_percentile"] > 85:
            return MarketRegime.HIGH_VOLATILITY

        # Trend-based regime
        if trend["direction"] == "strong_up" and momentum["state"] in ["strong_bullish", "bullish"]:
            return MarketRegime.STRONG_BULL
        elif trend["direction"] == "up" and momentum["adx"] > 20:
            return MarketRegime.BULL
        elif trend["direction"] == "strong_down" and momentum["state"] in ["strong_bearish", "bearish"]:
            return MarketRegime.STRONG_BEAR
        elif trend["direction"] == "down" and momentum["adx"] > 20:
            return MarketRegime.BEAR
        elif volatility["regime"] == "low":
            return MarketRegime.LOW_VOLATILITY
        else:
            return MarketRegime.SIDEWAYS

    def _calculate_regime_confidence(
        self,
        trend: Dict,
        volatility: Dict,
        momentum: Dict
    ) -> float:
        """Calculate confidence in regime classification"""
        confidence = 50  # Base confidence

        # Strong trend increases confidence
        if trend["strength"] > 70:
            confidence += 20
        elif trend["strength"] > 50:
            confidence += 10

        # ADX confirmation
        if momentum["adx"] > 25:
            confidence += 15
        elif momentum["adx"] > 20:
            confidence += 10

        # Alignment of indicators
        if trend["sma_aligned"]:
            confidence += 10

        # Volatility clarity
        if volatility["regime"] != "normal":
            confidence += 5

        return min(95, confidence)

    def get_adaptive_thresholds(self, regime_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get adaptive indicator thresholds based on market regime.

        Returns dynamically adjusted thresholds for RSI, Stochastic, etc.
        """
        primary_regime = regime_info.get("primary_regime", "sideways")
        volatility_regime = regime_info.get("volatility_regime", "normal")

        # Default thresholds
        thresholds = {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "rsi_extreme_oversold": 20,
            "rsi_extreme_overbought": 80,
            "stoch_oversold": 20,
            "stoch_overbought": 80,
            "adx_trend": 25,
            "volume_high": 1.5,
            "bb_touch_strength": 1.0
        }

        # Adjust for bull market
        if primary_regime in ["strong_bull", "bull"]:
            thresholds["rsi_oversold"] = 40  # RSI doesn't go as low in bull markets
            thresholds["rsi_extreme_oversold"] = 30
            thresholds["stoch_oversold"] = 30
            thresholds["bb_touch_strength"] = 1.2  # BB touches more significant

        # Adjust for bear market
        elif primary_regime in ["strong_bear", "bear"]:
            thresholds["rsi_overbought"] = 60  # RSI doesn't go as high in bear markets
            thresholds["rsi_extreme_overbought"] = 70
            thresholds["stoch_overbought"] = 70
            thresholds["bb_touch_strength"] = 1.2

        # Adjust for high volatility
        if volatility_regime == "high":
            thresholds["rsi_oversold"] = 25  # More extreme levels in high vol
            thresholds["rsi_overbought"] = 75
            thresholds["stoch_oversold"] = 15
            thresholds["stoch_overbought"] = 85
            thresholds["volume_high"] = 2.0  # Need higher volume for confirmation
            thresholds["bb_touch_strength"] = 0.8  # BB touches less significant

        # Adjust for low volatility
        elif volatility_regime == "low":
            thresholds["rsi_oversold"] = 35
            thresholds["rsi_overbought"] = 65
            thresholds["adx_trend"] = 20  # Lower ADX threshold
            thresholds["volume_high"] = 1.3

        return thresholds


# Global instance
_regime_detector_instance = None


def get_regime_detector() -> MarketRegimeDetector:
    """Get or create global regime detector instance"""
    global _regime_detector_instance
    if _regime_detector_instance is None:
        _regime_detector_instance = MarketRegimeDetector()
    return _regime_detector_instance
