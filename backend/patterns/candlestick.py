"""
Candlestick Pattern Detector
Detects common candlestick patterns using real price data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CandlestickPatternDetector:
    """
    Detects candlestick patterns in OHLC data
    """
    
    # Pattern definitions with signals
    PATTERNS = {
        # Bullish reversal patterns
        "hammer": {"type": "bullish_reversal", "strength": "medium"},
        "inverted_hammer": {"type": "bullish_reversal", "strength": "medium"},
        "bullish_engulfing": {"type": "bullish_reversal", "strength": "strong"},
        "piercing_line": {"type": "bullish_reversal", "strength": "medium"},
        "morning_star": {"type": "bullish_reversal", "strength": "strong"},
        "three_white_soldiers": {"type": "bullish_reversal", "strength": "strong"},
        
        # Bearish reversal patterns
        "hanging_man": {"type": "bearish_reversal", "strength": "medium"},
        "shooting_star": {"type": "bearish_reversal", "strength": "medium"},
        "bearish_engulfing": {"type": "bearish_reversal", "strength": "strong"},
        "dark_cloud_cover": {"type": "bearish_reversal", "strength": "medium"},
        "evening_star": {"type": "bearish_reversal", "strength": "strong"},
        "three_black_crows": {"type": "bearish_reversal", "strength": "strong"},
        
        # Continuation patterns
        "rising_three_methods": {"type": "bullish_continuation", "strength": "medium"},
        "falling_three_methods": {"type": "bearish_continuation", "strength": "medium"},
        
        # Indecision patterns
        "doji": {"type": "indecision", "strength": "weak"},
        "spinning_top": {"type": "indecision", "strength": "weak"},
        "harami": {"type": "indecision", "strength": "weak"}
    }
    
    def __init__(self):
        """Initialize pattern detector"""
        pass
    
    def detect_patterns(
        self,
        data: pd.DataFrame,
        lookback_periods: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Detect all candlestick patterns in data (REAL pattern detection)
        
        Args:
            data: DataFrame with OHLC data (Open, High, Low, Close)
            lookback_periods: Number of recent periods to analyze
            
        Returns:
            List of detected patterns with details
        """
        if data.empty or len(data) < 3:
            return []
        
        # Use recent data
        recent_data = data.tail(lookback_periods).copy()
        
        detected_patterns = []
        
        # Calculate body and shadow sizes
        recent_data['body'] = abs(recent_data['Close'] - recent_data['Open'])
        recent_data['upper_shadow'] = recent_data['High'] - recent_data[['Close', 'Open']].max(axis=1)
        recent_data['lower_shadow'] = recent_data[['Close', 'Open']].min(axis=1) - recent_data['Low']
        recent_data['is_bullish'] = recent_data['Close'] > recent_data['Open']
        recent_data['range'] = recent_data['High'] - recent_data['Low']
        
        # Detect each pattern type
        for i in range(len(recent_data)):
            date = recent_data.index[i]
            
            # Single candle patterns
            if i >= 0:
                # Doji
                if self._is_doji(recent_data.iloc[i]):
                    detected_patterns.append(self._create_pattern_result(
                        "doji", date, recent_data.iloc[i]
                    ))
                
                # Hammer / Hanging Man
                if self._is_hammer(recent_data.iloc[i]):
                    if i > 0 and not recent_data.iloc[i-1]['is_bullish']:
                        pattern_name = "hammer"
                    else:
                        pattern_name = "hanging_man"
                    detected_patterns.append(self._create_pattern_result(
                        pattern_name, date, recent_data.iloc[i]
                    ))
                
                # Inverted Hammer / Shooting Star
                if self._is_inverted_hammer(recent_data.iloc[i]):
                    if i > 0 and not recent_data.iloc[i-1]['is_bullish']:
                        pattern_name = "inverted_hammer"
                    else:
                        pattern_name = "shooting_star"
                    detected_patterns.append(self._create_pattern_result(
                        pattern_name, date, recent_data.iloc[i]
                    ))
                
                # Spinning Top
                if self._is_spinning_top(recent_data.iloc[i]):
                    detected_patterns.append(self._create_pattern_result(
                        "spinning_top", date, recent_data.iloc[i]
                    ))
            
            # Two candle patterns
            if i >= 1:
                current = recent_data.iloc[i]
                previous = recent_data.iloc[i-1]
                
                # Bullish Engulfing
                if self._is_bullish_engulfing(previous, current):
                    detected_patterns.append(self._create_pattern_result(
                        "bullish_engulfing", date, current, [previous, current]
                    ))
                
                # Bearish Engulfing
                if self._is_bearish_engulfing(previous, current):
                    detected_patterns.append(self._create_pattern_result(
                        "bearish_engulfing", date, current, [previous, current]
                    ))
                
                # Piercing Line
                if self._is_piercing_line(previous, current):
                    detected_patterns.append(self._create_pattern_result(
                        "piercing_line", date, current, [previous, current]
                    ))
                
                # Dark Cloud Cover
                if self._is_dark_cloud_cover(previous, current):
                    detected_patterns.append(self._create_pattern_result(
                        "dark_cloud_cover", date, current, [previous, current]
                    ))
                
                # Harami
                if self._is_harami(previous, current):
                    detected_patterns.append(self._create_pattern_result(
                        "harami", date, current, [previous, current]
                    ))
            
            # Three candle patterns
            if i >= 2:
                candle1 = recent_data.iloc[i-2]
                candle2 = recent_data.iloc[i-1]
                candle3 = recent_data.iloc[i]
                
                # Morning Star
                if self._is_morning_star(candle1, candle2, candle3):
                    detected_patterns.append(self._create_pattern_result(
                        "morning_star", date, candle3, [candle1, candle2, candle3]
                    ))
                
                # Evening Star
                if self._is_evening_star(candle1, candle2, candle3):
                    detected_patterns.append(self._create_pattern_result(
                        "evening_star", date, candle3, [candle1, candle2, candle3]
                    ))
                
                # Three White Soldiers
                if self._is_three_white_soldiers(candle1, candle2, candle3):
                    detected_patterns.append(self._create_pattern_result(
                        "three_white_soldiers", date, candle3, [candle1, candle2, candle3]
                    ))
                
                # Three Black Crows
                if self._is_three_black_crows(candle1, candle2, candle3):
                    detected_patterns.append(self._create_pattern_result(
                        "three_black_crows", date, candle3, [candle1, candle2, candle3]
                    ))
        
        return detected_patterns
    
    def get_recent_patterns(
        self,
        data: pd.DataFrame,
        days: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get patterns detected in recent days
        
        Args:
            data: OHLC data
            days: Number of recent days
            
        Returns:
            Recent patterns
        """
        all_patterns = self.detect_patterns(data)
        
        if not all_patterns:
            return []
        
        # Filter to recent days
        recent_date = data.index[-1]
        recent_patterns = [
            p for p in all_patterns
            if (recent_date - pd.to_datetime(p['date'])).days <= days
        ]
        
        return recent_patterns
    
    # Pattern detection methods
    
    def _is_doji(self, candle: pd.Series) -> bool:
        """Detect Doji pattern"""
        body_size = candle['body']
        range_size = candle['range']
        
        # Body is very small relative to range
        return body_size < (range_size * 0.1) if range_size > 0 else False
    
    def _is_hammer(self, candle: pd.Series) -> bool:
        """Detect Hammer pattern"""
        body = candle['body']
        lower_shadow = candle['lower_shadow']
        upper_shadow = candle['upper_shadow']
        range_size = candle['range']
        
        # Long lower shadow, small upper shadow, small body
        return (
            lower_shadow > (body * 2) and
            upper_shadow < (body * 0.5) and
            body > (range_size * 0.2)
        )
    
    def _is_inverted_hammer(self, candle: pd.Series) -> bool:
        """Detect Inverted Hammer pattern"""
        body = candle['body']
        lower_shadow = candle['lower_shadow']
        upper_shadow = candle['upper_shadow']
        range_size = candle['range']
        
        # Long upper shadow, small lower shadow, small body
        return (
            upper_shadow > (body * 2) and
            lower_shadow < (body * 0.5) and
            body > (range_size * 0.2)
        )
    
    def _is_spinning_top(self, candle: pd.Series) -> bool:
        """Detect Spinning Top pattern"""
        body = candle['body']
        upper_shadow = candle['upper_shadow']
        lower_shadow = candle['lower_shadow']
        range_size = candle['range']
        
        # Small body with similar upper and lower shadows
        return (
            body < (range_size * 0.3) and
            upper_shadow > body and
            lower_shadow > body and
            abs(upper_shadow - lower_shadow) < (range_size * 0.3)
        )
    
    def _is_bullish_engulfing(self, prev: pd.Series, curr: pd.Series) -> bool:
        """Detect Bullish Engulfing pattern"""
        # Previous is bearish, current is bullish
        # Current body engulfs previous body
        return (
            not prev['is_bullish'] and
            curr['is_bullish'] and
            curr['Open'] < prev['Close'] and
            curr['Close'] > prev['Open']
        )
    
    def _is_bearish_engulfing(self, prev: pd.Series, curr: pd.Series) -> bool:
        """Detect Bearish Engulfing pattern"""
        # Previous is bullish, current is bearish
        # Current body engulfs previous body
        return (
            prev['is_bullish'] and
            not curr['is_bullish'] and
            curr['Open'] > prev['Close'] and
            curr['Close'] < prev['Open']
        )
    
    def _is_piercing_line(self, prev: pd.Series, curr: pd.Series) -> bool:
        """Detect Piercing Line pattern"""
        # Previous bearish, current bullish
        # Current opens below previous close, closes above midpoint
        prev_midpoint = (prev['Open'] + prev['Close']) / 2
        return (
            not prev['is_bullish'] and
            curr['is_bullish'] and
            curr['Open'] < prev['Close'] and
            curr['Close'] > prev_midpoint and
            curr['Close'] < prev['Open']
        )
    
    def _is_dark_cloud_cover(self, prev: pd.Series, curr: pd.Series) -> bool:
        """Detect Dark Cloud Cover pattern"""
        # Previous bullish, current bearish
        # Current opens above previous close, closes below midpoint
        prev_midpoint = (prev['Open'] + prev['Close']) / 2
        return (
            prev['is_bullish'] and
            not curr['is_bullish'] and
            curr['Open'] > prev['Close'] and
            curr['Close'] < prev_midpoint and
            curr['Close'] > prev['Open']
        )
    
    def _is_harami(self, prev: pd.Series, curr: pd.Series) -> bool:
        """Detect Harami pattern"""
        # Current body is contained within previous body
        return (
            curr['Open'] > min(prev['Open'], prev['Close']) and
            curr['Open'] < max(prev['Open'], prev['Close']) and
            curr['Close'] > min(prev['Open'], prev['Close']) and
            curr['Close'] < max(prev['Open'], prev['Close'])
        )
    
    def _is_morning_star(self, c1: pd.Series, c2: pd.Series, c3: pd.Series) -> bool:
        """Detect Morning Star pattern"""
        # First bearish, second small, third bullish
        return (
            not c1['is_bullish'] and
            c2['body'] < (c1['body'] * 0.5) and
            c3['is_bullish'] and
            c3['Close'] > (c1['Open'] + c1['Close']) / 2
        )
    
    def _is_evening_star(self, c1: pd.Series, c2: pd.Series, c3: pd.Series) -> bool:
        """Detect Evening Star pattern"""
        # First bullish, second small, third bearish
        return (
            c1['is_bullish'] and
            c2['body'] < (c1['body'] * 0.5) and
            not c3['is_bullish'] and
            c3['Close'] < (c1['Open'] + c1['Close']) / 2
        )
    
    def _is_three_white_soldiers(self, c1: pd.Series, c2: pd.Series, c3: pd.Series) -> bool:
        """Detect Three White Soldiers pattern"""
        # Three consecutive bullish candles with higher closes
        return (
            c1['is_bullish'] and
            c2['is_bullish'] and
            c3['is_bullish'] and
            c2['Close'] > c1['Close'] and
            c3['Close'] > c2['Close'] and
            c2['Open'] > c1['Open'] and
            c3['Open'] > c2['Open']
        )
    
    def _is_three_black_crows(self, c1: pd.Series, c2: pd.Series, c3: pd.Series) -> bool:
        """Detect Three Black Crows pattern"""
        # Three consecutive bearish candles with lower closes
        return (
            not c1['is_bullish'] and
            not c2['is_bullish'] and
            not c3['is_bullish'] and
            c2['Close'] < c1['Close'] and
            c3['Close'] < c2['Close'] and
            c2['Open'] < c1['Open'] and
            c3['Open'] < c2['Open']
        )
    
    def _create_pattern_result(
        self,
        pattern_name: str,
        date: Any,
        candle: pd.Series,
        candles: Optional[List[pd.Series]] = None
    ) -> Dict[str, Any]:
        """Create pattern detection result"""
        pattern_info = self.PATTERNS.get(pattern_name, {})
        
        return {
            "pattern": pattern_name,
            "date": str(date),
            "type": pattern_info.get("type", "unknown"),
            "strength": pattern_info.get("strength", "weak"),
            "price": float(candle['Close']),
            "open": float(candle['Open']),
            "high": float(candle['High']),
            "low": float(candle['Low']),
            "close": float(candle['Close']),
            "description": self._get_pattern_description(pattern_name)
        }
    
    def _get_pattern_description(self, pattern_name: str) -> str:
        """Get human-readable pattern description"""
        descriptions = {
            "doji": "Market indecision with small body and equal shadows",
            "hammer": "Bullish reversal with long lower shadow",
            "inverted_hammer": "Bullish reversal with long upper shadow",
            "hanging_man": "Bearish reversal with long lower shadow",
            "shooting_star": "Bearish reversal with long upper shadow",
            "spinning_top": "Indecision with small body and long shadows",
            "bullish_engulfing": "Strong bullish reversal - current candle engulfs previous",
            "bearish_engulfing": "Strong bearish reversal - current candle engulfs previous",
            "piercing_line": "Bullish reversal - closes above midpoint of previous candle",
            "dark_cloud_cover": "Bearish reversal - closes below midpoint of previous candle",
            "morning_star": "Strong bullish reversal - three candle pattern",
            "evening_star": "Strong bearish reversal - three candle pattern",
            "three_white_soldiers": "Strong bullish continuation - three consecutive bullish candles",
            "three_black_crows": "Strong bearish continuation - three consecutive bearish candles",
            "harami": "Indecision - small candle within previous candle's body",
            "rising_three_methods": "Bullish continuation pattern",
            "falling_three_methods": "Bearish continuation pattern"
        }
        return descriptions.get(pattern_name, "Candlestick pattern detected")


# Global instance
_pattern_detector_instance = None


def get_pattern_detector() -> CandlestickPatternDetector:
    """Get or create global pattern detector instance"""
    global _pattern_detector_instance
    if _pattern_detector_instance is None:
        _pattern_detector_instance = CandlestickPatternDetector()
    return _pattern_detector_instance

