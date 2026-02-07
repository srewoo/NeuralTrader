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

        # Volume confirmation
        if 'Volume' in recent_data.columns and len(recent_data) >= 20:
            recent_data['vol_ma_20'] = recent_data['Volume'].rolling(20).mean()
            recent_data['volume_ratio'] = recent_data['Volume'] / recent_data['vol_ma_20'].replace(0, 1)
        else:
            recent_data['volume_ratio'] = 1.0

        # ATR calculation
        if len(recent_data) >= 14:
            tr = pd.concat([
                recent_data['High'] - recent_data['Low'],
                (recent_data['High'] - recent_data['Close'].shift(1)).abs(),
                (recent_data['Low'] - recent_data['Close'].shift(1)).abs()
            ], axis=1).max(axis=1)
            recent_data['atr_14'] = tr.rolling(14).mean()
        else:
            recent_data['atr_14'] = recent_data['range'].rolling(5).mean()

        # Detect each pattern type
        for i in range(len(recent_data)):
            date = recent_data.index[i]
            
            # Single candle patterns
            if i >= 0:
                # Doji
                if self._is_doji(recent_data.iloc[i]):
                    prev_c = list(recent_data.iloc[max(0,i-5):i+1]) if i >= 1 else None
                    detected_patterns.append(self._create_pattern_result(
                        "doji", date, recent_data.iloc[i], prev_c
                    ))
                
                # Hammer / Hanging Man
                if self._is_hammer(recent_data.iloc[i]):
                    if i > 0 and not recent_data.iloc[i-1]['is_bullish']:
                        pattern_name = "hammer"
                    else:
                        pattern_name = "hanging_man"
                    prev_c = list(recent_data.iloc[max(0,i-5):i+1]) if i >= 1 else None
                    detected_patterns.append(self._create_pattern_result(
                        pattern_name, date, recent_data.iloc[i], prev_c
                    ))
                
                # Inverted Hammer / Shooting Star
                if self._is_inverted_hammer(recent_data.iloc[i]):
                    if i > 0 and not recent_data.iloc[i-1]['is_bullish']:
                        pattern_name = "inverted_hammer"
                    else:
                        pattern_name = "shooting_star"
                    prev_c = list(recent_data.iloc[max(0,i-5):i+1]) if i >= 1 else None
                    detected_patterns.append(self._create_pattern_result(
                        pattern_name, date, recent_data.iloc[i], prev_c
                    ))
                
                # Spinning Top
                if self._is_spinning_top(recent_data.iloc[i]):
                    prev_c = list(recent_data.iloc[max(0,i-5):i+1]) if i >= 1 else None
                    detected_patterns.append(self._create_pattern_result(
                        "spinning_top", date, recent_data.iloc[i], prev_c
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

            # Five candle patterns
            if i >= 4:
                c1 = recent_data.iloc[i-4]
                c2 = recent_data.iloc[i-3]
                c3 = recent_data.iloc[i-2]
                c4 = recent_data.iloc[i-1]
                c5 = recent_data.iloc[i]

                if self._is_rising_three_methods(c1, c2, c3, c4, c5):
                    detected_patterns.append(self._create_pattern_result("rising_three_methods", date, c5, [c1, c2, c3, c4, c5]))
                if self._is_falling_three_methods(c1, c2, c3, c4, c5):
                    detected_patterns.append(self._create_pattern_result("falling_three_methods", date, c5, [c1, c2, c3, c4, c5]))

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
    
    def _is_rising_three_methods(self, c1, c2, c3, c4, c5):
        """Detect Rising Three Methods pattern (5-candle bullish continuation)"""
        return (c1['is_bullish'] and c1['body'] > c1['range'] * 0.5 and
                not c2['is_bullish'] and not c3['is_bullish'] and not c4['is_bullish'] and
                c2['body'] < c1['body'] * 0.5 and c3['body'] < c1['body'] * 0.5 and c4['body'] < c1['body'] * 0.5 and
                c2['Low'] > c1['Low'] and c3['Low'] > c1['Low'] and c4['Low'] > c1['Low'] and
                c5['is_bullish'] and c5['Close'] > c1['Close'])

    def _is_falling_three_methods(self, c1, c2, c3, c4, c5):
        """Detect Falling Three Methods pattern (5-candle bearish continuation)"""
        return (not c1['is_bullish'] and c1['body'] > c1['range'] * 0.5 and
                c2['is_bullish'] and c3['is_bullish'] and c4['is_bullish'] and
                c2['body'] < c1['body'] * 0.5 and c3['body'] < c1['body'] * 0.5 and c4['body'] < c1['body'] * 0.5 and
                c2['High'] < c1['High'] and c3['High'] < c1['High'] and c4['High'] < c1['High'] and
                not c5['is_bullish'] and c5['Close'] < c1['Close'])

    def _calculate_strength_score(self, pattern_name, candle, prev_candles=None):
        """Calculate 0-100 strength score based on pattern quality + volume + trend."""
        base_scores = {"strong": 70, "medium": 50, "weak": 30}
        base = base_scores.get(self.PATTERNS[pattern_name]["strength"], 50)

        # Volume confirmation bonus (up to +15)
        vol_ratio = candle.get('volume_ratio', 1.0) if isinstance(candle, dict) else getattr(candle, 'volume_ratio', 1.0)
        if vol_ratio > 1.5:
            base += 15
        elif vol_ratio > 1.2:
            base += 10
        elif vol_ratio < 0.7:
            base -= 10

        # ATR-relative body size bonus (meaningful candle vs noise)
        atr = candle.get('atr_14', None) if isinstance(candle, dict) else getattr(candle, 'atr_14', None)
        body = candle.get('body', 0) if isinstance(candle, dict) else getattr(candle, 'body', 0)
        if atr and atr > 0:
            body_atr_ratio = body / atr
            if body_atr_ratio > 0.8:
                base += 10  # Strong body relative to ATR
            elif body_atr_ratio > 0.5:
                base += 5

        # Trend context bonus (up to +10)
        pattern_type = self.PATTERNS[pattern_name]["type"]
        if prev_candles is not None and len(prev_candles) >= 5:
            # Check if 5-day trend aligns with reversal direction
            try:
                closes = [c['Close'] if isinstance(c, dict) else c.Close for c in prev_candles[-5:]]
                trend_up = closes[-1] > closes[0]
                if "bullish" in pattern_type and not trend_up:
                    base += 10  # Bullish reversal in downtrend = stronger
                elif "bearish" in pattern_type and trend_up:
                    base += 10  # Bearish reversal in uptrend = stronger
            except:
                pass

        return max(0, min(100, base))

    def _create_pattern_result(
        self,
        pattern_name: str,
        date: Any,
        candle: pd.Series,
        prev_candles: Optional[List[pd.Series]] = None
    ) -> Dict[str, Any]:
        """Create pattern detection result"""
        pattern_info = self.PATTERNS.get(pattern_name, {})
        price = float(candle['Close'])
        strength_score = self._calculate_strength_score(pattern_name, candle, prev_candles)

        # Format date and time for display
        date_str = str(date)
        time_str = None

        if hasattr(date, 'strftime'):
            date_str = date.strftime('%d/%m/%Y')
            time_str = date.strftime('%H:%M:%S')
        elif 'T' in date_str:
            # Split date and time
            parts = date_str.split('T')
            date_part = parts[0]
            if len(parts) > 1:
                # Extract time (HH:MM:SS)
                time_part = parts[1].split('.')[0] if '.' in parts[1] else parts[1]
                time_str = time_part[:8] if len(time_part) >= 8 else time_part

            # Convert YYYY-MM-DD to DD/MM/YYYY
            date_parts = date_part.split('-')
            if len(date_parts) == 3:
                date_str = f"{date_parts[2]}/{date_parts[1]}/{date_parts[0]}"

        # If no time or time is 00:00:00, use market close time (3:30 PM IST for Indian markets)
        if not time_str or time_str == "00:00:00":
            time_str = "15:30:00"

        implication = self._get_pattern_implication(pattern_name)

        return {
            "pattern": pattern_name,
            "date": str(date),
            "date_formatted": date_str,
            "time": time_str,
            "datetime_formatted": f"{date_str} {time_str}",
            "type": pattern_info.get("type", "unknown"),
            "strength": pattern_info.get("strength", "weak"),
            "price": price,
            "price_formatted": f"₹{price:,.2f}",
            "open": float(candle['Open']),
            "high": float(candle['High']),
            "low": float(candle['Low']),
            "close": float(candle['Close']),
            "description": self._get_pattern_description(pattern_name),
            "strength_score": strength_score,
            "implication": implication,
            "display": f"{pattern_name.replace('_', ' ').title()}\n{date_str} - ₹{price:,.2f}"
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

    def _get_pattern_implication(self, pattern_name: str) -> Dict[str, str]:
        """Get trading implication for the pattern"""
        implications = {
            "doji": {
                "direction": "neutral",
                "signal": "⚠️ Wait for confirmation",
                "meaning": "Buyers and sellers are balanced. Wait for next candle to confirm direction."
            },
            "hammer": {
                "direction": "bullish",
                "signal": "📈 Potential upside",
                "meaning": "Buyers rejected lower prices. Stock may reverse upward from here."
            },
            "inverted_hammer": {
                "direction": "bullish",
                "signal": "📈 Potential upside",
                "meaning": "After downtrend, buyers showing interest. Watch for confirmation."
            },
            "hanging_man": {
                "direction": "bearish",
                "signal": "📉 Potential downside",
                "meaning": "After uptrend, selling pressure emerging. Consider taking profits."
            },
            "shooting_star": {
                "direction": "bearish",
                "signal": "📉 Potential downside",
                "meaning": "Buyers failed to hold highs. Reversal likely, watch for follow-through."
            },
            "spinning_top": {
                "direction": "neutral",
                "signal": "⚠️ Wait for confirmation",
                "meaning": "Market is undecided. Trend may pause or reverse - wait for clarity."
            },
            "bullish_engulfing": {
                "direction": "bullish",
                "signal": "🚀 Strong buy signal",
                "meaning": "Buyers overpowered sellers completely. High probability of upward move."
            },
            "bearish_engulfing": {
                "direction": "bearish",
                "signal": "🔻 Strong sell signal",
                "meaning": "Sellers overpowered buyers completely. High probability of downward move."
            },
            "piercing_line": {
                "direction": "bullish",
                "signal": "📈 Bullish reversal",
                "meaning": "Strong buying after gap down. Recovery likely to continue."
            },
            "dark_cloud_cover": {
                "direction": "bearish",
                "signal": "📉 Bearish reversal",
                "meaning": "Strong selling after gap up. Decline likely to continue."
            },
            "morning_star": {
                "direction": "bullish",
                "signal": "🚀 Strong buy signal",
                "meaning": "Classic bottom reversal. High confidence upward move expected."
            },
            "evening_star": {
                "direction": "bearish",
                "signal": "🔻 Strong sell signal",
                "meaning": "Classic top reversal. High confidence downward move expected."
            },
            "three_white_soldiers": {
                "direction": "bullish",
                "signal": "🚀 Strong bullish momentum",
                "meaning": "Sustained buying pressure. Uptrend likely to continue."
            },
            "three_black_crows": {
                "direction": "bearish",
                "signal": "🔻 Strong bearish momentum",
                "meaning": "Sustained selling pressure. Downtrend likely to continue."
            },
            "harami": {
                "direction": "neutral",
                "signal": "⚠️ Trend weakening",
                "meaning": "Current trend losing momentum. Possible reversal or consolidation ahead."
            },
            "rising_three_methods": {
                "direction": "bullish",
                "signal": "📈 Bullish continuation",
                "meaning": "Brief pause in uptrend. Expect continuation higher."
            },
            "falling_three_methods": {
                "direction": "bearish",
                "signal": "📉 Bearish continuation",
                "meaning": "Brief pause in downtrend. Expect continuation lower."
            }
        }
        return implications.get(pattern_name, {
            "direction": "neutral",
            "signal": "Pattern detected",
            "meaning": "Analyze in context of overall trend."
        })


# Global instance
_pattern_detector_instance = None


def get_pattern_detector() -> CandlestickPatternDetector:
    """Get or create global pattern detector instance"""
    global _pattern_detector_instance
    if _pattern_detector_instance is None:
        _pattern_detector_instance = CandlestickPatternDetector()
    return _pattern_detector_instance

