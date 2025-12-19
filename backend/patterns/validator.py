
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class PatternValidator:
    """
    Validates candlestick patterns using statistical tests and walk-forward analysis.
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    def validate_pattern(
        self,
        pattern_data: Dict[str, Any],
        historical_data: pd.DataFrame,
        holding_period: int = 5
    ) -> Dict[str, Any]:
        """
        Validate a specific pattern occurrence against historical performance.

        Args:
            pattern_data: Data about the detected pattern (from CandlestickPatternDetector)
            historical_data: Full OHLCV dataframe
            holding_period: Number of days to look ahead for outcome

        Returns:
            Validation result dictionary
        """
        pattern_name = pattern_data['pattern']
        signal_type = pattern_data['type']
        
        # 1. Identify all historical occurrences of this pattern
        # Note: This requires re-running detection on history, or better, 
        # assuming the caller creates an instance that caches historical patterns.
        # For efficiency here, we'll assume we need to check history now.
        # In a production system, historical patterns should be pre-computed.
        
        # Simple simulation: Extract windows similar to current pattern? 
        # No, that's too complex. We need the actual detector.
        # To avoid circular imports or heavy re-computation, let's assume 
        # we can't fully re-detect all patterns cheaply here without access to the detector.
        # Design decision: The validator should ideally take a list of *all* detected patterns 
        # from the history, not just the current one.
        
        # However, since we are integrating into existing flow, we will simulate 
        # historical performance using a simplified approach or we need to import the detector.
        # Let's import the detector here to be robust.
        
        from .candlestick import get_pattern_detector
        detector = get_pattern_detector()
        all_patterns = detector.detect_patterns(historical_data)
        
        # Filter for the same pattern type
        same_patterns = [p for p in all_patterns if p['pattern'] == pattern_name]
        
        if len(same_patterns) < 10:
            return {
                "is_valid": False,
                "confidence": 0,
                "message": "Insufficient historical data (less than 10 occurrences)",
                "stats": {"count": len(same_patterns)}
            }
            
        # 2. Walk-Forward Analysis
        # Split into Discovery (70%) and Validation (30%)
        # Note: Time-based split is better for financial data
        split_idx = int(len(same_patterns) * 0.7)
        discovery_patterns = same_patterns[:split_idx]
        validation_patterns = same_patterns[split_idx:]
        
        discovery_stats = self._calculate_performance(discovery_patterns, historical_data, holding_period, signal_type)
        validation_stats = self._calculate_performance(validation_patterns, historical_data, holding_period, signal_type)
        
        # 3. Statistical Testing (Fisher's Exact Test or Binomial)
        # We test if the win rate is significantly > 50%
        # Use Binomial test for simplicity
        total_wins = discovery_stats['wins'] + validation_stats['wins']
        total_trades = discovery_stats['count'] + validation_stats['count']
        
        if total_trades > 0:
            p_value = stats.binomtest(total_wins, n=total_trades, p=0.5, alternative='greater').pvalue
        else:
            p_value = 1.0
            
        is_statistically_significant = p_value < (1 - self.confidence_level)
        
        # 4. Consistency Check
        # Validation win rate should not drop significantly compared to discovery
        is_robust = True
        if discovery_stats['win_rate'] > 0 and validation_stats['win_rate'] < (discovery_stats['win_rate'] * 0.7):
            is_robust = False
            
        # 5. Final Decision
        is_valid = (
            (discovery_stats['win_rate'] > 0.55 or validation_stats['win_rate'] > 0.55) and 
            is_statistically_significant
        )
        
        # Calculate final confidence score
        confidence = 0
        if is_valid:
            base_score = 50
            if is_robust: base_score += 20
            if p_value < 0.01: base_score += 20
            elif p_value < 0.05: base_score += 10
            confidence = min(100, base_score)
            
        return {
            "is_valid": is_valid,
            "confidence": confidence,
            "robust": is_robust,
            "p_value": round(p_value, 4),
            "stats": {
                "total_occurrences": total_trades,
                "win_rate": round((total_wins / total_trades) * 100, 1) if total_trades > 0 else 0,
                "discovery_win_rate": round(discovery_stats['win_rate'] * 100, 1),
                "validation_win_rate": round(validation_stats['win_rate'] * 100, 1),
                "avg_return": round(validation_stats['avg_return'] * 100, 2)
            }
        }

    def _calculate_performance(
        self, 
        patterns: List[Dict], 
        historical_data: pd.DataFrame, 
        periods: int, 
        direction: str
    ) -> Dict[str, Any]:
        """
        Calculate performance stats for a list of patterns.
        """
        wins = 0
        total_return = 0.0
        count = 0
        
        for p in patterns:
            date_str = p['date']
            try:
                # Find the index of the pattern date
                if isinstance(date_str, str):
                    date = pd.to_datetime(date_str)
                else:
                    date = date_str
                    
                # Get location in dataframe
                # Note: Assuming historical_data index is datetime
                idx_locs = historical_data.index.get_indexer([date], method='nearest')
                if idx_locs[0] == -1 or idx_locs[0] + periods >= len(historical_data):
                    continue
                    
                idx = idx_locs[0]
                entry_price = historical_data.iloc[idx]['Close']
                exit_price = historical_data.iloc[idx + periods]['Close']
                
                ret = (exit_price - entry_price) / entry_price
                
                if "bearish" in direction:
                    ret = -ret
                    
                total_return += ret
                if ret > 0:
                    wins += 1
                count += 1
                
            except Exception as e:
                logger.debug(f"Error calculating performance for pattern at {date_str}: {e}")
                continue
                
        return {
            "count": count,
            "wins": wins,
            "win_rate": (wins / count) if count > 0 else 0,
            "avg_return": (total_return / count) if count > 0 else 0
        }

# Global instance
_validator_instance = None

def get_pattern_validator() -> PatternValidator:
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = PatternValidator()
    return _validator_instance
