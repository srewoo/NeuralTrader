"""
Macro Analysis Module
Analyzes broader economic context: Market Regime, Yield Curves, Sector Rotation.
"""

import logging
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MacroAnalyzer:
    """
    Analyzes macroeconomic factors.
    """
    
    def analyze_macro_context(self, market_regime_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich market regime with macro interpretation.
        For now, this is a placeholder/wrapper that interprets the existing regime detector output
        in a 'macro' context, as we don't have direct feed to Bond Yields in this setup yet.
        """
        try:
            regime = market_regime_info.get('primary_regime', 'unknown')
            volatility = market_regime_info.get('volatility_regime', 'unknown')
            
            # Simple Cycle Map
            cycle_phase = "Unknown"
            strategy = "Balanced"
            
            if regime == "strong_bull":
                if volatility == "low_volatility":
                    cycle_phase = "Early Cycle / Mid Cycle"
                    strategy = "Risk On (Growth, Cyclicals)"
                else:
                    cycle_phase = "Late Cycle"
                    strategy = "Quality Growth / Momentum"
            elif regime == "bull":
                cycle_phase = "Mid Cycle"
                strategy = "Diversified Growth"
            elif regime == "bear":
                cycle_phase = "Recession / Downturn"
                strategy = "Defensive (Utilities, Staples, Gold)"
            elif regime == "strong_bear":
                cycle_phase = "Crisis / Crash"
                strategy = "Cash / Hedges / Deep Value"
            elif regime == "sideways":
                cycle_phase = "Transition / Uncertainty"
                strategy = "Income / Options Selling"
                
            return {
                "economic_cycle_proxy": cycle_phase,
                "recommended_macro_strategy": strategy,
                "regime_context": regime
            }
            
        except Exception as e:
            logger.error(f"Macro analysis failed: {e}")
            return {"error": str(e)}

_macro_analyzer = None

def get_macro_analyzer() -> MacroAnalyzer:
    global _macro_analyzer
    if _macro_analyzer is None:
        _macro_analyzer = MacroAnalyzer()
    return _macro_analyzer
