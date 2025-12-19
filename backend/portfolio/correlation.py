
import pandas as pd
import numpy as np
from typing import Dict, Any, List

def calculate_correlation_matrix(returns_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate correlation matrix for visualization
    """
    if returns_df.empty:
        return {}
        
    corr_matrix = returns_df.corr()
    
    # Format for frontend (e.g. heatmap)
    # List of x-axis (symbols), y-axis (symbols), and z (values)
    symbols = corr_matrix.columns.tolist()
    z_values = []
    
    for i in range(len(symbols)):
        row = []
        for j in range(len(symbols)):
            val = corr_matrix.iloc[i, j]
            # Replace NaN with 0
            if np.isnan(val):
                val = 0
            row.append(round(val, 2))
        z_values.append(row)
        
    return {
        "symbols": symbols,
        "matrix": z_values
    }

def calculate_diversification_score(
    weights: Dict[str, float], 
    returns_df: pd.DataFrame
) -> float:
    """
    Calculate a diversification score (0-100).
    Higher is better.
    
    Based on the difference between weighted average volatility 
    and actual portfolio volatility.
    """
    if returns_df.empty or len(weights) < 2:
        return 0.0
        
    valid_symbols = [s for s in weights.keys() if s in returns_df.columns]
    if len(valid_symbols) < 2:
        return 0.0
        
    w_vector = np.array([weights[s] for s in valid_symbols])
    w_vector = w_vector / np.sum(w_vector)
    
    # 1. Weighted Average Volatility (No diversification benefit)
    indiv_vols = returns_df[valid_symbols].std()
    weighted_avg_vol = np.sum(indiv_vols * w_vector)
    
    # 2. Actual Portfolio Volatility (With correlation benefit)
    cov_matrix = returns_df[valid_symbols].cov()
    portfolio_var = np.dot(w_vector.T, np.dot(cov_matrix, w_vector))
    portfolio_vol = np.sqrt(portfolio_var)
    
    # 3. Diversification Ratio
    # If correct_correlation = 1, port_vol = weighted_avg_vol -> Ratio = 1
    # If correlation < 1, port_vol < weighted_avg_vol -> Ratio > 1
    
    if portfolio_vol == 0:
        return 0.0
        
    div_ratio = weighted_avg_vol / portfolio_vol
    
    # Map ratio to 0-100 score
    # Ratio of 1.0 = Score 0 (Poor)
    # Ratio of 1.5+ = Score 100 (Excellent)
    
    score = (div_ratio - 1.0) * 200 # 1.5 -> 0.5 * 200 = 100
    return min(100.0, max(0.0, score))
