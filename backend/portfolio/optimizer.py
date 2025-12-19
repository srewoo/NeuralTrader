
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from scipy.optimize import minimize

def optimize_portfolio(
    returns_df: pd.DataFrame, 
    risk_free_rate: float = 0.06
) -> Dict[str, Any]:
    """
    Calculate Max Sharpe and Min Volatility portfolios
    
    Args:
        returns_df: DataFrame of returns
        risk_free_rate: Annual risk-free rate (decimal, e.g. 0.06 for 6%)
        
    Returns:
        Dictionary with 'max_sharpe' and 'min_vol' allocation info
    """
    if returns_df.empty:
        return {}
        
    num_assets = len(returns_df.columns)
    symbols = returns_df.columns.tolist()
    
    mean_daily_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    
    # Annualize
    ann_returns = mean_daily_returns * 252
    ann_cov = cov_matrix * 252
    
    # Helper functions for optimization
    def get_ret_vol_sr(weights):
        weights = np.array(weights)
        ret = np.sum(ann_returns * weights)
        vol = np.sqrt(np.dot(weights.T, np.dot(ann_cov, weights)))
        sr = (ret - risk_free_rate) / vol
        return np.array([ret, vol, sr])
        
    def neg_sharpe(weights):
        return -get_ret_vol_sr(weights)[2]
        
    def volatility(weights):
        return get_ret_vol_sr(weights)[1]
        
    # Constraints: sum of weights = 1, weights between 0 and 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    # Initial guess: Equal weights
    init_guess = num_assets * [1. / num_assets,]
    
    # 1. Max Sharpe
    opt_sharpe = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    max_sharpe_weights = opt_sharpe.x
    max_sharpe_stats = get_ret_vol_sr(max_sharpe_weights)
    
    # 2. Min Volatility
    opt_vol = minimize(volatility, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    min_vol_weights = opt_vol.x
    min_vol_stats = get_ret_vol_sr(min_vol_weights)
    
    # Format results
    def format_allocation(w):
        return {sym: round(val * 100, 2) for sym, val in zip(symbols, w) if val > 0.01}
        
    return {
        "max_sharpe": {
            "allocation": format_allocation(max_sharpe_weights),
            "metrics": {
                "expected_return": round(max_sharpe_stats[0] * 100, 2),
                "volatility": round(max_sharpe_stats[1] * 100, 2),
                "sharpe_ratio": round(max_sharpe_stats[2], 2)
            }
        },
        "min_vol": {
            "allocation": format_allocation(min_vol_weights),
            "metrics": {
                "expected_return": round(min_vol_stats[0] * 100, 2),
                "volatility": round(min_vol_stats[1] * 100, 2),
                "sharpe_ratio": round(min_vol_stats[2], 2)
            }
        }
    }
