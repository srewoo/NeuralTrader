
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from scipy.stats import norm

def calculate_var(
    returns: pd.Series, 
    confidence_level: float = 0.95, 
    method: str = "historical"
) -> float:
    """
    Calculate Value at Risk (VaR)
    
    Args:
        returns: Series of percentage returns
        confidence_level: Confidence level (0.90, 0.95, 0.99)
        method: 'historical' or 'parametric'
        
    Returns:
        VaR value (positive float representing loss %)
    """
    if returns.empty:
        return 0.0
        
    if method == "historical":
        # Historical Simulation
        var_percentile = np.percentile(returns, (1 - confidence_level) * 100)
        return abs(var_percentile)
        
    elif method == "parametric":
        # Variance-Covariance Method
        mu = np.mean(returns)
        sigma = np.std(returns)
        z_score = norm.ppf(1 - confidence_level)
        var = mu + z_score * sigma
        return abs(var)
        
    return 0.0

def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
    
    Args:
        returns: Series of percentage returns
        confidence_level: Confidence level
        
    Returns:
        CVaR value (positive float representing expected loss beyond VaR)
    """
    if returns.empty:
        return 0.0
        
    var = calculate_var(returns, confidence_level, method="historical")
    # Filter returns worse than VaR
    # Since var is positive loss, returns < -var are the tail losses
    tail_losses = returns[returns <= -var]
    
    if tail_losses.empty:
        return var
        
    return abs(tail_losses.mean())

def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """
    Calculate annualized volatility
    """
    if returns.empty:
        return 0.0
    
    std = returns.std()
    if annualize:
        # Assuming daily returns
        return std * np.sqrt(252)
    return std

def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculate Beta relative to market
    """
    if stock_returns.empty or market_returns.empty:
        return 1.0
        
    # Align dates
    common_idx = stock_returns.index.intersection(market_returns.index)
    if len(common_idx) < 30:
        return 1.0
        
    s_ret = stock_returns.loc[common_idx]
    m_ret = market_returns.loc[common_idx]
    
    covariance = np.cov(s_ret, m_ret)[0][1]
    variance = np.var(m_ret)
    
    if variance == 0:
        return 1.0
        
    return covariance / variance

def calculate_portfolio_risk(
    weights: Dict[str, float], 
    returns_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Calculate aggregate portfolio risk
    
    Args:
        weights: Dictionary of {symbol: weight} (must sum to 1.0)
        returns_df: DataFrame of returns for all symbols
        
    Returns:
        Dictionary of risk metrics
    """
    if returns_df.empty:
        return {}
    
    # Ensure weights match available columns
    valid_symbols = [s for s in weights.keys() if s in returns_df.columns]
    if not valid_symbols:
        return {}
        
    # Normalize weights (just in case)
    w_vector = np.array([weights[s] for s in valid_symbols])
    w_vector = w_vector / np.sum(w_vector)
    
    # Calculate portfolio returns series
    portfolio_returns = returns_df[valid_symbols].dot(w_vector)
    
    var_95 = calculate_var(portfolio_returns, 0.95)
    var_99 = calculate_var(portfolio_returns, 0.99)
    cvar_95 = calculate_cvar(portfolio_returns, 0.95)
    volatility = calculate_volatility(portfolio_returns)
    
    return {
        "var_95": round(var_95 * 100, 2),
        "var_99": round(var_99 * 100, 2),
        "cvar_95": round(cvar_95 * 100, 2),
        "volatility_annualized": round(volatility * 100, 2),
        "max_drawdown": _calculate_max_drawdown(portfolio_returns)
    }

def _calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate Max Drawdown from returns series"""
    # Reconstruct equity curve
    equity_curve = (1 + returns).cumprod()
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    return round(abs(drawdown.min()) * 100, 2)
