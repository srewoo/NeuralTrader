"""
Performance Metrics Calculator
Calculates various performance metrics for backtesting results
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime


class PerformanceMetrics:
    """
    Calculates comprehensive performance metrics for trading strategies
    """
    
    @staticmethod
    def calculate_all_metrics(
        equity_curve: pd.Series,
        returns: pd.Series,
        trades: pd.DataFrame,
        initial_capital: float,
        risk_free_rate: float = 0.05
    ) -> Dict[str, Any]:
        """
        Calculate all performance metrics
        
        Args:
            equity_curve: Time series of portfolio value
            returns: Time series of returns
            trades: DataFrame with trade records
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics.update(PerformanceMetrics._calculate_basic_metrics(
            equity_curve, initial_capital
        ))
        
        # Risk metrics
        metrics.update(PerformanceMetrics._calculate_risk_metrics(
            returns, risk_free_rate
        ))
        
        # Trade metrics
        metrics.update(PerformanceMetrics._calculate_trade_metrics(trades))
        
        # Drawdown metrics
        metrics.update(PerformanceMetrics._calculate_drawdown_metrics(equity_curve))
        
        return metrics
    
    @staticmethod
    def _calculate_basic_metrics(
        equity_curve: pd.Series,
        initial_capital: float
    ) -> Dict[str, Any]:
        """Calculate basic performance metrics"""
        final_value = equity_curve.iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # Calculate CAGR
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        cagr = ((final_value / initial_capital) ** (1 / years) - 1) if years > 0 else 0
        
        return {
            "initial_capital": initial_capital,
            "final_value": round(final_value, 2),
            "total_return": round(total_return * 100, 2),
            "total_return_pct": round(total_return * 100, 2),
            "cagr": round(cagr * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "duration_days": days,
            "duration_years": round(years, 2)
        }
    
    @staticmethod
    def _calculate_risk_metrics(
        returns: pd.Series,
        risk_free_rate: float
    ) -> Dict[str, Any]:
        """Calculate risk-adjusted metrics"""
        # Sharpe Ratio
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        sharpe_ratio = (
            np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            if excess_returns.std() > 0 else 0
        )
        
        # Sortino Ratio (using only downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = (
            np.sqrt(252) * excess_returns.mean() / downside_std
            if downside_std > 0 else 0
        )
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Calmar Ratio (CAGR / Max Drawdown)
        # Will be calculated after drawdown metrics
        
        return {
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "volatility": round(volatility * 100, 2),
            "volatility_pct": round(volatility * 100, 2)
        }
    
    @staticmethod
    def _calculate_trade_metrics(trades: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trade-related metrics"""
        if trades.empty:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "win_rate_pct": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "avg_trade_duration": 0
            }
        
        total_trades = len(trades)
        
        # Separate winning and losing trades
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Average win/loss
        avg_win = winning_trades['pnl'].mean() if win_count > 0 else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if loss_count > 0 else 0
        
        # Profit factor
        total_wins = winning_trades['pnl'].sum() if win_count > 0 else 0
        total_losses = abs(losing_trades['pnl'].sum()) if loss_count > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Average trade duration
        if 'duration_days' in trades.columns:
            avg_duration = trades['duration_days'].mean()
        else:
            avg_duration = 0
        
        # Largest win/loss
        largest_win = winning_trades['pnl'].max() if win_count > 0 else 0
        largest_loss = losing_trades['pnl'].min() if loss_count > 0 else 0
        
        return {
            "total_trades": total_trades,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "win_rate": round(win_rate, 4),
            "win_rate_pct": round(win_rate * 100, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "largest_win": round(largest_win, 2),
            "largest_loss": round(largest_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_trade_duration": round(avg_duration, 1)
        }
    
    @staticmethod
    def _calculate_drawdown_metrics(equity_curve: pd.Series) -> Dict[str, Any]:
        """Calculate drawdown metrics"""
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Max drawdown duration
        drawdown_series = drawdown < 0
        if drawdown_series.any():
            # Find drawdown periods
            drawdown_starts = drawdown_series & ~drawdown_series.shift(1, fill_value=False)
            drawdown_ends = ~drawdown_series & drawdown_series.shift(1, fill_value=False)
            
            if drawdown_starts.any() and drawdown_ends.any():
                start_dates = equity_curve.index[drawdown_starts]
                end_dates = equity_curve.index[drawdown_ends]
                
                # Calculate durations
                if len(end_dates) > 0 and len(start_dates) > 0:
                    # Match starts with ends
                    durations = []
                    for start in start_dates:
                        matching_ends = end_dates[end_dates > start]
                        if len(matching_ends) > 0:
                            duration = (matching_ends[0] - start).days
                            durations.append(duration)
                    
                    max_drawdown_duration = max(durations) if durations else 0
                else:
                    max_drawdown_duration = 0
            else:
                max_drawdown_duration = 0
        else:
            max_drawdown_duration = 0
        
        return {
            "max_drawdown": round(max_drawdown * 100, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "max_drawdown_duration": max_drawdown_duration
        }
    
    @staticmethod
    def calculate_calmar_ratio(cagr: float, max_drawdown: float) -> float:
        """
        Calculate Calmar Ratio
        
        Args:
            cagr: Compound Annual Growth Rate (decimal)
            max_drawdown: Maximum Drawdown (decimal, negative)
            
        Returns:
            Calmar ratio
        """
        if max_drawdown == 0:
            return 0
        return abs(cagr / max_drawdown)

