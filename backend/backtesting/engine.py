"""
Backtest Engine
Core backtesting engine for strategy evaluation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from .strategies import Strategy, SignalType
from .metrics import PerformanceMetrics
from .price_cache import get_price_cache

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtesting engine for evaluating trading strategies
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.0005     # 0.05%
    ):
        """
        Initialize backtest engine
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate (as decimal)
            slippage: Slippage rate (as decimal)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.price_cache = get_price_cache()
    
    def run_backtest(
        self,
        symbol: str,
        strategy: Strategy,
        start_date: str,
        end_date: str,
        position_size: float = 1.0  # Fraction of capital to use per trade
    ) -> Dict[str, Any]:
        """
        Run backtest for a strategy
        
        Args:
            symbol: Stock symbol
            strategy: Trading strategy
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            position_size: Position size as fraction of capital
            
        Returns:
            Backtest results with metrics
        """
        try:
            logger.info(f"Running backtest for {symbol} with {strategy.name}")
            
            # Get price data (REAL API CALL via cache)
            data = self.price_cache.get_prices(symbol, start_date, end_date)
            
            if data.empty:
                raise ValueError(f"No price data available for {symbol}")
            
            # Generate trading signals
            signals = strategy.generate_signals(data)
            
            # Simulate trades
            trades, equity_curve, position_history = self._simulate_trades(
                data, signals, position_size
            )
            
            # Calculate returns
            returns = equity_curve.pct_change().fillna(0)
            
            # Calculate performance metrics
            metrics = PerformanceMetrics.calculate_all_metrics(
                equity_curve=equity_curve,
                returns=returns,
                trades=trades,
                initial_capital=self.initial_capital
            )
            
            # Add Calmar ratio
            if metrics['max_drawdown'] != 0:
                metrics['calmar_ratio'] = round(
                    PerformanceMetrics.calculate_calmar_ratio(
                        metrics['cagr'] / 100,
                        metrics['max_drawdown'] / 100
                    ), 2
                )
            else:
                metrics['calmar_ratio'] = 0
            
            # Prepare result
            result = {
                "symbol": symbol,
                "strategy": strategy.name,
                "strategy_params": strategy.params,
                "start_date": start_date,
                "end_date": end_date,
                "metrics": metrics,
                "equity_curve": equity_curve.to_dict(),
                "trades": trades.to_dict('records') if not trades.empty else [],
                "total_signals": {
                    "buy": int((signals == SignalType.BUY.value).sum()),
                    "sell": int((signals == SignalType.SELL.value).sum()),
                    "hold": int((signals == SignalType.HOLD.value).sum())
                }
            }
            
            logger.info(
                f"Backtest complete: {metrics['total_return_pct']}% return, "
                f"{metrics['total_trades']} trades"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    def _simulate_trades(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        position_size: float
    ) -> tuple:
        """
        Simulate trades based on signals
        
        Args:
            data: Price data
            signals: Trading signals
            position_size: Position size fraction
            
        Returns:
            Tuple of (trades DataFrame, equity curve, position history)
        """
        cash = self.initial_capital
        position = 0  # Number of shares
        trades = []
        equity_curve = []
        position_history = []
        
        entry_price = None
        entry_date = None
        
        for date, row in data.iterrows():
            signal = signals.loc[date]
            price = row['Close']
            
            # Apply slippage
            if signal == SignalType.BUY.value:
                execution_price = price * (1 + self.slippage)
            elif signal == SignalType.SELL.value:
                execution_price = price * (1 - self.slippage)
            else:
                execution_price = price
            
            # Execute trades
            if signal == SignalType.BUY.value and position == 0:
                # Buy
                capital_to_use = cash * position_size
                shares_to_buy = int(capital_to_use / (execution_price * (1 + self.commission)))
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * execution_price * (1 + self.commission)
                    cash -= cost
                    position = shares_to_buy
                    entry_price = execution_price
                    entry_date = date
                    
                    logger.debug(f"BUY {shares_to_buy} @ {execution_price:.2f}")
            
            elif signal == SignalType.SELL.value and position > 0:
                # Sell
                proceeds = position * execution_price * (1 - self.commission)
                cash += proceeds
                
                # Record trade
                pnl = proceeds - (position * entry_price * (1 + self.commission))
                pnl_pct = (pnl / (position * entry_price)) * 100
                duration_days = (date - entry_date).days if entry_date else 0
                
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": execution_price,
                    "shares": position,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "duration_days": duration_days
                })
                
                logger.debug(f"SELL {position} @ {execution_price:.2f}, PnL: {pnl:.2f}")
                
                position = 0
                entry_price = None
                entry_date = None
            
            # Calculate equity
            position_value = position * price
            total_equity = cash + position_value
            equity_curve.append(total_equity)
            position_history.append(position)
        
        # Close any open positions at end
        if position > 0:
            final_price = data.iloc[-1]['Close']
            proceeds = position * final_price * (1 - self.commission)
            pnl = proceeds - (position * entry_price * (1 + self.commission))
            pnl_pct = (pnl / (position * entry_price)) * 100
            duration_days = (data.index[-1] - entry_date).days if entry_date else 0
            
            trades.append({
                "entry_date": entry_date,
                "exit_date": data.index[-1],
                "entry_price": entry_price,
                "exit_price": final_price,
                "shares": position,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "duration_days": duration_days
            })
            
            cash += proceeds
            equity_curve[-1] = cash
        
        # Create DataFrames
        trades_df = pd.DataFrame(trades)
        equity_series = pd.Series(equity_curve, index=data.index)
        position_series = pd.Series(position_history, index=data.index)
        
        return trades_df, equity_series, position_series
    
    def compare_strategies(
        self,
        symbol: str,
        strategies: List[Strategy],
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Compare multiple strategies
        
        Args:
            symbol: Stock symbol
            strategies: List of strategies
            start_date: Start date
            end_date: End date
            
        Returns:
            Comparison results
        """
        results = []
        
        for strategy in strategies:
            try:
                result = self.run_backtest(symbol, strategy, start_date, end_date)
                results.append({
                    "strategy": strategy.name,
                    "params": strategy.params,
                    "metrics": result['metrics']
                })
            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed: {e}")
        
        # Sort by total return
        results.sort(key=lambda x: x['metrics']['total_return_pct'], reverse=True)
        
        return {
            "symbol": symbol,
            "period": f"{start_date} to {end_date}",
            "strategies_compared": len(results),
            "results": results
        }

