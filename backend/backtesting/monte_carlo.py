"""
Monte Carlo Simulation for Backtesting
Randomizes trade sequences to assess strategy robustness and estimate
probability distributions of returns, drawdowns, and other metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import random

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation"""
    n_simulations: int = 1000  # Number of simulations to run
    confidence_levels: List[float] = None  # Percentiles to calculate
    randomize_trades: bool = True  # Shuffle trade order
    randomize_returns: bool = False  # Bootstrap from return distribution
    block_size: int = 5  # For block bootstrap (preserves some autocorrelation)
    seed: Optional[int] = None  # For reproducibility

    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [5, 10, 25, 50, 75, 90, 95]


class MonteCarloSimulator:
    """
    Monte Carlo simulation for trading strategy robustness testing.

    Methods:
    1. Trade Shuffling: Randomize order of trades to see range of outcomes
    2. Return Bootstrap: Sample from historical returns with replacement
    3. Block Bootstrap: Preserve some time-series structure
    """

    def __init__(self, config: Optional[MonteCarloConfig] = None):
        self.config = config or MonteCarloConfig()
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

    def run_simulation(
        self,
        trades: List[Dict[str, Any]],
        initial_capital: float = 100000
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation on a list of trades.

        Args:
            trades: List of trade dictionaries with 'net_pnl' or 'pnl_pct'
            initial_capital: Starting capital

        Returns:
            Simulation results with statistics
        """
        if not trades:
            return {"error": "No trades provided"}

        # Extract returns from trades
        returns = []
        for t in trades:
            if 'net_pnl_pct' in t:
                returns.append(t['net_pnl_pct'] / 100)
            elif 'pnl_pct' in t:
                returns.append(t['pnl_pct'] / 100)
            elif 'net_pnl' in t and 'entry_price' in t and 'shares' in t:
                invested = t['entry_price'] * t['shares']
                returns.append(t['net_pnl'] / invested if invested > 0 else 0)
            else:
                continue

        if not returns:
            return {"error": "Could not extract returns from trades"}

        # Run simulations
        simulation_results = []

        for _ in range(self.config.n_simulations):
            if self.config.randomize_trades:
                # Shuffle trade order
                sim_returns = random.sample(returns, len(returns))
            elif self.config.randomize_returns:
                # Bootstrap from returns
                sim_returns = np.random.choice(returns, size=len(returns), replace=True).tolist()
            else:
                sim_returns = returns

            # Calculate equity curve for this simulation
            equity = [initial_capital]
            for r in sim_returns:
                equity.append(equity[-1] * (1 + r))

            # Calculate metrics
            final_equity = equity[-1]
            total_return = (final_equity / initial_capital - 1) * 100
            max_drawdown = self._calculate_max_drawdown(equity)

            simulation_results.append({
                "final_equity": final_equity,
                "total_return_pct": total_return,
                "max_drawdown_pct": max_drawdown,
                "equity_curve": equity
            })

        # Aggregate results
        return self._aggregate_simulations(simulation_results, initial_capital, len(trades))

    def run_block_bootstrap(
        self,
        returns: List[float],
        initial_capital: float = 100000
    ) -> Dict[str, Any]:
        """
        Block bootstrap simulation - preserves some time structure.
        Better for time-series with autocorrelation.
        """
        if len(returns) < self.config.block_size:
            return {"error": "Not enough data for block bootstrap"}

        n_blocks = len(returns) // self.config.block_size
        blocks = [returns[i:i + self.config.block_size]
                  for i in range(0, len(returns) - self.config.block_size + 1)]

        simulation_results = []

        for _ in range(self.config.n_simulations):
            # Sample blocks with replacement
            selected_blocks = random.choices(blocks, k=n_blocks)
            sim_returns = [r for block in selected_blocks for r in block]

            # Calculate equity curve
            equity = [initial_capital]
            for r in sim_returns:
                equity.append(equity[-1] * (1 + r))

            final_equity = equity[-1]
            total_return = (final_equity / initial_capital - 1) * 100
            max_drawdown = self._calculate_max_drawdown(equity)

            simulation_results.append({
                "final_equity": final_equity,
                "total_return_pct": total_return,
                "max_drawdown_pct": max_drawdown
            })

        return self._aggregate_simulations(simulation_results, initial_capital, len(returns))

    def _calculate_max_drawdown(self, equity: List[float]) -> float:
        """Calculate maximum drawdown percentage"""
        peak = equity[0]
        max_dd = 0

        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _aggregate_simulations(
        self,
        results: List[Dict],
        initial_capital: float,
        n_trades: int
    ) -> Dict[str, Any]:
        """Aggregate simulation results into statistics"""

        final_equities = [r['final_equity'] for r in results]
        total_returns = [r['total_return_pct'] for r in results]
        max_drawdowns = [r['max_drawdown_pct'] for r in results]

        # Calculate percentiles
        percentiles = {}
        for level in self.config.confidence_levels:
            percentiles[f"p{level}"] = {
                "return_pct": round(np.percentile(total_returns, level), 2),
                "drawdown_pct": round(np.percentile(max_drawdowns, level), 2),
                "final_equity": round(np.percentile(final_equities, level), 2)
            }

        # Risk metrics
        var_95 = np.percentile(total_returns, 5)  # 5% worst case
        cvar_95 = np.mean([r for r in total_returns if r <= var_95])  # Expected shortfall

        # Probability of profit
        prob_profit = sum(1 for r in total_returns if r > 0) / len(total_returns) * 100

        # Probability of beating benchmarks
        prob_beat_fd = sum(1 for r in total_returns if r > 7) / len(total_returns) * 100  # Beat FD
        prob_beat_nifty = sum(1 for r in total_returns if r > 12) / len(total_returns) * 100  # Beat Nifty avg

        return {
            "n_simulations": self.config.n_simulations,
            "n_trades": n_trades,
            "initial_capital": initial_capital,

            "return_statistics": {
                "mean": round(np.mean(total_returns), 2),
                "median": round(np.median(total_returns), 2),
                "std": round(np.std(total_returns), 2),
                "min": round(min(total_returns), 2),
                "max": round(max(total_returns), 2),
                "skewness": round(float(pd.Series(total_returns).skew()), 2)
            },

            "drawdown_statistics": {
                "mean": round(np.mean(max_drawdowns), 2),
                "median": round(np.median(max_drawdowns), 2),
                "worst": round(max(max_drawdowns), 2),
                "best": round(min(max_drawdowns), 2)
            },

            "risk_metrics": {
                "var_95": round(var_95, 2),  # Value at Risk
                "cvar_95": round(cvar_95, 2),  # Conditional VaR (Expected Shortfall)
                "probability_of_profit": round(prob_profit, 1),
                "probability_beat_fd_7pct": round(prob_beat_fd, 1),
                "probability_beat_nifty_12pct": round(prob_beat_nifty, 1)
            },

            "percentiles": percentiles,

            "interpretation": self._interpret_results(
                np.mean(total_returns),
                np.std(total_returns),
                prob_profit,
                np.mean(max_drawdowns)
            )
        }

    def _interpret_results(
        self,
        mean_return: float,
        std_return: float,
        prob_profit: float,
        mean_drawdown: float
    ) -> Dict[str, Any]:
        """Interpret Monte Carlo results"""

        # Risk-adjusted return
        if std_return > 0:
            sharpe_estimate = mean_return / std_return
        else:
            sharpe_estimate = 0

        # Overall assessment
        if mean_return > 15 and prob_profit > 70 and mean_drawdown < 15:
            assessment = "Excellent"
            grade = "A"
        elif mean_return > 10 and prob_profit > 60 and mean_drawdown < 20:
            assessment = "Good"
            grade = "B"
        elif mean_return > 5 and prob_profit > 50:
            assessment = "Average"
            grade = "C"
        elif mean_return > 0:
            assessment = "Below Average"
            grade = "D"
        else:
            assessment = "Poor"
            grade = "F"

        recommendations = []

        if prob_profit < 50:
            recommendations.append("Strategy has less than 50% chance of profit - reconsider")
        if mean_drawdown > 25:
            recommendations.append("High average drawdown - consider tighter stops")
        if std_return > mean_return * 2:
            recommendations.append("High return variability - results are unpredictable")
        if mean_return < 7:
            recommendations.append("Returns may not beat a simple FD - question if worth the risk")

        if not recommendations:
            recommendations.append("Strategy shows reasonable risk-adjusted returns")

        return {
            "assessment": assessment,
            "grade": grade,
            "sharpe_estimate": round(sharpe_estimate, 2),
            "recommendations": recommendations
        }


class ParameterOptimizer:
    """
    Optimize strategy parameters using walk-forward methodology.
    Avoids overfitting by using out-of-sample validation.
    """

    def __init__(
        self,
        n_iterations: int = 100,
        optimization_metric: str = "sharpe",
        validation_split: float = 0.3
    ):
        self.n_iterations = n_iterations
        self.optimization_metric = optimization_metric
        self.validation_split = validation_split

    def optimize(
        self,
        strategy_class,
        param_ranges: Dict[str, Tuple[float, float]],
        data: pd.DataFrame,
        initial_capital: float = 100000
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters.

        Args:
            strategy_class: Strategy class to optimize
            param_ranges: Dict of parameter names to (min, max) tuples
            data: Price data
            initial_capital: Starting capital

        Returns:
            Optimal parameters and validation results
        """
        # Split data
        split_idx = int(len(data) * (1 - self.validation_split))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]

        best_params = None
        best_score = -np.inf
        all_results = []

        for i in range(self.n_iterations):
            # Random parameter sampling
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = random.randint(min_val, max_val)
                else:
                    params[param_name] = random.uniform(min_val, max_val)

            try:
                # Create strategy with these params
                strategy = strategy_class(params)

                # Backtest on training data
                signals = strategy.generate_signals(train_data)
                train_metrics = self._quick_backtest(train_data, signals, initial_capital)

                # Get optimization metric
                if self.optimization_metric == "sharpe":
                    score = train_metrics.get('sharpe', 0)
                elif self.optimization_metric == "return":
                    score = train_metrics.get('total_return', 0)
                elif self.optimization_metric == "sortino":
                    score = train_metrics.get('sortino', 0)
                else:
                    score = train_metrics.get('sharpe', 0)

                all_results.append({
                    "params": params,
                    "train_score": score,
                    "train_metrics": train_metrics
                })

                if score > best_score:
                    best_score = score
                    best_params = params

            except Exception as e:
                logger.warning(f"Optimization iteration {i} failed: {e}")
                continue

        if best_params is None:
            return {"error": "Optimization failed - no valid parameter combinations found"}

        # Validate best params on test data
        try:
            best_strategy = strategy_class(best_params)
            test_signals = best_strategy.generate_signals(test_data)
            test_metrics = self._quick_backtest(test_data, test_signals, initial_capital)
        except Exception as e:
            test_metrics = {"error": str(e)}

        # Check for overfitting
        train_return = max(r['train_metrics'].get('total_return', 0) for r in all_results if r['params'] == best_params)
        test_return = test_metrics.get('total_return', 0)

        if train_return > 0:
            overfit_ratio = test_return / train_return
        else:
            overfit_ratio = 0

        return {
            "best_params": best_params,
            "best_train_score": round(best_score, 4),
            "train_metrics": next((r['train_metrics'] for r in all_results if r['params'] == best_params), {}),
            "test_metrics": test_metrics,
            "overfit_analysis": {
                "train_return": round(train_return, 2),
                "test_return": round(test_return, 2),
                "overfit_ratio": round(overfit_ratio, 2),
                "is_overfit": overfit_ratio < 0.5,
                "recommendation": "Likely overfit - reduce complexity" if overfit_ratio < 0.5 else "Acceptable out-of-sample performance"
            },
            "iterations_run": len(all_results),
            "top_5_params": sorted(all_results, key=lambda x: x['train_score'], reverse=True)[:5]
        }

    def _quick_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        initial_capital: float
    ) -> Dict[str, float]:
        """Quick backtest for optimization - minimal overhead"""
        from .strategies import SignalType

        cash = initial_capital
        position = 0
        entry_price = 0
        returns = []

        for date, row in data.iterrows():
            if date not in signals.index:
                continue

            signal = signals.loc[date]
            price = row['Close']

            if signal == SignalType.BUY.value and position == 0:
                shares = int(cash * 0.95 / price)
                if shares > 0:
                    cash -= shares * price
                    position = shares
                    entry_price = price

            elif signal == SignalType.SELL.value and position > 0:
                cash += position * price
                ret = (price - entry_price) / entry_price
                returns.append(ret)
                position = 0

        # Close any open position
        if position > 0:
            final_price = data.iloc[-1]['Close']
            cash += position * final_price
            ret = (final_price - entry_price) / entry_price
            returns.append(ret)

        total_return = (cash / initial_capital - 1) * 100

        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns) if len(returns) > 1 else 0.0001
            sharpe = avg_return / std_return if std_return > 0 else 0

            negative_returns = [r for r in returns if r < 0]
            downside_std = np.std(negative_returns) if negative_returns else 0.0001
            sortino = avg_return / downside_std if downside_std > 0 else 0
        else:
            sharpe = 0
            sortino = 0

        return {
            "total_return": round(total_return, 2),
            "n_trades": len(returns),
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "win_rate": round(sum(1 for r in returns if r > 0) / len(returns) * 100, 1) if returns else 0
        }


# Global instances
_monte_carlo_simulator = None


def get_monte_carlo_simulator(config: Optional[MonteCarloConfig] = None) -> MonteCarloSimulator:
    """Get or create Monte Carlo simulator instance"""
    global _monte_carlo_simulator
    if _monte_carlo_simulator is None or config is not None:
        _monte_carlo_simulator = MonteCarloSimulator(config)
    return _monte_carlo_simulator
