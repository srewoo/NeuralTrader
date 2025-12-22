"""
Strategy Optimizer
Grid search and genetic algorithm optimization for trading strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from itertools import product
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

from .strategies import Strategy, SignalType, StrategyRegistry
from .metrics import PerformanceMetrics
from .price_cache import get_price_cache
from .walk_forward import IndianMarketCosts

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result from strategy optimization"""
    best_params: Dict[str, Any]
    best_score: float
    best_metrics: Dict[str, Any]
    all_results: List[Dict[str, Any]]
    optimization_method: str
    iterations: int
    symbol: str
    period: str


class StrategyOptimizer:
    """
    Optimizes trading strategy parameters using various methods:
    - Grid Search: Exhaustive search over parameter grid
    - Random Search: Random sampling of parameter space
    - Genetic Algorithm: Evolution-based optimization
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        use_realistic_costs: bool = True
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.use_realistic_costs = use_realistic_costs
        self.price_cache = get_price_cache()

        if use_realistic_costs:
            self.costs = IndianMarketCosts()
        else:
            self.costs = None

    def grid_search(
        self,
        symbol: str,
        strategy_name: str,
        param_grid: Dict[str, List[Any]],
        start_date: str,
        end_date: str,
        objective: str = "sharpe_ratio",
        train_pct: float = 0.7,
        max_workers: int = 4
    ) -> OptimizationResult:
        """
        Perform grid search optimization

        Args:
            symbol: Stock symbol
            strategy_name: Strategy name from registry
            param_grid: Dictionary of parameter -> list of values to try
            start_date: Start date
            end_date: End date
            objective: Metric to optimize (sharpe_ratio, total_return_pct, profit_factor)
            train_pct: Percentage for training (rest for validation)
            max_workers: Number of parallel workers

        Returns:
            OptimizationResult with best parameters
        """
        logger.info(f"Starting grid search for {strategy_name} on {symbol}")

        # Get data
        data = self.price_cache.get_prices(symbol, start_date, end_date)
        if data.empty or len(data) < 100:
            raise ValueError(f"Insufficient data for {symbol}")

        # Split into train/validation
        train_size = int(len(data) * train_pct)
        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:]

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(product(*param_values))

        logger.info(f"Testing {len(all_combinations)} parameter combinations")

        all_results = []
        best_score = float('-inf')
        best_params = None
        best_metrics = None

        # Run optimization in parallel
        def evaluate_params(params_tuple):
            params = dict(zip(param_names, params_tuple))
            try:
                strategy = StrategyRegistry.get_strategy(strategy_name, params)
                metrics = self._run_backtest(train_data, strategy)
                score = metrics.get(objective, 0)
                return {
                    "params": params,
                    "train_metrics": metrics,
                    "score": score,
                    "success": True
                }
            except Exception as e:
                logger.warning(f"Failed with params {params}: {e}")
                return {
                    "params": params,
                    "score": float('-inf'),
                    "success": False,
                    "error": str(e)
                }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(evaluate_params, combo): combo for combo in all_combinations}

            for future in as_completed(futures):
                result = future.result()
                if result["success"]:
                    all_results.append(result)
                    if result["score"] > best_score:
                        best_score = result["score"]
                        best_params = result["params"]
                        best_metrics = result["train_metrics"]

        # Validate best params on validation set
        if best_params:
            strategy = StrategyRegistry.get_strategy(strategy_name, best_params)
            val_metrics = self._run_backtest(val_data, strategy)
            best_metrics["validation_metrics"] = val_metrics

        logger.info(f"Grid search complete. Best score: {best_score:.4f}")

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_metrics=best_metrics,
            all_results=sorted(all_results, key=lambda x: x["score"], reverse=True)[:20],
            optimization_method="grid_search",
            iterations=len(all_combinations),
            symbol=symbol,
            period=f"{start_date} to {end_date}"
        )

    def random_search(
        self,
        symbol: str,
        strategy_name: str,
        param_ranges: Dict[str, Tuple[Any, Any]],
        start_date: str,
        end_date: str,
        n_iterations: int = 100,
        objective: str = "sharpe_ratio",
        train_pct: float = 0.7
    ) -> OptimizationResult:
        """
        Perform random search optimization

        Args:
            symbol: Stock symbol
            strategy_name: Strategy name
            param_ranges: Dictionary of parameter -> (min, max) ranges
            n_iterations: Number of random samples
            objective: Metric to optimize
            train_pct: Training percentage

        Returns:
            OptimizationResult with best parameters
        """
        logger.info(f"Starting random search for {strategy_name} on {symbol}")

        # Get data
        data = self.price_cache.get_prices(symbol, start_date, end_date)
        if data.empty or len(data) < 100:
            raise ValueError(f"Insufficient data for {symbol}")

        train_size = int(len(data) * train_pct)
        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:]

        all_results = []
        best_score = float('-inf')
        best_params = None
        best_metrics = None

        for i in range(n_iterations):
            # Sample random parameters
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = random.randint(min_val, max_val)
                else:
                    params[param_name] = random.uniform(min_val, max_val)

            try:
                strategy = StrategyRegistry.get_strategy(strategy_name, params)
                metrics = self._run_backtest(train_data, strategy)
                score = metrics.get(objective, 0)

                result = {
                    "params": params,
                    "train_metrics": metrics,
                    "score": score,
                    "success": True,
                    "iteration": i
                }
                all_results.append(result)

                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_metrics = metrics

            except Exception as e:
                logger.debug(f"Iteration {i} failed: {e}")

        # Validate best params
        if best_params:
            strategy = StrategyRegistry.get_strategy(strategy_name, best_params)
            val_metrics = self._run_backtest(val_data, strategy)
            best_metrics["validation_metrics"] = val_metrics

        logger.info(f"Random search complete. Best score: {best_score:.4f}")

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_metrics=best_metrics,
            all_results=sorted(all_results, key=lambda x: x["score"], reverse=True)[:20],
            optimization_method="random_search",
            iterations=n_iterations,
            symbol=symbol,
            period=f"{start_date} to {end_date}"
        )

    def genetic_algorithm(
        self,
        symbol: str,
        strategy_name: str,
        param_ranges: Dict[str, Tuple[Any, Any]],
        start_date: str,
        end_date: str,
        population_size: int = 50,
        generations: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        objective: str = "sharpe_ratio",
        train_pct: float = 0.7
    ) -> OptimizationResult:
        """
        Perform genetic algorithm optimization

        Args:
            symbol: Stock symbol
            strategy_name: Strategy name
            param_ranges: Dictionary of parameter -> (min, max)
            population_size: Size of population
            generations: Number of generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            objective: Metric to optimize
            train_pct: Training percentage

        Returns:
            OptimizationResult with best parameters
        """
        logger.info(f"Starting genetic algorithm for {strategy_name} on {symbol}")

        # Get data
        data = self.price_cache.get_prices(symbol, start_date, end_date)
        if data.empty or len(data) < 100:
            raise ValueError(f"Insufficient data for {symbol}")

        train_size = int(len(data) * train_pct)
        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:]

        param_names = list(param_ranges.keys())
        all_results = []

        # Initialize population
        population = [self._random_individual(param_ranges) for _ in range(population_size)]

        best_score = float('-inf')
        best_params = None
        best_metrics = None

        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                params = dict(zip(param_names, individual))
                try:
                    strategy = StrategyRegistry.get_strategy(strategy_name, params)
                    metrics = self._run_backtest(train_data, strategy)
                    score = metrics.get(objective, 0)
                    fitness_scores.append(score)

                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        best_metrics = metrics

                    all_results.append({
                        "params": params,
                        "score": score,
                        "generation": gen
                    })

                except Exception:
                    fitness_scores.append(float('-inf'))

            # Selection (tournament)
            new_population = []
            while len(new_population) < population_size:
                # Tournament selection
                tournament_size = 5
                tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
                winner = max(tournament, key=lambda x: x[1])[0]
                new_population.append(winner)

            # Crossover
            offspring = []
            for i in range(0, len(new_population) - 1, 2):
                if random.random() < crossover_rate:
                    parent1, parent2 = new_population[i], new_population[i + 1]
                    crossover_point = random.randint(1, len(param_names) - 1)
                    child1 = parent1[:crossover_point] + parent2[crossover_point:]
                    child2 = parent2[:crossover_point] + parent1[crossover_point:]
                    offspring.extend([child1, child2])
                else:
                    offspring.extend([new_population[i], new_population[i + 1]])

            # Mutation
            for i in range(len(offspring)):
                if random.random() < mutation_rate:
                    mutation_idx = random.randint(0, len(param_names) - 1)
                    param_name = param_names[mutation_idx]
                    min_val, max_val = param_ranges[param_name]
                    if isinstance(min_val, int):
                        offspring[i] = list(offspring[i])
                        offspring[i][mutation_idx] = random.randint(min_val, max_val)
                    else:
                        offspring[i] = list(offspring[i])
                        offspring[i][mutation_idx] = random.uniform(min_val, max_val)

            population = offspring[:population_size]

            logger.debug(f"Generation {gen + 1}: Best score = {best_score:.4f}")

        # Validate best params
        if best_params:
            strategy = StrategyRegistry.get_strategy(strategy_name, best_params)
            val_metrics = self._run_backtest(val_data, strategy)
            best_metrics["validation_metrics"] = val_metrics

        logger.info(f"Genetic algorithm complete. Best score: {best_score:.4f}")

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_metrics=best_metrics,
            all_results=sorted(all_results, key=lambda x: x["score"], reverse=True)[:20],
            optimization_method="genetic_algorithm",
            iterations=population_size * generations,
            symbol=symbol,
            period=f"{start_date} to {end_date}"
        )

    def _random_individual(self, param_ranges: Dict[str, Tuple[Any, Any]]) -> List[Any]:
        """Generate a random individual for genetic algorithm"""
        individual = []
        for param_name, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                individual.append(random.randint(min_val, max_val))
            else:
                individual.append(random.uniform(min_val, max_val))
        return individual

    def _run_backtest(
        self,
        data: pd.DataFrame,
        strategy: Strategy
    ) -> Dict[str, Any]:
        """Run backtest and return metrics"""
        signals = strategy.generate_signals(data)

        cash = self.initial_capital
        position = 0
        trades = []
        equity_curve = []
        entry_price = None

        for date, row in data.iterrows():
            if date not in signals.index:
                continue

            signal = signals.loc[date]
            price = row['Close']

            # Execute trades
            if signal == SignalType.BUY.value and position == 0:
                # Apply slippage on buy
                exec_price = price * (1 + self.slippage)
                shares = int((cash * 0.99) / exec_price)

                if shares > 0:
                    if self.costs:
                        buy_costs = self.costs.calculate_buy_cost(shares * exec_price)
                        total_cost = shares * exec_price + buy_costs['total']
                    else:
                        total_cost = shares * exec_price * (1 + self.commission)

                    if total_cost <= cash:
                        cash -= total_cost
                        position = shares
                        entry_price = exec_price

            elif signal == SignalType.SELL.value and position > 0:
                # Apply slippage on sell
                exec_price = price * (1 - self.slippage)

                if self.costs:
                    sell_costs = self.costs.calculate_sell_cost(position * exec_price)
                    proceeds = position * exec_price - sell_costs['total']
                else:
                    proceeds = position * exec_price * (1 - self.commission)

                pnl = proceeds - (position * entry_price)
                trades.append({
                    "entry_price": entry_price,
                    "exit_price": exec_price,
                    "shares": position,
                    "pnl": pnl,
                    "pnl_pct": (pnl / (position * entry_price)) * 100
                })

                cash += proceeds
                position = 0
                entry_price = None

            # Track equity
            position_value = position * price
            equity_curve.append(cash + position_value)

        # Calculate metrics
        if equity_curve:
            equity_series = pd.Series(equity_curve, index=data.index[:len(equity_curve)])
            returns = equity_series.pct_change().fillna(0)
            trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

            metrics = PerformanceMetrics.calculate_all_metrics(
                equity_curve=equity_series,
                returns=returns,
                trades=trades_df,
                initial_capital=self.initial_capital
            )
        else:
            metrics = {
                "total_return_pct": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0
            }

        return metrics

    def suggest_param_grid(self, strategy_name: str) -> Dict[str, List[Any]]:
        """Suggest parameter grid for common strategies"""
        grids = {
            "mean_reversion": {
                "oversold": [20, 25, 30, 35],
                "overbought": [65, 70, 75, 80],
                "rsi_period": [7, 14, 21]
            },
            "trend_following": {
                "fast_period": [5, 10, 15, 20],
                "slow_period": [30, 40, 50, 60]
            },
            "macd": {
                "fast": [8, 10, 12, 14],
                "slow": [20, 24, 26, 30],
                "signal": [7, 9, 11]
            },
            "bollinger_bands": {
                "period": [15, 20, 25],
                "std_dev": [1.5, 2.0, 2.5]
            },
            "momentum": {
                "lookback": [5, 10, 15, 20],
                "threshold": [0.01, 0.02, 0.03, 0.05]
            }
        }

        return grids.get(strategy_name, {})


# Singleton instance
_optimizer_instance: Optional[StrategyOptimizer] = None


def get_strategy_optimizer(
    initial_capital: float = 100000,
    use_realistic_costs: bool = True
) -> StrategyOptimizer:
    """Get or create strategy optimizer instance"""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = StrategyOptimizer(
            initial_capital=initial_capital,
            use_realistic_costs=use_realistic_costs
        )
    return _optimizer_instance
