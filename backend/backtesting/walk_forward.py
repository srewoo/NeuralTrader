"""
Walk-Forward Backtesting Engine
Implements proper train/test splits and realistic transaction costs
for Indian equity markets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from .strategies import Strategy, SignalType
from .metrics import PerformanceMetrics
from .price_cache import get_price_cache

logger = logging.getLogger(__name__)


@dataclass
class IndianMarketCosts:
    """
    Realistic transaction costs for Indian equity markets (NSE/BSE).
    All values in percentage or absolute amounts.
    """
    # Brokerage (varies by broker)
    brokerage_rate: float = 0.0003  # 0.03% for discount brokers like Zerodha
    brokerage_min: float = 0  # Minimum brokerage per order
    brokerage_max: float = 20  # Maximum brokerage per order (Zerodha style)

    # STT (Securities Transaction Tax)
    stt_buy: float = 0.0  # No STT on buy (delivery)
    stt_sell: float = 0.001  # 0.1% on sell (delivery)
    stt_intraday_sell: float = 0.00025  # 0.025% for intraday

    # Exchange Transaction Charges
    nse_charge: float = 0.0000297  # 0.00297%
    bse_charge: float = 0.0000375  # 0.00375%

    # GST (18% on brokerage + transaction charges)
    gst_rate: float = 0.18

    # SEBI Turnover Charges
    sebi_charge: float = 0.000001  # ₹10 per crore

    # Stamp Duty (state-wise, using average)
    stamp_duty_buy: float = 0.00015  # 0.015% on buy
    stamp_duty_sell: float = 0.0  # No stamp duty on sell

    # DP Charges (for delivery trades, per scrip sold)
    dp_charges: float = 15.93  # Per scrip sold (varies by broker)

    # Slippage estimate
    slippage: float = 0.001  # 0.1% average slippage

    def calculate_buy_cost(self, value: float, is_intraday: bool = False) -> Dict[str, float]:
        """Calculate total cost for buying shares"""
        brokerage = min(max(value * self.brokerage_rate, self.brokerage_min), self.brokerage_max)
        exchange_charge = value * self.nse_charge
        gst = (brokerage + exchange_charge) * self.gst_rate
        sebi = value * self.sebi_charge
        stamp_duty = value * self.stamp_duty_buy
        slippage_cost = value * self.slippage

        total = brokerage + exchange_charge + gst + sebi + stamp_duty + slippage_cost

        return {
            "brokerage": round(brokerage, 2),
            "exchange_charge": round(exchange_charge, 2),
            "gst": round(gst, 2),
            "sebi": round(sebi, 2),
            "stamp_duty": round(stamp_duty, 2),
            "slippage": round(slippage_cost, 2),
            "total": round(total, 2),
            "total_pct": round((total / value) * 100, 4)
        }

    def calculate_sell_cost(self, value: float, is_intraday: bool = False) -> Dict[str, float]:
        """Calculate total cost for selling shares"""
        brokerage = min(max(value * self.brokerage_rate, self.brokerage_min), self.brokerage_max)
        exchange_charge = value * self.nse_charge

        if is_intraday:
            stt = value * self.stt_intraday_sell
        else:
            stt = value * self.stt_sell

        gst = (brokerage + exchange_charge) * self.gst_rate
        sebi = value * self.sebi_charge
        dp = self.dp_charges if not is_intraday else 0
        slippage_cost = value * self.slippage

        total = brokerage + stt + exchange_charge + gst + sebi + dp + slippage_cost

        return {
            "brokerage": round(brokerage, 2),
            "stt": round(stt, 2),
            "exchange_charge": round(exchange_charge, 2),
            "gst": round(gst, 2),
            "sebi": round(sebi, 2),
            "dp_charges": round(dp, 2),
            "slippage": round(slippage_cost, 2),
            "total": round(total, 2),
            "total_pct": round((total / value) * 100, 4)
        }

    def calculate_roundtrip_cost(self, value: float, is_intraday: bool = False) -> Dict[str, float]:
        """Calculate total cost for a complete buy-sell roundtrip"""
        buy = self.calculate_buy_cost(value, is_intraday)
        sell = self.calculate_sell_cost(value, is_intraday)

        return {
            "buy_costs": buy,
            "sell_costs": sell,
            "total": round(buy['total'] + sell['total'], 2),
            "total_pct": round(buy['total_pct'] + sell['total_pct'], 4)
        }


class WalkForwardEngine:
    """
    Walk-Forward Backtesting Engine.

    Walk-forward testing uses rolling train/test splits to avoid look-ahead bias
    and provide more realistic performance estimates.

    Process:
    1. Divide data into multiple periods
    2. For each period: train on in-sample, test on out-of-sample
    3. Aggregate results to get realistic performance

    Example with 5 years of data and 20% test ratio:
    - Period 1: Train on Y1-Y4, Test on Y5
    - Period 2: Train on Y2-Y5, Test on Y6 (if available)
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        costs: Optional[IndianMarketCosts] = None,
        is_intraday: bool = False
    ):
        self.initial_capital = initial_capital
        self.costs = costs or IndianMarketCosts()
        self.is_intraday = is_intraday
        self.price_cache = get_price_cache()

    def run_walk_forward(
        self,
        symbol: str,
        strategy: Strategy,
        start_date: str,
        end_date: str,
        train_pct: float = 0.7,
        n_splits: int = 5,
        anchored: bool = False
    ) -> Dict[str, Any]:
        """
        Run walk-forward backtest.

        Args:
            symbol: Stock symbol
            strategy: Trading strategy
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            train_pct: Percentage of data for training in each split
            n_splits: Number of train/test splits
            anchored: If True, always start training from the beginning

        Returns:
            Walk-forward results with aggregated metrics
        """
        try:
            logger.info(f"Running walk-forward backtest for {symbol}")

            # Get price data
            data = self.price_cache.get_prices(symbol, start_date, end_date)

            if data.empty or len(data) < 100:
                raise ValueError(f"Insufficient data for {symbol}")

            # Create train/test splits
            splits = self._create_splits(data, train_pct, n_splits, anchored)

            # Run backtest on each split
            split_results = []
            all_trades = []
            all_equity = []

            for i, (train_data, test_data) in enumerate(splits):
                logger.info(f"Processing split {i+1}/{n_splits}")

                # Train strategy (optimize parameters if applicable)
                strategy.fit(train_data)

                # Test on out-of-sample data
                result = self._run_single_backtest(test_data, strategy)

                split_results.append({
                    "split": i + 1,
                    "train_start": train_data.index[0].strftime("%Y-%m-%d"),
                    "train_end": train_data.index[-1].strftime("%Y-%m-%d"),
                    "test_start": test_data.index[0].strftime("%Y-%m-%d"),
                    "test_end": test_data.index[-1].strftime("%Y-%m-%d"),
                    "train_size": len(train_data),
                    "test_size": len(test_data),
                    "metrics": result["metrics"],
                    "trades": len(result["trades"])
                })

                all_trades.extend(result["trades"])
                all_equity.append(result["equity_curve"])

            # Aggregate results
            aggregated = self._aggregate_results(split_results, all_trades, all_equity)

            return {
                "symbol": symbol,
                "strategy": strategy.name,
                "strategy_params": strategy.params,
                "period": f"{start_date} to {end_date}",
                "n_splits": n_splits,
                "train_pct": train_pct,
                "anchored": anchored,
                "split_results": split_results,
                "aggregated_metrics": aggregated,
                "all_trades": all_trades,
                "transaction_costs": {
                    "model": "Indian Market Costs",
                    "brokerage_rate": f"{self.costs.brokerage_rate * 100:.3f}%",
                    "stt_rate": f"{self.costs.stt_sell * 100:.2f}%",
                    "estimated_roundtrip": self.costs.calculate_roundtrip_cost(100000)
                },
                "robustness_score": self._calculate_robustness(split_results)
            }

        except Exception as e:
            logger.error(f"Walk-forward backtest failed: {e}")
            raise

    def _create_splits(
        self,
        data: pd.DataFrame,
        train_pct: float,
        n_splits: int,
        anchored: bool
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create train/test splits for walk-forward testing"""
        splits = []
        n = len(data)

        # Calculate window sizes
        test_size = int(n * (1 - train_pct) / n_splits)
        train_size = int(n * train_pct)

        for i in range(n_splits):
            if anchored:
                # Anchored: always start from beginning
                train_end = train_size + (i * test_size)
                train_start = 0
            else:
                # Rolling: shift both train and test windows
                train_start = i * test_size
                train_end = train_start + train_size

            test_start = train_end
            test_end = min(test_start + test_size, n)

            if test_end <= test_start:
                break

            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]

            if len(train_data) >= 50 and len(test_data) >= 10:
                splits.append((train_data, test_data))

        return splits

    def _run_single_backtest(
        self,
        data: pd.DataFrame,
        strategy: Strategy
    ) -> Dict[str, Any]:
        """Run backtest on a single data segment"""
        signals = strategy.generate_signals(data)

        # Simulate trades with realistic costs
        trades, equity_curve = self._simulate_trades_with_costs(data, signals)

        # Calculate metrics
        if len(equity_curve) > 0:
            returns = pd.Series(equity_curve).pct_change().fillna(0)
            equity_series = pd.Series(equity_curve, index=data.index[:len(equity_curve)])

            trades_df = pd.DataFrame(trades)
            metrics = PerformanceMetrics.calculate_all_metrics(
                equity_curve=equity_series,
                returns=returns,
                trades=trades_df,
                initial_capital=self.initial_capital
            )
        else:
            metrics = {
                "total_return_pct": 0,
                "total_trades": 0,
                "win_rate": 0,
                "max_drawdown": 0
            }

        return {
            "trades": trades,
            "equity_curve": equity_curve,
            "metrics": metrics
        }

    def _simulate_trades_with_costs(
        self,
        data: pd.DataFrame,
        signals: pd.Series
    ) -> Tuple[List[Dict], List[float]]:
        """Simulate trades with realistic Indian market costs"""
        cash = self.initial_capital
        position = 0
        trades = []
        equity_curve = []

        entry_price = None
        entry_date = None
        entry_costs = {}

        for date, row in data.iterrows():
            if date not in signals.index:
                continue

            signal = signals.loc[date]
            price = row['Close']

            # Execute trades
            if signal == SignalType.BUY.value and position == 0:
                # Calculate how much we can buy after costs
                # Estimate 0.5% for buy-side costs
                max_buy_value = cash * 0.995
                shares_to_buy = int(max_buy_value / price)

                if shares_to_buy > 0:
                    buy_value = shares_to_buy * price
                    buy_costs = self.costs.calculate_buy_cost(buy_value, self.is_intraday)

                    total_cost = buy_value + buy_costs['total']

                    if total_cost <= cash:
                        cash -= total_cost
                        position = shares_to_buy
                        entry_price = price
                        entry_date = date
                        entry_costs = buy_costs

            elif signal == SignalType.SELL.value and position > 0:
                # Sell with costs
                sell_value = position * price
                sell_costs = self.costs.calculate_sell_cost(sell_value, self.is_intraday)

                proceeds = sell_value - sell_costs['total']
                cash += proceeds

                # Calculate P&L including all costs
                gross_pnl = (price - entry_price) * position
                total_costs = entry_costs.get('total', 0) + sell_costs['total']
                net_pnl = gross_pnl - total_costs
                net_pnl_pct = (net_pnl / (position * entry_price)) * 100

                trades.append({
                    "entry_date": entry_date.strftime("%Y-%m-%d") if hasattr(entry_date, 'strftime') else str(entry_date),
                    "exit_date": date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(price, 2),
                    "shares": position,
                    "gross_pnl": round(gross_pnl, 2),
                    "buy_costs": round(entry_costs.get('total', 0), 2),
                    "sell_costs": round(sell_costs['total'], 2),
                    "total_costs": round(total_costs, 2),
                    "net_pnl": round(net_pnl, 2),
                    "net_pnl_pct": round(net_pnl_pct, 2),
                    "cost_impact_pct": round((total_costs / (position * entry_price)) * 100, 2)
                })

                position = 0
                entry_price = None
                entry_date = None
                entry_costs = {}

            # Calculate equity
            position_value = position * price
            total_equity = cash + position_value
            equity_curve.append(total_equity)

        return trades, equity_curve

    def _aggregate_results(
        self,
        split_results: List[Dict],
        all_trades: List[Dict],
        all_equity: List[List[float]]
    ) -> Dict[str, Any]:
        """Aggregate results across all splits"""
        if not split_results:
            return {}

        # Extract metrics from each split
        returns = [r['metrics'].get('total_return_pct', 0) for r in split_results]
        win_rates = [r['metrics'].get('win_rate', 0) for r in split_results]
        drawdowns = [r['metrics'].get('max_drawdown', 0) for r in split_results]
        trade_counts = [r['trades'] for r in split_results]

        # Calculate aggregated metrics
        avg_return = np.mean(returns)
        std_return = np.std(returns)

        # Profitable splits
        profitable_splits = sum(1 for r in returns if r > 0)

        # Overall trade statistics
        if all_trades:
            winning_trades = [t for t in all_trades if t.get('net_pnl', 0) > 0]
            losing_trades = [t for t in all_trades if t.get('net_pnl', 0) <= 0]

            avg_win = np.mean([t['net_pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([abs(t['net_pnl']) for t in losing_trades]) if losing_trades else 0

            total_costs = sum(t.get('total_costs', 0) for t in all_trades)
            avg_cost_impact = np.mean([t.get('cost_impact_pct', 0) for t in all_trades])
        else:
            avg_win = avg_loss = total_costs = avg_cost_impact = 0

        return {
            "avg_return_pct": round(avg_return, 2),
            "std_return_pct": round(std_return, 2),
            "min_return_pct": round(min(returns), 2),
            "max_return_pct": round(max(returns), 2),
            "avg_win_rate": round(np.mean(win_rates), 2),
            "avg_max_drawdown": round(np.mean(drawdowns), 2),
            "worst_drawdown": round(max(drawdowns), 2),
            "total_trades": len(all_trades),
            "profitable_splits": profitable_splits,
            "total_splits": len(split_results),
            "consistency_ratio": round(profitable_splits / len(split_results), 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(avg_win / avg_loss, 2) if avg_loss > 0 else 0,
            "total_transaction_costs": round(total_costs, 2),
            "avg_cost_impact_pct": round(avg_cost_impact, 2),
            "sharpe_estimate": round(avg_return / std_return, 2) if std_return > 0 else 0
        }

    def _calculate_robustness(self, split_results: List[Dict]) -> Dict[str, Any]:
        """Calculate robustness score based on consistency across splits"""
        if not split_results:
            return {"score": 0, "grade": "F", "issues": ["No results"]}

        returns = [r['metrics'].get('total_return_pct', 0) for r in split_results]
        win_rates = [r['metrics'].get('win_rate', 0) for r in split_results]

        score = 100
        issues = []

        # Penalize for inconsistent returns
        if np.std(returns) > np.mean(returns):
            score -= 20
            issues.append("High return variability across periods")

        # Penalize for negative splits
        negative_splits = sum(1 for r in returns if r < 0)
        if negative_splits > 0:
            penalty = (negative_splits / len(returns)) * 30
            score -= penalty
            issues.append(f"{negative_splits} out of {len(returns)} periods had losses")

        # Penalize for low win rates
        avg_win_rate = np.mean(win_rates)
        if avg_win_rate < 40:
            score -= 20
            issues.append(f"Low average win rate: {avg_win_rate:.1f}%")
        elif avg_win_rate < 50:
            score -= 10
            issues.append(f"Below-average win rate: {avg_win_rate:.1f}%")

        # Determine grade
        if score >= 80:
            grade = "A"
        elif score >= 70:
            grade = "B"
        elif score >= 60:
            grade = "C"
        elif score >= 50:
            grade = "D"
        else:
            grade = "F"

        if not issues:
            issues.append("Strategy shows good consistency")

        return {
            "score": max(0, round(score)),
            "grade": grade,
            "issues": issues
        }


# Global instance
_walk_forward_engine = None


def get_walk_forward_engine(
    initial_capital: float = 100000,
    is_intraday: bool = False
) -> WalkForwardEngine:
    """Get or create walk-forward engine instance"""
    global _walk_forward_engine
    if _walk_forward_engine is None:
        _walk_forward_engine = WalkForwardEngine(
            initial_capital=initial_capital,
            is_intraday=is_intraday
        )
    return _walk_forward_engine
