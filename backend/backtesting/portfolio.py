"""
Portfolio-Level Backtesting
Test strategies across multiple stocks with:
- Correlation analysis
- Portfolio optimization
- Risk-adjusted metrics
- Diversification scoring
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

from .strategies import Strategy, SignalType
from .price_cache import get_price_cache
from .walk_forward import IndianMarketCosts

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    """Configuration for portfolio backtesting"""
    initial_capital: float = 1000000  # ₹10 lakh
    max_positions: int = 10  # Maximum concurrent positions
    position_size_method: str = "equal"  # equal, risk_parity, kelly
    max_position_pct: float = 0.20  # Max 20% in single stock
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly, none
    stop_loss_pct: Optional[float] = 0.10  # 10% stop loss per position
    take_profit_pct: Optional[float] = 0.25  # 25% take profit


class CorrelationAnalyzer:
    """Analyze correlations between stocks"""

    def __init__(self):
        self.price_cache = get_price_cache()

    def calculate_correlation_matrix(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        return_period: str = "daily"  # daily, weekly, monthly
    ) -> Dict[str, Any]:
        """
        Calculate correlation matrix between stocks.

        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            return_period: Period for return calculation

        Returns:
            Correlation matrix and analysis
        """
        returns_data = {}

        for symbol in symbols:
            try:
                data = self.price_cache.get_prices(symbol, start_date, end_date)
                if data.empty:
                    continue

                # Calculate returns based on period
                if return_period == "daily":
                    returns = data['Close'].pct_change().dropna()
                elif return_period == "weekly":
                    weekly = data['Close'].resample('W').last()
                    returns = weekly.pct_change().dropna()
                elif return_period == "monthly":
                    monthly = data['Close'].resample('M').last()
                    returns = monthly.pct_change().dropna()
                else:
                    returns = data['Close'].pct_change().dropna()

                returns_data[symbol] = returns

            except Exception as e:
                logger.warning(f"Failed to get data for {symbol}: {e}")

        if len(returns_data) < 2:
            return {"error": "Need at least 2 stocks with valid data"}

        # Create DataFrame and align dates
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()

        if len(returns_df) < 20:
            return {"error": "Insufficient overlapping data"}

        # Calculate correlation matrix
        corr_matrix = returns_df.corr()

        # Calculate additional metrics
        avg_correlation = corr_matrix.values[np.triu_indices(len(symbols), k=1)].mean()

        # Find highly correlated pairs
        high_corr_pairs = []
        low_corr_pairs = []

        for i, sym1 in enumerate(corr_matrix.columns):
            for j, sym2 in enumerate(corr_matrix.columns):
                if i < j:
                    corr = corr_matrix.loc[sym1, sym2]
                    pair = {"stock1": sym1, "stock2": sym2, "correlation": round(corr, 3)}

                    if corr > 0.7:
                        high_corr_pairs.append(pair)
                    elif corr < 0.3:
                        low_corr_pairs.append(pair)

        # Diversification ratio
        individual_vols = returns_df.std()
        portfolio_vol = (returns_df.mean(axis=1)).std()
        diversification_ratio = individual_vols.mean() / portfolio_vol if portfolio_vol > 0 else 1

        return {
            "correlation_matrix": corr_matrix.round(3).to_dict(),
            "average_correlation": round(avg_correlation, 3),
            "highly_correlated_pairs": sorted(high_corr_pairs, key=lambda x: x['correlation'], reverse=True),
            "low_correlation_pairs": sorted(low_corr_pairs, key=lambda x: x['correlation']),
            "diversification_ratio": round(diversification_ratio, 2),
            "diversification_benefit": "Good" if diversification_ratio > 1.5 else "Moderate" if diversification_ratio > 1.2 else "Low",
            "n_stocks": len(returns_data),
            "data_points": len(returns_df),
            "period": f"{start_date} to {end_date}",
            "return_period": return_period,
            "recommendations": self._get_diversification_recommendations(
                avg_correlation, high_corr_pairs, diversification_ratio
            )
        }

    def _get_diversification_recommendations(
        self,
        avg_corr: float,
        high_corr_pairs: List[Dict],
        div_ratio: float
    ) -> List[str]:
        """Generate diversification recommendations"""
        recommendations = []

        if avg_corr > 0.6:
            recommendations.append("Portfolio is highly correlated - consider adding uncorrelated assets")

        if len(high_corr_pairs) > 3:
            symbols_to_review = set()
            for pair in high_corr_pairs[:3]:
                symbols_to_review.add(pair['stock1'])
                symbols_to_review.add(pair['stock2'])
            recommendations.append(f"Review positions: {', '.join(symbols_to_review)} - highly correlated")

        if div_ratio < 1.2:
            recommendations.append("Low diversification benefit - stocks move together")
        elif div_ratio > 2:
            recommendations.append("Excellent diversification - portfolio is well-hedged")

        if not recommendations:
            recommendations.append("Portfolio diversification appears adequate")

        return recommendations

    def calculate_beta(
        self,
        symbol: str,
        benchmark: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """Calculate beta of a stock relative to benchmark"""
        try:
            stock_data = self.price_cache.get_prices(symbol, start_date, end_date)
            bench_data = self.price_cache.get_prices(benchmark, start_date, end_date)

            if stock_data.empty or bench_data.empty:
                return {"error": "Could not fetch data"}

            stock_returns = stock_data['Close'].pct_change().dropna()
            bench_returns = bench_data['Close'].pct_change().dropna()

            # Align dates
            aligned = pd.concat([stock_returns, bench_returns], axis=1, keys=['stock', 'benchmark']).dropna()

            if len(aligned) < 20:
                return {"error": "Insufficient data"}

            # Calculate beta
            covariance = aligned['stock'].cov(aligned['benchmark'])
            variance = aligned['benchmark'].var()
            beta = covariance / variance if variance > 0 else 1

            # Calculate alpha (Jensen's alpha)
            risk_free_rate = 0.06 / 252  # ~6% annual, daily
            stock_excess = aligned['stock'].mean() - risk_free_rate
            bench_excess = aligned['benchmark'].mean() - risk_free_rate
            alpha = stock_excess - beta * bench_excess

            # R-squared
            correlation = aligned['stock'].corr(aligned['benchmark'])
            r_squared = correlation ** 2

            return {
                "symbol": symbol,
                "benchmark": benchmark,
                "beta": round(beta, 3),
                "alpha_daily": round(alpha, 6),
                "alpha_annual": round(alpha * 252, 4),
                "r_squared": round(r_squared, 3),
                "correlation": round(correlation, 3),
                "interpretation": {
                    "beta": "Aggressive" if beta > 1.2 else "Defensive" if beta < 0.8 else "Market-neutral",
                    "alpha": "Outperforming" if alpha > 0 else "Underperforming"
                },
                "period": f"{start_date} to {end_date}",
                "data_points": len(aligned)
            }

        except Exception as e:
            logger.error(f"Beta calculation failed: {e}")
            return {"error": str(e)}


class PortfolioBacktester:
    """
    Backtest trading strategies across a portfolio of stocks.
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        self.config = config or PortfolioConfig()
        self.price_cache = get_price_cache()
        self.costs = IndianMarketCosts()
        self.correlation_analyzer = CorrelationAnalyzer()

    def run_portfolio_backtest(
        self,
        symbols: List[str],
        strategy: Strategy,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Run backtest across multiple stocks.

        Args:
            symbols: List of stock symbols
            strategy: Trading strategy to use
            start_date: Start date
            end_date: End date

        Returns:
            Portfolio backtest results
        """
        logger.info(f"Running portfolio backtest for {len(symbols)} stocks")

        # Fetch all price data
        all_data = {}
        for symbol in symbols:
            try:
                data = self.price_cache.get_prices(symbol, start_date, end_date)
                if not data.empty and len(data) >= 50:
                    all_data[symbol] = data
            except Exception as e:
                logger.warning(f"Failed to get data for {symbol}: {e}")

        if len(all_data) < 2:
            return {"error": "Need at least 2 stocks with valid data"}

        # Generate signals for all stocks
        all_signals = {}
        for symbol, data in all_data.items():
            try:
                signals = strategy.generate_signals(data)
                all_signals[symbol] = signals
            except Exception as e:
                logger.warning(f"Failed to generate signals for {symbol}: {e}")

        # Run simulation
        result = self._simulate_portfolio(all_data, all_signals)

        # Calculate correlation of the portfolio
        corr_analysis = self.correlation_analyzer.calculate_correlation_matrix(
            list(all_data.keys()), start_date, end_date
        )

        result["correlation_analysis"] = corr_analysis
        result["strategy"] = strategy.name
        result["period"] = f"{start_date} to {end_date}"
        result["stocks_traded"] = list(all_data.keys())

        return result

    def _simulate_portfolio(
        self,
        all_data: Dict[str, pd.DataFrame],
        all_signals: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """Simulate portfolio trading"""

        cash = self.config.initial_capital
        positions = {}  # {symbol: {"shares": int, "entry_price": float, "entry_date": date}}
        trades = []
        equity_curve = []
        position_history = []

        # Get all unique dates across all stocks
        all_dates = set()
        for data in all_data.values():
            all_dates.update(data.index.tolist())
        all_dates = sorted(all_dates)

        for date in all_dates:
            # Check for exits first (stop loss, take profit, sell signals)
            symbols_to_exit = []

            for symbol, pos in positions.items():
                if symbol not in all_data or date not in all_data[symbol].index:
                    continue

                current_price = all_data[symbol].loc[date, 'Close']
                entry_price = pos['entry_price']
                pnl_pct = (current_price - entry_price) / entry_price

                exit_reason = None

                # Check stop loss
                if self.config.stop_loss_pct and pnl_pct < -self.config.stop_loss_pct:
                    exit_reason = "stop_loss"

                # Check take profit
                elif self.config.take_profit_pct and pnl_pct > self.config.take_profit_pct:
                    exit_reason = "take_profit"

                # Check sell signal
                elif symbol in all_signals and date in all_signals[symbol].index:
                    if all_signals[symbol].loc[date] == SignalType.SELL.value:
                        exit_reason = "signal"

                if exit_reason:
                    symbols_to_exit.append((symbol, current_price, exit_reason))

            # Execute exits
            for symbol, exit_price, reason in symbols_to_exit:
                pos = positions[symbol]
                sell_value = pos['shares'] * exit_price
                sell_costs = self.costs.calculate_sell_cost(sell_value)

                proceeds = sell_value - sell_costs['total']
                cash += proceeds

                gross_pnl = (exit_price - pos['entry_price']) * pos['shares']
                net_pnl = proceeds - (pos['shares'] * pos['entry_price'])

                trades.append({
                    "symbol": symbol,
                    "entry_date": pos['entry_date'].strftime("%Y-%m-%d") if hasattr(pos['entry_date'], 'strftime') else str(pos['entry_date']),
                    "exit_date": date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date),
                    "entry_price": round(pos['entry_price'], 2),
                    "exit_price": round(exit_price, 2),
                    "shares": pos['shares'],
                    "gross_pnl": round(gross_pnl, 2),
                    "net_pnl": round(net_pnl, 2),
                    "exit_reason": reason
                })

                del positions[symbol]

            # Check for new entries
            if len(positions) < self.config.max_positions:
                for symbol, signals in all_signals.items():
                    if symbol in positions:
                        continue
                    if symbol not in all_data or date not in all_data[symbol].index:
                        continue
                    if date not in signals.index:
                        continue

                    if signals.loc[date] == SignalType.BUY.value:
                        current_price = all_data[symbol].loc[date, 'Close']

                        # Calculate position size
                        if self.config.position_size_method == "equal":
                            position_value = min(
                                cash / (self.config.max_positions - len(positions)),
                                self.config.initial_capital * self.config.max_position_pct
                            )
                        else:
                            position_value = cash * self.config.max_position_pct

                        if position_value < 10000:  # Minimum ₹10k position
                            continue

                        shares = int(position_value / current_price)
                        if shares < 1:
                            continue

                        buy_value = shares * current_price
                        buy_costs = self.costs.calculate_buy_cost(buy_value)
                        total_cost = buy_value + buy_costs['total']

                        if total_cost > cash:
                            continue

                        cash -= total_cost
                        positions[symbol] = {
                            "shares": shares,
                            "entry_price": current_price,
                            "entry_date": date
                        }

                        if len(positions) >= self.config.max_positions:
                            break

            # Calculate portfolio value
            portfolio_value = cash
            for symbol, pos in positions.items():
                if symbol in all_data and date in all_data[symbol].index:
                    portfolio_value += pos['shares'] * all_data[symbol].loc[date, 'Close']

            equity_curve.append({
                "date": date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date),
                "equity": round(portfolio_value, 2),
                "cash": round(cash, 2),
                "n_positions": len(positions)
            })

            position_history.append({
                "date": date,
                "positions": {s: p['shares'] for s, p in positions.items()}
            })

        # Close remaining positions at end
        for symbol, pos in list(positions.items()):
            if symbol in all_data:
                final_price = all_data[symbol].iloc[-1]['Close']
                sell_value = pos['shares'] * final_price
                sell_costs = self.costs.calculate_sell_cost(sell_value)
                proceeds = sell_value - sell_costs['total']
                cash += proceeds

                gross_pnl = (final_price - pos['entry_price']) * pos['shares']
                net_pnl = proceeds - (pos['shares'] * pos['entry_price'])

                trades.append({
                    "symbol": symbol,
                    "entry_date": pos['entry_date'].strftime("%Y-%m-%d") if hasattr(pos['entry_date'], 'strftime') else str(pos['entry_date']),
                    "exit_date": "End",
                    "entry_price": round(pos['entry_price'], 2),
                    "exit_price": round(final_price, 2),
                    "shares": pos['shares'],
                    "gross_pnl": round(gross_pnl, 2),
                    "net_pnl": round(net_pnl, 2),
                    "exit_reason": "end_of_period"
                })

        # Calculate metrics
        final_equity = cash
        total_return = (final_equity / self.config.initial_capital - 1) * 100

        if trades:
            trades_df = pd.DataFrame(trades)
            winning_trades = trades_df[trades_df['net_pnl'] > 0]
            losing_trades = trades_df[trades_df['net_pnl'] <= 0]

            win_rate = len(winning_trades) / len(trades_df) * 100
            avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades['net_pnl'].mean()) if len(losing_trades) > 0 else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

            total_costs = sum(
                self.costs.calculate_buy_cost(t['entry_price'] * t['shares'])['total'] +
                self.costs.calculate_sell_cost(t['exit_price'] * t['shares'])['total']
                for t in trades
            )
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            total_costs = 0

        # Calculate drawdown
        equity_values = [e['equity'] for e in equity_curve]
        max_drawdown = self._calculate_max_drawdown(equity_values)

        # By symbol performance
        symbol_performance = {}
        for t in trades:
            sym = t['symbol']
            if sym not in symbol_performance:
                symbol_performance[sym] = {"trades": 0, "pnl": 0, "wins": 0}
            symbol_performance[sym]["trades"] += 1
            symbol_performance[sym]["pnl"] += t['net_pnl']
            if t['net_pnl'] > 0:
                symbol_performance[sym]["wins"] += 1

        for sym in symbol_performance:
            symbol_performance[sym]["win_rate"] = round(
                symbol_performance[sym]["wins"] / symbol_performance[sym]["trades"] * 100, 1
            )
            symbol_performance[sym]["pnl"] = round(symbol_performance[sym]["pnl"], 2)

        return {
            "initial_capital": self.config.initial_capital,
            "final_equity": round(final_equity, 2),
            "total_return_pct": round(total_return, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "total_trades": len(trades),
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "total_transaction_costs": round(total_costs, 2),
            "trades": trades,
            "equity_curve": equity_curve[::max(1, len(equity_curve)//100)],  # Sample for response size
            "symbol_performance": symbol_performance,
            "config": {
                "max_positions": self.config.max_positions,
                "position_size_method": self.config.position_size_method,
                "max_position_pct": self.config.max_position_pct,
                "stop_loss_pct": self.config.stop_loss_pct,
                "take_profit_pct": self.config.take_profit_pct
            }
        }

    def _calculate_max_drawdown(self, equity: List[float]) -> float:
        """Calculate maximum drawdown percentage"""
        if not equity:
            return 0

        peak = equity[0]
        max_dd = 0

        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd

        return max_dd


# Global instances
_portfolio_backtester = None
_correlation_analyzer = None


def get_portfolio_backtester(config: Optional[PortfolioConfig] = None) -> PortfolioBacktester:
    """Get or create portfolio backtester instance"""
    global _portfolio_backtester
    if _portfolio_backtester is None or config is not None:
        _portfolio_backtester = PortfolioBacktester(config)
    return _portfolio_backtester


def get_correlation_analyzer() -> CorrelationAnalyzer:
    """Get or create correlation analyzer instance"""
    global _correlation_analyzer
    if _correlation_analyzer is None:
        _correlation_analyzer = CorrelationAnalyzer()
    return _correlation_analyzer
