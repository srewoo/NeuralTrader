"""
Regime-Specific Strategy Testing
Test strategies across different market conditions:
- Bull markets
- Bear markets
- Sideways/Range-bound
- High volatility
- Low volatility
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .strategies import Strategy, SignalType
from .price_cache import get_price_cache
from .walk_forward import IndianMarketCosts

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    SIDEWAYS = "sideways"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class RegimeDetector:
    """Detect market regimes from price data"""

    def __init__(
        self,
        lookback_trend: int = 50,
        lookback_volatility: int = 20
    ):
        self.lookback_trend = lookback_trend
        self.lookback_volatility = lookback_volatility

    def detect_regimes(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect market regime for each date in the data.

        Returns Series with regime labels.
        """
        regimes = pd.Series(index=data.index, dtype=str)

        # Calculate indicators
        close = data['Close']

        # Trend detection using SMA
        sma_short = close.rolling(20).mean()
        sma_long = close.rolling(50).mean()

        # Rate of change
        roc = close.pct_change(20) * 100

        # Volatility (ATR-based)
        high = data['High']
        low = data['Low']
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        atr_pct = (atr / close) * 100

        # Historical volatility percentile
        vol_20 = close.pct_change().rolling(20).std() * np.sqrt(252) * 100
        vol_percentile = vol_20.rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
            if len(x) >= 20 else 50,
            raw=False
        )

        for i in range(max(self.lookback_trend, 50), len(data)):
            date = data.index[i]

            # Trend regime
            if pd.isna(sma_short.iloc[i]) or pd.isna(sma_long.iloc[i]):
                trend = MarketRegime.SIDEWAYS
            elif close.iloc[i] > sma_short.iloc[i] > sma_long.iloc[i] and roc.iloc[i] > 10:
                trend = MarketRegime.STRONG_BULL
            elif close.iloc[i] > sma_short.iloc[i] and sma_short.iloc[i] > sma_long.iloc[i]:
                trend = MarketRegime.BULL
            elif close.iloc[i] < sma_short.iloc[i] < sma_long.iloc[i] and roc.iloc[i] < -10:
                trend = MarketRegime.STRONG_BEAR
            elif close.iloc[i] < sma_short.iloc[i] and sma_short.iloc[i] < sma_long.iloc[i]:
                trend = MarketRegime.BEAR
            else:
                trend = MarketRegime.SIDEWAYS

            # Volatility regime
            vol_pct = vol_percentile.iloc[i] if not pd.isna(vol_percentile.iloc[i]) else 50

            if vol_pct > 80:
                vol_regime = MarketRegime.HIGH_VOLATILITY
            elif vol_pct < 20:
                vol_regime = MarketRegime.LOW_VOLATILITY
            else:
                vol_regime = None

            # Combine - volatility takes precedence if extreme
            if vol_regime == MarketRegime.HIGH_VOLATILITY:
                regimes.iloc[i] = f"{trend.value}+high_vol"
            elif vol_regime == MarketRegime.LOW_VOLATILITY:
                regimes.iloc[i] = f"{trend.value}+low_vol"
            else:
                regimes.iloc[i] = trend.value

        return regimes

    def get_regime_periods(self, data: pd.DataFrame) -> Dict[str, List[Tuple[str, str]]]:
        """
        Get date ranges for each regime.

        Returns dict mapping regime to list of (start_date, end_date) tuples.
        """
        regimes = self.detect_regimes(data)

        periods = {}
        current_regime = None
        period_start = None

        for date, regime in regimes.items():
            if pd.isna(regime):
                continue

            if regime != current_regime:
                # End previous period
                if current_regime is not None:
                    if current_regime not in periods:
                        periods[current_regime] = []
                    periods[current_regime].append((
                        period_start.strftime("%Y-%m-%d"),
                        date.strftime("%Y-%m-%d")
                    ))

                # Start new period
                current_regime = regime
                period_start = date

        # Close final period
        if current_regime is not None:
            if current_regime not in periods:
                periods[current_regime] = []
            periods[current_regime].append((
                period_start.strftime("%Y-%m-%d"),
                data.index[-1].strftime("%Y-%m-%d")
            ))

        return periods


class RegimeStrategyTester:
    """Test strategies across different market regimes"""

    def __init__(self):
        self.price_cache = get_price_cache()
        self.regime_detector = RegimeDetector()
        self.costs = IndianMarketCosts()

    def test_strategy_by_regime(
        self,
        symbol: str,
        strategy: Strategy,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000
    ) -> Dict[str, Any]:
        """
        Test a strategy and break down performance by market regime.

        Args:
            symbol: Stock symbol
            strategy: Trading strategy
            start_date: Start date
            end_date: End date
            initial_capital: Starting capital

        Returns:
            Performance breakdown by regime
        """
        try:
            # Get data
            data = self.price_cache.get_prices(symbol, start_date, end_date)

            if data.empty or len(data) < 100:
                return {"error": "Insufficient data"}

            # Detect regimes
            regimes = self.regime_detector.detect_regimes(data)

            # Generate signals
            signals = strategy.generate_signals(data)

            # Run backtest and tag trades with regime
            trades, equity_curve = self._backtest_with_regimes(
                data, signals, regimes, initial_capital
            )

            # Analyze by regime
            regime_analysis = self._analyze_by_regime(trades, regimes)

            # Overall metrics
            final_equity = equity_curve[-1] if equity_curve else initial_capital
            total_return = (final_equity / initial_capital - 1) * 100

            # Regime distribution
            regime_counts = regimes.value_counts().to_dict()
            total_days = len(regimes.dropna())
            regime_distribution = {
                k: round(v / total_days * 100, 1)
                for k, v in regime_counts.items()
            }

            return {
                "symbol": symbol,
                "strategy": strategy.name,
                "period": f"{start_date} to {end_date}",
                "overall_metrics": {
                    "initial_capital": initial_capital,
                    "final_equity": round(final_equity, 2),
                    "total_return_pct": round(total_return, 2),
                    "total_trades": len(trades)
                },
                "regime_distribution": regime_distribution,
                "performance_by_regime": regime_analysis,
                "best_regime": self._find_best_regime(regime_analysis),
                "worst_regime": self._find_worst_regime(regime_analysis),
                "recommendations": self._generate_recommendations(regime_analysis)
            }

        except Exception as e:
            logger.error(f"Regime testing failed: {e}")
            return {"error": str(e)}

    def _backtest_with_regimes(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        regimes: pd.Series,
        initial_capital: float
    ) -> Tuple[List[Dict], List[float]]:
        """Run backtest and tag each trade with its regime"""

        cash = initial_capital
        position = 0
        entry_price = 0
        entry_date = None
        entry_regime = None
        trades = []
        equity_curve = []

        for date, row in data.iterrows():
            if date not in signals.index:
                equity_curve.append(cash + position * row['Close'])
                continue

            signal = signals.loc[date]
            price = row['Close']
            regime = regimes.loc[date] if date in regimes.index else None

            if signal == SignalType.BUY.value and position == 0:
                shares = int(cash * 0.95 / price)
                if shares > 0:
                    buy_cost = self.costs.calculate_buy_cost(shares * price)['total']
                    cash -= shares * price + buy_cost
                    position = shares
                    entry_price = price
                    entry_date = date
                    entry_regime = regime

            elif signal == SignalType.SELL.value and position > 0:
                sell_value = position * price
                sell_cost = self.costs.calculate_sell_cost(sell_value)['total']
                proceeds = sell_value - sell_cost
                cash += proceeds

                net_pnl = proceeds - (position * entry_price)
                pnl_pct = (price / entry_price - 1) * 100

                trades.append({
                    "entry_date": entry_date.strftime("%Y-%m-%d") if hasattr(entry_date, 'strftime') else str(entry_date),
                    "exit_date": date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(price, 2),
                    "shares": position,
                    "net_pnl": round(net_pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "entry_regime": entry_regime,
                    "exit_regime": regime
                })

                position = 0
                entry_price = 0
                entry_date = None
                entry_regime = None

            equity_curve.append(cash + position * price)

        # Close open position
        if position > 0:
            final_price = data.iloc[-1]['Close']
            sell_cost = self.costs.calculate_sell_cost(position * final_price)['total']
            proceeds = position * final_price - sell_cost
            cash += proceeds

            net_pnl = proceeds - (position * entry_price)
            pnl_pct = (final_price / entry_price - 1) * 100

            trades.append({
                "entry_date": entry_date.strftime("%Y-%m-%d") if hasattr(entry_date, 'strftime') else str(entry_date),
                "exit_date": "End",
                "entry_price": round(entry_price, 2),
                "exit_price": round(final_price, 2),
                "shares": position,
                "net_pnl": round(net_pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "entry_regime": entry_regime,
                "exit_regime": regimes.iloc[-1] if len(regimes) > 0 else None
            })

            equity_curve[-1] = cash

        return trades, equity_curve

    def _analyze_by_regime(
        self,
        trades: List[Dict],
        regimes: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze performance by regime"""

        regime_trades = {}

        for trade in trades:
            regime = trade.get('entry_regime')
            if regime is None:
                continue

            # Simplify regime name (remove volatility suffix for grouping)
            base_regime = regime.split('+')[0]

            if base_regime not in regime_trades:
                regime_trades[base_regime] = []
            regime_trades[base_regime].append(trade)

        analysis = {}

        for regime, rtrades in regime_trades.items():
            if not rtrades:
                continue

            pnls = [t['net_pnl'] for t in rtrades]
            pnl_pcts = [t['pnl_pct'] for t in rtrades]

            winning = [t for t in rtrades if t['net_pnl'] > 0]
            losing = [t for t in rtrades if t['net_pnl'] <= 0]

            analysis[regime] = {
                "total_trades": len(rtrades),
                "winning_trades": len(winning),
                "losing_trades": len(losing),
                "win_rate": round(len(winning) / len(rtrades) * 100, 1),
                "total_pnl": round(sum(pnls), 2),
                "avg_pnl": round(np.mean(pnls), 2),
                "avg_pnl_pct": round(np.mean(pnl_pcts), 2),
                "best_trade": round(max(pnl_pcts), 2),
                "worst_trade": round(min(pnl_pcts), 2),
                "profit_factor": round(
                    sum(t['net_pnl'] for t in winning) / abs(sum(t['net_pnl'] for t in losing))
                    if losing and sum(t['net_pnl'] for t in losing) != 0 else 0,
                    2
                )
            }

        return analysis

    def _find_best_regime(self, analysis: Dict) -> Dict[str, Any]:
        """Find the regime where strategy performs best"""
        if not analysis:
            return {"regime": None, "reason": "No data"}

        best = max(analysis.items(), key=lambda x: x[1].get('avg_pnl_pct', 0))
        return {
            "regime": best[0],
            "avg_return": best[1]['avg_pnl_pct'],
            "win_rate": best[1]['win_rate'],
            "trades": best[1]['total_trades']
        }

    def _find_worst_regime(self, analysis: Dict) -> Dict[str, Any]:
        """Find the regime where strategy performs worst"""
        if not analysis:
            return {"regime": None, "reason": "No data"}

        worst = min(analysis.items(), key=lambda x: x[1].get('avg_pnl_pct', 0))
        return {
            "regime": worst[0],
            "avg_return": worst[1]['avg_pnl_pct'],
            "win_rate": worst[1]['win_rate'],
            "trades": worst[1]['total_trades']
        }

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on regime analysis"""
        recommendations = []

        for regime, metrics in analysis.items():
            if metrics['win_rate'] < 40:
                recommendations.append(
                    f"Consider disabling strategy during {regime} regimes (win rate: {metrics['win_rate']}%)"
                )

            if metrics['avg_pnl_pct'] < -2:
                recommendations.append(
                    f"Strategy loses money in {regime} markets - add regime filter"
                )

            if metrics['win_rate'] > 60 and metrics['avg_pnl_pct'] > 3:
                recommendations.append(
                    f"Strategy excels in {regime} markets - consider increasing position size"
                )

        # Check for regime sensitivity
        if analysis:
            returns = [m['avg_pnl_pct'] for m in analysis.values()]
            if max(returns) - min(returns) > 10:
                recommendations.append(
                    "Strategy is highly regime-sensitive - consider adding regime detection filter"
                )

        if not recommendations:
            recommendations.append("Strategy shows consistent performance across regimes")

        return recommendations


# Global instances
_regime_tester = None


def get_regime_tester() -> RegimeStrategyTester:
    """Get or create regime tester instance"""
    global _regime_tester
    if _regime_tester is None:
        _regime_tester = RegimeStrategyTester()
    return _regime_tester
