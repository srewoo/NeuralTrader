"""
Backtesting Engine
Replays historical data to test signal alignment strategy performance.
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import ta

from data_providers.provider_manager import get_provider_manager

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtests the signal alignment strategy on historical OHLCV data.

    Strategy:
    - Compute technical indicators at each bar
    - Check alignment ratio (same logic as validator_agent.py)
    - BUY signal when alignment >= 0.7
    - SELL signal when alignment <= 0.3
    - Simulate trades: 5% target, 3% stop-loss, 20-day max hold
    """

    TARGET_PCT = 0.05    # 5% profit target
    STOP_LOSS_PCT = 0.03 # 3% stop loss
    MAX_HOLD_DAYS = 20   # Maximum holding period
    MIN_WARMUP = 200     # Minimum bars for indicator warmup

    async def run_backtest(
        self,
        symbol: str,
        period: str = "2y",
        api_keys: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Run backtest for a symbol over the specified period."""
        logger.info(f"Starting backtest for {symbol} over {period}")

        provider = get_provider_manager(api_keys)
        hist = await provider.get_historical_data(symbol, period=period, interval="1d")

        if hist is None or len(hist) < self.MIN_WARMUP + 50:
            return {
                "symbol": symbol,
                "error": f"Insufficient data (need {self.MIN_WARMUP + 50} bars, got {len(hist) if hist is not None else 0})",
                "total_signals": 0
            }

        trades = []
        i = self.MIN_WARMUP  # Start after warmup

        while i < len(hist):
            indicators = self._compute_indicators(hist, i)
            if indicators is None:
                i += 1
                continue

            # Check buy alignment
            buy_ratio = self._check_alignment(indicators, "BUY")
            sell_ratio = self._check_alignment(indicators, "SELL")

            signal = None
            if buy_ratio >= 0.7:
                signal = "BUY"
            elif sell_ratio >= 0.7:  # Note: sell_ratio is alignment for SELL recommendation
                signal = "SELL"

            if signal:
                trade = self._simulate_trade(hist, i, signal)
                if trade:
                    trades.append(trade)
                    # Skip past the trade holding period to avoid overlapping trades
                    i += trade["hold_days"] + 1
                    continue

            i += 1

        return self._calculate_stats(symbol, period, trades)

    def _compute_indicators(self, df: pd.DataFrame, idx: int) -> Optional[Dict[str, Any]]:
        """Compute technical indicators at a specific bar index."""
        try:
            if idx < 50:
                return None

            # Use data up to and including idx
            window = df.iloc[:idx + 1]
            close = window['Close']
            high = window['High']
            low = window['Low']
            volume = window['Volume']

            current_price = float(close.iloc[-1])

            # RSI
            rsi_series = ta.momentum.RSIIndicator(close, window=14).rsi()
            rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else None

            # MACD
            macd_ind = ta.trend.MACD(close)
            macd = float(macd_ind.macd().iloc[-1]) if not pd.isna(macd_ind.macd().iloc[-1]) else None
            macd_signal = float(macd_ind.macd_signal().iloc[-1]) if not pd.isna(macd_ind.macd_signal().iloc[-1]) else None

            # SMAs
            sma_20 = float(ta.trend.SMAIndicator(close, window=20).sma_indicator().iloc[-1])
            sma_50 = float(ta.trend.SMAIndicator(close, window=50).sma_indicator().iloc[-1])

            # Stochastic
            stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
            stochastic_k = float(stoch.stoch().iloc[-1]) if not pd.isna(stoch.stoch().iloc[-1]) else None

            # Volume ratio
            avg_vol_20 = float(volume.rolling(20).mean().iloc[-1])
            volume_ratio = float(volume.iloc[-1] / avg_vol_20) if avg_vol_20 > 0 else None

            return {
                "current_price": current_price,
                "rsi": rsi,
                "macd": macd,
                "macd_signal": macd_signal,
                "sma_20": sma_20,
                "sma_50": sma_50,
                "stochastic_k": stochastic_k,
                "volume_ratio": volume_ratio
            }
        except Exception as e:
            logger.debug(f"Indicator computation failed at idx {idx}: {e}")
            return None

    def _check_alignment(self, indicators: Dict[str, Any], recommendation: str) -> float:
        """
        Check signal alignment ratio — exact replica of validator_agent.py logic.
        Returns alignment_ratio (0.0 to 1.0).
        """
        alignment_count = 0
        total_checks = 0

        rsi = indicators.get("rsi")
        macd = indicators.get("macd")
        macd_signal = indicators.get("macd_signal")
        sma_20 = indicators.get("sma_20")
        sma_50 = indicators.get("sma_50")
        current_price = indicators.get("current_price")
        stochastic_k = indicators.get("stochastic_k")
        volume_ratio = indicators.get("volume_ratio")

        # RSI alignment
        if rsi is not None:
            total_checks += 1
            if recommendation == "BUY" and rsi < 70:
                alignment_count += 1
            elif recommendation == "SELL" and rsi > 30:
                alignment_count += 1
            elif recommendation == "HOLD":
                alignment_count += 1

        # MACD alignment
        if macd is not None and macd_signal is not None:
            total_checks += 1
            if recommendation == "BUY" and macd > macd_signal:
                alignment_count += 1
            elif recommendation == "SELL" and macd < macd_signal:
                alignment_count += 1
            elif recommendation == "HOLD":
                alignment_count += 1

        # Trend alignment
        if sma_20 and sma_50 and current_price:
            total_checks += 1
            if recommendation == "BUY" and (current_price > sma_20 or current_price > sma_50):
                alignment_count += 1
            elif recommendation == "SELL" and (current_price < sma_20 or current_price < sma_50):
                alignment_count += 1
            elif recommendation == "HOLD":
                alignment_count += 1

        # Stochastic alignment
        if stochastic_k is not None:
            total_checks += 1
            if recommendation == "BUY" and stochastic_k < 80:
                alignment_count += 1
            elif recommendation == "SELL" and stochastic_k > 20:
                alignment_count += 1
            elif recommendation == "HOLD":
                alignment_count += 1

        # Volume alignment
        if volume_ratio is not None:
            total_checks += 1
            if volume_ratio >= 0.7:
                alignment_count += 1

        if total_checks < 3:
            return 0.0

        return alignment_count / total_checks

    def _simulate_trade(self, df: pd.DataFrame, entry_idx: int, direction: str) -> Optional[Dict[str, Any]]:
        """Simulate a trade from entry_idx with 5% target, 3% stop, 20-day max."""
        if entry_idx >= len(df) - 1:
            return None

        entry_price = float(df['Close'].iloc[entry_idx])
        entry_date = str(df.index[entry_idx].date()) if hasattr(df.index[entry_idx], 'date') else str(df.index[entry_idx])

        if direction == "BUY":
            target = entry_price * (1 + self.TARGET_PCT)
            stop = entry_price * (1 - self.STOP_LOSS_PCT)
        else:  # SELL
            target = entry_price * (1 - self.TARGET_PCT)
            stop = entry_price * (1 + self.STOP_LOSS_PCT)

        # Walk forward through subsequent bars
        for offset in range(1, min(self.MAX_HOLD_DAYS + 1, len(df) - entry_idx)):
            bar_idx = entry_idx + offset
            bar_high = float(df['High'].iloc[bar_idx])
            bar_low = float(df['Low'].iloc[bar_idx])
            bar_close = float(df['Close'].iloc[bar_idx])
            exit_date = str(df.index[bar_idx].date()) if hasattr(df.index[bar_idx], 'date') else str(df.index[bar_idx])

            if direction == "BUY":
                # Check stop loss first (intrabar)
                if bar_low <= stop:
                    return self._trade_result(direction, entry_price, stop, entry_date, exit_date, offset, "stop_loss")
                # Check target
                if bar_high >= target:
                    return self._trade_result(direction, entry_price, target, entry_date, exit_date, offset, "target_hit")
            else:  # SELL
                if bar_high >= stop:
                    return self._trade_result(direction, entry_price, stop, entry_date, exit_date, offset, "stop_loss")
                if bar_low <= target:
                    return self._trade_result(direction, entry_price, target, entry_date, exit_date, offset, "target_hit")

        # Max hold reached — exit at last available close
        last_idx = min(entry_idx + self.MAX_HOLD_DAYS, len(df) - 1)
        exit_price = float(df['Close'].iloc[last_idx])
        exit_date = str(df.index[last_idx].date()) if hasattr(df.index[last_idx], 'date') else str(df.index[last_idx])
        hold_days = last_idx - entry_idx
        return self._trade_result(direction, entry_price, exit_price, entry_date, exit_date, hold_days, "max_hold")

    def _trade_result(self, direction, entry, exit_price, entry_date, exit_date, hold_days, exit_reason):
        """Build a trade result dict."""
        if direction == "BUY":
            return_pct = ((exit_price - entry) / entry) * 100
        else:
            return_pct = ((entry - exit_price) / entry) * 100

        return {
            "direction": direction,
            "entry_price": round(entry, 2),
            "exit_price": round(exit_price, 2),
            "entry_date": entry_date,
            "exit_date": exit_date,
            "hold_days": hold_days,
            "return_pct": round(return_pct, 2),
            "profitable": return_pct > 0,
            "exit_reason": exit_reason
        }

    def _calculate_stats(self, symbol: str, period: str, trades: List[Dict]) -> Dict[str, Any]:
        """Calculate backtest statistics from trade list."""
        if not trades:
            return {
                "symbol": symbol,
                "period": period,
                "total_signals": 0,
                "buy_signals": 0,
                "sell_signals": 0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "profit_factor": 0.0,
                "avg_hold_days": 0.0,
                "trades": []
            }

        buy_trades = [t for t in trades if t["direction"] == "BUY"]
        sell_trades = [t for t in trades if t["direction"] == "SELL"]
        winners = [t for t in trades if t["profitable"]]
        losers = [t for t in trades if not t["profitable"]]

        returns = [t["return_pct"] for t in trades]
        win_rate = (len(winners) / len(trades)) * 100
        avg_return = sum(returns) / len(returns)
        avg_hold = sum(t["hold_days"] for t in trades) / len(trades)

        # Profit factor
        gross_profit = sum(t["return_pct"] for t in winners) if winners else 0
        gross_loss = abs(sum(t["return_pct"] for t in losers)) if losers else 0.001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown (cumulative returns)
        cumulative = 0
        peak = 0
        max_dd = 0
        for t in trades:
            cumulative += t["return_pct"]
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (annualized, assuming ~252 trading days)
        if len(returns) >= 2:
            avg_r = np.mean(returns)
            std_r = np.std(returns, ddof=1)
            trades_per_year = 252 / avg_hold if avg_hold > 0 else 12
            sharpe = (avg_r / std_r) * np.sqrt(trades_per_year) if std_r > 0 else 0.0
        else:
            sharpe = 0.0

        return {
            "symbol": symbol,
            "period": period,
            "total_signals": len(trades),
            "buy_signals": len(buy_trades),
            "sell_signals": len(sell_trades),
            "win_rate": round(win_rate, 1),
            "avg_return": round(avg_return, 2),
            "max_drawdown": round(max_dd, 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "profit_factor": round(float(profit_factor), 2),
            "avg_hold_days": round(avg_hold, 1),
            "exit_reasons": {
                "target_hit": len([t for t in trades if t["exit_reason"] == "target_hit"]),
                "stop_loss": len([t for t in trades if t["exit_reason"] == "stop_loss"]),
                "max_hold": len([t for t in trades if t["exit_reason"] == "max_hold"])
            },
            "trades": trades
        }


# Singleton
_backtest_engine: Optional[BacktestEngine] = None


def get_backtest_engine() -> BacktestEngine:
    global _backtest_engine
    if _backtest_engine is None:
        _backtest_engine = BacktestEngine()
    return _backtest_engine
