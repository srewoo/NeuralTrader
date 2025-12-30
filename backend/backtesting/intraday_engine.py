"""
Intraday Backtesting Engine
Supports 1m, 5m, 15m, 30m, 1h timeframes with intraday-specific cost modeling
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, time
from dataclasses import dataclass, field
import logging
from enum import Enum

from .strategies import Strategy, SignalType
from .metrics import PerformanceMetrics
from .spread_model import SpreadModel, OrderSide, get_spread_model_for_stock

logger = logging.getLogger(__name__)


class TimeFrame(Enum):
    """Supported intraday timeframes"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    DAY_1 = "1d"


@dataclass
class IntradayCosts:
    """
    Indian market costs for intraday trading.
    Lower STT than delivery, no DP charges.
    """
    # Brokerage (flat fee or percentage, whichever is lower)
    brokerage_pct: float = 0.0003  # 0.03% (Zerodha style)
    brokerage_min: float = 0.0
    brokerage_max: float = 20.0  # Rs 20 cap per order

    # STT for intraday (only on sell side)
    stt_sell: float = 0.00025  # 0.025% (vs 0.1% for delivery)

    # Exchange charges
    nse_charge: float = 0.0000297  # 0.00297%

    # GST on brokerage
    gst_rate: float = 0.18  # 18%

    # SEBI charges
    sebi_charge: float = 0.000001  # Rs 10 per crore

    # Stamp duty (only on buy)
    stamp_duty_buy: float = 0.00003  # 0.003% for intraday

    # Slippage (higher for intraday due to volatility)
    slippage: float = 0.001  # 0.1%

    def calculate_buy_cost(self, price: float, quantity: int) -> Dict[str, float]:
        """Calculate all costs for a buy order"""
        turnover = price * quantity

        brokerage = min(turnover * self.brokerage_pct, self.brokerage_max)
        exchange = turnover * self.nse_charge
        gst = (brokerage + exchange) * self.gst_rate
        sebi = turnover * self.sebi_charge
        stamp = turnover * self.stamp_duty_buy
        slippage_cost = turnover * self.slippage

        total = brokerage + exchange + gst + sebi + stamp + slippage_cost

        return {
            "brokerage": brokerage,
            "exchange_charges": exchange,
            "gst": gst,
            "sebi_charges": sebi,
            "stamp_duty": stamp,
            "slippage": slippage_cost,
            "total": total,
            "total_pct": (total / turnover) * 100 if turnover > 0 else 0
        }

    def calculate_sell_cost(self, price: float, quantity: int) -> Dict[str, float]:
        """Calculate all costs for a sell order"""
        turnover = price * quantity

        brokerage = min(turnover * self.brokerage_pct, self.brokerage_max)
        stt = turnover * self.stt_sell
        exchange = turnover * self.nse_charge
        gst = (brokerage + exchange) * self.gst_rate
        sebi = turnover * self.sebi_charge
        slippage_cost = turnover * self.slippage

        total = brokerage + stt + exchange + gst + sebi + slippage_cost

        return {
            "brokerage": brokerage,
            "stt": stt,
            "exchange_charges": exchange,
            "gst": gst,
            "sebi_charges": sebi,
            "slippage": slippage_cost,
            "total": total,
            "total_pct": (total / turnover) * 100 if turnover > 0 else 0
        }

    def calculate_roundtrip_cost(self, price: float, quantity: int) -> float:
        """Calculate total cost for a complete buy-sell roundtrip"""
        buy_costs = self.calculate_buy_cost(price, quantity)
        sell_costs = self.calculate_sell_cost(price, quantity)
        return buy_costs["total"] + sell_costs["total"]


@dataclass
class IntradaySession:
    """Indian market session timings"""
    pre_open_start: time = field(default_factory=lambda: time(9, 0))
    pre_open_end: time = field(default_factory=lambda: time(9, 8))
    market_open: time = field(default_factory=lambda: time(9, 15))
    market_close: time = field(default_factory=lambda: time(15, 30))
    post_market_end: time = field(default_factory=lambda: time(16, 0))

    # Trading restrictions
    no_fresh_positions_after: time = field(default_factory=lambda: time(15, 15))

    def is_trading_hours(self, t: time) -> bool:
        """Check if time is within trading hours"""
        return self.market_open <= t <= self.market_close

    def can_open_position(self, t: time) -> bool:
        """Check if new positions can be opened"""
        return self.market_open <= t <= self.no_fresh_positions_after


class IntradayBacktestEngine:
    """
    Backtesting engine optimized for intraday trading strategies.

    Features:
    - Multiple timeframe support (1m, 5m, 15m, 30m, 1h)
    - Intraday-specific cost model (lower STT, no DP charges)
    - Session time enforcement
    - Auto square-off at session end
    - Per-day performance tracking
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        interval: str = "5m",
        cost_model: Optional[IntradayCosts] = None,
        spread_model: Optional[SpreadModel] = None,
        session: Optional[IntradaySession] = None,
        auto_square_off: bool = True,
        max_trades_per_day: int = 10
    ):
        """
        Initialize intraday backtest engine.

        Args:
            initial_capital: Starting capital
            interval: Timeframe (1m, 5m, 15m, 30m, 1h)
            cost_model: Intraday cost model
            spread_model: Bid-ask spread model
            session: Market session timings
            auto_square_off: Auto close positions at session end
            max_trades_per_day: Maximum trades allowed per day
        """
        self.initial_capital = initial_capital
        self.interval = TimeFrame(interval)
        self.cost_model = cost_model or IntradayCosts()
        self.spread_model = spread_model or get_spread_model_for_stock()
        self.session = session or IntradaySession()
        self.auto_square_off = auto_square_off
        self.max_trades_per_day = max_trades_per_day

    def run_intraday_backtest(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        position_size: float = 1.0
    ) -> Dict[str, Any]:
        """
        Run intraday backtest on provided data.

        Args:
            data: Intraday OHLCV data with DatetimeIndex
            strategy: Trading strategy
            position_size: Fraction of capital per trade

        Returns:
            Backtest results with metrics
        """
        if data.empty:
            raise ValueError("No data provided for backtest")

        logger.info(f"Running intraday backtest with {self.interval.value} data")

        # Generate signals
        signals = strategy.generate_signals(data)

        # Simulate trades with intraday rules
        trades, equity_curve, daily_pnl = self._simulate_intraday_trades(
            data, signals, position_size
        )

        # Calculate metrics
        returns = equity_curve.pct_change().fillna(0)

        metrics = PerformanceMetrics.calculate_all_metrics(
            equity_curve=equity_curve,
            returns=returns,
            trades=trades,
            initial_capital=self.initial_capital
        )

        # Add intraday-specific metrics
        metrics["interval"] = self.interval.value
        metrics["trading_days"] = len(daily_pnl)
        metrics["avg_trades_per_day"] = len(trades) / max(len(daily_pnl), 1)
        metrics["profitable_days"] = sum(1 for pnl in daily_pnl.values() if pnl > 0)
        metrics["losing_days"] = sum(1 for pnl in daily_pnl.values() if pnl < 0)
        metrics["daily_win_rate"] = (
            metrics["profitable_days"] / metrics["trading_days"] * 100
            if metrics["trading_days"] > 0 else 0
        )

        return {
            "strategy": strategy.name,
            "interval": self.interval.value,
            "metrics": metrics,
            "trades": trades.to_dict('records') if not trades.empty else [],
            "equity_curve": equity_curve.to_dict(),
            "daily_pnl": daily_pnl
        }

    def _simulate_intraday_trades(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        position_size: float
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, float]]:
        """
        Simulate intraday trades with session rules.

        Returns:
            Tuple of (trades_df, equity_curve, daily_pnl_dict)
        """
        cash = self.initial_capital
        position = 0
        trades = []
        equity_curve = []
        daily_pnl = {}

        entry_price = None
        entry_time = None
        current_date = None
        trades_today = 0

        # Pre-calculate volume stats for spread model
        avg_volume = data['Volume'].mean() if 'Volume' in data.columns else None

        for timestamp, row in data.iterrows():
            signal = signals.loc[timestamp]
            price = row['Close']
            volume = row.get('Volume', None)

            # Extract date and time
            if hasattr(timestamp, 'date'):
                trade_date = timestamp.date()
                trade_time = timestamp.time()
            else:
                trade_date = timestamp
                trade_time = time(12, 0)  # Default noon if no time

            # Reset daily counter on new day
            if current_date != trade_date:
                # Auto square-off at end of previous day
                if self.auto_square_off and position > 0 and current_date is not None:
                    # Force close position
                    exec_price = self.spread_model.get_execution_price(
                        price, OrderSide.SELL, volume, avg_volume, 0.20
                    )
                    sell_costs = self.cost_model.calculate_sell_cost(exec_price, position)
                    proceeds = position * exec_price - sell_costs["total"]
                    pnl = proceeds - (position * entry_price)

                    trades.append({
                        "entry_time": entry_time,
                        "exit_time": timestamp,
                        "entry_price": entry_price,
                        "exit_price": exec_price,
                        "shares": position,
                        "pnl": pnl,
                        "pnl_pct": (pnl / (position * entry_price)) * 100,
                        "exit_reason": "auto_square_off"
                    })

                    cash += proceeds
                    position = 0
                    entry_price = None

                current_date = trade_date
                trades_today = 0

            # Check session timing
            if not self.session.is_trading_hours(trade_time):
                equity_curve.append(cash + position * price)
                continue

            # Check if we can open new positions
            can_trade = (
                self.session.can_open_position(trade_time) and
                trades_today < self.max_trades_per_day
            )

            # Get execution price with spread
            if signal == SignalType.BUY.value:
                exec_price = self.spread_model.get_execution_price(
                    price, OrderSide.BUY, volume, avg_volume, 0.20
                )
            elif signal == SignalType.SELL.value:
                exec_price = self.spread_model.get_execution_price(
                    price, OrderSide.SELL, volume, avg_volume, 0.20
                )
            else:
                exec_price = price

            # Execute trades
            if signal == SignalType.BUY.value and position == 0 and can_trade:
                # Calculate costs
                capital_to_use = cash * position_size
                buy_costs = self.cost_model.calculate_buy_cost(exec_price, 1)
                cost_per_share = exec_price + buy_costs["total"]

                shares_to_buy = int(capital_to_use / cost_per_share)

                if shares_to_buy > 0:
                    total_costs = self.cost_model.calculate_buy_cost(exec_price, shares_to_buy)
                    total_cost = shares_to_buy * exec_price + total_costs["total"]

                    cash -= total_cost
                    position = shares_to_buy
                    entry_price = exec_price
                    entry_time = timestamp
                    trades_today += 1

                    logger.debug(f"BUY {shares_to_buy} @ {exec_price:.2f}")

            elif signal == SignalType.SELL.value and position > 0:
                # Sell position
                sell_costs = self.cost_model.calculate_sell_cost(exec_price, position)
                proceeds = position * exec_price - sell_costs["total"]
                pnl = proceeds - (position * entry_price)
                pnl_pct = (pnl / (position * entry_price)) * 100

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": timestamp,
                    "entry_price": entry_price,
                    "exit_price": exec_price,
                    "shares": position,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "costs": sell_costs["total"],
                    "exit_reason": "signal"
                })

                # Track daily PnL
                date_str = str(trade_date)
                daily_pnl[date_str] = daily_pnl.get(date_str, 0) + pnl

                cash += proceeds
                position = 0
                entry_price = None
                trades_today += 1

                logger.debug(f"SELL @ {exec_price:.2f}, PnL: {pnl:.2f}")

            # Track equity
            equity_curve.append(cash + position * price)

        # Final square-off if needed
        if position > 0:
            final_price = data.iloc[-1]['Close']
            exec_price = self.spread_model.get_execution_price(
                final_price, OrderSide.SELL, None, avg_volume, 0.20
            )
            sell_costs = self.cost_model.calculate_sell_cost(exec_price, position)
            proceeds = position * exec_price - sell_costs["total"]
            pnl = proceeds - (position * entry_price)

            trades.append({
                "entry_time": entry_time,
                "exit_time": data.index[-1],
                "entry_price": entry_price,
                "exit_price": exec_price,
                "shares": position,
                "pnl": pnl,
                "pnl_pct": (pnl / (position * entry_price)) * 100,
                "exit_reason": "end_of_backtest"
            })

            cash += proceeds
            equity_curve[-1] = cash

        # Create DataFrames
        trades_df = pd.DataFrame(trades)
        equity_series = pd.Series(equity_curve, index=data.index)

        return trades_df, equity_series, daily_pnl


def fetch_intraday_data(
    symbol: str,
    interval: str = "5m",
    days: int = 30,
    provider: str = "yfinance"
) -> pd.DataFrame:
    """
    Fetch intraday data for backtesting.

    Args:
        symbol: Stock symbol
        interval: Timeframe (1m, 5m, 15m, 30m, 1h)
        days: Number of days of data
        provider: Data provider (yfinance, twelvedata, alpaca)

    Returns:
        DataFrame with OHLCV data
    """
    if provider == "yfinance":
        import yfinance as yf

        # yfinance limits: 1m (7 days), 5m/15m/30m (60 days), 1h (730 days)
        if interval == "1m" and days > 7:
            days = 7
        elif interval in ["5m", "15m", "30m"] and days > 60:
            days = 60

        ticker = yf.Ticker(symbol)
        data = ticker.history(period=f"{days}d", interval=interval)

        return data

    else:
        raise ValueError(f"Provider {provider} not supported for intraday data")
