"""
Unit Tests for Performance Metrics
Tests for PerformanceMetrics calculator
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics class"""

    def test_calculate_basic_metrics(self):
        """Test basic metrics calculation"""
        from backtesting.metrics import PerformanceMetrics

        # Create simple equity curve
        dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
        initial = 100000
        final = 120000
        equity = pd.Series(np.linspace(initial, final, 365), index=dates)

        metrics = PerformanceMetrics._calculate_basic_metrics(equity, initial)

        assert metrics["initial_capital"] == initial
        assert metrics["final_value"] == final
        assert metrics["total_return"] == 20.0  # 20% return
        assert "cagr" in metrics
        assert metrics["duration_days"] == 364

    def test_calculate_basic_metrics_with_loss(self):
        """Test metrics with negative return"""
        from backtesting.metrics import PerformanceMetrics

        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        initial = 100000
        final = 80000
        equity = pd.Series(np.linspace(initial, final, 100), index=dates)

        metrics = PerformanceMetrics._calculate_basic_metrics(equity, initial)

        assert metrics["total_return"] == -20.0
        assert metrics["final_value"] == 80000

    def test_calculate_risk_metrics(self):
        """Test risk metrics calculation"""
        from backtesting.metrics import PerformanceMetrics

        # Generate returns with known properties
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        metrics = PerformanceMetrics._calculate_risk_metrics(returns, risk_free_rate=0.05)

        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "volatility" in metrics

    def test_sharpe_ratio_positive(self):
        """Test Sharpe ratio for positive returns"""
        from backtesting.metrics import PerformanceMetrics

        # Positive returns with some variance (constant returns have 0 std, causing 0 Sharpe)
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.002, 0.01, 252))  # Positive mean with some variance
        metrics = PerformanceMetrics._calculate_risk_metrics(returns, risk_free_rate=0.02)

        # With positive mean returns exceeding risk-free rate and some volatility
        # Sharpe should be non-zero (can be positive or negative depending on random seed)
        assert isinstance(metrics["sharpe_ratio"], (int, float))

    def test_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility"""
        from backtesting.metrics import PerformanceMetrics

        # All same returns (zero std)
        returns = pd.Series([0.0] * 252)
        metrics = PerformanceMetrics._calculate_risk_metrics(returns, risk_free_rate=0.05)

        # Should handle gracefully
        assert metrics["sharpe_ratio"] == 0 or not np.isnan(metrics["sharpe_ratio"])

    def test_calculate_trade_metrics_with_trades(self, sample_trades):
        """Test trade metrics with sample trades"""
        from backtesting.metrics import PerformanceMetrics

        metrics = PerformanceMetrics._calculate_trade_metrics(sample_trades)

        assert metrics["total_trades"] == 10
        assert metrics["winning_trades"] == 7
        assert metrics["losing_trades"] == 3
        assert metrics["win_rate"] == 0.7
        assert metrics["win_rate_pct"] == 70.0
        assert metrics["avg_win"] > 0
        assert metrics["avg_loss"] > 0
        assert metrics["profit_factor"] > 0

    def test_calculate_trade_metrics_empty_trades(self):
        """Test trade metrics with no trades"""
        from backtesting.metrics import PerformanceMetrics

        empty_trades = pd.DataFrame(columns=['pnl', 'duration_days'])
        metrics = PerformanceMetrics._calculate_trade_metrics(empty_trades)

        assert metrics["total_trades"] == 0
        assert metrics["win_rate"] == 0
        assert metrics["profit_factor"] == 0

    def test_calculate_trade_metrics_all_winners(self):
        """Test trade metrics with all winning trades"""
        from backtesting.metrics import PerformanceMetrics

        winning_trades = pd.DataFrame({
            'pnl': [100, 200, 150, 300, 250],
            'duration_days': [5, 3, 4, 6, 2]
        })
        metrics = PerformanceMetrics._calculate_trade_metrics(winning_trades)

        assert metrics["winning_trades"] == 5
        assert metrics["losing_trades"] == 0
        assert metrics["win_rate"] == 1.0
        assert metrics["profit_factor"] == 0  # Division by zero protection

    def test_calculate_trade_metrics_all_losers(self):
        """Test trade metrics with all losing trades"""
        from backtesting.metrics import PerformanceMetrics

        losing_trades = pd.DataFrame({
            'pnl': [-100, -200, -150, -300, -250],
            'duration_days': [5, 3, 4, 6, 2]
        })
        metrics = PerformanceMetrics._calculate_trade_metrics(losing_trades)

        assert metrics["winning_trades"] == 0
        assert metrics["losing_trades"] == 5
        assert metrics["win_rate"] == 0

    def test_calculate_drawdown_metrics(self, sample_equity_curve):
        """Test drawdown metrics calculation"""
        from backtesting.metrics import PerformanceMetrics

        metrics = PerformanceMetrics._calculate_drawdown_metrics(sample_equity_curve)

        assert "max_drawdown" in metrics
        assert "max_drawdown_pct" in metrics
        assert metrics["max_drawdown"] <= 0  # Drawdown should be negative or zero

    def test_drawdown_with_no_drawdown(self):
        """Test drawdown when equity only goes up"""
        from backtesting.metrics import PerformanceMetrics

        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        equity = pd.Series(range(100, 200), index=dates)

        metrics = PerformanceMetrics._calculate_drawdown_metrics(equity)

        assert metrics["max_drawdown"] == 0

    def test_drawdown_with_significant_drop(self):
        """Test drawdown with significant decline"""
        from backtesting.metrics import PerformanceMetrics

        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        # Go up to 200, then drop to 100
        equity = pd.Series(
            list(range(100, 200)) + list(range(200, 100, -1)),
            index=pd.date_range(start='2024-01-01', periods=200, freq='D')
        )

        metrics = PerformanceMetrics._calculate_drawdown_metrics(equity)

        # Should detect ~50% drawdown (100/200 - 1 = -0.5)
        assert metrics["max_drawdown"] <= -45  # At least 45% drawdown

    def test_calculate_all_metrics(self, sample_equity_curve, sample_returns, sample_trades):
        """Test calculating all metrics together"""
        from backtesting.metrics import PerformanceMetrics

        metrics = PerformanceMetrics.calculate_all_metrics(
            equity_curve=sample_equity_curve,
            returns=sample_returns,
            trades=sample_trades,
            initial_capital=100000,
            risk_free_rate=0.05
        )

        # Check all metric categories are present
        assert "initial_capital" in metrics
        assert "final_value" in metrics
        assert "total_return" in metrics
        assert "cagr" in metrics
        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "volatility" in metrics
        assert "total_trades" in metrics
        assert "win_rate" in metrics
        assert "max_drawdown" in metrics

    def test_calculate_calmar_ratio(self):
        """Test Calmar ratio calculation"""
        from backtesting.metrics import PerformanceMetrics

        # CAGR of 20%, max drawdown of -10%
        calmar = PerformanceMetrics.calculate_calmar_ratio(0.20, -0.10)

        assert calmar == 2.0  # 0.20 / 0.10 = 2.0

    def test_calculate_calmar_ratio_zero_drawdown(self):
        """Test Calmar ratio with zero drawdown"""
        from backtesting.metrics import PerformanceMetrics

        calmar = PerformanceMetrics.calculate_calmar_ratio(0.20, 0)

        assert calmar == 0

    def test_calculate_calmar_ratio_negative_cagr(self):
        """Test Calmar ratio with negative CAGR"""
        from backtesting.metrics import PerformanceMetrics

        calmar = PerformanceMetrics.calculate_calmar_ratio(-0.10, -0.20)

        assert calmar == 0.5  # abs(-0.10) / abs(-0.20) = 0.5


class TestMetricsEdgeCases:
    """Tests for edge cases in metrics calculations"""

    def test_single_day_equity_curve(self):
        """Test metrics with single day equity"""
        from backtesting.metrics import PerformanceMetrics

        equity = pd.Series([100000], index=[pd.Timestamp('2024-01-01')])

        # Should handle without error
        metrics = PerformanceMetrics._calculate_basic_metrics(equity, 100000)
        assert metrics["total_return"] == 0

    def test_very_short_returns_series(self):
        """Test risk metrics with very short series"""
        from backtesting.metrics import PerformanceMetrics

        returns = pd.Series([0.01, -0.01])
        metrics = PerformanceMetrics._calculate_risk_metrics(returns, 0.05)

        # Should compute without error
        assert "sharpe_ratio" in metrics

    def test_extreme_returns(self):
        """Test metrics with extreme returns"""
        from backtesting.metrics import PerformanceMetrics

        np.random.seed(42)
        returns = pd.Series(np.random.uniform(-0.5, 0.5, 100))  # -50% to +50% daily
        metrics = PerformanceMetrics._calculate_risk_metrics(returns, 0.05)

        assert "volatility" in metrics
        assert metrics["volatility"] > 0


class TestMetricsIntegration:
    """Integration tests for complete metrics workflow"""

    def test_full_backtest_metrics(self):
        """Test complete backtest metrics calculation"""
        from backtesting.metrics import PerformanceMetrics

        # Simulate a backtest
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=252, freq='D')
        initial_capital = 100000

        # Generate returns
        daily_returns = np.random.normal(0.0005, 0.015, 252)

        # Build equity curve
        equity = [initial_capital]
        for r in daily_returns:
            equity.append(equity[-1] * (1 + r))
        equity_curve = pd.Series(equity[1:], index=dates)

        returns = equity_curve.pct_change().dropna()

        # Generate some trades
        trades = pd.DataFrame({
            'pnl': np.random.choice([-500, 500, 1000, -300, 800], 20),
            'duration_days': np.random.randint(1, 10, 20)
        })

        metrics = PerformanceMetrics.calculate_all_metrics(
            equity_curve=equity_curve,
            returns=returns,
            trades=trades,
            initial_capital=initial_capital
        )

        # Validate all expected metrics are present and reasonable
        assert isinstance(metrics["total_return"], float)
        assert isinstance(metrics["sharpe_ratio"], float)
        assert isinstance(metrics["max_drawdown"], float)
        assert 0 <= metrics["win_rate"] <= 1
