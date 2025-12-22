"""
Risk Management Module
Position sizing, stop-loss, drawdown limits, Kelly Criterion
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Comprehensive risk management for trading
    """

    def __init__(
        self,
        max_position_size: float = 0.05,  # 5% max per stock
        max_sector_exposure: float = 0.20,  # 20% max per sector
        max_portfolio_leverage: float = 1.0,  # No leverage by default
        max_drawdown_limit: float = 0.15,  # 15% max drawdown
        kelly_fraction: float = 0.25,  # Quarter Kelly for safety
        default_stop_loss_pct: float = 0.05,  # 5% stop loss
        atr_multiplier: float = 2.0,  # 2x ATR for stop loss
    ):
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.max_portfolio_leverage = max_portfolio_leverage
        self.max_drawdown_limit = max_drawdown_limit
        self.kelly_fraction = kelly_fraction
        self.default_stop_loss_pct = default_stop_loss_pct
        self.atr_multiplier = atr_multiplier

        self.peak_portfolio_value = 0.0
        self.current_drawdown = 0.0

    def calculate_position_size_kelly(
        self,
        portfolio_value: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion

        Kelly % = (Win Rate * Avg Win - (1 - Win Rate) * Avg Loss) / Avg Win

        Args:
            portfolio_value: Total portfolio value
            win_rate: Historical win rate (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount (positive number)

        Returns:
            Position size in currency
        """
        if win_rate <= 0 or win_rate >= 1:
            logger.warning(f"Invalid win rate: {win_rate}")
            return 0.0

        if avg_win <= 0 or avg_loss <= 0:
            logger.warning(f"Invalid avg win/loss: {avg_win}, {avg_loss}")
            return 0.0

        # Kelly formula
        kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

        # Apply fraction for safety (quarter Kelly is conservative)
        adjusted_kelly = kelly_pct * self.kelly_fraction

        # Ensure it's within limits
        adjusted_kelly = max(0.0, min(adjusted_kelly, self.max_position_size))

        position_size = portfolio_value * adjusted_kelly

        logger.info(f"Kelly position size: ${position_size:,.2f} ({adjusted_kelly*100:.2f}% of portfolio)")
        return position_size

    def calculate_position_size_fixed_fractional(
        self,
        portfolio_value: float,
        risk_per_trade: float = 0.02,  # 2% risk per trade
        entry_price: float = 0.0,
        stop_loss_price: float = 0.0
    ) -> int:
        """
        Calculate position size using fixed fractional method

        Position Size = (Portfolio Value * Risk %) / (Entry Price - Stop Loss Price)

        Args:
            portfolio_value: Total portfolio value
            risk_per_trade: Percentage of portfolio to risk (0.02 = 2%)
            entry_price: Entry price per share
            stop_loss_price: Stop loss price per share

        Returns:
            Number of shares to buy
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            logger.error("Invalid entry or stop loss price")
            return 0

        if stop_loss_price >= entry_price:
            logger.error("Stop loss must be below entry price")
            return 0

        risk_amount = portfolio_value * risk_per_trade
        price_risk_per_share = entry_price - stop_loss_price

        shares = int(risk_amount / price_risk_per_share)

        # Ensure position doesn't exceed max position size
        max_shares = int((portfolio_value * self.max_position_size) / entry_price)
        shares = min(shares, max_shares)

        position_value = shares * entry_price
        position_pct = (position_value / portfolio_value) * 100

        logger.info(f"Fixed fractional position: {shares} shares @ ${entry_price:.2f} = ${position_value:,.2f} ({position_pct:.2f}%)")
        return shares

    def calculate_position_size_volatility_based(
        self,
        portfolio_value: float,
        target_volatility: float,
        stock_volatility: float,
        stock_price: float
    ) -> int:
        """
        Calculate position size based on volatility

        Position Size = (Portfolio Value * Target Vol) / Stock Vol

        Args:
            portfolio_value: Total portfolio value
            target_volatility: Target portfolio volatility (e.g., 0.10 = 10%)
            stock_volatility: Stock's annualized volatility
            stock_price: Current stock price

        Returns:
            Number of shares to buy
        """
        if stock_volatility <= 0:
            logger.error("Invalid stock volatility")
            return 0

        position_value = (portfolio_value * target_volatility) / stock_volatility
        shares = int(position_value / stock_price)

        # Ensure within max position size
        max_shares = int((portfolio_value * self.max_position_size) / stock_price)
        shares = min(shares, max_shares)

        logger.info(f"Volatility-based position: {shares} shares (stock vol: {stock_volatility*100:.1f}%)")
        return shares

    def calculate_stop_loss_atr(
        self,
        entry_price: float,
        atr: float,
        side: str = "BUY"
    ) -> float:
        """
        Calculate stop loss using ATR (Average True Range)

        Args:
            entry_price: Entry price
            atr: Average True Range value
            side: BUY or SELL

        Returns:
            Stop loss price
        """
        if side == "BUY":
            stop_loss = entry_price - (atr * self.atr_multiplier)
        else:
            stop_loss = entry_price + (atr * self.atr_multiplier)

        logger.info(f"ATR stop loss: ${stop_loss:.2f} (ATR: ${atr:.2f}, multiplier: {self.atr_multiplier}x)")
        return stop_loss

    def calculate_stop_loss_percentage(
        self,
        entry_price: float,
        stop_loss_pct: Optional[float] = None,
        side: str = "BUY"
    ) -> float:
        """
        Calculate stop loss as percentage below entry

        Args:
            entry_price: Entry price
            stop_loss_pct: Stop loss percentage (default: self.default_stop_loss_pct)
            side: BUY or SELL

        Returns:
            Stop loss price
        """
        pct = stop_loss_pct or self.default_stop_loss_pct

        if side == "BUY":
            stop_loss = entry_price * (1 - pct)
        else:
            stop_loss = entry_price * (1 + pct)

        logger.info(f"Percentage stop loss: ${stop_loss:.2f} ({pct*100:.1f}% from entry)")
        return stop_loss

    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,
        trailing_pct: float = 0.05,
        side: str = "BUY"
    ) -> Tuple[float, bool]:
        """
        Calculate trailing stop loss

        Args:
            entry_price: Original entry price
            current_price: Current market price
            highest_price: Highest price since entry
            trailing_pct: Trailing percentage (default: 5%)
            side: BUY or SELL

        Returns:
            (stop_loss_price, should_trigger)
        """
        if side == "BUY":
            stop_loss = highest_price * (1 - trailing_pct)
            should_trigger = current_price <= stop_loss
        else:
            stop_loss = highest_price * (1 + trailing_pct)
            should_trigger = current_price >= stop_loss

        return stop_loss, should_trigger

    def check_position_size_limits(
        self,
        symbol: str,
        position_value: float,
        portfolio_value: float,
        sector: Optional[str] = None,
        sector_positions: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, str]:
        """
        Check if position size is within limits

        Args:
            symbol: Stock symbol
            position_value: Value of proposed position
            portfolio_value: Total portfolio value
            sector: Sector of the stock (optional)
            sector_positions: Dict of sector -> total value (optional)

        Returns:
            (is_valid, reason)
        """
        # Check single stock limit
        position_pct = position_value / portfolio_value
        if position_pct > self.max_position_size:
            return False, f"Exceeds max position size ({self.max_position_size*100:.1f}%)"

        # Check sector limit
        if sector and sector_positions:
            current_sector_value = sector_positions.get(sector, 0.0)
            new_sector_value = current_sector_value + position_value
            sector_pct = new_sector_value / portfolio_value

            if sector_pct > self.max_sector_exposure:
                return False, f"Exceeds max sector exposure ({self.max_sector_exposure*100:.1f}%)"

        return True, "OK"

    def update_drawdown(self, current_portfolio_value: float):
        """
        Update drawdown tracking

        Args:
            current_portfolio_value: Current portfolio value
        """
        # Update peak
        if current_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_portfolio_value

        # Calculate drawdown
        if self.peak_portfolio_value > 0:
            self.current_drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
        else:
            self.current_drawdown = 0.0

        logger.debug(f"Current drawdown: {self.current_drawdown*100:.2f}%")

    def check_drawdown_limit(self) -> Tuple[bool, float]:
        """
        Check if current drawdown exceeds limit

        Returns:
            (is_within_limit, current_drawdown)
        """
        is_within_limit = self.current_drawdown <= self.max_drawdown_limit

        if not is_within_limit:
            logger.warning(f"Drawdown limit exceeded: {self.current_drawdown*100:.2f}% > {self.max_drawdown_limit*100:.1f}%")

        return is_within_limit, self.current_drawdown

    def calculate_risk_reward_ratio(
        self,
        entry_price: float,
        target_price: float,
        stop_loss_price: float
    ) -> float:
        """
        Calculate risk/reward ratio

        Args:
            entry_price: Entry price
            target_price: Target/take-profit price
            stop_loss_price: Stop loss price

        Returns:
            Risk/Reward ratio (higher is better)
        """
        potential_reward = abs(target_price - entry_price)
        potential_risk = abs(entry_price - stop_loss_price)

        if potential_risk == 0:
            return 0.0

        ratio = potential_reward / potential_risk

        logger.info(f"Risk/Reward ratio: {ratio:.2f}:1")
        return ratio

    def should_take_trade(
        self,
        risk_reward_ratio: float,
        min_ratio: float = 2.0,
        win_rate: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Determine if trade meets risk management criteria

        Args:
            risk_reward_ratio: Calculated R/R ratio
            min_ratio: Minimum acceptable R/R ratio (default: 2:1)
            win_rate: Historical win rate (optional)

        Returns:
            (should_take, reason)
        """
        if risk_reward_ratio < min_ratio:
            return False, f"R/R ratio {risk_reward_ratio:.2f} < minimum {min_ratio:.2f}"

        # If win rate is provided, check expected value
        if win_rate is not None:
            # Expected value = (Win Rate * Reward) - (Loss Rate * Risk)
            # For R/R ratio, assume risk = 1, reward = ratio
            expected_value = (win_rate * risk_reward_ratio) - ((1 - win_rate) * 1)

            if expected_value <= 0:
                return False, f"Negative expected value: {expected_value:.2f}"

            logger.info(f"Expected value: {expected_value:.2f}")

        return True, "Trade meets risk criteria"

    def calculate_portfolio_heat(
        self,
        open_positions: List[Dict],
        portfolio_value: float
    ) -> float:
        """
        Calculate total portfolio heat (risk across all positions)

        Args:
            open_positions: List of dicts with 'risk_amount' key
            portfolio_value: Total portfolio value

        Returns:
            Portfolio heat as percentage
        """
        total_risk = sum(pos.get('risk_amount', 0) for pos in open_positions)
        heat = (total_risk / portfolio_value) * 100 if portfolio_value > 0 else 0

        logger.info(f"Portfolio heat: {heat:.2f}%")
        return heat

    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary"""
        return {
            "max_position_size_pct": self.max_position_size * 100,
            "max_sector_exposure_pct": self.max_sector_exposure * 100,
            "max_drawdown_limit_pct": self.max_drawdown_limit * 100,
            "current_drawdown_pct": self.current_drawdown * 100,
            "kelly_fraction": self.kelly_fraction,
            "default_stop_loss_pct": self.default_stop_loss_pct * 100,
            "atr_multiplier": self.atr_multiplier,
            "peak_portfolio_value": self.peak_portfolio_value,
            "drawdown_limit_breached": self.current_drawdown > self.max_drawdown_limit
        }


# Singleton instance
_risk_manager: Optional[RiskManager] = None


def get_risk_manager() -> RiskManager:
    """Get or create risk manager singleton"""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager
