"""
Bid-Ask Spread Modeling for Realistic Backtesting

Models spread as a function of:
- Base spread (market structure)
- Volume (liquidity impact)
- Volatility (risk premium)
- Price level (tick size effects)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum
import numpy as np


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class SpreadModelConfig:
    """Configuration for spread modeling"""
    # Base spread as percentage (0.05% = 5 basis points)
    base_spread_pct: float = 0.0005

    # Volume impact: spread widens for low-volume stocks
    # Formula: spread *= (1 + volume_factor / volume_ratio)
    # where volume_ratio = current_volume / avg_volume
    volume_factor: float = 0.02
    min_volume_ratio: float = 0.1  # Floor to prevent extreme spreads

    # Volatility impact: spread widens in volatile conditions
    # Formula: spread *= (1 + volatility_factor * normalized_volatility)
    volatility_factor: float = 0.5
    volatility_lookback: int = 20  # Days for volatility calculation

    # Price level adjustments (tick size effects for Indian markets)
    # NSE tick sizes: <Rs.250 = 0.05, >=Rs.250 = 0.10
    use_tick_size_adjustment: bool = True

    # Maximum spread cap (prevent unrealistic spreads)
    max_spread_pct: float = 0.02  # 2% maximum

    # Minimum spread floor
    min_spread_pct: float = 0.0001  # 1 basis point minimum


@dataclass
class SpreadModel:
    """
    Models bid-ask spread for realistic trade execution in backtests.

    Usage:
        model = SpreadModel()
        exec_price = model.get_execution_price(
            price=1500.0,
            side=OrderSide.BUY,
            volume=1000000,
            avg_volume=2000000,
            volatility=0.02
        )
    """
    config: SpreadModelConfig = field(default_factory=SpreadModelConfig)

    def calculate_spread(
        self,
        price: float,
        volume: Optional[float] = None,
        avg_volume: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate the bid-ask spread as a percentage.

        Args:
            price: Current stock price
            volume: Current day's volume (optional)
            avg_volume: Average daily volume (optional)
            volatility: Annualized volatility as decimal (optional)

        Returns:
            Spread as decimal (e.g., 0.001 = 0.1%)
        """
        spread = self.config.base_spread_pct

        # Volume adjustment
        if volume is not None and avg_volume is not None and avg_volume > 0:
            volume_ratio = max(volume / avg_volume, self.config.min_volume_ratio)
            # Low volume = wider spread
            if volume_ratio < 1.0:
                spread *= (1 + self.config.volume_factor / volume_ratio)

        # Volatility adjustment
        if volatility is not None and volatility > 0:
            # Normalize volatility (assuming 20% is "normal")
            normalized_vol = volatility / 0.20
            spread *= (1 + self.config.volatility_factor * (normalized_vol - 1))

        # Tick size adjustment for Indian markets
        if self.config.use_tick_size_adjustment and price > 0:
            tick_size = 0.10 if price >= 250 else 0.05
            # Spread should be at least 1 tick as percentage
            min_tick_spread = tick_size / price
            spread = max(spread, min_tick_spread)

        # Apply caps
        spread = max(spread, self.config.min_spread_pct)
        spread = min(spread, self.config.max_spread_pct)

        return spread

    def get_bid_ask(
        self,
        mid_price: float,
        volume: Optional[float] = None,
        avg_volume: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate bid and ask prices from mid price.

        Args:
            mid_price: Mid/last traded price
            volume: Current volume
            avg_volume: Average volume
            volatility: Current volatility

        Returns:
            Tuple of (bid_price, ask_price)
        """
        spread = self.calculate_spread(mid_price, volume, avg_volume, volatility)
        half_spread = spread / 2

        bid_price = mid_price * (1 - half_spread)
        ask_price = mid_price * (1 + half_spread)

        # Round to tick size
        tick_size = 0.10 if mid_price >= 250 else 0.05
        bid_price = round(bid_price / tick_size) * tick_size
        ask_price = round(ask_price / tick_size) * tick_size

        return bid_price, ask_price

    def get_execution_price(
        self,
        price: float,
        side: OrderSide,
        volume: Optional[float] = None,
        avg_volume: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> float:
        """
        Get the execution price for a trade, accounting for spread.

        BUY orders execute at ask (price + half_spread)
        SELL orders execute at bid (price - half_spread)

        Args:
            price: Last traded / mid price
            side: OrderSide.BUY or OrderSide.SELL
            volume: Current volume
            avg_volume: Average volume
            volatility: Current volatility

        Returns:
            Execution price after spread
        """
        bid, ask = self.get_bid_ask(price, volume, avg_volume, volatility)

        if side == OrderSide.BUY:
            return ask
        else:
            return bid

    def calculate_spread_cost(
        self,
        price: float,
        quantity: int,
        volume: Optional[float] = None,
        avg_volume: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate the total spread cost for a round-trip trade.

        Args:
            price: Stock price
            quantity: Number of shares
            volume: Current volume
            avg_volume: Average volume
            volatility: Volatility

        Returns:
            Total spread cost in currency (buy spread + sell spread)
        """
        spread = self.calculate_spread(price, volume, avg_volume, volatility)
        # Round-trip pays full spread
        return price * quantity * spread

    def estimate_market_impact(
        self,
        price: float,
        quantity: int,
        avg_daily_volume: float,
        participation_rate: float = 0.10
    ) -> float:
        """
        Estimate additional market impact for larger orders.

        Uses square-root impact model: impact = sigma * sqrt(Q / ADV)

        Args:
            price: Stock price
            quantity: Order quantity
            avg_daily_volume: Average daily volume
            participation_rate: Target participation rate (default 10%)

        Returns:
            Estimated market impact as percentage
        """
        if avg_daily_volume <= 0:
            return 0.0

        # Participation = order_size / ADV
        participation = quantity / avg_daily_volume

        # Square-root impact model
        # Typical coefficient for Indian markets: 0.1 to 0.3
        impact_coefficient = 0.15
        impact = impact_coefficient * np.sqrt(participation)

        # Cap at reasonable level
        return min(impact, 0.05)  # Max 5% impact


@dataclass
class RealisticExecutionModel:
    """
    Combines spread model with slippage and market impact for
    realistic order execution simulation.
    """
    spread_model: SpreadModel = field(default_factory=SpreadModel)

    # Additional slippage (order execution delay, etc.)
    base_slippage_pct: float = 0.0005  # 5 basis points

    def get_fill_price(
        self,
        price: float,
        side: OrderSide,
        quantity: int,
        volume: Optional[float] = None,
        avg_volume: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> Tuple[float, dict]:
        """
        Get realistic fill price including spread, slippage, and market impact.

        Returns:
            Tuple of (fill_price, cost_breakdown)
        """
        # Get spread-adjusted price
        exec_price = self.spread_model.get_execution_price(
            price, side, volume, avg_volume, volatility
        )

        # Calculate spread cost
        spread = self.spread_model.calculate_spread(price, volume, avg_volume, volatility)

        # Add slippage
        slippage = self.base_slippage_pct
        if side == OrderSide.BUY:
            exec_price *= (1 + slippage)
        else:
            exec_price *= (1 - slippage)

        # Market impact for larger orders
        market_impact = 0.0
        if avg_volume and avg_volume > 0:
            market_impact = self.spread_model.estimate_market_impact(
                price, quantity, avg_volume
            )
            if side == OrderSide.BUY:
                exec_price *= (1 + market_impact)
            else:
                exec_price *= (1 - market_impact)

        # Cost breakdown
        cost_breakdown = {
            "spread_pct": spread,
            "slippage_pct": slippage,
            "market_impact_pct": market_impact,
            "total_execution_cost_pct": spread/2 + slippage + market_impact,
            "price_before_costs": price,
            "fill_price": exec_price
        }

        return exec_price, cost_breakdown


# Preset configurations for different market conditions
SPREAD_PRESETS = {
    "liquid_large_cap": SpreadModelConfig(
        base_spread_pct=0.0003,  # 3 bps
        volume_factor=0.01,
        volatility_factor=0.3,
        max_spread_pct=0.005
    ),
    "mid_cap": SpreadModelConfig(
        base_spread_pct=0.0005,  # 5 bps
        volume_factor=0.02,
        volatility_factor=0.5,
        max_spread_pct=0.01
    ),
    "small_cap": SpreadModelConfig(
        base_spread_pct=0.001,  # 10 bps
        volume_factor=0.05,
        volatility_factor=0.8,
        max_spread_pct=0.02
    ),
    "penny_stock": SpreadModelConfig(
        base_spread_pct=0.005,  # 50 bps
        volume_factor=0.1,
        volatility_factor=1.0,
        max_spread_pct=0.05
    )
}


def get_spread_model_for_stock(
    market_cap: Optional[float] = None,
    avg_volume: Optional[float] = None
) -> SpreadModel:
    """
    Factory function to get appropriate spread model based on stock characteristics.

    Args:
        market_cap: Market cap in crores (for Indian stocks)
        avg_volume: Average daily volume in shares

    Returns:
        SpreadModel with appropriate configuration
    """
    # Determine stock category
    if market_cap is not None:
        if market_cap >= 50000:  # 50,000 Cr = Large cap
            config = SPREAD_PRESETS["liquid_large_cap"]
        elif market_cap >= 10000:  # 10,000 Cr = Mid cap
            config = SPREAD_PRESETS["mid_cap"]
        elif market_cap >= 1000:  # 1,000 Cr = Small cap
            config = SPREAD_PRESETS["small_cap"]
        else:
            config = SPREAD_PRESETS["penny_stock"]
    elif avg_volume is not None:
        if avg_volume >= 5000000:  # 50 lakh shares
            config = SPREAD_PRESETS["liquid_large_cap"]
        elif avg_volume >= 1000000:  # 10 lakh shares
            config = SPREAD_PRESETS["mid_cap"]
        elif avg_volume >= 100000:  # 1 lakh shares
            config = SPREAD_PRESETS["small_cap"]
        else:
            config = SPREAD_PRESETS["penny_stock"]
    else:
        # Default to mid-cap settings
        config = SPREAD_PRESETS["mid_cap"]

    return SpreadModel(config=config)
