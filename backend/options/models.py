"""
Pydantic Models for Options Data

Models for:
- Option contracts
- Options chain
- Greeks
- IV surface data
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import date, datetime
from enum import Enum


class OptionTypeEnum(str, Enum):
    """Option type enumeration"""
    CALL = "CE"
    PUT = "PE"


class Greeks(BaseModel):
    """Option Greeks values"""
    delta: float = Field(..., description="Delta - price sensitivity")
    gamma: float = Field(..., description="Gamma - delta sensitivity")
    theta: float = Field(..., description="Theta - time decay (per day)")
    vega: float = Field(..., description="Vega - volatility sensitivity (per 1%)")
    rho: float = Field(..., description="Rho - interest rate sensitivity (per 1%)")


class OptionContract(BaseModel):
    """Single option contract data"""
    symbol: str = Field(..., description="Option symbol (e.g., NIFTY2430224500CE)")
    underlying: str = Field(..., description="Underlying symbol (e.g., NIFTY)")
    strike: float = Field(..., description="Strike price")
    expiry: date = Field(..., description="Expiry date")
    option_type: OptionTypeEnum = Field(..., description="Call (CE) or Put (PE)")

    # Pricing
    ltp: Optional[float] = Field(None, description="Last traded price")
    bid: Optional[float] = Field(None, description="Best bid price")
    ask: Optional[float] = Field(None, description="Best ask price")
    bid_qty: Optional[int] = Field(None, description="Bid quantity")
    ask_qty: Optional[int] = Field(None, description="Ask quantity")

    # Volume & OI
    volume: Optional[int] = Field(None, description="Traded volume")
    open_interest: Optional[int] = Field(None, description="Open interest")
    oi_change: Optional[int] = Field(None, description="Change in OI")

    # Greeks (if calculated)
    iv: Optional[float] = Field(None, description="Implied volatility")
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None

    # Theoretical price
    theoretical_price: Optional[float] = Field(
        None, description="Black-Scholes theoretical price"
    )

    # Contract details
    lot_size: int = Field(default=50, description="Lot size for the contract")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "NIFTY24JAN24500CE",
                "underlying": "NIFTY",
                "strike": 24500,
                "expiry": "2024-01-25",
                "option_type": "CE",
                "ltp": 125.50,
                "bid": 125.00,
                "ask": 126.00,
                "volume": 150000,
                "open_interest": 5000000,
                "iv": 0.15,
                "delta": 0.45,
                "lot_size": 50
            }
        }


class OptionChain(BaseModel):
    """Complete options chain for a symbol"""
    underlying: str = Field(..., description="Underlying symbol")
    spot_price: float = Field(..., description="Current spot price")
    expiry: date = Field(..., description="Expiry date for this chain")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Chain data
    calls: List[OptionContract] = Field(default_factory=list)
    puts: List[OptionContract] = Field(default_factory=list)

    # Summary stats
    atm_strike: float = Field(..., description="At-the-money strike")
    total_call_oi: Optional[int] = Field(None, description="Total call OI")
    total_put_oi: Optional[int] = Field(None, description="Total put OI")
    pcr_oi: Optional[float] = Field(None, description="Put-Call ratio by OI")
    pcr_volume: Optional[float] = Field(None, description="Put-Call ratio by volume")

    # Max Pain
    max_pain_strike: Optional[float] = Field(
        None, description="Max pain strike price"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "underlying": "NIFTY",
                "spot_price": 24350.50,
                "expiry": "2024-01-25",
                "atm_strike": 24350,
                "total_call_oi": 15000000,
                "total_put_oi": 12000000,
                "pcr_oi": 0.80,
                "max_pain_strike": 24400
            }
        }


class IVPoint(BaseModel):
    """Single point on IV surface"""
    strike: float
    expiry_days: int
    iv: float
    option_type: OptionTypeEnum


class IVSurface(BaseModel):
    """Implied Volatility surface data"""
    underlying: str
    spot_price: float
    timestamp: datetime = Field(default_factory=datetime.now)
    points: List[IVPoint] = Field(default_factory=list)

    # Smile/Skew metrics
    atm_iv: Optional[float] = Field(None, description="ATM implied volatility")
    skew_25d: Optional[float] = Field(None, description="25-delta skew")
    term_structure: Optional[Dict[int, float]] = Field(
        None, description="IV by days to expiry"
    )


class GreeksRequest(BaseModel):
    """Request model for Greeks calculation"""
    spot_price: float = Field(..., gt=0, description="Current spot price")
    strike: float = Field(..., gt=0, description="Strike price")
    expiry_days: int = Field(..., ge=0, description="Days to expiry")
    volatility: float = Field(
        default=0.20, gt=0, le=5, description="Annualized volatility"
    )
    risk_free_rate: float = Field(
        default=0.065, ge=0, le=0.5, description="Risk-free rate"
    )
    option_type: OptionTypeEnum = Field(
        default=OptionTypeEnum.CALL, description="Call or Put"
    )


class GreeksResponse(BaseModel):
    """Response model for Greeks calculation"""
    spot_price: float
    strike: float
    expiry_days: int
    time_to_expiry_years: float
    option_type: OptionTypeEnum
    volatility: float
    risk_free_rate: float

    # Calculated values
    option_price: float = Field(..., description="Theoretical option price")
    greeks: Greeks

    # Moneyness
    moneyness: str = Field(..., description="ITM, ATM, or OTM")
    intrinsic_value: float
    time_value: float


class IVRequest(BaseModel):
    """Request model for implied volatility calculation"""
    option_price: float = Field(..., gt=0, description="Market price of option")
    spot_price: float = Field(..., gt=0, description="Current spot price")
    strike: float = Field(..., gt=0, description="Strike price")
    expiry_days: int = Field(..., ge=0, description="Days to expiry")
    risk_free_rate: float = Field(
        default=0.065, ge=0, le=0.5, description="Risk-free rate"
    )
    option_type: OptionTypeEnum = Field(
        default=OptionTypeEnum.CALL, description="Call or Put"
    )


class IVResponse(BaseModel):
    """Response model for implied volatility calculation"""
    option_price: float
    spot_price: float
    strike: float
    expiry_days: int
    option_type: OptionTypeEnum
    implied_volatility: float = Field(..., description="Calculated IV (annualized)")
    implied_volatility_pct: float = Field(..., description="IV as percentage")


class OptionsChainRequest(BaseModel):
    """Request model for fetching options chain"""
    symbol: str = Field(..., description="Underlying symbol (e.g., NIFTY, BANKNIFTY)")
    expiry: Optional[date] = Field(None, description="Specific expiry date")
    include_greeks: bool = Field(
        default=True, description="Calculate and include Greeks"
    )
    strikes_around_atm: Optional[int] = Field(
        default=10, ge=1, le=50,
        description="Number of strikes above/below ATM to include"
    )


class MaxPainResult(BaseModel):
    """Max Pain calculation result"""
    underlying: str
    expiry: date
    spot_price: float
    max_pain_strike: float
    max_pain_value: float = Field(..., description="Total pain value at max pain strike")
    strikes_analyzed: int
    call_oi_at_max_pain: int
    put_oi_at_max_pain: int
