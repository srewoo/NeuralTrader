"""
Options API Router

Provides endpoints for:
- Options chain data
- Greeks calculation
- Implied volatility calculation
- IV surface
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import date
import logging

from .black_scholes import GreeksCalculator, OptionType
from .options_chain import get_options_fetcher
from .models import (
    GreeksRequest, GreeksResponse, Greeks,
    IVRequest, IVResponse,
    OptionTypeEnum, OptionsChainRequest
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/options", tags=["Options"])


@router.get("/{symbol}/chain")
async def get_options_chain(
    symbol: str,
    expiry: Optional[date] = None,
    strikes_around_atm: int = Query(default=10, ge=1, le=50),
    include_greeks: bool = True
):
    """
    Get options chain for a symbol.

    Parameters:
    - symbol: Underlying symbol (e.g., NIFTY, BANKNIFTY, RELIANCE)
    - expiry: Specific expiry date (optional, defaults to nearest)
    - strikes_around_atm: Number of strikes above/below ATM
    - include_greeks: Calculate and include Greeks

    Returns complete options chain with calls, puts, OI, and Greeks.
    """
    try:
        fetcher = get_options_fetcher()
        chain = await fetcher.get_options_chain(
            symbol=symbol,
            expiry=expiry,
            strikes_around_atm=strikes_around_atm,
            include_greeks=include_greeks
        )

        if chain is None:
            raise HTTPException(
                status_code=404,
                detail=f"Options chain not found for {symbol}"
            )

        return chain.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get options chain: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/greeks")
async def calculate_greeks(request: GreeksRequest):
    """
    Calculate option Greeks using Black-Scholes model.

    Input:
    - spot_price: Current underlying price
    - strike: Strike price
    - expiry_days: Days to expiration
    - volatility: Annualized volatility (default 0.20)
    - risk_free_rate: Risk-free rate (default 0.065 for India)
    - option_type: CE (Call) or PE (Put)

    Returns all Greeks (Delta, Gamma, Theta, Vega, Rho) and theoretical price.
    """
    try:
        calc = GreeksCalculator(risk_free_rate=request.risk_free_rate)
        T = calc.days_to_years(request.expiry_days)

        bs_type = OptionType.CALL if request.option_type == OptionTypeEnum.CALL else OptionType.PUT

        greeks_result = calc.calculate_all_greeks(
            S=request.spot_price,
            K=request.strike,
            T=T,
            sigma=request.volatility,
            option_type=bs_type
        )

        # Calculate moneyness
        if request.option_type == OptionTypeEnum.CALL:
            intrinsic = max(0, request.spot_price - request.strike)
            if request.spot_price > request.strike * 1.02:
                moneyness = "ITM"
            elif request.spot_price < request.strike * 0.98:
                moneyness = "OTM"
            else:
                moneyness = "ATM"
        else:
            intrinsic = max(0, request.strike - request.spot_price)
            if request.spot_price < request.strike * 0.98:
                moneyness = "ITM"
            elif request.spot_price > request.strike * 1.02:
                moneyness = "OTM"
            else:
                moneyness = "ATM"

        time_value = greeks_result.option_price - intrinsic

        return GreeksResponse(
            spot_price=request.spot_price,
            strike=request.strike,
            expiry_days=request.expiry_days,
            time_to_expiry_years=round(T, 4),
            option_type=request.option_type,
            volatility=request.volatility,
            risk_free_rate=request.risk_free_rate,
            option_price=round(greeks_result.option_price, 2),
            greeks=Greeks(
                delta=round(greeks_result.delta, 4),
                gamma=round(greeks_result.gamma, 6),
                theta=round(greeks_result.theta, 4),
                vega=round(greeks_result.vega, 4),
                rho=round(greeks_result.rho, 4)
            ),
            moneyness=moneyness,
            intrinsic_value=round(intrinsic, 2),
            time_value=round(time_value, 2)
        )

    except Exception as e:
        logger.error(f"Failed to calculate Greeks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/iv")
async def calculate_implied_volatility(request: IVRequest):
    """
    Calculate implied volatility from option market price.

    Uses Brent's method (falling back to Newton-Raphson) to find
    the volatility that matches the market price.

    Input:
    - option_price: Market price of the option
    - spot_price: Current underlying price
    - strike: Strike price
    - expiry_days: Days to expiration
    - risk_free_rate: Risk-free rate (default 0.065)
    - option_type: CE (Call) or PE (Put)

    Returns implied volatility as decimal and percentage.
    """
    try:
        calc = GreeksCalculator(risk_free_rate=request.risk_free_rate)
        T = calc.days_to_years(request.expiry_days)

        bs_type = OptionType.CALL if request.option_type == OptionTypeEnum.CALL else OptionType.PUT

        iv = calc.calculate_implied_volatility(
            option_price=request.option_price,
            S=request.spot_price,
            K=request.strike,
            T=T,
            option_type=bs_type
        )

        return IVResponse(
            option_price=request.option_price,
            spot_price=request.spot_price,
            strike=request.strike,
            expiry_days=request.expiry_days,
            option_type=request.option_type,
            implied_volatility=round(iv, 4),
            implied_volatility_pct=round(iv * 100, 2)
        )

    except Exception as e:
        logger.error(f"Failed to calculate IV: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/iv-surface")
async def get_iv_surface(symbol: str):
    """
    Get implied volatility surface for a symbol.

    Returns IV data across multiple strikes and expiries,
    useful for volatility analysis and smile/skew visualization.
    """
    try:
        fetcher = get_options_fetcher()
        surface = await fetcher.get_iv_surface(symbol)

        if surface is None:
            raise HTTPException(
                status_code=404,
                detail=f"IV surface not found for {symbol}"
            )

        return surface.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get IV surface: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/max-pain")
async def get_max_pain(
    symbol: str,
    expiry: Optional[date] = None
):
    """
    Calculate Max Pain strike for an option series.

    Max Pain is the strike price where option writers have
    minimum payout (maximum pain to option buyers).

    Returns the max pain strike and related OI data.
    """
    try:
        fetcher = get_options_fetcher()
        chain = await fetcher.get_options_chain(
            symbol=symbol,
            expiry=expiry,
            strikes_around_atm=30,  # Get more strikes for accurate max pain
            include_greeks=False
        )

        if chain is None:
            raise HTTPException(
                status_code=404,
                detail=f"Options data not found for {symbol}"
            )

        return {
            "symbol": symbol,
            "expiry": chain.expiry.isoformat(),
            "spot_price": chain.spot_price,
            "max_pain_strike": chain.max_pain_strike,
            "atm_strike": chain.atm_strike,
            "pcr_oi": chain.pcr_oi,
            "pcr_volume": chain.pcr_volume,
            "total_call_oi": chain.total_call_oi,
            "total_put_oi": chain.total_put_oi
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get max pain: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/pcr")
async def get_put_call_ratio(
    symbol: str,
    expiry: Optional[date] = None
):
    """
    Get Put-Call Ratio for a symbol.

    Returns both OI-based and volume-based PCR, along with
    interpretation for sentiment analysis.
    """
    try:
        fetcher = get_options_fetcher()
        chain = await fetcher.get_options_chain(
            symbol=symbol,
            expiry=expiry,
            strikes_around_atm=30,
            include_greeks=False
        )

        if chain is None:
            raise HTTPException(
                status_code=404,
                detail=f"Options data not found for {symbol}"
            )

        # Interpret PCR
        pcr = chain.pcr_oi or 0
        if pcr > 1.2:
            sentiment = "Bearish"
            interpretation = "High put buying indicates bearish sentiment"
        elif pcr < 0.8:
            sentiment = "Bullish"
            interpretation = "High call buying indicates bullish sentiment"
        else:
            sentiment = "Neutral"
            interpretation = "Balanced put-call activity"

        return {
            "symbol": symbol,
            "expiry": chain.expiry.isoformat(),
            "spot_price": chain.spot_price,
            "pcr_oi": chain.pcr_oi,
            "pcr_volume": chain.pcr_volume,
            "total_call_oi": chain.total_call_oi,
            "total_put_oi": chain.total_put_oi,
            "sentiment": sentiment,
            "interpretation": interpretation
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get PCR: {e}")
        raise HTTPException(status_code=500, detail=str(e))
