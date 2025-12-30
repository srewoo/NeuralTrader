"""
Black-Scholes Options Pricing and Greeks Calculator

Implements:
- Option pricing (Call/Put)
- All Greeks: Delta, Gamma, Theta, Vega, Rho
- Implied Volatility calculation (Newton-Raphson)

Indian Market Considerations:
- Default risk-free rate: RBI repo rate (~6.5%)
- Dividend yield for index options
- NSE option lot sizes
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

try:
    from scipy.stats import norm
    from scipy.optimize import brentq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    norm = None
    brentq = None

logger = logging.getLogger(__name__)


class OptionType(Enum):
    """Option type"""
    CALL = "CE"
    PUT = "PE"


@dataclass
class GreeksResult:
    """Result of Greeks calculation"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    option_price: float


class GreeksCalculator:
    """
    Black-Scholes Greeks calculator for Indian market options.

    Usage:
        calc = GreeksCalculator()
        greeks = calc.calculate_all_greeks(
            S=2450,      # Spot price
            K=2500,      # Strike price
            T=0.0833,    # Time to expiry (in years, ~30 days)
            r=0.065,     # Risk-free rate (6.5%)
            sigma=0.20,  # Volatility (20%)
            option_type=OptionType.CALL
        )
    """

    # Default parameters for Indian market
    DEFAULT_RISK_FREE_RATE = 0.065  # RBI repo rate
    TRADING_DAYS_PER_YEAR = 252
    CALENDAR_DAYS_PER_YEAR = 365

    def __init__(self, risk_free_rate: Optional[float] = None):
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required. Run: pip install scipy")

        self.risk_free_rate = risk_free_rate or self.DEFAULT_RISK_FREE_RATE

    def _d1(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """Calculate d1 in Black-Scholes formula"""
        if T <= 0 or sigma <= 0:
            return 0.0
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    def _d2(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """Calculate d2 in Black-Scholes formula"""
        if T <= 0 or sigma <= 0:
            return 0.0
        return self._d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    def calculate_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType
    ) -> float:
        """
        Calculate option price using Black-Scholes model.

        Args:
            S: Current stock/index price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility (annualized)
            option_type: OptionType.CALL or OptionType.PUT

        Returns:
            Option price
        """
        if T <= 0:
            # At expiration, return intrinsic value
            if option_type == OptionType.CALL:
                return max(0, S - K)
            else:
                return max(0, K - S)

        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)

        if option_type == OptionType.CALL:
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return max(0, price)

    def calculate_delta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType
    ) -> float:
        """
        Calculate Delta - sensitivity to underlying price.

        Delta measures the rate of change of option price with respect
        to changes in the underlying asset's price.

        Call Delta: 0 to 1
        Put Delta: -1 to 0
        """
        if T <= 0:
            if option_type == OptionType.CALL:
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0

        d1 = self._d1(S, K, T, r, sigma)

        if option_type == OptionType.CALL:
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    def calculate_gamma(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate Gamma - rate of change of Delta.

        Gamma measures the rate of change in Delta with respect to
        changes in the underlying price. Same for calls and puts.
        """
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0

        d1 = self._d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    def calculate_theta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        per_day: bool = True
    ) -> float:
        """
        Calculate Theta - time decay.

        Theta measures the rate of decline in option value due to
        the passage of time.

        Args:
            per_day: If True, returns daily theta (default)
                     If False, returns annualized theta
        """
        if T <= 0:
            return 0.0

        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)

        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

        if option_type == OptionType.CALL:
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            theta = term1 + term2
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            theta = term1 + term2

        if per_day:
            return theta / self.CALENDAR_DAYS_PER_YEAR
        return theta

    def calculate_vega(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        per_pct: bool = True
    ) -> float:
        """
        Calculate Vega - sensitivity to volatility.

        Vega measures the rate of change of option price with respect
        to changes in implied volatility. Same for calls and puts.

        Args:
            per_pct: If True, returns vega per 1% change in volatility
                     If False, returns vega per unit change
        """
        if T <= 0:
            return 0.0

        d1 = self._d1(S, K, T, r, sigma)
        vega = S * norm.pdf(d1) * np.sqrt(T)

        if per_pct:
            return vega / 100  # Per 1% change
        return vega

    def calculate_rho(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        per_pct: bool = True
    ) -> float:
        """
        Calculate Rho - sensitivity to interest rate.

        Rho measures the rate of change of option price with respect
        to changes in the risk-free interest rate.

        Args:
            per_pct: If True, returns rho per 1% change in rate
        """
        if T <= 0:
            return 0.0

        d2 = self._d2(S, K, T, r, sigma)

        if option_type == OptionType.CALL:
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

        if per_pct:
            return rho / 100
        return rho

    def calculate_all_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: Optional[float] = None,
        sigma: float = 0.20,
        option_type: OptionType = OptionType.CALL
    ) -> GreeksResult:
        """
        Calculate all Greeks at once.

        Args:
            S: Current stock/index price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate (defaults to instance rate)
            sigma: Volatility (annualized)
            option_type: Call or Put

        Returns:
            GreeksResult with all Greeks and option price
        """
        r = r if r is not None else self.risk_free_rate

        return GreeksResult(
            delta=self.calculate_delta(S, K, T, r, sigma, option_type),
            gamma=self.calculate_gamma(S, K, T, r, sigma),
            theta=self.calculate_theta(S, K, T, r, sigma, option_type),
            vega=self.calculate_vega(S, K, T, r, sigma),
            rho=self.calculate_rho(S, K, T, r, sigma, option_type),
            option_price=self.calculate_price(S, K, T, r, sigma, option_type)
        )

    def calculate_implied_volatility(
        self,
        option_price: float,
        S: float,
        K: float,
        T: float,
        r: Optional[float] = None,
        option_type: OptionType = OptionType.CALL,
        precision: float = 1e-6,
        max_iterations: int = 100
    ) -> float:
        """
        Calculate Implied Volatility using Brent's method.

        Args:
            option_price: Market price of the option
            S: Current stock/index price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            option_type: Call or Put
            precision: Convergence precision
            max_iterations: Maximum iterations

        Returns:
            Implied volatility (annualized)
        """
        r = r if r is not None else self.risk_free_rate

        if T <= 0:
            return 0.0

        # Check intrinsic value
        if option_type == OptionType.CALL:
            intrinsic = max(0, S - K * np.exp(-r * T))
        else:
            intrinsic = max(0, K * np.exp(-r * T) - S)

        if option_price < intrinsic:
            logger.warning("Option price below intrinsic value")
            return 0.0

        def objective(sigma):
            return self.calculate_price(S, K, T, r, sigma, option_type) - option_price

        try:
            # Brent's method is more robust than Newton-Raphson
            iv = brentq(objective, 0.001, 5.0, xtol=precision, maxiter=max_iterations)
            return iv
        except ValueError:
            # Fall back to Newton-Raphson if Brent fails
            return self._newton_raphson_iv(
                option_price, S, K, T, r, option_type, precision, max_iterations
            )

    def _newton_raphson_iv(
        self,
        option_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: OptionType,
        precision: float,
        max_iterations: int
    ) -> float:
        """Newton-Raphson fallback for IV calculation"""
        sigma = 0.25  # Initial guess

        for _ in range(max_iterations):
            price = self.calculate_price(S, K, T, r, sigma, option_type)
            vega = self.calculate_vega(S, K, T, r, sigma, per_pct=False)

            if vega < 1e-10:
                break

            diff = option_price - price
            if abs(diff) < precision:
                return sigma

            sigma = sigma + diff / vega
            sigma = max(0.001, min(sigma, 5.0))  # Keep in bounds

        logger.warning(f"IV did not converge. Last estimate: {sigma:.4f}")
        return sigma

    def calculate_iv_surface(
        self,
        S: float,
        strikes: list,
        expiries: list,
        option_prices: Dict[Tuple[float, float], float],
        option_type: OptionType = OptionType.CALL
    ) -> Dict[Tuple[float, float], float]:
        """
        Calculate IV surface for multiple strikes and expiries.

        Args:
            S: Current price
            strikes: List of strike prices
            expiries: List of expiry times (in years)
            option_prices: Dict mapping (strike, expiry) to price
            option_type: Call or Put

        Returns:
            Dict mapping (strike, expiry) to implied volatility
        """
        iv_surface = {}

        for K in strikes:
            for T in expiries:
                key = (K, T)
                if key in option_prices:
                    try:
                        iv = self.calculate_implied_volatility(
                            option_prices[key], S, K, T,
                            option_type=option_type
                        )
                        iv_surface[key] = iv
                    except Exception as e:
                        logger.warning(f"IV calc failed for {key}: {e}")

        return iv_surface

    def days_to_years(self, days: int, trading_days: bool = False) -> float:
        """
        Convert days to years for time to expiry.

        Args:
            days: Number of days
            trading_days: If True, uses trading days (252)
                         If False, uses calendar days (365)
        """
        if trading_days:
            return days / self.TRADING_DAYS_PER_YEAR
        return days / self.CALENDAR_DAYS_PER_YEAR


def calculate_greeks_for_chain(
    spot_price: float,
    options_data: list,
    risk_free_rate: float = 0.065
) -> list:
    """
    Calculate Greeks for an entire options chain.

    Args:
        spot_price: Current spot price
        options_data: List of dicts with strike, expiry, option_type, iv
        risk_free_rate: Risk-free rate

    Returns:
        List with Greeks added to each option
    """
    calc = GreeksCalculator(risk_free_rate=risk_free_rate)
    results = []

    for option in options_data:
        strike = option.get('strike')
        expiry_days = option.get('expiry_days', 30)
        option_type = OptionType.CALL if option.get('type', 'CE') == 'CE' else OptionType.PUT
        iv = option.get('iv', 0.20)

        T = calc.days_to_years(expiry_days)

        greeks = calc.calculate_all_greeks(
            S=spot_price,
            K=strike,
            T=T,
            sigma=iv,
            option_type=option_type
        )

        result = {
            **option,
            'delta': round(greeks.delta, 4),
            'gamma': round(greeks.gamma, 6),
            'theta': round(greeks.theta, 4),
            'vega': round(greeks.vega, 4),
            'rho': round(greeks.rho, 4),
            'theoretical_price': round(greeks.option_price, 2)
        }
        results.append(result)

    return results
