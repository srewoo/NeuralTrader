"""
Options Chain Data Fetcher

Fetches options chain data from NSE and calculates:
- Greeks for all strikes
- Max Pain
- Put-Call ratios
- IV surface
"""

import aiohttp
import asyncio
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
import json

from .models import (
    OptionContract, OptionChain, OptionTypeEnum,
    MaxPainResult, IVSurface, IVPoint
)
from .black_scholes import GreeksCalculator, OptionType

logger = logging.getLogger(__name__)


# NSE Option Chain URLs
NSE_OPTIONS_URL = "https://www.nseindia.com/api/option-chain-indices"
NSE_EQUITY_OPTIONS_URL = "https://www.nseindia.com/api/option-chain-equities"

# Common headers for NSE
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/option-chain",
}

# Index lot sizes
INDEX_LOT_SIZES = {
    "NIFTY": 50,
    "BANKNIFTY": 15,
    "FINNIFTY": 40,
    "MIDCPNIFTY": 75,
}


class OptionsChainFetcher:
    """
    Fetches and processes options chain data from NSE.

    Features:
    - Fetches live options chain
    - Calculates Greeks for all strikes
    - Computes Max Pain
    - Generates IV surface
    """

    def __init__(self, cache_ttl: int = 60):
        """
        Initialize fetcher.

        Args:
            cache_ttl: Cache time-to-live in seconds (default 60s)
        """
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self.greeks_calc = GreeksCalculator()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=NSE_HEADERS)
            # Warm up session with main page
            try:
                async with self._session.get("https://www.nseindia.com") as resp:
                    await resp.text()
            except Exception:
                pass
        return self._session

    async def close(self):
        """Close the session"""
        if self._session and not self._session.closed:
            await self._session.close()

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid"""
        if key not in self._cache:
            return False
        cached_time, _ = self._cache[key]
        return (datetime.now() - cached_time).seconds < self.cache_ttl

    async def fetch_nse_options_chain(
        self,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch raw options chain data from NSE.

        Args:
            symbol: Index or equity symbol

        Returns:
            Raw JSON response from NSE
        """
        cache_key = f"nse_chain_{symbol}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key][1]

        session = await self._get_session()

        # Determine URL based on symbol
        if symbol.upper() in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]:
            url = f"{NSE_OPTIONS_URL}?symbol={symbol.upper()}"
        else:
            url = f"{NSE_EQUITY_OPTIONS_URL}?symbol={symbol.upper()}"

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._cache[cache_key] = (datetime.now(), data)
                    return data
                else:
                    logger.warning(f"NSE returned status {resp.status}")
                    return None
        except Exception as e:
            logger.error(f"Failed to fetch NSE options chain: {e}")
            return None

    async def get_options_chain(
        self,
        symbol: str,
        expiry: Optional[date] = None,
        strikes_around_atm: int = 10,
        include_greeks: bool = True
    ) -> Optional[OptionChain]:
        """
        Get processed options chain with Greeks.

        Args:
            symbol: Underlying symbol
            expiry: Specific expiry (None = nearest)
            strikes_around_atm: Number of strikes above/below ATM
            include_greeks: Whether to calculate Greeks

        Returns:
            OptionChain model with all data
        """
        raw_data = await self.fetch_nse_options_chain(symbol)

        if not raw_data or "records" not in raw_data:
            logger.warning(f"No options data available for {symbol}")
            return None

        records = raw_data["records"]
        spot_price = records.get("underlyingValue", 0)
        expiry_dates = records.get("expiryDates", [])

        if not expiry_dates:
            return None

        # Select expiry
        if expiry:
            target_expiry = expiry.strftime("%d-%b-%Y")
        else:
            target_expiry = expiry_dates[0]  # Nearest expiry

        # Parse expiry date
        try:
            expiry_date = datetime.strptime(target_expiry, "%d-%b-%Y").date()
        except ValueError:
            expiry_date = date.today() + timedelta(days=7)

        # Calculate days to expiry
        days_to_expiry = max(0, (expiry_date - date.today()).days)

        # Get lot size
        lot_size = INDEX_LOT_SIZES.get(symbol.upper(), 50)

        # Find ATM strike
        all_strikes = set()
        for record in records.get("data", []):
            if record.get("expiryDate") == target_expiry:
                all_strikes.add(record.get("strikePrice"))

        all_strikes = sorted(all_strikes)
        if not all_strikes:
            return None

        # Find ATM
        atm_strike = min(all_strikes, key=lambda x: abs(x - spot_price))
        atm_idx = all_strikes.index(atm_strike)

        # Filter strikes around ATM
        start_idx = max(0, atm_idx - strikes_around_atm)
        end_idx = min(len(all_strikes), atm_idx + strikes_around_atm + 1)
        selected_strikes = set(all_strikes[start_idx:end_idx])

        # Process options data
        calls = []
        puts = []
        total_call_oi = 0
        total_put_oi = 0
        total_call_volume = 0
        total_put_volume = 0

        for record in records.get("data", []):
            if record.get("expiryDate") != target_expiry:
                continue

            strike = record.get("strikePrice")
            if strike not in selected_strikes:
                continue

            # Process Call
            ce_data = record.get("CE", {})
            if ce_data:
                call = self._create_option_contract(
                    ce_data, symbol, strike, expiry_date,
                    OptionTypeEnum.CALL, lot_size,
                    spot_price, days_to_expiry, include_greeks
                )
                calls.append(call)
                total_call_oi += ce_data.get("openInterest", 0)
                total_call_volume += ce_data.get("totalTradedVolume", 0)

            # Process Put
            pe_data = record.get("PE", {})
            if pe_data:
                put = self._create_option_contract(
                    pe_data, symbol, strike, expiry_date,
                    OptionTypeEnum.PUT, lot_size,
                    spot_price, days_to_expiry, include_greeks
                )
                puts.append(put)
                total_put_oi += pe_data.get("openInterest", 0)
                total_put_volume += pe_data.get("totalTradedVolume", 0)

        # Sort by strike
        calls.sort(key=lambda x: x.strike)
        puts.sort(key=lambda x: x.strike)

        # Calculate ratios
        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        pcr_volume = total_put_volume / total_call_volume if total_call_volume > 0 else 0

        # Calculate Max Pain
        max_pain = self._calculate_max_pain(
            calls, puts, list(selected_strikes), spot_price
        )

        return OptionChain(
            underlying=symbol.upper(),
            spot_price=spot_price,
            expiry=expiry_date,
            calls=calls,
            puts=puts,
            atm_strike=atm_strike,
            total_call_oi=total_call_oi,
            total_put_oi=total_put_oi,
            pcr_oi=round(pcr_oi, 2),
            pcr_volume=round(pcr_volume, 2),
            max_pain_strike=max_pain
        )

    def _create_option_contract(
        self,
        data: Dict,
        symbol: str,
        strike: float,
        expiry: date,
        option_type: OptionTypeEnum,
        lot_size: int,
        spot_price: float,
        days_to_expiry: int,
        include_greeks: bool
    ) -> OptionContract:
        """Create OptionContract from raw data"""
        ltp = data.get("lastPrice", 0)
        iv = data.get("impliedVolatility", 0)

        contract = OptionContract(
            symbol=data.get("identifier", f"{symbol}{expiry}{strike}{option_type.value}"),
            underlying=symbol.upper(),
            strike=strike,
            expiry=expiry,
            option_type=option_type,
            ltp=ltp,
            bid=data.get("bidprice"),
            ask=data.get("askPrice"),
            bid_qty=data.get("bidQty"),
            ask_qty=data.get("askQty"),
            volume=data.get("totalTradedVolume"),
            open_interest=data.get("openInterest"),
            oi_change=data.get("changeinOpenInterest"),
            iv=iv / 100 if iv > 0 else None,  # Convert from percentage
            lot_size=lot_size
        )

        # Calculate Greeks if requested
        if include_greeks and iv > 0:
            T = self.greeks_calc.days_to_years(days_to_expiry)
            bs_type = OptionType.CALL if option_type == OptionTypeEnum.CALL else OptionType.PUT

            greeks = self.greeks_calc.calculate_all_greeks(
                S=spot_price,
                K=strike,
                T=T,
                sigma=iv / 100,
                option_type=bs_type
            )

            contract.delta = round(greeks.delta, 4)
            contract.gamma = round(greeks.gamma, 6)
            contract.theta = round(greeks.theta, 4)
            contract.vega = round(greeks.vega, 4)
            contract.rho = round(greeks.rho, 4)
            contract.theoretical_price = round(greeks.option_price, 2)

        return contract

    def _calculate_max_pain(
        self,
        calls: List[OptionContract],
        puts: List[OptionContract],
        strikes: List[float],
        spot_price: float
    ) -> Optional[float]:
        """
        Calculate Max Pain strike.

        Max Pain is the strike where option writers have minimum payout.
        """
        if not calls or not puts:
            return None

        # Create OI lookup
        call_oi = {c.strike: c.open_interest or 0 for c in calls}
        put_oi = {p.strike: p.open_interest or 0 for p in puts}

        min_pain = float('inf')
        max_pain_strike = strikes[0]

        for test_strike in strikes:
            total_pain = 0

            # Call pain: OI * max(0, spot - strike)
            for strike, oi in call_oi.items():
                pain = oi * max(0, test_strike - strike)
                total_pain += pain

            # Put pain: OI * max(0, strike - spot)
            for strike, oi in put_oi.items():
                pain = oi * max(0, strike - test_strike)
                total_pain += pain

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_strike

        return max_pain_strike

    async def get_iv_surface(
        self,
        symbol: str,
        expiries: Optional[List[date]] = None
    ) -> Optional[IVSurface]:
        """
        Get IV surface for multiple expiries.

        Args:
            symbol: Underlying symbol
            expiries: List of expiry dates (None = all available)

        Returns:
            IVSurface with IV data points
        """
        raw_data = await self.fetch_nse_options_chain(symbol)

        if not raw_data or "records" not in raw_data:
            return None

        records = raw_data["records"]
        spot_price = records.get("underlyingValue", 0)
        available_expiries = records.get("expiryDates", [])

        points = []

        for expiry_str in available_expiries:
            try:
                expiry_date = datetime.strptime(expiry_str, "%d-%b-%Y").date()
            except ValueError:
                continue

            if expiries and expiry_date not in expiries:
                continue

            days_to_expiry = max(0, (expiry_date - date.today()).days)

            for record in records.get("data", []):
                if record.get("expiryDate") != expiry_str:
                    continue

                strike = record.get("strikePrice")

                # Call IV
                ce_data = record.get("CE", {})
                if ce_data and ce_data.get("impliedVolatility"):
                    points.append(IVPoint(
                        strike=strike,
                        expiry_days=days_to_expiry,
                        iv=ce_data["impliedVolatility"] / 100,
                        option_type=OptionTypeEnum.CALL
                    ))

                # Put IV
                pe_data = record.get("PE", {})
                if pe_data and pe_data.get("impliedVolatility"):
                    points.append(IVPoint(
                        strike=strike,
                        expiry_days=days_to_expiry,
                        iv=pe_data["impliedVolatility"] / 100,
                        option_type=OptionTypeEnum.PUT
                    ))

        # Calculate ATM IV (average of ATM call and put)
        atm_ivs = [
            p.iv for p in points
            if abs(p.strike - spot_price) / spot_price < 0.01
        ]
        atm_iv = sum(atm_ivs) / len(atm_ivs) if atm_ivs else None

        return IVSurface(
            underlying=symbol.upper(),
            spot_price=spot_price,
            points=points,
            atm_iv=round(atm_iv, 4) if atm_iv else None
        )


# Singleton instance
_fetcher_instance: Optional[OptionsChainFetcher] = None


def get_options_fetcher() -> OptionsChainFetcher:
    """Get singleton fetcher instance"""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = OptionsChainFetcher()
    return _fetcher_instance


async def fetch_options_chain(
    symbol: str,
    expiry: Optional[date] = None,
    include_greeks: bool = True
) -> Optional[OptionChain]:
    """
    Convenience function to fetch options chain.

    Args:
        symbol: Underlying symbol
        expiry: Specific expiry date
        include_greeks: Calculate Greeks

    Returns:
        OptionChain or None
    """
    fetcher = get_options_fetcher()
    return await fetcher.get_options_chain(
        symbol=symbol,
        expiry=expiry,
        include_greeks=include_greeks
    )
