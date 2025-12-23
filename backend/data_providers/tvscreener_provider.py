"""
TradingView Screener Data Provider
FREE real-time Indian stock data via tvscreener library
No API key required!
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import logging
import asyncio
from data_providers.base_provider import BaseDataProvider, StockData

try:
    from tvscreener import StockScreener, StockField, Market, Filter
    from tvscreener import STOCK_PRICE_FIELDS, STOCK_VALUATION_FIELDS, STOCK_PROFITABILITY_FIELDS
    TV_SCREENER_AVAILABLE = True
except ImportError:
    TV_SCREENER_AVAILABLE = False
    logging.warning("tvscreener library not installed. Install with: pip install tvscreener")

logger = logging.getLogger(__name__)


class TVScreenerProvider(BaseDataProvider):
    """
    TradingView Screener Provider

    Advantages:
    - ✅ FREE (no API key required)
    - ✅ Real-time prices for NSE/BSE stocks
    - ✅ Fundamental data (P/E, ROE, Debt/Equity, etc.)
    - ✅ Built-in screener queries
    - ✅ No rate limits
    - ✅ High data quality from TradingView

    Usage:
        provider = TVScreenerProvider()
        data = await provider.get_quote("RELIANCE")
    """

    def __init__(self):
        """Initialize TV Screener provider (no API key needed!)"""
        super().__init__(name="TVScreener", api_key="NOT_REQUIRED")

    def _check_availability(self) -> bool:
        """Check if TVScreener is available"""
        return TV_SCREENER_AVAILABLE

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for TradingView format"""
        symbol = symbol.upper().replace('.NS', '').replace('.BO', '')
        return symbol

    async def get_quote(self, symbol: str) -> Optional[StockData]:
        """Get current quote from TradingView Screener"""
        if not self.is_available:
            return None

        try:
            symbol = self._normalize_symbol(symbol)

            # Query TradingView for stock data
            screener = StockScreener()
            screener.set_markets(Market.INDIA)

            # Select fields we need using predefined groups
            fields = (
                STOCK_PRICE_FIELDS[:8] +
                [STOCK_VALUATION_FIELDS[2]] +  # Include P/E
                [
                    StockField.DEBT_TO_EQUITY_FQ,
                    StockField.DIVIDENDS_YIELD_CURRENT,
                ]
            )
            screener.select(*fields)

            # Set range to get results
            screener.set_range(0, 50)  # Get first 50 to find our stock

            # Get data
            result = screener.get()

            if result.empty:
                logger.warning(f"No data found for {symbol}")
                return None

            # Find the stock (prefer NSE over BSE)
            stock_row = None
            for _, row in result.iterrows():
                row_symbol = row.get('Name', '')
                if row_symbol.upper() == symbol:
                    # Prefer NSE listing
                    if 'NSE:' in row.get('Symbol', ''):
                        stock_row = row
                        break
                    elif stock_row is None:  # Take BSE if no NSE found yet
                        stock_row = row

            if stock_row is None:
                logger.warning(f"No data found for {symbol}")
                return None

            # Extract data
            current_price = float(stock_row.get('Price', 0))
            change_percent = float(stock_row.get('Change %', 0))
            # Calculate absolute change from percentage
            change = current_price * (change_percent / 100)
            previous_close = current_price - change
            volume = int(stock_row.get('Volume', 0)) if stock_row.get('Volume') else 0
            market_cap = float(stock_row.get('Market Capitalization', 0))
            pe_ratio = float(stock_row.get('Price to Earnings Ratio (TTM)', 0)) if stock_row.get('Price to Earnings Ratio (TTM)') else None

            logger.info(f"TVScreener: {symbol} @ ₹{current_price} ({change_percent:+.2f}%)")

            return StockData(
                symbol=symbol,
                name=stock_row.get('Description', symbol),
                current_price=current_price,
                previous_close=previous_close,
                volume=volume,
                market_cap=market_cap,
                pe_ratio=pe_ratio,
                week_52_high=current_price,  # Approximation
                week_52_low=current_price,    # Approximation
                provider="TVScreener"
            )

        except Exception as e:
            logger.error(f"Error fetching {symbol} from TVScreener: {e}", exc_info=True)
            return None

    async def get_historical_data(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Get historical data (not supported by TVScreener)"""
        logger.warning("TVScreener has limited historical data. Using yfinance fallback.")
        return None

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limit information"""
        return {
            "provider": "TVScreener",
            "calls_per_minute": "unlimited",
            "calls_per_day": "unlimited",
            "cost": "FREE",
            "notes": "No rate limits - free service!"
        }

    def screen_stocks(
        self,
        pe_max: Optional[float] = None,
        roe_min: Optional[float] = None,
        debt_to_equity_max: Optional[float] = None,
        market_cap_min: Optional[float] = None,
        dividend_yield_min: Optional[float] = None,
        profit_margin_min: Optional[float] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Screen stocks based on fundamental criteria"""
        if not self.is_available:
            return []

        try:
            screener = StockScreener()
            screener.set_markets(Market.INDIA)

            # Use predefined field groups plus specific fields we need
            fields = (
                STOCK_PRICE_FIELDS[:3] +
                STOCK_VALUATION_FIELDS[:5] +
                STOCK_PROFITABILITY_FIELDS[:6] +
                [
                    StockField.DEBT_TO_EQUITY_FQ,
                    StockField.DIVIDENDS_YIELD_CURRENT,
                ]
            )
            screener.select(*fields)

            # Note: Filters are complex with tvscreener API, so we'll filter in Python instead
            # Set range to get more results for filtering
            screener.set_range(0, min(limit * 5, 200))  # Get more results to filter

            # Execute query
            result = screener.get()

            if result.empty:
                logger.info("No stocks match screening criteria")
                return []

            # Filter results in Python
            filtered_stocks = []
            for _, row in result.iterrows():
                # Extract values with proper column names from DataFrame
                pe = row.get('Price to Earnings Ratio (TTM)')
                roe = row.get('Return on Equity (TTM)')
                de = row.get('Debt to Equity FQ')
                market_cap = row.get('Market Capitalization', 0)
                div_yield = row.get('Dividend Yield % (Current)')
                profit_margin = row.get('Net Margin (TTM)')

                # Apply filters
                if pe_max is not None and (pe is None or pe > pe_max):
                    continue
                if roe_min is not None and (roe is None or roe < roe_min):
                    continue
                if debt_to_equity_max is not None and de is not None and de > debt_to_equity_max:
                    continue
                if market_cap_min is not None and market_cap < (market_cap_min * 10000000):
                    continue
                if dividend_yield_min is not None and div_yield is not None and div_yield < dividend_yield_min:
                    continue
                if profit_margin_min is not None and (profit_margin is None or profit_margin < profit_margin_min):
                    continue

                filtered_stocks.append({
                    "symbol": row.get('Name', ''),
                    "name": row.get('Description', ''),
                    "price": float(row.get('Price', 0)),
                    "market_cap": float(market_cap),
                    "pe_ratio": float(pe) if pe is not None else None,
                    "roe": float(roe) if roe is not None else None,
                    "debt_to_equity": float(de) if de is not None else None,
                    "dividend_yield": float(div_yield) if div_yield is not None else None,
                    "profit_margin": float(profit_margin) if profit_margin is not None else None,
                })

                if len(filtered_stocks) >= limit:
                    break

            logger.info(f"Screener found {len(filtered_stocks)} stocks matching criteria")
            return filtered_stocks

        except Exception as e:
            logger.error(f"Error screening stocks: {e}", exc_info=True)
            return []


def get_tvscreener_provider() -> TVScreenerProvider:
    """Get TVScreener provider instance"""
    return TVScreenerProvider()


# ============== Dynamic Stock List Functions ==============

# Cache for stock list
_stock_cache = {
    "stocks": [],
    "last_updated": None,
    "cache_duration": timedelta(hours=6)  # Refresh every 6 hours
}


def get_all_indian_stocks(
    min_market_cap: float = 100,  # Minimum market cap in crores
    max_stocks: int = 2000,
    force_refresh: bool = False
) -> List[Dict[str, Any]]:
    """
    Fetch all Indian stocks from NSE/BSE using TradingView Screener.

    Args:
        min_market_cap: Minimum market cap in crores (default 100 Cr)
        max_stocks: Maximum number of stocks to fetch
        force_refresh: Force refresh cache

    Returns:
        List of stock dictionaries with symbol, name, sector, market_cap
    """
    global _stock_cache

    # Check cache
    if not force_refresh and _stock_cache["stocks"]:
        if _stock_cache["last_updated"]:
            cache_age = datetime.now() - _stock_cache["last_updated"]
            if cache_age < _stock_cache["cache_duration"]:
                logger.info(f"Using cached stock list ({len(_stock_cache['stocks'])} stocks)")
                return _stock_cache["stocks"]

    if not TV_SCREENER_AVAILABLE:
        logger.error("tvscreener not available - cannot fetch dynamic stock list")
        return []

    try:
        logger.info(f"Fetching all Indian stocks from TradingView (min market cap: {min_market_cap} Cr)...")

        screener = StockScreener()
        screener.set_markets(Market.INDIA)

        # Select essential fields
        fields = [
            StockField.NAME,
            StockField.DESCRIPTION,
            StockField.PRICE,
            StockField.CHANGE,
            StockField.CHANGE_ABS,
            StockField.VOLUME,
            StockField.MARKET_CAPITALIZATION,
            StockField.SECTOR,
            StockField.INDUSTRY,
        ]

        # Add valuation fields if available
        try:
            fields.extend([
                StockField.PRICE_TO_EARNINGS_TTM,
                StockField.HIGH_52_WEEK,
                StockField.LOW_52_WEEK,
            ])
        except AttributeError:
            pass

        screener.select(*fields)

        # Fetch in batches
        all_stocks = []
        batch_size = 500
        offset = 0
        seen_symbols = set()

        while len(all_stocks) < max_stocks:
            screener.set_range(offset, offset + batch_size)
            result = screener.get()

            if result.empty:
                break

            for _, row in result.iterrows():
                symbol = row.get('Name', '')
                if not symbol or symbol in seen_symbols:
                    continue

                # Get market cap (in crores for filtering)
                market_cap = row.get('Market Capitalization', 0)
                market_cap_cr = market_cap / 10000000 if market_cap else 0

                # Filter by minimum market cap
                if market_cap_cr < min_market_cap:
                    continue

                seen_symbols.add(symbol)

                stock_data = {
                    "symbol": symbol,
                    "name": row.get('Description', symbol),
                    "sector": row.get('Sector', 'N/A'),
                    "industry": row.get('Industry', 'N/A'),
                    "market_cap": market_cap,
                    "market_cap_cr": round(market_cap_cr, 2),
                    "price": float(row.get('Price', 0)) if row.get('Price') else None,
                    "change_percent": float(row.get('Change %', 0)) if row.get('Change %') else 0,
                }

                # Add optional fields
                try:
                    stock_data["pe_ratio"] = float(row.get('Price to Earnings Ratio (TTM)', 0)) if row.get('Price to Earnings Ratio (TTM)') else None
                    stock_data["week_52_high"] = float(row.get('52 Week High', 0)) if row.get('52 Week High') else None
                    stock_data["week_52_low"] = float(row.get('52 Week Low', 0)) if row.get('52 Week Low') else None
                except:
                    pass

                all_stocks.append(stock_data)

                if len(all_stocks) >= max_stocks:
                    break

            offset += batch_size

            # Safety check - don't loop forever
            if offset > 5000:
                break

        # Sort by market cap (largest first)
        all_stocks.sort(key=lambda x: x.get('market_cap', 0) or 0, reverse=True)

        # Update cache
        _stock_cache["stocks"] = all_stocks
        _stock_cache["last_updated"] = datetime.now()

        logger.info(f"Successfully fetched {len(all_stocks)} Indian stocks from TradingView")
        return all_stocks

    except Exception as e:
        logger.error(f"Error fetching Indian stocks: {e}", exc_info=True)
        # Return cached data if available
        if _stock_cache["stocks"]:
            logger.info("Returning cached stock list due to error")
            return _stock_cache["stocks"]
        return []


async def get_all_indian_stocks_async(
    min_market_cap: float = 100,
    max_stocks: int = 2000,
    force_refresh: bool = False
) -> List[Dict[str, Any]]:
    """Async wrapper for get_all_indian_stocks"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: get_all_indian_stocks(min_market_cap, max_stocks, force_refresh)
    )


def search_indian_stocks(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Search Indian stocks by symbol or name.

    Args:
        query: Search query (symbol or name)
        limit: Maximum results to return

    Returns:
        List of matching stocks
    """
    # Get cached stocks first
    stocks = get_all_indian_stocks()

    if not stocks:
        logger.warning("No stocks in cache for search")
        return []

    query_upper = query.upper().strip()

    # Search by symbol and name
    results = []
    exact_matches = []
    prefix_matches = []
    contains_matches = []

    for stock in stocks:
        symbol = stock.get('symbol', '').upper()
        name = stock.get('name', '').upper()

        # Exact symbol match (highest priority)
        if symbol == query_upper:
            exact_matches.append(stock)
        # Symbol starts with query
        elif symbol.startswith(query_upper):
            prefix_matches.append(stock)
        # Symbol or name contains query
        elif query_upper in symbol or query_upper in name:
            contains_matches.append(stock)

    # Combine results with priority
    results = exact_matches + prefix_matches + contains_matches

    return results[:limit]


def get_stocks_by_sector(sector: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Get stocks filtered by sector"""
    stocks = get_all_indian_stocks()

    sector_lower = sector.lower()
    filtered = [
        s for s in stocks
        if sector_lower in s.get('sector', '').lower() or
           sector_lower in s.get('industry', '').lower()
    ]

    return filtered[:limit]


def get_top_stocks_by_market_cap(limit: int = 200) -> List[Dict[str, Any]]:
    """Get top stocks sorted by market cap"""
    stocks = get_all_indian_stocks()
    # Already sorted by market cap in get_all_indian_stocks
    return stocks[:limit]


def clear_stock_cache():
    """Clear the stock cache to force refresh on next call"""
    global _stock_cache
    _stock_cache["stocks"] = []
    _stock_cache["last_updated"] = None
    logger.info("Stock cache cleared")
