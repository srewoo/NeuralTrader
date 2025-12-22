"""
TradingView Screener Data Provider
FREE real-time Indian stock data via tvscreener library
No API key required!
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd
import logging
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
