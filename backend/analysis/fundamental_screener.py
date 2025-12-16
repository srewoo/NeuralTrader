"""
Fundamental Screener
Screen stocks based on fundamental metrics using Yahoo Finance data.
Supports PE, PB, debt/equity, ROE, dividend yield, market cap, and more.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ScreenerCriteria(Enum):
    """Available screening criteria"""
    PE_RATIO = "pe_ratio"
    PB_RATIO = "pb_ratio"
    DEBT_TO_EQUITY = "debt_to_equity"
    ROE = "roe"
    ROA = "roa"
    DIVIDEND_YIELD = "dividend_yield"
    MARKET_CAP = "market_cap"
    REVENUE_GROWTH = "revenue_growth"
    PROFIT_MARGIN = "profit_margin"
    CURRENT_RATIO = "current_ratio"
    QUICK_RATIO = "quick_ratio"
    EPS_GROWTH = "eps_growth"
    PRICE_TO_SALES = "price_to_sales"
    PRICE_TO_BOOK = "price_to_book"
    BETA = "beta"


@dataclass
class ScreenerFilter:
    """A single filter criterion"""
    metric: str
    operator: str  # "gt", "lt", "eq", "gte", "lte", "between"
    value: float
    value2: Optional[float] = None  # For "between" operator


class FundamentalScreener:
    """
    Screen stocks based on fundamental metrics.
    Uses Yahoo Finance for fundamental data.
    """

    # Mapping of our metric names to Yahoo Finance field names
    METRIC_MAPPING = {
        "pe_ratio": "trailingPE",
        "forward_pe": "forwardPE",
        "pb_ratio": "priceToBook",
        "debt_to_equity": "debtToEquity",
        "roe": "returnOnEquity",
        "roa": "returnOnAssets",
        "dividend_yield": "dividendYield",
        "market_cap": "marketCap",
        "revenue_growth": "revenueGrowth",
        "profit_margin": "profitMargins",
        "operating_margin": "operatingMargins",
        "current_ratio": "currentRatio",
        "quick_ratio": "quickRatio",
        "eps": "trailingEps",
        "forward_eps": "forwardEps",
        "price_to_sales": "priceToSalesTrailing12Months",
        "enterprise_value": "enterpriseValue",
        "ev_to_ebitda": "enterpriseToEbitda",
        "ev_to_revenue": "enterpriseToRevenue",
        "beta": "beta",
        "52w_high": "fiftyTwoWeekHigh",
        "52w_low": "fiftyTwoWeekLow",
        "50d_avg": "fiftyDayAverage",
        "200d_avg": "twoHundredDayAverage",
        "avg_volume": "averageVolume",
        "shares_outstanding": "sharesOutstanding",
        "float_shares": "floatShares",
        "held_by_institutions": "heldPercentInstitutions",
        "held_by_insiders": "heldPercentInsiders",
        "book_value": "bookValue",
        "earnings_growth": "earningsGrowth",
        "revenue_per_share": "revenuePerShare",
        "gross_margins": "grossMargins",
        "ebitda_margins": "ebitdaMargins",
        "free_cashflow": "freeCashflow",
        "operating_cashflow": "operatingCashflow",
        "total_debt": "totalDebt",
        "total_cash": "totalCash",
    }

    # Sector benchmarks for Indian markets (approximate)
    SECTOR_BENCHMARKS = {
        "Technology": {"pe_ratio": 25, "roe": 0.20, "debt_to_equity": 0.3},
        "Financial Services": {"pe_ratio": 15, "roe": 0.15, "debt_to_equity": 5.0},
        "Healthcare": {"pe_ratio": 30, "roe": 0.18, "debt_to_equity": 0.4},
        "Consumer Cyclical": {"pe_ratio": 35, "roe": 0.15, "debt_to_equity": 0.5},
        "Consumer Defensive": {"pe_ratio": 40, "roe": 0.25, "debt_to_equity": 0.3},
        "Energy": {"pe_ratio": 10, "roe": 0.12, "debt_to_equity": 0.8},
        "Industrials": {"pe_ratio": 20, "roe": 0.14, "debt_to_equity": 0.6},
        "Basic Materials": {"pe_ratio": 12, "roe": 0.12, "debt_to_equity": 0.5},
        "Communication Services": {"pe_ratio": 18, "roe": 0.10, "debt_to_equity": 1.0},
        "Utilities": {"pe_ratio": 12, "roe": 0.10, "debt_to_equity": 1.5},
        "Real Estate": {"pe_ratio": 20, "roe": 0.08, "debt_to_equity": 1.0},
    }

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=10)

    def _get_indian_symbol(self, symbol: str) -> str:
        """Add .NS suffix for NSE stocks"""
        symbol = symbol.upper().strip()
        if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
            return f"{symbol}.NS"
        return symbol

    def _fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Fetch fundamental data for a single stock"""
        try:
            ticker_symbol = self._get_indian_symbol(symbol)
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info

            if not info or info.get('regularMarketPrice') is None:
                # Try BSE
                ticker_symbol = symbol.upper().replace('.NS', '') + '.BO'
                ticker = yf.Ticker(ticker_symbol)
                info = ticker.info

            if not info:
                return {"symbol": symbol, "error": "No data available"}

            # Extract all fundamental metrics
            fundamentals = {
                "symbol": symbol.upper().replace('.NS', '').replace('.BO', ''),
                "name": info.get('longName', info.get('shortName', symbol)),
                "sector": info.get('sector', 'N/A'),
                "industry": info.get('industry', 'N/A'),
                "current_price": info.get('regularMarketPrice', info.get('currentPrice')),
            }

            # Add all mapped metrics
            for our_key, yf_key in self.METRIC_MAPPING.items():
                value = info.get(yf_key)
                if value is not None:
                    # Convert percentages
                    if our_key in ['dividend_yield', 'roe', 'roa', 'profit_margin',
                                   'operating_margin', 'revenue_growth', 'earnings_growth',
                                   'gross_margins', 'ebitda_margins', 'held_by_institutions',
                                   'held_by_insiders']:
                        if isinstance(value, (int, float)) and value < 1:
                            value = value * 100  # Convert to percentage
                    fundamentals[our_key] = round(value, 4) if isinstance(value, float) else value

            # Calculate additional metrics
            if fundamentals.get('market_cap') and fundamentals.get('market_cap') > 0:
                fundamentals['market_cap_cr'] = round(fundamentals['market_cap'] / 10000000, 2)

            # Valuation score (simple)
            fundamentals['valuation_score'] = self._calculate_valuation_score(fundamentals)

            # Quality score
            fundamentals['quality_score'] = self._calculate_quality_score(fundamentals)

            return fundamentals

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}

    def _calculate_valuation_score(self, data: Dict) -> Optional[float]:
        """Calculate a simple valuation score (0-100, lower PE/PB = better)"""
        try:
            score = 50  # Start neutral

            pe = data.get('pe_ratio')
            pb = data.get('pb_ratio')
            div_yield = data.get('dividend_yield', 0)

            if pe:
                if pe < 10:
                    score += 20
                elif pe < 15:
                    score += 15
                elif pe < 20:
                    score += 10
                elif pe < 25:
                    score += 5
                elif pe > 50:
                    score -= 15
                elif pe > 35:
                    score -= 10

            if pb:
                if pb < 1:
                    score += 15
                elif pb < 2:
                    score += 10
                elif pb < 3:
                    score += 5
                elif pb > 5:
                    score -= 10

            if div_yield:
                if div_yield > 4:
                    score += 10
                elif div_yield > 2:
                    score += 5

            return max(0, min(100, score))
        except:
            return None

    def _calculate_quality_score(self, data: Dict) -> Optional[float]:
        """Calculate quality score based on profitability and financial health"""
        try:
            score = 50

            roe = data.get('roe')
            roa = data.get('roa')
            debt_to_equity = data.get('debt_to_equity')
            current_ratio = data.get('current_ratio')
            profit_margin = data.get('profit_margin')

            if roe:
                if roe > 25:
                    score += 20
                elif roe > 20:
                    score += 15
                elif roe > 15:
                    score += 10
                elif roe > 10:
                    score += 5
                elif roe < 5:
                    score -= 10

            if roa:
                if roa > 15:
                    score += 10
                elif roa > 10:
                    score += 5

            if debt_to_equity is not None:
                if debt_to_equity < 0.3:
                    score += 15
                elif debt_to_equity < 0.5:
                    score += 10
                elif debt_to_equity < 1:
                    score += 5
                elif debt_to_equity > 2:
                    score -= 15
                elif debt_to_equity > 1.5:
                    score -= 10

            if current_ratio:
                if current_ratio > 2:
                    score += 10
                elif current_ratio > 1.5:
                    score += 5
                elif current_ratio < 1:
                    score -= 15

            if profit_margin:
                if profit_margin > 20:
                    score += 10
                elif profit_margin > 15:
                    score += 5

            return max(0, min(100, score))
        except:
            return None

    def _apply_filter(self, value: Any, filter_def: ScreenerFilter) -> bool:
        """Apply a single filter to a value"""
        if value is None:
            return False

        try:
            value = float(value)
            op = filter_def.operator
            target = float(filter_def.value)

            if op == "gt":
                return value > target
            elif op == "gte":
                return value >= target
            elif op == "lt":
                return value < target
            elif op == "lte":
                return value <= target
            elif op == "eq":
                return abs(value - target) < 0.001
            elif op == "between" and filter_def.value2 is not None:
                return target <= value <= float(filter_def.value2)
            else:
                return False
        except:
            return False

    async def screen_stocks(
        self,
        symbols: List[str],
        filters: List[Dict[str, Any]],
        sort_by: Optional[str] = None,
        sort_order: str = "desc",
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Screen stocks based on fundamental criteria.

        Args:
            symbols: List of stock symbols to screen
            filters: List of filter dictionaries with keys: metric, operator, value, value2
            sort_by: Metric to sort by
            sort_order: "asc" or "desc"
            limit: Maximum results to return

        Returns:
            Dictionary with screened results and metadata
        """
        try:
            # Fetch fundamentals for all symbols concurrently
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(self.executor, self._fetch_fundamentals, symbol)
                for symbol in symbols
            ]
            all_data = await asyncio.gather(*tasks)

            # Filter out errors
            valid_data = [d for d in all_data if 'error' not in d]

            # Convert filters to ScreenerFilter objects
            parsed_filters = []
            for f in filters:
                parsed_filters.append(ScreenerFilter(
                    metric=f.get('metric'),
                    operator=f.get('operator', 'gte'),
                    value=f.get('value', 0),
                    value2=f.get('value2')
                ))

            # Apply filters
            filtered = []
            for data in valid_data:
                passes_all = True
                for f in parsed_filters:
                    value = data.get(f.metric)
                    if not self._apply_filter(value, f):
                        passes_all = False
                        break
                if passes_all:
                    filtered.append(data)

            # Sort results
            if sort_by and filtered:
                reverse = sort_order == "desc"
                filtered.sort(
                    key=lambda x: x.get(sort_by, 0) or 0,
                    reverse=reverse
                )

            # Limit results
            filtered = filtered[:limit]

            return {
                "total_screened": len(symbols),
                "passed_filters": len(filtered),
                "filters_applied": [f.__dict__ for f in parsed_filters],
                "results": filtered,
                "sort_by": sort_by,
                "sort_order": sort_order
            }

        except Exception as e:
            logger.error(f"Screening failed: {e}")
            return {"error": str(e)}

    async def get_stock_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Get complete fundamental data for a single stock"""
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(self.executor, self._fetch_fundamentals, symbol)

        if 'error' in data:
            return data

        # Add sector comparison
        sector = data.get('sector')
        if sector and sector in self.SECTOR_BENCHMARKS:
            benchmarks = self.SECTOR_BENCHMARKS[sector]
            comparison = {}

            pe = data.get('pe_ratio')
            if pe and benchmarks.get('pe_ratio'):
                diff = ((pe - benchmarks['pe_ratio']) / benchmarks['pe_ratio']) * 100
                comparison['pe_vs_sector'] = {
                    "value": round(pe, 2),
                    "benchmark": benchmarks['pe_ratio'],
                    "diff_pct": round(diff, 1),
                    "assessment": "Expensive" if diff > 20 else "Cheap" if diff < -20 else "Fair"
                }

            roe = data.get('roe')
            if roe and benchmarks.get('roe'):
                roe_decimal = roe / 100 if roe > 1 else roe
                diff = ((roe_decimal - benchmarks['roe']) / benchmarks['roe']) * 100
                comparison['roe_vs_sector'] = {
                    "value": round(roe, 2),
                    "benchmark": benchmarks['roe'] * 100,
                    "diff_pct": round(diff, 1),
                    "assessment": "Strong" if diff > 20 else "Weak" if diff < -20 else "Average"
                }

            data['sector_comparison'] = comparison

        # Add investment style classification
        data['investment_style'] = self._classify_investment_style(data)

        return data

    def _classify_investment_style(self, data: Dict) -> Dict[str, Any]:
        """Classify stock into investment styles (Value, Growth, Blend, etc.)"""
        pe = data.get('pe_ratio')
        pb = data.get('pb_ratio')
        roe = data.get('roe')
        revenue_growth = data.get('revenue_growth')
        earnings_growth = data.get('earnings_growth')
        div_yield = data.get('dividend_yield')
        market_cap = data.get('market_cap_cr', 0)

        styles = []
        scores = {
            "value": 0,
            "growth": 0,
            "quality": 0,
            "dividend": 0,
            "momentum": 0
        }

        # Value characteristics
        if pe and pe < 15:
            scores["value"] += 30
        if pb and pb < 2:
            scores["value"] += 20
        if div_yield and div_yield > 2:
            scores["value"] += 20

        # Growth characteristics
        if revenue_growth and revenue_growth > 15:
            scores["growth"] += 30
        if earnings_growth and earnings_growth > 20:
            scores["growth"] += 30
        if pe and pe > 25:
            scores["growth"] += 10

        # Quality characteristics
        if roe and roe > 18:
            scores["quality"] += 40
        if data.get('debt_to_equity') and data['debt_to_equity'] < 0.5:
            scores["quality"] += 30

        # Dividend characteristics
        if div_yield and div_yield > 3:
            scores["dividend"] += 50

        # Determine primary style
        primary_style = max(scores, key=scores.get)
        primary_score = scores[primary_style]

        # Size classification
        if market_cap > 50000:
            size = "Large Cap"
        elif market_cap > 10000:
            size = "Mid Cap"
        elif market_cap > 2000:
            size = "Small Cap"
        else:
            size = "Micro Cap"

        return {
            "primary_style": primary_style.title(),
            "style_scores": scores,
            "size_category": size,
            "suitable_for": self._get_suitable_investors(primary_style, size)
        }

    def _get_suitable_investors(self, style: str, size: str) -> List[str]:
        """Get investor profiles this stock is suitable for"""
        suitable = []

        if style == "value":
            suitable.extend(["Long-term investors", "Warren Buffett followers", "Contrarian investors"])
        elif style == "growth":
            suitable.extend(["Growth investors", "Aggressive investors", "Wealth builders"])
        elif style == "quality":
            suitable.extend(["Conservative investors", "Quality-focused investors", "Long-term holders"])
        elif style == "dividend":
            suitable.extend(["Income seekers", "Retirees", "Passive income investors"])

        if size == "Large Cap":
            suitable.append("Risk-averse investors")
        elif size in ["Small Cap", "Micro Cap"]:
            suitable.append("High-risk tolerance investors")

        return suitable

    def get_preset_screens(self) -> List[Dict[str, Any]]:
        """Get predefined screening presets"""
        return [
            {
                "name": "Value Picks",
                "description": "Low PE, Low PB, High Dividend stocks",
                "filters": [
                    {"metric": "pe_ratio", "operator": "lt", "value": 15},
                    {"metric": "pb_ratio", "operator": "lt", "value": 2},
                    {"metric": "dividend_yield", "operator": "gt", "value": 2}
                ],
                "sort_by": "dividend_yield",
                "sort_order": "desc"
            },
            {
                "name": "Quality Growth",
                "description": "High ROE, Low Debt, Growing earnings",
                "filters": [
                    {"metric": "roe", "operator": "gt", "value": 18},
                    {"metric": "debt_to_equity", "operator": "lt", "value": 0.5},
                    {"metric": "profit_margin", "operator": "gt", "value": 10}
                ],
                "sort_by": "roe",
                "sort_order": "desc"
            },
            {
                "name": "Dividend Champions",
                "description": "High yield, stable companies",
                "filters": [
                    {"metric": "dividend_yield", "operator": "gt", "value": 3},
                    {"metric": "pe_ratio", "operator": "lt", "value": 25},
                    {"metric": "market_cap", "operator": "gt", "value": 10000000000}
                ],
                "sort_by": "dividend_yield",
                "sort_order": "desc"
            },
            {
                "name": "Growth Rockets",
                "description": "High growth, high momentum stocks",
                "filters": [
                    {"metric": "revenue_growth", "operator": "gt", "value": 20},
                    {"metric": "roe", "operator": "gt", "value": 15}
                ],
                "sort_by": "revenue_growth",
                "sort_order": "desc"
            },
            {
                "name": "Financially Strong",
                "description": "Low debt, high liquidity",
                "filters": [
                    {"metric": "debt_to_equity", "operator": "lt", "value": 0.3},
                    {"metric": "current_ratio", "operator": "gt", "value": 1.5},
                    {"metric": "roe", "operator": "gt", "value": 12}
                ],
                "sort_by": "current_ratio",
                "sort_order": "desc"
            },
            {
                "name": "Undervalued Large Caps",
                "description": "Large caps trading below fair value",
                "filters": [
                    {"metric": "market_cap", "operator": "gt", "value": 50000000000},
                    {"metric": "pe_ratio", "operator": "lt", "value": 20},
                    {"metric": "pb_ratio", "operator": "lt", "value": 3}
                ],
                "sort_by": "pe_ratio",
                "sort_order": "asc"
            }
        ]


# Global instance
_screener_instance = None


def get_fundamental_screener() -> FundamentalScreener:
    """Get or create global screener instance"""
    global _screener_instance
    if _screener_instance is None:
        _screener_instance = FundamentalScreener()
    return _screener_instance
