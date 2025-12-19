"""
IEX Cloud Data Provider
Real-time and historical market data using IEX Cloud API
"""

import logging
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class IEXCloudProvider:
    """
    IEX Cloud data provider for market data
    Supports stocks, fundamentals, news, and more
    """

    BASE_URL = "https://cloud.iexapis.com/stable"

    def __init__(self, api_key: str, sandbox: bool = False):
        """
        Initialize IEX Cloud provider

        Args:
            api_key: IEX Cloud API token
            sandbox: Use sandbox environment (default: False)
        """
        self.api_key = api_key
        self.sandbox = sandbox

        if sandbox:
            self.base_url = "https://sandbox.iexapis.com/stable"
        else:
            self.base_url = self.BASE_URL

        logger.info(f"IEX Cloud provider initialized (sandbox={sandbox})")

    async def _request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request to IEX Cloud"""
        if params is None:
            params = {}

        params["token"] = self.api_key

        url = f"{self.base_url}/{endpoint}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"IEX API error: {response.status} - {await response.text()}")
                        return None
        except Exception as e:
            logger.error(f"IEX request failed: {e}")
            return None

    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time quote for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Dict with quote data
        """
        data = await self._request(f"stock/{symbol}/quote")

        if data:
            return {
                "symbol": symbol,
                "price": data.get("latestPrice"),
                "change": data.get("change"),
                "change_percent": data.get("changePercent"),
                "volume": data.get("latestVolume"),
                "market_cap": data.get("marketCap"),
                "pe_ratio": data.get("peRatio"),
                "week_52_high": data.get("week52High"),
                "week_52_low": data.get("week52Low"),
                "avg_volume": data.get("avgTotalVolume"),
                "timestamp": data.get("latestUpdate")
            }

        return None

    async def get_historical_prices(
        self,
        symbol: str,
        range_period: str = "1m"
    ) -> Optional[pd.DataFrame]:
        """
        Get historical price data

        Args:
            symbol: Stock symbol
            range_period: Time range (5d, 1m, 3m, 6m, 1y, 2y, 5y, max)

        Returns:
            DataFrame with OHLCV data
        """
        data = await self._request(f"stock/{symbol}/chart/{range_period}")

        if data and isinstance(data, list):
            df = pd.DataFrame(data)

            # Rename columns to match our standard format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'date': 'Date'
            })

            # Set date as index
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

            return df[['Open', 'High', 'Low', 'Close', 'Volume']]

        return None

    async def get_company_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company information"""
        data = await self._request(f"stock/{symbol}/company")

        if data:
            return {
                "symbol": symbol,
                "name": data.get("companyName"),
                "description": data.get("description"),
                "CEO": data.get("CEO"),
                "sector": data.get("sector"),
                "industry": data.get("industry"),
                "employees": data.get("employees"),
                "website": data.get("website"),
                "exchange": data.get("exchange")
            }

        return None

    async def get_key_stats(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get key statistics"""
        data = await self._request(f"stock/{symbol}/stats")

        if data:
            return {
                "symbol": symbol,
                "market_cap": data.get("marketcap"),
                "pe_ratio": data.get("peRatio"),
                "forward_pe": data.get("forwardPERatio"),
                "peg_ratio": data.get("pegRatio"),
                "price_to_sales": data.get("priceToSales"),
                "price_to_book": data.get("priceToBook"),
                "dividend_yield": data.get("dividendYield"),
                "profit_margin": data.get("profitMargin"),
                "debt_to_equity": data.get("debtToEquity"),
                "roe": data.get("returnOnEquity"),
                "roa": data.get("returnOnAssets"),
                "revenue_growth": data.get("revenuePerShareTTM"),
                "earnings_growth": data.get("ttmEPS")
            }

        return None

    async def get_news(self, symbol: str, last_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get latest news for a symbol

        Args:
            symbol: Stock symbol
            last_n: Number of articles to retrieve

        Returns:
            List of news articles
        """
        data = await self._request(f"stock/{symbol}/news/last/{last_n}")

        if data and isinstance(data, list):
            articles = []
            for item in data:
                articles.append({
                    "headline": item.get("headline"),
                    "summary": item.get("summary"),
                    "source": item.get("source"),
                    "url": item.get("url"),
                    "datetime": item.get("datetime"),
                    "related": item.get("related")
                })
            return articles

        return []

    async def get_earnings(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """Get earnings data"""
        data = await self._request(f"stock/{symbol}/earnings")

        if data and "earnings" in data:
            return data["earnings"]

        return None

    async def get_financials(self, symbol: str, period: str = "annual") -> Optional[List[Dict[str, Any]]]:
        """
        Get financial statements

        Args:
            symbol: Stock symbol
            period: 'annual' or 'quarter'

        Returns:
            List of financial statements
        """
        data = await self._request(f"stock/{symbol}/financials", {"period": period})

        if data and "financials" in data:
            return data["financials"]

        return None

    async def get_cash_flow(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """Get cash flow statement"""
        data = await self._request(f"stock/{symbol}/cash-flow")

        if data and "cashflow" in data:
            return data["cashflow"]

        return None

    async def get_balance_sheet(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """Get balance sheet"""
        data = await self._request(f"stock/{symbol}/balance-sheet")

        if data and "balancesheet" in data:
            return data["balancesheet"]

        return None


# Singleton instance
_iex_instance: Optional[IEXCloudProvider] = None


def get_iex_provider(api_key: str, sandbox: bool = False) -> IEXCloudProvider:
    """Get or create IEX Cloud provider instance"""
    global _iex_instance

    if _iex_instance is None:
        _iex_instance = IEXCloudProvider(api_key, sandbox)

    return _iex_instance
