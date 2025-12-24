"""
FII/DII Flow Tracker
Fetches and analyzes Foreign Institutional Investor (FII) and
Domestic Institutional Investor (DII) flow data from NSE.
"""

import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import json
import re
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class FIIDIITracker:
    """
    Track FII and DII flows in Indian markets.
    Data sourced from NSE India.
    """

    # NSE endpoints (may need periodic updates as NSE changes URLs)
    NSE_FII_DII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"
    NSE_BASE_URL = "https://www.nseindia.com"

    # Headers to mimic browser request (NSE blocks direct API calls)
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/reports-indices-historical-index-data",
        "Connection": "keep-alive",
    }

    def __init__(self):
        self.session = None
        self.cookies = None

    async def _get_session(self):
        """Get or create aiohttp session with NSE cookies"""
        if self.session is None:
            # Create connector with SSL verification disabled for NSE India
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self.session = aiohttp.ClientSession(headers=self.HEADERS, connector=connector)

            # First visit NSE homepage to get cookies
            try:
                async with self.session.get(self.NSE_BASE_URL, timeout=10) as response:
                    self.cookies = response.cookies
            except Exception as e:
                logger.warning(f"Failed to get NSE cookies: {e}")

        return self.session

    async def fetch_daily_fii_dii(self) -> Dict[str, Any]:
        """
        Fetch today's FII/DII data from NSE.
        Returns buy/sell values for both categories.
        """
        try:
            session = await self._get_session()

            async with session.get(
                self.NSE_FII_DII_URL,
                cookies=self.cookies,
                timeout=15
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_fii_dii_data(data)
                else:
                    logger.warning(f"NSE API returned status {response.status}")
                    return await self._get_fallback_data()

        except Exception as e:
            logger.error(f"Failed to fetch FII/DII data: {e}")
            return await self._get_fallback_data()

    def _parse_fii_dii_data(self, raw_data: Any) -> Dict[str, Any]:
        """Parse NSE FII/DII response"""
        try:
            result = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "fii": {
                    "buy_value": 0,
                    "sell_value": 0,
                    "net_value": 0
                },
                "dii": {
                    "buy_value": 0,
                    "sell_value": 0,
                    "net_value": 0
                },
                "source": "NSE India",
                "timestamp": datetime.now().isoformat()
            }

            if isinstance(raw_data, list):
                for item in raw_data:
                    category = item.get('category', '').upper()
                    if 'FII' in category or 'FPI' in category:
                        result['fii']['buy_value'] = self._parse_value(item.get('buyValue', 0))
                        result['fii']['sell_value'] = self._parse_value(item.get('sellValue', 0))
                        result['fii']['net_value'] = self._parse_value(item.get('netValue', 0))
                    elif 'DII' in category:
                        result['dii']['buy_value'] = self._parse_value(item.get('buyValue', 0))
                        result['dii']['sell_value'] = self._parse_value(item.get('sellValue', 0))
                        result['dii']['net_value'] = self._parse_value(item.get('netValue', 0))

            # Calculate net position
            result['net_institutional'] = result['fii']['net_value'] + result['dii']['net_value']

            # Add interpretation
            result['interpretation'] = self._interpret_flows(result)

            return result

        except Exception as e:
            logger.error(f"Error parsing FII/DII data: {e}")
            return {"error": str(e)}

    def _parse_value(self, value: Any) -> float:
        """Parse value from string or number"""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Remove commas and convert
            clean = value.replace(',', '').replace('₹', '').strip()
            try:
                return float(clean)
            except:
                return 0.0
        return 0.0

    async def _get_fallback_data(self) -> Dict[str, Any]:
        """
        Fallback when live API fails.
        Returns sample data structure with a note.
        """
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "fii": {
                "buy_value": None,
                "sell_value": None,
                "net_value": None
            },
            "dii": {
                "buy_value": None,
                "sell_value": None,
                "net_value": None
            },
            "net_institutional": None,
            "source": "Fallback - Live data unavailable",
            "note": "NSE API may be temporarily unavailable. Try again later.",
            "timestamp": datetime.now().isoformat()
        }

    def _interpret_flows(self, data: Dict) -> Dict[str, Any]:
        """Interpret FII/DII flows for trading signals"""
        fii_net = data['fii']['net_value'] or 0
        dii_net = data['dii']['net_value'] or 0
        total_net = fii_net + dii_net

        interpretation = {
            "fii_stance": "Neutral",
            "dii_stance": "Neutral",
            "overall_bias": "Neutral",
            "signal_strength": "Weak",
            "commentary": ""
        }

        # FII interpretation (values in crores)
        if fii_net > 1000:
            interpretation["fii_stance"] = "Strong Buying"
        elif fii_net > 500:
            interpretation["fii_stance"] = "Moderate Buying"
        elif fii_net > 0:
            interpretation["fii_stance"] = "Mild Buying"
        elif fii_net < -1000:
            interpretation["fii_stance"] = "Strong Selling"
        elif fii_net < -500:
            interpretation["fii_stance"] = "Moderate Selling"
        elif fii_net < 0:
            interpretation["fii_stance"] = "Mild Selling"

        # DII interpretation
        if dii_net > 1000:
            interpretation["dii_stance"] = "Strong Buying"
        elif dii_net > 500:
            interpretation["dii_stance"] = "Moderate Buying"
        elif dii_net > 0:
            interpretation["dii_stance"] = "Mild Buying"
        elif dii_net < -1000:
            interpretation["dii_stance"] = "Strong Selling"
        elif dii_net < -500:
            interpretation["dii_stance"] = "Moderate Selling"
        elif dii_net < 0:
            interpretation["dii_stance"] = "Mild Selling"

        # Overall bias
        if total_net > 1500:
            interpretation["overall_bias"] = "Strongly Bullish"
            interpretation["signal_strength"] = "Strong"
        elif total_net > 500:
            interpretation["overall_bias"] = "Bullish"
            interpretation["signal_strength"] = "Moderate"
        elif total_net > 0:
            interpretation["overall_bias"] = "Mildly Bullish"
            interpretation["signal_strength"] = "Weak"
        elif total_net < -1500:
            interpretation["overall_bias"] = "Strongly Bearish"
            interpretation["signal_strength"] = "Strong"
        elif total_net < -500:
            interpretation["overall_bias"] = "Bearish"
            interpretation["signal_strength"] = "Moderate"
        elif total_net < 0:
            interpretation["overall_bias"] = "Mildly Bearish"
            interpretation["signal_strength"] = "Weak"

        # Generate commentary
        if fii_net > 0 and dii_net > 0:
            interpretation["commentary"] = "Both FIIs and DIIs are buying - strong institutional support for markets."
        elif fii_net < 0 and dii_net < 0:
            interpretation["commentary"] = "Both FIIs and DIIs are selling - institutional exodus, caution advised."
        elif fii_net < 0 and dii_net > 0:
            interpretation["commentary"] = "FIIs selling but DIIs absorbing - domestic institutions providing support."
        elif fii_net > 0 and dii_net < 0:
            interpretation["commentary"] = "FIIs buying while DIIs booking profits - foreign money driving markets."
        else:
            interpretation["commentary"] = "Institutional activity is mixed - no clear direction."

        return interpretation

    async def get_historical_flows(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get historical FII/DII data.
        Note: This would need web scraping or stored data.
        For now, returns structure for future implementation.
        """
        # TODO: Implement historical data fetching
        # Options:
        # 1. Store daily data in MongoDB
        # 2. Scrape NSE historical reports
        # 3. Use third-party data providers

        return {
            "message": "Historical data feature coming soon",
            "recommendation": "Store daily FII/DII data to build history",
            "implementation": [
                "1. Create MongoDB collection for fii_dii_history",
                "2. Run daily cron job to fetch and store data",
                "3. Query collection for historical analysis"
            ]
        }

    async def get_monthly_summary(self) -> Dict[str, Any]:
        """
        Get month-to-date FII/DII summary.
        Requires historical data storage.
        """
        # Placeholder for when historical data is available
        return {
            "period": "MTD",
            "message": "Monthly summary requires historical data storage",
            "current_day": await self.fetch_daily_fii_dii()
        }

    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None


class BulkBlockDealsTracker:
    """
    Track Bulk and Block deals from NSE/BSE.
    Bulk deals: > 0.5% of equity
    Block deals: > 5 lakh shares or ₹10 crore value
    """

    NSE_BULK_DEALS_URL = "https://www.nseindia.com/api/snapshot-capital-market-largedeal"
    NSE_BLOCK_DEALS_URL = "https://www.nseindia.com/api/block-deal"

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/market-data/bulk-and-block-deals",
    }

    def __init__(self):
        self.session = None
        self.cookies = None

    async def _get_session(self):
        """Get or create session with cookies"""
        if self.session is None:
            self.session = aiohttp.ClientSession(headers=self.HEADERS)
            try:
                async with self.session.get("https://www.nseindia.com", timeout=10) as response:
                    self.cookies = response.cookies
            except:
                pass
        return self.session

    async def fetch_bulk_deals(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Fetch bulk deals for a given date (or today)"""
        try:
            session = await self._get_session()

            async with session.get(
                self.NSE_BULK_DEALS_URL,
                cookies=self.cookies,
                timeout=15
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_bulk_deals(data)
                else:
                    return {"error": f"API returned status {response.status}", "deals": []}

        except Exception as e:
            logger.error(f"Failed to fetch bulk deals: {e}")
            return {"error": str(e), "deals": []}

    async def fetch_block_deals(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Fetch block deals for a given date (or today)"""
        try:
            session = await self._get_session()

            async with session.get(
                self.NSE_BLOCK_DEALS_URL,
                cookies=self.cookies,
                timeout=15
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_block_deals(data)
                else:
                    return {"error": f"API returned status {response.status}", "deals": []}

        except Exception as e:
            logger.error(f"Failed to fetch block deals: {e}")
            return {"error": str(e), "deals": []}

    def _parse_bulk_deals(self, raw_data: Any) -> Dict[str, Any]:
        """Parse bulk deals response"""
        deals = []

        try:
            data_list = raw_data.get('data', []) if isinstance(raw_data, dict) else raw_data

            for item in data_list:
                deal = {
                    "symbol": item.get('symbol', item.get('SYMBOL', '')),
                    "name": item.get('securityName', item.get('NAME', '')),
                    "client_name": item.get('clientName', item.get('CLIENT_NAME', '')),
                    "deal_type": item.get('buySell', item.get('BUYSELL', '')),
                    "quantity": self._parse_number(item.get('quantity', item.get('QTY', 0))),
                    "price": self._parse_number(item.get('avgPrice', item.get('PRICE', 0))),
                    "trade_date": item.get('dealDate', item.get('DATE', '')),
                }
                deal["value_cr"] = round((deal["quantity"] * deal["price"]) / 10000000, 2)

                # Classify the deal
                if 'BUY' in deal['deal_type'].upper():
                    deal['signal'] = 'Bullish'
                    deal['signal_icon'] = '🟢'
                else:
                    deal['signal'] = 'Bearish'
                    deal['signal_icon'] = '🔴'

                deals.append(deal)

        except Exception as e:
            logger.error(f"Error parsing bulk deals: {e}")

        # Group by symbol
        by_symbol = {}
        for deal in deals:
            sym = deal['symbol']
            if sym not in by_symbol:
                by_symbol[sym] = {"buys": [], "sells": []}
            if deal['signal'] == 'Bullish':
                by_symbol[sym]['buys'].append(deal)
            else:
                by_symbol[sym]['sells'].append(deal)

        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_deals": len(deals),
            "deals": deals,
            "by_symbol": by_symbol,
            "summary": self._summarize_deals(deals)
        }

    def _parse_block_deals(self, raw_data: Any) -> Dict[str, Any]:
        """Parse block deals response"""
        deals = []

        try:
            data_list = raw_data.get('data', []) if isinstance(raw_data, dict) else raw_data

            for item in data_list:
                deal = {
                    "symbol": item.get('symbol', ''),
                    "name": item.get('securityName', ''),
                    "client_name": item.get('clientName', 'N/A'),
                    "deal_type": item.get('buySell', ''),
                    "quantity": self._parse_number(item.get('quantity', 0)),
                    "price": self._parse_number(item.get('tradedPrice', 0)),
                    "trade_date": item.get('dealDate', ''),
                }
                deal["value_cr"] = round((deal["quantity"] * deal["price"]) / 10000000, 2)

                if 'BUY' in deal.get('deal_type', '').upper():
                    deal['signal'] = 'Bullish'
                else:
                    deal['signal'] = 'Bearish'

                deals.append(deal)

        except Exception as e:
            logger.error(f"Error parsing block deals: {e}")

        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_deals": len(deals),
            "deals": deals,
            "summary": self._summarize_deals(deals)
        }

    def _parse_number(self, value: Any) -> float:
        """Parse number from various formats"""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            clean = value.replace(',', '').replace(' ', '')
            try:
                return float(clean)
            except:
                return 0.0
        return 0.0

    def _summarize_deals(self, deals: List[Dict]) -> Dict[str, Any]:
        """Summarize deal activity"""
        buys = [d for d in deals if d.get('signal') == 'Bullish']
        sells = [d for d in deals if d.get('signal') == 'Bearish']

        buy_value = sum(d.get('value_cr', 0) for d in buys)
        sell_value = sum(d.get('value_cr', 0) for d in sells)

        return {
            "total_buys": len(buys),
            "total_sells": len(sells),
            "buy_value_cr": round(buy_value, 2),
            "sell_value_cr": round(sell_value, 2),
            "net_value_cr": round(buy_value - sell_value, 2),
            "bias": "Bullish" if buy_value > sell_value else "Bearish" if sell_value > buy_value else "Neutral"
        }

    async def close(self):
        """Close session"""
        if self.session:
            await self.session.close()
            self.session = None


# Global instances
_fii_dii_tracker = None
_bulk_block_tracker = None


def get_fii_dii_tracker() -> FIIDIITracker:
    """Get or create FII/DII tracker instance"""
    global _fii_dii_tracker
    if _fii_dii_tracker is None:
        _fii_dii_tracker = FIIDIITracker()
    return _fii_dii_tracker


def get_bulk_block_tracker() -> BulkBlockDealsTracker:
    """Get or create bulk/block deals tracker instance"""
    global _bulk_block_tracker
    if _bulk_block_tracker is None:
        _bulk_block_tracker = BulkBlockDealsTracker()
    return _bulk_block_tracker
