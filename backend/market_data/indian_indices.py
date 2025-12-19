"""
Indian Stock Market Indices Data
Fetches BSE, NIFTY indices with gainers/losers
"""

from typing import Dict, List, Any, Optional
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


# Indian Indices Symbols (Yahoo Finance format)
INDIAN_INDICES = {
    "NIFTY_50": "^NSEI",
    "SENSEX": "^BSESN",
    "NIFTY_BANK": "^NSEBANK",
    "NIFTY_IT": "NIFTY_IT.NS",
    "NIFTY_METAL": "^CNXMETAL",
    "NIFTY_PHARMA": "NIFTY_PHARMA.NS",
    "NIFTY_AUTO": "NIFTY_AUTO.NS",
    "NIFTY_FMCG": "NIFTY_FMCG.NS",
    "NIFTY_REALTY": "NIFTY_REALTY.NS",
    "NIFTY_ENERGY": "NIFTY_ENERGY.NS",
}

# Index constituents (major stocks in each index)
INDEX_CONSTITUENTS = {
    "NIFTY_50": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS",
        "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
        "HCLTECH.NS", "WIPRO.NS", "TITAN.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
        "SUNPHARMA.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "M&M.NS",
        "TATAMOTORS.NS", "TECHM.NS", "ADANIPORTS.NS", "DIVISLAB.NS", "BRITANNIA.NS"
    ],
    "NIFTY_BANK": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS",
        "BANDHANBNK.NS", "AUBANK.NS"
    ],
    "NIFTY_METAL": [
        "TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "COALINDIA.NS", "VEDL.NS",
        "HINDZINC.NS", "NMDC.NS", "NATIONALUM.NS", "SAIL.NS", "JINDALSTEL.NS",
        "RATNAMANI.NS", "WELCORP.NS", "WELSPUNIND.NS", "APARINDS.NS"
    ],
    "NIFTY_IT": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
        "LTIM.NS", "MPHASIS.NS", "COFORGE.NS", "PERSISTENT.NS", "LTTS.NS"
    ],
    "SENSEX": [
        "RELIANCE.BO", "TCS.BO", "HDFCBANK.BO", "INFY.BO", "ICICIBANK.BO",
        "HINDUNILVR.BO", "ITC.BO", "SBIN.BO", "BHARTIARTL.BO", "BAJFINANCE.BO",
        "KOTAKBANK.BO", "LT.BO", "AXISBANK.BO", "ASIANPAINT.BO", "MARUTI.BO",
        "HCLTECH.BO", "WIPRO.BO", "TITAN.BO", "ULTRACEMCO.BO", "NESTLEIND.BO",
        "SUNPHARMA.BO", "ONGC.BO", "NTPC.BO", "POWERGRID.BO", "M&M.BO",
        "TATAMOTORS.BO", "TECHM.BO", "ADANIPORTS.BO", "DIVISLAB.BO", "BRITANNIA.BO"
    ]
}


class IndianIndicesData:
    """Fetch Indian stock market indices and constituents data"""

    def __init__(self):
        self.indices = INDIAN_INDICES
        self.constituents = INDEX_CONSTITUENTS
        self.executor = ThreadPoolExecutor(max_workers=10)

        # Cache for market overview (to avoid Yahoo Finance rate limits)
        self._cache = {}
        self._cache_ttl = 30  # seconds

    def _get_index_data_sync(self, index_name: str) -> Optional[Dict[str, Any]]:
        """Synchronous version for thread pool execution"""
        try:
            symbol = self.indices.get(index_name)
            if not symbol:
                logger.error(f"Unknown index: {index_name}")
                return None

            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="2d")  # Get last 2 days

            # If Yahoo Finance doesn't have the index, calculate synthetic index from constituents
            if hist.empty and index_name == "NIFTY_METAL":
                logger.info(f"Calculating synthetic {index_name} from constituents")
                return self._calculate_synthetic_metal_index_sync()

            if hist.empty:
                logger.error(f"No data for {index_name}")
                return None

            current_price = float(hist['Close'].iloc[-1])
            previous_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
            change = current_price - previous_close
            change_percent = (change / previous_close * 100) if previous_close else 0

            return {
                "name": index_name,
                "symbol": symbol,
                "current_value": round(current_price, 2),
                "previous_close": round(previous_close, 2),
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "high": round(float(hist['High'].iloc[-1]), 2),
                "low": round(float(hist['Low'].iloc[-1]), 2),
                "volume": int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching {index_name}: {e}")
            # Fallback to synthetic calculation for NIFTY_METAL
            if index_name == "NIFTY_METAL":
                return self._calculate_synthetic_metal_index_sync()
            return None

    async def get_index_data(self, index_name: str) -> Optional[Dict[str, Any]]:
        """Async wrapper that runs sync code in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._get_index_data_sync, index_name)

    def _calculate_synthetic_metal_index_sync(self) -> Optional[Dict[str, Any]]:
        """Calculate NIFTY METAL index from constituent stocks when Yahoo Finance doesn't have it"""
        try:
            symbols = self.constituents.get("NIFTY_METAL", [])
            if not symbols:
                return None

            total_change_percent = 0
            valid_stocks = 0
            total_high = 0
            total_low = 0

            for symbol in symbols[:5]:  # Use top 5 stocks for speed
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="2d")

                    if hist.empty or len(hist) < 2:
                        continue

                    current_price = float(hist['Close'].iloc[-1])
                    previous_close = float(hist['Close'].iloc[-2])
                    change_percent = (current_price - previous_close) / previous_close * 100

                    total_change_percent += change_percent
                    total_high += float(hist['High'].iloc[-1])
                    total_low += float(hist['Low'].iloc[-1])
                    valid_stocks += 1

                except Exception:
                    continue

            if valid_stocks == 0:
                return None

            avg_change_percent = total_change_percent / valid_stocks
            # Use a base value for display purposes
            base_value = 10000
            current_value = base_value * (1 + avg_change_percent / 100)
            previous_close = base_value
            change = current_value - previous_close

            return {
                "name": "NIFTY_METAL",
                "symbol": "NIFTY_METAL.NS (Synthetic)",
                "current_value": round(current_value, 2),
                "previous_close": round(previous_close, 2),
                "change": round(change, 2),
                "change_percent": round(avg_change_percent, 2),
                "high": round(current_value * 1.005, 2),  # Estimated
                "low": round(current_value * 0.995, 2),   # Estimated
                "volume": 0,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error calculating synthetic NIFTY_METAL: {e}")
            return None

    async def get_all_indices(self) -> List[Dict[str, Any]]:
        """Get data for all configured indices"""
        indices_data = []

        for index_name in self.indices.keys():
            data = await self.get_index_data(index_name)
            if data:
                indices_data.append(data)

        return indices_data

    def _get_constituents_performance_sync(self, index_name: str) -> Dict[str, Any]:
        """Synchronous version for thread pool execution"""
        try:
            symbols = self.constituents.get(index_name, [])
            if not symbols:
                logger.error(f"No constituents found for {index_name}")
                return {"gainers": [], "losers": []}

            stocks_data = []

            # Fetch data for all constituents
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="2d")

                    if hist.empty or len(hist) < 2:
                        continue

                    current_price = float(hist['Close'].iloc[-1])
                    previous_close = float(hist['Close'].iloc[-2])
                    change = current_price - previous_close
                    change_percent = (change / previous_close * 100) if previous_close else 0

                    # Get company name
                    info = ticker.info
                    company_name = info.get('longName', symbol.replace('.NS', '').replace('.BO', ''))

                    stocks_data.append({
                        "symbol": symbol,
                        "name": company_name,
                        "current_price": round(current_price, 2),
                        "previous_close": round(previous_close, 2),
                        "change": round(change, 2),
                        "change_percent": round(change_percent, 2),
                        "volume": int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0
                    })

                except Exception as e:
                    logger.debug(f"Error fetching {symbol}: {e}")
                    continue

            # Sort by change_percent
            stocks_data.sort(key=lambda x: x['change_percent'], reverse=True)

            # Get top 5 gainers and losers
            gainers = stocks_data[:5]
            losers = stocks_data[-5:][::-1]  # Reverse to show worst first

            return {
                "index": index_name,
                "gainers": gainers,
                "losers": losers,
                "total_stocks": len(stocks_data),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching constituents for {index_name}: {e}")
            return {"gainers": [], "losers": []}

    async def get_constituents_performance(self, index_name: str) -> Dict[str, Any]:
        """Async wrapper that runs sync code in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._get_constituents_performance_sync, index_name)

    async def get_market_overview(self) -> Dict[str, Any]:
        """
        Get complete market overview with all indices and top gainers/losers
        Optimized to fetch only displayed indices and make concurrent requests
        Includes caching to avoid Yahoo Finance rate limits

        Returns:
            Dictionary with market overview data
        """
        try:
            # Check cache
            cache_key = "market_overview"
            if cache_key in self._cache:
                cached_data, cached_time = self._cache[cache_key]
                if (datetime.now() - cached_time).total_seconds() < self._cache_ttl:
                    logger.info("Returning cached market overview")
                    return cached_data

            # Only fetch the 4 major indices displayed in the UI
            major_indices = ["NIFTY_50", "SENSEX", "NIFTY_BANK", "NIFTY_METAL"]

            # Fetch all data concurrently for better performance
            indices_tasks = [self.get_index_data(idx) for idx in major_indices]
            movers_tasks = [self.get_constituents_performance(idx) for idx in major_indices]

            # Execute all tasks concurrently
            indices_results, movers_results = await asyncio.gather(
                asyncio.gather(*indices_tasks, return_exceptions=True),
                asyncio.gather(*movers_tasks, return_exceptions=True)
            )

            # Filter out None and error results
            indices = [idx for idx in indices_results if idx and not isinstance(idx, Exception)]

            # Build market movers dict
            market_movers = {}
            for idx_name, movers in zip(major_indices, movers_results):
                if movers and not isinstance(movers, Exception):
                    market_movers[idx_name] = movers
                else:
                    market_movers[idx_name] = {"gainers": [], "losers": []}

            result = {
                "indices": indices,
                "market_movers": market_movers,
                "timestamp": datetime.now().isoformat()
            }

            # Cache the result
            self._cache[cache_key] = (result, datetime.now())

            return result

        except Exception as e:
            logger.error(f"Error fetching market overview: {e}")
            # Return cached data if available, even if expired
            if cache_key in self._cache:
                cached_data, _ = self._cache[cache_key]
                logger.warning("Returning expired cached data due to error")
                return cached_data
            return {"indices": [], "market_movers": {}}


# Singleton instance to preserve cache
_indices_data_instance = None

def get_indian_indices_data() -> IndianIndicesData:
    """Factory function to create/return singleton IndianIndicesData instance"""
    global _indices_data_instance
    if _indices_data_instance is None:
        _indices_data_instance = IndianIndicesData()
    return _indices_data_instance
