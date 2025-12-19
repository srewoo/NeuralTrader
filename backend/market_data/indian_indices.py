"""
Indian Stock Market Indices Data
Fetches BSE, NIFTY indices with gainers/losers
"""

from typing import Dict, List, Any, Optional
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

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

    async def get_index_data(self, index_name: str) -> Optional[Dict[str, Any]]:
        """
        Get current data for a specific index

        Args:
            index_name: Name of the index (e.g., 'NIFTY_50', 'SENSEX')

        Returns:
            Dictionary with index data
        """
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
                return await self._calculate_synthetic_metal_index()

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
                return await self._calculate_synthetic_metal_index()
            return None

    async def _calculate_synthetic_metal_index(self) -> Optional[Dict[str, Any]]:
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

    async def get_constituents_performance(self, index_name: str) -> Dict[str, Any]:
        """
        Get performance of all stocks in an index

        Args:
            index_name: Name of the index

        Returns:
            Dictionary with gainers and losers
        """
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

    async def get_market_overview(self) -> Dict[str, Any]:
        """
        Get complete market overview with all indices and top gainers/losers

        Returns:
            Dictionary with market overview data
        """
        try:
            # Get all indices
            indices = await self.get_all_indices()

            # Get gainers/losers for major indices
            nifty_50_performance = await self.get_constituents_performance("NIFTY_50")
            bank_nifty_performance = await self.get_constituents_performance("NIFTY_BANK")
            nifty_metal_performance = await self.get_constituents_performance("NIFTY_METAL")
            sensex_performance = await self.get_constituents_performance("SENSEX")

            return {
                "indices": indices,
                "market_movers": {
                    "NIFTY_50": nifty_50_performance,
                    "NIFTY_BANK": bank_nifty_performance,
                    "NIFTY_METAL": nifty_metal_performance,
                    "SENSEX": sensex_performance
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching market overview: {e}")
            return {"indices": [], "market_movers": {}}


def get_indian_indices_data() -> IndianIndicesData:
    """Factory function to create IndianIndicesData instance"""
    return IndianIndicesData()
