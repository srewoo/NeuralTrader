"""
Alpaca Live Market Data Integration
Real-time and historical market data using Alpaca Markets API
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class AlpacaLiveDataProvider:
    """
    Alpaca Markets data provider for real-time and historical data
    Supports stocks, crypto, and more
    """

    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        """
        Initialize Alpaca data provider

        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            paper: Use paper trading endpoint (default: True for safety)
        """
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.live import StockDataStream
            from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
            from alpaca.data.timeframe import TimeFrame

            self.api_key = api_key
            self.api_secret = api_secret
            self.paper = paper

            # Historical data client
            self.historical_client = StockHistoricalDataClient(api_key, api_secret)

            # Live data stream
            self.stream = StockDataStream(api_key, api_secret)

            # Import request types
            self.StockBarsRequest = StockBarsRequest
            self.StockLatestQuoteRequest = StockLatestQuoteRequest
            self.TimeFrame = TimeFrame

            self._running = False
            self._subscribers = {}  # symbol -> callbacks

            logger.info(f"Alpaca data provider initialized (paper={paper})")

        except ImportError:
            logger.error("alpaca-py not installed. Install with: pip install alpaca-py")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca client: {e}")
            raise

    async def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest quote for a symbol

        Args:
            symbol: Stock symbol (e.g., "AAPL")

        Returns:
            Dict with quote data or None if failed
        """
        try:
            request = self.StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = await asyncio.to_thread(
                self.historical_client.get_stock_latest_quote,
                request
            )

            if symbol in quotes:
                quote = quotes[symbol]
                return {
                    "symbol": symbol,
                    "bid": float(quote.bid_price),
                    "ask": float(quote.ask_price),
                    "bid_size": quote.bid_size,
                    "ask_size": quote.ask_size,
                    "timestamp": quote.timestamp.isoformat()
                }
            return None

        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return None

    async def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime = None,
        timeframe: str = "1Day"
    ) -> Optional[pd.DataFrame]:
        """
        Get historical bar data

        Args:
            symbol: Stock symbol
            start: Start date
            end: End date (default: now)
            timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            if end is None:
                end = datetime.now()

            # Map timeframe string to TimeFrame enum
            timeframe_map = {
                "1Min": self.TimeFrame.Minute,
                "5Min": self.TimeFrame(5, self.TimeFrame.Unit.Minute),
                "15Min": self.TimeFrame(15, self.TimeFrame.Unit.Minute),
                "1Hour": self.TimeFrame.Hour,
                "1Day": self.TimeFrame.Day,
            }

            tf = timeframe_map.get(timeframe, self.TimeFrame.Day)

            request = self.StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end
            )

            bars = await asyncio.to_thread(
                self.historical_client.get_stock_bars,
                request
            )

            if symbol in bars:
                df = bars[symbol].df
                # Rename columns to match our standard format
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                return df

            return None

        except Exception as e:
            logger.error(f"Failed to get historical bars for {symbol}: {e}")
            return None

    async def subscribe_to_quotes(self, symbol: str, callback):
        """
        Subscribe to real-time quotes for a symbol

        Args:
            symbol: Stock symbol
            callback: Async function to call with quote data
        """
        if symbol not in self._subscribers:
            self._subscribers[symbol] = []

        self._subscribers[symbol].append(callback)

        # Register handler with Alpaca stream
        async def quote_handler(data):
            quote_data = {
                "symbol": data.symbol,
                "bid": float(data.bid_price),
                "ask": float(data.ask_price),
                "bid_size": data.bid_size,
                "ask_size": data.ask_size,
                "timestamp": data.timestamp.isoformat()
            }

            # Call all subscribers for this symbol
            for cb in self._subscribers.get(data.symbol, []):
                try:
                    await cb(quote_data)
                except Exception as e:
                    logger.error(f"Error in quote callback: {e}")

        self.stream.subscribe_quotes(quote_handler, symbol)
        logger.info(f"Subscribed to quotes for {symbol}")

    async def subscribe_to_trades(self, symbol: str, callback):
        """
        Subscribe to real-time trades for a symbol

        Args:
            symbol: Stock symbol
            callback: Async function to call with trade data
        """
        async def trade_handler(data):
            trade_data = {
                "symbol": data.symbol,
                "price": float(data.price),
                "size": data.size,
                "timestamp": data.timestamp.isoformat(),
                "conditions": data.conditions
            }

            try:
                await callback(trade_data)
            except Exception as e:
                logger.error(f"Error in trade callback: {e}")

        self.stream.subscribe_trades(trade_handler, symbol)
        logger.info(f"Subscribed to trades for {symbol}")

    async def start_stream(self):
        """Start the real-time data stream"""
        if self._running:
            logger.warning("Stream already running")
            return

        self._running = True
        logger.info("Starting Alpaca data stream...")

        try:
            await asyncio.to_thread(self.stream.run)
        except Exception as e:
            logger.error(f"Stream error: {e}")
            self._running = False

    async def stop_stream(self):
        """Stop the real-time data stream"""
        if not self._running:
            return

        self._running = False
        try:
            await asyncio.to_thread(self.stream.close)
            logger.info("Alpaca data stream stopped")
        except Exception as e:
            logger.error(f"Error stopping stream: {e}")


# Singleton instance
_alpaca_instance: Optional[AlpacaLiveDataProvider] = None


def get_alpaca_provider(api_key: str, api_secret: str, paper: bool = True) -> AlpacaLiveDataProvider:
    """Get or create Alpaca provider instance"""
    global _alpaca_instance

    if _alpaca_instance is None:
        _alpaca_instance = AlpacaLiveDataProvider(api_key, api_secret, paper)

    return _alpaca_instance
