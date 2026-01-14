"""
Market Data Stream
Provides live market data by polling real data providers with GBM simulation fallback.
"""

import asyncio
import random
import logging
import math
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from .connection_manager import get_connection_manager
from analysis.enhanced_analyzer import get_enhanced_analyzer

logger = logging.getLogger(__name__)


class MarketStream:
    """
    Live market data stream with real data polling and GBM fallback.

    Polls real data from yfinance/providers every 5 seconds per symbol.
    Falls back to Geometric Brownian Motion simulation if live data fails.
    """

    # Rate limiting: minimum seconds between fetches per symbol
    FETCH_INTERVAL_SECONDS = 5

    def __init__(self):
        self.active = False
        self.manager = get_connection_manager()
        self.analyzer = None  # Lazy load to avoid circular import issues at startup
        self.alert_manager = None  # Lazy load alert manager
        self.watched_symbols = {}  # Dynamic - populated from user subscriptions or watchlist
        self.default_volatility = 0.002  # Default volatility for simulation fallback
        self._background_tasks = set()  # Track background tasks to prevent garbage collection

        # Live data tracking
        self._last_fetch_time: Dict[str, datetime] = {}  # Rate limiting
        self._live_data_enabled = True  # Can be disabled for testing
        self._data_provider = None  # Lazy loaded
        self._consecutive_failures: Dict[str, int] = {}  # Track failures per symbol

    def _get_data_provider(self):
        """Lazy load data provider factory"""
        if self._data_provider is None:
            try:
                from data_providers.factory import get_data_provider_factory
                self._data_provider = get_data_provider_factory()
                logger.info("Data provider factory initialized for live streaming")
            except Exception as e:
                logger.warning(f"Failed to initialize data provider: {e}")
                self._live_data_enabled = False
        return self._data_provider

    async def add_symbol(self, symbol: str, initial_price: float = None):
        """Add a symbol to watch dynamically"""
        if symbol not in self.watched_symbols:
            if initial_price is None:
                # Fetch real price from data provider
                try:
                    factory = self._get_data_provider()
                    if factory:
                        quote = await factory.get_quote(symbol)
                        if quote:
                            initial_price = quote.get('price') or quote.get('current_price', 100.0)
                        else:
                            initial_price = 100.0
                    else:
                        initial_price = 100.0
                except Exception:
                    initial_price = 100.0  # Fallback

            self.watched_symbols[symbol] = {
                "price": initial_price,
                "volatility": self.default_volatility,
                "prev_price": initial_price,
                "is_live": False  # Track if data is from live source
            }
            self._last_fetch_time[symbol] = datetime.min  # Allow immediate fetch
            self._consecutive_failures[symbol] = 0
            logger.info(f"Added {symbol} to market stream at price {initial_price}")

    async def _fetch_live_price(self, symbol: str) -> Optional[Dict]:
        """
        Fetch live price from data provider with rate limiting.

        Returns dict with price data or None if should use simulation.
        """
        if not self._live_data_enabled:
            return None

        # Rate limiting check
        last_fetch = self._last_fetch_time.get(symbol, datetime.min)
        time_since_fetch = (datetime.now() - last_fetch).total_seconds()

        if time_since_fetch < self.FETCH_INTERVAL_SECONDS:
            # Use cached price with small simulation variation
            return None

        try:
            factory = self._get_data_provider()
            if not factory:
                return None

            quote = await factory.get_quote(symbol)

            if quote:
                self._last_fetch_time[symbol] = datetime.now()
                self._consecutive_failures[symbol] = 0

                # Extract price from various possible formats
                price = None
                if hasattr(quote, 'price'):
                    price = quote.price
                elif isinstance(quote, dict):
                    price = quote.get('price') or quote.get('current_price') or quote.get('regularMarketPrice')

                if price and price > 0:
                    return {
                        "price": float(price),
                        "volume": quote.get('volume', 0) if isinstance(quote, dict) else getattr(quote, 'volume', 0),
                        "is_live": True
                    }

        except Exception as e:
            self._consecutive_failures[symbol] = self._consecutive_failures.get(symbol, 0) + 1

            # Only log every 5 failures to reduce noise
            if self._consecutive_failures[symbol] % 5 == 1:
                logger.warning(f"Live fetch failed for {symbol} ({self._consecutive_failures[symbol]}x): {e}")

            # If too many consecutive failures, reduce fetch attempts
            if self._consecutive_failures[symbol] > 20:
                # Increase rate limit for problematic symbols
                self._last_fetch_time[symbol] = datetime.now()

        return None

    def _simulate_tick(self, symbol: str, current_price: float, volatility: float) -> Dict:
        """
        Generate simulated price tick using Geometric Brownian Motion.

        Used as fallback when live data is unavailable.
        """
        shock = random.gauss(0, volatility)
        new_price = current_price * (1 + shock)

        return {
            "price": new_price,
            "change_pct": shock * 100,
            "volume": random.randint(100, 5000),
            "is_live": False
        }

    async def start_stream(self):
        """Start the market data stream"""
        if self.active:
            return

        self.active = True
        logger.info("Market Stream Started (Live Data Mode)")

        try:
            # Initialize analyzer (optional - stream works without it)
            if not self.analyzer:
                try:
                    self.analyzer = get_enhanced_analyzer()
                except Exception as e:
                    logger.warning(f"Enhanced analyzer not available: {e}")
                    self.analyzer = None

            while self.active:
                try:
                    # Skip if no symbols to watch
                    if not self.watched_symbols:
                        await asyncio.sleep(5)
                        continue

                    for symbol, data in list(self.watched_symbols.items()):
                        current_price = data["price"]
                        prev_price = data.get("prev_price", current_price)

                        # Try to fetch live price
                        live_data = await self._fetch_live_price(symbol)

                        if live_data:
                            # Use live data
                            new_price = live_data["price"]
                            change_pct = ((new_price - current_price) / current_price * 100) if current_price > 0 else 0
                            volume = live_data.get("volume", 0)
                            is_live = True
                        else:
                            # Fall back to GBM simulation
                            tick_data = self._simulate_tick(symbol, current_price, data["volatility"])
                            new_price = tick_data["price"]
                            change_pct = tick_data["change_pct"]
                            volume = tick_data["volume"]
                            is_live = False

                        # Update state
                        self.watched_symbols[symbol]["price"] = new_price
                        self.watched_symbols[symbol]["prev_price"] = current_price
                        self.watched_symbols[symbol]["is_live"] = is_live

                        # Create tick payload
                        tick = {
                            "symbol": symbol,
                            "price": round(new_price, 2),
                            "change_pct": round(change_pct, 3),
                            "volume": volume,
                            "timestamp": datetime.now().isoformat(),
                            "is_live": is_live  # Let frontend know data source
                        }

                        # Broadcast
                        await self.manager.broadcast_ticker(symbol, tick)

                        # Check price alerts (with error handling to prevent stream crash)
                        try:
                            await self._check_price_alerts(symbol, new_price, prev_price)
                        except Exception as alert_err:
                            logger.warning(f"Alert check failed for {symbol}: {alert_err}")

                        # Smart Trigger: Run analysis if price moves > 0.5% (Flash Move)
                        # Only if analyzer is available
                        if self.analyzer:
                            shock = abs(change_pct / 100) if change_pct else 0
                            is_flash_move = shock > 0.005
                            is_random_check = random.random() < 0.005  # Reduced random checks

                            if is_flash_move or is_random_check:
                                logger.info(f"Triggering Live Analysis for {symbol} (Change: {change_pct:.4f}%, Live: {is_live})")

                                # Run analysis in background (don't block stream)
                                task = asyncio.create_task(self._run_live_analysis(symbol, new_price, shock))
                                self._background_tasks.add(task)
                                task.add_done_callback(self._background_tasks.discard)

                    # Sleep to mimic tick frequency (1 second updates)
                    await asyncio.sleep(1)

                except Exception as e:
                    logger.error(f"Stream tick error: {e}", exc_info=True)
                    await asyncio.sleep(5)  # Retry delay

        except Exception as e:
            logger.error(f"Fatal stream error: {e}", exc_info=True)
            self.active = False

    async def _run_live_analysis(self, symbol: str, price: float, shock: float):
        """Run quick analysis and broadcast if interesting"""
        try:
             # We skip deep fundamentals for speed in live mode
             result = await self.analyzer.analyze_stock(
                 symbol,
                 include_fundamentals=False,
                 include_patterns=True
             )

             recommendation = result.get('recommendation', 'HOLD')
             confidence = result.get('confidence', 0)

             # Broadcast only if it's a tradeable signal
             if recommendation != 'HOLD' and confidence > 60:
                 alert = {
                     "type": "opportunity",
                     "symbol": symbol,
                     "title": f"Live Opportunity: {recommendation}",
                     "message": f"{symbol} flashed {recommendation} ({confidence}% conf) at Rs.{price:.2f}",
                     "details": {
                         "price": price,
                         "shock": shock,
                         "recommendation": recommendation,
                         "confidence": confidence
                     },
                     "timestamp": datetime.now().isoformat()
                 }
                 await self.manager.broadcast_alert(alert)

        except Exception as e:
            logger.error(f"Live analysis failed for {symbol}: {e}")

    async def _check_price_alerts(self, symbol: str, current_price: float, previous_price: float):
        """Check if any price alerts should trigger for this symbol"""
        try:
            # Lazy load alert manager to avoid circular imports
            if self.alert_manager is None:
                from alerts.alert_manager import get_alert_manager
                self.alert_manager = get_alert_manager()

            # Get all active price alerts for this symbol
            from alerts.alert_manager import AlertStatus, AlertType
            active_alerts = [
                alert for alert in self.alert_manager.alerts.values()
                if alert.symbol == symbol
                and alert.status == AlertStatus.ACTIVE
                and alert.alert_type == AlertType.PRICE
            ]

            # Check each alert
            for alert in active_alerts:
                should_trigger = self.alert_manager.check_price_alert(
                    alert, current_price, previous_price
                )

                if should_trigger:
                    logger.info(f"Alert triggered: {alert.alert_id} for {symbol} at Rs.{current_price:.2f}")
                    await self.alert_manager.trigger_alert(alert)

        except Exception as e:
            logger.error(f"Error checking price alerts for {symbol}: {e}")

    async def stop_stream(self):
        """Stop the market data stream"""
        self.active = False
        logger.info("Market Stream Stopped")

    def get_stream_status(self) -> Dict:
        """Get current stream status"""
        live_symbols = [s for s, d in self.watched_symbols.items() if d.get("is_live")]
        return {
            "active": self.active,
            "total_symbols": len(self.watched_symbols),
            "live_symbols": len(live_symbols),
            "simulated_symbols": len(self.watched_symbols) - len(live_symbols),
            "live_data_enabled": self._live_data_enabled,
            "symbols": list(self.watched_symbols.keys())
        }


# Global instance
stream = MarketStream()

def get_market_stream() -> MarketStream:
    return stream
