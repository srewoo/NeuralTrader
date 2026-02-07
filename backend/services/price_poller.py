"""
Real-Time Price Polling Service
Polls stock prices every 30 seconds during Indian market hours.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Set
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

# IST offset
IST_OFFSET = timedelta(hours=5, minutes=30)


class PricePollerService:
    """
    Background service that polls stock prices at regular intervals.
    Only active during Indian market hours (Mon-Fri 9:15-15:30 IST).
    """

    def __init__(self, db=None, poll_interval: int = 30):
        self.db = db
        self.poll_interval = poll_interval
        self.active_symbols: Set[str] = set()
        self._price_cache: Dict[str, Dict[str, Any]] = {}
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        """Start the background polling loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(f"Price poller started (interval: {self.poll_interval}s)")

    async def stop(self):
        """Stop the background polling loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Price poller stopped")

    def add_symbol(self, symbol: str):
        """Add a symbol to the watch list."""
        self.active_symbols.add(symbol.upper().replace('.NS', '').replace('.BO', ''))

    def remove_symbol(self, symbol: str):
        """Remove a symbol from the watch list."""
        clean = symbol.upper().replace('.NS', '').replace('.BO', '')
        self.active_symbols.discard(clean)

    def get_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest cached price for a symbol."""
        clean = symbol.upper().replace('.NS', '').replace('.BO', '')
        return self._price_cache.get(clean)

    def _is_market_open(self) -> bool:
        """Check if Indian stock market is currently open."""
        now_utc = datetime.now(timezone.utc)
        now_ist = now_utc + IST_OFFSET

        # Weekday check (Mon=0 to Fri=4)
        if now_ist.weekday() > 4:
            return False

        # Market hours: 9:15 to 15:30 IST
        market_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)

        return market_open <= now_ist <= market_close

    async def _poll_loop(self):
        """Main polling loop."""
        from data_providers.provider_manager import get_provider_manager

        while self._running:
            try:
                if not self._is_market_open():
                    # Market closed — sleep longer
                    await asyncio.sleep(60)
                    continue

                if not self.active_symbols:
                    await asyncio.sleep(self.poll_interval)
                    continue

                provider = get_provider_manager()

                for symbol in list(self.active_symbols):
                    try:
                        quote = await provider.get_quote(symbol)
                        if quote:
                            price_data = {
                                "symbol": symbol,
                                "price": quote.current_price if hasattr(quote, 'current_price') else quote.get('current_price'),
                                "change": quote.change if hasattr(quote, 'change') else quote.get('change'),
                                "change_percent": quote.change_percent if hasattr(quote, 'change_percent') else quote.get('change_percent'),
                                "volume": quote.volume if hasattr(quote, 'volume') else quote.get('volume'),
                                "timestamp": datetime.now(timezone.utc).isoformat()
                            }
                            self._price_cache[symbol] = price_data

                            # Persist to MongoDB
                            if self.db:
                                await self.db.live_prices.update_one(
                                    {"symbol": symbol},
                                    {"$set": price_data},
                                    upsert=True
                                )
                    except Exception as e:
                        logger.debug(f"Failed to poll {symbol}: {e}")

                await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Poll loop error: {e}")
                await asyncio.sleep(self.poll_interval)


# Singleton
_poller_instance: Optional[PricePollerService] = None


def get_price_poller(db=None, poll_interval: int = 30) -> PricePollerService:
    """Get or create singleton price poller."""
    global _poller_instance
    if _poller_instance is None:
        _poller_instance = PricePollerService(db=db, poll_interval=poll_interval)
    return _poller_instance
