"""
Market Data Simulator
Generates synthetic market data to mimic a live feed.
"""

import asyncio
import random
import logging
import math
from typing import Dict, List, Optional
from datetime import datetime
from .connection_manager import get_connection_manager
from analysis.enhanced_analyzer import get_enhanced_analyzer

logger = logging.getLogger(__name__)

class MarketStream:
    """
    Simulates a live market feed using Geometric Brownian Motion.
    """
    def __init__(self):
        self.active = False
        self.manager = get_connection_manager()
        self.analyzer = None  # Lazy load to avoid circular import issues at startup
        self.alert_manager = None  # Lazy load alert manager
        self.watched_symbols = {}  # Dynamic - populated from user subscriptions or watchlist
        self.default_volatility = 0.002  # Default volatility for new symbols
        self._background_tasks = set()  # Track background tasks to prevent garbage collection
        
    async def add_symbol(self, symbol: str, initial_price: float = None):
        """Add a symbol to watch dynamically"""
        if symbol not in self.watched_symbols:
            if initial_price is None:
                # Fetch real price from data provider
                try:
                    from data_providers.factory import get_data_provider_factory
                    factory = get_data_provider_factory()
                    quote = await factory.get_quote(symbol)
                    initial_price = quote.get('price', 100.0)
                except Exception:
                    initial_price = 100.0  # Fallback
            
            self.watched_symbols[symbol] = {
                "price": initial_price,
                "volatility": self.default_volatility,
                "prev_price": initial_price
            }
            logger.info(f"Added {symbol} to market stream")
    
    async def start_stream(self):
        """Start the simulation loop"""
        if self.active:
            return
            
        self.active = True
        logger.info("Market Stream Started")
        
        try:
            # Initialize analyzer
            if not self.analyzer:
                self.analyzer = get_enhanced_analyzer()
            
            while self.active:
                try:
                    # Skip if no symbols to watch
                    if not self.watched_symbols:
                        await asyncio.sleep(5)
                        continue
                    
                    for symbol, data in self.watched_symbols.items():
                        # Generate next price tick
                        volatility = data["volatility"]
                        shock = random.gauss(0, volatility)
                        new_price = data["price"] * (1 + shock)
                        
                        # Get previous price for alert checking
                        prev_price = data.get("prev_price", new_price)

                        # Update state
                        self.watched_symbols[symbol]["price"] = new_price
                        self.watched_symbols[symbol]["prev_price"] = new_price

                        # Create tick payload
                        tick = {
                            "symbol": symbol,
                            "price": round(new_price, 2),
                            "change_pct": round(shock * 100, 3),
                            "volume": random.randint(100, 5000),
                            "timestamp": datetime.now().isoformat()
                        }

                        # Broadcast
                        await self.manager.broadcast_ticker(symbol, tick)

                        # Check price alerts (with error handling to prevent stream crash)
                        try:
                            await self._check_price_alerts(symbol, new_price, prev_price)
                        except Exception as alert_err:
                            logger.warning(f"Alert check failed for {symbol}: {alert_err}")
                        
                        # Smart Trigger: Run analysis if price moves > 0.5% in one tick (Flash Move)
                        # OR randomly for demo purposes (1% chance per tick)
                        is_flash_move = abs(shock) > 0.005
                        is_random_check = random.random() < 0.01
                        
                        if is_flash_move or is_random_check:
                            logger.info(f"⚡ Triggering Live Analysis for {symbol} (Shock: {shock:.4f})")

                            # Run analysis in background (don't block stream)
                            task = asyncio.create_task(self._run_live_analysis(symbol, new_price, shock))
                            self._background_tasks.add(task)
                            task.add_done_callback(self._background_tasks.discard)
                    
                    # Sleep to mimic tick frequency (e.g., 1 second updates)
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Stream tick error: {e}", exc_info=True)
                    await asyncio.sleep(5) # Retry delay
                    
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
                     "message": f"{symbol} flashed {recommendation} ({confidence}% conf) at ₹{price:.2f}",
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
                    logger.info(f"🔔 Alert triggered: {alert.alert_id} for {symbol} at ₹{current_price:.2f}")
                    await self.alert_manager.trigger_alert(alert)

        except Exception as e:
            logger.error(f"Error checking price alerts for {symbol}: {e}")

    async def stop_stream(self):
        """Stop the simulation symbol"""
        self.active = False
        logger.info("Market Stream Stopped")

# Global instance
stream = MarketStream()

def get_market_stream() -> MarketStream:
    return stream

