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
        self.watched_symbols = {
            "RELIANCE": {"price": 2500.0, "volatility": 0.002},
            "TCS": {"price": 3500.0, "volatility": 0.0015},
            "INFY": {"price": 1500.0, "volatility": 0.0025},
            "HDFCBANK": {"price": 1600.0, "volatility": 0.0018},
            "ICICIBANK": {"price": 950.0, "volatility": 0.002},
            "NTPC": {"price": 322.55, "volatility": 0.0020},
            "POWERGRID": {"price": 268.05, "volatility": 0.0018},
            "SBIN": {"price": 780.0, "volatility": 0.0022},
            "TATAMOTORS": {"price": 750.0, "volatility": 0.0025},
            "WIPRO": {"price": 450.0, "volatility": 0.0019}
        }
        
    async def start_stream(self):
        """Start the simulation loop"""
        if self.active:
            return
            
        self.active = True
        logger.info("Market Stream Started")
        
        # Initialize analyzer
        if not self.analyzer:
             self.analyzer = get_enhanced_analyzer()
        
        while self.active:
            try:
                for symbol, data in self.watched_symbols.items():
                    # Generate next price tick
                    # Geometric Brownian Motion step: P_t = P_{t-1} * e^((mu - 0.5*sigma^2)*dt + sigma*dW)
                    # Simplified: P_new = P_old * (1 + random_shock)
                    
                    volatility = data["volatility"]
                    shock = random.gauss(0, volatility)
                    new_price = data["price"] * (1 + shock)
                    
                    # Update state
                    self.watched_symbols[symbol]["price"] = new_price
                    
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
                    
                    # Smart Trigger: Run analysis if price moves > 0.5% in one tick (Flash Move)
                    # OR randomly for demo purposes (1% chance per tick)
                    is_flash_move = abs(shock) > 0.005
                    is_random_check = random.random() < 0.01
                    
                    if is_flash_move or is_random_check:
                        logger.info(f"⚡ Triggering Live Analysis for {symbol} (Shock: {shock:.4f})")
                        
                        # Run analysis in background (don't block stream)
                        asyncio.create_task(self._run_live_analysis(symbol, new_price, shock))
                
                # Sleep to mimic tick frequency (e.g., 1 second updates)
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Stream Error: {e}")
                await asyncio.sleep(5) # Retry delay

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

    async def stop_stream(self):
        """Stop the simulation symbol"""
        self.active = False
        logger.info("Market Stream Stopped")

# Global instance
stream = MarketStream()

def get_market_stream() -> MarketStream:
    return stream
