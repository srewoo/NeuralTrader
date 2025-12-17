"""
Auto-Discovery Pipeline for Pattern Mining
Automatically discovers and validates trading patterns from historical data
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorDatabase
import pandas as pd
import yfinance as yf
import asyncio
import logging

from .pattern_mining import get_pattern_miner
from data_providers.provider_manager import get_provider_manager

logger = logging.getLogger(__name__)


class AutoDiscoveryPipeline:
    """
    Automatically discovers patterns from historical market data
    Runs periodically to keep pattern library updated
    """

    # NIFTY 50 top stocks for pattern discovery
    NIFTY_50_SYMBOLS = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "BAJFINANCE",
        "KOTAKBANK", "LT", "HCLTECH", "ASIANPAINT", "AXISBANK",
        "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND",
        "WIPRO", "ADANIENT", "BAJAJFINSV", "ONGC", "NTPC",
        "POWERGRID", "TATAMOTORS", "M&M", "TECHM", "TATASTEEL"
    ]

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.pattern_miner = get_pattern_miner(db)
        self.results_collection = db.discovery_runs

    async def run_discovery(
        self,
        symbols: Optional[List[str]] = None,
        lookback_years: int = 5,
        api_keys: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run pattern discovery across multiple stocks

        Args:
            symbols: List of symbols to analyze (defaults to NIFTY 50 subset)
            lookback_years: Years of historical data to analyze
            api_keys: Optional API keys for data providers

        Returns:
            Discovery run summary
        """
        start_time = datetime.now(timezone.utc)
        symbols = symbols or self.NIFTY_50_SYMBOLS[:10]  # Process 10 stocks per run

        logger.info(f"🔍 Starting pattern discovery for {len(symbols)} symbols")

        discovered_patterns = []
        failed_symbols = []

        provider_manager = get_provider_manager(api_keys)

        for symbol in symbols:
            try:
                logger.info(f"Mining patterns for {symbol}...")

                # Fetch historical data
                hist = await provider_manager.get_historical_data(
                    symbol,
                    period=f"{lookback_years}y",
                    interval="1d"
                )

                if hist is None or len(hist) < 200:
                    logger.warning(f"Insufficient data for {symbol}")
                    failed_symbols.append(symbol)
                    continue

                # Mine RSI oversold patterns
                rsi_pattern = await self.pattern_miner.mine_rsi_oversold_patterns(
                    symbol=symbol,
                    historical_data=hist,
                    rsi_threshold=30,
                    holding_period_days=5,
                    volume_spike_threshold=1.5
                )

                if rsi_pattern:
                    pattern_id = await self.pattern_miner.store_discovered_pattern(rsi_pattern)
                    discovered_patterns.append({
                        "pattern_id": pattern_id,
                        "symbol": symbol,
                        "type": "rsi_oversold",
                        "success_rate": rsi_pattern["statistics"]["success_rate"]
                    })

                # Mine breakout patterns
                breakout_pattern = await self.pattern_miner.mine_breakout_patterns(
                    symbol=symbol,
                    historical_data=hist,
                    lookback_period=20,
                    breakout_threshold=0.03,
                    holding_period_days=10
                )

                if breakout_pattern:
                    pattern_id = await self.pattern_miner.store_discovered_pattern(breakout_pattern)
                    discovered_patterns.append({
                        "pattern_id": pattern_id,
                        "symbol": symbol,
                        "type": "breakout",
                        "success_rate": breakout_pattern["statistics"]["success_rate"]
                    })

                # Add small delay to avoid rate limits
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                failed_symbols.append(symbol)

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        # Store discovery run results
        run_summary = {
            "run_id": str(datetime.now(timezone.utc).timestamp()),
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration,
            "symbols_processed": len(symbols) - len(failed_symbols),
            "symbols_failed": len(failed_symbols),
            "failed_symbols": failed_symbols,
            "patterns_discovered": len(discovered_patterns),
            "patterns": discovered_patterns,
            "lookback_years": lookback_years
        }

        await self.results_collection.insert_one(run_summary)

        logger.info(
            f"✅ Discovery complete: {len(discovered_patterns)} patterns found "
            f"from {len(symbols) - len(failed_symbols)} symbols in {duration:.1f}s"
        )

        return run_summary

    async def get_discovery_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get history of discovery runs"""
        cursor = self.results_collection.find().sort("start_time", -1).limit(limit)
        runs = await cursor.to_list(length=limit)

        for run in runs:
            run['id'] = str(run.pop('_id'))

        return runs

    async def schedule_periodic_discovery(self, interval_hours: int = 24):
        """
        Schedule periodic pattern discovery
        This can be run as a background task

        Args:
            interval_hours: Hours between discovery runs
        """
        logger.info(f"Scheduled pattern discovery every {interval_hours} hours")

        while True:
            try:
                # Run discovery
                await self.run_discovery()

                # Wait for next run
                await asyncio.sleep(interval_hours * 3600)

            except Exception as e:
                logger.error(f"Error in scheduled discovery: {e}")
                # Wait 1 hour before retrying on error
                await asyncio.sleep(3600)


class PatternValidator:
    """
    Validates discovered patterns against live market data
    Ensures patterns remain valid over time
    """

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.pattern_miner = get_pattern_miner(db)

    async def validate_pattern(
        self,
        pattern_id: str,
        recent_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Validate if a pattern still works on recent data

        Args:
            pattern_id: ID of pattern to validate
            recent_data: Recent historical data to test against

        Returns:
            Validation results
        """
        # Get pattern from database
        pattern = await self.db.discovered_patterns.find_one({"_id": pattern_id})

        if not pattern:
            return {"error": "Pattern not found"}

        # Re-mine the pattern on recent data
        if pattern["pattern_type"] == "rsi_oversold_volume_spike":
            conditions = pattern["conditions"]
            recent_pattern = await self.pattern_miner.mine_rsi_oversold_patterns(
                symbol=pattern["symbol"],
                historical_data=recent_data,
                rsi_threshold=conditions["rsi_threshold"],
                holding_period_days=conditions["holding_period_days"],
                volume_spike_threshold=conditions["volume_spike_threshold"]
            )

        elif pattern["pattern_type"] == "breakout_above_resistance":
            conditions = pattern["conditions"]
            recent_pattern = await self.pattern_miner.mine_breakout_patterns(
                symbol=pattern["symbol"],
                historical_data=recent_data,
                lookback_period=conditions["lookback_period"],
                breakout_threshold=conditions["breakout_threshold"],
                holding_period_days=conditions["holding_period_days"]
            )

        else:
            return {"error": "Unknown pattern type"}

        if not recent_pattern:
            return {
                "valid": False,
                "reason": "Pattern no longer meets minimum criteria"
            }

        # Compare success rates
        original_success_rate = pattern["statistics"]["success_rate"]
        recent_success_rate = recent_pattern["statistics"]["success_rate"]
        degradation = original_success_rate - recent_success_rate

        validation = {
            "valid": recent_success_rate >= 0.55,  # Still above 55%
            "original_success_rate": original_success_rate,
            "recent_success_rate": recent_success_rate,
            "degradation": degradation,
            "recommendation": (
                "Pattern still valid" if degradation < 0.1
                else "Pattern degrading, consider updating" if degradation < 0.2
                else "Pattern invalid, should be archived"
            )
        }

        # Update pattern in database with validation results
        await self.db.discovered_patterns.update_one(
            {"_id": pattern_id},
            {
                "$set": {
                    "last_validated": datetime.now(timezone.utc),
                    "validation_results": validation
                }
            }
        )

        return validation


# Singleton instance
_discovery_pipeline: Optional[AutoDiscoveryPipeline] = None


def get_discovery_pipeline(db: AsyncIOMotorDatabase) -> AutoDiscoveryPipeline:
    """Get or create singleton instance"""
    global _discovery_pipeline
    if _discovery_pipeline is None:
        _discovery_pipeline = AutoDiscoveryPipeline(db)
    return _discovery_pipeline
