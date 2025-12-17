"""
Pattern Mining from Historical Data
Automatically discovers recurring patterns and validates their success rates
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from motor.motor_asyncio import AsyncIOMotorDatabase
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PatternMatch:
    """Represents a single instance where a pattern was detected"""
    symbol: str
    detection_date: datetime
    entry_price: float
    exit_price: Optional[float]
    exit_date: Optional[datetime]
    return_pct: Optional[float]
    success: bool
    conditions_met: Dict[str, Any]


class PatternMiner:
    """
    Mines historical data to discover profitable trading patterns
    Auto-validates patterns and ingests successful ones into RAG system
    """

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.patterns_collection = db.discovered_patterns
        self.min_occurrences = 10  # Minimum pattern occurrences to consider valid
        self.min_success_rate = 0.60  # 60% success rate threshold

    async def mine_rsi_oversold_patterns(
        self,
        symbol: str,
        historical_data: pd.DataFrame,
        rsi_threshold: float = 30,
        holding_period_days: int = 5,
        volume_spike_threshold: float = 1.5
    ) -> Optional[Dict[str, Any]]:
        """
        Mine RSI oversold + volume spike patterns

        Args:
            symbol: Stock symbol
            historical_data: DataFrame with OHLCV + indicators
            rsi_threshold: RSI below this value triggers pattern
            holding_period_days: Days to hold after signal
            volume_spike_threshold: Volume ratio to consider a spike

        Returns:
            Pattern summary with success rate
        """
        try:
            import ta

            # Calculate RSI if not present
            if 'RSI' not in historical_data.columns:
                historical_data['RSI'] = ta.momentum.RSIIndicator(
                    historical_data['Close'], window=14
                ).rsi()

            # Calculate volume ratio
            historical_data['Volume_MA20'] = historical_data['Volume'].rolling(20).mean()
            historical_data['Volume_Ratio'] = historical_data['Volume'] / historical_data['Volume_MA20']

            # Detect pattern occurrences
            pattern_matches = []

            for i in range(50, len(historical_data) - holding_period_days - 1):
                current_rsi = historical_data['RSI'].iloc[i]
                volume_ratio = historical_data['Volume_Ratio'].iloc[i]

                # Check if pattern conditions are met
                if current_rsi < rsi_threshold and volume_ratio > volume_spike_threshold:
                    entry_price = historical_data['Close'].iloc[i]
                    entry_date = historical_data.index[i]

                    # Check exit after holding period
                    exit_idx = i + holding_period_days
                    exit_price = historical_data['Close'].iloc[exit_idx]
                    exit_date = historical_data.index[exit_idx]

                    return_pct = ((exit_price - entry_price) / entry_price) * 100
                    success = return_pct > 0  # Profit = success

                    match = PatternMatch(
                        symbol=symbol,
                        detection_date=entry_date,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        exit_date=exit_date,
                        return_pct=return_pct,
                        success=success,
                        conditions_met={
                            'rsi': round(current_rsi, 2),
                            'volume_ratio': round(volume_ratio, 2),
                            'entry_price': round(entry_price, 2)
                        }
                    )
                    pattern_matches.append(match)

            if len(pattern_matches) < self.min_occurrences:
                logger.info(
                    f"Pattern RSI<{rsi_threshold} + Volume>{volume_spike_threshold}x "
                    f"has insufficient occurrences ({len(pattern_matches)}) for {symbol}"
                )
                return None

            # Calculate statistics
            success_count = sum(1 for m in pattern_matches if m.success)
            success_rate = success_count / len(pattern_matches)
            avg_return = np.mean([m.return_pct for m in pattern_matches])
            profitable_returns = [m.return_pct for m in pattern_matches if m.success]
            avg_profitable_return = np.mean(profitable_returns) if profitable_returns else 0

            if success_rate < self.min_success_rate:
                logger.info(
                    f"Pattern success rate {success_rate:.1%} below threshold "
                    f"{self.min_success_rate:.1%} for {symbol}"
                )
                return None

            # This is a valid pattern! Prepare to store it
            pattern_summary = {
                "pattern_type": "rsi_oversold_volume_spike",
                "symbol": symbol,
                "description": (
                    f"When {symbol} RSI drops below {rsi_threshold} AND "
                    f"volume spikes {volume_spike_threshold}x average, "
                    f"price recovered {avg_profitable_return:+.1f}% within "
                    f"{holding_period_days} days in {success_rate:.0%} of cases (N={len(pattern_matches)})"
                ),
                "conditions": {
                    "rsi_threshold": rsi_threshold,
                    "volume_spike_threshold": volume_spike_threshold,
                    "holding_period_days": holding_period_days
                },
                "statistics": {
                    "total_occurrences": len(pattern_matches),
                    "success_count": success_count,
                    "success_rate": round(success_rate, 4),
                    "average_return": round(avg_return, 2),
                    "average_profitable_return": round(avg_profitable_return, 2),
                    "best_return": round(max(m.return_pct for m in pattern_matches), 2),
                    "worst_return": round(min(m.return_pct for m in pattern_matches), 2)
                },
                "discovered_at": datetime.now(timezone.utc),
                "data_period": {
                    "start": str(historical_data.index[0]),
                    "end": str(historical_data.index[-1])
                },
                "matches": [
                    {
                        "date": str(m.detection_date),
                        "entry_price": m.entry_price,
                        "exit_price": m.exit_price,
                        "return_pct": round(m.return_pct, 2),
                        "success": m.success,
                        "conditions": m.conditions_met
                    }
                    for m in pattern_matches[-10:]  # Store last 10 matches
                ]
            }

            logger.info(
                f"✅ Discovered valid pattern for {symbol}: "
                f"{success_rate:.0%} success rate over {len(pattern_matches)} occurrences"
            )

            return pattern_summary

        except Exception as e:
            logger.error(f"Error mining RSI pattern for {symbol}: {e}")
            return None

    async def mine_breakout_patterns(
        self,
        symbol: str,
        historical_data: pd.DataFrame,
        lookback_period: int = 20,
        breakout_threshold: float = 0.03,  # 3% above resistance
        holding_period_days: int = 10
    ) -> Optional[Dict[str, Any]]:
        """
        Mine breakout patterns (price breaking above recent highs)

        Args:
            symbol: Stock symbol
            historical_data: DataFrame with OHLCV
            lookback_period: Days to determine resistance level
            breakout_threshold: % above resistance to confirm breakout
            holding_period_days: Days to hold after breakout

        Returns:
            Pattern summary with success rate
        """
        try:
            # Calculate rolling high (resistance)
            historical_data['Resistance'] = historical_data['High'].rolling(lookback_period).max()

            pattern_matches = []

            for i in range(lookback_period + 1, len(historical_data) - holding_period_days - 1):
                current_close = historical_data['Close'].iloc[i]
                resistance = historical_data['Resistance'].iloc[i-1]  # Previous resistance

                # Check breakout condition
                breakout_pct = ((current_close - resistance) / resistance)

                if breakout_pct > breakout_threshold:
                    entry_price = current_close
                    entry_date = historical_data.index[i]

                    # Check exit after holding period
                    exit_idx = i + holding_period_days
                    exit_price = historical_data['Close'].iloc[exit_idx]
                    exit_date = historical_data.index[exit_idx]

                    return_pct = ((exit_price - entry_price) / entry_price) * 100
                    success = return_pct > 2.0  # Need at least 2% profit

                    match = PatternMatch(
                        symbol=symbol,
                        detection_date=entry_date,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        exit_date=exit_date,
                        return_pct=return_pct,
                        success=success,
                        conditions_met={
                            'breakout_pct': round(breakout_pct * 100, 2),
                            'resistance': round(resistance, 2),
                            'entry_price': round(entry_price, 2)
                        }
                    )
                    pattern_matches.append(match)

            if len(pattern_matches) < self.min_occurrences:
                return None

            success_count = sum(1 for m in pattern_matches if m.success)
            success_rate = success_count / len(pattern_matches)

            if success_rate < self.min_success_rate:
                return None

            avg_return = np.mean([m.return_pct for m in pattern_matches if m.success])

            pattern_summary = {
                "pattern_type": "breakout_above_resistance",
                "symbol": symbol,
                "description": (
                    f"When {symbol} breaks {breakout_threshold*100:.0f}% above {lookback_period}-day resistance, "
                    f"price gained {avg_return:+.1f}% within {holding_period_days} days "
                    f"in {success_rate:.0%} of cases (N={len(pattern_matches)})"
                ),
                "conditions": {
                    "lookback_period": lookback_period,
                    "breakout_threshold": breakout_threshold,
                    "holding_period_days": holding_period_days
                },
                "statistics": {
                    "total_occurrences": len(pattern_matches),
                    "success_count": success_count,
                    "success_rate": round(success_rate, 4),
                    "average_return": round(avg_return, 2)
                },
                "discovered_at": datetime.now(timezone.utc)
            }

            logger.info(
                f"✅ Discovered breakout pattern for {symbol}: "
                f"{success_rate:.0%} success rate"
            )

            return pattern_summary

        except Exception as e:
            logger.error(f"Error mining breakout pattern for {symbol}: {e}")
            return None

    async def store_discovered_pattern(self, pattern: Dict[str, Any]) -> str:
        """
        Store a discovered pattern in the database

        Args:
            pattern: Pattern summary dictionary

        Returns:
            Pattern ID
        """
        # Check if similar pattern already exists
        existing = await self.patterns_collection.find_one({
            "pattern_type": pattern["pattern_type"],
            "symbol": pattern["symbol"]
        })

        if existing:
            # Update existing pattern with new data
            await self.patterns_collection.update_one(
                {"_id": existing["_id"]},
                {
                    "$set": {
                        "statistics": pattern["statistics"],
                        "last_updated": datetime.now(timezone.utc)
                    },
                    "$inc": {"update_count": 1}
                }
            )
            pattern_id = str(existing["_id"])
            logger.info(f"Updated existing pattern: {pattern_id}")
        else:
            # Insert new pattern
            pattern["update_count"] = 1
            result = await self.patterns_collection.insert_one(pattern)
            pattern_id = str(result.inserted_id)
            logger.info(f"Stored new pattern: {pattern_id}")

        return pattern_id

    async def get_patterns_for_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Get all discovered patterns for a symbol"""
        cursor = self.patterns_collection.find({"symbol": symbol})
        patterns = await cursor.to_list(length=100)

        for pattern in patterns:
            pattern['id'] = str(pattern.pop('_id'))

        return patterns

    async def get_all_valid_patterns(self, min_success_rate: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get all patterns above a certain success rate"""
        query = {}
        if min_success_rate:
            query["statistics.success_rate"] = {"$gte": min_success_rate}

        cursor = self.patterns_collection.find(query).sort("statistics.success_rate", -1)
        patterns = await cursor.to_list(length=500)

        for pattern in patterns:
            pattern['id'] = str(pattern.pop('_id'))

        return patterns

    async def create_indexes(self):
        """Create database indexes"""
        await self.patterns_collection.create_index([("symbol", 1)])
        await self.patterns_collection.create_index([("pattern_type", 1)])
        await self.patterns_collection.create_index([("statistics.success_rate", -1)])
        await self.patterns_collection.create_index([("symbol", 1), ("pattern_type", 1)])
        logger.info("Created indexes for discovered_patterns collection")


# Singleton instance
_pattern_miner: Optional[PatternMiner] = None


def get_pattern_miner(db: AsyncIOMotorDatabase) -> PatternMiner:
    """Get or create singleton instance"""
    global _pattern_miner
    if _pattern_miner is None:
        _pattern_miner = PatternMiner(db)
    return _pattern_miner
