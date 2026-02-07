"""
Track Record Service
Records every AI recommendation and tracks actual outcomes at 1/5/20 days.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


class TrackRecordService:
    """
    Tracks AI recommendation accuracy by recording predictions
    and checking actual price outcomes at 1-day, 5-day, and 20-day intervals.
    """

    def __init__(self, db):
        self.db = db

    async def record_recommendation(self, analysis_result: Dict[str, Any]):
        """Record a new recommendation for tracking."""
        try:
            symbol = analysis_result.get("symbol")
            recommendation = analysis_result.get("recommendation")
            if not symbol or not recommendation or recommendation == "HOLD":
                return  # Only track BUY/SELL

            record = {
                "symbol": symbol,
                "recommendation": recommendation,
                "confidence": analysis_result.get("confidence", 0),
                "entry_price": analysis_result.get("entry_price"),
                "target_price": analysis_result.get("target_price"),
                "stop_loss": analysis_result.get("stop_loss"),
                "model_used": analysis_result.get("model_used", ""),
                "quality_score": analysis_result.get("quality_score", 0),
                "created_at": datetime.now(timezone.utc),
                "outcomes": {
                    "1d": None,
                    "5d": None,
                    "20d": None
                },
                "status": "pending"
            }

            await self.db.recommendations_track.insert_one(record)
            logger.info(f"Recorded {recommendation} recommendation for {symbol}")

        except Exception as e:
            logger.warning(f"Failed to record recommendation: {e}")

    async def check_outcomes(self):
        """Check outcomes for pending recommendations."""
        try:
            from data_providers.provider_manager import get_provider_manager

            pending = await self.db.recommendations_track.find(
                {"status": "pending"}
            ).to_list(length=100)

            if not pending:
                return

            provider = get_provider_manager()
            now = datetime.now(timezone.utc)

            for record in pending:
                symbol = record["symbol"]
                created_at = record["created_at"]
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))

                entry_price = record.get("entry_price")
                if not entry_price:
                    continue

                target_price = record.get("target_price")
                stop_loss = record.get("stop_loss")
                recommendation = record["recommendation"]
                outcomes = record.get("outcomes", {})

                # Check each time horizon
                intervals = {"1d": 1, "5d": 5, "20d": 20}
                updated = False

                for key, days in intervals.items():
                    if outcomes.get(key) is not None:
                        continue  # Already filled

                    elapsed = (now - created_at).days
                    if elapsed < days:
                        continue  # Not enough time has passed

                    try:
                        quote = await provider.get_quote(symbol)
                        if quote:
                            current_price = quote.current_price if hasattr(quote, 'current_price') else quote.get('current_price')
                            if current_price and entry_price:
                                if recommendation == "BUY":
                                    return_pct = ((current_price - entry_price) / entry_price) * 100
                                else:
                                    return_pct = ((entry_price - current_price) / entry_price) * 100

                                hit_target = False
                                hit_stop = False
                                if target_price:
                                    if recommendation == "BUY":
                                        hit_target = current_price >= target_price
                                    else:
                                        hit_target = current_price <= target_price
                                if stop_loss:
                                    if recommendation == "BUY":
                                        hit_stop = current_price <= stop_loss
                                    else:
                                        hit_stop = current_price >= stop_loss

                                outcomes[key] = {
                                    "price": round(current_price, 2),
                                    "return_pct": round(return_pct, 2),
                                    "profitable": return_pct > 0,
                                    "hit_target": hit_target,
                                    "hit_stop": hit_stop,
                                    "checked_at": now.isoformat()
                                }
                                updated = True
                    except Exception as e:
                        logger.debug(f"Failed to check outcome for {symbol} at {key}: {e}")

                if updated:
                    # Check if all outcomes are filled
                    all_filled = all(outcomes.get(k) is not None for k in intervals)
                    update_data = {"outcomes": outcomes}
                    if all_filled:
                        update_data["status"] = "completed"

                    await self.db.recommendations_track.update_one(
                        {"_id": record["_id"]},
                        {"$set": update_data}
                    )

        except Exception as e:
            logger.warning(f"Outcome check failed: {e}")

    async def get_stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get accuracy statistics."""
        try:
            query = {}
            if symbol:
                query["symbol"] = symbol.upper()

            records = await self.db.recommendations_track.find(query).to_list(length=1000)

            if not records:
                return {
                    "total_recommendations": 0,
                    "completed": 0,
                    "buy_accuracy_1d": 0.0,
                    "buy_accuracy_5d": 0.0,
                    "buy_accuracy_20d": 0.0,
                    "sell_accuracy_1d": 0.0,
                    "sell_accuracy_5d": 0.0,
                    "sell_accuracy_20d": 0.0,
                    "avg_return_1d": 0.0,
                    "avg_return_5d": 0.0,
                    "avg_return_20d": 0.0,
                    "target_hit_rate": 0.0,
                    "stop_loss_hit_rate": 0.0
                }

            completed = [r for r in records if r.get("status") == "completed"]

            def calc_accuracy(recs, interval, direction):
                filtered = [r for r in recs if r["recommendation"] == direction
                           and r.get("outcomes", {}).get(interval) is not None]
                if not filtered:
                    return 0.0
                profitable = sum(1 for r in filtered if r["outcomes"][interval].get("profitable", False))
                return round((profitable / len(filtered)) * 100, 1)

            def calc_avg_return(recs, interval):
                returns = []
                for r in recs:
                    outcome = r.get("outcomes", {}).get(interval)
                    if outcome and outcome.get("return_pct") is not None:
                        returns.append(outcome["return_pct"])
                return round(sum(returns) / len(returns), 2) if returns else 0.0

            # Target and stop hit rates
            all_outcomes = []
            for r in records:
                for interval in ["1d", "5d", "20d"]:
                    outcome = r.get("outcomes", {}).get(interval)
                    if outcome:
                        all_outcomes.append(outcome)

            target_hits = sum(1 for o in all_outcomes if o.get("hit_target"))
            stop_hits = sum(1 for o in all_outcomes if o.get("hit_stop"))
            total_outcomes = len(all_outcomes) or 1

            return {
                "total_recommendations": len(records),
                "completed": len(completed),
                "pending": len(records) - len(completed),
                "buy_accuracy_1d": calc_accuracy(records, "1d", "BUY"),
                "buy_accuracy_5d": calc_accuracy(records, "5d", "BUY"),
                "buy_accuracy_20d": calc_accuracy(records, "20d", "BUY"),
                "sell_accuracy_1d": calc_accuracy(records, "1d", "SELL"),
                "sell_accuracy_5d": calc_accuracy(records, "5d", "SELL"),
                "sell_accuracy_20d": calc_accuracy(records, "20d", "SELL"),
                "avg_return_1d": calc_avg_return(records, "1d"),
                "avg_return_5d": calc_avg_return(records, "5d"),
                "avg_return_20d": calc_avg_return(records, "20d"),
                "target_hit_rate": round((target_hits / total_outcomes) * 100, 1),
                "stop_loss_hit_rate": round((stop_hits / total_outcomes) * 100, 1)
            }

        except Exception as e:
            logger.warning(f"Stats calculation failed: {e}")
            return {"error": str(e)}

    async def get_recent_track(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent recommendations with outcomes."""
        try:
            records = await self.db.recommendations_track.find(
                {}, {"_id": 0}
            ).sort("created_at", -1).limit(limit).to_list(length=limit)

            # Serialize datetime objects
            for r in records:
                if isinstance(r.get("created_at"), datetime):
                    r["created_at"] = r["created_at"].isoformat()

            return records

        except Exception as e:
            logger.warning(f"Failed to get recent track records: {e}")
            return []

    async def _outcome_check_loop(self):
        """Background loop that checks outcomes every hour."""
        while True:
            try:
                await self.check_outcomes()
            except Exception as e:
                logger.warning(f"Outcome check loop error: {e}")
            await asyncio.sleep(3600)  # Every hour


# Singleton
_track_service: Optional[TrackRecordService] = None


def get_track_record_service(db=None) -> TrackRecordService:
    """Get or create singleton track record service."""
    global _track_service
    if _track_service is None:
        if db is None:
            raise ValueError("db is required for first initialization")
        _track_service = TrackRecordService(db=db)
    return _track_service
