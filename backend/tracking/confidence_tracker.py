"""
Confidence Tracking System
Stores predictions and compares them with actual outcomes to measure
the accuracy of the AI recommendations over time.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from motor.motor_asyncio import AsyncIOMotorClient
import yfinance as yf
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ConfidenceTracker:
    """
    Track predictions and measure accuracy over time.

    Features:
    - Store every prediction with timestamp
    - Check actual price after 7/14/30 days
    - Calculate hit rates for different confidence levels
    - Track performance by recommendation type (BUY/SELL/HOLD)
    """

    def __init__(self, db):
        """
        Initialize tracker with MongoDB database.

        Args:
            db: Motor async MongoDB database instance
        """
        self.db = db
        self.collection = db.prediction_tracking

    async def record_prediction(
        self,
        symbol: str,
        recommendation: str,
        confidence: float,
        entry_price: float,
        target_price: Optional[float],
        stop_loss: Optional[float],
        reasoning: str,
        model_used: str,
        additional_data: Optional[Dict] = None
    ) -> str:
        """
        Record a new prediction for tracking.

        Args:
            symbol: Stock symbol
            recommendation: BUY/SELL/HOLD
            confidence: Confidence score (0-100)
            entry_price: Price at time of recommendation
            target_price: Target price (if applicable)
            stop_loss: Stop loss price (if applicable)
            reasoning: AI reasoning
            model_used: Model that made the prediction
            additional_data: Any additional context

        Returns:
            Prediction ID
        """
        prediction = {
            "symbol": symbol.upper(),
            "recommendation": recommendation.upper(),
            "confidence": confidence,
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_loss": stop_loss,
            "reasoning": reasoning[:500] if reasoning else None,  # Truncate for storage
            "model_used": model_used,
            "additional_data": additional_data,
            "created_at": datetime.utcnow(),
            "status": "pending",  # pending, verified, expired
            "verification": {
                "7_day": None,
                "14_day": None,
                "30_day": None
            }
        }

        result = await self.collection.insert_one(prediction)
        logger.info(f"Recorded prediction for {symbol}: {recommendation} @ {entry_price}")

        return str(result.inserted_id)

    async def verify_predictions(self) -> Dict[str, Any]:
        """
        Verify pending predictions by checking current prices.
        Should be run periodically (e.g., daily cron job).
        """
        now = datetime.utcnow()
        verified_count = 0
        errors = []

        # Find predictions that need verification
        pending = await self.collection.find({
            "status": "pending"
        }).to_list(length=1000)

        for pred in pending:
            try:
                created = pred['created_at']
                days_elapsed = (now - created).days

                if days_elapsed < 7:
                    continue  # Too early to verify

                symbol = pred['symbol']
                entry_price = pred['entry_price']
                recommendation = pred['recommendation']

                # Fetch current price
                ticker = yf.Ticker(f"{symbol}.NS")
                hist = ticker.history(period="1d")

                if hist.empty:
                    continue

                current_price = float(hist['Close'].iloc[-1])
                price_change_pct = ((current_price - entry_price) / entry_price) * 100

                # Determine if prediction was correct
                verification = pred.get('verification', {})

                # 7-day verification
                if days_elapsed >= 7 and verification.get('7_day') is None:
                    verification['7_day'] = self._evaluate_prediction(
                        recommendation, entry_price, current_price, pred.get('target_price'), pred.get('stop_loss')
                    )
                    verification['7_day']['verified_at'] = now
                    verification['7_day']['price_at_verification'] = current_price

                # 14-day verification
                if days_elapsed >= 14 and verification.get('14_day') is None:
                    verification['14_day'] = self._evaluate_prediction(
                        recommendation, entry_price, current_price, pred.get('target_price'), pred.get('stop_loss')
                    )
                    verification['14_day']['verified_at'] = now
                    verification['14_day']['price_at_verification'] = current_price

                # 30-day verification (final)
                if days_elapsed >= 30 and verification.get('30_day') is None:
                    verification['30_day'] = self._evaluate_prediction(
                        recommendation, entry_price, current_price, pred.get('target_price'), pred.get('stop_loss')
                    )
                    verification['30_day']['verified_at'] = now
                    verification['30_day']['price_at_verification'] = current_price

                    # Mark as verified after 30 days
                    status = "verified"
                else:
                    status = "pending"

                # Update the record
                await self.collection.update_one(
                    {"_id": pred['_id']},
                    {
                        "$set": {
                            "verification": verification,
                            "status": status,
                            "last_checked": now
                        }
                    }
                )

                verified_count += 1

            except Exception as e:
                errors.append(f"{pred.get('symbol', 'unknown')}: {str(e)}")
                logger.error(f"Error verifying prediction: {e}")

        return {
            "predictions_checked": len(pending),
            "predictions_updated": verified_count,
            "errors": errors,
            "timestamp": now.isoformat()
        }

    def _evaluate_prediction(
        self,
        recommendation: str,
        entry_price: float,
        current_price: float,
        target_price: Optional[float],
        stop_loss: Optional[float]
    ) -> Dict[str, Any]:
        """Evaluate if a prediction was correct"""
        price_change_pct = ((current_price - entry_price) / entry_price) * 100

        result = {
            "entry_price": entry_price,
            "exit_price": current_price,
            "price_change_pct": round(price_change_pct, 2)
        }

        if recommendation == "BUY":
            # BUY is correct if price went up
            if price_change_pct > 5:
                result["outcome"] = "strong_win"
                result["correct"] = True
            elif price_change_pct > 0:
                result["outcome"] = "win"
                result["correct"] = True
            elif price_change_pct > -5:
                result["outcome"] = "minor_loss"
                result["correct"] = False
            else:
                result["outcome"] = "loss"
                result["correct"] = False

            # Check if target was hit
            if target_price and current_price >= target_price:
                result["target_hit"] = True
            else:
                result["target_hit"] = False

            # Check if stop loss was hit
            if stop_loss and current_price <= stop_loss:
                result["stop_hit"] = True
            else:
                result["stop_hit"] = False

        elif recommendation == "SELL":
            # SELL is correct if price went down
            if price_change_pct < -5:
                result["outcome"] = "strong_win"
                result["correct"] = True
            elif price_change_pct < 0:
                result["outcome"] = "win"
                result["correct"] = True
            elif price_change_pct < 5:
                result["outcome"] = "minor_loss"
                result["correct"] = False
            else:
                result["outcome"] = "loss"
                result["correct"] = False

        else:  # HOLD
            # HOLD is correct if price stayed within +/- 5%
            if -5 <= price_change_pct <= 5:
                result["outcome"] = "correct"
                result["correct"] = True
            else:
                result["outcome"] = "incorrect"
                result["correct"] = False

        return result

    async def get_accuracy_stats(
        self,
        days_back: int = 90,
        min_confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get accuracy statistics for predictions.

        Args:
            days_back: Number of days to look back
            min_confidence: Only include predictions above this confidence

        Returns:
            Accuracy statistics
        """
        cutoff = datetime.utcnow() - timedelta(days=days_back)

        query = {
            "status": "verified",
            "created_at": {"$gte": cutoff}
        }

        if min_confidence:
            query["confidence"] = {"$gte": min_confidence}

        predictions = await self.collection.find(query).to_list(length=10000)

        if not predictions:
            return {
                "message": "No verified predictions found",
                "period_days": days_back,
                "total_predictions": 0
            }

        # Calculate stats for each timeframe
        stats = {}
        for period in ['7_day', '14_day', '30_day']:
            period_stats = self._calculate_period_stats(predictions, period)
            stats[period] = period_stats

        # Overall stats
        overall = {
            "total_predictions": len(predictions),
            "period_days": days_back,
            "by_recommendation": self._stats_by_recommendation(predictions),
            "by_confidence_band": self._stats_by_confidence(predictions),
            "by_period": stats
        }

        return overall

    def _calculate_period_stats(self, predictions: List[Dict], period: str) -> Dict[str, Any]:
        """Calculate statistics for a specific period"""
        verified = [p for p in predictions if p.get('verification', {}).get(period)]

        if not verified:
            return {"verified_count": 0}

        correct = sum(1 for p in verified if p['verification'][period].get('correct', False))
        total = len(verified)

        # Calculate average returns
        returns = []
        for p in verified:
            v = p['verification'][period]
            if v.get('price_change_pct') is not None:
                if p['recommendation'] == 'BUY':
                    returns.append(v['price_change_pct'])
                elif p['recommendation'] == 'SELL':
                    returns.append(-v['price_change_pct'])

        return {
            "verified_count": total,
            "correct_count": correct,
            "accuracy_pct": round((correct / total) * 100, 2) if total > 0 else 0,
            "avg_return_pct": round(np.mean(returns), 2) if returns else 0,
            "median_return_pct": round(np.median(returns), 2) if returns else 0,
            "best_return_pct": round(max(returns), 2) if returns else 0,
            "worst_return_pct": round(min(returns), 2) if returns else 0
        }

    def _stats_by_recommendation(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Calculate stats grouped by recommendation type"""
        results = {}

        for rec_type in ['BUY', 'SELL', 'HOLD']:
            preds = [p for p in predictions if p['recommendation'] == rec_type]

            if not preds:
                results[rec_type.lower()] = {"count": 0}
                continue

            # Use 30-day verification
            verified = [p for p in preds if p.get('verification', {}).get('30_day')]
            correct = sum(1 for p in verified if p['verification']['30_day'].get('correct', False))

            results[rec_type.lower()] = {
                "count": len(preds),
                "verified": len(verified),
                "correct": correct,
                "accuracy_pct": round((correct / len(verified)) * 100, 2) if verified else 0
            }

        return results

    def _stats_by_confidence(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Calculate stats grouped by confidence bands"""
        bands = [
            ("very_high", 85, 100),
            ("high", 70, 85),
            ("medium", 55, 70),
            ("low", 40, 55),
            ("very_low", 0, 40)
        ]

        results = {}

        for band_name, min_conf, max_conf in bands:
            preds = [p for p in predictions if min_conf <= p.get('confidence', 0) < max_conf]

            if not preds:
                results[band_name] = {"count": 0}
                continue

            verified = [p for p in preds if p.get('verification', {}).get('30_day')]
            correct = sum(1 for p in verified if p['verification']['30_day'].get('correct', False))

            results[band_name] = {
                "confidence_range": f"{min_conf}-{max_conf}%",
                "count": len(preds),
                "verified": len(verified),
                "correct": correct,
                "accuracy_pct": round((correct / len(verified)) * 100, 2) if verified else 0
            }

        return results

    async def get_recent_predictions(
        self,
        limit: int = 20,
        symbol: Optional[str] = None,
        recommendation: Optional[str] = None
    ) -> List[Dict]:
        """Get recent predictions with their outcomes"""
        query = {}

        if symbol:
            query["symbol"] = symbol.upper()
        if recommendation:
            query["recommendation"] = recommendation.upper()

        predictions = await self.collection.find(
            query,
            {"reasoning": 0}  # Exclude reasoning for list view
        ).sort("created_at", -1).limit(limit).to_list(length=limit)

        # Format for response
        results = []
        for p in predictions:
            p['_id'] = str(p['_id'])
            results.append(p)

        return results

    async def get_prediction_details(self, prediction_id: str) -> Optional[Dict]:
        """Get detailed prediction with full verification history"""
        from bson import ObjectId

        try:
            pred = await self.collection.find_one({"_id": ObjectId(prediction_id)})
            if pred:
                pred['_id'] = str(pred['_id'])
                return pred
        except:
            pass

        return None

    async def get_leaderboard(self, period_days: int = 30) -> Dict[str, Any]:
        """Get accuracy leaderboard by symbol"""
        cutoff = datetime.utcnow() - timedelta(days=period_days)

        predictions = await self.collection.find({
            "status": "verified",
            "created_at": {"$gte": cutoff}
        }).to_list(length=10000)

        # Group by symbol
        by_symbol = {}
        for p in predictions:
            sym = p['symbol']
            if sym not in by_symbol:
                by_symbol[sym] = {"total": 0, "correct": 0, "returns": []}

            by_symbol[sym]["total"] += 1

            v30 = p.get('verification', {}).get('30_day', {})
            if v30.get('correct'):
                by_symbol[sym]["correct"] += 1

            if v30.get('price_change_pct') is not None:
                if p['recommendation'] == 'BUY':
                    by_symbol[sym]["returns"].append(v30['price_change_pct'])
                elif p['recommendation'] == 'SELL':
                    by_symbol[sym]["returns"].append(-v30['price_change_pct'])

        # Calculate scores and sort
        leaderboard = []
        for symbol, data in by_symbol.items():
            if data["total"] >= 3:  # Minimum 3 predictions
                accuracy = (data["correct"] / data["total"]) * 100
                avg_return = np.mean(data["returns"]) if data["returns"] else 0

                leaderboard.append({
                    "symbol": symbol,
                    "total_predictions": data["total"],
                    "correct_predictions": data["correct"],
                    "accuracy_pct": round(accuracy, 1),
                    "avg_return_pct": round(avg_return, 2)
                })

        # Sort by accuracy
        leaderboard.sort(key=lambda x: x["accuracy_pct"], reverse=True)

        return {
            "period_days": period_days,
            "top_performers": leaderboard[:10],
            "worst_performers": sorted(leaderboard, key=lambda x: x["accuracy_pct"])[:10],
            "total_symbols_tracked": len(leaderboard)
        }


# Factory function
def get_confidence_tracker(db) -> ConfidenceTracker:
    """Create confidence tracker with database"""
    return ConfidenceTracker(db)
