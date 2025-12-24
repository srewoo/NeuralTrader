"""
AI Analysis Background Tasks
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any

from celery_app import celery_app

logger = logging.getLogger(__name__)


def run_async(coro):
    """Run async function in sync context"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, max_retries=2)
def update_predictions(self):
    """
    Update AI predictions for top stocks.
    Runs every 30 minutes during market hours.
    """
    try:
        from data_providers.tvscreener_provider import get_all_indian_stocks
        from agents.ensemble_analyzer import get_ensemble_analyzer
        from database.mongo_client import get_mongo_client
        import os

        logger.info("Starting AI predictions update...")

        # Get top stocks
        stocks = get_all_indian_stocks(max_stocks=50)

        if not stocks:
            logger.warning("No stocks for predictions")
            return {"status": "no_stocks", "count": 0}

        analyzer = get_ensemble_analyzer()
        mongo = get_mongo_client()
        db = mongo.get_database()

        updated_count = 0
        errors = []

        for stock in stocks[:20]:  # Limit to top 20
            try:
                symbol = stock.get("symbol", "")
                if not symbol:
                    continue

                # Run ensemble analysis
                async def analyze():
                    return await analyzer.analyze_stock(symbol)

                result = run_async(analyze())

                if result:
                    # Store prediction in MongoDB
                    prediction_doc = {
                        "symbol": symbol,
                        "prediction": result.get("recommendation"),
                        "confidence": result.get("confidence"),
                        "target_price": result.get("target_price"),
                        "analysis": result.get("analysis", {}),
                        "models_used": result.get("models_used", []),
                        "created_at": datetime.now(),
                        "valid_until": datetime.now() + timedelta(hours=4)
                    }

                    db.predictions.update_one(
                        {"symbol": symbol},
                        {"$set": prediction_doc},
                        upsert=True
                    )

                    updated_count += 1
                    logger.debug(f"Updated prediction for {symbol}")

            except Exception as e:
                logger.warning(f"Failed to update prediction for {stock.get('symbol')}: {e}")
                errors.append(str(e))
                continue

        logger.info(f"Updated predictions for {updated_count} stocks")
        return {
            "status": "success",
            "updated": updated_count,
            "errors": len(errors)
        }

    except Exception as e:
        logger.error(f"Predictions update failed: {e}")
        raise self.retry(exc=e, countdown=120)


@celery_app.task(bind=True)
def daily_market_analysis(self):
    """
    Generate daily market analysis report.
    Runs at 9:30 AM IST on weekdays.
    """
    try:
        from agents.ensemble_analyzer import get_ensemble_analyzer
        from database.mongo_client import get_mongo_client
        import os

        logger.info("Starting daily market analysis...")

        analyzer = get_ensemble_analyzer()
        mongo = get_mongo_client()
        db = mongo.get_database()

        # Get market overview analysis
        async def get_analysis():
            return await analyzer.get_market_overview()

        overview = run_async(get_analysis())

        if overview:
            # Store daily report
            report_doc = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "analysis": overview,
                "created_at": datetime.now(),
                "type": "daily_overview"
            }

            db.market_reports.update_one(
                {"date": report_doc["date"], "type": "daily_overview"},
                {"$set": report_doc},
                upsert=True
            )

            logger.info("Daily market analysis completed")
            return {"status": "success", "date": report_doc["date"]}

        return {"status": "no_data"}

    except Exception as e:
        logger.error(f"Daily market analysis failed: {e}")
        return {"status": "error", "message": str(e)}


@celery_app.task(bind=True)
def track_prediction_accuracy(self):
    """
    Track and record prediction accuracy.
    Runs daily at 4:30 PM IST after market close.
    """
    try:
        from database.mongo_client import get_mongo_client
        import yfinance as yf

        logger.info("Tracking prediction accuracy...")

        mongo = get_mongo_client()
        db = mongo.get_database()

        # Get predictions from today
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        predictions = list(db.predictions.find({
            "created_at": {"$gte": today}
        }))

        if not predictions:
            logger.info("No predictions to track today")
            return {"status": "no_predictions"}

        correct = 0
        total = 0
        results = []

        for pred in predictions:
            try:
                symbol = pred.get("symbol")
                predicted = pred.get("prediction")  # BUY, SELL, HOLD

                if not symbol or not predicted:
                    continue

                # Get current price
                yf_symbol = f"{symbol}.NS"
                ticker = yf.Ticker(yf_symbol)
                hist = ticker.history(period="1d")

                if hist.empty:
                    continue

                # Calculate actual performance
                open_price = hist.iloc[0]["Open"]
                close_price = hist.iloc[-1]["Close"]
                change_pct = ((close_price - open_price) / open_price) * 100

                # Determine if prediction was correct
                actual = "HOLD"
                if change_pct > 1:
                    actual = "BUY"
                elif change_pct < -1:
                    actual = "SELL"

                is_correct = (
                    (predicted == "BUY" and change_pct > 0) or
                    (predicted == "SELL" and change_pct < 0) or
                    (predicted == "HOLD" and abs(change_pct) < 1)
                )

                if is_correct:
                    correct += 1
                total += 1

                results.append({
                    "symbol": symbol,
                    "predicted": predicted,
                    "actual": actual,
                    "change_pct": round(change_pct, 2),
                    "is_correct": is_correct
                })

            except Exception as e:
                logger.warning(f"Failed to track {pred.get('symbol')}: {e}")
                continue

        # Calculate accuracy
        accuracy = (correct / total * 100) if total > 0 else 0

        # Store accuracy record
        accuracy_doc = {
            "date": today.strftime("%Y-%m-%d"),
            "total_predictions": total,
            "correct": correct,
            "accuracy": round(accuracy, 2),
            "results": results,
            "created_at": datetime.now()
        }

        db.prediction_accuracy.insert_one(accuracy_doc)

        logger.info(f"Prediction accuracy: {accuracy:.2f}% ({correct}/{total})")
        return {
            "status": "success",
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }

    except Exception as e:
        logger.error(f"Accuracy tracking failed: {e}")
        return {"status": "error", "message": str(e)}


@celery_app.task(bind=True)
def analyze_stock_async(self, symbol: str, user_id: str = "default"):
    """
    Run full AI analysis for a stock asynchronously.
    Returns task ID for polling.
    """
    try:
        from agents.ensemble_analyzer import get_ensemble_analyzer
        from database.mongo_client import get_mongo_client

        logger.info(f"Starting async analysis for {symbol}")

        analyzer = get_ensemble_analyzer()

        async def analyze():
            return await analyzer.analyze_stock(symbol, include_news=True)

        result = run_async(analyze())

        if result:
            # Store in MongoDB
            mongo = get_mongo_client()
            db = mongo.get_database()

            analysis_doc = {
                "symbol": symbol,
                "user_id": user_id,
                "analysis": result,
                "created_at": datetime.now(),
                "task_id": self.request.id
            }

            db.analyses.insert_one(analysis_doc)

            logger.info(f"Analysis completed for {symbol}")
            return {
                "status": "success",
                "symbol": symbol,
                "result": result
            }

        return {"status": "failed", "symbol": symbol}

    except Exception as e:
        logger.error(f"Async analysis failed for {symbol}: {e}")
        return {"status": "error", "message": str(e)}
