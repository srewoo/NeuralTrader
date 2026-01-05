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


@celery_app.task(bind=True, max_retries=1)
def generate_stock_recommendations(self, limit: int = 200):
    """
    Generate comprehensive stock recommendations for top NSE/BSE stocks.
    Runs every 4 hours to keep data fresh.
    Users always see cached results instantly.

    Args:
        limit: Number of stocks to analyze (default: 200 for top NSE/BSE stocks)
    """
    try:
        import yfinance as yf
        from datetime import timezone
        from data_providers.tvscreener_provider import get_all_indian_stocks_async
        from agents.ensemble_analyzer import get_ensemble_analyzer
        from database.mongo_client import get_mongo_client

        logger.info(f"🚀 Starting background stock recommendations generation for {limit} stocks...")

        # Get MongoDB connection
        mongo = get_mongo_client()
        db = mongo.get_database()

        # Fetch stocks dynamically
        async def fetch_stocks():
            return await get_all_indian_stocks_async(
                min_market_cap=100,  # 100 Cr minimum
                max_stocks=limit * 2  # Fetch more for filtering
            )

        dynamic_stocks = run_async(fetch_stocks())

        if not dynamic_stocks:
            logger.warning("No stocks fetched from TradingView, using fallback")
            from data_providers.indian_stocks import NIFTY_100_STOCKS
            stocks_to_analyze = [s['symbol'] for s in NIFTY_100_STOCKS[:limit]]
        else:
            stocks_to_analyze = [s['symbol'] for s in dynamic_stocks[:limit]]

        logger.info(f"📊 Analyzing {len(stocks_to_analyze)} stocks...")

        # Use ensemble analyzer
        analyzer = get_ensemble_analyzer()

        buy_recs = []
        sell_recs = []
        processed = 0

        # Process stocks (sequential to avoid overwhelming APIs)
        for symbol in stocks_to_analyze:
            try:
                # Try both NSE (.NS) and BSE (.BO) exchanges
                ticker = None
                exchange = None
                info = {}

                # Try NSE first (more liquid, primary exchange)
                try:
                    nse_ticker = f"{symbol}.NS"
                    nse_data = yf.Ticker(nse_ticker)
                    nse_info = nse_data.info

                    # Check if valid data exists
                    if nse_info and nse_info.get('regularMarketPrice'):
                        ticker = nse_ticker
                        exchange = "NSE"
                        info = nse_info
                except Exception:
                    pass

                # If NSE fails, try BSE
                if not ticker:
                    try:
                        bse_ticker = f"{symbol}.BO"
                        bse_data = yf.Ticker(bse_ticker)
                        bse_info = bse_data.info

                        if bse_info and bse_info.get('regularMarketPrice'):
                            ticker = bse_ticker
                            exchange = "BSE"
                            info = bse_info
                    except Exception:
                        pass

                # Skip if no valid data from either exchange
                if not ticker or not info:
                    logger.debug(f"No valid data for {symbol} on NSE or BSE")
                    continue

                stock_info = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "name": info.get('longName', symbol),
                    "sector": info.get('sector', 'N/A')
                }

                logger.debug(f"Analyzing {symbol} on {exchange}")

                # Analyze stock
                async def analyze():
                    return await analyzer.analyze_stock(symbol, include_news=True)

                result = run_async(analyze())

                if result.get('error'):
                    continue

                recommendation = result.get('recommendation', 'HOLD')
                confidence = result.get('confidence', 0)

                # Only include strong signals (60%+ confidence)
                if recommendation == 'HOLD' or confidence < 60.0:
                    continue

                # Build recommendation object
                signals = []
                for signal in result.get('signals', []):
                    signals.append(signal.get('description', signal.get('indicator', '')))

                regime = result.get('market_regime', {})
                if regime.get('primary_regime'):
                    signals.insert(0, f"Regime: {regime.get('primary_regime').replace('_', ' ').title()}")

                price_targets = result.get('price_targets', {})
                indicators = result.get('indicators', {})
                fundamentals = result.get('fundamentals', {})

                # Extract key fundamental metrics
                fundamental_metrics = None
                if fundamentals:
                    fundamental_metrics = {
                        "pe_ratio": fundamentals.get('pe_ratio'),
                        "pb_ratio": fundamentals.get('pb_ratio'),
                        "debt_to_equity": fundamentals.get('debt_to_equity'),
                        "roe": fundamentals.get('roe'),
                        "profit_margin": fundamentals.get('profit_margin'),
                        "dividend_yield": fundamentals.get('dividend_yield'),
                        "market_cap": fundamentals.get('market_cap'),
                        "fundamental_score": result.get('fundamental_score')
                    }

                rec_obj = {
                    "symbol": symbol,
                    "exchange": exchange,  # NSE or BSE
                    "name": stock_info.get("name", symbol),
                    "sector": stock_info.get("sector", "N/A"),
                    "recommendation": recommendation,
                    "confidence": round(confidence, 1),
                    "current_price": result.get('current_price', 0),
                    "target_price": price_targets.get('target'),
                    "stop_loss": price_targets.get('stop_loss'),
                    "signals": signals[:10],
                    "indicators": {
                        "rsi": indicators.get('rsi'),
                        "macd": indicators.get('macd'),
                        "adx": indicators.get('adx'),
                    },
                    "fundamentals": fundamental_metrics,
                    "market_regime": regime.get('primary_regime', 'unknown'),
                    "sentiment_score": result.get('sentiment', {}).get('score', 0),
                }

                if recommendation == 'BUY':
                    buy_recs.append(rec_obj)
                elif recommendation == 'SELL':
                    sell_recs.append(rec_obj)

                processed += 1

                if processed % 10 == 0:
                    logger.info(f"✅ Processed {processed}/{len(stocks_to_analyze)} stocks")

            except Exception as e:
                logger.warning(f"Failed to analyze {symbol}: {e}")
                continue

        # Sort by confidence
        buy_recs.sort(key=lambda x: x['confidence'], reverse=True)
        sell_recs.sort(key=lambda x: x['confidence'], reverse=True)

        # Count stocks by exchange
        all_recs = buy_recs + sell_recs
        nse_count = sum(1 for r in all_recs if r.get('exchange') == 'NSE')
        bse_count = sum(1 for r in all_recs if r.get('exchange') == 'BSE')

        # Calculate market sentiment
        market_sentiment = "Neutral"
        if len(buy_recs) > len(sell_recs) * 1.5:
            market_sentiment = "Bullish"
        elif len(sell_recs) > len(buy_recs) * 1.5:
            market_sentiment = "Bearish"

        # Calculate averages
        avg_buy_conf = sum(r['confidence'] for r in buy_recs) / len(buy_recs) if buy_recs else 0
        avg_sell_conf = sum(r['confidence'] for r in sell_recs) / len(sell_recs) if sell_recs else 0

        # Analyze sector diversification
        def analyze_sector_diversification(recommendations):
            sector_counts = {}
            for rec in recommendations:
                sector = rec.get('sector', 'Unknown')
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

            total = len(recommendations)
            warnings = []

            for sector, count in sector_counts.items():
                percentage = (count / total) * 100 if total > 0 else 0
                if percentage > 40:  # More than 40% in one sector
                    warnings.append({
                        "sector": sector,
                        "count": count,
                        "percentage": round(percentage, 1),
                        "severity": "high",
                        "message": f"{count} stocks ({percentage:.1f}%) from {sector} sector - High concentration risk"
                    })
                elif percentage > 25:  # More than 25% in one sector
                    warnings.append({
                        "sector": sector,
                        "count": count,
                        "percentage": round(percentage, 1),
                        "severity": "medium",
                        "message": f"{count} stocks ({percentage:.1f}%) from {sector} sector - Consider diversification"
                    })

            return {
                "warnings": warnings,
                "sector_breakdown": [{
                    "sector": sector,
                    "count": count,
                    "percentage": round((count / total) * 100, 1) if total > 0 else 0
                } for sector, count in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)]
            }

        buy_diversification = analyze_sector_diversification(buy_recs)
        sell_diversification = analyze_sector_diversification(sell_recs)

        # Get historical accuracy stats
        historical_accuracy = None
        try:
            from tracking.confidence_tracker import ConfidenceTracker
            tracker = ConfidenceTracker(db)

            # Use sync version for Celery task
            async def get_stats():
                return await tracker.get_accuracy_stats(days_back=30)

            import asyncio
            accuracy_stats = asyncio.run(get_stats()) if asyncio.get_event_loop().is_running() else run_async(get_stats())

            if accuracy_stats and accuracy_stats.get('total_predictions', 0) > 0:
                historical_accuracy = {
                    "total_predictions": accuracy_stats.get('total_predictions', 0),
                    "period_days": accuracy_stats.get('period_days', 30),
                    "by_recommendation": accuracy_stats.get('by_recommendation', {}),
                    "by_confidence_band": accuracy_stats.get('by_confidence_band', {})
                }
        except Exception as e:
            logger.warning(f"Failed to get historical accuracy: {e}")

        # Build recommendations document
        recommendations_data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "analysis_type": "enhanced_background",
            "total_stocks_analyzed": len(stocks_to_analyze),
            "min_confidence_threshold": 60.0,
            "sentiment_enabled": True,
            "backtest_enabled": False,
            "summary": {
                "total_buy_signals": len(buy_recs),
                "total_sell_signals": len(sell_recs),
                "market_sentiment": market_sentiment,
                "avg_buy_confidence": round(avg_buy_conf, 1),
                "avg_sell_confidence": round(avg_sell_conf, 1),
                "nse_stocks": nse_count,
                "bse_stocks": bse_count,
                "historical_accuracy": historical_accuracy,
            },
            "buy_recommendations": buy_recs[:50],  # Top 50 BUY signals
            "sell_recommendations": sell_recs[:30],  # Top 30 SELL signals
            "diversification_analysis": {
                "buy": buy_diversification,
                "sell": sell_diversification
            }
        }

        # Save to MongoDB (upsert to replace old data)
        from datetime import timedelta
        one_hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

        existing = db.recommendations.find_one(
            {"generated_at": {"$gte": one_hour_ago}},
            sort=[("generated_at", -1)]
        )

        if existing:
            db.recommendations.update_one(
                {"_id": existing["_id"]},
                {"$set": recommendations_data}
            )
            logger.info("✅ Updated existing recommendation")
        else:
            db.recommendations.insert_one(recommendations_data.copy())
            logger.info("✅ Inserted new recommendation")

        logger.info(f"🎉 Background analysis complete! {len(buy_recs)} BUY ({avg_buy_conf:.1f}% avg), {len(sell_recs)} SELL ({avg_sell_conf:.1f}% avg) | NSE: {nse_count}, BSE: {bse_count}")

        return {
            "status": "success",
            "stocks_analyzed": len(stocks_to_analyze),
            "buy_signals": len(buy_recs),
            "sell_signals": len(sell_recs),
            "market_sentiment": market_sentiment
        }

    except Exception as e:
        logger.error(f"❌ Background recommendations generation failed: {e}")
        raise self.retry(exc=e, countdown=600)  # Retry after 10 minutes
