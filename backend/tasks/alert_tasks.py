"""
Alert Checking Background Tasks
"""

import logging
from datetime import datetime
from typing import List, Dict, Any

from celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3)
def check_price_alerts(self):
    """
    Check all active price alerts.
    Runs every minute during market hours.
    """
    try:
        from database.mongo_client import get_mongo_client
        import yfinance as yf

        logger.info("Checking price alerts...")

        mongo = get_mongo_client()
        db = mongo.get_database()

        # Get all active alerts
        active_alerts = list(db.alerts.find({"active": True}))

        if not active_alerts:
            return {"status": "no_alerts", "checked": 0}

        triggered = []
        checked = 0

        # Group alerts by symbol to minimize API calls
        alerts_by_symbol = {}
        for alert in active_alerts:
            symbol = alert.get("symbol")
            if symbol not in alerts_by_symbol:
                alerts_by_symbol[symbol] = []
            alerts_by_symbol[symbol].append(alert)

        for symbol, symbol_alerts in alerts_by_symbol.items():
            try:
                # Get current price
                yf_symbol = f"{symbol}.NS"
                ticker = yf.Ticker(yf_symbol)
                info = ticker.info

                current_price = info.get("regularMarketPrice")
                if not current_price:
                    continue

                checked += len(symbol_alerts)

                for alert in symbol_alerts:
                    alert_type = alert.get("alert_type", "")
                    target_price = alert.get("target_price")
                    condition = alert.get("condition", "above")

                    should_trigger = False

                    if alert_type == "price":
                        if condition == "above" and current_price >= target_price:
                            should_trigger = True
                        elif condition == "below" and current_price <= target_price:
                            should_trigger = True

                    elif alert_type == "percent_change":
                        prev_close = info.get("previousClose", current_price)
                        change_pct = ((current_price - prev_close) / prev_close) * 100

                        if condition == "above" and change_pct >= target_price:
                            should_trigger = True
                        elif condition == "below" and change_pct <= target_price:
                            should_trigger = True

                    if should_trigger:
                        # Trigger the alert
                        trigger_alert.delay(
                            str(alert["_id"]),
                            current_price,
                            alert.get("user_id", "default")
                        )
                        triggered.append(alert["_id"])

            except Exception as e:
                logger.warning(f"Failed to check alerts for {symbol}: {e}")
                continue

        logger.info(f"Checked {checked} alerts, triggered {len(triggered)}")
        return {
            "status": "success",
            "checked": checked,
            "triggered": len(triggered)
        }

    except Exception as e:
        logger.error(f"Alert check failed: {e}")
        raise self.retry(exc=e, countdown=30)


@celery_app.task(bind=True)
def trigger_alert(self, alert_id: str, current_price: float, user_id: str):
    """
    Process a triggered alert.
    """
    try:
        from database.mongo_client import get_mongo_client
        from bson.objectid import ObjectId

        logger.info(f"Processing triggered alert: {alert_id}")

        mongo = get_mongo_client()
        db = mongo.get_database()

        # Get the alert
        alert = db.alerts.find_one({"_id": ObjectId(alert_id)})
        if not alert:
            logger.warning(f"Alert {alert_id} not found")
            return {"status": "not_found"}

        # Update alert as triggered
        db.alerts.update_one(
            {"_id": ObjectId(alert_id)},
            {
                "$set": {
                    "active": False,
                    "triggered_at": datetime.now(),
                    "triggered_price": current_price
                }
            }
        )

        # Create notification
        notification = {
            "user_id": user_id,
            "type": "alert_triggered",
            "alert_id": alert_id,
            "symbol": alert.get("symbol"),
            "message": f"Alert triggered for {alert.get('symbol')}: Price is now ₹{current_price:.2f}",
            "target_price": alert.get("target_price"),
            "current_price": current_price,
            "created_at": datetime.now(),
            "read": False
        }

        db.notifications.insert_one(notification)

        logger.info(f"Alert {alert_id} triggered successfully")
        return {"status": "triggered", "alert_id": alert_id}

    except Exception as e:
        logger.error(f"Failed to trigger alert {alert_id}: {e}")
        return {"status": "error", "message": str(e)}


@celery_app.task(bind=True)
def create_smart_alert(self, symbol: str, user_id: str, alert_config: Dict[str, Any]):
    """
    Create a smart alert based on technical analysis.
    """
    try:
        from database.mongo_client import get_mongo_client
        import yfinance as yf
        import pandas as pd

        logger.info(f"Creating smart alert for {symbol}")

        # Get historical data
        yf_symbol = f"{symbol}.NS"
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period="1mo")

        if hist.empty:
            return {"status": "no_data"}

        current_price = hist.iloc[-1]["Close"]

        # Calculate support/resistance levels
        high_20d = hist["High"].tail(20).max()
        low_20d = hist["Low"].tail(20).min()
        resistance = high_20d
        support = low_20d

        # Calculate RSI
        delta = hist["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        # Determine alert type based on config
        alert_type = alert_config.get("type", "price_breakout")

        mongo = get_mongo_client()
        db = mongo.get_database()

        alerts_created = []

        if alert_type == "price_breakout":
            # Create alerts for support/resistance breakout
            # Resistance breakout
            db.alerts.insert_one({
                "user_id": user_id,
                "symbol": symbol,
                "alert_type": "price",
                "condition": "above",
                "target_price": resistance * 1.01,  # 1% above resistance
                "message": f"Resistance breakout at ₹{resistance:.2f}",
                "active": True,
                "smart_alert": True,
                "created_at": datetime.now()
            })
            alerts_created.append("resistance_breakout")

            # Support breakdown
            db.alerts.insert_one({
                "user_id": user_id,
                "symbol": symbol,
                "alert_type": "price",
                "condition": "below",
                "target_price": support * 0.99,  # 1% below support
                "message": f"Support breakdown at ₹{support:.2f}",
                "active": True,
                "smart_alert": True,
                "created_at": datetime.now()
            })
            alerts_created.append("support_breakdown")

        elif alert_type == "rsi_extreme":
            # Oversold alert
            if current_rsi > 30:
                db.alerts.insert_one({
                    "user_id": user_id,
                    "symbol": symbol,
                    "alert_type": "rsi",
                    "condition": "below",
                    "target_price": 30,
                    "message": "RSI oversold - potential buying opportunity",
                    "active": True,
                    "smart_alert": True,
                    "created_at": datetime.now()
                })
                alerts_created.append("rsi_oversold")

            # Overbought alert
            if current_rsi < 70:
                db.alerts.insert_one({
                    "user_id": user_id,
                    "symbol": symbol,
                    "alert_type": "rsi",
                    "condition": "above",
                    "target_price": 70,
                    "message": "RSI overbought - potential selling opportunity",
                    "active": True,
                    "smart_alert": True,
                    "created_at": datetime.now()
                })
                alerts_created.append("rsi_overbought")

        logger.info(f"Created {len(alerts_created)} smart alerts for {symbol}")
        return {
            "status": "success",
            "symbol": symbol,
            "alerts_created": alerts_created,
            "analysis": {
                "current_price": round(current_price, 2),
                "resistance": round(resistance, 2),
                "support": round(support, 2),
                "rsi": round(current_rsi, 2)
            }
        }

    except Exception as e:
        logger.error(f"Failed to create smart alert for {symbol}: {e}")
        return {"status": "error", "message": str(e)}
