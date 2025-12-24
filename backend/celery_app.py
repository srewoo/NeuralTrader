"""
Celery Application Configuration
Background job processing for NeuralTrader
"""

import os
from celery import Celery
from celery.schedules import crontab
from dotenv import load_dotenv

load_dotenv()

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery app
celery_app = Celery(
    "neuraltrader",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "tasks.market_tasks",
        "tasks.ai_tasks",
        "tasks.alert_tasks",
        "tasks.news_tasks"
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=True,

    # Task execution settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,

    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour

    # Rate limiting
    task_default_rate_limit="10/s",

    # Retry settings
    task_default_retry_delay=30,
    task_max_retries=3,

    # Beat scheduler for periodic tasks
    beat_schedule={
        # Update market data every 5 minutes during market hours
        "update-market-data": {
            "task": "tasks.market_tasks.update_market_data",
            "schedule": crontab(minute="*/5", hour="9-16", day_of_week="1-5"),
        },
        # Update news every 15 minutes
        "update-news": {
            "task": "tasks.news_tasks.fetch_latest_news",
            "schedule": crontab(minute="*/15"),
        },
        # Check price alerts every minute during market hours
        "check-alerts": {
            "task": "tasks.alert_tasks.check_price_alerts",
            "schedule": crontab(minute="*", hour="9-16", day_of_week="1-5"),
        },
        # Update AI predictions every 30 minutes during market hours
        "update-ai-predictions": {
            "task": "tasks.ai_tasks.update_predictions",
            "schedule": crontab(minute="*/30", hour="9-16", day_of_week="1-5"),
        },
        # Daily market analysis at 9:30 AM IST
        "daily-market-analysis": {
            "task": "tasks.ai_tasks.daily_market_analysis",
            "schedule": crontab(hour=9, minute=30, day_of_week="1-5"),
        },
        # Update stock cache every 6 hours
        "refresh-stock-cache": {
            "task": "tasks.market_tasks.refresh_stock_cache",
            "schedule": crontab(minute=0, hour="*/6"),
        },
        # Track prediction accuracy daily at EOD
        "track-prediction-accuracy": {
            "task": "tasks.ai_tasks.track_prediction_accuracy",
            "schedule": crontab(hour=16, minute=30, day_of_week="1-5"),
        },
    },
)


if __name__ == "__main__":
    celery_app.start()
