"""
News Fetching Background Tasks
"""

import logging
from datetime import datetime
from typing import List, Dict, Any

from celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3)
def fetch_latest_news(self):
    """
    Fetch latest news from all sources.
    Runs every 15 minutes.
    """
    try:
        from news.sources import get_news_aggregator
        from database.mongo_client import get_mongo_client
        import redis
        import json
        import os

        logger.info("Fetching latest news...")

        aggregator = get_news_aggregator()
        articles = aggregator.fetch_latest_news(limit=50)

        if not articles:
            logger.warning("No news articles fetched")
            return {"status": "no_articles", "count": 0}

        # Store in MongoDB
        mongo = get_mongo_client()
        db = mongo.get_database()

        new_count = 0
        for article in articles:
            try:
                # Use link as unique identifier
                result = db.news.update_one(
                    {"link": article.get("link")},
                    {
                        "$set": {
                            **article,
                            "updated_at": datetime.now()
                        },
                        "$setOnInsert": {
                            "created_at": datetime.now()
                        }
                    },
                    upsert=True
                )
                if result.upserted_id:
                    new_count += 1
            except Exception as e:
                logger.warning(f"Failed to store article: {e}")
                continue

        # Cache latest news in Redis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        redis_client = redis.from_url(redis_url)
        redis_client.setex(
            "news:latest",
            900,  # 15 minutes TTL
            json.dumps(articles[:20])  # Cache top 20
        )

        logger.info(f"Fetched {len(articles)} articles, {new_count} new")
        return {
            "status": "success",
            "total": len(articles),
            "new": new_count
        }

    except Exception as e:
        logger.error(f"News fetch failed: {e}")
        raise self.retry(exc=e, countdown=60)


@celery_app.task(bind=True)
def fetch_stock_news(self, symbol: str):
    """
    Fetch news for a specific stock.
    On-demand task.
    """
    try:
        from news.sources import get_news_aggregator
        from database.mongo_client import get_mongo_client
        import redis
        import json
        import os

        logger.info(f"Fetching news for {symbol}")

        aggregator = get_news_aggregator()
        articles = aggregator.fetch_stock_news(symbol, limit=20)

        if not articles:
            logger.info(f"No news for {symbol}")
            return {"status": "no_articles", "symbol": symbol}

        # Cache in Redis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        redis_client = redis.from_url(redis_url)
        redis_client.setex(
            f"news:{symbol}:latest",
            1800,  # 30 minutes TTL
            json.dumps(articles)
        )

        # Store in MongoDB with symbol tag
        mongo = get_mongo_client()
        db = mongo.get_database()

        for article in articles:
            try:
                db.news.update_one(
                    {"link": article.get("link")},
                    {
                        "$set": {
                            **article,
                            "updated_at": datetime.now()
                        },
                        "$setOnInsert": {
                            "created_at": datetime.now()
                        },
                        "$addToSet": {
                            "symbols": symbol
                        }
                    },
                    upsert=True
                )
            except Exception as e:
                continue

        logger.info(f"Fetched {len(articles)} articles for {symbol}")
        return {
            "status": "success",
            "symbol": symbol,
            "count": len(articles)
        }

    except Exception as e:
        logger.error(f"Stock news fetch failed for {symbol}: {e}")
        return {"status": "error", "message": str(e)}


@celery_app.task(bind=True)
def analyze_news_sentiment(self, symbol: str = None):
    """
    Analyze sentiment of recent news.
    """
    try:
        from news.sources import get_news_aggregator
        from agents.ensemble_analyzer import get_ensemble_analyzer
        from database.mongo_client import get_mongo_client
        import asyncio

        logger.info(f"Analyzing news sentiment for {symbol or 'market'}")

        aggregator = get_news_aggregator()

        if symbol:
            articles = aggregator.fetch_stock_news(symbol, limit=10)
        else:
            articles = aggregator.fetch_latest_news(limit=20)

        if not articles:
            return {"status": "no_articles"}

        # Combine article texts
        combined_text = "\n\n".join([
            f"{a.get('title', '')}: {a.get('description', '')}"
            for a in articles
        ])

        # Use AI to analyze sentiment
        analyzer = get_ensemble_analyzer()

        async def analyze():
            prompt = f"""Analyze the sentiment of these news articles about {'the stock ' + symbol if symbol else 'the Indian stock market'}.

Articles:
{combined_text}

Provide:
1. Overall sentiment (Bullish/Bearish/Neutral)
2. Sentiment score (-100 to +100)
3. Key themes
4. Important events
5. Market impact assessment

Format as JSON."""
            return await analyzer.get_ai_response(prompt)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(analyze())
        loop.close()

        # Store sentiment analysis
        mongo = get_mongo_client()
        db = mongo.get_database()

        sentiment_doc = {
            "symbol": symbol or "MARKET",
            "sentiment": result,
            "articles_analyzed": len(articles),
            "created_at": datetime.now()
        }

        db.sentiment_analysis.insert_one(sentiment_doc)

        logger.info(f"Sentiment analysis completed for {symbol or 'market'}")
        return {
            "status": "success",
            "symbol": symbol,
            "sentiment": result
        }

    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return {"status": "error", "message": str(e)}


@celery_app.task(bind=True)
def get_trending_topics(self):
    """
    Extract trending topics from recent news.
    """
    try:
        from news.sources import get_news_aggregator
        import redis
        import json
        import os

        logger.info("Extracting trending topics...")

        aggregator = get_news_aggregator()
        topics = aggregator.get_trending_topics(limit=20)

        # Cache in Redis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        redis_client = redis.from_url(redis_url)
        redis_client.setex(
            "news:trending",
            1800,  # 30 minutes TTL
            json.dumps(topics)
        )

        logger.info(f"Found {len(topics)} trending topics")
        return {
            "status": "success",
            "topics": topics
        }

    except Exception as e:
        logger.error(f"Trending topics extraction failed: {e}")
        return {"status": "error", "message": str(e)}
