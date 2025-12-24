"""
Market Data Background Tasks
"""

import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any

from celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3)
def update_market_data(self):
    """
    Update market data for watched stocks.
    Runs every 5 minutes during market hours.
    """
    try:
        from data_providers.tvscreener_provider import get_all_indian_stocks
        from database.mongo_client import get_mongo_client
        import redis
        import json
        import os

        logger.info("Starting market data update...")

        # Get Redis client for caching
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        redis_client = redis.from_url(redis_url)

        # Get top stocks to update
        stocks = get_all_indian_stocks(max_stocks=100)

        if not stocks:
            logger.warning("No stocks to update")
            return {"status": "no_stocks", "count": 0}

        updated_count = 0

        for stock in stocks:
            try:
                symbol = stock.get("symbol", "")
                if not symbol:
                    continue

                # Cache the stock data in Redis
                cache_key = f"stock:{symbol}:data"
                redis_client.setex(
                    cache_key,
                    300,  # 5 minutes TTL
                    json.dumps({
                        "symbol": symbol,
                        "name": stock.get("name", symbol),
                        "price": stock.get("close"),
                        "change": stock.get("change"),
                        "change_percent": stock.get("change_percent"),
                        "volume": stock.get("volume"),
                        "market_cap": stock.get("market_cap_basic"),
                        "updated_at": datetime.now().isoformat()
                    })
                )
                updated_count += 1

            except Exception as e:
                logger.warning(f"Failed to update {symbol}: {e}")
                continue

        logger.info(f"Updated market data for {updated_count} stocks")
        return {"status": "success", "updated": updated_count}

    except Exception as e:
        logger.error(f"Market data update failed: {e}")
        raise self.retry(exc=e, countdown=60)


@celery_app.task(bind=True, max_retries=3)
def refresh_stock_cache(self):
    """
    Refresh the full stock list cache.
    Runs every 6 hours.
    """
    try:
        from data_providers.tvscreener_provider import get_all_indian_stocks, clear_stock_cache

        logger.info("Refreshing stock cache...")

        # Clear existing cache
        clear_stock_cache()

        # Fetch fresh stock list
        stocks = get_all_indian_stocks(force_refresh=True)

        count = len(stocks) if stocks else 0
        logger.info(f"Stock cache refreshed with {count} stocks")

        return {"status": "success", "count": count}

    except Exception as e:
        logger.error(f"Stock cache refresh failed: {e}")
        raise self.retry(exc=e, countdown=300)


@celery_app.task(bind=True)
def fetch_live_price(self, symbol: str):
    """
    Fetch live price for a single stock.
    Can be called on-demand.
    """
    try:
        import yfinance as yf
        import redis
        import json
        import os

        logger.info(f"Fetching live price for {symbol}")

        # Add .NS suffix if needed
        yf_symbol = symbol if ".NS" in symbol or ".BO" in symbol else f"{symbol}.NS"

        ticker = yf.Ticker(yf_symbol)
        info = ticker.info

        if not info or "regularMarketPrice" not in info:
            logger.warning(f"No price data for {symbol}")
            return None

        price_data = {
            "symbol": symbol.replace(".NS", "").replace(".BO", ""),
            "current_price": info.get("regularMarketPrice"),
            "previous_close": info.get("previousClose"),
            "change": info.get("regularMarketChange"),
            "change_percent": info.get("regularMarketChangePercent"),
            "volume": info.get("volume"),
            "high": info.get("dayHigh"),
            "low": info.get("dayLow"),
            "open": info.get("open"),
            "timestamp": datetime.now().isoformat()
        }

        # Cache in Redis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        redis_client = redis.from_url(redis_url)
        redis_client.setex(
            f"price:{symbol}:latest",
            60,  # 1 minute TTL
            json.dumps(price_data)
        )

        return price_data

    except Exception as e:
        logger.error(f"Failed to fetch live price for {symbol}: {e}")
        return None


@celery_app.task(bind=True)
def batch_fetch_prices(self, symbols: List[str]):
    """
    Fetch prices for multiple stocks in batch.
    """
    try:
        import yfinance as yf
        import redis
        import json
        import os

        logger.info(f"Batch fetching prices for {len(symbols)} stocks")

        # Prepare symbols with suffix
        yf_symbols = [
            s if ".NS" in s or ".BO" in s else f"{s}.NS"
            for s in symbols
        ]

        # Batch download
        data = yf.download(
            yf_symbols,
            period="1d",
            interval="1m",
            progress=False
        )

        results = []
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        redis_client = redis.from_url(redis_url)

        for symbol, yf_symbol in zip(symbols, yf_symbols):
            try:
                if len(symbols) == 1:
                    latest = data.iloc[-1] if not data.empty else None
                else:
                    latest = data[yf_symbol].iloc[-1] if yf_symbol in data.columns.get_level_values(1) else None

                if latest is not None:
                    price_data = {
                        "symbol": symbol,
                        "current_price": float(latest.get("Close", 0)),
                        "high": float(latest.get("High", 0)),
                        "low": float(latest.get("Low", 0)),
                        "volume": int(latest.get("Volume", 0)),
                        "timestamp": datetime.now().isoformat()
                    }
                    results.append(price_data)

                    # Cache
                    redis_client.setex(
                        f"price:{symbol}:latest",
                        60,
                        json.dumps(price_data)
                    )

            except Exception as e:
                logger.warning(f"Failed to process {symbol}: {e}")
                continue

        logger.info(f"Batch fetch completed: {len(results)}/{len(symbols)} successful")
        return results

    except Exception as e:
        logger.error(f"Batch fetch failed: {e}")
        return []
