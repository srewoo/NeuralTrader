"""
News API Rate Limiter and Caching System
Prevents rate limit exhaustion for NewsAPI, Alpha Vantage, and other limited APIs
"""

import redis
import json
import logging
import time
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from functools import wraps
import asyncio
import pytz

logger = logging.getLogger(__name__)

# Indian market timezone
IST = pytz.timezone('Asia/Kolkata')


class NewsRateLimiter:
    """
    Rate limiter with Redis caching for news APIs

    Features:
    - Redis caching to minimize API calls
    - Rate limiting per API provider
    - Exponential backoff on failures
    - Request deduplication
    """

    # Rate limits for different providers (requests per minute)
    # Conservative limits to prevent quota exhaustion
    RATE_LIMITS = {
        # News Providers
        "newsapi": 2,  # 100/day free = ~4/hour → conservative 2/min
        "rss": 30,  # RSS feeds can handle more

        # Market Data Providers (from screenshot)
        "finnhub": 50,  # 60/min free → conservative 50/min
        "alpaca": 150,  # 200/min free → conservative 150/min
        "fmp": 4,  # 250/day free = ~10/hour → conservative 4/min
        "iex": 5,  # May be down, conservative limit
        "polygon": 4,  # 5/min free → conservative 4/min
        "twelvedata": 6,  # 8/min, 800/day free → conservative 6/min
        "yfinance": 100,  # Unofficial, be conservative
        "alpha_vantage": 5,  # 5/minute limit

        # Indian Market Specific
        "fii_dii": 10,  # NSE India API
        "nse": 10,  # NSE general
        "bse": 10,  # BSE
        "angelone": 20,  # Angel One (if used)
    }

    # Cache TTL (Time To Live) in seconds
    CACHE_TTL = {
        "market_news": 300,  # 5 minutes
        "trending": 600,  # 10 minutes
        "symbol_news": 300,  # 5 minutes
        "fii_dii": 1800,  # 30 minutes
        "sentiment": 300,  # 5 minutes
        "market_overview": 120,  # 2 minutes (needs to be fresh)
        "stock_quote": 60,  # 1 minute
        "index_data": 120,  # 2 minutes
    }

    def __init__(self, redis_client: Optional[redis.Redis] = None, enforce_market_hours: bool = True):
        """
        Initialize rate limiter with Redis client

        Args:
            redis_client: Optional Redis client instance
            enforce_market_hours: If True, restrict API calls to 9 AM - 5 PM IST
        """
        if redis_client is None:
            try:
                self.redis = redis.Redis(
                    host='localhost',
                    port=6379,
                    db=0,
                    decode_responses=True,
                    socket_connect_timeout=5
                )
                # Test connection
                self.redis.ping()
                self.enabled = True
                logger.info("Rate limiter initialized with Redis caching")
            except Exception as e:
                logger.warning(f"Redis not available for rate limiting: {e}")
                self.redis = None
                self.enabled = False
        else:
            self.redis = redis_client
            self.enabled = True

        self.enforce_market_hours = enforce_market_hours

    def _get_cache_key(self, provider: str, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key from provider, endpoint, and params"""
        # Sort params for consistent key
        sorted_params = json.dumps(params, sort_keys=True)
        return f"news_cache:{provider}:{endpoint}:{sorted_params}"

    def _get_rate_limit_key(self, provider: str) -> str:
        """Generate rate limit key for provider"""
        return f"rate_limit:{provider}:{int(time.time() / 60)}"

    def is_market_hours(self) -> bool:
        """
        Check if current time is within market hours (9 AM - 5 PM IST)

        Returns:
            True if within market hours, False otherwise
        """
        if not self.enforce_market_hours:
            return True

        now_ist = datetime.now(IST)
        current_hour = now_ist.hour
        current_day = now_ist.weekday()  # 0 = Monday, 6 = Sunday

        # Check if it's a weekday (Monday-Friday)
        if current_day >= 5:  # Saturday or Sunday
            logger.debug("Outside market hours: Weekend")
            return False

        # Check if it's between 9 AM and 5 PM IST
        if 9 <= current_hour < 17:  # 9 AM to 4:59 PM
            return True

        logger.debug(f"Outside market hours: {current_hour}:00 IST")
        return False

    def check_rate_limit(self, provider: str, skip_market_hours: bool = False) -> bool:
        """
        Check if API call is within rate limit and market hours

        Args:
            provider: API provider name (newsapi, alpha_vantage, etc.)
            skip_market_hours: If True, bypass market hours check (for news, FII/DII)

        Returns:
            True if within limit, False if rate limited
        """
        if not self.enabled:
            return True

        # Check market hours (unless skipped for non-market-dependent data)
        if not skip_market_hours and not self.is_market_hours():
            logger.info(f"API call blocked: Outside market hours (9 AM - 5 PM IST)")
            return False

        try:
            limit = self.RATE_LIMITS.get(provider, 10)
            key = self._get_rate_limit_key(provider)

            # Increment counter
            current = self.redis.incr(key)

            # Set expiry on first request
            if current == 1:
                self.redis.expire(key, 60)  # 1 minute window

            if current > limit:
                logger.warning(f"Rate limit exceeded for {provider}: {current}/{limit} requests/minute")
                return False

            logger.debug(f"Rate limit OK for {provider}: {current}/{limit} requests/minute")
            return True

        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Allow on error

    def get_cached(self, cache_type: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get cached data if available and not expired

        Args:
            cache_type: Type of cache (market_news, trending, etc.)
            params: Request parameters for cache key

        Returns:
            Cached data or None
        """
        if not self.enabled:
            return None

        try:
            key = self._get_cache_key(cache_type, cache_type, params)
            cached = self.redis.get(key)

            if cached:
                data = json.loads(cached)
                logger.info(f"Cache HIT for {cache_type} (params: {params})")
                return data

            logger.debug(f"Cache MISS for {cache_type}")
            return None

        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
            return None

    def set_cache(self, cache_type: str, params: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """
        Cache data with TTL

        Args:
            cache_type: Type of cache
            params: Request parameters for cache key
            data: Data to cache

        Returns:
            True if cached successfully
        """
        if not self.enabled:
            return False

        try:
            key = self._get_cache_key(cache_type, cache_type, params)
            ttl = self.CACHE_TTL.get(cache_type, 300)

            # Add cache metadata
            data_with_meta = {
                **data,
                "_cached_at": datetime.now().isoformat(),
                "_ttl": ttl
            }

            self.redis.setex(
                key,
                ttl,
                json.dumps(data_with_meta)
            )

            logger.debug(f"Cached {cache_type} for {ttl}s")
            return True

        except Exception as e:
            logger.error(f"Cache write failed: {e}")
            return False

    def clear_cache(self, pattern: str = "news_cache:*"):
        """Clear cache by pattern"""
        if not self.enabled:
            return

        try:
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache entries")
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")

    def with_cache_and_rate_limit(
        self,
        provider: str,
        cache_type: str,
        params_extractor: Optional[Callable] = None
    ):
        """
        Decorator to add caching and rate limiting to API functions

        Args:
            provider: API provider name
            cache_type: Cache category
            params_extractor: Function to extract cache params from function args

        Usage:
            @rate_limiter.with_cache_and_rate_limit("newsapi", "market_news")
            async def fetch_news(...):
                ...
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract params for cache key
                if params_extractor:
                    params = params_extractor(*args, **kwargs)
                else:
                    params = {"args": str(args), "kwargs": str(kwargs)}

                # Try cache first
                cached = self.get_cached(cache_type, params)
                if cached:
                    return cached

                # Check rate limit
                if not self.check_rate_limit(provider):
                    logger.warning(f"Rate limited - using stale cache or empty response")
                    # Try to get stale cache (even if expired)
                    # Return empty fallback
                    return {"error": "rate_limited", "message": "Too many requests. Please try again later."}

                # Call actual function
                try:
                    result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)

                    # Cache result
                    if result and not result.get("error"):
                        self.set_cache(cache_type, params, result)

                    return result

                except Exception as e:
                    logger.error(f"API call failed: {e}")
                    # Try stale cache on error
                    return {"error": "api_error", "message": str(e)}

            return wrapper
        return decorator


# Global rate limiter instance
_rate_limiter = None


def get_rate_limiter() -> NewsRateLimiter:
    """Get or create global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = NewsRateLimiter()
    return _rate_limiter


# Exponential backoff for retries
async def exponential_backoff(attempt: int, max_attempts: int = 3, base_delay: float = 1.0):
    """
    Exponential backoff delay

    Args:
        attempt: Current attempt number (0-indexed)
        max_attempts: Maximum retry attempts
        base_delay: Base delay in seconds
    """
    if attempt >= max_attempts:
        raise Exception(f"Max retries ({max_attempts}) exceeded")

    delay = base_delay * (2 ** attempt)
    logger.info(f"Retry {attempt + 1}/{max_attempts} after {delay}s")
    await asyncio.sleep(delay)


async def with_retry(
    func: Callable,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    *args,
    **kwargs
):
    """
    Execute function with exponential backoff retry

    Args:
        func: Async function to execute
        max_attempts: Maximum retry attempts
        base_delay: Base delay for backoff
        *args, **kwargs: Function arguments

    Returns:
        Function result
    """
    for attempt in range(max_attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            await exponential_backoff(attempt, max_attempts, base_delay)
