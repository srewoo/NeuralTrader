"""
Redis Caching Layer
Provides fast caching for market data, analysis results, and API responses
"""

import json
import logging
import pickle
import hashlib
from collections import OrderedDict
from typing import Any, Optional, Union, Callable
from datetime import datetime, timedelta
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)

# Try to import redis
try:
    import redis.asyncio as aioredis
    import redis as sync_redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not installed. Install with: pip install redis")


class RedisCache:
    """
    Redis-based caching layer with automatic fallback to in-memory cache

    Features:
    - Async operations for non-blocking I/O
    - Automatic serialization/deserialization
    - TTL-based expiration
    - Namespace support for different cache types
    - Fallback to in-memory cache if Redis is unavailable
    """

    # Maximum entries for in-memory fallback cache (LRU eviction)
    MAX_MEMORY_ENTRIES = 1000

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl: int = 300,  # 5 minutes default
        namespace: str = "neuraltrader",
        use_fallback: bool = True,
        max_memory_entries: int = None
    ):
        """
        Initialize Redis cache

        Args:
            redis_url: Redis connection URL
            default_ttl: Default time-to-live in seconds
            namespace: Prefix for all keys
            use_fallback: Use in-memory fallback if Redis unavailable
            max_memory_entries: Max entries for in-memory cache (LRU eviction)
        """
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.namespace = namespace
        self.use_fallback = use_fallback
        self.max_memory_entries = max_memory_entries or self.MAX_MEMORY_ENTRIES

        self._redis: Optional[Any] = None
        self._sync_redis: Optional[Any] = None
        self._connected = False

        # In-memory fallback cache with LRU eviction (OrderedDict maintains insertion order)
        self._memory_cache: OrderedDict = OrderedDict()
        self._memory_ttls: dict = {}

    async def connect(self) -> bool:
        """Connect to Redis server"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis library not available, using memory fallback")
            return False

        try:
            self._redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,  # We handle encoding ourselves
                socket_timeout=5.0,
                socket_connect_timeout=5.0
            )

            # Test connection
            await self._redis.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.redis_url}")
            return True

        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using memory fallback.")
            self._connected = False
            return False

    def connect_sync(self) -> bool:
        """Connect to Redis synchronously"""
        if not REDIS_AVAILABLE:
            return False

        try:
            self._sync_redis = sync_redis.from_url(
                self.redis_url,
                decode_responses=False,
                socket_timeout=5.0
            )
            self._sync_redis.ping()
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to Redis (sync): {e}")
            return False

    async def disconnect(self):
        """Disconnect from Redis"""
        if self._redis:
            await self._redis.close()
            self._connected = False

    def _make_key(self, key: str) -> str:
        """Create namespaced key"""
        return f"{self.namespace}:{key}"

    def _evict_lru_if_needed(self):
        """Evict least recently used entries if memory cache exceeds limit"""
        while len(self._memory_cache) >= self.max_memory_entries:
            # Remove oldest entry (first item in OrderedDict)
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
            if oldest_key in self._memory_ttls:
                del self._memory_ttls[oldest_key]
            logger.debug(f"LRU evicted: {oldest_key}")

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        try:
            # Try JSON first for common types
            return json.dumps(value, default=str).encode('utf-8')
        except (TypeError, ValueError):
            # Fall back to pickle for complex objects
            return pickle.dumps(value)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize stored value"""
        try:
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return pickle.loads(data)

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        full_key = self._make_key(key)

        if self._connected and self._redis:
            try:
                data = await self._redis.get(full_key)
                if data:
                    return self._deserialize(data)
                return None
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")

        # Fallback to memory cache
        if self.use_fallback and full_key in self._memory_cache:
            # Check TTL
            expiry = self._memory_ttls.get(full_key)
            if expiry and datetime.now() > expiry:
                del self._memory_cache[full_key]
                del self._memory_ttls[full_key]
                return None
            # Move to end (most recently used) for LRU
            self._memory_cache.move_to_end(full_key)
            return self._memory_cache.get(full_key)

        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None uses default)

        Returns:
            True if successful
        """
        full_key = self._make_key(key)
        ttl = ttl or self.default_ttl
        data = self._serialize(value)

        if self._connected and self._redis:
            try:
                await self._redis.set(full_key, data, ex=ttl)
                return True
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")

        # Fallback to memory cache
        if self.use_fallback:
            # Evict LRU entries if cache is full
            self._evict_lru_if_needed()
            self._memory_cache[full_key] = value
            self._memory_ttls[full_key] = datetime.now() + timedelta(seconds=ttl)
            return True

        return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        full_key = self._make_key(key)

        if self._connected and self._redis:
            try:
                await self._redis.delete(full_key)
                return True
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")

        if self.use_fallback and full_key in self._memory_cache:
            del self._memory_cache[full_key]
            if full_key in self._memory_ttls:
                del self._memory_ttls[full_key]
            return True

        return False

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        full_key = self._make_key(key)

        if self._connected and self._redis:
            try:
                return await self._redis.exists(full_key) > 0
            except Exception:
                pass

        if self.use_fallback:
            return full_key in self._memory_cache

        return False

    async def clear_namespace(self, pattern: str = "*") -> int:
        """Clear all keys matching pattern in namespace"""
        full_pattern = self._make_key(pattern)
        count = 0

        if self._connected and self._redis:
            try:
                cursor = 0
                while True:
                    cursor, keys = await self._redis.scan(cursor, match=full_pattern, count=100)
                    if keys:
                        await self._redis.delete(*keys)
                        count += len(keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.warning(f"Redis clear failed: {e}")

        if self.use_fallback:
            keys_to_delete = [k for k in self._memory_cache if k.startswith(self.namespace)]
            for k in keys_to_delete:
                del self._memory_cache[k]
                if k in self._memory_ttls:
                    del self._memory_ttls[k]
                count += 1

        return count

    async def get_stats(self) -> dict:
        """Get cache statistics"""
        stats = {
            "redis_connected": self._connected,
            "redis_url": self.redis_url,
            "namespace": self.namespace,
            "default_ttl": self.default_ttl,
            "memory_cache_size": len(self._memory_cache)
        }

        if self._connected and self._redis:
            try:
                info = await self._redis.info()
                stats.update({
                    "redis_version": info.get("redis_version"),
                    "used_memory": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                    "total_keys": info.get("db0", {}).get("keys", 0) if isinstance(info.get("db0"), dict) else 0
                })
            except Exception as e:
                logger.warning(f"Failed to get Redis stats: {e}")

        return stats


# Cache key generators for different data types
class CacheKeys:
    """Standardized cache key generators"""

    @staticmethod
    def quote(symbol: str) -> str:
        """Cache key for stock quotes"""
        return f"quote:{symbol.upper()}"

    @staticmethod
    def historical(symbol: str, period: str, interval: str) -> str:
        """Cache key for historical data"""
        return f"historical:{symbol.upper()}:{period}:{interval}"

    @staticmethod
    def analysis(symbol: str, model: str) -> str:
        """Cache key for analysis results"""
        h = hashlib.md5(f"{symbol}:{model}".encode()).hexdigest()[:8]
        return f"analysis:{symbol.upper()}:{h}"

    @staticmethod
    def technical(symbol: str) -> str:
        """Cache key for technical indicators"""
        return f"technical:{symbol.upper()}"

    @staticmethod
    def news(symbol: str = None) -> str:
        """Cache key for news"""
        return f"news:{symbol.upper() if symbol else 'market'}"

    @staticmethod
    def screener(filters_hash: str) -> str:
        """Cache key for screener results"""
        return f"screener:{filters_hash}"


# Cache TTL presets for different data types
class CacheTTL:
    """Time-to-live presets in seconds"""

    QUOTE = 60  # 1 minute for live quotes
    HISTORICAL_INTRADAY = 300  # 5 minutes for intraday data
    HISTORICAL_DAILY = 3600  # 1 hour for daily data
    TECHNICAL = 120  # 2 minutes for technical indicators
    ANALYSIS = 1800  # 30 minutes for AI analysis
    NEWS = 600  # 10 minutes for news
    SCREENER = 300  # 5 minutes for screener results
    FUNDAMENTAL = 86400  # 24 hours for fundamental data


def cached(
    key_generator: Callable,
    ttl: int = 300,
    cache_instance: Optional[RedisCache] = None
):
    """
    Decorator for caching async function results

    Args:
        key_generator: Function to generate cache key from function args
        ttl: Time-to-live in seconds
        cache_instance: RedisCache instance (uses global if None)

    Usage:
        @cached(lambda symbol: CacheKeys.quote(symbol), ttl=CacheTTL.QUOTE)
        async def get_quote(symbol: str):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = cache_instance or get_cache()

            # Generate cache key
            try:
                key = key_generator(*args, **kwargs)
            except Exception:
                # If key generation fails, just call the function
                return await func(*args, **kwargs)

            # Try to get from cache
            cached_value = await cache.get(key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {key}")
                return cached_value

            # Call function and cache result
            result = await func(*args, **kwargs)
            if result is not None:
                await cache.set(key, result, ttl)
                logger.debug(f"Cached {key} for {ttl}s")

            return result

        return wrapper
    return decorator


# Global cache instance
_cache_instance: Optional[RedisCache] = None


async def init_cache(
    redis_url: str = "redis://localhost:6379",
    default_ttl: int = 300,
    namespace: str = "neuraltrader"
) -> RedisCache:
    """
    Initialize global cache instance

    Args:
        redis_url: Redis connection URL
        default_ttl: Default TTL
        namespace: Key namespace

    Returns:
        Configured RedisCache instance
    """
    global _cache_instance

    _cache_instance = RedisCache(
        redis_url=redis_url,
        default_ttl=default_ttl,
        namespace=namespace
    )

    await _cache_instance.connect()
    return _cache_instance


def get_cache() -> RedisCache:
    """
    Get global cache instance

    Returns:
        RedisCache instance (creates one with fallback if needed)
    """
    global _cache_instance

    if _cache_instance is None:
        _cache_instance = RedisCache()
        # Note: Not connected to Redis, will use memory fallback

    return _cache_instance


async def shutdown_cache():
    """Shutdown global cache"""
    global _cache_instance

    if _cache_instance:
        await _cache_instance.disconnect()
        _cache_instance = None
