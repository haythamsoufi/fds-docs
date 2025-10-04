"""Redis cache implementation."""

import json
import pickle
from typing import Any, Optional, Union
import logging
from datetime import timedelta
import warnings

from .config import settings

logger = logging.getLogger(__name__)

# Suppress cryptography deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="cryptography")

# Try to import Redis and check compatibility
try:
    import redis
    REDIS_AVAILABLE = True
    logger.info("Redis libraries available")
except ImportError as e:
    REDIS_AVAILABLE = False
    logger.info(f"Redis not available: {e}. Using in-memory cache fallback.")


class CacheManager:
    """Cache manager with Redis support and in-memory fallback."""
    
    def __init__(self):
        self.redis: Optional[Any] = None
        self._memory_cache: dict = {}
        
    async def connect(self):
        """Connect to cache (Redis or in-memory fallback)."""
        if not settings.use_redis:
            logger.info("Redis disabled, using in-memory cache fallback")
            self._memory_cache = {}
            return

        if not REDIS_AVAILABLE:
            logger.info("Redis not available, using in-memory cache fallback")
            self._memory_cache = {}
            return

        try:
            # Import aioredis only when needed
            import aioredis
            # Try to connect to Redis
            self.redis = aioredis.from_url(settings.redis_url)
            # Test the connection
            await self.redis.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using in-memory cache fallback.")
            self.redis = None
            self._memory_cache = {}
            
    async def disconnect(self):
        """Disconnect from cache."""
        if self.redis:
            await self.redis.close()
            logger.info("Disconnected from Redis")
        else:
            self._memory_cache.clear()
            logger.info("Cleared in-memory cache")
            
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not REDIS_AVAILABLE or not self.redis:
            return self._memory_cache.get(key)
            
        try:
            value = await self.redis.get(key)
            if value is None:
                return None
            return pickle.loads(value)
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            # Fallback to memory cache
            return self._memory_cache.get(key)
            
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """Set value in cache."""
        if not REDIS_AVAILABLE or not self.redis:
            # Simple in-memory cache without TTL for now
            self._memory_cache[key] = value
            return True
            
        try:
            serialized_value = pickle.dumps(value)
            if ttl is None:
                ttl = settings.cache_ttl
            
            await self.redis.set(key, serialized_value, ex=ttl)
            return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            # Fallback to memory cache
            self._memory_cache[key] = value
            return True
            
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not REDIS_AVAILABLE or not self.redis:
            if key in self._memory_cache:
                del self._memory_cache[key]
                return True
            return False
            
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            # Fallback to memory cache
            if key in self._memory_cache:
                del self._memory_cache[key]
                return True
            return False
            
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not REDIS_AVAILABLE or not self.redis:
            return key in self._memory_cache
            
        try:
            return bool(await self.redis.exists(key))
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            # Fallback to memory cache
            return key in self._memory_cache
            
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern."""
        if not REDIS_AVAILABLE:
            # Simple pattern matching for in-memory cache
            import fnmatch
            keys_to_delete = [key for key in self._memory_cache.keys() if fnmatch.fnmatch(key, pattern)]
            for key in keys_to_delete:
                del self._memory_cache[key]
            return len(keys_to_delete)
            
        if not self.redis:
            return 0
            
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                return await self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache clear pattern error for pattern {pattern}: {e}")
            return 0
            
    def generate_key(self, prefix: str, *args) -> str:
        """Generate cache key."""
        key_parts = [prefix] + [str(arg) for arg in args]
        return ":".join(key_parts)


# Global cache instance
cache = CacheManager()
