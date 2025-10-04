"""Performance monitoring and optimization service."""

import asyncio
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

from src.core.config import settings
from src.core.cache import cache
from src.core.monitoring import monitoring_service
from src.core.database import get_db_session
from sqlalchemy import select, text

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for different operations."""
    operation_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_count: int = 0
    success_count: int = 0
    last_updated: float = field(default_factory=time.time)

    def add_measurement(self, duration: float, success: bool = True):
        """Add a performance measurement."""
        self.operation_times.append(duration)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        self.last_updated = time.time()

    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.operation_times:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "error_rate": 0.0
            }

        times = list(self.operation_times)
        return {
            "count": len(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "p95": statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times),
            "p99": statistics.quantiles(times, n=100)[98] if len(times) >= 100 else max(times),
            "error_rate": self.error_count / max(self.success_count + self.error_count, 1)
        }


class QueryOptimizer:
    """Optimizes query performance and caching strategies."""

    def __init__(self):
        self.query_cache = {}
        self.query_patterns = defaultdict(PerformanceMetrics)
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "hit_rate": 0.0
        }

    async def optimize_query(self, query: str, search_type: str = "keyword") -> Dict[str, Any]:
        """Optimize query execution based on historical performance."""
        query_hash = hash(query + search_type)

        # Check if we have cached optimization strategy
        if query_hash in self.query_cache:
            strategy = self.query_cache[query_hash]
            await monitoring_service.track_cache_operation(
                operation="query_optimization_hit",
                cache_type="query_strategy",
                hit=True
            )
            return strategy

        # Analyze query characteristics
        query_length = len(query.split())
        query_complexity = self._assess_query_complexity(query)

        # Determine optimal strategy based on query characteristics
        if query_complexity == "simple" and query_length < 5:
            strategy = {
                "search_type": "keyword",
                "max_results": min(settings.retrieval_k, 3),
                "use_cache": True,
                "cache_ttl": 300,  # 5 minutes for simple queries
                "expected_duration": 0.1
            }
        elif query_complexity == "complex" or query_length > 15:
            strategy = {
                "search_type": "semantic",
                "max_results": settings.retrieval_k,
                "use_cache": True,
                "cache_ttl": 600,  # 10 minutes for complex queries
                "expected_duration": 0.5
            }
        else:
            strategy = {
                "search_type": "hybrid",
                "max_results": settings.retrieval_k,
                "use_cache": True,
                "cache_ttl": 300,
                "expected_duration": 0.3
            }

        # Cache the strategy
        self.query_cache[query_hash] = strategy

        await monitoring_service.track_cache_operation(
            operation="query_optimization_miss",
            cache_type="query_strategy",
            hit=False
        )

        return strategy

    def _assess_query_complexity(self, query: str) -> str:
        """Assess query complexity."""
        query_lower = query.lower()

        # Simple patterns
        simple_indicators = [
            "what is", "who is", "when", "where", "how many",
            "tell me", "explain", "define"
        ]

        # Complex patterns
        complex_indicators = [
            "analyze", "evaluate", "compare", "difference",
            "impact", "effect", "why", "how does"
        ]

        simple_count = sum(1 for indicator in simple_indicators if indicator in query_lower)
        complex_count = sum(1 for indicator in complex_indicators if indicator in query_lower)

        if complex_count > simple_count:
            return "complex"
        elif simple_count > 0:
            return "simple"
        else:
            return "medium"

    async def record_query_performance(self, query: str, search_type: str,
                                     duration: float, success: bool = True):
        """Record query performance for optimization."""
        pattern_key = f"{search_type}_{self._assess_query_complexity(query)}"
        self.query_patterns[pattern_key].add_measurement(duration, success)

    async def get_optimization_stats(self) -> Dict[str, Any]:
        """Get query optimization statistics."""
        return {
            "cache_stats": self.cache_stats,
            "pattern_performance": {
                pattern: metrics.get_stats()
                for pattern, metrics in self.query_patterns.items()
            },
            "cached_strategies": len(self.query_cache)
        }


class CacheOptimizer:
    """Optimizes caching strategies based on access patterns."""

    def __init__(self):
        self.access_patterns = defaultdict(list)
        self.cache_warmup_candidates = set()
        self.last_analysis = time.time()

    async def analyze_access_patterns(self):
        """Analyze cache access patterns to optimize strategies."""
        current_time = time.time()

        # Only analyze every 5 minutes
        if current_time - self.last_analysis < 300:
            return

        self.last_analysis = current_time

        # Analyze access patterns
        for key, accesses in self.access_patterns.items():
            if len(accesses) >= 3:  # At least 3 accesses
                # Calculate access frequency
                time_span = max(accesses[-1] - accesses[0], 1)
                frequency = len(accesses) / time_span

                if frequency > 0.1:  # More than 0.1 accesses per second
                    self.cache_warmup_candidates.add(key)

        # Log analysis results
        await monitoring_service.log_event(
            "cache_analysis_completed",
            {
                "patterns_analyzed": len(self.access_patterns),
                "warmup_candidates": len(self.cache_warmup_candidates)
            }
        )

    async def record_cache_access(self, key: str, hit: bool):
        """Record cache access for pattern analysis."""
        current_time = time.time()
        self.access_patterns[key].append(current_time)

        # Keep only recent accesses (last hour)
        cutoff_time = current_time - 3600
        self.access_patterns[key] = [
            t for t in self.access_patterns[key]
            if t > cutoff_time
        ]

    async def get_cache_recommendations(self) -> Dict[str, Any]:
        """Get cache optimization recommendations."""
        return {
            "warmup_candidates": list(self.cache_warmup_candidates),
            "frequent_access_keys": [
                key for key, accesses in self.access_patterns.items()
                if len(accesses) > 10
            ],
            "analysis_timestamp": self.last_analysis
        }


class DatabaseOptimizer:
    """Optimizes database operations and query performance."""

    def __init__(self):
        self.query_stats = defaultdict(PerformanceMetrics)
        self.connection_pool_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "idle_connections": 0
        }

    async def optimize_database_operations(self):
        """Analyze and optimize database operations."""
        # Check connection pool health
        try:
            async with get_db_session() as session:
                # Test query performance
                start_time = time.time()
                result = await session.execute(text("SELECT 1"))
                result.fetchone()
                query_time = time.time() - start_time

                # Record database operation metrics
                await monitoring_service.track_db_operation(
                    operation="health_check",
                    duration=query_time,
                    success=True
                )

        except Exception as e:
            await monitoring_service.track_db_operation(
                operation="health_check",
                duration=0,
                success=False
            )
            logger.error(f"Database health check failed: {e}")

    async def record_query_execution(self, operation: str, duration: float, success: bool = True):
        """Record database query execution metrics."""
        self.query_stats[operation].add_measurement(duration, success)

        # Update connection pool stats (simplified)
        self.connection_pool_stats["active_connections"] += 1
        if self.connection_pool_stats["active_connections"] > 10:
            # Simulate connection pool management
            self.connection_pool_stats["idle_connections"] = max(
                0, self.connection_pool_stats["idle_connections"] - 1
            )

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database performance statistics."""
        return {
            "query_performance": {
                operation: metrics.get_stats()
                for operation, metrics in self.query_stats.items()
            },
            "connection_pool": self.connection_pool_stats,
            "slow_queries": [
                operation for operation, metrics in self.query_stats.items()
                if metrics.get_stats()["p95"] > 1.0  # Queries slower than 1 second
            ]
        }


class PerformanceMonitor:
    """Main performance monitoring and optimization service."""

    def __init__(self):
        self.query_optimizer = QueryOptimizer()
        self.cache_optimizer = CacheOptimizer()
        self.database_optimizer = DatabaseOptimizer()

        # Performance tracking
        self.start_time = time.time()
        self.is_monitoring = False

    async def start_monitoring(self):
        """Start performance monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True

        # Start background monitoring tasks
        asyncio.create_task(self._periodic_optimization())
        asyncio.create_task(self._database_health_monitor())

        logger.info("Performance monitoring started")

    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        logger.info("Performance monitoring stopped")

    async def _periodic_optimization(self):
        """Run periodic optimization tasks."""
        while self.is_monitoring:
            try:
                # Analyze cache patterns
                await self.cache_optimizer.analyze_access_patterns()

                # Optimize database operations
                await self.database_optimizer.optimize_database_operations()

                # Update system metrics
                await monitoring_service.update_system_metrics()

                # Wait 5 minutes before next optimization run
                await asyncio.sleep(300)

            except Exception as e:
                logger.error(f"Error in periodic optimization: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    async def _database_health_monitor(self):
        """Monitor database health continuously."""
        while self.is_monitoring:
            try:
                await self.database_optimizer.optimize_database_operations()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Database health monitor error: {e}")
                await asyncio.sleep(60)

    async def record_performance_event(self, event_type: str, data: Dict[str, Any]):
        """Record a performance-related event."""
        await monitoring_service.log_event(
            f"performance_{event_type}",
            data
        )

    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            "uptime_seconds": time.time() - self.start_time,
            "monitoring_active": self.is_monitoring,
            "query_optimization": await self.query_optimizer.get_optimization_stats(),
            "cache_optimization": await self.cache_optimizer.get_cache_recommendations(),
            "database_performance": await self.database_optimizer.get_database_stats(),
            "timestamp": time.time()
        }

    async def optimize_query_execution(self, query: str, search_type: str = "keyword") -> Dict[str, Any]:
        """Get optimized execution strategy for a query."""
        return await self.query_optimizer.optimize_query(query, search_type)

    async def record_query_performance(self, query: str, search_type: str,
                                     duration: float, success: bool = True):
        """Record query performance metrics."""
        await self.query_optimizer.record_query_performance(query, search_type, duration, success)

    async def record_cache_access(self, key: str, hit: bool):
        """Record cache access patterns."""
        await self.cache_optimizer.record_cache_access(key, hit)


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
