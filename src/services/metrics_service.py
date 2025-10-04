"""Metrics service for RAG system monitoring and observability."""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio

from src.core.config import settings

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available. Install prometheus-client for metrics support.")


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_id: str
    timestamp: datetime
    query_length: int
    response_time: float
    retrieved_chunks: int
    answer_length: int
    confidence_score: float
    search_type: str
    embedding_time: float
    retrieval_time: float
    generation_time: float
    reranking_time: Optional[float] = None
    cache_hit: bool = False
    error: Optional[str] = None


@dataclass
class SystemMetrics:
    """System-level metrics."""
    timestamp: datetime
    active_connections: int
    memory_usage: float
    cpu_usage: float
    disk_usage: float
    vector_db_connections: int
    cache_size: int
    queue_size: int


class MetricsCollector:
    """Collects and stores metrics for the RAG system."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.query_metrics: deque = deque(maxlen=max_history)
        self.system_metrics: deque = deque(maxlen=max_history)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.performance_stats: Dict[str, List[float]] = defaultdict(list)
        
        # Initialize Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        self.registry = CollectorRegistry()
        
        # Query metrics
        self.query_counter = Counter(
            'rag_queries_total',
            'Total number of queries processed',
            ['search_type', 'status'],
            registry=self.registry
        )
        
        self.query_duration = Histogram(
            'rag_query_duration_seconds',
            'Time spent processing queries',
            ['search_type'],
            registry=self.registry
        )
        
        self.retrieval_duration = Histogram(
            'rag_retrieval_duration_seconds',
            'Time spent on retrieval',
            ['search_type'],
            registry=self.registry
        )
        
        self.embedding_duration = Histogram(
            'rag_embedding_duration_seconds',
            'Time spent generating embeddings',
            registry=self.registry
        )
        
        self.generation_duration = Histogram(
            'rag_generation_duration_seconds',
            'Time spent on answer generation',
            registry=self.registry
        )
        
        self.reranking_duration = Histogram(
            'rag_reranking_duration_seconds',
            'Time spent on reranking',
            registry=self.registry
        )
        
        # Quality metrics
        self.confidence_score = Histogram(
            'rag_confidence_score',
            'Distribution of confidence scores',
            ['confidence_level'],
            registry=self.registry
        )
        
        self.retrieved_chunks = Histogram(
            'rag_retrieved_chunks_count',
            'Number of chunks retrieved per query',
            registry=self.registry
        )
        
        self.answer_length = Histogram(
            'rag_answer_length',
            'Length of generated answers',
            registry=self.registry
        )
        
        # System metrics
        self.active_connections = Gauge(
            'rag_active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'rag_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'rag_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.vector_db_connections = Gauge(
            'rag_vector_db_connections',
            'Number of vector database connections',
            registry=self.registry
        )
        
        self.cache_size = Gauge(
            'rag_cache_size',
            'Cache size in bytes',
            registry=self.registry
        )
        
        self.queue_size = Gauge(
            'rag_queue_size',
            'Background queue size',
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'rag_cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'rag_cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        # Error metrics
        self.error_counter = Counter(
            'rag_errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.registry
        )
    
    def record_query_metrics(self, metrics: QueryMetrics):
        """Record metrics for a single query."""
        self.query_metrics.append(metrics)
        
        if PROMETHEUS_AVAILABLE:
            # Update Prometheus metrics
            status = 'error' if metrics.error else 'success'
            self.query_counter.labels(
                search_type=metrics.search_type,
                status=status
            ).inc()
            
            self.query_duration.labels(
                search_type=metrics.search_type
            ).observe(metrics.response_time)
            
            self.retrieval_duration.labels(
                search_type=metrics.search_type
            ).observe(metrics.retrieval_time)
            
            self.embedding_duration.observe(metrics.embedding_time)
            self.generation_duration.observe(metrics.generation_time)
            
            if metrics.reranking_time:
                self.reranking_duration.observe(metrics.reranking_time)
            
            # Confidence score categorization
            if metrics.confidence_score >= 0.8:
                confidence_level = 'high'
            elif metrics.confidence_score >= 0.6:
                confidence_level = 'medium'
            else:
                confidence_level = 'low'
            
            self.confidence_score.labels(
                confidence_level=confidence_level
            ).observe(metrics.confidence_score)
            
            self.retrieved_chunks.observe(metrics.retrieved_chunks)
            self.answer_length.observe(metrics.answer_length)
            
            if metrics.cache_hit:
                self.cache_hits.labels(cache_type='query').inc()
            else:
                self.cache_misses.labels(cache_type='query').inc()
            
            if metrics.error:
                self.error_counter.labels(
                    error_type=metrics.error,
                    component='query'
                ).inc()
    
    def record_system_metrics(self, metrics: SystemMetrics):
        """Record system-level metrics."""
        self.system_metrics.append(metrics)
        
        if PROMETHEUS_AVAILABLE:
            self.active_connections.set(metrics.active_connections)
            self.memory_usage.set(metrics.memory_usage)
            self.cpu_usage.set(metrics.cpu_usage)
            self.vector_db_connections.set(metrics.vector_db_connections)
            self.cache_size.set(metrics.cache_size)
            self.queue_size.set(metrics.queue_size)
    
    def record_cache_metrics(self, cache_type: str, hit: bool):
        """Record cache hit/miss metrics."""
        if PROMETHEUS_AVAILABLE:
            if hit:
                self.cache_hits.labels(cache_type=cache_type).inc()
            else:
                self.cache_misses.labels(cache_type=cache_type).inc()
    
    def record_error(self, error_type: str, component: str):
        """Record error metrics."""
        self.error_counts[f"{component}:{error_type}"] += 1
        
        if PROMETHEUS_AVAILABLE:
            self.error_counter.labels(
                error_type=error_type,
                component=component
            ).inc()
    
    def get_query_stats(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get query statistics for a time window."""
        if time_window:
            cutoff = datetime.now() - time_window
            metrics = [m for m in self.query_metrics if m.timestamp >= cutoff]
        else:
            metrics = list(self.query_metrics)
        
        if not metrics:
            return {}
        
        response_times = [m.response_time for m in metrics]
        confidence_scores = [m.confidence_score for m in metrics]
        retrieved_chunks = [m.retrieved_chunks for m in metrics]
        answer_lengths = [m.answer_length for m in metrics]
        
        # Error rates by search type
        search_type_stats = defaultdict(lambda: {'total': 0, 'errors': 0})
        for m in metrics:
            search_type_stats[m.search_type]['total'] += 1
            if m.error:
                search_type_stats[m.search_type]['errors'] += 1
        
        error_rates = {
            st: stats['errors'] / stats['total'] if stats['total'] > 0 else 0
            for st, stats in search_type_stats.items()
        }
        
        return {
            'total_queries': len(metrics),
            'avg_response_time': sum(response_times) / len(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'avg_confidence': sum(confidence_scores) / len(confidence_scores),
            'avg_retrieved_chunks': sum(retrieved_chunks) / len(retrieved_chunks),
            'avg_answer_length': sum(answer_lengths) / len(answer_lengths),
            'error_rates': error_rates,
            'cache_hit_rate': sum(1 for m in metrics if m.cache_hit) / len(metrics),
            'search_type_distribution': dict(search_type_stats)
        }
    
    def get_system_stats(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get system statistics for a time window."""
        if time_window:
            cutoff = datetime.now() - time_window
            metrics = [m for m in self.system_metrics if m.timestamp >= cutoff]
        else:
            metrics = list(self.system_metrics)
        
        if not metrics:
            return {}
        
        memory_usage = [m.memory_usage for m in metrics]
        cpu_usage = [m.cpu_usage for m in metrics]
        active_connections = [m.active_connections for m in metrics]
        
        return {
            'avg_memory_usage': sum(memory_usage) / len(memory_usage),
            'max_memory_usage': max(memory_usage),
            'avg_cpu_usage': sum(cpu_usage) / len(cpu_usage),
            'max_cpu_usage': max(cpu_usage),
            'avg_active_connections': sum(active_connections) / len(active_connections),
            'max_active_connections': max(active_connections),
            'current_connections': active_connections[-1] if active_connections else 0
        }
    
    def get_performance_trends(self, time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Get performance trends over time."""
        cutoff = datetime.now() - time_window
        metrics = [m for m in self.query_metrics if m.timestamp >= cutoff]
        
        if not metrics:
            return {}
        
        # Group metrics by time intervals (5-minute buckets)
        bucket_size = timedelta(minutes=5)
        buckets = defaultdict(list)
        
        for metric in metrics:
            bucket_time = metric.timestamp.replace(
                minute=(metric.timestamp.minute // 5) * 5,
                second=0,
                microsecond=0
            )
            buckets[bucket_time].append(metric)
        
        trends = []
        for bucket_time in sorted(buckets.keys()):
            bucket_metrics = buckets[bucket_time]
            trends.append({
                'timestamp': bucket_time.isoformat(),
                'query_count': len(bucket_metrics),
                'avg_response_time': sum(m.response_time for m in bucket_metrics) / len(bucket_metrics),
                'avg_confidence': sum(m.confidence_score for m in bucket_metrics) / len(bucket_metrics),
                'error_rate': sum(1 for m in bucket_metrics if m.error) / len(bucket_metrics)
            })
        
        return {
            'time_window': str(time_window),
            'trends': trends
        }
    
    def export_prometheus_metrics(self) -> str:
        """Export Prometheus metrics in text format."""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(self.registry).decode('utf-8')
        else:
            return "# Prometheus client not available\n"
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        recent_metrics = list(self.query_metrics)[-100:] if self.query_metrics else []
        recent_system = list(self.system_metrics)[-10:] if self.system_metrics else []
        
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'prometheus_available': PROMETHEUS_AVAILABLE
        }
        
        if recent_metrics:
            avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
            error_rate = sum(1 for m in recent_metrics if m.error) / len(recent_metrics)
            avg_confidence = sum(m.confidence_score for m in recent_metrics) / len(recent_metrics)
            
            health_status.update({
                'avg_response_time': avg_response_time,
                'error_rate': error_rate,
                'avg_confidence': avg_confidence,
                'recent_queries': len(recent_metrics)
            })
            
            # Health checks
            if avg_response_time > 10:  # 10 second threshold
                health_status['status'] = 'degraded'
                health_status['issues'] = health_status.get('issues', []) + ['high_response_time']
            
            if error_rate > 0.05:  # 5% error rate threshold
                health_status['status'] = 'degraded'
                health_status['issues'] = health_status.get('issues', []) + ['high_error_rate']
            
            if avg_confidence < 0.5:  # Low confidence threshold
                health_status['status'] = 'degraded'
                health_status['issues'] = health_status.get('issues', []) + ['low_confidence']
        
        if recent_system:
            current_memory = recent_system[-1].memory_usage
            current_cpu = recent_system[-1].cpu_usage
            
            health_status.update({
                'memory_usage': current_memory,
                'cpu_usage': current_cpu
            })
            
            if current_memory > 0.9:  # 90% memory threshold
                health_status['status'] = 'degraded'
                health_status['issues'] = health_status.get('issues', []) + ['high_memory_usage']
            
            if current_cpu > 0.9:  # 90% CPU threshold
                health_status['status'] = 'degraded'
                health_status['issues'] = health_status.get('issues', []) + ['high_cpu_usage']
        
        return health_status


# Global metrics collector instance
metrics_collector = MetricsCollector()


class MetricsMiddleware:
    """Middleware for automatically collecting metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
    
    async def record_query_metrics(
        self,
        query: str,
        response_time: float,
        retrieved_chunks: int,
        answer: str,
        confidence_score: float,
        search_type: str,
        embedding_time: float = 0.0,
        retrieval_time: float = 0.0,
        generation_time: float = 0.0,
        reranking_time: Optional[float] = None,
        cache_hit: bool = False,
        error: Optional[str] = None
    ):
        """Record metrics for a query."""
        query_metrics = QueryMetrics(
            query_id=f"query_{int(time.time() * 1000)}",
            timestamp=datetime.now(),
            query_length=len(query),
            response_time=response_time,
            retrieved_chunks=retrieved_chunks,
            answer_length=len(answer),
            confidence_score=confidence_score,
            search_type=search_type,
            embedding_time=embedding_time,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            reranking_time=reranking_time,
            cache_hit=cache_hit,
            error=error
        )
        
        self.metrics_collector.record_query_metrics(query_metrics)
    
    async def record_system_metrics(self):
        """Record current system metrics."""
        try:
            import psutil
            
            system_metrics = SystemMetrics(
                timestamp=datetime.now(),
                active_connections=0,  # Would need to track actual connections
                memory_usage=psutil.virtual_memory().used,
                cpu_usage=psutil.cpu_percent(),
                disk_usage=psutil.disk_usage('/').used,
                vector_db_connections=0,  # Would need to track actual connections
                cache_size=0,  # Would need to get from cache service
                queue_size=0  # Would need to get from Celery/queue
            )
            
            self.metrics_collector.record_system_metrics(system_metrics)
            
        except ImportError:
            logger.warning("psutil not available for system metrics")
        except Exception as e:
            logger.error(f"Error recording system metrics: {e}")


# Global metrics middleware instance
metrics_middleware = MetricsMiddleware(metrics_collector)
