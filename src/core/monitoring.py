"""Comprehensive monitoring system with metrics, tracing, and structured logging."""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
import json

from prometheus_client import (
    Counter, Histogram, Gauge, Info, Enum,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from fastapi import Request, Response
import structlog
from structlog.contextvars import merge_contextvars
import sentry_sdk
from sentry_sdk.integrations.asyncio import AsyncioIntegration

from .config import settings


@dataclass
class MetricsCollector:
    """Collects and exposes system metrics."""

    def __init__(self):
        # Use a single registry to avoid duplicates
        self.registry = CollectorRegistry()

        # Request metrics
        self.request_count = Counter(
            'fds_requests_total', 'Total number of requests', ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        self.request_duration = Histogram(
            'fds_request_duration_seconds', 'Request duration in seconds', ['method', 'endpoint'],
            registry=self.registry
        )
        self.active_requests = Gauge(
            'fds_active_requests', 'Number of active requests',
            registry=self.registry
        )

        # Database metrics
        self.db_connections = Gauge(
            'fds_db_connections', 'Database connection pool size',
            registry=self.registry
        )
        self.db_query_duration = Histogram(
            'fds_db_query_duration_seconds', 'Database query duration in seconds', ['operation'],
            registry=self.registry
        )
        self.db_connection_errors = Counter(
            'fds_db_connection_errors_total', 'Total database connection errors',
            registry=self.registry
        )

        # Cache metrics
        self.cache_hits = Counter(
            'fds_cache_hits_total', 'Total cache hits', ['cache_type'],
            registry=self.registry
        )
        self.cache_misses = Counter(
            'fds_cache_misses_total', 'Total cache misses', ['cache_type'],
            registry=self.registry
        )
        self.cache_operations = Counter(
            'fds_cache_operations_total', 'Total cache operations', ['operation', 'cache_type'],
            registry=self.registry
        )

        # AI/LLM metrics
        self.llm_requests = Counter(
            'fds_llm_requests_total', 'Total LLM requests', ['model', 'provider'],
            registry=self.registry
        )
        self.llm_tokens = Counter(
            'fds_llm_tokens_total', 'Total tokens processed', ['model', 'direction'],
            registry=self.registry
        )
        self.llm_duration = Histogram(
            'fds_llm_duration_seconds', 'LLM request duration in seconds', ['model', 'provider'],
            registry=self.registry
        )
        self.llm_errors = Counter(
            'fds_llm_errors_total', 'Total LLM errors', ['model', 'error_type'],
            registry=self.registry
        )

        # Document processing metrics
        self.documents_processed = Counter(
            'fds_documents_processed_total', 'Total documents processed', ['status', 'format'],
            registry=self.registry
        )
        self.document_processing_duration = Histogram(
            'fds_document_processing_seconds', 'Document processing duration in seconds',
            registry=self.registry
        )
        self.chunks_created = Counter(
            'fds_chunks_created_total', 'Total chunks created',
            registry=self.registry
        )

        # Embedding metrics
        self.embeddings_generated = Counter(
            'fds_embeddings_generated_total', 'Total embeddings generated',
            registry=self.registry
        )
        self.embedding_duration = Histogram(
            'fds_embedding_duration_seconds', 'Embedding generation duration in seconds',
            registry=self.registry
        )

        # Search/Retrieval metrics
        self.search_requests = Counter(
            'fds_search_requests_total', 'Total search requests', ['search_type'],
            registry=self.registry
        )
        self.search_duration = Histogram(
            'fds_search_duration_seconds', 'Search duration in seconds', ['search_type'],
            registry=self.registry
        )
        self.retrieved_chunks = Counter(
            'fds_retrieved_chunks_total', 'Total chunks retrieved', ['search_type'],
            registry=self.registry
        )

        # System metrics
        self.system_info = Info(
            'fds_system_info', 'System information',
            registry=self.registry
        )
        self.memory_usage = Gauge(
            'fds_memory_usage_bytes', 'Memory usage in bytes',
            registry=self.registry
        )
        self.cpu_usage = Gauge(
            'fds_cpu_usage_percent', 'CPU usage percentage',
            registry=self.registry
        )


class MonitoringService:
    """Comprehensive monitoring service."""

    def __init__(self):
        self.metrics = MetricsCollector()
        self.registry = self.metrics.registry  # Use the registry from MetricsCollector
        self.logger = self._setup_structured_logging()
        self._setup_sentry()
        self._setup_system_info()

    def _setup_structured_logging(self) -> structlog.BoundLogger:
        """Setup structured logging with JSON output."""
        # Configure structlog processors
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ]

        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        return structlog.get_logger()

    def _setup_sentry(self):
        """Setup Sentry error tracking."""
        if settings.enable_sentry and settings.sentry_dsn:
            sentry_sdk.init(
                dsn=settings.sentry_dsn,
                integrations=[AsyncioIntegration()],
                traces_sample_rate=1.0,
                environment="local" if settings.debug else "production"
            )

    def _setup_system_info(self):
        """Setup system information metrics."""
        import platform
        import sys

        system_info = {
            "version": settings.api_version,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.platform(),
            "architecture": platform.machine()
        }

        self.metrics.system_info.info(system_info)

    async def track_request(self, request: Request, response: Response):
        """Track HTTP request metrics."""
        method = request.method
        endpoint = request.url.path
        status_code = response.status_code

        # Record request count and duration
        with self.metrics.request_duration.labels(method=method, endpoint=endpoint).time():
            self.metrics.request_count.labels(
                method=method,
                endpoint=endpoint,
                status=status_code
            ).inc()

        # Track active requests
        self.metrics.active_requests.inc()
        await asyncio.sleep(0)  # Allow other tasks to run
        self.metrics.active_requests.dec()

    async def track_db_operation(self, operation: str, duration: float, success: bool = True):
        """Track database operation metrics."""
        self.metrics.db_query_duration.labels(operation=operation).observe(duration)
        if not success:
            self.metrics.db_connection_errors.inc()

    async def track_cache_operation(self, operation: str, cache_type: str, hit: bool = True):
        """Track cache operation metrics."""
        self.metrics.cache_operations.labels(operation=operation, cache_type=cache_type).inc()

        if hit:
            self.metrics.cache_hits.labels(cache_type=cache_type).inc()
        else:
            self.metrics.cache_misses.labels(cache_type=cache_type).inc()

    async def track_llm_request(self, model: str, provider: str, tokens: int,
                               duration: float, success: bool = True):
        """Track LLM request metrics."""
        self.metrics.llm_requests.labels(model=model, provider=provider).inc()
        self.metrics.llm_duration.labels(model=model, provider=provider).observe(duration)

        if tokens > 0:
            self.metrics.llm_tokens.labels(model=model, direction="input").inc(tokens)

        if not success:
            self.metrics.llm_errors.labels(model=model, error_type="api_error").inc()

    async def track_document_processing(self, status: str, format_type: str, duration: float):
        """Track document processing metrics."""
        self.metrics.documents_processed.labels(status=status, format=format_type).inc()
        self.metrics.document_processing_duration.observe(duration)

    async def track_embedding_generation(self, count: int, duration: float):
        """Track embedding generation metrics."""
        self.metrics.embeddings_generated.inc(count)
        self.metrics.embedding_duration.observe(duration)

    async def track_search_operation(self, search_type: str, duration: float, results_count: int):
        """Track search operation metrics."""
        self.metrics.search_requests.labels(search_type=search_type).inc()
        self.metrics.search_duration.labels(search_type=search_type).observe(duration)
        self.metrics.retrieved_chunks.labels(search_type=search_type).inc(results_count)

    async def update_system_metrics(self):
        """Update system-level metrics."""
        try:
            import psutil
            process = psutil.Process()

            # Memory usage
            memory_info = process.memory_info()
            self.metrics.memory_usage.set(memory_info.rss)

            # CPU usage
            cpu_percent = process.cpu_percent()
            self.metrics.cpu_usage.set(cpu_percent)

        except ImportError:
            self.logger.warning("psutil not available for system metrics")
        except Exception as e:
            self.logger.error("Failed to update system metrics", error=str(e))

    @asynccontextmanager
    async def track_performance(self, operation: str, operation_type: str = "general"):
        """Context manager to track operation performance."""
        start_time = time.time()

        try:
            # Log operation start
            self.logger.info(
                "Operation started",
                operation=operation,
                operation_type=operation_type
            )

            yield

            duration = time.time() - start_time

            # Log successful operation
            self.logger.info(
                "Operation completed",
                operation=operation,
                operation_type=operation_type,
                duration=duration,
                success=True
            )

        except Exception as e:
            duration = time.time() - start_time

            # Log failed operation
            self.logger.error(
                "Operation failed",
                operation=operation,
                operation_type=operation_type,
                duration=duration,
                success=False,
                error=str(e)
            )

            raise

    def get_metrics_text(self) -> str:
        """Get Prometheus metrics as text."""
        return generate_latest(self.registry).decode('utf-8')

    async def log_event(self, event_type: str, data: Dict[str, Any], level: str = "info"):
        """Log structured event."""
        log_data = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            **data
        }

        if level == "info":
            self.logger.info("Event logged", **log_data)
        elif level == "warning":
            self.logger.warning("Event logged", **log_data)
        elif level == "error":
            self.logger.error("Event logged", **log_data)
        elif level == "debug":
            self.logger.debug("Event logged", **log_data)


# Global monitoring service instance
monitoring_service = MonitoringService()
