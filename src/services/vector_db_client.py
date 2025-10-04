"""Vector database client with backpressure and retry logic."""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Callable, TypeVar, Union
from dataclasses import dataclass
from enum import Enum
import functools

from src.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar('T')


class OperationType(str, Enum):
    """Vector DB operation types for backpressure tracking."""
    SEARCH = "search"
    ADD = "add"
    UPSERT = "upsert"
    DELETE = "delete"
    COUNT = "count"


@dataclass
class OperationStats:
    """Statistics for vector DB operations."""
    operation_type: OperationType
    start_time: float
    end_time: Optional[float] = None
    retry_count: int = 0
    success: bool = False
    error: Optional[str] = None


class VectorDBBackpressureManager:
    """Manages backpressure and retry logic for vector database operations."""

    def __init__(self):
        self.active_operations: Dict[OperationType, int] = {
            op_type: 0 for op_type in OperationType
        }
        self.operation_history: List[OperationStats] = []
        self.max_retries = settings.vector_db_max_retries
        self.retry_delay = settings.vector_db_retry_delay
        # Reduce backpressure threshold to prevent overwhelming the system
        # Lower threshold for DELETE operations to prevent timeouts
        self.backpressure_threshold = min(settings.vector_db_backpressure_threshold, 5)
        self.operation_timeout = settings.vector_db_operation_timeout
        # Add timeout for backpressure waiting to prevent infinite loops
        self.backpressure_wait_timeout = 60.0  # 60 seconds max wait

    def _check_backpressure(self, operation_type: OperationType) -> bool:
        """Check if we should apply backpressure for this operation type."""
        total_active = sum(self.active_operations.values())
        return total_active >= self.backpressure_threshold

    async def _wait_for_backpressure(self, operation_type: OperationType):
        """Wait if backpressure is needed with timeout."""
        start_time = time.time()
        while self._check_backpressure(operation_type):
            elapsed = time.time() - start_time
            if elapsed > self.backpressure_wait_timeout:
                logger.error(f"Backpressure wait timeout for {operation_type} after {elapsed:.1f}s, proceeding anyway")
                break
            logger.warning(f"Backpressure applied for {operation_type}, waiting... (elapsed: {elapsed:.1f}s)")
            await asyncio.sleep(0.5)  # Longer wait to reduce log spam

    def _record_operation_start(self, operation_type: OperationType) -> OperationStats:
        """Record the start of an operation."""
        stats = OperationStats(
            operation_type=operation_type,
            start_time=time.time()
        )
        self.active_operations[operation_type] += 1
        return stats

    def _record_operation_end(self, stats: OperationStats, success: bool, error: str = None):
        """Record the end of an operation."""
        stats.end_time = time.time()
        stats.success = success
        stats.error = error
        self.active_operations[stats.operation_type] -= 1
        
        # Keep only recent history (last 1000 operations)
        self.operation_history.append(stats)
        if len(self.operation_history) > 1000:
            self.operation_history = self.operation_history[-1000:]

    async def execute_with_retry(
        self,
        operation_type: OperationType,
        operation: Callable[[], Any],
        *args,
        **kwargs
    ) -> Any:
        """Execute an operation with retry logic and backpressure management."""
        
        # Check backpressure
        await self._wait_for_backpressure(operation_type)
        
        stats = self._record_operation_start(operation_type)
        
        for attempt in range(self.max_retries + 1):
            try:
                # Add timeout to the operation
                result = await asyncio.wait_for(
                    operation(*args, **kwargs),
                    timeout=self.operation_timeout
                )
                
                self._record_operation_end(stats, success=True)
                logger.debug(f"Vector DB {operation_type} succeeded on attempt {attempt + 1}")
                return result
                
            except asyncio.TimeoutError:
                error_msg = f"Vector DB {operation_type} timed out after {self.operation_timeout}s"
                logger.warning(error_msg)
                # Always record operation end on timeout to free up the slot
                self._record_operation_end(stats, success=False, error=error_msg)
                if attempt == self.max_retries:
                    raise
                    
            except Exception as e:
                error_msg = f"Vector DB {operation_type} failed: {str(e)}"
                logger.warning(f"{error_msg} (attempt {attempt + 1}/{self.max_retries + 1})")
                
                if attempt == self.max_retries:
                    # Always record operation end on final failure
                    self._record_operation_end(stats, success=False, error=error_msg)
                    raise
                
                # Exponential backoff with jitter
                delay = self.retry_delay * (2 ** attempt) + (time.time() % 1)
                await asyncio.sleep(delay)
        
        # Should not reach here
        self._record_operation_end(stats, success=False, error="Max retries exceeded")
        raise RuntimeError("Max retries exceeded")

    def get_operation_stats(self) -> Dict[str, Any]:
        """Get current operation statistics."""
        recent_ops = [op for op in self.operation_history if op.end_time and (time.time() - op.end_time) < 300]  # Last 5 minutes
        
        stats = {
            "active_operations": dict(self.active_operations),
            "total_active": sum(self.active_operations.values()),
            "recent_operations": len(recent_ops),
            "recent_success_rate": 0.0,
            "average_duration": 0.0,
            "backpressure_applied": sum(self.active_operations.values()) >= self.backpressure_threshold
        }
        
        if recent_ops:
            successful_ops = [op for op in recent_ops if op.success]
            stats["recent_success_rate"] = len(successful_ops) / len(recent_ops)
            
            durations = [op.end_time - op.start_time for op in recent_ops if op.end_time]
            if durations:
                stats["average_duration"] = sum(durations) / len(durations)
        
        return stats

    def reset_backpressure_state(self):
        """Reset backpressure state in case of stuck operations."""
        logger.warning("Resetting backpressure state due to stuck operations")
        self.active_operations = {op_type: 0 for op_type in OperationType}
        # Keep recent history but clear very old entries
        cutoff_time = time.time() - 300  # 5 minutes ago
        self.operation_history = [
            op for op in self.operation_history 
            if op.start_time > cutoff_time
        ]


def with_retry_and_backpressure(operation_type: OperationType):
    """Decorator to add retry logic and backpressure to vector DB operations."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Get backpressure manager from the first argument (self)
            if args and hasattr(args[0], '_backpressure_manager'):
                backpressure_manager = args[0]._backpressure_manager
            else:
                # Fallback: create a new manager
                backpressure_manager = VectorDBBackpressureManager()
            
            return await backpressure_manager.execute_with_retry(
                operation_type,
                func,
                *args,
                **kwargs
            )
        return wrapper
    return decorator


class ResilientVectorStore:
    """Vector store wrapper with built-in retry and backpressure logic."""
    
    def __init__(self, vector_store):
        self._vector_store = vector_store
        self._backpressure_manager = VectorDBBackpressureManager()

    @with_retry_and_backpressure(OperationType.SEARCH)
    async def search(self, *args, **kwargs):
        """Search with retry and backpressure."""
        return await self._vector_store._search_impl(*args, **kwargs)

    @with_retry_and_backpressure(OperationType.ADD)
    async def add_documents(self, *args, **kwargs):
        """Add documents with retry and backpressure."""
        return await self._vector_store._add_documents_impl(*args, **kwargs)

    @with_retry_and_backpressure(OperationType.DELETE)
    async def delete_document(self, *args, **kwargs):
        """Delete document with retry and backpressure."""
        return await self._vector_store._delete_document_impl(*args, **kwargs)

    async def get_operation_stats(self) -> Dict[str, Any]:
        """Get operation statistics."""
        return self._backpressure_manager.get_operation_stats()

    def reset_backpressure_state(self):
        """Reset backpressure state in case of stuck operations."""
        self._backpressure_manager.reset_backpressure_state()

    def __getattr__(self, name):
        """Delegate other attributes to the underlying vector store."""
        return getattr(self._vector_store, name)


# Global backpressure manager instance
backpressure_manager = VectorDBBackpressureManager()
