"""Metrics and monitoring routes for RAG system observability."""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Response
from datetime import datetime, timedelta
import json

from src.services.metrics_service import metrics_collector, metrics_middleware

router = APIRouter()


@router.get("/health")
async def get_health_status() -> Dict[str, Any]:
    """Get system health status."""
    try:
        return metrics_collector.get_health_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prometheus")
async def get_prometheus_metrics() -> Response:
    """Get Prometheus metrics in text format."""
    try:
        metrics_text = metrics_collector.export_prometheus_metrics()
        return Response(
            content=metrics_text,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query-stats")
async def get_query_stats(
    time_window: Optional[str] = None
) -> Dict[str, Any]:
    """Get query statistics."""
    try:
        if time_window:
            # Parse time window (e.g., "1h", "30m", "1d")
            time_window_td = _parse_time_window(time_window)
        else:
            time_window_td = None
        
        stats = metrics_collector.get_query_stats(time_window_td)
        return {
            "stats": stats,
            "time_window": time_window,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system-stats")
async def get_system_stats(
    time_window: Optional[str] = None
) -> Dict[str, Any]:
    """Get system statistics."""
    try:
        if time_window:
            time_window_td = _parse_time_window(time_window)
        else:
            time_window_td = None
        
        stats = metrics_collector.get_system_stats(time_window_td)
        return {
            "stats": stats,
            "time_window": time_window,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance-trends")
async def get_performance_trends(
    time_window: str = "1h"
) -> Dict[str, Any]:
    """Get performance trends over time."""
    try:
        time_window_td = _parse_time_window(time_window)
        trends = metrics_collector.get_performance_trends(time_window_td)
        
        return {
            "trends": trends,
            "time_window": time_window,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard-data")
async def get_dashboard_data() -> Dict[str, Any]:
    """Get comprehensive dashboard data."""
    try:
        # Get all metrics
        query_stats = metrics_collector.get_query_stats(timedelta(hours=1))
        system_stats = metrics_collector.get_system_stats(timedelta(hours=1))
        performance_trends = metrics_collector.get_performance_trends(timedelta(hours=1))
        health_status = metrics_collector.get_health_status()
        
        # Calculate derived metrics
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "health": health_status,
            "query_metrics": {
                "current": query_stats,
                "trends": performance_trends.get("trends", [])[-12:],  # Last 12 data points
                "alerts": _check_query_alerts(query_stats)
            },
            "system_metrics": {
                "current": system_stats,
                "alerts": _check_system_alerts(system_stats)
            },
            "performance_summary": {
                "avg_response_time": query_stats.get("avg_response_time", 0),
                "error_rate": query_stats.get("error_rates", {}),
                "cache_hit_rate": query_stats.get("cache_hit_rate", 0),
                "throughput": query_stats.get("total_queries", 0),
                "avg_confidence": query_stats.get("avg_confidence", 0)
            },
            "status": "success"
        }
        
        return dashboard_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_alerts() -> Dict[str, Any]:
    """Get current system alerts."""
    try:
        query_stats = metrics_collector.get_query_stats(timedelta(hours=1))
        system_stats = metrics_collector.get_system_stats(timedelta(hours=1))
        
        alerts = []
        
        # Check query alerts
        query_alerts = _check_query_alerts(query_stats)
        alerts.extend(query_alerts)
        
        # Check system alerts
        system_alerts = _check_system_alerts(system_stats)
        alerts.extend(system_alerts)
        
        return {
            "alerts": alerts,
            "total_alerts": len(alerts),
            "critical_alerts": len([a for a in alerts if a.get("severity") == "critical"]),
            "warning_alerts": len([a for a in alerts if a.get("severity") == "warning"]),
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/record-query")
async def record_query_metrics(
    query: str,
    response_time: float,
    retrieved_chunks: int,
    answer: str,
    confidence_score: float,
    search_type: str = "hybrid",
    embedding_time: float = 0.0,
    retrieval_time: float = 0.0,
    generation_time: float = 0.0,
    reranking_time: Optional[float] = None,
    cache_hit: bool = False,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """Record metrics for a single query."""
    try:
        await metrics_middleware.record_query_metrics(
            query=query,
            response_time=response_time,
            retrieved_chunks=retrieved_chunks,
            answer=answer,
            confidence_score=confidence_score,
            search_type=search_type,
            embedding_time=embedding_time,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            reranking_time=reranking_time,
            cache_hit=cache_hit,
            error=error
        )
        
        return {
            "message": "Query metrics recorded successfully",
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/record-system")
async def record_system_metrics() -> Dict[str, Any]:
    """Record current system metrics."""
    try:
        await metrics_middleware.record_system_metrics()
        
        return {
            "message": "System metrics recorded successfully",
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _parse_time_window(time_window: str) -> timedelta:
    """Parse time window string to timedelta."""
    if time_window.endswith('s'):
        return timedelta(seconds=int(time_window[:-1]))
    elif time_window.endswith('m'):
        return timedelta(minutes=int(time_window[:-1]))
    elif time_window.endswith('h'):
        return timedelta(hours=int(time_window[:-1]))
    elif time_window.endswith('d'):
        return timedelta(days=int(time_window[:-1]))
    else:
        raise ValueError(f"Invalid time window format: {time_window}")


def _check_query_alerts(query_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Check for query-related alerts."""
    alerts = []
    
    if query_stats:
        avg_response_time = query_stats.get("avg_response_time", 0)
        if avg_response_time > 10:  # 10 seconds
            alerts.append({
                "type": "high_response_time",
                "severity": "critical",
                "message": f"Average response time is {avg_response_time:.2f}s (threshold: 10s)",
                "value": avg_response_time,
                "threshold": 10.0
            })
        elif avg_response_time > 5:  # 5 seconds
            alerts.append({
                "type": "high_response_time",
                "severity": "warning",
                "message": f"Average response time is {avg_response_time:.2f}s (threshold: 5s)",
                "value": avg_response_time,
                "threshold": 5.0
            })
        
        error_rates = query_stats.get("error_rates", {})
        for search_type, rate in error_rates.items():
            if rate > 0.1:  # 10% error rate
                alerts.append({
                    "type": "high_error_rate",
                    "severity": "critical",
                    "message": f"Error rate for {search_type} is {rate:.1%} (threshold: 10%)",
                    "value": rate,
                    "threshold": 0.1,
                    "search_type": search_type
                })
            elif rate > 0.05:  # 5% error rate
                alerts.append({
                    "type": "high_error_rate",
                    "severity": "warning",
                    "message": f"Error rate for {search_type} is {rate:.1%} (threshold: 5%)",
                    "value": rate,
                    "threshold": 0.05,
                    "search_type": search_type
                })
        
        avg_confidence = query_stats.get("avg_confidence", 0)
        if avg_confidence < 0.3:  # 30% confidence
            alerts.append({
                "type": "low_confidence",
                "severity": "critical",
                "message": f"Average confidence is {avg_confidence:.1%} (threshold: 30%)",
                "value": avg_confidence,
                "threshold": 0.3
            })
        elif avg_confidence < 0.5:  # 50% confidence
            alerts.append({
                "type": "low_confidence",
                "severity": "warning",
                "message": f"Average confidence is {avg_confidence:.1%} (threshold: 50%)",
                "value": avg_confidence,
                "threshold": 0.5
            })
        
        cache_hit_rate = query_stats.get("cache_hit_rate", 0)
        if cache_hit_rate < 0.5:  # 50% cache hit rate
            alerts.append({
                "type": "low_cache_hit_rate",
                "severity": "warning",
                "message": f"Cache hit rate is {cache_hit_rate:.1%} (threshold: 50%)",
                "value": cache_hit_rate,
                "threshold": 0.5
            })
    
    return alerts


def _check_system_alerts(system_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Check for system-related alerts."""
    alerts = []
    
    if system_stats:
        memory_usage = system_stats.get("avg_memory_usage", 0)
        if memory_usage > 0.9:  # 90% memory usage
            alerts.append({
                "type": "high_memory_usage",
                "severity": "critical",
                "message": f"Memory usage is {memory_usage:.1%} (threshold: 90%)",
                "value": memory_usage,
                "threshold": 0.9
            })
        elif memory_usage > 0.8:  # 80% memory usage
            alerts.append({
                "type": "high_memory_usage",
                "severity": "warning",
                "message": f"Memory usage is {memory_usage:.1%} (threshold: 80%)",
                "value": memory_usage,
                "threshold": 0.8
            })
        
        cpu_usage = system_stats.get("avg_cpu_usage", 0)
        if cpu_usage > 0.9:  # 90% CPU usage
            alerts.append({
                "type": "high_cpu_usage",
                "severity": "critical",
                "message": f"CPU usage is {cpu_usage:.1%} (threshold: 90%)",
                "value": cpu_usage,
                "threshold": 0.9
            })
        elif cpu_usage > 0.8:  # 80% CPU usage
            alerts.append({
                "type": "high_cpu_usage",
                "severity": "warning",
                "message": f"CPU usage is {cpu_usage:.1%} (threshold: 80%)",
                "value": cpu_usage,
                "threshold": 0.8
            })
        
        active_connections = system_stats.get("current_connections", 0)
        if active_connections > 1000:  # 1000 connections
            alerts.append({
                "type": "high_connection_count",
                "severity": "warning",
                "message": f"Active connections: {active_connections} (threshold: 1000)",
                "value": active_connections,
                "threshold": 1000
            })
    
    return alerts
