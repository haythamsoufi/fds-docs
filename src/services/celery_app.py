"""Celery application initialization for background workers."""

from celery import Celery
from src.core.config import settings


def create_celery_app() -> Celery:
    broker_url = settings.redis_url if settings.use_redis else "memory://"
    backend_url = settings.redis_url if settings.use_redis else None

    app = Celery(
        "rag_workers",
        broker=broker_url,
        backend=backend_url,
        include=["src.services.tasks"],
    )

    # Reasonable defaults; can be tuned via env
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
    )
    return app


celery_app = create_celery_app()


