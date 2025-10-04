"""Enterprise RAG System - Production-ready document question answering system."""

__version__ = "1.0.0"
__author__ = "Enterprise RAG Team"
__description__ = "Enterprise-grade RAG system for document question answering"

# Core imports for easy access
from .core.config import settings
# from .core.models import (
#     DocumentStatus,
#     QueryIntent,
#     DocumentResponse,
#     QueryRequest,
#     QueryResponse,
#     HealthCheck
# )

__all__ = [
    "settings",
    "__version__",
    "__author__",
    "__description__"
]
