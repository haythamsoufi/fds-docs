"""API layer for the Enterprise RAG System."""

# Main FastAPI application
from .main import app

# Route modules
from .routes import documents, queries, admin

__all__ = [
    "app",
    "documents",
    "queries", 
    "admin"
]
