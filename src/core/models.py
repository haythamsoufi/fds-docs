"""Core data models for the RAG system."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import Column, Integer, String, DateTime, Text, Float, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
import uuid

Base = declarative_base()


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class QueryIntent(str, Enum):
    """Query intent classification."""
    FACTUAL = "factual"
    COMPARISON = "comparison"
    ANALYTICAL = "analytical"
    PROCEDURAL = "procedural"
    UNKNOWN = "unknown"


# SQLAlchemy Models
class DocumentModel(Base):
    """Document metadata model."""
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False, unique=True)
    file_hash = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String, nullable=False)
    status = Column(String, nullable=False, default=DocumentStatus.PENDING)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    # Rename attribute to avoid SQLAlchemy reserved name; keep DB column as 'metadata'
    extra_metadata = Column("metadata", JSON, nullable=True)
    content_preview = Column(Text, nullable=True)
    chunk_count = Column(Integer, default=0)


class ChunkModel(Base):
    """Document chunk model."""
    __tablename__ = "chunks"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String, nullable=False)
    embedding_id = Column(String, nullable=True)  # Vector DB ID
    # Avoid reserved name
    extra_metadata = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class QueryModel(Base):
    """Query analytics model."""
    __tablename__ = "queries"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query_text = Column(Text, nullable=False)
    query_hash = Column(String, nullable=False)
    intent = Column(String, nullable=True)
    response_time = Column(Float, nullable=False)
    retrieved_chunks = Column(Integer, nullable=False)
    user_feedback = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    extra_metadata = Column("metadata", JSON, nullable=True)


# Pydantic Models
class DocumentCreate(BaseModel):
    """Document creation model."""
    filename: str
    filepath: str
    file_hash: str
    file_size: int
    mime_type: str
    metadata: Optional[Dict[str, Any]] = None


class DocumentResponse(BaseModel):
    """Document response model."""
    id: str
    filename: str
    filepath: str
    file_size: int
    mime_type: str
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None
    # Map DB attribute 'extra_metadata' to API field 'metadata'
    metadata: Optional[Dict[str, Any]] = Field(default=None, alias="extra_metadata")
    chunk_count: int = 0
    
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class ChunkCreate(BaseModel):
    """Chunk creation model."""
    document_id: str
    chunk_index: int
    content: str
    content_hash: str
    metadata: Optional[Dict[str, Any]] = None


class ChunkResponse(BaseModel):
    """Chunk response model."""
    id: str
    document_id: str
    chunk_index: int
    content: str
    # Map DB attribute 'extra_metadata' to API field 'metadata'
    metadata: Optional[Dict[str, Any]] = Field(default=None, alias="extra_metadata")
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class QueryRequest(BaseModel):
    """Query request model."""
    query: str
    max_results: int = Field(default=5, ge=1, le=20)
    include_metadata: bool = True
    filters: Optional[Dict[str, Any]] = None


class RetrievedChunk(BaseModel):
    """Retrieved chunk with relevance score."""
    chunk: ChunkResponse
    score: float
    document: DocumentResponse


class QueryResponse(BaseModel):
    """Query response model."""
    query: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    response_time: float
    intent: Optional[QueryIntent] = None
    confidence: Optional[float] = None
    # Optional citations for UI; derived from top retrieved chunks
    citations: Optional[List[Dict[str, Any]]] = None


class ProcessingStatus(BaseModel):
    """Processing status model."""
    total_documents: int
    processed_documents: int
    failed_documents: int
    processing_documents: int
    last_updated: datetime


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    components: Dict[str, str]
