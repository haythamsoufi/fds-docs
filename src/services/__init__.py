"""Services layer for the Enterprise RAG System."""

# Document Processing
from .document_processor import DocumentProcessor, DocumentChangeHandler

# Text Processing
from .text_splitter import (
    AdaptiveTextSplitter,
    SemanticSplitter,
    StructuralSplitter,
    LegalDocumentSplitter,
    TextChunk,
    BaseSplitter
)

# Embedding Services
from .embedding_service import EmbeddingService, VectorStore

# Retrieval Services
from .retrieval_service import (
    HybridRetriever,
    QueryProcessor,
    KeywordSearcher,
    SemanticSearcher,
    SearchResult,
    SearchType
)

__all__ = [
    # Document Processing
    "DocumentProcessor",
    "DocumentChangeHandler",
    
    # Text Processing
    "AdaptiveTextSplitter",
    "SemanticSplitter",
    "StructuralSplitter", 
    "LegalDocumentSplitter",
    "TextChunk",
    "BaseSplitter",
    
    # Embedding Services
    "EmbeddingService",
    "VectorStore",
    
    # Retrieval Services
    "HybridRetriever",
    "QueryProcessor",
    "KeywordSearcher",
    "SemanticSearcher",
    "SearchResult",
    "SearchType"
]
