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

# Enhanced Retrieval Services
from .enhanced_retrieval_service import EnhancedRetriever

# Multimodal Processing
from .multimodal_document_processor import MultimodalDocumentProcessor
from .table_extraction_service import TableExtractionService
from .chart_extraction_service import ChartExtractionService
from .structured_data_service import StructuredDataService

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
    "SearchType",
    
    # Enhanced Retrieval Services
    "EnhancedRetriever",
    
    # Multimodal Processing
    "MultimodalDocumentProcessor",
    "TableExtractionService",
    "ChartExtractionService",
    "StructuredDataService"
]
