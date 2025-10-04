"""Configuration management for the RAG system."""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    api_title: str = "Enterprise RAG System"
    api_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    debug: bool = False
    
    # Database Configuration
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/app.db",
        env="DATABASE_URL"
    )
    
    # Redis Configuration
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    use_redis: bool = Field(
        default=False,
        env="USE_REDIS"
    )
    
    # Vector Database Configuration
    vector_db_path: str = Field(
        default="./data/vectordb",
        env="VECTOR_DB_PATH"
    )
    
    # Document Processing
    documents_path: str = Field(
        default="./data/documents",
        env="DOCUMENTS_PATH"
    )
    max_file_size: int = Field(default=0, env="MAX_FILE_SIZE")  # 0 = no limit, >0 = max bytes
    supported_formats: List[str] = Field(default=[".pdf", ".docx", ".txt"], env="SUPPORTED_FORMATS")
    # Text Processing Configuration
    chunk_size: int = Field(default=400, env="CHUNK_SIZE")  # Reduced from 500 to 400 for token limits
    chunk_overlap: int = Field(default=80, env="CHUNK_OVERLAP")  # Reduced from 100 to 80
    # OCR Configuration
    ocr_enabled: bool = Field(default=True, env="OCR_ENABLED")
    ocr_dpi: int = Field(default=300, env="OCR_DPI")
    
    # Cloud OCR Configuration (Azure Computer Vision)
    azure_vision_key: Optional[str] = Field(default=None, env="AZURE_VISION_KEY")
    azure_vision_endpoint: Optional[str] = Field(default=None, env="AZURE_VISION_ENDPOINT")
    
    # Cloud OCR Configuration (Google Vision API)
    google_vision_credentials: Optional[str] = Field(default=None, env="GOOGLE_VISION_CREDENTIALS")
    
    # PDF Processing Configuration
    pdf_strict_mode: bool = Field(default=False, env="PDF_STRICT_MODE")  # More lenient PDF parsing
    pdf_fallback_ocr: bool = Field(default=True, env="PDF_FALLBACK_OCR")  # Use OCR for problematic PDFs
    
    # Embedding Configuration
    # Backward-compatible single model name
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        env="EMBEDDING_MODEL"
    )
    # Separate passage and query models (optional). Falls back to embedding_model when not set.
    embedding_model_passage: str = Field(
        default="BAAI/bge-small-en-v1.5",
        env="EMBEDDING_MODEL_PASSAGE"
    )
    embedding_model_query: str = Field(
        default="BAAI/bge-small-en-v1.5",
        env="EMBEDDING_MODEL_QUERY"
    )
    # Instruction prefixes for models that require them (e.g., BGE/E5)
    embedding_passage_instruction: str = Field(
        default="Represent this passage for retrieval: ",
        env="EMBEDDING_PASSAGE_INSTRUCTION"
    )
    embedding_query_instruction: str = Field(
        default="Represent this query for retrieval: ",
        env="EMBEDDING_QUERY_INSTRUCTION"
    )
    # Embedding cache/versioning control
    embedding_version: int = Field(
        default=1,
        env="EMBEDDING_VERSION"
    )
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    default_llm_model: str = Field(default="gpt-3.5-turbo", env="DEFAULT_LLM_MODEL")
    max_tokens: int = Field(default=1000, env="MAX_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    # Local LLM (OpenAI-compatible) Configuration
    use_local_llm: bool = Field(default=False, env="USE_LOCAL_LLM")
    local_llm_base_url: Optional[str] = Field(default=None, env="LOCAL_LLM_BASE_URL")
    local_llm_model: Optional[str] = Field(default=None, env="LOCAL_LLM_MODEL")
    # LLM Response Configuration
    # 0 = unlimited (do not send max_tokens to the API)
    llm_response_max_tokens: int = Field(default=0, env="LLM_RESPONSE_MAX_TOKENS")
    llm_temperature: float = Field(default=0.1, env="LLM_TEMPERATURE")
    llm_timeout: float = Field(default=0.0, env="LLM_TIMEOUT")
    
    # Token Management Configuration
    max_input_tokens: int = Field(default=512, env="MAX_INPUT_TOKENS")  # Model input token limit
    max_context_tokens: int = Field(default=400, env="MAX_CONTEXT_TOKENS")  # Context window for processing
    
    # Retrieval Configuration
    retrieval_k: int = Field(default=100, env="RETRIEVAL_K")
    rerank_top_k: int = Field(default=32, env="RERANK_TOP_K")
    similarity_threshold: float = Field(default=0.0, env="SIMILARITY_THRESHOLD")
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        env="CROSS_ENCODER_MODEL"
    )
    # Enable cross-encoder reranking for higher-quality results (can be disabled via env)
    use_cross_encoder: bool = Field(default=True, env="USE_CROSS_ENCODER")
    # Confidence and no-answer thresholds
    no_answer_threshold: float = Field(default=0.3, env="NO_ANSWER_THRESHOLD")
    confidence_calibration_enabled: bool = Field(default=True, env="CONFIDENCE_CALIBRATION_ENABLED")
    min_chunk_score_threshold: float = Field(default=0.2, env="MIN_CHUNK_SCORE_THRESHOLD")
    # Search Configuration
    search_type: str = Field(default="hybrid", env="SEARCH_TYPE")  # semantic, keyword, hybrid
    mmr_lambda_balance: float = Field(default=0.7, env="MMR_LAMBDA_BALANCE")
    bm25_k1: float = Field(default=1.5, env="BM25_K1")
    bm25_b: float = Field(default=0.75, env="BM25_B")

    # Performance
    batch_size: int = Field(
        default=32,
        env="BATCH_SIZE"
    )
    cache_ttl: int = Field(
        default=3600,
        env="CACHE_TTL"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL"
    )
    
    # Performance Configuration
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    max_concurrent_requests: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    # Query caching configuration
    query_cache_ttl: int = Field(default=1800, env="QUERY_CACHE_TTL")  # 30 minutes
    query_embedding_cache_ttl: int = Field(default=86400, env="QUERY_EMBEDDING_CACHE_TTL")  # 24 hours
    # Vector DB backpressure and retry configuration
    vector_db_max_retries: int = Field(default=3, env="VECTOR_DB_MAX_RETRIES")
    vector_db_retry_delay: float = Field(default=1.0, env="VECTOR_DB_RETRY_DELAY")
    vector_db_backpressure_threshold: int = Field(default=100, env="VECTOR_DB_BACKPRESSURE_THRESHOLD")
    vector_db_operation_timeout: float = Field(default=10.0, env="VECTOR_DB_OPERATION_TIMEOUT")
    
    # Monitoring
    enable_metrics: bool = True
    log_level: str = "INFO"
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    enable_sentry: bool = Field(default=False, env="ENABLE_SENTRY")
    metrics_port: int = 9090
    enable_tracing: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables


# Global settings instance
settings = Settings()
