# FDS Docs - Enterprise RAG System

A production-ready, enterprise-grade Retrieval-Augmented Generation (RAG) system for document question answering with advanced features including incremental processing, hybrid retrieval, intelligent caching, and comprehensive monitoring.

## 🚀 Features

### Core Capabilities
- **Multi-format Document Processing**: PDF, DOCX, TXT with OCR support for scanned documents
- **Advanced Text Processing**: Unicode normalization, hyphenation fixes, boilerplate removal
- **Intelligent Chunking**: Token-based, sentence-aware splitting with structural metadata preservation
- **Deduplication**: SimHash-based near-duplicate detection to prevent redundant chunks
- **Hybrid Retrieval**: Combines semantic (vector) and BM25 keyword search with cross-encoder reranking
- **Dual Embedding Models**: Separate passage and query models (BGE/E5) with instruction formatting
- **Confidence Calibration**: No-answer thresholds and calibrated confidence scoring
- **Advanced Caching**: Multi-level caching with Redis for optimal performance
- **Real-time Monitoring**: Comprehensive analytics, health monitoring, and quality assessment

### Enterprise Features
- **Scalable Architecture**: Microservices-based design with async processing and Celery workers
- **Production Ready**: Docker support, proper logging, error handling, and rollback procedures
- **API-First Design**: RESTful API with OpenAPI documentation and comprehensive admin endpoints
- **Database Integration**: PostgreSQL with async SQLAlchemy and database migrations
- **Background Processing**: Celery integration for document processing and embedding tasks
- **Quality Assurance**: RAGAS evaluation pipeline with automated quality metrics
- **Monitoring & Observability**: Prometheus metrics, Grafana dashboards, and real-time alerting
- **Security**: Authentication, rate limiting, input validation, and secure error handling

## 📋 Document Processing Methodologies

### Advanced Text Processing Pipeline

Our RAG system implements state-of-the-art document processing methodologies:

#### 1. **Multi-Format Document Processing**
- **PDF Processing**: PyPDF for text extraction with OCR fallback for scanned documents
- **DOCX Processing**: python-docx for structured document parsing
- **OCR Integration**: Tesseract OCR with configurable DPI and preprocessing
- **Table Extraction**: pdfplumber for reliable table data extraction

#### 2. **Intelligent Text Normalization**
- **Unicode Normalization**: NFC normalization for consistent character encoding
- **Hyphenation Fixes**: Automatic hyphenated word reconstruction
- **Control Character Removal**: Clean text by removing unwanted control characters
- **Boilerplate Removal**: Intelligent detection and removal of headers, footers, page numbers
- **Whitespace Normalization**: Consistent spacing and line break handling

#### 3. **Advanced Chunking Strategies**
- **Token-Based Chunking**: Configurable chunk sizes with token counting
- **Sentence-Aware Splitting**: Preserves sentence boundaries for better context
- **Overlap Management**: Configurable overlap between chunks for context continuity
- **Structural Metadata**: Preserves headers, sections, and document hierarchy
- **Adaptive Chunking**: Different strategies for different document types

#### 4. **Deduplication & Quality Control**
- **SimHash Algorithm**: Near-duplicate detection using locality-sensitive hashing
- **Content Similarity**: Configurable similarity thresholds for duplicate detection
- **Metadata Preservation**: Maintains document structure while removing duplicates
- **Quality Filtering**: Removes low-quality chunks and empty content

#### 5. **Embedding Generation**
- **Dual Embedding Models**: Separate models for passages and queries (BGE/E5)
- **Instruction Formatting**: Proper prompt engineering for retrieval-optimized embeddings
- **Batch Processing**: Optimized embedding generation for large document sets
- **Cache Management**: Versioned embedding cache with TTL management
- **L2 Normalization**: Consistent vector normalization for optimal similarity search

#### 6. **Vector Store Optimization**
- **HNSW Indexing**: Hierarchical Navigable Small World graphs for fast retrieval
- **Metadata Indexing**: Efficient filtering on document metadata
- **Backpressure Control**: Queue management for high-load scenarios
- **Retry Logic**: Exponential backoff for failed operations
- **Operation Monitoring**: Comprehensive metrics for vector store operations

### Retrieval & Generation Methodologies

#### 1. **Hybrid Search Architecture**
- **Semantic Search**: Vector similarity using advanced embedding models
- **BM25 Keyword Search**: Traditional full-text search for exact matches
- **Reciprocal Rank Fusion**: Combines multiple ranking signals optimally
- **Cross-Encoder Reranking**: Re-scores top candidates for improved relevance
- **MMR Diversification**: Maximal Marginal Relevance for result diversity

#### 2. **Confidence & Quality Assessment**
- **No-Answer Thresholds**: Configurable confidence thresholds for response generation
- **Confidence Calibration**: Calibrated confidence scores for reliability assessment
- **RAGAS Evaluation**: Automated quality metrics using industry-standard evaluation framework
- **Performance Monitoring**: Continuous quality assessment and optimization

#### 3. **Context Management**
- **Source-Labeled Context**: Clear attribution of information sources
- **Citation Integration**: Inline citations with expandable evidence blocks
- **Context Packing**: Optimized context assembly for LLM input
- **Refusal Rules**: Intelligent handling of low-confidence scenarios

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FDS RAG System v2.0                         │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (React + TypeScript)                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   Query UI      │ │   Monitoring    │ │   Admin Panel   │   │
│  │   Citations     │ │   Dashboards    │ │   Management    │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  API Layer (FastAPI)                                           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Queries │ │Documents│ │  Admin  │ │Metrics  │ │Evaluation│   │
│  │  API    │ │   API   │ │   API   │ │   API   │ │   API    │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  Core Services Layer                                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │  Document   │ │ Embedding   │ │ Retrieval   │ │ Generation  │ │
│  │ Processor   │ │  Service    │ │  Service    │ │  Service    │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │ OCR Service │ │  Metrics    │ │ Evaluation  │ │ Confidence  │ │
│  │             │ │  Service    │ │  Service    │ │ Calibrator  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Data & Infrastructure Layer                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │   SQLite    │ │  ChromaDB   │ │    Redis    │ │   Celery    │ │
│  │  Database   │ │Vector Store │ │   Cache     │ │   Workers   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  External Dependencies                                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │  OpenAI     │ │  Sentence   │ │  Tesseract  │ │  Prometheus │ │
│  │   API       │ │Transformers │ │    OCR      │ │ + Grafana   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Option 1: Production Mode (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fds-docs
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp env.example .env
   # Edit .env with your settings
   ```

4. **Run the system**:
   ```bash
   python run.py
   ```

5. **Access the system**:
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - React UI: http://localhost:3000 (when running separately)

### Option 2: Docker (Production)

1. **Configure environment**:
   ```bash
   cp env.example .env
   # Edit .env with your settings
   ```

2. **Start with Docker**:
   ```bash
   docker-compose up -d
   ```

3. **Access the system**:
   - Full Application: http://localhost (UI + API)
   - API Direct: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Option 3: Development Mode

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   ```bash
   cp env.example .env
   # Edit .env with your database and Redis URLs
   ```

3. **Run the API directly**:
   ```bash
   python -m src.api.main
   ```

## 📁 Project Structure

```
fds-docs/
├── src/                    # Main application code
│   ├── api/               # FastAPI application
│   │   ├── main.py        # Main API entry point
│   │   └── routes/        # API route modules
│   │       ├── admin.py       # Admin endpoints
│   │       ├── documents.py   # Document management
│   │       ├── queries.py     # Query processing
│   │       ├── evaluation.py  # RAGAS evaluation
│   │       └── metrics.py     # Monitoring metrics
│   ├── core/              # Core components
│   │   ├── config.py      # Configuration management
│   │   ├── database.py    # Database setup
│   │   ├── cache.py       # Caching layer
│   │   ├── models.py      # Data models
│   │   └── monitoring.py  # System monitoring
│   ├── services/          # Business logic services
│   │   ├── document_processor.py    # Document processing pipeline
│   │   ├── embedding_service.py     # Embedding generation
│   │   ├── retrieval_service.py     # Search and retrieval
│   │   ├── ocr_service.py          # OCR processing
│   │   ├── evaluation_service.py   # RAGAS evaluation
│   │   ├── metrics_service.py      # Metrics collection
│   │   ├── confidence_calibrator.py # Confidence scoring
│   │   ├── vector_db_client.py     # Vector store operations
│   │   ├── text_splitter.py        # Text chunking
│   │   ├── celery_app.py           # Background tasks
│   │   └── tasks.py                # Celery task definitions
│   └── web/               # Web interfaces
│       └── streamlit_app.py
├── ui/                    # React frontend
│   ├── src/
│   │   ├── components/    # Reusable UI components
│   │   │   ├── CitationBlock.tsx    # Citation display
│   │   │   ├── EvidenceBlock.tsx    # Evidence visualization
│   │   │   ├── Layout.tsx           # Main layout
│   │   │   └── SystemStatus.tsx     # System status
│   │   ├── pages/         # Application pages
│   │   │   ├── Query.tsx            # Query interface
│   │   │   ├── Monitoring.tsx       # Monitoring dashboard
│   │   │   ├── Documents.tsx        # Document management
│   │   │   ├── Analytics.tsx        # Analytics view
│   │   │   ├── Dashboard.tsx        # Main dashboard
│   │   │   └── Settings.tsx         # Settings panel
│   │   └── services/      # API service layer
│   └── package.json
├── docs/                  # Documentation
│   ├── RAG_UPGRADE_PLAN.md   # Upgrade plan
│   ├── MIGRATION_GUIDE.md    # Migration instructions
│   ├── ROLLBACK_PROCEDURES.md # Rollback procedures
│   └── README.md             # Documentation overview
├── grafana/              # Monitoring setup
│   ├── dashboard.json        # Grafana dashboard config
│   ├── docker-compose.yml    # Monitoring stack
│   ├── prometheus.yml        # Prometheus config
│   └── grafana-datasources.yml # Data source config
├── data/                  # Data storage
│   ├── app.db            # SQLite database
│   ├── documents/        # Document storage
│   ├── vectordb/         # Vector database
│   └── evaluations/      # Evaluation results
├── models/                # LLM models
├── run.py                 # Main entry point
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Docker configuration
└── README.md             # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Database Configuration
DATABASE_URL=sqlite+aiosqlite:///./data/app.db
REDIS_URL=redis://localhost:6379/0
USE_REDIS=false

# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
DEBUG=false

# Document Processing
DOCUMENTS_PATH=./data/documents
VECTOR_DB_PATH=./data/vectordb
MAX_FILE_SIZE=0  # No limit

# Text Processing Configuration
CHUNK_SIZE=500
CHUNK_OVERLAP=100
MAX_CHUNKS_PER_DOCUMENT=1000

# OCR Configuration
OCR_ENABLED=true
OCR_DPI=300
TESSERACT_CONFIG=--oem 3 --psm 6

# Embedding Configuration
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_MODEL_PASSAGE=BAAI/bge-small-en-v1.5
EMBEDDING_MODEL_QUERY=BAAI/bge-small-en-v1.5
EMBEDDING_PASSAGE_INSTRUCTION=Represent this passage for retrieval: 
EMBEDDING_QUERY_INSTRUCTION=Represent this query for retrieval: 
EMBEDDING_VERSION=1
EMBEDDING_DIMENSION=384

# LLM Configuration
OPENAI_API_KEY=your_openai_api_key_here
DEFAULT_LLM_MODEL=gpt-3.5-turbo
MAX_TOKENS=1000
TEMPERATURE=0.7

# Retrieval Configuration
RETRIEVAL_K=5
RERANK_TOP_K=10
SIMILARITY_THRESHOLD=0.3
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
USE_CROSS_ENCODER=true
NO_ANSWER_THRESHOLD=0.3
CONFIDENCE_CALIBRATION_ENABLED=true
MIN_CHUNK_SCORE_THRESHOLD=0.2

# Search Configuration
SEARCH_TYPE=hybrid  # semantic, keyword, hybrid
MMR_LAMBDA_BALANCE=0.7
BM25_K1=1.5
BM25_B=0.75

# Performance & Caching
BATCH_SIZE=32
MAX_CONCURRENT_REQUESTS=100
CACHE_TTL=3600
QUERY_CACHE_TTL=3600
QUERY_EMBEDDING_CACHE_TTL=86400

# Vector DB Configuration
VECTOR_DB_MAX_RETRIES=3
VECTOR_DB_RETRY_DELAY=1.0
VECTOR_DB_BACKPRESSURE_THRESHOLD=100
VECTOR_DB_OPERATION_TIMEOUT=30.0

# Monitoring
LOG_LEVEL=INFO
ENABLE_METRICS=true
```

## 📊 API Endpoints

### Documents
- `GET /api/v1/documents` - List documents
- `GET /api/v1/documents/{id}` - Get document details
- `POST /api/v1/upload` - Upload document
- `DELETE /api/v1/documents/{id}` - Delete document

### Queries
- `POST /api/v1/query` - Search documents with hybrid retrieval
- `GET /api/v1/queries/history` - Get query history

### Admin
- `GET /api/v1/admin/stats` - System statistics
- `POST /api/v1/admin/clear-cache` - Clear cache
- `POST /api/v1/admin/reprocess-documents` - Reprocess documents
- `POST /api/v1/admin/reindex` - Reindex documents with options
- `POST /api/v1/admin/cache-control` - Advanced cache operations

### Evaluation
- `POST /api/v1/evaluation/run` - Run RAGAS evaluation
- `GET /api/v1/evaluation/datasets` - List evaluation datasets
- `GET /api/v1/evaluation/results/{id}` - Get evaluation results

### Metrics
- `GET /api/v1/metrics/prometheus` - Prometheus metrics
- `GET /api/v1/metrics/system-stats` - System statistics

### Health
- `GET /health` - Health check

## 🎯 React UI

The system includes a modern React-based user interface located in the `ui/` directory:

### Features
- **Dashboard**: System overview and metrics
- **Document Management**: Upload, view, and manage documents
- **Query Interface**: Advanced search with hybrid retrieval and citations
- **Monitoring**: Real-time system health and performance dashboards
- **Analytics**: Performance monitoring and usage statistics
- **Settings**: System configuration and management
- **Evidence Visualization**: Expandable evidence blocks with query highlighting

### Setup React UI

1. **Navigate to UI directory**:
   ```bash
   cd ui
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start development server**:
   ```bash
   npm run dev
   ```

4. **Build for production**:
   ```bash
   npm run build
   ```

## 🐳 Docker Deployment

### Production Docker Setup

1. **Build the image**:
   ```bash
   docker build -t fds-docs .
   ```

2. **Run with external services**:
   ```bash
   docker run -d \
     --name fds-docs \
     -p 8000:8000 \
     -e DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db \
     -e REDIS_URL=redis://host:6379/0 \
     -v ./data:/app/data \
     fds-docs
   ```

### Docker Compose

The system includes a complete Docker setup with:
- **PostgreSQL** database
- **Redis** cache
- **FastAPI** backend
- **React UI** frontend
- **Nginx** reverse proxy

Simply run:
```bash
docker-compose up -d
```

## 📈 Monitoring & Observability

### Health Checks
- **API Health**: `GET /health`
- **Database**: Connection status
- **Redis**: Cache status
- **Embeddings**: Model loading status
- **Vector DB**: Operation status and performance

### Prometheus Metrics
- **Query Metrics**: Response times, confidence scores, retrieved chunks
- **System Metrics**: CPU, memory, disk usage, network I/O
- **Cache Metrics**: Hit/miss rates, TTL statistics
- **Vector DB Metrics**: Operation counts, retry statistics, active operations
- **Embedding Metrics**: Generation times, batch sizes
- **Document Processing**: Processing times, success/failure rates

### Grafana Dashboards
- **System Overview**: Real-time system health and resource usage
- **Query Performance**: Response times, throughput, confidence scores
- **Cache Performance**: Hit rates, cache efficiency
- **Vector DB Operations**: Operation latency, retry rates, backpressure
- **Document Processing**: Processing pipeline performance
- **Error Tracking**: Error rates, failure patterns

### Quality Assessment
- **RAGAS Evaluation**: Automated quality metrics for RAG responses
- **Sample Datasets**: Pre-configured evaluation datasets
- **Performance Benchmarking**: Continuous quality monitoring

### Logging
- **Structured Logging**: JSON format with correlation IDs
- **Log Levels**: DEBUG, INFO, WARNING, ERROR
- **Request/Response Logging**: Complete API request tracing
- **Error Tracking**: Comprehensive error logging and alerting

## 🔒 Security

### Authentication
- JWT token-based authentication
- Role-based access control
- API key authentication for services

### Security Features
- Input validation and sanitization
- Rate limiting
- CORS configuration
- SQL injection prevention
- XSS protection

## 🚀 Production Deployment

### Prerequisites
- Python 3.8+
- PostgreSQL 12+
- Redis 6+
- Node.js 16+ (for React UI)

### Deployment Steps

1. **Setup external services**:
   - Configure PostgreSQL database
   - Setup Redis cache
   - Configure reverse proxy (nginx)

2. **Deploy application**:
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Run database migrations
   python -c "import asyncio; from src.core.database import create_tables; asyncio.run(create_tables())"
   
   # Start application
   python run.py
   ```

3. **Configure monitoring**:
   - Setup Prometheus metrics
   - Configure log aggregation
   - Setup alerting

### Performance Optimization

- **Database**: Connection pooling, query optimization, async operations
- **Cache**: Multi-level caching (query embeddings, retrieval results), Redis clustering
- **Embeddings**: Model caching, batch processing, dual embedding models
- **Vector Store**: HNSW indexing, metadata filtering, L2 normalization
- **Retrieval**: Hybrid search (semantic + BM25), cross-encoder reranking, MMR diversification
- **API**: Request batching, response compression, backpressure control
- **Background Processing**: Celery workers for document processing and embedding tasks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Documentation

### Comprehensive Documentation
- **[RAG Upgrade Plan](docs/RAG_UPGRADE_PLAN.md)**: Complete system overview and deployment guide
- **[Migration Guide](docs/MIGRATION_GUIDE.md)**: Step-by-step migration instructions
- **[Rollback Procedures](docs/ROLLBACK_PROCEDURES.md)**: Emergency and gradual rollback options
- **[System Architecture](SYSTEM_ARCHITECTURE.md)**: Detailed technical documentation

### Key Features Documentation
- **Document Processing**: Advanced text processing, OCR, and chunking methodologies
- **Embedding System**: Dual embedding models with instruction formatting
- **Retrieval System**: Hybrid search with cross-encoder reranking
- **Quality Assurance**: RAGAS evaluation and confidence calibration
- **Monitoring**: Prometheus metrics and Grafana dashboards

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the comprehensive documentation in the `docs/` directory
- Review the API documentation at `/docs`
- Consult the system architecture documentation