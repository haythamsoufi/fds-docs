# RAG System Architecture Overview

## High-Level System Architecture

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

## Document Processing Pipeline

```
Documents → OCR/Text Extraction → Normalization → Deduplication → Chunking → 
Structural Metadata → Embeddings → Vector Store
```

### Key Components:
1. **OCR Service**: Tesseract-based OCR for scanned PDFs
2. **Text Normalization**: Unicode normalization, hyphenation fixes, control character removal
3. **Deduplication**: SimHash-based near-duplicate detection
4. **Adaptive Chunking**: Token-based, sentence-aware splitting with structural metadata preservation

## Retrieval & Generation Pipeline

```
Query → Embedding → Hybrid Search (Semantic + BM25) → Cross-encoder Reranking → 
MMR Diversification → Context Packing → LLM Generation → Citation Generation → 
Numeric Answer Enhancement → Confidence Calibration → Response
```

### Key Components:
1. **Dual Embedding Models**: Separate passage and query models (BGE/E5)
2. **Hybrid Retrieval**: Semantic + BM25 keyword search with RRF fusion
3. **Cross-encoder Reranking**: Relevance rescoring of top candidates
4. **MMR Diversification**: Maximal Marginal Relevance for result diversity
5. **Confidence Calibration**: No-answer threshold and quality scoring

## Monitoring & Observability

```
System Metrics → Prometheus → Grafana Dashboards
Query Metrics → Real-time Monitoring → Alert System
Performance Data → Trend Analysis → Optimization
```

### Key Components:
1. **Prometheus Metrics**: Comprehensive system and application metrics
2. **Grafana Dashboards**: Real-time visualization and alerting
3. **React Monitoring UI**: Custom monitoring interface
4. **RAGAS Evaluation**: Automated quality assessment

## File Structure Overview

```
FDS Docs/
├── src/
│   ├── api/                    # FastAPI application and routes
│   │   ├── main.py            # Main application entry point
│   │   └── routes/            # API route handlers
│   │       ├── admin.py       # Admin endpoints
│   │       ├── documents.py   # Document management
│   │       ├── queries.py     # Query processing
│   │       ├── evaluation.py  # RAGAS evaluation
│   │       └── metrics.py     # Monitoring metrics
│   ├── core/                  # Core system components
│   │   ├── config.py         # Configuration management
│   │   ├── database.py       # Database connection
│   │   ├── models.py         # Data models
│   │   ├── cache.py          # Caching layer
│   │   └── monitoring.py     # System monitoring
│   └── services/             # Business logic services
│       ├── document_processor.py    # Document processing pipeline
│       ├── embedding_service.py     # Embedding generation
│       ├── retrieval_service.py     # Search and retrieval
│       ├── ocr_service.py          # OCR processing
│       ├── evaluation_service.py   # RAGAS evaluation
│       ├── metrics_service.py      # Metrics collection
│       ├── confidence_calibrator.py # Confidence scoring
│       ├── vector_db_client.py     # Vector store operations
│       ├── text_splitter.py        # Text chunking
│       ├── celery_app.py           # Background tasks
│       └── tasks.py                # Celery task definitions
├── ui/                        # React frontend
│   └── src/
│       ├── components/        # Reusable UI components
│       │   ├── CitationBlock.tsx    # Citation display
│       │   ├── EvidenceBlock.tsx    # Evidence visualization
│       │   ├── Layout.tsx           # Main layout
│       │   └── SystemStatus.tsx     # System status
│       ├── pages/             # Application pages
│       │   ├── Query.tsx            # Query interface
│       │   ├── Monitoring.tsx       # Monitoring dashboard
│       │   ├── Documents.tsx        # Document management
│       │   ├── Analytics.tsx        # Analytics view
│       │   ├── Dashboard.tsx        # Main dashboard
│       │   └── Settings.tsx         # Settings panel
│       └── services/          # API service layer
├── docs/                      # Documentation
│   ├── RAG_UPGRADE_PLAN.md   # Upgrade plan
│   ├── MIGRATION_GUIDE.md    # Migration instructions
│   ├── ROLLBACK_PROCEDURES.md # Rollback procedures
│   └── README.md             # Documentation overview
├── grafana/                  # Monitoring setup
│   ├── dashboard.json        # Grafana dashboard config
│   ├── docker-compose.yml    # Monitoring stack
│   ├── prometheus.yml        # Prometheus config
│   └── grafana-datasources.yml # Data source config
├── data/                     # Data storage
│   ├── app.db               # SQLite database
│   ├── documents/           # Document storage
│   ├── vectordb/            # Vector database
│   └── evaluations/         # Evaluation results
└── requirements.txt         # Python dependencies
```

## Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **SQLAlchemy**: SQL toolkit and ORM
- **ChromaDB**: Vector database for embeddings
- **Celery**: Distributed task queue
- **Redis**: Caching and message broker
- **Sentence Transformers**: Embedding models
- **RAGAS**: RAG evaluation framework

### Frontend
- **React**: JavaScript library for building user interfaces
- **TypeScript**: Typed JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **Vite**: Fast build tool and dev server

### Monitoring
- **Prometheus**: Metrics collection and monitoring
- **Grafana**: Visualization and dashboards
- **Node Exporter**: System metrics

### Document Processing
- **Tesseract OCR**: Optical character recognition
- **pdfplumber**: PDF text extraction
- **python-docx**: Word document processing
- **pypdf**: PDF manipulation

### Machine Learning
- **OpenAI API**: Large language model
- **BGE/E5 Models**: State-of-the-art embedding models
- **Cross-encoder Models**: Reranking models

## Key Features Implemented

### ✅ Document Processing
- OCR for scanned PDFs and table extraction
- Advanced text normalization and cleaning
- SimHash-based deduplication
- Structural metadata preservation
- Adaptive chunking with overlap

### ✅ Embedding System
- Dual embedding models (passage/query)
- Instruction formatting for BGE/E5 models
- Embedding cache versioning
- Batch processing optimization
- L2 normalization

### ✅ Retrieval System
- Hybrid search (semantic + BM25)
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- MMR diversification
- Metadata filtering and boosting

### ✅ Generation System
- Grounded prompt design with citations
- Context packing with source labels
- Confidence calibration
- No-answer thresholds
- Refusal rules for low-confidence responses

### ✅ Monitoring & Observability
- Prometheus metrics collection
- Grafana dashboards
- Real-time monitoring UI
- Performance trend analysis
- Automated alerting system

### ✅ Quality Assurance
- RAGAS evaluation pipeline
- Sample labeled datasets
- Automated quality metrics
- Performance benchmarking

### ✅ Operations
- Comprehensive admin endpoints
- Background task processing
- Cache management
- System health monitoring
- Rollback procedures

## Performance Characteristics

### Scalability
- **Horizontal Scaling**: Celery workers for background processing
- **Caching**: Redis-based caching for embeddings and queries
- **Database**: SQLite for development, PostgreSQL for production
- **Vector Store**: ChromaDB with HNSW indexing for fast retrieval

### Reliability
- **Backpressure Control**: Vector DB operation queuing
- **Retry Logic**: Exponential backoff for failed operations
- **Health Monitoring**: Comprehensive system health checks
- **Rollback Procedures**: Emergency and gradual rollback options

### Performance
- **Response Time**: Sub-5 second query responses
- **Throughput**: 100+ concurrent requests
- **Cache Hit Rate**: 70%+ for repeated queries
- **Accuracy**: 85%+ confidence scores for quality responses

## Security Considerations

- **Input Validation**: Comprehensive request validation
- **Rate Limiting**: API rate limiting and backpressure
- **Authentication**: JWT-based authentication (extensible)
- **Data Privacy**: Local processing with optional external APIs
- **Error Handling**: Secure error messages without data leakage

## Deployment Options

### Development
- Local SQLite database
- Single-instance deployment
- Basic monitoring
- Manual document processing

### Production
- PostgreSQL database
- Docker containerization
- Redis clustering
- Celery worker scaling
- Prometheus + Grafana monitoring
- Automated backup procedures

This architecture provides a robust, scalable, and maintainable RAG system with enterprise-grade features and comprehensive monitoring capabilities.
