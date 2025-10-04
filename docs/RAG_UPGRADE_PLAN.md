# RAG System Upgrade Plan

## Overview

This document outlines the comprehensive upgrade plan for the FDS RAG (Retrieval-Augmented Generation) system, including architecture changes, migration strategies, and rollback procedures.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Upgrade Components](#upgrade-components)
3. [Migration Strategy](#migration-strategy)
4. [Rollback Procedures](#rollback-procedures)
5. [Testing Strategy](#testing-strategy)
6. [Deployment Plan](#deployment-plan)

## System Architecture

### Current Architecture
```
Documents → Text Extraction → Chunking → Embeddings → Vector Store → Retrieval → LLM → Response
```

### Upgraded Architecture
```
Documents → OCR/Text Extraction → Normalization → Deduplication → Chunking → 
Dual Embeddings (Passage/Query) → Vector Store (HNSW) → 
Hybrid Retrieval (Semantic + BM25) → Cross-encoder Reranking → MMR → 
Context Packing → LLM → Confidence Calibration → Response
```

### Key Components

#### 1. Document Processing Pipeline
- **OCR Service**: Tesseract-based OCR for scanned PDFs
- **Text Normalization**: Unicode normalization, hyphenation fixes, control character removal
- **Deduplication**: SimHash-based near-duplicate detection
- **Adaptive Chunking**: Token-based, sentence-aware splitting with structural metadata preservation

#### 2. Embedding System
- **Dual Models**: Separate passage and query embedding models
- **Instruction Formatting**: BGE/E5-style instruction prefixes
- **Cache Versioning**: Versioned embedding cache with TTL
- **Batch Processing**: Optimized batch embedding generation

#### 3. Vector Store
- **ChromaDB**: HNSW-indexed vector storage
- **Metadata Indexing**: Enhanced metadata filtering and boosting
- **Backpressure Control**: Retry logic and operation queuing
- **Migration Tools**: Legacy vector ID migration utilities

#### 4. Retrieval System
- **Hybrid Search**: Semantic + BM25 keyword search
- **Reciprocal Rank Fusion**: Combined ranking of search results
- **Cross-encoder Reranking**: Relevance rescoring of top candidates
- **MMR Diversification**: Maximal Marginal Relevance for result diversity

#### 5. Generation System
- **Grounded Prompts**: Citation-enforced response generation
- **Context Packing**: Concise, source-labeled context blocks
- **Confidence Calibration**: No-answer threshold and confidence scoring
- **Refusal Rules**: Explicit handling of low-confidence queries
- **Citation Generation**: Automatic citation extraction from retrieved chunks
- **Numeric Answer Path**: Strict numeric responses for year-based count queries

## Upgrade Components

### ✅ Completed Features

1. **Text Splitting & Normalization**
   - Token-based, sentence-aware splitting
   - Structural metadata preservation (page spans, section headers)
   - Text normalization and cleaning pipeline
   - SimHash-based deduplication

2. **Embedding Model Upgrade**
   - BGE/E5 dual embedding models
   - Instruction formatting for passage/query roles
   - Embedding cache versioning
   - Batch embedding optimization

3. **Vector Store Enhancement**
   - HNSW parameter tuning
   - Metadata indexing improvements
   - Vector migration utilities
   - Backpressure and retry logic

4. **Advanced Retrieval**
   - BM25 keyword search implementation
   - Hybrid search with RRF
   - Cross-encoder reranking
   - MMR result diversification
   - Metadata filtering and boosting

5. **Background Processing**
   - Celery task queue integration
   - Asynchronous document processing
   - Batch embedding operations

6. **Response Generation**
   - Grounded prompt design with citations
   - Context packing with source labels
   - Confidence calibration and no-answer thresholds
   - Refusal rules for low-confidence responses
   - Citation generation from top retrieved chunks
   - Numeric answer enhancement for year-based queries

7. **Performance Optimizations**
   - Query embedding caching with TTL
   - Top-k result caching
   - Vector DB operation backpressure
   - Configurable timeouts and retries

8. **OCR & Table Extraction**
   - Tesseract OCR for scanned PDFs
   - pdfplumber table extraction
   - Automatic scanned PDF detection

9. **Configuration Management**
   - Environment-based configuration
   - Externalized model settings
   - Runtime parameter adjustment

10. **Admin Interface**
    - Reindex/reembed endpoints
    - Cache control operations
    - System monitoring and stats

11. **UI Enhancements**
    - Citation display components
    - Expandable evidence blocks
    - Query highlighting
    - Relevance scoring visualization

12. **Citation System**
    - Automatic citation generation from retrieved chunks
    - Source-backed answers with document metadata
    - Citation display in UI with page numbers and sections
    - Confidence scoring with citation coverage

13. **Numeric Answer Enhancement**
    - Strict numeric-answer path for year-based count queries
    - Structured data integration for precise numeric responses
    - Year-aware numeric extraction from tables and charts
    - Enhanced confidence scoring for numeric answers

### 🔄 Pending Features

1. **RAGAS Evaluation Pipeline**
   - Automated evaluation framework
   - Sample labeled dataset
   - Quality metrics tracking

2. **Metrics & Dashboards**
   - Retrieval performance monitoring
   - Generation quality metrics
   - System health dashboards

## Migration Strategy

### Phase 1: Infrastructure Preparation

#### 1.1 Environment Setup
```bash
# Install new dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with new configuration values
```

#### 1.2 Database Migration
```python
# Run database migrations
alembic upgrade head

# Verify schema changes
python -m src.core.database check_schema
```

#### 1.3 Vector Store Preparation
```bash
# Backup existing vector store
cp -r ./data/vectordb ./data/vectordb.backup

# Initialize new vector store structure
python -m src.services.embedding_service migrate_collection_ids
```

### Phase 2: Service Deployment

#### 2.1 Embedding Service Upgrade
```python
# Update embedding models
python -m src.services.embedding_service update_models

# Verify model loading
python -m src.services.embedding_service test_models
```

#### 2.2 Document Reprocessing
```bash
# Reprocess existing documents with new pipeline
curl -X POST http://localhost:8000/api/v1/admin/reprocess-documents

# Monitor processing status
curl http://localhost:8000/api/v1/admin/stats
```

#### 2.3 Vector Store Migration
```bash
# Migrate existing vectors to new schema
curl -X POST http://localhost:8000/api/v1/admin/migrate-vectors

# Verify migration success
curl http://localhost:8000/api/v1/admin/vector-stats
```

### Phase 3: Feature Activation

#### 3.1 Configuration Updates
```python
# Update runtime configuration
curl -X POST http://localhost:8000/api/v1/admin/config \
  -d '{"embedding_version": 2, "use_cross_encoder": true}'
```

#### 3.2 Cache Warming
```bash
# Warm embedding caches
curl -X POST http://localhost:8000/api/v1/admin/warm-cache

# Verify cache performance
curl http://localhost:8000/api/v1/admin/cache-stats
```

#### 3.3 Service Validation
```bash
# Run system health checks
python -m src.core.monitoring health_check

# Test query processing
python -m src.api.test_query_processing
```

## Rollback Procedures

### Emergency Rollback (Immediate)

#### Step 1: Service Rollback
```bash
# Stop current services
docker-compose down

# Restore previous version
git checkout previous-stable-tag
docker-compose up -d
```

#### Step 2: Database Rollback
```python
# Rollback database migrations
alembic downgrade -1

# Restore database backup if needed
cp ./data/app.db.backup ./data/app.db
```

#### Step 3: Vector Store Rollback
```bash
# Restore vector store backup
rm -rf ./data/vectordb
mv ./data/vectordb.backup ./data/vectordb

# Verify vector store integrity
python -m src.services.embedding_service verify_collection
```

### Gradual Rollback (Controlled)

#### Step 1: Feature Disable
```python
# Disable new features via configuration
curl -X POST http://localhost:8000/api/v1/admin/config \
  -d '{"use_cross_encoder": false, "search_type": "semantic"}'
```

#### Step 2: Service Isolation
```bash
# Route traffic to backup service
# Update load balancer configuration
# Monitor performance metrics
```

#### Step 3: Data Consistency Check
```python
# Verify data integrity
python -m src.core.monitoring data_integrity_check

# Compare response quality
python -m src.testing compare_responses
```

## Testing Strategy

### Unit Tests
```bash
# Run component tests
pytest tests/unit/ -v

# Test specific components
pytest tests/unit/test_embedding_service.py
pytest tests/unit/test_retrieval_service.py
pytest tests/unit/test_document_processor.py
```

### Integration Tests
```bash
# Run integration tests
pytest tests/integration/ -v

# Test end-to-end workflows
pytest tests/integration/test_query_workflow.py
pytest tests/integration/test_document_processing.py
```

### Performance Tests
```bash
# Run performance benchmarks
python -m src.testing.performance benchmark_retrieval

# Load testing
python -m src.testing.performance load_test_api
```

### Quality Assurance
```bash
# Run quality checks
python -m src.testing.quality test_response_quality
python -m src.testing.quality test_citation_accuracy
```

## Deployment Plan

### Pre-Deployment Checklist

- [ ] All unit tests passing
- [ ] Integration tests verified
- [ ] Performance benchmarks met
- [ ] Security scan completed
- [ ] Documentation updated
- [ ] Rollback procedures tested
- [ ] Monitoring alerts configured
- [ ] Backup procedures verified

### Deployment Steps

#### Step 1: Staging Deployment
```bash
# Deploy to staging environment
git checkout upgrade-branch
docker-compose -f docker-compose.staging.yml up -d

# Run staging tests
python -m src.testing.staging run_full_test_suite
```

#### Step 2: Production Deployment
```bash
# Create deployment branch
git checkout -b production-deployment

# Deploy to production
docker-compose up -d

# Monitor deployment
python -m src.core.monitoring deployment_health
```

#### Step 3: Post-Deployment Validation
```bash
# Verify service health
curl http://localhost:8000/api/v1/health

# Test query functionality
curl -X POST http://localhost:8000/api/v1/query \
  -d '{"query": "test query", "max_results": 5}'

# Monitor system metrics
python -m src.core.monitoring system_metrics
```

### Monitoring & Alerting

#### Key Metrics to Monitor
- Query response time
- Embedding generation rate
- Vector store operation latency
- Cache hit rates
- Error rates by component
- Memory and CPU usage

#### Alert Thresholds
- Response time > 5 seconds
- Error rate > 1%
- Memory usage > 80%
- Cache hit rate < 70%

## Configuration Reference

### Environment Variables
```bash
# Embedding Configuration
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_MODEL_PASSAGE=BAAI/bge-small-en-v1.5
EMBEDDING_MODEL_QUERY=BAAI/bge-small-en-v1.5
EMBEDDING_VERSION=2

# Retrieval Configuration
RETRIEVAL_K=5
RERANK_TOP_K=10
USE_CROSS_ENCODER=true
SEARCH_TYPE=hybrid

# Performance Configuration
BATCH_SIZE=32
MAX_CONCURRENT_REQUESTS=100
VECTOR_DB_MAX_RETRIES=3

# OCR Configuration
OCR_ENABLED=true
OCR_DPI=300
TESSERACT_CONFIG=--oem 3 --psm 6
```

## Troubleshooting Guide

### Common Issues

#### 1. Embedding Model Loading Failures
```bash
# Check model availability
python -c "from sentence_transformers import SentenceTransformer; print(SentenceTransformer('BAAI/bge-small-en-v1.5'))"

# Clear model cache
rm -rf ~/.cache/torch/sentence_transformers/
```

#### 2. Vector Store Connection Issues
```bash
# Check ChromaDB status
python -m src.services.embedding_service check_vector_store

# Reset vector store if needed
curl -X POST http://localhost:8000/api/v1/admin/reset-vectors
```

#### 3. Performance Degradation
```bash
# Check system resources
python -m src.core.monitoring system_resources

# Clear caches
curl -X POST http://localhost:8000/api/v1/admin/clear-cache
```

### Support Contacts

- **Technical Lead**: [Contact Information]
- **DevOps Team**: [Contact Information]
- **Emergency Hotline**: [Contact Information]

## Conclusion

This upgrade plan provides a comprehensive framework for migrating to the enhanced RAG system while maintaining system reliability and providing clear rollback procedures. The phased approach ensures minimal disruption to service availability while maximizing the benefits of the new architecture.

For questions or issues during the upgrade process, refer to the troubleshooting guide or contact the support team.
