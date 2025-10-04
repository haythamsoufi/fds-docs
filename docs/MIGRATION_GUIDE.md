# RAG System Migration Guide

## Overview

This guide provides step-by-step instructions for migrating from the legacy RAG system to the enhanced version with advanced retrieval, dual embeddings, and improved response generation.

## Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Git
- Access to the production environment
- Backup of current system state

## Pre-Migration Checklist

### 1. System Backup
```bash
# Create system backup
mkdir -p backups/$(date +%Y%m%d_%H%M%S)
cp -r data/ backups/$(date +%Y%m%d_%H%M%S)/data/
cp app.db backups/$(date +%Y%m%d_%H%M%S)/app.db

# Backup configuration
cp .env backups/$(date +%Y%m%d_%H%M%S)/.env
cp docker-compose.yml backups/$(date +%Y%m%d_%H%M%S)/docker-compose.yml
```

### 2. Environment Preparation
```bash
# Update environment configuration
cp .env.example .env.new
# Edit .env.new with new configuration values

# Install new dependencies
pip install -r requirements.txt
```

### 3. Service Dependencies
```bash
# Install OCR dependencies (if not already installed)
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install poppler-utils

# Verify installations
tesseract --version
pdftoppm -h
```

## Migration Steps

### Step 1: Database Schema Migration

#### 1.1 Run Database Migrations
```bash
# Apply database schema changes
alembic upgrade head

# Verify schema updates
python -c "
from src.core.database import engine
from sqlalchemy import inspect
inspector = inspect(engine)
print('Tables:', inspector.get_table_names())
"
```

#### 1.2 Data Integrity Check
```python
# Verify data integrity
python -c "
from src.core.database import get_db_session
from src.core.models import DocumentModel, ChunkModel
import asyncio

async def check_data():
    async with get_db_session() as db:
        from sqlalchemy import select, func
        
        # Count documents
        doc_count = await db.execute(select(func.count(DocumentModel.id)))
        print(f'Documents: {doc_count.scalar()}')
        
        # Count chunks
        chunk_count = await db.execute(select(func.count(ChunkModel.id)))
        print(f'Chunks: {chunk_count.scalar()}')

asyncio.run(check_data())
"
```

### Step 2: Embedding Service Migration

#### 2.1 Update Embedding Models
```python
# Update embedding service configuration
python -c "
from src.services.embedding_service import EmbeddingService
from src.core.config import settings

# Initialize new embedding service
embedding_service = EmbeddingService()

# Test model loading
print('Passage model:', embedding_service.passage_model_name)
print('Query model:', embedding_service.query_model_name)
print('Dimension:', embedding_service.dimension)
"
```

#### 2.2 Migrate Existing Embeddings
```bash
# Run vector migration
curl -X POST http://localhost:8000/api/v1/admin/migrate-vectors \
  -H "Content-Type: application/json"

# Monitor migration progress
curl http://localhost:8000/api/v1/admin/stats
```

### Step 3: Document Reprocessing

#### 3.1 Reprocess Documents with New Pipeline
```bash
# Trigger document reprocessing
curl -X POST http://localhost:8000/api/v1/admin/reprocess-documents \
  -H "Content-Type: application/json"

# Monitor processing status
watch -n 5 'curl -s http://localhost:8000/api/v1/admin/stats | jq'
```

#### 3.2 Verify Processing Results
```python
# Check processing results
python -c "
from src.core.database import get_db_session
from src.core.models import DocumentModel, ChunkModel
from sqlalchemy import select, func
import asyncio

async def check_processing():
    async with get_db_session() as db:
        # Check document status
        result = await db.execute(
            select(DocumentModel.status, func.count(DocumentModel.id))
            .group_by(DocumentModel.status)
        )
        for status, count in result:
            print(f'{status}: {count}')
        
        # Check chunks with embeddings
        chunk_result = await db.execute(
            select(func.count(ChunkModel.id)).where(ChunkModel.embedding_id.isnot(None))
        )
        total_chunks = await db.execute(select(func.count(ChunkModel.id)))
        print(f'Chunks with embeddings: {chunk_result.scalar()}/{total_chunks.scalar()}')

asyncio.run(check_processing())
"
```

### Step 4: Vector Store Migration

#### 4.1 Initialize New Vector Store
```python
# Initialize vector store with new configuration
python -c "
from src.services.embedding_service import EmbeddingService, VectorStore

# Create embedding service
embedding_service = EmbeddingService()
vector_store = VectorStore(embedding_service)

# Test vector store operations
print('Vector store initialized successfully')
"
```

#### 4.2 Populate Vector Store
```bash
# Populate vector store with new embeddings
curl -X POST http://localhost:8000/api/v1/admin/populate-vectors \
  -H "Content-Type: application/json"

# Verify vector store population
curl http://localhost:8000/api/v1/admin/vector-stats
```

### Step 5: Configuration Migration

#### 5.1 Update Runtime Configuration
```python
# Update configuration settings
python -c "
from src.core.config import settings

# Verify new configuration
print('Embedding model:', settings.embedding_model)
print('Embedding version:', settings.embedding_version)
print('Search type:', settings.search_type)
print('Cross-encoder enabled:', settings.use_cross_encoder)
print('OCR enabled:', settings.ocr_enabled)
"
```

#### 5.2 Cache Configuration
```bash
# Clear old caches
curl -X POST http://localhost:8000/api/v1/admin/clear-cache

# Warm new caches
curl -X POST http://localhost:8000/api/v1/admin/warm-cache
```

### Step 6: Service Validation

#### 6.1 Health Checks
```bash
# Run comprehensive health checks
python -c "
import requests
import json

# Test API endpoints
endpoints = [
    'http://localhost:8000/api/v1/health',
    'http://localhost:8000/api/v1/admin/stats',
    'http://localhost:8000/api/v1/documents'
]

for endpoint in endpoints:
    try:
        response = requests.get(endpoint)
        print(f'{endpoint}: {response.status_code}')
    except Exception as e:
        print(f'{endpoint}: ERROR - {e}')
"
```

#### 6.2 Query Testing
```bash
# Test query functionality
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the company policy on remote work?",
    "max_results": 5,
    "filters": {
      "search_type": "hybrid"
    }
  }'
```

### Step 7: Performance Validation

#### 7.1 Benchmark Tests
```bash
# Run performance benchmarks
python -m src.testing.performance benchmark_retrieval

# Test response times
python -c "
import time
import requests

def test_query_performance():
    query = 'test query'
    times = []
    
    for i in range(10):
        start = time.time()
        response = requests.post('http://localhost:8000/api/v1/query', 
                               json={'query': query, 'max_results': 5})
        end = time.time()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    print(f'Average response time: {avg_time:.2f}s')
    print(f'Min: {min(times):.2f}s, Max: {max(times):.2f}s')

test_query_performance()
"
```

#### 7.2 Memory and Resource Monitoring
```bash
# Monitor system resources
python -c "
import psutil
import time

def monitor_resources():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    print(f'CPU Usage: {cpu_percent}%')
    print(f'Memory Usage: {memory.percent}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)')

monitor_resources()
"
```

## Post-Migration Validation

### 1. Functional Testing
```bash
# Run comprehensive functional tests
pytest tests/integration/test_query_workflow.py -v
pytest tests/integration/test_document_processing.py -v
pytest tests/integration/test_embedding_service.py -v
```

### 2. Quality Assurance
```bash
# Test response quality
python -m src.testing.quality test_response_quality

# Test citation accuracy
python -m src.testing.quality test_citation_accuracy
```

### 3. User Acceptance Testing
```bash
# Test UI functionality
npm run test:ui

# Test API endpoints
python -m src.testing.api test_all_endpoints
```

## Rollback Procedures

### Emergency Rollback
```bash
# Stop services
docker-compose down

# Restore previous version
git checkout previous-stable-tag

# Restore backups
cp backups/$(date +%Y%m%d_%H%M%S)/data/* data/
cp backups/$(date +%Y%m%d_%H%M%S)/app.db app.db

# Restart services
docker-compose up -d
```

### Gradual Rollback
```bash
# Disable new features
curl -X POST http://localhost:8000/api/v1/admin/config \
  -H "Content-Type: application/json" \
  -d '{
    "use_cross_encoder": false,
    "search_type": "semantic",
    "ocr_enabled": false
  }'

# Monitor system behavior
python -m src.core.monitoring system_metrics
```

## Troubleshooting

### Common Issues

#### 1. Embedding Model Loading Errors
```bash
# Clear model cache
rm -rf ~/.cache/torch/sentence_transformers/

# Reinstall models
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
print('Model loaded successfully')
"
```

#### 2. Vector Store Connection Issues
```bash
# Check ChromaDB status
python -c "
import chromadb
client = chromadb.PersistentClient(path='./data/vectordb')
print('ChromaDB connection successful')
print('Collections:', [c.name for c in client.list_collections()])
"
```

#### 3. OCR Service Issues
```bash
# Test Tesseract installation
tesseract --version

# Test PDF processing
python -c "
from src.services.ocr_service import ocr_service
print('OCR Status:', ocr_service.get_ocr_status())
"
```

#### 4. Performance Issues
```bash
# Check system resources
htop

# Monitor API performance
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/api/v1/query \
  -X POST -H "Content-Type: application/json" \
  -d '{"query": "test", "max_results": 5}'
```

### Support Contacts

- **Technical Lead**: [Contact Information]
- **DevOps Team**: [Contact Information]
- **Emergency Hotline**: [Contact Information]

## Migration Checklist

### Pre-Migration
- [ ] System backup completed
- [ ] Environment configuration updated
- [ ] Dependencies installed
- [ ] Service dependencies verified

### Migration Steps
- [ ] Database schema migrated
- [ ] Embedding service updated
- [ ] Documents reprocessed
- [ ] Vector store migrated
- [ ] Configuration updated
- [ ] Services validated
- [ ] Performance tested

### Post-Migration
- [ ] Functional tests passed
- [ ] Quality assurance completed
- [ ] User acceptance testing passed
- [ ] Monitoring configured
- [ ] Documentation updated

### Rollback Preparation
- [ ] Rollback procedures tested
- [ ] Backup verification completed
- [ ] Emergency contacts confirmed

## Conclusion

This migration guide provides comprehensive steps for upgrading to the enhanced RAG system. Follow the steps carefully and ensure all validations pass before considering the migration complete. Keep rollback procedures ready in case of issues during the migration process.
