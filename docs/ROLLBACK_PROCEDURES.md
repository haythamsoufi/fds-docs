# RAG System Rollback Procedures

## Overview

This document provides detailed rollback procedures for the RAG system upgrade, including emergency rollback, gradual rollback, and data recovery procedures.

## Rollback Scenarios

### 1. Emergency Rollback (Complete System Failure)
- System is completely unresponsive
- Critical data corruption detected
- Security breach suspected
- Performance degradation > 90%

### 2. Gradual Rollback (Feature-Specific Issues)
- Specific features causing problems
- Performance issues with new components
- User complaints about response quality
- Partial system degradation

### 3. Data Recovery Rollback
- Data corruption in vector store
- Embedding inconsistencies
- Document processing errors
- Database integrity issues

## Emergency Rollback Procedures

### Step 1: Immediate Service Shutdown
```bash
# Stop all services immediately
docker-compose down --remove-orphans

# Kill any remaining processes
pkill -f "uvicorn\|gunicorn\|celery"

# Verify all processes stopped
ps aux | grep -E "(uvicorn|gunicorn|celery)"
```

### Step 2: Restore Previous Version
```bash
# Navigate to project directory
cd /path/to/fds-docs

# Create emergency backup of current state
mkdir -p emergency_backup/$(date +%Y%m%d_%H%M%S)
cp -r . emergency_backup/$(date +%Y%m%d_%H%M%S)/

# Restore previous stable version
git stash
git checkout previous-stable-tag
git checkout -b emergency-rollback-$(date +%Y%m%d_%H%M%S)
```

### Step 3: Database Rollback
```bash
# Stop database services
docker-compose -f docker-compose.db.yml down

# Restore database backup
cp backups/$(date +%Y%m%d_%H%M%S)/app.db ./data/app.db

# Restore database schema if needed
alembic downgrade -1

# Verify database integrity
python -c "
from src.core.database import engine
from sqlalchemy import text
with engine.connect() as conn:
    result = conn.execute(text('SELECT COUNT(*) FROM documents'))
    print(f'Documents: {result.scalar()}')
"
```

### Step 4: Vector Store Rollback
```bash
# Remove current vector store
rm -rf ./data/vectordb

# Restore vector store backup
cp -r backups/$(date +%Y%m%d_%H%M%S)/data/vectordb ./data/

# Verify vector store integrity
python -c "
import chromadb
try:
    client = chromadb.PersistentClient(path='./data/vectordb')
    collections = client.list_collections()
    print(f'Collections: {[c.name for c in collections]}')
except Exception as e:
    print(f'Vector store error: {e}')
"
```

### Step 5: Configuration Rollback
```bash
# Restore configuration
cp backups/$(date +%Y%m%d_%H%M%S)/.env .env

# Restore Docker configuration
cp backups/$(date +%Y%m%d_%H%M%S)/docker-compose.yml docker-compose.yml
```

### Step 6: Service Restart
```bash
# Restart services with previous configuration
docker-compose up -d

# Monitor service startup
docker-compose logs -f

# Verify services are running
curl http://localhost:8000/api/v1/health
```

### Step 7: Validation
```bash
# Test basic functionality
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "max_results": 3}'

# Check system status
curl http://localhost:8000/api/v1/admin/stats
```

## Gradual Rollback Procedures

### Step 1: Disable Problematic Features
```bash
# Disable cross-encoder reranking
curl -X POST http://localhost:8000/api/v1/admin/config \
  -H "Content-Type: application/json" \
  -d '{"use_cross_encoder": false}'

# Switch to semantic-only search
curl -X POST http://localhost:8000/api/v1/admin/config \
  -H "Content-Type: application/json" \
  -d '{"search_type": "semantic"}'

# Disable OCR if causing issues
curl -X POST http://localhost:8000/api/v1/admin/config \
  -H "Content-Type: application/json" \
  -d '{"ocr_enabled": false}'
```

### Step 2: Revert Embedding Models
```bash
# Revert to previous embedding model
curl -X POST http://localhost:8000/api/v1/admin/config \
  -H "Content-Type: application/json" \
  -d '{
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_version": 1
  }'

# Clear embedding cache
curl -X POST http://localhost:8000/api/v1/admin/cache-control \
  -H "Content-Type: application/json" \
  -d '{"cache_type": "embeddings", "action": "clear"}'
```

### Step 3: Adjust Performance Settings
```bash
# Reduce batch size
curl -X POST http://localhost:8000/api/v1/admin/config \
  -H "Content-Type: application/json" \
  -d '{"batch_size": 16}'

# Reduce concurrent requests
curl -X POST http://localhost:8000/api/v1/admin/config \
  -H "Content-Type: application/json" \
  -d '{"max_concurrent_requests": 50}'

# Increase timeouts
curl -X POST http://localhost:8000/api/v1/admin/config \
  -H "Content-Type: application/json" \
  -d '{
    "vector_db_operation_timeout": 60.0,
    "llm_timeout": 60.0
  }'
```

### Step 4: Monitor System Behavior
```bash
# Monitor system metrics
python -c "
import time
import requests

def monitor_system():
    for i in range(10):
        try:
            response = requests.get('http://localhost:8000/api/v1/admin/stats')
            stats = response.json()
            print(f'Iteration {i+1}: {stats}')
        except Exception as e:
            print(f'Error: {e}')
        time.sleep(30)

monitor_system()
"
```

## Data Recovery Rollback

### Step 1: Database Recovery
```bash
# Check database integrity
python -c "
from src.core.database import engine
from sqlalchemy import text
import sqlite3

# SQLite integrity check
conn = sqlite3.connect('./data/app.db')
cursor = conn.cursor()
cursor.execute('PRAGMA integrity_check')
result = cursor.fetchone()
print(f'Database integrity: {result[0]}')
conn.close()
"

# Repair database if needed
sqlite3 ./data/app.db "PRAGMA integrity_check; VACUUM;"
```

### Step 2: Vector Store Recovery
```bash
# Check vector store integrity
python -c "
import chromadb
import os

try:
    client = chromadb.PersistentClient(path='./data/vectordb')
    collections = client.list_collections()
    
    for collection in collections:
        count = collection.count()
        print(f'Collection {collection.name}: {count} items')
        
except Exception as e:
    print(f'Vector store error: {e}')
    print('Attempting to rebuild vector store...')
"
```

### Step 3: Rebuild Vector Store
```bash
# Clear corrupted vector store
rm -rf ./data/vectordb

# Rebuild from database
curl -X POST http://localhost:8000/api/v1/admin/populate-vectors \
  -H "Content-Type: application/json"

# Monitor rebuild progress
watch -n 5 'curl -s http://localhost:8000/api/v1/admin/stats | jq'
```

### Step 4: Document Reprocessing
```bash
# Reprocess documents if needed
curl -X POST http://localhost:8000/api/v1/admin/reprocess-documents \
  -H "Content-Type: application/json"

# Monitor processing
curl -s http://localhost:8000/api/v1/admin/stats | jq '.documents'
```

## Automated Rollback Scripts

### Emergency Rollback Script
```bash
#!/bin/bash
# emergency_rollback.sh

set -e

echo "Starting emergency rollback..."

# Configuration
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
PROJECT_DIR="/path/to/fds-docs"
PREVIOUS_TAG="v1.2.0"

# Step 1: Stop services
echo "Stopping services..."
cd $PROJECT_DIR
docker-compose down --remove-orphans

# Step 2: Create emergency backup
echo "Creating emergency backup..."
mkdir -p emergency_backup/$(date +%Y%m%d_%H%M%S)
cp -r . emergency_backup/$(date +%Y%m%d_%H%M%S)/

# Step 3: Restore previous version
echo "Restoring previous version..."
git stash
git checkout $PREVIOUS_TAG
git checkout -b emergency-rollback-$(date +%Y%m%d_%H%M%S)

# Step 4: Restore data
echo "Restoring data..."
if [ -d "$BACKUP_DIR" ]; then
    cp $BACKUP_DIR/app.db ./data/
    cp -r $BACKUP_DIR/data/vectordb ./data/
    cp $BACKUP_DIR/.env .env
fi

# Step 5: Restart services
echo "Restarting services..."
docker-compose up -d

# Step 6: Validate
echo "Validating rollback..."
sleep 10
curl -f http://localhost:8000/api/v1/health || exit 1

echo "Emergency rollback completed successfully"
```

### Gradual Rollback Script
```bash
#!/bin/bash
# gradual_rollback.sh

set -e

echo "Starting gradual rollback..."

# Configuration
API_BASE="http://localhost:8000/api/v1"

# Step 1: Disable new features
echo "Disabling new features..."
curl -X POST $API_BASE/admin/config \
  -H "Content-Type: application/json" \
  -d '{"use_cross_encoder": false}'

curl -X POST $API_BASE/admin/config \
  -H "Content-Type: application/json" \
  -d '{"search_type": "semantic"}'

# Step 2: Revert embedding models
echo "Reverting embedding models..."
curl -X POST $API_BASE/admin/config \
  -H "Content-Type: application/json" \
  -d '{"embedding_model": "all-MiniLM-L6-v2", "embedding_version": 1}'

# Step 3: Clear caches
echo "Clearing caches..."
curl -X POST $API_BASE/admin/cache-control \
  -H "Content-Type: application/json" \
  -d '{"cache_type": "all", "action": "clear"}'

# Step 4: Monitor system
echo "Monitoring system for 5 minutes..."
for i in {1..10}; do
    echo "Check $i/10"
    curl -s $API_BASE/admin/stats | jq '.documents'
    sleep 30
done

echo "Gradual rollback completed"
```

## Rollback Validation

### Health Check Script
```bash
#!/bin/bash
# validate_rollback.sh

echo "Validating rollback..."

# Check API health
echo "Checking API health..."
curl -f http://localhost:8000/api/v1/health || {
    echo "API health check failed"
    exit 1
}

# Check database
echo "Checking database..."
python -c "
from src.core.database import engine
from sqlalchemy import text
with engine.connect() as conn:
    result = conn.execute(text('SELECT COUNT(*) FROM documents'))
    print(f'Documents: {result.scalar()}')
"

# Check vector store
echo "Checking vector store..."
python -c "
import chromadb
client = chromadb.PersistentClient(path='./data/vectordb')
collections = client.list_collections()
print(f'Collections: {[c.name for c in collections]}')
"

# Test query functionality
echo "Testing query functionality..."
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "max_results": 3}' || {
    echo "Query test failed"
    exit 1
}

echo "Rollback validation completed successfully"
```

## Monitoring and Alerting

### Rollback Triggers
```python
# rollback_monitor.py

import time
import requests
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RollbackMonitor:
    def __init__(self, api_base="http://localhost:8000/api/v1"):
        self.api_base = api_base
        self.error_count = 0
        self.max_errors = 5
        
    def check_system_health(self):
        """Check system health and trigger rollback if needed"""
        try:
            # Check API health
            response = requests.get(f"{self.api_base}/health", timeout=10)
            if response.status_code != 200:
                self.error_count += 1
                logger.error(f"API health check failed: {response.status_code}")
                return False
                
            # Check response time
            start_time = time.time()
            response = requests.post(f"{self.api_base}/query", 
                                   json={"query": "test", "max_results": 3}, 
                                   timeout=30)
            response_time = time.time() - start_time
            
            if response_time > 10:  # 10 second threshold
                self.error_count += 1
                logger.error(f"Response time too slow: {response_time:.2f}s")
                return False
                
            # Reset error count on success
            self.error_count = 0
            return True
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Health check error: {e}")
            return False
    
    def should_rollback(self):
        """Determine if rollback should be triggered"""
        return self.error_count >= self.max_errors
    
    def trigger_rollback(self):
        """Trigger emergency rollback"""
        logger.critical("Triggering emergency rollback")
        # Execute rollback script
        import subprocess
        subprocess.run(["./emergency_rollback.sh"])
    
    def monitor(self):
        """Main monitoring loop"""
        logger.info("Starting rollback monitoring")
        
        while True:
            if not self.check_system_health():
                if self.should_rollback():
                    self.trigger_rollback()
                    break
                    
            time.sleep(60)  # Check every minute

if __name__ == "__main__":
    monitor = RollbackMonitor()
    monitor.monitor()
```

## Rollback Communication

### Rollback Notification Template
```bash
#!/bin/bash
# notify_rollback.sh

# Send notification about rollback
curl -X POST "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "🚨 RAG System Rollback Initiated",
    "blocks": [
      {
        "type": "section",
        "text": {
          "type": "mrkdwn",
          "text": "*RAG System Rollback*\n\n*Time:* '$(date)'\n*Type:* Emergency Rollback\n*Reason:* System failure detected\n*Status:* In progress"
        }
      }
    ]
  }'
```

## Post-Rollback Actions

### 1. Root Cause Analysis
```bash
# Collect logs for analysis
mkdir -p rollback_analysis/$(date +%Y%m%d_%H%M%S)
docker-compose logs > rollback_analysis/$(date +%Y%m%d_%H%M%S)/docker.logs
cp -r logs/ rollback_analysis/$(date +%Y%m%d_%H%M%S)/

# System state snapshot
python -c "
import psutil
import json
import datetime

state = {
    'timestamp': datetime.datetime.now().isoformat(),
    'cpu_percent': psutil.cpu_percent(),
    'memory_percent': psutil.virtual_memory().percent,
    'disk_usage': psutil.disk_usage('/').percent
}

with open('rollback_analysis/$(date +%Y%m%d_%H%M%S)/system_state.json', 'w') as f:
    json.dump(state, f, indent=2)
"
```

### 2. Service Validation
```bash
# Comprehensive service validation
python -c "
import requests
import time

def validate_services():
    endpoints = [
        '/api/v1/health',
        '/api/v1/admin/stats',
        '/api/v1/documents'
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f'http://localhost:8000{endpoint}')
            print(f'{endpoint}: {response.status_code}')
        except Exception as e:
            print(f'{endpoint}: ERROR - {e}')

validate_services()
"
```

### 3. Performance Monitoring
```bash
# Monitor performance post-rollback
python -c "
import time
import requests

def monitor_performance():
    times = []
    for i in range(5):
        start = time.time()
        response = requests.post('http://localhost:8000/api/v1/query', 
                               json={'query': 'test', 'max_results': 3})
        end = time.time()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    print(f'Average response time: {avg_time:.2f}s')
    print(f'Min: {min(times):.2f}s, Max: {max(times):.2f}s')

monitor_performance()
"
```

## Conclusion

These rollback procedures provide comprehensive coverage for different failure scenarios. The emergency rollback should be used for complete system failures, while gradual rollback is appropriate for feature-specific issues. Always validate the rollback success and monitor system behavior after rollback completion.

Remember to:
1. Keep backups up to date
2. Test rollback procedures regularly
3. Monitor system health continuously
4. Document all rollback events
5. Conduct post-rollback analysis
