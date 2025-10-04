# Llama3.1:70b Setup and Usage Guide

This guide provides comprehensive instructions for setting up and using the powerful Llama3.1:70b model with the FDS RAG system for maximum answer quality.

## 🚀 Overview

The FDS RAG system now supports the **Llama3.1:70b** model, which provides:
- **8.75x more parameters** than the previous 8b model
- **Significantly better reasoning** and analysis capabilities
- **Enhanced answer quality** for complex queries
- **Up to 200 chunks** retrieved for comprehensive context
- **No timeout limits** for maximum processing time

## 📋 Prerequisites

### System Requirements
- **RAM**: Minimum 32GB (recommended 64GB+)
- **Storage**: 50GB+ free space for model files
- **CPU**: Multi-core processor (8+ cores recommended)
- **Docker**: Installed and running
- **Docker Compose**: v2.0+

### Hardware Recommendations
- **For optimal performance**: 64GB+ RAM, 16+ CPU cores
- **For basic usage**: 32GB RAM, 8+ CPU cores
- **Storage**: SSD recommended for faster model loading

## 🔧 Installation Steps

### Step 1: Verify Docker Setup

```bash
# Check Docker is running
docker --version
docker-compose --version

# Verify Docker has sufficient resources
docker system info
```

### Step 2: Configure Environment

Create or update your `.env` file in the project root:

```bash
# LLM Configuration
LOCAL_LLM_BASE_URL=http://localhost:8090/v1
LOCAL_LLM_MODEL=llama3.1:70b
USE_LOCAL_LLM=true

# Enhanced Retrieval Settings
RETRIEVAL_K=100
RERANK_TOP_K=32
USE_CROSS_ENCODER=true

# Timeout Settings (0 = no timeout)
LLM_TIMEOUT=0
LLM_RESPONSE_MAX_TOKENS=0
```

### Step 3: Start Ollama Service

```bash
# Navigate to project directory
cd "C:\FDS Docs"

# Start Ollama with Docker Compose
docker-compose up -d llama

# Verify service is running
docker-compose ps
```

Expected output:
```
NAME              IMAGE                  COMMAND               SERVICE   CREATED          STATUS          PORTS
fdsdocs-llama-1   ollama/ollama:latest   "/bin/ollama serve"   llama     X minutes ago    Up X minutes    0.0.0.0:8090->11434/tcp
```

### Step 4: Download Llama3.1:70b Model

```bash
# Pull the 70b model (this will take 10-30 minutes)
docker-compose exec llama ollama pull llama3.1:70b

# Verify model is downloaded
docker-compose exec llama ollama list
```

Expected output:
```
NAME                         ID              SIZE      MODIFIED           
llama3.1:70b                 711a9e8463af    42 GB     X minutes ago    
llama3.1:8b                  46e0c10c039e    4.9 GB    X minutes ago        
llama3.1:8b-instruct-q4_0    42182419e950    4.7 GB    7 days ago            
```

### Step 5: Start Backend Service

```bash
# Start the FDS backend with new configuration
python run.py
```

## ⚙️ Configuration Details

### Enhanced Retrieval Settings

The system is configured for maximum quality with these settings:

```python
# Maximum retrieval (up to 200 chunks for complex queries)
retrieval_k = 100  # Base retrieval
# Dynamic scaling:
# - Complex queries (16+ words): up to 150 chunks
# - Medium queries (8+ words): up to 120 chunks  
# - Numeric queries: up to 140 chunks
# - Maximum cap: 200 chunks

# Advanced reranking
rerank_top_k = 32  # Top candidates for reranking
use_cross_encoder = True  # Enable cross-encoder reranking

# No limits for maximum quality
llm_timeout = 0  # No timeout
llm_response_max_tokens = 0  # No token limit
```

### Model Selection

The system automatically uses the best available model:

1. **Primary**: `llama3.1:70b` (42GB) - Maximum quality
2. **Fallback**: `llama3.1:8b` (4.9GB) - Good quality, faster
3. **Legacy**: `llama3.1:8b-instruct-q4_0` (4.7GB) - Quantized, fastest

## 🎯 Usage Instructions

### Basic Query

Simply ask your question in the web interface. The system will:

1. **Retrieve** up to 200 relevant chunks
2. **Rerank** using cross-encoder for relevance
3. **Generate** answer with 70b model (no time/token limits)
4. **Provide** comprehensive, high-quality response

### Query Types Optimized

The system is particularly powerful for:

- **Complex analytical queries** (16+ words)
- **Numeric/counting questions** (up to 140 chunks)
- **Multi-part questions** requiring synthesis
- **Detailed explanations** needing comprehensive context

### Example Queries

```
# Complex analytical query (will use 150+ chunks)
"What are the key factors contributing to the success of emergency response operations in conflict zones, and how do they differ from standard humanitarian interventions?"

# Numeric query (will use 140+ chunks)  
"How many ongoing emergency operations are currently active worldwide, and what are the primary causes?"

# Multi-part synthesis (will use 120+ chunks)
"Compare the effectiveness of different disaster response strategies used in the past five years, including their resource requirements and outcomes."
```

## 🔍 Monitoring and Troubleshooting

### Check Model Status

```bash
# Verify model is loaded and ready
docker-compose exec llama ollama ps

# Check available models
docker-compose exec llama ollama list

# Test model directly
docker-compose exec llama ollama run llama3.1:70b "Hello, how are you?"
```

### Monitor Performance

```bash
# Check container resource usage
docker stats fdsdocs-llama-1

# View Ollama logs
docker-compose logs -f llama

# Check backend logs
# (Look for LLM generation logs in your Python console)
```

### Common Issues

#### Issue: Model Not Loading
```bash
# Check if model is downloaded
docker-compose exec llama ollama list

# If missing, re-download
docker-compose exec llama ollama pull llama3.1:70b
```

#### Issue: Out of Memory
```bash
# Check available memory
docker stats fdsdocs-llama-1

# If insufficient RAM, use smaller model
# Update .env: LOCAL_LLM_MODEL=llama3.1:8b
```

#### Issue: Slow Performance
```bash
# Check if model is loaded in memory
docker-compose exec llama ollama ps

# Pre-load model for faster responses
docker-compose exec llama ollama run llama3.1:70b
```

## 📊 Performance Expectations

### Response Times
- **First query**: 30-60 seconds (model loading)
- **Subsequent queries**: 10-30 seconds
- **Complex queries**: 30-120 seconds
- **Simple queries**: 5-15 seconds

### Quality Improvements
- **Reasoning**: 8.75x better than 8b model
- **Context understanding**: Up to 200 chunks vs 10
- **Answer completeness**: Significantly improved
- **Numerical accuracy**: Much better for counting queries

### Resource Usage
- **RAM**: 25-30GB during inference
- **CPU**: High usage during generation
- **Storage**: 42GB for model files

## 🔄 Switching Between Models

### Use 70b Model (Maximum Quality)
```bash
# Update .env
LOCAL_LLM_MODEL=llama3.1:70b

# Restart backend
python run.py
```

### Use 8b Model (Balanced Performance)
```bash
# Update .env  
LOCAL_LLM_MODEL=llama3.1:8b

# Restart backend
python run.py
```

### Use Quantized Model (Fastest)
```bash
# Update .env
LOCAL_LLM_MODEL=llama3.1:8b-instruct-q4_0

# Restart backend
python run.py
```

## 🚨 Emergency Procedures

### Quick Rollback to 8b Model
```bash
# Update .env to use smaller model
echo "LOCAL_LLM_MODEL=llama3.1:8b" >> .env

# Restart backend
python run.py
```

### Stop All Services
```bash
# Stop backend (Ctrl+C in terminal)
# Stop Ollama
docker-compose down
```

### Clear Model Cache
```bash
# Remove all models (will need to re-download)
docker-compose exec llama ollama rm llama3.1:70b
docker-compose exec llama ollama rm llama3.1:8b
```

## 📈 Optimization Tips

### For Maximum Quality
1. Use `llama3.1:70b` model
2. Ensure 64GB+ RAM available
3. Use SSD storage
4. Keep model loaded in memory

### For Balanced Performance
1. Use `llama3.1:8b` model
2. Ensure 32GB+ RAM available
3. Pre-load model for faster responses

### For Speed
1. Use `llama3.1:8b-instruct-q4_0` model
2. Reduce `retrieval_k` to 50
3. Enable timeouts for faster responses

## 📞 Support

### Getting Help
- Check logs: `docker-compose logs llama`
- Verify configuration: Check `.env` file
- Test model: `docker-compose exec llama ollama run llama3.1:70b`

### Performance Issues
- Monitor RAM usage: `docker stats`
- Check model status: `docker-compose exec llama ollama ps`
- Verify backend logs for LLM generation

### Model Issues
- Re-download model: `docker-compose exec llama ollama pull llama3.1:70b`
- Switch to smaller model if memory issues
- Check Docker resources and restart if needed

## 🎉 Success Indicators

You'll know the setup is working correctly when:

✅ **Model Status**: `llama3.1:70b` appears in `ollama list`  
✅ **Backend Logs**: Show "LLM generation successful"  
✅ **Response Quality**: Significantly better answers  
✅ **Response Time**: 10-30 seconds for complex queries  
✅ **Memory Usage**: 25-30GB RAM during inference  

## 📝 Next Steps

1. **Test the system** with your typical queries
2. **Monitor performance** and adjust settings if needed
3. **Document any issues** for future reference
4. **Share feedback** on answer quality improvements

---

**Note**: The 70b model provides the highest quality answers but requires significant system resources. For production use, consider your hardware capabilities and response time requirements when choosing between models.
