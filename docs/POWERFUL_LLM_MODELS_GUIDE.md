# Powerful LLM Models for GPU Hosting Guide

This guide provides comprehensive information about the most powerful open-source LLM models that can be used with the FDS RAG system when hosting on a VM with powerful hardware and GPU acceleration.

## 🏆 Top Tier Models (Most Powerful)

### **1. Qwen2.5-72B** ⭐ **RECOMMENDED**
- **Parameters**: 72 billion
- **GPU Memory**: 40-48GB VRAM
- **Ollama Support**: ✅ Available
- **Why it's powerful**:
  - Superior reasoning and analysis capabilities
  - Excellent multilingual support (100+ languages)
  - Strong performance in complex tasks
  - Better than Llama3.1-70B in most benchmarks
- **Configuration**:
```bash
LOCAL_LLM_MODEL=qwen2.5:72b
```

### **2. Mixtral-8x22B** 
- **Parameters**: 141B total, 39B active (sparse)
- **GPU Memory**: 24-32GB VRAM
- **Ollama Support**: ✅ Available
- **Why it's powerful**:
  - Mixture of Experts architecture
  - 64K context window
  - Excellent for coding and mathematics
  - Efficient inference despite large size
- **Configuration**:
```bash
LOCAL_LLM_MODEL=mixtral:8x22b
```

### **3. DeepSeek-V2-236B** 
- **Parameters**: 236 billion
- **GPU Memory**: 80GB+ VRAM (requires multiple GPUs)
- **Ollama Support**: ✅ Available
- **Why it's powerful**:
  - Largest open-source model available
  - Exceptional reasoning capabilities
  - Strong in mathematics and coding
  - 64K context window
- **Configuration**:
```bash
LOCAL_LLM_MODEL=deepseek-coder:33b  # Smaller variant
# or
LOCAL_LLM_MODEL=deepseek-v2:236b    # Full model (requires massive GPU)
```

## 🚀 High-Performance Alternatives

### **4. CodeLlama-2-70B**
- **Parameters**: 70 billion
- **GPU Memory**: 40-48GB VRAM
- **Ollama Support**: ✅ Available
- **Why it's powerful**:
  - Specialized for code generation
  - Excellent for technical documents
  - Strong reasoning capabilities
- **Configuration**:
```bash
LOCAL_LLM_MODEL=codellama:70b
```

### **5. Yi-1.5-34B**
- **Parameters**: 34 billion
- **GPU Memory**: 20-24GB VRAM
- **Ollama Support**: ✅ Available
- **Why it's powerful**:
  - Excellent performance-to-size ratio
  - Strong multilingual support
  - Efficient resource usage
- **Configuration**:
```bash
LOCAL_LLM_MODEL=yi:34b
```

## 💡 Recommendations by Use Case

### **For Maximum Quality (if you have 40GB+ VRAM)**
```bash
LOCAL_LLM_MODEL=qwen2.5:72b
```
- Best overall performance
- Excellent for complex reasoning
- Superior multilingual capabilities

### **For Balanced Performance (24-32GB VRAM)**
```bash
LOCAL_LLM_MODEL=mixtral:8x22b
```
- Excellent performance with efficient resource usage
- Great for coding and mathematics
- 64K context window

### **For Technical Documents (40GB+ VRAM)**
```bash
LOCAL_LLM_MODEL=codellama:70b
```
- Specialized for technical content
- Excellent code understanding
- Strong reasoning capabilities

## 🔧 Implementation Steps

### **1. Update Your Environment**
```bash
# In your .env file
LOCAL_LLM_MODEL=qwen2.5:72b  # or your chosen model
LOCAL_LLM_BASE_URL=http://localhost:8090/v1
USE_LOCAL_LLM=true

# Enhanced settings for powerful models
RETRIEVAL_K=150
RERANK_TOP_K=32
USE_CROSS_ENCODER=true
LLM_TIMEOUT=0
LLM_RESPONSE_MAX_TOKENS=0
```

### **2. Update Docker Compose for GPU**
```yaml
services:
  llama:
    image: ollama/ollama:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - OLLAMA_KEEP_ALIVE=30m
      - OLLAMA_GPU_LAYERS=35  # Adjust based on your GPU
    volumes:
      - ollama:/root/.ollama
    restart: unless-stopped
```

### **3. Download the Model**
```bash
# Start Ollama with GPU support
docker-compose up -d llama

# Download your chosen model
docker-compose exec llama ollama pull qwen2.5:72b

# Verify model is available
docker-compose exec llama ollama list
```

## 📊 Hardware Requirements Comparison

| Model | VRAM Required | RAM Required | Performance |
|-------|---------------|---------------|-------------|
| **Qwen2.5-72B** | 40-48GB | 64GB+ | ⭐⭐⭐⭐⭐ |
| **Mixtral-8x22B** | 24-32GB | 48GB+ | ⭐⭐⭐⭐⭐ |
| **DeepSeek-V2-236B** | 80GB+ | 128GB+ | ⭐⭐⭐⭐⭐ |
| **CodeLlama-2-70B** | 40-48GB | 64GB+ | ⭐⭐⭐⭐ |
| **Yi-1.5-34B** | 20-24GB | 32GB+ | ⭐⭐⭐⭐ |

## 🎯 Performance Expectations

### **Qwen2.5-72B** (Recommended)
- **Response Time**: 15-45 seconds
- **Quality**: Exceptional reasoning and analysis
- **Multilingual**: Excellent support for 100+ languages
- **Context**: 32K tokens

### **Mixtral-8x22B**
- **Response Time**: 10-30 seconds
- **Quality**: Excellent for complex tasks
- **Context**: 64K tokens
- **Efficiency**: Very efficient resource usage

### **DeepSeek-V2-236B**
- **Response Time**: 30-120 seconds
- **Quality**: Best available (if you have the hardware)
- **Context**: 64K tokens
- **Requirements**: Massive GPU setup needed

## 🚀 Quick Start with Qwen2.5-72B

```bash
# 1. Update your .env
echo "LOCAL_LLM_MODEL=qwen2.5:72b" >> .env

# 2. Start with GPU support
docker-compose up -d llama

# 3. Download the model
docker-compose exec llama ollama pull qwen2.5:72b

# 4. Start your backend
python run.py
```

## 🔍 Model Comparison with Llama3.1-70B

| Feature | Llama3.1-70B | Qwen2.5-72B | Mixtral-8x22B | DeepSeek-V2-236B |
|---------|--------------|--------------|---------------|------------------|
| **Parameters** | 70B | 72B | 141B (39B active) | 236B |
| **Context Window** | 128K | 32K | 64K | 64K |
| **Multilingual** | Good | Excellent | Good | Good |
| **Code Generation** | Good | Excellent | Excellent | Excellent |
| **Reasoning** | Good | Excellent | Excellent | Exceptional |
| **VRAM Required** | 40GB | 40-48GB | 24-32GB | 80GB+ |
| **Inference Speed** | Medium | Medium | Fast | Slow |

## 🛠️ Advanced Configuration

### **GPU Optimization Settings**
```bash
# Environment variables for optimal GPU usage
CUDA_VISIBLE_DEVICES=0
OLLAMA_GPU_LAYERS=35  # Adjust based on your GPU memory
OLLAMA_KEEP_ALIVE=30m
OLLAMA_NUM_PARALLEL=1
OLLAMA_MAX_LOADED_MODELS=1
```

### **Memory Optimization**
```bash
# For systems with limited VRAM
OLLAMA_GPU_LAYERS=20  # Reduce GPU layers
OLLAMA_MMAP=true      # Use memory mapping
OLLAMA_MLOCK=false    # Disable memory locking
```

### **Performance Tuning**
```bash
# For maximum performance
RETRIEVAL_K=200       # Maximum context retrieval
RERANK_TOP_K=32       # Advanced reranking
USE_CROSS_ENCODER=true # Enable cross-encoder
LLM_TIMEOUT=0         # No timeout limits
LLM_RESPONSE_MAX_TOKENS=0 # No token limits
```

## 🔧 Troubleshooting

### **Common Issues**

#### **Out of Memory Errors**
```bash
# Check GPU memory usage
nvidia-smi

# Reduce GPU layers
OLLAMA_GPU_LAYERS=20

# Use smaller model
LOCAL_LLM_MODEL=qwen2.5:32b  # Smaller variant
```

#### **Slow Performance**
```bash
# Pre-load model
docker-compose exec llama ollama run qwen2.5:72b

# Check if model is loaded
docker-compose exec llama ollama ps
```

#### **Model Not Found**
```bash
# List available models
docker-compose exec llama ollama list

# Pull the model
docker-compose exec llama ollama pull qwen2.5:72b
```

## 📈 Monitoring and Optimization

### **Performance Monitoring**
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor Ollama logs
docker-compose logs -f llama

# Check model status
docker-compose exec llama ollama ps
```

### **Resource Optimization**
- **For 24GB VRAM**: Use Mixtral-8x22B or Yi-1.5-34B
- **For 40GB VRAM**: Use Qwen2.5-72B or CodeLlama-2-70B
- **For 80GB+ VRAM**: Use DeepSeek-V2-236B or multiple models

## 🎉 Success Indicators

You'll know the setup is working correctly when:

✅ **Model Status**: Your chosen model appears in `ollama list`  
✅ **Backend Logs**: Show "LLM generation successful"  
✅ **Response Quality**: Significantly better answers than Llama3.1-70B  
✅ **Response Time**: 10-45 seconds for complex queries  
✅ **Memory Usage**: Appropriate VRAM usage for your model  

## 📝 Next Steps

1. **Choose your model** based on your hardware capabilities
2. **Update configuration** with the recommended settings
3. **Test the system** with your typical queries
4. **Monitor performance** and adjust settings if needed
5. **Document any issues** for future reference

---

**Note**: These models provide significantly better performance than Llama3.1-70B but require substantial GPU resources. Choose based on your hardware capabilities and response time requirements.

**Qwen2.5-72B** is recommended as the best overall choice for most use cases, offering excellent performance with reasonable hardware requirements.
