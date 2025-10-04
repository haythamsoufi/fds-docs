# Llama3.1:70b Quick Reference

## 🚀 Quick Setup (5 minutes)

```bash
# 1. Start Ollama
cd "C:\FDS Docs"
docker-compose up -d llama

# 2. Download model (10-30 minutes)
docker-compose exec llama ollama pull llama3.1:70b

# 3. Configure environment
echo "LOCAL_LLM_BASE_URL=http://localhost:8090/v1" > .env
echo "LOCAL_LLM_MODEL=llama3.1:70b" >> .env
echo "USE_LOCAL_LLM=true" >> .env

# 4. Start backend
python run.py
```

## ⚡ Quick Commands

```bash
# Check model status
docker-compose exec llama ollama list

# Test model
docker-compose exec llama ollama run llama3.1:70b "Hello"

# Check memory usage
docker stats fdsdocs-llama-1

# View logs
docker-compose logs -f llama
```

## 🔧 Quick Configuration

### Maximum Quality (70b model)
```env
LOCAL_LLM_MODEL=llama3.1:70b
RETRIEVAL_K=100
LLM_TIMEOUT=0
LLM_RESPONSE_MAX_TOKENS=0
```

### Balanced Performance (8b model)
```env
LOCAL_LLM_MODEL=llama3.1:8b
RETRIEVAL_K=50
LLM_TIMEOUT=60
LLM_RESPONSE_MAX_TOKENS=1000
```

### Fast Performance (quantized)
```env
LOCAL_LLM_MODEL=llama3.1:8b-instruct-q4_0
RETRIEVAL_K=20
LLM_TIMEOUT=30
LLM_RESPONSE_MAX_TOKENS=500
```

## 🚨 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | `docker-compose exec llama ollama pull llama3.1:70b` |
| Out of memory | Switch to `llama3.1:8b` model |
| Slow responses | Check `docker-compose exec llama ollama ps` |
| Backend errors | Check `.env` file configuration |

## 📊 Performance Expectations

| Model | RAM Usage | Response Time | Quality |
|-------|-----------|---------------|---------|
| 70b | 25-30GB | 10-30s | Maximum |
| 8b | 8-12GB | 3-10s | High |
| 8b-q4 | 4-6GB | 1-5s | Good |

## 🎯 Best Practices

1. **For maximum quality**: Use 70b with 64GB+ RAM
2. **For balanced performance**: Use 8b with 32GB+ RAM  
3. **For speed**: Use quantized model with 16GB+ RAM
4. **Always check**: `docker-compose exec llama ollama ps` before queries
5. **Monitor resources**: `docker stats` during heavy usage

## 📞 Emergency Rollback

```bash
# Quick switch to smaller model
echo "LOCAL_LLM_MODEL=llama3.1:8b" > .env
python run.py
```

---

**Full Documentation**: See [LLAMA_SETUP_GUIDE.md](LLAMA_SETUP_GUIDE.md) for complete instructions.
