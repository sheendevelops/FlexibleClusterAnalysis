# Ollama Setup Guide for AISapien

AISapien now supports **Ollama** for completely offline AI operation. This eliminates the need for internet connectivity and API costs while maintaining full functionality.

## Quick Setup

### 1. Install Ollama

**Linux/macOS:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from: https://ollama.com/download/windows

### 2. Download a Model

Choose one of these models based on your system:

**Recommended: Llama 3.1 (8B)**
```bash
ollama pull llama3.1
```

**For faster performance: Llama 3.2 (3B)**
```bash
ollama pull llama3.2
```

**For better quality: Llama 3.1 (70B)** (requires 64GB+ RAM)
```bash
ollama pull llama3.1:70b
```

### 3. Configure AISapien

Add to your `.env` file:
```env
# Use Ollama instead of OpenAI
LLM_BACKEND=ollama

# Ollama configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1

# Database (still required)
DATABASE_URL=postgresql://user:password@localhost:5432/aisapien
```

### 4. Start Services

```bash
# Start Ollama (if not running as service)
ollama serve

# In another terminal, start AISapien
streamlit run app.py --server.port 5000
```

## Model Recommendations

| Model | Size | RAM Required | Speed | Quality |
|-------|------|--------------|-------|---------|
| llama3.2:3b | 3B | 8GB | Fast | Good |
| llama3.1 | 8B | 16GB | Medium | Better |
| llama3.1:70b | 70B | 64GB | Slow | Best |
| mixtral | 8x7B | 32GB | Medium | Excellent |

## Switching Between Backends

You can easily switch between Ollama and OpenAI:

**Use Ollama (Offline):**
```env
LLM_BACKEND=ollama
OLLAMA_MODEL=llama3.1
```

**Use OpenAI (Online):**
```env
LLM_BACKEND=openai
OPENAI_API_KEY=your_key_here
```

## Troubleshooting

**Ollama not connecting?**
- Check if Ollama is running: `ollama list`
- Verify the service: `curl http://localhost:11434/api/version`
- Check firewall settings

**Model not found?**
- List available models: `ollama list`
- Pull the model: `ollama pull llama3.1`

**Slow responses?**
- Use a smaller model (llama3.2:3b)
- Check system resources (CPU/RAM usage)
- Consider GPU acceleration if available

**Memory issues?**
- Reduce model size
- Close other applications
- Use swap file for larger models

## Performance Optimization

### GPU Acceleration
If you have an NVIDIA GPU:
```bash
# Ollama automatically uses GPU if available
# Check GPU usage: nvidia-smi
```

### Model Management
```bash
# List downloaded models
ollama list

# Remove unused models
ollama rm model_name

# Update a model
ollama pull model_name
```

### System Tuning
- Allocate sufficient RAM
- Use SSD storage for better model loading
- Close unnecessary applications

## Benefits of Ollama

✅ **Complete Privacy**: All processing happens locally  
✅ **No API Costs**: No usage fees or token limits  
✅ **Offline Operation**: Works without internet  
✅ **Customizable**: Use any compatible model  
✅ **Performance**: Can be faster than API calls  

## Model Comparison

**For Development/Testing:** llama3.2:3b  
**For Production:** llama3.1  
**For Best Quality:** llama3.1:70b or mixtral  

## Integration Status

AISapien's Ollama integration includes:
- ✅ Conscience Model (ethics analysis)
- ✅ Logic Model (efficiency analysis) 
- ✅ Personality Model (user preferences)
- ✅ Master Model (decision synthesis)
- ✅ Content filtering and emotion detection
- ✅ Learning from documents and URLs
- ✅ Database storage of all interactions

All features work identically whether using Ollama or OpenAI.