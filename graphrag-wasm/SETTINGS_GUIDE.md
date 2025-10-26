# Settings Guide - GraphRAG WASM Configuration

This guide explains how to configure embedding and LLM providers in GraphRAG WASM.

## Accessing Settings

Click the **"Settings"** tab in the navigation bar to access the configuration panel.

## Embedding Provider Configuration

### Available Providers

#### 1. ONNX Runtime Web (Local - Default)
**Best for**: Privacy, offline use, no API costs

- **Provider**: ONNX Runtime Web
- **Runs**: 100% in browser (WebGPU accelerated)
- **Models**:
  - `all-MiniLM-L6-v2` (384 dim, fast)
  - `all-mpnet-base-v2` (768 dim, quality)
- **API Key**: Not required ✅
- **Cost**: Free

**Setup**: No setup needed! Works out of the box.

#### 2. OpenAI
**Best for**: High-quality embeddings, production use

- **Provider**: OpenAI
- **Models**:
  - `text-embedding-3-small` (1536 dim, $0.02/1M tokens)
  - `text-embedding-3-large` (3072 dim, $0.13/1M tokens)
  - `text-embedding-ada-002` (1536 dim, legacy)
- **API Key**: Required (from platform.openai.com)
- **Cost**: Pay per token

**Setup**:
1. Get API key from https://platform.openai.com/api-keys
2. Select "OpenAI" provider
3. Choose model
4. Enter API key
5. Click "Save Settings"

#### 3. Voyage AI
**Best for**: State-of-the-art retrieval quality

- **Provider**: Voyage AI
- **Models**:
  - `voyage-3` (1024 dim, best quality)
  - `voyage-3-lite` (512 dim, fast)
  - `voyage-code-2` (1536 dim, for code)
- **API Key**: Required (from docs.voyageai.com)
- **Cost**: Pay per token

#### 4. Cohere
**Best for**: Multilingual support

- **Provider**: Cohere
- **Models**:
  - `embed-english-v3.0` (1024 dim)
  - `embed-multilingual-v3.0` (1024 dim)
  - `embed-english-light-v3.0` (384 dim)
- **API Key**: Required (from dashboard.cohere.com)
- **Cost**: Pay per token

#### 5. Jina AI
**Best for**: Long context embeddings

- **Provider**: Jina AI
- **Models**:
  - `jina-embeddings-v2-base-en` (768 dim)
  - `jina-embeddings-v2-small-en` (512 dim)
- **API Key**: Required (from jina.ai)
- **Cost**: Pay per token

#### 6. Mistral
**Best for**: European-based provider

- **Provider**: Mistral
- **Models**:
  - `mistral-embed` (1024 dim)
- **API Key**: Required (from console.mistral.ai)
- **Cost**: Pay per token

#### 7. Together AI
**Best for**: Open-source models

- **Provider**: Together AI
- **Models**:
  - `togethercomputer/m2-bert-80M-8k-retrieval` (768 dim)
  - `BAAI/bge-large-en-v1.5` (1024 dim)
- **API Key**: Required (from api.together.xyz)
- **Cost**: Pay per token

### Choosing an Embedding Provider

**For Privacy & Offline Use**:
- ✅ ONNX Runtime Web (local)

**For Best Quality**:
- ✅ Voyage AI (voyage-3)
- ✅ OpenAI (text-embedding-3-large)

**For Low Cost**:
- ✅ ONNX Runtime Web (free)
- ✅ OpenAI (text-embedding-3-small)

**For Multilingual**:
- ✅ Cohere (embed-multilingual-v3.0)

**For Code**:
- ✅ Voyage AI (voyage-code-2)

## LLM Provider Configuration

### Available Providers

#### 1. WebLLM (In-Browser - Default)
**Best for**: Privacy, offline use, no server setup

- **Provider**: WebLLM
- **Runs**: 100% in browser (WebGPU accelerated)
- **Models**:
  - `Llama-3.2-1B-Instruct-q4f16_1-MLC` (1.2GB, 62 tok/s)
  - `Phi-3-mini-4k-instruct-q4f16_1-MLC` (2.4GB, 40 tok/s)
  - `Qwen2-1.5B-Instruct-q4f16_1-MLC` (1.6GB, 50 tok/s)
  - `gemma-2b-it-q4f16_1-MLC` (2.0GB, 45 tok/s)
- **Server**: Not required ✅
- **Cost**: Free

**Pros**:
- ✅ Complete privacy (no data leaves browser)
- ✅ Works offline after first load
- ✅ GPU-accelerated via WebGPU
- ✅ No server setup

**Cons**:
- ⚠️ First load downloads model (~1-2GB)
- ⚠️ Requires modern browser with WebGPU
- ⚠️ Limited to smaller models (1-3B params)

**Setup**: No setup needed! Works out of the box.

#### 2. Ollama (Local Server)
**Best for**: Larger models, better quality, full GPU utilization

- **Provider**: Ollama HTTP
- **Runs**: Local Ollama server via HTTP
- **Models**:
  - `llama3.1:8b` (Best balance)
  - `qwen2.5:7b` (Reasoning)
  - `mistral:7b` (Fast)
  - `llama3.1:70b` (Highest quality)
  - `qwen2.5:32b` (Advanced)
- **Server**: Required (Ollama must be running)
- **Cost**: Free (runs locally)

**Pros**:
- ✅ Larger models (7B, 13B, 70B+)
- ✅ Better quality responses
- ✅ Full GPU utilization (CUDA/Metal)
- ✅ Works on older browsers

**Cons**:
- ⚠️ Requires Ollama server installation
- ⚠️ CORS configuration needed
- ⚠️ Data sent to localhost server

**Setup**:
1. Install Ollama:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. Pull a model:
   ```bash
   ollama pull llama3.1:8b
   ```

3. Start server with CORS:
   ```bash
   OLLAMA_ORIGINS="http://localhost:8080" ollama serve
   ```

4. In Settings:
   - Select "Ollama (Local Server)"
   - Choose model
   - Set endpoint: `http://localhost:11434`
   - Click "Save Settings"

### Choosing an LLM Provider

**For Privacy & Simplicity**:
- ✅ WebLLM (Phi-3 or Llama 3.2)

**For Best Quality**:
- ✅ Ollama (Llama 3.1 70B or Qwen 2.5 32B)

**For Speed**:
- ✅ WebLLM (Llama 3.2 1B)
- ✅ Ollama (Mistral 7B)

**For Reasoning**:
- ✅ Ollama (Qwen 2.5 7B or 32B)

## Temperature Control

The temperature slider controls creativity vs. precision:

- **0.0 - 0.3**: Precise, factual, consistent
  - Use for: Question answering, fact extraction

- **0.4 - 0.7**: Balanced (Default: 0.7)
  - Use for: General queries, summaries

- **0.8 - 1.0**: Creative, varied, exploratory
  - Use for: Creative writing, brainstorming

## Cache Settings

**Enable Model Caching** (Toggle):
- ✅ **ON**: Models cached in browser (faster subsequent loads)
- ❌ **OFF**: Models downloaded fresh each time

**Recommendation**: Keep ON for better performance.

## Saving Settings

Settings are automatically saved to **IndexedDB** when you click "💾 Save Settings".

### What Gets Saved:
- Embedding provider & model
- LLM provider & model
- Ollama endpoint
- Temperature
- Cache preferences
- API keys (stored securely in browser)

### Settings Persistence:
- Settings persist across browser sessions
- Settings are local to this browser/device
- Clearing browser data will reset settings

## API Key Security

### Security Best Practices:

✅ **Good**:
- API keys stored in browser's IndexedDB (encrypted)
- Keys never sent to any server except the provider's API
- Keys are local to your browser

⚠️ **Important**:
- Don't share screenshots with API keys visible
- Use read-only API keys when possible
- Rotate keys regularly
- Clear browser data to remove stored keys

### Provider-Specific Security:

**OpenAI**:
- Create project-specific keys
- Set usage limits
- Monitor usage at platform.openai.com

**Voyage AI**:
- Use separate keys per application
- Monitor at dashboard.voyageai.com

**Cohere**:
- Use trial keys for testing
- Production keys for production
- Monitor at dashboard.cohere.com

## Example Configurations

### Configuration 1: Privacy-Focused (Default)
```
Embedding Provider: ONNX Runtime Web
Embedding Model: all-MiniLM-L6-v2
LLM Provider: WebLLM
LLM Model: Phi-3-mini-4k-instruct-q4f16_1-MLC
Temperature: 0.7
Cache: ON
```
**Use case**: Personal use, no internet required after first load

### Configuration 2: Quality-Focused
```
Embedding Provider: Voyage AI
Embedding Model: voyage-3
API Key: [your-voyage-key]
LLM Provider: Ollama
LLM Model: llama3.1:70b
Endpoint: http://localhost:11434
Temperature: 0.7
Cache: ON
```
**Use case**: Research, high-quality analysis

### Configuration 3: Cost-Optimized
```
Embedding Provider: OpenAI
Embedding Model: text-embedding-3-small
API Key: [your-openai-key]
LLM Provider: WebLLM
LLM Model: Llama-3.2-1B-Instruct-q4f16_1-MLC
Temperature: 0.7
Cache: ON
```
**Use case**: Production with budget constraints

### Configuration 4: Multilingual
```
Embedding Provider: Cohere
Embedding Model: embed-multilingual-v3.0
API Key: [your-cohere-key]
LLM Provider: Ollama
LLM Model: qwen2.5:7b
Endpoint: http://localhost:11434
Temperature: 0.7
Cache: ON
```
**Use case**: Non-English documents

## Troubleshooting

### "Failed to save settings"
- **Cause**: IndexedDB not available or blocked
- **Solution**: Check browser privacy settings, allow storage

### "API Key invalid"
- **Cause**: Incorrect API key format
- **Solution**: Copy-paste key directly from provider dashboard

### "Ollama not available"
- **Cause**: Ollama server not running or CORS not configured
- **Solution**:
  ```bash
  OLLAMA_ORIGINS="http://localhost:8080" ollama serve
  ```

### "Model not loading" (WebLLM)
- **Cause**: Browser doesn't support WebGPU
- **Solution**: Use Chrome 113+, Edge 113+, or switch to Ollama

### "Embedding failed"
- **Cause**: Provider API error or rate limit
- **Solution**: Check API key, usage limits, or switch to ONNX

## Advanced Tips

1. **Mix and Match**: Use ONNX for embeddings (free) + Ollama for LLM (quality)

2. **Fallback Strategy**: Start with WebLLM, upgrade to Ollama when needed

3. **Cost Control**: Monitor API usage at provider dashboards

4. **Performance**: ONNX + WebLLM = No network latency

5. **Quality**: Voyage + Ollama 70B = Best possible results

## Need Help?

- 📖 **Main README**: [README.md](./README.md)
- 🤖 **Ollama Guide**: [OLLAMA_INTEGRATION.md](./OLLAMA_INTEGRATION.md)
- 🐛 **Issues**: [GitHub Issues](https://github.com/anthropics/claude-code/issues)

---

**Status**: ✅ Settings panel fully functional with 7 embedding providers and 2 LLM providers!
