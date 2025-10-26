# Embedding Configuration Guide

This guide shows how to configure embedding providers in GraphRAG Core using TOML configuration files.

## Overview

GraphRAG Core supports **8 embedding providers** via a unified configuration system:

1. **HuggingFace Hub** - Free, downloadable models (offline-capable)
2. **OpenAI** - Production-grade API embeddings
3. **Voyage AI** - Recommended by Anthropic
4. **Cohere** - Multilingual support
5. **Jina AI** - Cost-optimized ($0.02/1M tokens)
6. **Mistral AI** - RAG-optimized embeddings
7. **Together AI** - Cheapest option ($0.008/1M tokens)
8. **Ollama** - Local LLM embeddings with GPU support

## Quick Start

### Option 1: TOML Configuration (Recommended)

Add an `[embeddings]` section to your GraphRAG configuration file:

```toml
# config.toml

[embeddings]
backend = "huggingface"
model = "sentence-transformers/all-MiniLM-L6-v2"
dimension = 384
batch_size = 32
cache_dir = "~/.cache/huggingface"
```

### Option 2: Standalone Embeddings Config

Use the dedicated embeddings configuration file (see `examples/embeddings.toml`):

```toml
# embeddings.toml

[embeddings]
provider = "openai"
model = "text-embedding-3-small"
api_key = "sk-..."  # Or set OPENAI_API_KEY env var
batch_size = 100
dimensions = 1536
```

## Configuration by Provider

### 1. HuggingFace Hub (Free, Offline)

**Features:**
- ✅ Free, no API key required
- ✅ Offline-capable (models cached locally)
- ✅ Wide selection of models
- ⚠️ First run downloads model (~100MB)

**Configuration:**
```toml
[embeddings]
backend = "huggingface"
model = "sentence-transformers/all-MiniLM-L6-v2"  # Default, recommended
# model = "BAAI/bge-large-en-v1.5"  # High quality alternative
# model = "intfloat/e5-small-v2"    # E5 family
dimension = 384  # Depends on model
batch_size = 32
cache_dir = "~/.cache/huggingface"  # Optional
```

**Recommended Models:**

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | ⚡⚡⚡ | ★★★ | General (default) |
| all-mpnet-base-v2 | 768 | ⚡⚡ | ★★★★ | Balanced |
| BAAI/bge-large-en-v1.5 | 1024 | ⚡ | ★★★★★ | Best quality |
| paraphrase-multilingual-MiniLM | 384 | ⚡⚡⚡ | ★★★ | 50+ languages |

**Enable Feature:**
```bash
cargo build --features huggingface-hub
```

---

### 2. OpenAI (Production Quality)

**Features:**
- ✅ Best quality embeddings
- ✅ High throughput (100-2048 batch size)
- 💰 $0.13 per 1M tokens

**Configuration:**
```toml
[embeddings]
backend = "openai"
model = "text-embedding-3-small"  # or "text-embedding-3-large"
api_key = "sk-..."  # Or set OPENAI_API_KEY env var
dimension = 1536  # 3-small: 1536, 3-large: 3072
batch_size = 100
```

**Environment Variable:**
```bash
export OPENAI_API_KEY="sk-..."
```

---

### 3. Voyage AI (Recommended by Anthropic)

**Features:**
- ✅ Recommended by Anthropic
- ✅ Domain-specific models (code, finance, law)
- ✅ High quality for RAG applications
- 💰 Competitive pricing

**Configuration:**
```toml
[embeddings]
backend = "voyage"  # or "voyageai" or "voyage-ai"
model = "voyage-3-large"  # General purpose
# model = "voyage-code-3"     # Code search
# model = "voyage-finance-2"  # Finance documents
# model = "voyage-law-2"      # Legal documents
api_key = "pa-..."  # Or set VOYAGE_API_KEY env var
dimension = 1024
batch_size = 128
```

**Environment Variable:**
```bash
export VOYAGE_API_KEY="pa-..."
```

**Available Models:**

| Model | Dimensions | Use Case |
|-------|------------|----------|
| voyage-3-large | 1024 | General purpose (best) |
| voyage-code-3 | 1024 | Code search |
| voyage-finance-2 | 1024 | Finance documents |
| voyage-law-2 | 1024 | Legal documents |

---

### 4. Cohere (Multilingual)

**Features:**
- ✅ Excellent multilingual support (100+ languages)
- ✅ Fast inference
- 💰 $0.10 per 1M tokens

**Configuration:**
```toml
[embeddings]
backend = "cohere"
model = "embed-english-v3.0"       # English only
# model = "embed-multilingual-v3.0"  # 100+ languages
# model = "embed-english-light-v3.0" # Faster, smaller
api_key = "..."  # Or set COHERE_API_KEY env var
dimension = 1024
batch_size = 96  # Max per request
```

**Environment Variable:**
```bash
export COHERE_API_KEY="..."
```

---

### 5. Jina AI (Cost Optimized)

**Features:**
- ✅ Very cost-effective ($0.02/1M tokens)
- ✅ High throughput (200+ batch size)
- ✅ Multimodal support (v4)
- 💰 Best price/performance ratio

**Configuration:**
```toml
[embeddings]
backend = "jina"  # or "jinaai" or "jina-ai"
model = "jina-embeddings-v3"
# model = "jina-embeddings-v4"  # Multimodal (text + images)
# model = "jina-clip-v2"        # Text-to-image
api_key = "jina_..."  # Or set JINA_API_KEY env var
dimension = 1024
batch_size = 200
```

**Environment Variable:**
```bash
export JINA_API_KEY="jina_..."
```

---

### 6. Mistral AI (RAG Optimized)

**Features:**
- ✅ Optimized for RAG applications
- ✅ Code embedding support
- 💰 $0.10 per 1M tokens

**Configuration:**
```toml
[embeddings]
backend = "mistral"  # or "mistralai" or "mistral-ai"
model = "mistral-embed"
# model = "codestral-embed"  # For code
api_key = "..."  # Or set MISTRAL_API_KEY env var
dimension = 1024
batch_size = 50
```

**Environment Variable:**
```bash
export MISTRAL_API_KEY="..."
```

---

### 7. Together AI (Cheapest)

**Features:**
- ✅ Most cost-effective ($0.008/1M tokens)
- ✅ High throughput
- ✅ Multiple model choices
- 💰 Best for large-scale deployments

**Configuration:**
```toml
[embeddings]
backend = "together"  # or "togetherai" or "together-ai"
model = "BAAI/bge-large-en-v1.5"
# model = "BAAI/bge-base-en-v1.5"
# model = "WhereIsAI/UAE-Large-V1"
api_key = "..."  # Or set TOGETHER_API_KEY env var
dimension = 1024
batch_size = 128
```

**Environment Variable:**
```bash
export TOGETHER_API_KEY="..."
```

---

### 8. Ollama (Local, GPU-Accelerated)

**Features:**
- ✅ Local inference with GPU support
- ✅ No API costs
- ✅ Privacy-first (data never leaves your machine)
- ⚠️ Requires Ollama installation

**Configuration:**
```toml
[embeddings]
backend = "ollama"
model = "nomic-embed-text"
dimension = 768
batch_size = 32

[ollama]
enabled = true
host = "http://localhost"
port = 11434
embedding_model = "nomic-embed-text"
```

**Setup:**
```bash
# Install and start Ollama
ollama serve &
ollama pull nomic-embed-text
```

---

## Integration with GraphRAG Config

### Full Configuration Example

```toml
# my_config.toml - Complete GraphRAG configuration with embeddings

[general]
input_document_path = "path/to/document.txt"
output_dir = "./output/my_project"

[pipeline]
chunk_size = 800
chunk_overlap = 200

# Embedding configuration
[embeddings]
backend = "huggingface"  # or openai, voyage, cohere, jina, mistral, together, ollama
model = "sentence-transformers/all-MiniLM-L6-v2"
dimension = 384
batch_size = 32
fallback_to_hash = true  # Fallback if provider fails
cache_dir = "~/.cache/huggingface"

# If using API providers, set api_key here or via environment variable
# api_key = "your-api-key-here"

[graph]
max_connections = 10
similarity_threshold = 0.8

[retrieval]
top_k = 10
search_algorithm = "cosine"
```

### Using Multiple Configs

You can have different config files for different providers:

```bash
# For development (free, offline)
cp config_huggingface.toml dev_config.toml

# For production (high quality)
cp config_openai.toml prod_config.toml

# For cost optimization
cp config_together.toml budget_config.toml
```

---

## Programmatic Usage

### Using from Rust Code

```rust
use graphrag_core::config::Config;
use graphrag_core::embeddings::config::EmbeddingProviderConfig;

// Load main config
let config = Config::from_toml_file("config.toml")?;

// Access embedding settings
println!("Backend: {}", config.embeddings.backend);
println!("Model: {:?}", config.embeddings.model);
println!("Dimensions: {}", config.embeddings.dimension);

// Or use standalone embeddings config
let emb_config = EmbeddingProviderConfig::from_toml_file("embeddings.toml")?;
let embedding_config = emb_config.to_embedding_config()?;

// Create provider from config
#[cfg(feature = "huggingface-hub")]
{
    use graphrag_core::embeddings::huggingface::HuggingFaceEmbeddings;
    let mut hf = HuggingFaceEmbeddings::from_config(&embedding_config);
    hf.initialize().await?;
}

#[cfg(feature = "ureq")]
{
    use graphrag_core::embeddings::api_providers::HttpEmbeddingProvider;
    let provider = HttpEmbeddingProvider::from_config(&embedding_config)?;
    let embedding = provider.embed("Your text").await?;
}
```

---

## Provider Selection Guide

### By Use Case

| Use Case | Provider | Reason |
|----------|----------|--------|
| **Development/Testing** | HuggingFace | Free, offline, no setup |
| **Production (Quality)** | OpenAI or Voyage | Best accuracy |
| **Production (Cost)** | Together AI | Most cost-effective |
| **Multilingual** | Cohere | 100+ languages |
| **Code Search** | Voyage (code-3) | Optimized for code |
| **Privacy/Offline** | HuggingFace or Ollama | Local inference |
| **Finance/Legal** | Voyage (finance-2/law-2) | Domain-specific |

### By Budget

| Budget | Monthly Tokens | Provider | Cost |
|--------|---------------|----------|------|
| **Free** | Unlimited | HuggingFace | $0 |
| **Low** | <100M | Together AI | $0.80 |
| **Medium** | 100M-1B | Jina AI | $2-20 |
| **High** | >1B | OpenAI, Voyage | $20-130+ |

### By Performance

| Metric | Best Choice | Performance |
|--------|-------------|-------------|
| **Speed** | ONNX Web (WASM) | 3-8ms per chunk |
| **Quality** | OpenAI 3-large | MTEB: 64.6 |
| **Throughput** | Jina AI | 200+ batch size |
| **Latency** | Local (HF/Ollama) | No network delay |

---

## Environment Variables Reference

Instead of putting API keys in config files, use environment variables:

```bash
# API Keys (recommended method)
export OPENAI_API_KEY="sk-..."
export VOYAGE_API_KEY="pa-..."
export COHERE_API_KEY="..."
export JINA_API_KEY="jina_..."
export MISTRAL_API_KEY="..."
export TOGETHER_API_KEY="..."

# HuggingFace cache location (optional)
export HF_HOME="~/.cache/huggingface"
export HF_HUB_CACHE="~/.cache/huggingface/hub"

# Enable model downloads in tests
export ENABLE_DOWNLOAD_TESTS=1
```

---

## Examples

### Run Examples

```bash
# 1. Basic embeddings demo (all providers)
cd graphrag-core
cargo run --example embeddings_demo --features "huggingface-hub,ureq"

# 2. Config-based usage
cargo run --example embeddings_from_config --features "huggingface-hub,ureq"

# 3. With API keys
OPENAI_API_KEY=sk-... \
VOYAGE_API_KEY=pa-... \
cargo run --example embeddings_demo --features ureq
```

### Example Files

- `examples/embeddings.toml` - Comprehensive config with all providers
- `examples/embeddings_demo.rs` - Test all providers
- `examples/embeddings_from_config.rs` - Load and use config
- `config-huggingface.toml` - HuggingFace setup
- `config-openai.toml` - OpenAI setup
- `config-voyage.toml` - Voyage AI setup

---

## Troubleshooting

### HuggingFace: Model Download Fails

```bash
# Check internet connection
ping huggingface.co

# Set cache directory explicitly
export HF_HOME="/path/to/cache"

# Enable debug logging
RUST_LOG=debug cargo run --example embeddings_demo
```

### API Providers: Authentication Error

```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Test API key
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models
```

### Build Errors: Feature Not Enabled

```bash
# Enable required features
cargo build --features "huggingface-hub,ureq"

# Check enabled features
cargo tree --features huggingface-hub
```

---

## Migration Guide

### From Old Config Format

**Old:**
```toml
[embeddings]
dimension = 384
backend = "hash"
```

**New:**
```toml
[embeddings]
backend = "huggingface"
model = "sentence-transformers/all-MiniLM-L6-v2"
dimension = 384
batch_size = 32
cache_dir = "~/.cache/huggingface"
fallback_to_hash = true  # Keeps compatibility
```

---

## See Also

- [Embeddings README](src/embeddings/README.md) - Technical details
- [HOW_IT_WORKS.md](../HOW_IT_WORKS.md) - Pipeline overview
- [LLM_PROVIDERS.md](../graphrag-wasm/LLM_PROVIDERS.md) - Provider comparison

---

**Last Updated:** 2025-10-07
**GraphRAG Core Version:** 0.1.0
