# LLM Providers for GraphRAG

This document lists OpenAI-compatible API providers that can be used for natural language answer synthesis in the GraphRAG pipeline.

## Overview

All providers listed below support OpenAI-compatible API formats, making them drop-in replacements for the LLM synthesis component. The modular trait-based architecture allows easy switching between providers.

## Multi-Provider Aggregators

### OpenRouter
- **Website**: https://openrouter.ai
- **Models**: 300+ models from multiple providers
- **Features**: Single API for accessing various LLM providers
- **Pricing**: Pay-per-use, varies by model
- **Best For**: Flexibility, accessing multiple models without multiple API keys

### LiteLLM
- **Website**: https://litellm.ai
- **Models**: 100+ models from 50+ providers
- **Features**: Unified interface, load balancing, fallbacks
- **Pricing**: Open-source proxy, pay provider costs
- **Best For**: Self-hosted unified gateway

### Portkey
- **Website**: https://portkey.ai
- **Models**: All major providers
- **Features**: AI gateway, fallbacks, caching, analytics
- **Pricing**: Freemium with usage-based pricing
- **Best For**: Production deployments with reliability requirements

## High-Performance Inference

### Groq
- **Website**: https://groq.com
- **Models**: Llama 3.x, Mixtral, Gemma
- **Features**: Ultra-fast inference with LPU technology
- **Performance**: ~0.13s first token, 500+ tok/s
- **Pricing**: Generous free tier, competitive paid
- **Best For**: Speed-critical applications

### Together AI
- **Website**: https://together.ai
- **Models**: 200+ open-source models
- **Features**: Sub-100ms latency, custom deployments
- **Pricing**: Pay-per-token, volume discounts
- **Best For**: Open-source models at scale

### Fireworks AI
- **Website**: https://fireworks.ai
- **Models**: Llama, Mistral, StarCoder, and more
- **Features**: Fast inference, fine-tuning support
- **Pricing**: Competitive per-token pricing
- **Best For**: Production inference for popular models

### Cerebras
- **Website**: https://cerebras.ai
- **Models**: Llama 3.x, Mistral
- **Features**: Specialized hardware acceleration
- **Performance**: Very high throughput
- **Pricing**: Contact for enterprise pricing
- **Best For**: Large-scale deployments

### SambaNova
- **Website**: https://sambanova.ai
- **Models**: Llama, custom models
- **Features**: High-performance cloud inference
- **Performance**: ~0.13s first token (fastest tier)
- **Pricing**: Enterprise pricing
- **Best For**: Enterprise deployments with performance needs

## Major AI Providers

### Anthropic
- **Website**: https://anthropic.com
- **Models**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
- **Features**: Long context (200K tokens), vision, function calling
- **Pricing**: Usage-based, different rates per model
- **Best For**: High-quality reasoning, long documents, safety-focused applications
- **API Format**: Native API + OpenAI-compatible endpoints

### OpenAI
- **Website**: https://openai.com
- **Models**: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
- **Features**: Function calling, vision, embeddings
- **Pricing**: Usage-based per model
- **Best For**: General-purpose applications, established ecosystem
- **API Format**: Native OpenAI format (the standard)

### DeepSeek
- **Website**: https://deepseek.com
- **Models**: DeepSeek-V3, DeepSeek-R1
- **Features**: Reasoning models, competitive pricing
- **Pricing**: Very cost-effective
- **Best For**: Cost-conscious deployments, reasoning tasks
- **API Format**: OpenAI-compatible

### Perplexity
- **Website**: https://perplexity.ai
- **Models**: Llama 3.1 Sonar, Perplexity custom models
- **Features**: Search-focused, web grounding
- **Pricing**: Subscription + API credits
- **Best For**: Search-enhanced responses, factual queries
- **API Format**: OpenAI-compatible

## Cost-Effective Providers

### DeepInfra
- **Website**: https://deepinfra.com
- **Models**: 50+ open-source models
- **Features**: Cost-effective inference
- **Pricing**: Very competitive per-token rates
- **Best For**: Budget-conscious projects

### Hugging Face
- **Website**: https://huggingface.co
- **Models**: Thousands of open-source models
- **Features**: Inference API, free tier available
- **Pricing**: Free tier + pro tiers
- **Best For**: Experimenting with various models

### Novita AI
- **Website**: https://novita.ai
- **Models**: 200+ APIs, various model families
- **Features**: Serverless GPU, no cold starts
- **Pricing**: Pay-as-you-go
- **Best For**: Serverless deployments

## Specialized Providers

### Replicate
- **Website**: https://replicate.com
- **Models**: Open-source models (Llama, SDXL, etc.)
- **Features**: Run models via API, custom deployments
- **Pricing**: Per-second billing
- **Best For**: Open-source model hosting

### Anyscale
- **Website**: https://anyscale.com
- **Models**: Ray-based inference platform
- **Features**: Scalable inference, fine-tuning
- **Pricing**: Enterprise pricing
- **Best For**: Ray ecosystem users

## Local/Browser Inference

### WebLLM (Current Implementation)
- **Website**: https://webllm.mlc.ai
- **Models**: Phi-3, Llama 3.2, Qwen2, Gemma
- **Features**: Browser-based inference with WebGPU
- **Performance**: 40-62 tok/s with WebGPU
- **Pricing**: Free (runs in browser)
- **Best For**: Privacy-focused, offline applications, demos

### Ollama (Local)
- **Website**: https://ollama.ai
- **Models**: Llama, Mistral, Phi, many others
- **Features**: Local inference, OpenAI-compatible API
- **Performance**: Depends on hardware
- **Pricing**: Free (runs locally)
- **Best For**: Privacy, offline use, development

### ONNX Runtime (Local)
- **Models**: Custom ONNX-exported models
- **Features**: Cross-platform inference
- **Performance**: Hardware-dependent
- **Pricing**: Free (runs locally)
- **Best For**: Custom models, embedded deployments

## Performance Comparison

| Provider | First Token Latency | Throughput | Cost (GPT-3.5 equiv) |
|----------|-------------------|------------|---------------------|
| **Groq** | ~0.13s | 500+ tok/s | Low |
| **SambaNova** | ~0.13s | 400+ tok/s | Medium |
| **Together AI** | <0.1s | 300+ tok/s | Low |
| **Fireworks** | <0.2s | 250+ tok/s | Low |
| **Cerebras** | <0.15s | 350+ tok/s | Medium |
| **Anthropic** | ~0.5s | 100+ tok/s | Medium-High |
| **OpenAI** | ~0.4s | 80+ tok/s | Medium |
| **DeepSeek** | ~0.3s | 150+ tok/s | Very Low |
| **WebLLM** | N/A | 40-62 tok/s | Free |
| **Ollama** | Variable | Variable | Free |

*Note: Performance varies by model size, hardware, and network conditions*

## Cost Comparison (Approximate)

| Provider | Input (per 1M tokens) | Output (per 1M tokens) |
|----------|----------------------|------------------------|
| **DeepSeek** | $0.14 | $0.28 |
| **Groq** | $0.05 | $0.08 |
| **Together AI** | $0.20 | $0.20 |
| **DeepInfra** | $0.13 | $0.13 |
| **Anthropic (Haiku)** | $0.25 | $1.25 |
| **Anthropic (Sonnet)** | $3.00 | $15.00 |
| **OpenAI (GPT-3.5)** | $0.50 | $1.50 |
| **OpenAI (GPT-4)** | $10.00 | $30.00 |
| **WebLLM** | Free | Free |
| **Ollama** | Free | Free |

*Prices as of 2025 - check provider websites for current rates*

## Integration Architecture

All providers can be integrated using a common trait-based architecture:

```rust
#[async_trait(?Send)]
pub trait LLMProvider {
    async fn initialize(&mut self) -> Result<(), LLMError>;
    async fn synthesize(&self, query: &str, context: &str) -> Result<String, LLMError>;
    fn is_available(&self) -> bool;
    fn provider_name(&self) -> &str;
}
```

### Example Implementations

#### Anthropic Provider
```rust
pub struct AnthropicProvider {
    api_key: String,
    model: String, // e.g., "claude-3-5-sonnet-20241022"
    client: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait(?Send)]
impl LLMProvider for AnthropicProvider {
    async fn synthesize(&self, query: &str, context: &str) -> Result<String, LLMError> {
        let response = self.client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&json!({
                "model": self.model,
                "max_tokens": 512,
                "messages": [{
                    "role": "user",
                    "content": format!("Question: {}\n\nContext: {}", query, context)
                }]
            }))
            .send()
            .await?;

        // Parse response...
        Ok(answer)
    }
}
```

#### Groq Provider
```rust
pub struct GroqProvider {
    api_key: String,
    model: String, // e.g., "llama-3.1-70b-versatile"
    client: reqwest::Client,
}

#[async_trait(?Send)]
impl LLMProvider for GroqProvider {
    async fn synthesize(&self, query: &str, context: &str) -> Result<String, LLMError> {
        let response = self.client
            .post("https://api.groq.com/openai/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&json!({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant..."},
                    {"role": "user", "content": format!("Question: {}\n\nContext: {}", query, context)}
                ],
                "temperature": 0.7,
                "max_tokens": 512
            }))
            .send()
            .await?;

        // Parse OpenAI-compatible response...
        Ok(answer)
    }
}
```

#### OpenRouter Provider (Multi-Model)
```rust
pub struct OpenRouterProvider {
    api_key: String,
    model: String, // e.g., "anthropic/claude-3.5-sonnet"
    client: reqwest::Client,
}

#[async_trait(?Send)]
impl LLMProvider for OpenRouterProvider {
    async fn synthesize(&self, query: &str, context: &str) -> Result<String, LLMError> {
        let response = self.client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&json!({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant..."},
                    {"role": "user", "content": format!("Question: {}\n\nContext: {}", query, context)}
                ]
            }))
            .send()
            .await?;

        Ok(answer)
    }
}
```

## Recommendation Matrix

| Use Case | Recommended Provider | Alternative |
|----------|---------------------|-------------|
| **Production (Quality)** | Anthropic Claude 3.5 Sonnet | OpenAI GPT-4 |
| **Production (Speed)** | Groq | Together AI |
| **Production (Cost)** | DeepSeek | DeepInfra |
| **Development/Testing** | Groq (free tier) | WebLLM (local) |
| **Privacy/Offline** | Ollama | WebLLM |
| **Research** | Hugging Face | Replicate |
| **Multi-Model Flexibility** | OpenRouter | LiteLLM |
| **Enterprise** | Anthropic | SambaNova |

## Configuration Example

```rust
// In your GraphRAG config
pub struct LLMConfig {
    pub provider: LLMProviderType,
    pub api_key: Option<String>,
    pub model: String,
    pub temperature: f32,
    pub max_tokens: u32,
}

pub enum LLMProviderType {
    WebLLM,
    Anthropic,
    OpenAI,
    Groq,
    DeepSeek,
    Together,
    Ollama,
    OpenRouter,
    Custom(Box<dyn LLMProvider>),
}
```

## Best Practices

1. **API Key Security**: Never hardcode API keys, use environment variables
2. **Rate Limiting**: Implement backoff for API providers
3. **Fallbacks**: Configure backup providers for production
4. **Caching**: Cache LLM responses for repeated queries
5. **Monitoring**: Track costs and latency per provider
6. **Error Handling**: Graceful degradation when LLM unavailable

## Future Enhancements

- [ ] Multi-provider fallback system
- [ ] Automatic provider selection based on query type
- [ ] Cost tracking and budgeting
- [ ] Response caching layer
- [ ] A/B testing between providers
- [ ] Streaming support for all providers
- [ ] Custom fine-tuned models

## Embedding Models

Many providers also offer embedding models for vector search in GraphRAG pipelines. Here's a comprehensive list:

### OpenAI Embeddings
- **Website**: https://platform.openai.com/docs/models/embeddings
- **Models**:
  - `text-embedding-3-large`: 3072 dimensions, most capable
  - `text-embedding-3-small`: 1536 dimensions, cost-effective
  - `text-embedding-ada-002`: 1536 dimensions (legacy)
- **Features**: Multilingual, high quality
- **Pricing**: $0.13/1M tokens (small), $0.02/1M tokens (large)
- **Best For**: Production applications, established ecosystem

### Voyage AI Embeddings (Recommended by Anthropic)
- **Website**: https://docs.voyageai.com
- **Models**:
  - `voyage-3-large`: 1024D, general-purpose, multilingual (default)
  - `voyage-3.5`: Optimized general-purpose, multilingual
  - `voyage-3.5-lite`: Optimized for latency and cost
  - `voyage-code-3`: Specialized for code retrieval
  - `voyage-finance-2`: Optimized for finance
  - `voyage-law-2`: Specialized for legal documents
- **Features**: 32K context length, flexible dimensions (256-2048)
- **Pricing**: Competitive pricing, free tier available
- **Best For**: Domain-specific applications (code, finance, legal)

### Cohere Embeddings
- **Website**: https://docs.cohere.com/docs/embeddings
- **Models**:
  - `embed-v4`: Latest version, multimodal support
  - `embed-english-v3.0`: 1024D, MTEB 64.5, BEIR 55.9
  - `embed-multilingual-v3.0`: Multilingual support
  - `embed-english-light-v3.0`: Lightweight version
- **Features**: Multiple embedding types (int8, binary, base64), image support
- **Pricing**: Free tier + usage-based
- **Best For**: Multilingual applications, compression-sensitive use cases

### Jina AI Embeddings
- **Website**: https://jina.ai/embeddings
- **Models**:
  - `jina-embeddings-v4`: 3.8B parameters, multimodal (text + images)
  - `jina-clip-v2`: Text-to-image, image-to-text, 89 languages
  - `jina-embeddings-v3`: Multilingual, 2nd on MTEB English leaderboard
- **Features**: 8192 token input, multimodal, 89 languages
- **API**: `https://api.jina.ai/v1/embeddings`
- **Best For**: Multimodal applications, multilingual search

### Mistral AI Embeddings
- **Website**: https://docs.mistral.ai/capabilities/embeddings
- **Models**:
  - `mistral-embed`: 8K context, optimized for RAG
  - `codestral-embed`: Specialized for code retrieval
- **Features**: Large context window, code-optimized option
- **Pricing**: Competitive pricing
- **Best For**: RAG applications, code search

### Together AI Embeddings
- **Website**: https://docs.together.ai/docs/embedding-models
- **Models**:
  - `BAAI/bge-base-en-v1.5`: Base model
  - `BAAI/bge-large-en-v1.5`: Large model
  - `WhereIsAI/UAE-Large-V1`: 1024D embeddings
- **Features**: OpenAI-compatible API, multiple open-source models
- **Pricing**: Pay-per-use
- **Best For**: Open-source model preference

### Hugging Face Embeddings
- **Website**: https://huggingface.co/models?pipeline_tag=sentence-similarity
- **Models**: Thousands of open-source models including:
  - `sentence-transformers/all-MiniLM-L6-v2`: 384D (fast, lightweight)
  - `sentence-transformers/all-mpnet-base-v2`: 768D (high quality)
  - `BAAI/bge-large-en-v1.5`: 1024D
- **Features**: Free inference API, self-hosted options
- **Best For**: Experimentation, self-hosted deployments

### Local Embeddings (Current Implementation)
- **ONNX Runtime Web**: MiniLM-L6 (384D) with WebGPU acceleration
- **Ollama**: Various models (nomic-embed-text, mxbai-embed-large)
- **Features**: Privacy-focused, offline capable, no API costs
- **Best For**: Privacy requirements, offline applications

### Providers WITHOUT Embeddings

❌ **Anthropic**: No native embeddings (recommends Voyage AI)
❌ **Groq**: No embeddings API (only LLM inference)
❌ **DeepSeek**: No embeddings API found
❌ **Fireworks AI**: Primarily LLM inference
❌ **Cerebras**: No embeddings API
❌ **SambaNova**: No embeddings API

## Embeddings Comparison

| Provider | Best Model | Dimensions | Context | Price (per 1M tokens) | MTEB Score |
|----------|-----------|------------|---------|----------------------|------------|
| **OpenAI** | text-embedding-3-large | 3072 | 8K | $0.13 | ~64.6 |
| **Voyage AI** | voyage-3-large | 1024 | 32K | Competitive | ~65.0 |
| **Cohere** | embed-english-v3.0 | 1024 | 512 | $0.10 | 64.5 |
| **Jina AI** | jina-embeddings-v3 | 1024 | 8K | $0.02 | ~63.0 |
| **Mistral** | mistral-embed | 1024 | 8K | $0.10 | ~62.0 |
| **Together AI** | BAAI/bge-large-en-v1.5 | 1024 | 512 | $0.008 | 63.98 |
| **ONNX (Local)** | MiniLM-L6 | 384 | 512 | Free | ~56.0 |

*MTEB = Massive Text Embedding Benchmark*

## Embedding Provider Trait

```rust
#[async_trait(?Send)]
pub trait EmbeddingProvider {
    async fn initialize(&mut self) -> Result<(), EmbeddingError>;
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;
    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, EmbeddingError>;
    fn dimensions(&self) -> usize;
    fn is_available(&self) -> bool;
    fn provider_name(&self) -> &str;
}
```

### Example: Voyage AI Embeddings

```rust
pub struct VoyageEmbeddings {
    api_key: String,
    model: String, // e.g., "voyage-3-large"
    client: reqwest::Client,
}

#[async_trait(?Send)]
impl EmbeddingProvider for VoyageEmbeddings {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let response = self.client
            .post("https://api.voyageai.com/v1/embeddings")
            .bearer_auth(&self.api_key)
            .json(&json!({
                "model": self.model,
                "input": text,
                "input_type": "document"
            }))
            .send()
            .await?;

        let result: VoyageResponse = response.json().await?;
        Ok(result.data[0].embedding)
    }

    fn dimensions(&self) -> usize {
        1024
    }
}
```

### Example: Cohere Embeddings

```rust
pub struct CohereEmbeddings {
    api_key: String,
    model: String, // e.g., "embed-english-v3.0"
    client: reqwest::Client,
}

#[async_trait(?Send)]
impl EmbeddingProvider for CohereEmbeddings {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let response = self.client
            .post("https://api.cohere.ai/v1/embed")
            .bearer_auth(&self.api_key)
            .json(&json!({
                "model": self.model,
                "texts": [text],
                "input_type": "search_document",
                "embedding_types": ["float"]
            }))
            .send()
            .await?;

        let result: CohereResponse = response.json().await?;
        Ok(result.embeddings[0].clone())
    }

    fn dimensions(&self) -> usize {
        1024
    }
}
```

## Embeddings Recommendation Matrix

| Use Case | Recommended Provider | Alternative |
|----------|---------------------|-------------|
| **General Purpose** | Voyage AI voyage-3-large | OpenAI text-embedding-3-large |
| **Code Search** | Voyage AI voyage-code-3 | Mistral codestral-embed |
| **Multilingual** | Cohere embed-multilingual-v3.0 | Jina jina-embeddings-v3 |
| **Cost-Optimized** | Together AI BAAI/bge-large | Jina AI jina-embeddings-v3 |
| **Domain-Specific (Finance)** | Voyage AI voyage-finance-2 | Custom fine-tuned |
| **Domain-Specific (Legal)** | Voyage AI voyage-law-2 | Custom fine-tuned |
| **Privacy/Offline** | ONNX Runtime (MiniLM) | Ollama nomic-embed-text |
| **Multimodal** | Jina AI jina-embeddings-v4 | Cohere embed-v4 |

## References

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [Groq Documentation](https://console.groq.com/docs)
- [Anthropic API Reference](https://docs.anthropic.com/claude/reference)
- [Together AI Documentation](https://docs.together.ai)
- [WebLLM Documentation](https://webllm.mlc.ai)
- [Ollama Documentation](https://ollama.ai/docs)
- [Voyage AI Documentation](https://docs.voyageai.com)
- [Cohere Embeddings](https://docs.cohere.com/docs/embeddings)
- [Jina AI Embeddings](https://jina.ai/embeddings)
- [Mistral Embeddings](https://docs.mistral.ai/capabilities/embeddings)

## GraphRAG Core Integration

All embedding providers are now integrated in `graphrag-core` for easy use across the entire project:

### Available Modules

1. **`graphrag_core::embeddings::huggingface`** - Hugging Face Hub integration
   - Feature flag: `huggingface-hub`
   - Download and cache models from HF Hub
   - 11 recommended models pre-configured

2. **`graphrag_core::embeddings::api_providers`** - HTTP API providers
   - Feature flag: `ureq` (enabled by default)
   - OpenAI, Voyage AI, Cohere, Jina AI, Mistral, Together AI
   - Unified `HttpEmbeddingProvider` interface

### Usage Example

```rust
use graphrag_core::embeddings::{EmbeddingProvider, EmbeddingConfig, EmbeddingProviderType};
use graphrag_core::embeddings::api_providers::HttpEmbeddingProvider;

#[tokio::main]
async fn main() -> Result<()> {
    // Option 1: Direct creation
    let provider = HttpEmbeddingProvider::openai(
        "sk-...".to_string(),
        "text-embedding-3-small".to_string(),
    );

    let embedding = provider.embed("Your text here").await?;
    println!("Embedding dimensions: {}", provider.dimensions());

    // Option 2: From configuration
    let config = EmbeddingConfig {
        provider: EmbeddingProviderType::VoyageAI,
        model: "voyage-3-large".to_string(),
        api_key: Some("pa-...".to_string()),
        cache_dir: None,
        batch_size: 32,
    };

    let provider = HttpEmbeddingProvider::from_config(&config)?;
    let embeddings = provider.embed_batch(&["text1", "text2"]).await?;

    Ok(())
}
```

### Running the Demo

```bash
# Test Hugging Face Hub (downloads models)
ENABLE_DOWNLOAD_TESTS=1 cargo run --example embeddings_demo --features huggingface-hub

# Test API providers (requires API keys)
OPENAI_API_KEY=sk-... \
VOYAGE_API_KEY=pa-... \
COHERE_API_KEY=... \
cargo run --example embeddings_demo --features ureq
```

### Feature Flags

Enable embedding providers in your `Cargo.toml`:

```toml
[dependencies]
graphrag-core = { path = "../graphrag-core", features = ["huggingface-hub", "ureq"] }
```

Available features:
- `huggingface-hub` - Hugging Face Hub model downloads
- `ureq` - HTTP-based API providers (enabled by default)
- `neural-embeddings` - Local model inference with Candle (coming soon)

## License

This document is part of the GraphRAG project. See main repository LICENSE file.
