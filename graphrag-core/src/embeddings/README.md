# Embedding Providers for GraphRAG Core

This module provides a unified interface for embedding generation across multiple providers.

## Overview

The `embeddings` module offers three types of providers:

1. **Hugging Face Hub** - Download and use free open-source models
2. **API Providers** - Use hosted embedding services (OpenAI, Voyage AI, Cohere, etc.)
3. **Local Models** (coming soon) - Run models locally with Candle/ONNX

All providers implement the `EmbeddingProvider` trait for consistency.

## Quick Start

### Hugging Face Hub (Free Models)

```rust
use graphrag_core::embeddings::huggingface::HuggingFaceEmbeddings;
use graphrag_core::embeddings::EmbeddingProvider;

#[tokio::main]
async fn main() -> Result<()> {
    let mut embeddings = HuggingFaceEmbeddings::new(
        "sentence-transformers/all-MiniLM-L6-v2",
        None, // Use default cache directory
    );

    // Download model (only once, then cached)
    embeddings.initialize().await?;

    // Generate embedding
    let embedding = embeddings.embed("Your text here").await?;
    println!("Dimensions: {}", embedding.len()); // 384

    Ok(())
}
```

**Cargo.toml:**
```toml
graphrag-core = { features = ["huggingface-hub"] }
```

### API Providers

```rust
use graphrag_core::embeddings::api_providers::HttpEmbeddingProvider;
use graphrag_core::embeddings::EmbeddingProvider;

#[tokio::main]
async fn main() -> Result<()> {
    // OpenAI
    let openai = HttpEmbeddingProvider::openai(
        std::env::var("OPENAI_API_KEY")?,
        "text-embedding-3-small".to_string(),
    );
    let embedding = openai.embed("Your text").await?;

    // Voyage AI (recommended by Anthropic)
    let voyage = HttpEmbeddingProvider::voyage_ai(
        std::env::var("VOYAGE_API_KEY")?,
        "voyage-3-large".to_string(),
    );
    let embedding = voyage.embed("Your text").await?;

    // Cohere
    let cohere = HttpEmbeddingProvider::cohere(
        std::env::var("COHERE_API_KEY")?,
        "embed-english-v3.0".to_string(),
    );
    let embedding = cohere.embed("Your text").await?;

    Ok(())
}
```

**Cargo.toml:**
```toml
graphrag-core = { features = ["ureq"] } # Enabled by default
```

### Configuration-Based

```rust
use graphrag_core::embeddings::{
    EmbeddingConfig, EmbeddingProviderType,
    api_providers::HttpEmbeddingProvider,
};

let config = EmbeddingConfig {
    provider: EmbeddingProviderType::VoyageAI,
    model: "voyage-3-large".to_string(),
    api_key: Some("pa-...".to_string()),
    cache_dir: None,
    batch_size: 32,
};

let provider = HttpEmbeddingProvider::from_config(&config)?;
```

## Trait: `EmbeddingProvider`

All providers implement this async trait:

```rust
#[async_trait::async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Initialize the provider (e.g., download models)
    async fn initialize(&mut self) -> Result<()>;

    /// Generate embedding for single text
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Generate embeddings for multiple texts
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Get embedding dimensions
    fn dimensions(&self) -> usize;

    /// Check if provider is ready
    fn is_available(&self) -> bool;

    /// Get provider name
    fn provider_name(&self) -> &str;
}
```

## Supported Providers

### Hugging Face Hub

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| **all-MiniLM-L6-v2** | 384 | ⚡⚡⚡ | ★★★ | General (default) |
| all-mpnet-base-v2 | 768 | ⚡⚡ | ★★★★ | High quality |
| BAAI/bge-small-en-v1.5 | 384 | ⚡⚡⚡ | ★★★ | Efficient |
| BAAI/bge-base-en-v1.5 | 768 | ⚡⚡ | ★★★★ | Balanced |
| BAAI/bge-large-en-v1.5 | 1024 | ⚡ | ★★★★★ | Best quality |
| intfloat/e5-small-v2 | 384 | ⚡⚡⚡ | ★★★ | E5 family |
| paraphrase-multilingual-MiniLM | 384 | ⚡⚡⚡ | ★★★ | 50+ languages |

**Get recommendations:**
```rust
let models = HuggingFaceEmbeddings::recommended_models();
for (id, desc, dims) in models {
    println!("{}: {} ({}D)", id, desc, dims);
}
```

### API Providers

| Provider | Best Model | Dimensions | Cost (per 1M tokens) |
|----------|-----------|------------|----------------------|
| **OpenAI** | text-embedding-3-large | 3072 | $0.13 |
| **Voyage AI** | voyage-3-large | 1024 | Competitive |
| **Cohere** | embed-english-v3.0 | 1024 | $0.10 |
| **Jina AI** | jina-embeddings-v3 | 1024 | $0.02 |
| **Mistral** | mistral-embed | 1024 | $0.10 |
| **Together AI** | BAAI/bge-large-en-v1.5 | 1024 | $0.008 |

### Domain-Specific Models (Voyage AI)

```rust
// Code search
let code = HttpEmbeddingProvider::voyage_ai(api_key, "voyage-code-3");

// Finance documents
let finance = HttpEmbeddingProvider::voyage_ai(api_key, "voyage-finance-2");

// Legal documents
let legal = HttpEmbeddingProvider::voyage_ai(api_key, "voyage-law-2");
```

## Examples

### Batch Processing

```rust
let texts = vec![
    "First document",
    "Second document",
    "Third document",
];

let embeddings = provider.embed_batch(&texts).await?;

for (i, embedding) in embeddings.iter().enumerate() {
    println!("Document {}: {} dimensions", i + 1, embedding.len());
}
```

### Error Handling

```rust
match provider.embed("text").await {
    Ok(embedding) => {
        println!("Success: {} dimensions", embedding.len());
    }
    Err(GraphRAGError::Embedding { message }) => {
        eprintln!("Embedding failed: {}", message);
    }
    Err(e) => {
        eprintln!("Unexpected error: {:?}", e);
    }
}
```

### Testing Multiple Providers

Run the full demo:

```bash
# Hugging Face (downloads model)
ENABLE_DOWNLOAD_TESTS=1 \
cargo run --example embeddings_demo --features huggingface-hub

# API providers (requires keys)
OPENAI_API_KEY=sk-... \
VOYAGE_API_KEY=pa-... \
COHERE_API_KEY=... \
JINA_API_KEY=jina_... \
MISTRAL_API_KEY=... \
TOGETHER_API_KEY=... \
cargo run --example embeddings_demo --features ureq
```

## Feature Flags

Enable in `Cargo.toml`:

```toml
[dependencies]
graphrag-core = {
    version = "0.1.0",
    features = [
        "huggingface-hub",  # Hugging Face Hub downloads
        "ureq",             # API providers (default)
        "neural-embeddings" # Local inference (coming soon)
    ]
}
```

## Environment Variables

### API Keys

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Voyage AI
export VOYAGE_API_KEY="pa-..."

# Cohere
export COHERE_API_KEY="..."

# Jina AI
export JINA_API_KEY="jina_..."

# Mistral AI
export MISTRAL_API_KEY="..."

# Together AI
export TOGETHER_API_KEY="..."
```

### Testing

```bash
# Enable HuggingFace model downloads in tests
export ENABLE_DOWNLOAD_TESTS=1
```

## Performance Tips

1. **Batch Processing**: Use `embed_batch()` when possible for better throughput
2. **Caching**: Hugging Face models are cached after first download
3. **Dimensions**: Smaller dimensions (384) are faster than larger (1024+)
4. **Provider Selection**:
   - Local/offline: Hugging Face Hub
   - Speed: Together AI, Jina AI
   - Quality: OpenAI, Voyage AI
   - Cost: Together AI ($0.008/1M)

## Roadmap

- [x] Hugging Face Hub integration
- [x] API providers (OpenAI, Voyage, Cohere, Jina, Mistral, Together)
- [x] Unified trait interface
- [x] Batch processing
- [ ] Local model inference with Candle
- [ ] ONNX Runtime integration
- [ ] Batch API optimization for supported providers
- [ ] Embedding caching layer
- [ ] Rate limiting and retry logic

## License

Part of the GraphRAG project. See repository LICENSE.
