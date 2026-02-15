# Ollama Integration Guide

## Overview

GraphRAG Core now includes complete integration with [Ollama](https://ollama.com/) for local LLM inference and embeddings. This integration provides:

- **Text Embeddings**: Generate vector embeddings using models like `nomic-embed-text`
- **Text Generation**: Chat and completion using models like `llama3.2`, `mistral`, etc.
- **Service Registry**: Automatic service registration through `ServiceConfig`

## Prerequisites

1. **Install Ollama**: Download from [ollama.com](https://ollama.com/)
2. **Pull Models**:
   ```bash
   ollama pull nomic-embed-text:latest
   ollama pull llama3.2:latest
   ```
3. **Start Ollama Server**:
   ```bash
   ollama serve  # Runs on http://localhost:11434
   ```

## Basic Usage

### Using ServiceConfig (Recommended)

The simplest way to use Ollama services:

```rust
use graphrag_core::core::{ServiceConfig, ServiceRegistry};

// Create configuration
let config = ServiceConfig {
    ollama_base_url: Some("http://localhost:11434".to_string()),
    embedding_model: Some("nomic-embed-text:latest".to_string()),
    language_model: Some("llama3.2:latest".to_string()),
    vector_dimension: Some(768),  // nomic-embed-text dimension
    ..Default::default()
};

// Build registry with Ollama services
let registry = config.build_registry().build();

// Services are now registered and ready to use!
```

### Using Adapters Directly

For more control, use the adapter types:

```rust
use graphrag_core::core::ollama_adapters::{
    OllamaEmbedderAdapter,
    OllamaLanguageModelAdapter
};
use graphrag_core::core::traits::{AsyncEmbedder, AsyncLanguageModel};
use graphrag_core::ollama::OllamaConfig;

// Create embedder
let embedder = OllamaEmbedderAdapter::new("nomic-embed-text:latest", 768);

// Generate embeddings
let embedding = embedder.embed("Hello, world!").await?;
println!("Embedding dimension: {}", embedding.len());

// Create language model
let ollama_config = OllamaConfig {
    enabled: true,
    host: "http://localhost".to_string(),
    port: 11434,
    chat_model: "llama3.2:latest".to_string(),
    ..Default::default()
};

let llm = OllamaLanguageModelAdapter::new(ollama_config);

// Generate text
let response = llm.complete("Explain quantum computing in simple terms").await?;
println!("Response: {}", response);
```

## Supported Models

### Embedding Models

| Model | Dimension | Use Case |
|-------|-----------|----------|
| `nomic-embed-text:latest` | 768 | General-purpose text embeddings |
| `mxbai-embed-large` | 1024 | High-quality embeddings |
| `all-minilm` | 384 | Lightweight, fast embeddings |

### Language Models

| Model | Size | Use Case |
|-------|------|----------|
| `llama3.2:3b` | 3B | Fast, lightweight generation |
| `llama3.2:latest` | 8B | Balanced quality/speed |
| `mistral:latest` | 7B | High-quality responses |
| `codellama:latest` | 7B | Code generation |

## Configuration Options

### ServiceConfig Fields

```rust
pub struct ServiceConfig {
    /// Base URL for Ollama API (default: "http://localhost:11434")
    pub ollama_base_url: Option<String>,

    /// Model for text embeddings (e.g., "nomic-embed-text:latest")
    pub embedding_model: Option<String>,

    /// Model for text generation (e.g., "llama3.2:latest")
    pub language_model: Option<String>,

    /// Embedding vector dimension (must match model output)
    pub vector_dimension: Option<usize>,

    // ... other fields
}
```

### OllamaConfig Fields

```rust
pub struct OllamaConfig {
    /// Enable Ollama integration
    pub enabled: bool,

    /// Ollama host URL
    pub host: String,

    /// Ollama port (default: 11434)
    pub port: u16,

    /// Embedding model name
    pub embedding_model: String,

    /// Chat/generation model name
    pub chat_model: String,

    /// Timeout in seconds (default: 30)
    pub timeout_seconds: u64,

    /// Maximum retry attempts (default: 3)
    pub max_retries: u32,

    /// Fallback to hash-based IDs on error
    pub fallback_to_hash: bool,

    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,

    /// Temperature for generation (0.0 - 1.0)
    pub temperature: Option<f32>,
}
```

## Advanced Usage

### Batch Processing

```rust
use graphrag_core::core::traits::AsyncEmbedder;

let embedder = OllamaEmbedderAdapter::new("nomic-embed-text:latest", 768);

// Batch embed multiple texts
let texts = vec!["First text", "Second text", "Third text"];
let embeddings = embedder.embed_batch(&texts).await?;

println!("Generated {} embeddings", embeddings.len());
```

### Custom Generation Parameters

```rust
use graphrag_core::core::traits::{AsyncLanguageModel, GenerationParams};

let llm = OllamaLanguageModelAdapter::new(ollama_config);

let params = GenerationParams {
    max_tokens: Some(500),
    temperature: Some(0.8),
    top_p: Some(0.9),
    stop_sequences: Some(vec!["END".to_string()]),
};

let response = llm.complete_with_params("Write a short story", params).await?;
```

Note: Current implementation uses default parameters. Custom parameters support coming soon.

### Service Registry Pattern

```rust
use graphrag_core::core::{RegistryBuilder, ServiceRegistry};

// Build registry manually
let mut registry = RegistryBuilder::new()
    .with_service(embedder)
    .with_service(llm)
    .with_service(vector_store)
    .build();

// Retrieve services by type
let embedder = registry.get::<OllamaEmbedderAdapter>()?;
let llm = registry.get::<OllamaLanguageModelAdapter>()?;
```

## Feature Flags

Add these features to your `Cargo.toml`:

```toml
[dependencies]
graphrag-core = { version = "*", features = ["async", "ollama", "memory-storage"] }
```

Available features:
- `async`: Required for async traits (AsyncEmbedder, AsyncLanguageModel)
- `ollama`: Enables Ollama integration and adapters
- `memory-storage`: In-memory vector store
- `ureq`: HTTP client for Ollama API calls

## Troubleshooting

### "Connection refused" Error

**Problem**: Cannot connect to Ollama server

**Solution**:
```bash
# Start Ollama server
ollama serve

# Verify it's running
curl http://localhost:11434/api/version
```

### "Model not found" Error

**Problem**: Requested model not available

**Solution**:
```bash
# Pull the model first
ollama pull nomic-embed-text:latest
ollama pull llama3.2:latest

# List available models
ollama list
```

### Dimension Mismatch Error

**Problem**: Embedding dimension doesn't match model output

**Solution**: Use correct dimensions for each model:
- `nomic-embed-text`: 768
- `mxbai-embed-large`: 1024
- `all-minilm`: 384

```rust
let embedder = OllamaEmbedderAdapter::new("nomic-embed-text:latest", 768);
//                                                                   ^^^
//                                                                   Must match model!
```

## API Reference

See the [Ollama API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md) for detailed API specifications.

## Performance Tips

1. **Use Batch Processing**: Batch multiple embedding requests for better throughput
2. **Model Selection**: Choose smaller models (3B) for faster responses, larger (8B+) for quality
3. **Local Deployment**: Keep Ollama on the same machine for lowest latency
4. **GPU Acceleration**: Ollama automatically uses GPU when available
5. **Connection Pooling**: ServiceRegistry reuses connections efficiently

## Examples

See `graphrag-core/examples/` for complete working examples:
- `ollama_embeddings.rs` - Text embedding generation
- `ollama_generation.rs` - Text completion and chat
- `ollama_rag.rs` - Complete RAG pipeline with Ollama

## Sources

- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Ollama Embeddings Guide](https://docs.ollama.com/capabilities/embeddings)
- [Embedding Models Blog](https://ollama.com/blog/embedding-models)

---

**Contributing**: Found a bug or want to improve Ollama integration? Open an issue or PR at the GraphRAG-RS repository.
