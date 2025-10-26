# GraphRAG Core

The core library for GraphRAG-rs, providing portable functionality for both native and WASM deployments.

## Overview

`graphrag-core` is the foundational library that powers GraphRAG-rs. It provides:

- **Embedding Generation**: 8 provider backends (HuggingFace, OpenAI, Voyage AI, Cohere, Jina, Mistral, Together AI, Ollama)
- **Entity Extraction**: TRUE LLM-based gleaning extraction with multi-round refinement (Microsoft GraphRAG-style)
- **Graph Construction**: Incremental updates, PageRank, community detection
- **Retrieval Strategies**: Vector, BM25, PageRank, hybrid, adaptive
- **Configuration System**: Comprehensive TOML-based configuration
- **Cross-Platform**: Works on native (Linux, macOS, Windows) and WASM

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
graphrag-core = { path = "../graphrag-core", features = ["huggingface-hub", "ureq"] }
```

### Basic Usage

```rust
use graphrag_core::embeddings::huggingface::HuggingFaceEmbeddings;
use graphrag_core::embeddings::EmbeddingProvider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create embedding provider
    let mut embeddings = HuggingFaceEmbeddings::new(
        "sentence-transformers/all-MiniLM-L6-v2",
        None,
    );

    // Initialize (downloads model if needed)
    embeddings.initialize().await?;

    // Generate embedding
    let embedding = embeddings.embed("Your text here").await?;
    println!("Generated {} dimensions", embedding.len());

    Ok(())
}
```

## Features

### Embedding Providers

GraphRAG Core supports 8 embedding backends via feature flags:

```toml
[features]
# Free, offline embedding models
huggingface-hub = ["hf-hub", "dirs"]

# API-based embedding providers
ureq = ["dep:ureq"]  # Enables OpenAI, Voyage, Cohere, Jina, Mistral, Together

# Local inference (coming soon)
neural-embeddings = ["candle-core"]
```

**Supported Providers:**

| Provider | Cost | Quality | Feature Flag | Use Case |
|----------|------|---------|--------------|----------|
| **HuggingFace** | Free | ★★★★ | `huggingface-hub` | Offline, 100+ models |
| **OpenAI** | $0.13/1M | ★★★★★ | `ureq` | Best quality |
| **Voyage AI** | Medium | ★★★★★ | `ureq` | Anthropic recommended |
| **Cohere** | $0.10/1M | ★★★★ | `ureq` | Multilingual (100+ langs) |
| **Jina AI** | $0.02/1M | ★★★★ | `ureq` | Cost-optimized |
| **Mistral** | $0.10/1M | ★★★★ | `ureq` | RAG-optimized |
| **Together AI** | $0.008/1M | ★★★★ | `ureq` | Cheapest |
| **Ollama** | Free | ★★★★ | (via config) | Local GPU |

See [EMBEDDINGS_CONFIG.md](EMBEDDINGS_CONFIG.md) for detailed configuration.

### Configuration System

GraphRAG Core uses TOML for configuration:

```toml
# config.toml

[embeddings]
backend = "huggingface"
model = "sentence-transformers/all-MiniLM-L6-v2"
dimension = 384
batch_size = 32
cache_dir = "~/.cache/huggingface"

[graph]
max_connections = 10
similarity_threshold = 0.8

[retrieval]
top_k = 10
search_algorithm = "cosine"
```

Load configuration:

```rust
use graphrag_core::config::Config;

let config = Config::from_toml_file("config.toml")?;
```

## Examples

### Embedding Providers

```bash
# Demo all providers
cargo run --example embeddings_demo --features "huggingface-hub,ureq"

# Config-based usage
cargo run --example embeddings_from_config --features "huggingface-hub,ureq"

# With API keys
OPENAI_API_KEY=sk-... \
cargo run --example embeddings_demo --features ureq
```

### Entity Extraction (Real LLM-Based Gleaning)

GraphRAG Core implements TRUE LLM-based entity extraction with iterative refinement:

```rust
use graphrag_core::entity::gleaning_extractor::GleaningEntityExtractor;
use graphrag_core::entity::GleaningConfig;
use graphrag_core::ollama::OllamaClient;

// Create Ollama client
let ollama_client = OllamaClient::new(ollama_config);

// Configure gleaning (Microsoft GraphRAG-style)
let gleaning_config = GleaningConfig {
    max_gleaning_rounds: 4,  // Default: 4 rounds like Microsoft GraphRAG
    completion_threshold: 0.8,
    entity_confidence_threshold: 0.7,
    use_llm_completion_check: true,  // LLM judges completion
    entity_types: vec!["PERSON".to_string(), "ORGANIZATION".to_string(), "LOCATION".to_string()],
    temperature: 0.1,
    max_tokens: 1500,
};

// Create extractor
let extractor = GleaningEntityExtractor::new(ollama_client, gleaning_config);

// Extract with real LLM calls (takes 15-30 seconds per chunk per round)
let (entities, relationships) = extractor.extract_with_gleaning(chunk).await?;
```

**Performance Expectations**:
- **Small document** (5-10 pages): 5-15 minutes
- **Medium document** (50-100 pages): 30-60 minutes
- **Large document** (500-1000 pages): 2-4 hours

This is REAL LLM processing with actual API calls, not pattern matching!

### Graph Construction

```rust
use graphrag_core::graph::incremental::IncrementalGraph;

let mut graph = IncrementalGraph::new();
graph.add_document("document1", entities1)?;
graph.add_document("document2", entities2)?;  // Automatic deduplication!
```

## Module Structure

```
graphrag-core/
├── src/
│   ├── embeddings/          # Embedding generation
│   │   ├── mod.rs           # EmbeddingProvider trait
│   │   ├── huggingface.rs   # HuggingFace Hub integration
│   │   ├── api_providers.rs # 6 API providers
│   │   ├── config.rs        # TOML configuration
│   │   └── README.md        # Technical documentation
│   ├── entity/              # Entity extraction
│   │   ├── mod.rs           # Entity types and base traits
│   │   ├── prompts.rs       # Microsoft GraphRAG-style LLM prompts
│   │   ├── llm_extractor.rs # Real LLM entity extraction with Ollama
│   │   ├── gleaning_extractor.rs  # Multi-round gleaning orchestrator
│   │   └── semantic_merging.rs    # Entity deduplication
│   ├── graph/               # Knowledge graph
│   │   ├── mod.rs           # Graph construction
│   │   ├── incremental.rs   # Incremental updates
│   │   └── pagerank.rs      # Fast-GraphRAG retrieval
│   ├── retrieval/           # Search strategies
│   │   ├── mod.rs           # Vector similarity
│   │   ├── bm25.rs          # Keyword search
│   │   ├── hybrid.rs        # Multi-strategy fusion
│   │   └── pagerank_retrieval.rs  # Graph-based search
│   ├── config/              # Configuration system
│   │   ├── mod.rs           # Main Config struct
│   │   └── toml_config.rs   # TOML parsing
│   └── core/                # Core traits and types
│       ├── error.rs         # Error types
│       ├── traits.rs        # Pluggable abstractions
│       └── registry.rs      # Component registry
└── examples/
    ├── embeddings_demo.rs           # Test all providers
    ├── embeddings_from_config.rs    # Config-based usage
    └── embeddings.toml              # Example configuration
```

## Performance

### Embedding Speed

| Provider | Latency | Throughput | GPU Support |
|----------|---------|------------|-------------|
| HuggingFace (local) | 50-100ms | 10-20 chunks/s | ❌ |
| ONNX Runtime Web | 3-8ms | 125-333 chunks/s | ✅ WebGPU |
| Ollama | 100-200ms | 5-10 chunks/s | ✅ CUDA/Metal |
| API Providers | 50-200ms | Varies | ☁️ Cloud |

### Memory Usage

- HuggingFace models: ~100-500MB (cached on disk)
- Graph (10K entities): ~50MB
- Embeddings cache: Configurable (default: 10K vectors)

## Advanced Features

### LightRAG (Dual-Level Retrieval)

```toml
[retrieval]
strategy = "hybrid"
enable_lightrag = true  # 6000x token reduction!
```

### PageRank (Fast-GraphRAG)

```toml
[graph]
enable_pagerank = true  # 27x performance boost
```

### Intelligent Caching

```toml
[generation]
enable_caching = true  # 80%+ hit rate, 6x cost reduction
```

## API Providers Setup

### Environment Variables

```bash
# API Keys (recommended)
export OPENAI_API_KEY="sk-..."
export VOYAGE_API_KEY="pa-..."
export COHERE_API_KEY="..."
export JINA_API_KEY="jina_..."
export MISTRAL_API_KEY="..."
export TOGETHER_API_KEY="..."

# HuggingFace cache
export HF_HOME="~/.cache/huggingface"
```

### Configuration File

```toml
[embeddings]
backend = "openai"
model = "text-embedding-3-small"
# api_key = "sk-..."  # Or use OPENAI_API_KEY env var
dimension = 1536
batch_size = 100
```

## Testing

```bash
# Run all tests
cargo test --all-features

# Test embeddings
cargo test --features "huggingface-hub,ureq" embeddings

# Test with model downloads (slow)
ENABLE_DOWNLOAD_TESTS=1 cargo test --features huggingface-hub
```

## Documentation

- **[ENTITY_EXTRACTION.md](ENTITY_EXTRACTION.md)** - Complete guide to TRUE LLM-based gleaning extraction
- **[EMBEDDINGS_CONFIG.md](EMBEDDINGS_CONFIG.md)** - Complete embedding configuration guide
- **[src/embeddings/README.md](src/embeddings/README.md)** - Technical embedding documentation
- **[../REAL_LLM_GLEANING_IMPLEMENTATION.md](../REAL_LLM_GLEANING_IMPLEMENTATION.md)** - Implementation details and architecture
- **[../HOW_IT_WORKS.md](../HOW_IT_WORKS.md)** - Pipeline overview
- **[../README.md](../README.md)** - Main project documentation

## Feature Flags

```toml
[features]
# Embedding providers
huggingface-hub = ["hf-hub", "dirs"]  # Free, offline models
ureq = ["dep:ureq"]                   # API providers

# Graph backends
memory-storage = []                    # In-memory (default)
persistent-storage = ["lancedb"]       # LanceDB embedded

# Processing
parallel-processing = ["rayon"]        # Multi-threading
caching = ["moka"]                     # LLM response cache

# Advanced features
incremental = []                       # Zero-downtime updates
pagerank = []                          # Fast-GraphRAG
lightrag = []                          # Dual-level retrieval
rograg = []                            # Query decomposition

# LLM integrations
ollama = []                            # Ollama local models
function-calling = []                  # Function calling support
```

## Cross-Platform Support

GraphRAG Core is designed to work across platforms:

- ✅ **Linux** - Full support with all features
- ✅ **macOS** - Full support with Metal GPU acceleration
- ✅ **Windows** - Full support with CUDA GPU acceleration
- ✅ **WASM** - Core functionality (see graphrag-wasm crate)

## Contributing

See [../CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.

## License

MIT License - see [../LICENSE](../LICENSE) for details.

---

**Part of the GraphRAG-rs project** | [Main README](../README.md) | [Architecture](../ARCHITECTURE.md) | [How It Works](../HOW_IT_WORKS.md)
