# GraphRAG Core

The core library for GraphRAG-rs, providing portable functionality for both native and WASM deployments.

## Overview

`graphrag-core` is the foundational library that powers GraphRAG-rs. It provides:

- **Embedding Generation**: 8 provider backends (HuggingFace, OpenAI, Voyage AI, Cohere, Jina, Mistral, Together AI, Ollama)
- **Entity Extraction**: TRUE LLM-based gleaning extraction with multi-round refinement (Microsoft GraphRAG-style)
- **Graph Construction**: Incremental updates, PageRank, community detection
- **Retrieval Strategies**: Vector, BM25, PageRank, hybrid, adaptive
- **Configuration System**: Hierarchical TOML-based configuration with environment variable overrides
- **Cross-Platform**: Works on native (Linux, macOS, Windows) and WASM

## Quick Start (5 Lines!)

```rust
use graphrag_core::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let mut graphrag = GraphRAG::quick_start("Your document text here").await?;
    let answer = graphrag.ask("What is the main topic?").await?;
    println!("{}", answer);
    Ok(())
}
```

**Or with detailed explanations:**

```rust
let explained = graphrag.ask_explained("What is the main topic?").await?;
println!("Answer: {}", explained.answer);
println!("Confidence: {:.0}%", explained.confidence * 100.0);
for step in &explained.reasoning_steps {
    println!("Step {}: {}", step.step_number, step.description);
}
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
# Choose a feature bundle:
graphrag-core = { version = "0.1", features = ["starter"] }  # Basic setup
# OR
graphrag-core = { version = "0.1", features = ["full"] }     # Production-ready
# OR
graphrag-core = { version = "0.1", features = ["research"] } # Advanced features
```

### Feature Bundles

| Bundle | Description | Includes |
|--------|-------------|----------|
| **`starter`** | Minimal setup to get started | async, ollama, memory-storage, basic-retrieval |
| **`full`** | Production-ready with common features | starter + pagerank, lightrag, caching, parallel-processing, leiden |
| **`wasm-bundle`** | Browser-safe features only | memory-storage, basic-retrieval, leiden |
| **`research`** | Advanced experimental features | full + rograg, cross-encoder, incremental, monitoring |

## Three Ways to Configure

### 1. TypedBuilder (Compile-Time Safety)

```rust
use graphrag_core::prelude::*;

// Build won't compile until required fields are set!
let graphrag = TypedBuilder::new()
    .with_output_dir("./output")    // Required
    .with_ollama()                   // Required: choose LLM backend
    .with_chunk_size(512)            // Optional
    .with_top_k(10)                  // Optional
    .build()?;
```

**Available LLM backends:**
- `.with_ollama()` - Local Ollama (recommended)
- `.with_ollama_custom("host", 8080, "model")` - Custom Ollama config
- `.with_hash_embeddings()` - Offline, no LLM needed
- `.with_candle_embeddings()` - Local neural embeddings

### 2. Hierarchical Config (with figment)

Enable with the `hierarchical-config` feature:

```rust
// Loads configuration from 5 sources (in priority order):
// 1. Code defaults (lowest priority)
// 2. ~/.graphrag/config.toml (user config)
// 3. ./graphrag.toml (project config)
// 4. Environment variables (GRAPHRAG_*)
// 5. Builder overrides (highest priority)

let config = Config::load()?;  // Automatically merges all sources
let graphrag = GraphRAG::new(config)?;
```

**Environment variable overrides:**
```bash
export GRAPHRAG_OLLAMA_HOST=my-server
export GRAPHRAG_OLLAMA_PORT=8080
export GRAPHRAG_CHUNK_SIZE=1000
```

### 3. TOML Configuration File

```toml
# graphrag.toml
output_dir = "./output"
approach = "hybrid"  # semantic, algorithmic, or hybrid
chunk_size = 1000
chunk_overlap = 200

[embeddings]
backend = "ollama"
dimension = 768
model = "nomic-embed-text:latest"

[ollama]
enabled = true
host = "localhost"
port = 11434
chat_model = "llama3.2:3b"

[entities]
min_confidence = 0.7
use_gleaning = true
max_gleaning_rounds = 3
entity_types = ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "EVENT"]
```

Load with:
```rust
let config = Config::from_toml_file("graphrag.toml")?;
let graphrag = GraphRAG::new(config)?;
```

## Sectoral Templates

Pre-configured templates for specific domains:

| Template | Best For | Entity Types |
|----------|----------|--------------|
| `general.toml` | Mixed documents | PERSON, ORGANIZATION, LOCATION, DATE, EVENT |
| `legal.toml` | Contracts, agreements | PARTY, JURISDICTION, CLAUSE_TYPE, OBLIGATION |
| `medical.toml` | Clinical notes | PATIENT, DIAGNOSIS, MEDICATION, SYMPTOM |
| `financial.toml` | Reports, filings | COMPANY, TICKER, MONETARY_VALUE, METRIC |
| `technical.toml` | API docs, code | FUNCTION, CLASS, MODULE, API_ENDPOINT |

**Using templates:**
```rust
let config = Config::from_toml_file("templates/legal.toml")?;
```

**Or via CLI:**
```bash
graphrag-cli setup --template legal
```

## Explained Answers

Get transparency into how answers are generated:

```rust
let explained = graphrag.ask_explained("Who founded the company?").await?;

// Access detailed information:
println!("Answer: {}", explained.answer);
println!("Confidence: {:.0}%", explained.confidence * 100.0);

// Reasoning trace
for step in &explained.reasoning_steps {
    println!("{}. {} (confidence: {:.0}%)",
        step.step_number,
        step.description,
        step.confidence * 100.0
    );
}

// Source references
for source in &explained.sources {
    println!("Source: {} ({:?})", source.id, source.source_type);
    println!("  Excerpt: {}", source.excerpt);
}

// Or get formatted output
println!("{}", explained.format_display());
```

**Output:**
```
**Answer:** John Smith founded Acme Corp in 2015.

**Confidence:** 85%

**Reasoning:**
1. Analyzed query: "Who founded the company?" (confidence: 95%)
2. Found 3 relevant entities (confidence: 85%)
3. Retrieved 5 relevant text chunks (confidence: 85%)
4. Synthesized answer from retrieved information (confidence: 85%)

**Sources:**
1. [TextChunk] chunk_123 (relevance: 92%)
2. [Entity] john_smith (relevance: 88%)
```

## Error Handling

Errors implement standard `std::error::Error` and carry descriptive messages:

```rust
match graphrag.ask("question").await {
    Ok(answer) => println!("{}", answer),
    Err(e) => {
        println!("Error: {}", e);
    }
}
```

## CLI Setup Wizard

Interactive configuration wizard:

```bash
graphrag-cli setup

# With template:
graphrag-cli setup --template legal

# Custom output:
graphrag-cli setup --output ./my-config.toml
```

**Wizard prompts:**
1. Select use case (General, Legal, Medical, Financial, Technical)
2. Choose LLM provider (Ollama or pattern-based)
3. Configure Ollama settings (if selected)
4. Set output directory

## Full Usage Example

```rust
use graphrag_core::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Option 1: Quick start (simplest)
    let mut graphrag = GraphRAG::quick_start("Your document text").await?;

    // Option 2: TypedBuilder (compile-time safe)
    let mut graphrag = TypedBuilder::new()
        .with_output_dir("./output")
        .with_ollama()
        .with_chunk_size(512)
        .build_and_init()?;

    // Add documents
    graphrag.add_document_from_text("Document content here")?;

    // Build knowledge graph
    graphrag.build_graph().await?;

    // Query
    let answer = graphrag.ask("What are the main topics?").await?;
    println!("{}", answer);

    // Or with explanations
    let explained = graphrag.ask_explained("What are the main topics?").await?;
    println!("{}", explained.format_display());

    Ok(())
}
```

## Embedding Providers

GraphRAG Core supports 8 embedding backends:

| Provider | Cost | Quality | Feature Flag | Use Case |
|----------|------|---------|--------------|----------|
| **HuggingFace** | Free | ★★★★ | `huggingface-hub` | Offline, 100+ models |
| **OpenAI** | $0.13/1M | ★★★★★ | `ureq` | Best quality |
| **Voyage AI** | Medium | ★★★★★ | `ureq` | Anthropic recommended |
| **Cohere** | $0.10/1M | ★★★★ | `ureq` | Multilingual (100+ langs) |
| **Jina AI** | $0.02/1M | ★★★★ | `ureq` | Cost-optimized |
| **Mistral** | $0.10/1M | ★★★★ | `ureq` | RAG-optimized |
| **Together AI** | $0.008/1M | ★★★★ | `ureq` | Cheapest |
| **Ollama** | Free | ★★★★ | `ollama` + `async` | Local GPU + LLM |

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

### RoGRAG (Logic Form Reasoning)
```rust
// Enable with feature flag: rograg
let answer = graphrag.ask_with_reasoning("Why did X cause Y?").await?;
```

### Intelligent Caching
```toml
[generation]
enable_caching = true  # 80%+ hit rate, 6x cost reduction
```

## Pipeline Architecture

GraphRAG uses a configurable pipeline with different methods for each phase:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         build_graph()                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐                                                    │
│  │    CHUNKING     │  TextProcessor splits document into chunks         │
│  │  (always runs)  │  Configurable: chunk_size, chunk_overlap           │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    ENTITY EXTRACTION                             │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │   │
│  │  │   Algorithmic   │  │    Semantic     │  │     Hybrid      │  │   │
│  │  │ (pattern-based) │  │  (LLM-based)    │  │ (both + fusion) │  │   │
│  │  │    ⚡ Fast      │  │  🎯 Accurate    │  │  ⚖️ Balanced    │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  RELATIONSHIP EXTRACTION                         │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │   │
│  │  │  Co-occurrence  │  │    LLM-based    │  │    Gleaning     │  │   │
│  │  │ entity proximity│  │ GraphRAG method │  │ multi-round LLM │  │   │
│  │  │    ⚡ Fast      │  │  🎯 Semantic    │  │  🔄 Iterative   │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘  │   │
│  │  Optional: config.graph.extract_relationships = true/false       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │    GRAPH        │  Entities + Relationships → KnowledgeGraph        │
│  │  CONSTRUCTION   │  Supports: PageRank, Community Detection          │
│  └─────────────────┘                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           ask() / query                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐                                                    │
│  │    EMBEDDING    │  Generated on-demand (lazy evaluation)             │
│  │   GENERATION    │  8 providers: Ollama, OpenAI, HuggingFace, etc.   │
│  └────────┬────────┘                                                    │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     RETRIEVAL STRATEGIES                         │   │
│  │  Vector │ BM25 │ PageRank │ Hybrid │ Adaptive │ LightRAG         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │     ANSWER      │  LLM synthesis (if Ollama enabled)                │
│  │   GENERATION    │  Or: concatenated search results                   │
│  └─────────────────┘                                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase Configuration Quick Reference

| Phase | Key Parameters | Config |
|-------|----------------|--------|
| **1. Chunking** | `chunk_size`, `chunk_overlap` | `chunk_size = 1000` |
| **2. Entity Extraction** | `approach`, `entity_types`, `use_gleaning` | `approach = "hybrid"` |
| **3. Relationship Extraction** | `extract_relationships`, `use_gleaning` | `[graph] extract_relationships = true` |
| **4. Graph Construction** | `enable_pagerank`, `max_connections` | `[graph] enable_pagerank = true` |
| **5. Embedding** | `backend`, `dimension`, `model` | `[embeddings] backend = "ollama"` |
| **6. Retrieval** | `strategy`, `top_k` | `[retrieval] strategy = "hybrid"` |
| **7. Answer Generation** | `chat_model`, `temperature` | `[ollama] enabled = true` |

### Method Selection by Phase

| Phase | Methods Available | Config Setting |
|-------|-------------------|----------------|
| **Entity Extraction** | Algorithmic / Semantic / Hybrid | `approach = "algorithmic\|semantic\|hybrid"` |
| **Relationship Extraction** | Co-occurrence / LLM-based / Gleaning | `entities.use_gleaning = true\|false` |
| **Embedding** | Ollama / Hash / OpenAI / HuggingFace / 8 providers | `embeddings.backend = "ollama"` |
| **Retrieval** | Vector / BM25 / PageRank / Hybrid / Adaptive / LightRAG | `retrieval.strategy = "hybrid"` |

### Key Notes

- **Embedding is NOT part of `build_graph()`** - generated lazily during queries
- **Relationship extraction is optional** - controlled by `config.graph.extract_relationships`
- **Gleaning extracts entities AND relationships together** in multi-round LLM calls
- **See [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) for full parameter reference**

## Module Structure

```
graphrag-core/
├── src/
│   ├── builder/         # TypedBuilder with type-state pattern
│   ├── config/          # Hierarchical configuration (figment)
│   ├── core/            # Core traits, errors with suggestions
│   ├── embeddings/      # 8 embedding providers
│   ├── entity/          # LLM-based gleaning extraction
│   ├── graph/           # Knowledge graph construction
│   ├── retrieval/       # ExplainedAnswer, search strategies
│   └── templates/       # Sectoral configuration templates
└── examples/
```

## Testing

```bash
# Quick test with starter features
cargo test --features starter

# Full test suite
cargo test --all-features

# Test specific modules
cargo test --features starter builder::
cargo test --features starter retrieval::
```

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute getting started guide
- **[PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md)** - Pipeline phases and methods
- **[templates/README.md](templates/README.md)** - Sectoral template guide
- **[EMBEDDINGS_CONFIG.md](EMBEDDINGS_CONFIG.md)** - Embedding configuration
- **[ENTITY_EXTRACTION.md](ENTITY_EXTRACTION.md)** - LLM-based extraction guide
- **[OLLAMA_INTEGRATION.md](OLLAMA_INTEGRATION.md)** - Ollama setup guide

## Cross-Platform Support

- ✅ **Linux** - Full support with all features
- ✅ **macOS** - Full support with Metal GPU acceleration
- ✅ **Windows** - Full support with CUDA GPU acceleration
- ✅ **WASM** - Core functionality (use `wasm-bundle` feature)

## License

MIT License - see [../LICENSE](../LICENSE) for details.

---

**Part of the GraphRAG-rs project** | [Main README](../README.md) | [Quick Start](QUICKSTART.md)
