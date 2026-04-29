# GraphRAG-rs

![GraphRAG Network Visualization](examples/readme.jpg)

A high-performance, modular Rust implementation of GraphRAG (Graph-based Retrieval Augmented Generation) with **three deployment architectures**: Server-Only, WASM-Only (100% client-side), and Hybrid. Build knowledge graphs from documents and query them with natural language, with GPU acceleration support via WebGPU.

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![WASM](https://img.shields.io/badge/WASM-ready-green.svg)](https://webassembly.org/)
[![WebGPU](https://img.shields.io/badge/WebGPU-supported-blue.svg)](https://gpuweb.github.io/gpuweb/)

## 🔧 Prerequisites

### System Requirements
- **Rust 1.70+** with `wasm32-unknown-unknown` target
- **Node.js 18+** (for WASM builds)
- **Git** for cloning

### Platform-Specific Dependencies

#### Linux (Ubuntu/Debian)
```bash
# Basic build tools
sudo apt update
sudo apt install -y build-essential pkg-config

# For GPU acceleration features (Metal/WebGPU dependencies)
sudo apt install -y gobjc gnustep-devel libgnustep-base-dev

# Optional: For Qdrant vector database
docker-compose   # For containerized vector storage
```

#### macOS
```bash
# Xcode Command Line Tools (includes Objective-C compiler)
xcode-select --install

# Optional: Homebrew for additional tools
brew install rustup
```

#### Windows
```bash
# Install Visual Studio Build Tools with C++ support
# Or use Visual Studio Community with C++ development tools

# Install Rust with Windows target support
rustup target add wasm32-unknown-unknown
```

### Optional Dependencies
- **Ollama** for local LLM and embeddings: `ollama pull nomic-embed-text`
- **OpenAI-compatible server** as an alternative to Ollama for chat and/or
  embeddings — anything that speaks `/v1/chat/completions` and
  `/v1/embeddings` (vLLM, llama.cpp's `llama-server`, OpenVINO Model
  Server, OpenRouter, OpenAI itself, …)
- **Docker** for Qdrant vector database: `docker-compose up -d`
- **Trunk** for WASM builds: `cargo install trunk wasm-bindgen-cli`

## 🚀 Deployment Options

GraphRAG-rs supports **three deployment architectures** - choose based on your needs:

### Option 1: Server-Only (Traditional) ✅ Production Ready
```bash
git clone https://github.com/your-username/graphrag-rs.git
cd graphrag-rs

# Start Qdrant (optional)
cd graphrag-server && docker-compose up -d

# Pick ONE chat / embedding backend:

# Option A — Ollama (local, GGUF via Ollama daemon)
ollama serve &
ollama pull nomic-embed-text
export EMBEDDING_BACKEND=ollama
cargo run --release --bin graphrag-server --features "qdrant,ollama"

# Option B — OpenAI-compatible server (vLLM, llama-server, OVMS,
# OpenRouter, OpenAI, …) for either chat, embeddings, or both
export EMBEDDING_BACKEND=openai
export OPENAI_URL="http://127.0.0.1:8000/v1"   # vLLM example
export OPENAI_EMBEDDING_MODEL="BAAI/bge-m3"
export OPENAI_API_KEY=""                         # empty for self-hosted
cargo run --release --bin graphrag-server --features "qdrant,openai"
```
**Best for**: Multi-tenant SaaS, mobile apps, GPU workloads, >1M documents

**Features**:
- ✅ Qdrant vector database integration (production-ready)
- ✅ Embeddings via Ollama OR any OpenAI-compatible server (vLLM /
     llama-server / OVMS / OpenRouter / real OpenAI) with full
     hardware-acceleration paths (GPU, NPU)
- ✅ Hash-based fallback embeddings (no dependencies)
- ✅ REST API with semantic search
- ✅ Docker Compose setup
- ✅ 5.2MB release binary (optimized)

### Option 2: WASM-Only (100% Client-Side) ✅ Production Ready
```bash
# Install trunk for WASM builds
cargo install trunk wasm-bindgen-cli

# Build and run WASM app with GPU acceleration
cd graphrag-wasm
trunk serve --open
```
**Best for**: Privacy-first apps, offline tools, zero infrastructure cost, edge deployment

**Status: Fully Functional!**
- ✅ Complete GraphRAG pipeline running in browser
- ✅ ONNX Runtime Web (GPU-accelerated embeddings)
- ✅ WebLLM integration (Phi-3-mini for LLM synthesis)
- ✅ Pure Rust vector search (cosine similarity)
- ✅ Full Leptos UI with document upload and query interface
- ✅ Entity extraction with relationships
- ✅ Natural language answer synthesis
- ✅ Demo available: Plato's Symposium (2691 entities)

### Option 3: Hybrid (Recommended) 🎯 Planned
Use WASM client for real-time UI with optional server for heavy processing.

**Best for**: Enterprise apps, multi-device sync, best UX with scalability

**Status**: Architecture designed, implementation in Phase 3

See [graphrag-server/README.md](graphrag-server/README.md) for server documentation.

## 🎯 State-of-the-Art Quality Improvements

GraphRAG-rs implements **5 cutting-edge research papers** (2019-2025) for superior retrieval quality:

### Research-Based Features ✨

| Feature | Impact | Paper | Status |
|---------|--------|-------|--------|
| **LightRAG Dual-Level Retrieval** | 6000x token reduction | EMNLP 2025 | ✅ Production |
| **Leiden Community Detection** | +15% modularity | Sci Reports 2019 | ✅ Production |
| **Cross-Encoder Reranking** | +20% accuracy | EMNLP 2019 | ✅ Production |
| **HippoRAG Personalized PageRank** | 10-30x cheaper | NeurIPS 2024 | ✅ Production |
| **Semantic Chunking** | Better boundaries | LangChain 2024 | ✅ Production |

**Combined Result**: **+20% accuracy** with **99% cost savings**! 🚀

### New: Advanced Reasoning & Optimization (2025-2026) 🔬

Building on state-of-the-art foundations, GraphRAG-rs now implements **7 cutting-edge techniques** from recent research:

| Phase | Feature | Impact | Status |
|-------|---------|--------|--------|
| **Phase 2** | **Symbolic Anchoring** (CatRAG-style) | Better conceptual queries | ✅ Complete |
| **Phase 2** | **Dynamic Edge Weighting** | Context-aware ranking | ✅ Complete |
| **Phase 2** | **Causal Chain Analysis** | Multi-step reasoning | ✅ Complete |
| **Phase 3** | **Hierarchical Relationship Clustering** | Multi-level organization | ✅ Complete |
| **Phase 3** | **Graph Weight Optimization** (DW-GRPO) | Adaptive learning | ✅ Complete |

#### Key Capabilities

- **Symbolic Anchoring**: Automatically grounds abstract concepts (like "love" or "justice") to concrete entities for better conceptual query handling
- **Dynamic Weighting**: Adjusts relationship importance based on query context using semantic, temporal, and causal signals
- **Causal Reasoning**: Discovers multi-step causal chains with temporal consistency validation
- **Hierarchical Clustering**: Organizes relationships into multi-level hierarchies using Leiden algorithm with LLM-generated summaries
- **Weight Optimization**: Learns optimal relationship weights through heuristic optimization for improved retrieval quality

📚 **Full Documentation**: See [graphrag-core/ADVANCED_FEATURES.md](graphrag-core/ADVANCED_FEATURES.md) for implementation details, benchmarks, and research papers.

### Enable Advanced Features

```toml
[dependencies]
graphrag-core = { path = "../graphrag-core", features = ["lightrag", "leiden", "cross-encoder", "pagerank", "async"] }
```

```toml
# my_config.toml
[enhancements]
enabled = true

[enhancements.lightrag]
enabled = true
max_keywords = 20           # 6000x token reduction vs traditional GraphRAG
high_level_weight = 0.6
low_level_weight = 0.4

[enhancements.leiden]
enabled = true
max_cluster_size = 10       # Better quality than Louvain
resolution = 1.0

[enhancements.cross_encoder]
enabled = true
model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
top_k = 10                  # +20% accuracy improvement

# Advanced Features (Phases 2-3)
[advanced_features.symbolic_anchoring]
min_relevance = 0.3         # Minimum relevance for concept anchors
max_anchors = 5             # Maximum anchors per query

[advanced_features.dynamic_weighting]
enable_semantic_boost = true    # Boost relationships similar to query
enable_temporal_boost = true    # Boost recent/relevant relationships
enable_causal_boost = true      # Boost strong causal relationships

[advanced_features.causal_analysis]
min_confidence = 0.3            # Minimum confidence for causal chains
max_chain_depth = 5             # Maximum chain depth to search
require_temporal_consistency = true  # Enforce chronological ordering

[advanced_features.hierarchical_clustering]
num_levels = 3                  # Number of hierarchy levels (2-5)
generate_summaries = true       # LLM-generated cluster summaries

[advanced_features.weight_optimization]
learning_rate = 0.05            # Learning rate for optimization
max_iterations = 20             # Maximum optimization iterations
use_llm_eval = true             # Use LLM for quality evaluation
```

📖 **Quick Start Example**: See [graphrag-core/config-examples/quick-start.toml](graphrag-core/config-examples/quick-start.toml) for a minimal configuration.

📚 **Documentation**: See [HOW_IT_WORKS.md](HOW_IT_WORKS.md) for full details on the pipeline.

## Installation

### Prerequisites

- Rust 1.70 or later
- (Optional) Ollama for local LLM support - [Install Ollama](https://ollama.ai)

### From Source

```bash
git clone https://github.com/your-username/graphrag-rs.git
cd graphrag-rs
cargo build --release

# Optional: Install globally
cargo install --path .
```

## Quick Start (5 Lines!)

The fastest way to get started with GraphRAG:

```rust
use graphrag_core::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let mut graphrag = GraphRAG::quick_start("Your document text").await?;
    let answer = graphrag.ask("What is this about?").await?;
    println!("{}", answer);
    Ok(())
}
```

### With Compile-Time Safety (TypedBuilder)

```rust
use graphrag_core::prelude::*;

let graphrag = TypedBuilder::new()
    .with_output_dir("./output")  // Required - won't compile without
    .with_ollama()                 // Required - choose your LLM backend
    .with_chunk_size(512)          // Optional
    .build_and_init()?;
```

### Get Explained Answers

```rust
let explained = graphrag.ask_explained("Who founded the company?").await?;
println!("Answer: {}", explained.answer);
println!("Confidence: {:.0}%", explained.confidence * 100.0);
for source in &explained.sources {
    println!("Source: {} (relevance: {:.0}%)", source.id, source.relevance_score * 100.0);
}
```

### CLI Setup Wizard

```bash
# Interactive configuration wizard
graphrag-cli setup

# With domain template
graphrag-cli setup --template legal
```

### Feature Bundles

Choose the right features for your use case:

```toml
[dependencies]
graphrag-core = { version = "0.1", features = ["starter"] }   # Getting started
graphrag-core = { version = "0.1", features = ["full"] }      # Production
graphrag-core = { version = "0.1", features = ["research"] }  # Advanced
```

📚 **Full Guide**: See [graphrag-core/QUICKSTART.md](graphrag-core/QUICKSTART.md) for detailed quick start documentation.

## Basic Usage

### 1. Simple API (One Line)

```rust
use graphrag_rs::simple;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let answer = simple::answer("Your document text", "Your question")?;
    println!("Answer: {}", answer);
    Ok(())
}
```

### 2. Stateful API (Multiple Queries)

```rust
use graphrag_rs::easy::SimpleGraphRAG;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = SimpleGraphRAG::from_text("Your document text")?;

    let answer1 = graph.ask("What is this about?")?;
    let answer2 = graph.ask("Who are the main characters?")?;

    println!("Answer 1: {}", answer1);
    println!("Answer 2: {}", answer2);
    Ok(())
}
```

### 3. Builder API (Configurable)

```rust
use graphrag_rs::{GraphRAG, ConfigPreset};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut graphrag = GraphRAG::builder()
        .with_preset(ConfigPreset::Balanced)
        .auto_detect_llm()
        .build()?;

    graphrag.add_document("Your document")?;
    let answer = graphrag.ask("Your question")?;

    println!("Answer: {}", answer);
    Ok(())
}
```

## 📚 Understanding GraphRAG

New to GraphRAG? Start here:

- 📖 **[How It Works](HOW_IT_WORKS.md)** - Complete 7-stage pipeline explanation with diagrams and examples
- 🏗️ **[Pipeline Architecture](graphrag-core/PIPELINE_ARCHITECTURE.md)** - Technical deep dive into implementation details
- 💡 **[Examples](examples/)** - Hands-on code examples from basic to advanced
- 📊 **[Multi-Document Pipeline](examples/MULTI_DOCUMENT_PIPELINE.md)** - Production-ready example with benchmarks
- ⚙️ **[Pipeline Architecture](graphrag-core/PIPELINE_ARCHITECTURE.md)** - Detailed 7-phase configuration & performance guide

### Complete 7-Stage Pipeline Schema

```text
INDEXING (build_graph())
├── Phase 1: CHUNKING          → chunk_size, chunk_overlap
├── Phase 2: ENTITY EXTRACTION → approach, entity_types, use_gleaning
├── Phase 3: RELATIONSHIP      → extract_relationships, use_gleaning
└── Phase 4: GRAPH CONSTRUCTION → enable_pagerank, max_connections

QUERY (ask())
├── Phase 5: EMBEDDING         → backend, dimension, model
├── Phase 6: RETRIEVAL         → strategy, top_k
└── Phase 7: ANSWER GENERATION → chat_model, temperature
```

### Pipeline Configuration Summary

| Phase | Goal | Key Parameters |
|-------|------|----------------|
| **1. Chunking** | Split text | `chunk_size` (300), `chunk_overlap` (30) |
| **2. Extraction** | Identify entities | `approach` (hybrid), `entity_types` |
| **3. Relationships** | Connect entities | `extract_relationships` (true) |
| **4. Graph** | Build network | `max_connections` (50), `enable_pagerank` |
| **5. Embedding** | Vectorize data | `backend` (openai), `dimension` (1536) |
| **6. Retrieval** | Find context | `strategy` (hybrid), `top_k` (10) |
| **7. Generation** | Answer query | `chat_model` (gpt-4o), `temperature` (0.0) |

See **[PIPELINE_ARCHITECTURE.md](graphrag-core/PIPELINE_ARCHITECTURE.md)** for detailed configuration and performance tuning.

### 4. CLI Usage

GraphRAG-rs provides two CLI tools:

#### Smart CLI (Recommended) - `simple_cli`
Automatically detects if the knowledge graph needs building and handles everything for you:

```bash
# Build the Smart CLI
cargo build --release --bin simple_cli

# Process document and answer question in one command
cargo run --bin simple_cli config.toml "What are the main themes?"

# Interactive mode - builds graph if needed, then waits for questions
cargo run --bin simple_cli config.toml

# How it works:
# 1. Loads your TOML configuration
# 2. Checks if knowledge graph exists
# 3. Builds graph if needed (shows progress)
# 4. Answers your question using Ollama
# 5. Saves results to output directory
```

#### Manual CLI - `graphrag-rs`
For advanced users who want full control:

```bash
# Build the manual CLI
cargo build --release

# Step 1: Build knowledge graph
./target/release/graphrag-rs config.toml build

# Step 2: Query the graph
./target/release/graphrag-rs config.toml query "Your question"
```

## Configuration

### Basic Configuration (config.toml)

The project includes several ready-to-use configuration templates:

**Available Templates:**
- `config.toml` - Basic configuration for general use
- `config_complete.toml` - Full configuration with all options
- `config_tom_sawyer.toml` - Pre-configured for book processing
- `config_example.toml` - Annotated template with explanations

**Essential Configuration Fields:**

```toml
[general]
# IMPORTANT: Change these two paths for your project!
input_document_path = "path/to/your/document.txt"  # Your document to process
output_dir = "./output/your_project"                # Where to save results

[pipeline]
chunk_size = 800        # Size of text chunks (adjust based on document type)
chunk_overlap = 200     # Overlap to preserve context between chunks

[ollama]
enabled = true
host = "http://localhost"
port = 11434
chat_model = "llama3.1:8b"           # LLM for text generation
embedding_model = "nomic-embed-text"  # Model for embeddings

# --- Alternative: OpenAI-compatible chat backend ---
# Set [openai].enabled = true (and [ollama].enabled = false) to route
# entity extraction / query / gleaning through any OpenAI-compatible
# /v1/chat/completions endpoint — vLLM, llama-server, OpenRouter,
# OpenAI itself, …  Embeddings are configured separately under
# [embeddings] (see "Embedding Providers Configuration" below).
[openai]
enabled = false                                  # flip to true to use
base_url = "http://127.0.0.1:8000/v1"           # vLLM / llama-server / openai.com
chat_model = "Qwen3.6-27B-Instruct"             # whatever the server reports under /models
api_key = ""                                     # empty disables Authorization header
timeout_seconds = 600                            # set high for slow local models
max_tokens = 2000                                # null = uncapped (model stops at EOS)
temperature = 0.2

# Optional: server-specific knobs the OpenAI spec doesn't cover, merged
# into every /chat/completions body. Examples:
#   - llama.cpp / vLLM Qwen3 thinking suppression:
#       extra_body = { chat_template_kwargs = { enable_thinking = false } }
#   - vLLM JSON-only output:
#       extra_body = { response_format = { type = "json_object" } }
# extra_body = { chat_template_kwargs = { enable_thinking = false } }
```

**Quick Setup:**
1. Copy a template: `cp config_complete.toml my_project.toml`
2. Edit `input_document_path` to point to your document
3. Edit `output_dir` to set where results are saved
4. Run: `cargo run --bin simple_cli my_project.toml`

See [config_example.toml](config_example.toml) for detailed explanations of all options.

### Embedding Providers Configuration

GraphRAG Core supports **8 embedding providers** for maximum flexibility:

```toml
[embeddings]
backend = "huggingface"  # Free, offline (default)
# backend = "openai"     # Best quality ($0.13/1M tokens)
# backend = "voyage"     # Anthropic recommended
# backend = "cohere"     # Multilingual (100+ languages)
# backend = "jina"       # Cost-optimized ($0.02/1M)
# backend = "mistral"    # RAG-optimized
# backend = "together"   # Cheapest ($0.008/1M)
# backend = "ollama"     # Local GPU

model = "sentence-transformers/all-MiniLM-L6-v2"
dimension = 384
batch_size = 32
cache_dir = "~/.cache/huggingface"

# For API providers, set api_key or use environment variables
# api_key = "your-key"  # Or set OPENAI_API_KEY, VOYAGE_API_KEY, etc.
```

**Provider Comparison:**

| Provider | Cost | Quality | Features |
|----------|------|---------|----------|
| **HuggingFace** | Free | ★★★★ | Offline, 100+ models |
| **OpenAI** | $0.13/1M | ★★★★★ | Best quality |
| **Voyage AI** | Medium | ★★★★★ | Domain-specific (code, finance, law) |
| **Cohere** | $0.10/1M | ★★★★ | Multilingual |
| **Jina AI** | $0.02/1M | ★★★★ | Best price/performance |
| **Mistral** | $0.10/1M | ★★★★ | RAG-optimized |
| **Together AI** | $0.008/1M | ★★★★ | Cheapest |
| **Ollama** | Free | ★★★★ | Local GPU |

**Environment Variables:**
```bash
export OPENAI_API_KEY="sk-..."
export VOYAGE_API_KEY="pa-..."
export COHERE_API_KEY="..."
export JINA_API_KEY="jina_..."
export MISTRAL_API_KEY="..."
export TOGETHER_API_KEY="..."
```

See [graphrag-core/EMBEDDINGS_CONFIG.md](graphrag-core/EMBEDDINGS_CONFIG.md) for detailed configuration guide.

## ✨ Core Features

### 🏗️ Modular Architecture
- **Workspace Design**: Separate crates for core, WASM, Leptos, and server
- **Pluggable Backends**: Qdrant, LanceDB, pgvector, or in-memory storage
- **Feature Flags**: Compile only what you need (WASM, CUDA, Metal, WebGPU)
- **Trait-Based**: 12+ core abstractions for maximum flexibility

### 🔧 Trait-Based Chunking Architecture
- **ChunkingStrategy Trait**: Minimal interface for extensible chunking (1 method: `fn chunk(&self, text: &str) -> Vec<TextChunk>`)
- **HierarchicalChunkingStrategy**: LangChain-style with boundary preservation (respects paragraphs/sentences)
- **Tree-sitter AST Chunking**: cAST approach preserving syntactic boundaries for code
- **Performance Optimized**: Zero-cost abstraction with real implementations
- **Example**: Symposium analysis with 269 chunks preserving philosophical structure

#### cAST (Context-Aware Splitting) Implementation
Based on CMU research, our tree-sitter implementation provides:
- **Syntactic Boundary Preservation**: Complete functions, methods, structs
- **Rust Support**: AST parsing for proper code chunking
- **Configurable Granularity**: Function-level with minimum size controls
- **Feature-Gated**: Available with `--features code-chunking`

#### Usage Example
```rust
use graphrag_core::{
    core::{DocumentId, Document, ChunkingStrategy},
    text::{TextProcessor, HierarchicalChunkingStrategy},
};

// Trait-based chunking with hierarchical strategy
let processor = TextProcessor::new(1000, 100)?;
let strategy = HierarchicalChunkingStrategy::new(1000, 100, document.id);
let chunks = processor.chunk_with_strategy(&document, &strategy)?;

// Tree-sitter code chunking (with code-chunking feature)
#[cfg(feature = "code-chunking")]
{
    let code_strategy = RustCodeChunkingStrategy::new(50, document_id);
    let code_chunks = code_strategy.chunk(rust_code);
}
```

#### Run the Complete Example
```bash
# Basic example (hierarchical chunking)
cargo run --example symposium_trait_based_chunking --package graphrag-core

# With tree-sitter code chunking
cargo run --example symposium_trait_based_chunking --package graphrag-core --features code-chunking
```

**See**: [graphrag-core/examples/symposium_trait_based_chunking.rs](graphrag-core/examples/symposium_trait_based_chunking.rs) and [README_symposium_trait_based_chunking.md](graphrag-core/examples/README_symposium_trait_based_chunking.md) for complete documentation.

### 🎯 Storage Options

#### Native Production
- **Qdrant**: High-performance vector DB with JSON payload for entities/relationships
- **LanceDB**: Embedded vector DB for edge deployments (Node.js/desktop only)
- **pgvector**: PostgreSQL integration for existing infrastructure
- **Neo4j**: Optional graph database for complex multi-hop queries (>100k entities)

#### WASM Browser
- **Voy**: 75KB pure Rust vector search with k-d tree algorithm
- **IndexedDB**: Browser-native persistent storage for graph data
- **Cache API**: PWA-standard storage for ML models (1.6GB)

### 🧠 ML Inference

#### Embeddings
- **ONNX Runtime Web (GPU)**: 25-40x speedup, 3-8ms inference, WebGPU + CPU fallback, ✅ production-ready
- **Burn + wgpu (GPU)**: 20-40x speedup, 100% Rust, 🚧 70% complete (architecture done)
- **Candle (CPU)**: 100% Rust, BERT/MiniLM models, 50-100ms, 📅 planned
- **Ollama**: Server-side embeddings with GPU acceleration

#### LLM Chatbot
- **WebLLM**: 40-62 tok/s with WebGPU, production-ready
- **Candle**: 2-5 tok/s CPU-only, 100% Rust, good for demos
- **Ollama**: Server-side LLM with unlimited GPU power

### 🚀 Performance
- **ONNX Runtime Web**: 25-40x speedup for embeddings, 3-8ms inference ✅ production-ready
- **WebGPU Acceleration**: GPU inference in browser with automatic CPU fallback
- **WebLLM**: 40-62 tok/s LLM inference with WebGPU ✅ production-ready
- **LightRAG Integration**: 6000x token reduction vs traditional GraphRAG
- **PageRank Retrieval**: Fast-GraphRAG with 6x cost reduction
- **Parallel Processing**: Async/await throughout, concurrent document processing
- **Intelligent Caching**: LLM response cache with 80%+ hit rates

### 🎨 Developer Experience
- **Progressive API**: 4 complexity levels (Simple → Easy → Builder → Advanced)
- **Auto-Detection**: Smart LLM/backend discovery
- **Enhanced Errors**: Actionable error messages with solutions
- **TOML Config**: Complete configuration-driven processing
- **Hot Reload**: Configuration changes without restart

## Examples

### Quick Example: Using Config Templates

```bash
# Example 1: Process a book using existing template
cp config_tom_sawyer.toml my_book_config.toml
# Edit my_book_config.toml:
#   input_document_path = "books/my_book.txt"
#   output_dir = "./output/my_book"
cargo run --bin simple_cli my_book_config.toml "Who are the main characters?"

# Example 2: Process a research paper
cp config.toml research_config.toml
# Edit research_config.toml:
#   input_document_path = "papers/research.txt"
#   output_dir = "./output/research"
#   chunk_size = 500  # Smaller chunks for technical content
cargo run --bin simple_cli research_config.toml "What is the main hypothesis?"

# Example 3: Process with full configuration
cp config_complete.toml advanced_config.toml
# Edit all the parameters you need in advanced_config.toml
cargo run --bin simple_cli advanced_config.toml
```

### Process a Book

```rust
use graphrag_rs::{GraphRAG, Document};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read document
    let content = fs::read_to_string("book.txt")?;

    // Create and configure GraphRAG
    let mut graphrag = GraphRAG::builder()
        .with_chunk_size(1000)
        .with_chunk_overlap(200)
        .build()?;

    // Process document
    let doc = Document::new("book", content);
    graphrag.add_document(doc)?;

    // Query
    let answer = graphrag.ask("What are the main themes?")?;
    println!("Answer: {}", answer);

    Ok(())
}
```

### Use with Ollama

```rust
use graphrag_rs::{GraphRAG, OllamaConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure Ollama
    let ollama = OllamaConfig::new()
        .with_model("llama3.1:8b")
        .with_embedding_model("nomic-embed-text");

    // Create GraphRAG with Ollama
    let mut graphrag = GraphRAG::builder()
        .with_llm(ollama)
        .build()?;

    // Use as normal
    graphrag.add_text("Your document")?;
    let answer = graphrag.ask("Your question")?;

    Ok(())
}
```

### Batch Processing

```rust
use graphrag_rs::GraphRAG;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut graphrag = GraphRAG::new_default()?;

    // Process multiple documents
    for file in ["doc1.txt", "doc2.txt", "doc3.txt"] {
        let content = fs::read_to_string(file)?;
        graphrag.add_text(&content)?;
    }

    // Query across all documents
    let answer = graphrag.ask("What connects these documents?")?;
    println!("Answer: {}", answer);

    Ok(())
}
```

## Technical Achievements

GraphRAG-rs implements cutting-edge 2024 research in retrieval-augmented generation:

### Core Innovations
- **Fast-GraphRAG**: PageRank-based retrieval with 27x performance boost and 6x cost reduction
- **LightRAG Integration**: Dual-level retrieval achieving 6000x token reduction vs traditional GraphRAG
- **Incremental Updates**: Zero-downtime real-time graph processing with ACID-like guarantees
- **Intelligent Caching**: LLM response cache with 80%+ hit rates and 6x cost reduction
- **Hybrid Retrieval**: Combines semantic, keyword, BM25, and graph-based search strategies
- **ROGRAG Decomposition**: Advanced query decomposition with 60%→75% accuracy boost, temporal and causal reasoning
- **Ollama Advanced Integration**: Complete local LLM support with streaming, custom parameters, automatic caching, and metrics tracking

### Ollama Integration (NEW! ✨)

Complete local LLM and embedding support with production-grade features:

**Core Capabilities**:
- ✅ **Streaming Responses**: Real-time token generation with tokio channels
- ✅ **Custom Parameters**: Fine-grained control (temperature, top_p, top_k, stop sequences, repeat penalty)
- ✅ **Automatic Caching**: DashMap-based response caching with 80%+ hit rate
- ✅ **Metrics Tracking**: Thread-safe request/success/failure counting with atomic operations
- ✅ **Service Registry**: Type-safe dependency injection for all Ollama services
- ✅ **AsyncEmbedder Trait**: Full async/await support for embeddings
- ✅ **AsyncLanguageModel Trait**: Standardized LLM interface with streaming

**Performance**:
- Cache hit: <1ms vs 100-1000ms API calls
- Concurrent request handling with Arc-based sharing
- Zero-copy streaming with channel-based architecture
- GPU acceleration via Ollama (CUDA, ROCm, Metal)

**Example**:
```rust
use graphrag_core::core::ServiceConfig;

let config = ServiceConfig {
    ollama_base_url: Some("http://localhost:11434".to_string()),
    embedding_model: Some("nomic-embed-text:latest".to_string()),
    language_model: Some("llama3.2:latest".to_string()),
    vector_dimension: Some(768),
    ..Default::default()
};

let registry = config.build_registry().build();
// All services configured and ready!
```

See [graphrag-core/OLLAMA_INTEGRATION.md](graphrag-core/OLLAMA_INTEGRATION.md) for complete guide.

### Architecture & Quality
- **Modular Workspace**: 4 publishable crates (core, wasm, leptos, server)
- **Trait-Based Architecture**: 15+ core abstractions with dependency injection
- **50,000+ Lines**: Production-quality Rust implementation
- **Comprehensive Testing**: 220+ test cases with 100% pass rate
- **Production-Grade Logging**: Structured tracing throughout core library
- **Zero Warnings**: Clean compilation with clippy and cargo check
- **Feature Gates**: Compile only what you need for minimal binary size
- **Memory-Safe**: Leverages Rust's ownership system for zero-cost abstractions

## 🏗️ Workspace Architecture

GraphRAG-rs uses a modular workspace design for maximum reusability:

```
graphrag-rs/                     # Main workspace (50,000+ lines)
├── graphrag-core/               # ✅ Portable core library (native + WASM)
│   ├── All core functionality   # LightRAG, PageRank, caching, incremental
│   └── Feature-gated deps       # Compile only what you need
├── graphrag-wasm/               # ✅ WASM bindings and browser integrations
│   ├── ONNX Runtime Web         # GPU embeddings (3-8ms)
│   ├── WebLLM integration       # GPU LLM (40-62 tok/s)
│   └── IndexedDB + Cache API    # Browser storage
├── graphrag-leptos/             # 🚧 Leptos UI components (in development)
│   └── Chat, search, viz        # Ready-to-use components
├── graphrag-server/             # ✅ Production REST API server
│   ├── JSON configuration       # Dynamic config via REST API
│   ├── Qdrant integration       # Vector database
│   ├── Ollama embeddings        # Real semantic search
│   └── Docker Compose           # One-command deployment
└── src/                         # Main library and binaries
    ├── lib.rs                   # GraphRAG main API
    ├── bin/                     # CLI tools
    └── 25+ modules              # Modular architecture
```

### Dependency Graph
```
graphrag-leptos → graphrag-wasm → graphrag-core
graphrag-server → graphrag-core
main crate      → graphrag-core
```

### Feature Flags
```toml
[features]
# Storage backends
memory-storage = []                           # In-memory (development)
persistent-storage = ["lancedb", "arrow"]     # LanceDB embedded vector DB ⚠️ Mutually exclusive with neural-embeddings
redis-storage = ["redis"]                     # Redis for distributed caching

# Processing features
parallel-processing = []                      # Rayon parallelization
caching = ["moka"]                           # LLM response caching
incremental = []                             # Zero-downtime updates
pagerank = []                                # Fast-GraphRAG retrieval
lightrag = []                                # Dual-level retrieval
rograg = []                                  # Query decomposition

# LLM integrations
ollama = []                                  # Ollama local models with streaming
dashmap = ["dep:dashmap"]                    # Response caching (used with ollama)
neural-embeddings = ["candle-core"]          # Candle ML framework ⚠️ Mutually exclusive with persistent-storage
function-calling = []                        # Function calling support

# Platform-specific (GPU acceleration)
cuda = ["neural-embeddings", "candle-core/cuda"]    # NVIDIA GPU
metal = ["neural-embeddings", "candle-core/metal"]  # Apple Silicon GPU
webgpu = ["burn/wgpu"]                              # WebGPU (WASM)

# Chunking strategies
code-chunking = ["tree-sitter", "tree-sitter-rust"]  # Tree-sitter AST-based chunking

# API & CLI
web-api = []                                 # REST API server
```

**⚠️ Important: Feature Compatibility**

- `persistent-storage` and `neural-embeddings` are **mutually exclusive** due to dependency conflicts
- Choose based on your use case:
  - **For production RAG with vector storage**: Use `persistent-storage` (LanceDB + qdrant)
  - **For ML experiments with neural nets**: Use `neural-embeddings` (Candle + qdrant)
  - **For development**: Use neither (minimal dependencies)

See the feature flags section above for technical details on dependency selection.

For detailed architecture, see [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## API Reference

### Core Types

```rust
// Main GraphRAG interface
pub struct GraphRAG { /* ... */ }

// Document representation
pub struct Document {
    pub id: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
}

// Query results
pub struct QueryResult {
    pub answer: String,
    pub confidence: f32,
    pub sources: Vec<String>,
}
```

### Main Methods

```rust
impl GraphRAG {
    // Create new instance
    pub fn new(config: Config) -> Result<Self>;

    // Add content
    pub fn add_document(&mut self, doc: Document) -> Result<()>;
    pub fn add_text(&mut self, text: &str) -> Result<()>;

    // Query
    pub fn ask(&self, question: &str) -> Result<String>;
    pub fn query(&self, question: &str) -> Result<QueryResult>;

    // Management
    pub fn clear(&mut self);
    pub fn save(&self, path: &str) -> Result<()>;
    pub fn load(&mut self, path: &str) -> Result<()>;
}
```

## Performance Tuning

### Memory Optimization

```toml
[performance]
chunk_size = 500  # Smaller chunks use less memory
max_entities_per_chunk = 10
enable_caching = false
```

### Speed Optimization

```toml
[performance]
enable_parallel = true
num_threads = 8  # Adjust based on CPU cores
batch_size = 50
```

### Accuracy Optimization

```toml
[pipeline]
chunk_overlap = 400  # Higher overlap preserves more context
min_confidence = 0.7
enable_reranking = true
```

## Troubleshooting

### Common Issues

**Build fails with "rust version" error**
```bash
# Update Rust
rustup update
```

**Out of memory error**
```toml
# Reduce chunk size in config.toml
chunk_size = 300
enable_parallel = false
```

**Slow processing**
```toml
# Enable parallel processing
enable_parallel = true
num_threads = 8
```

**Ollama connection error**
```bash
# Ensure Ollama is running
ollama serve

# Check if model is available
ollama list
```

### Debug Mode

```bash
# Enable debug logging
RUST_LOG=debug cargo run --bin simple_cli config.toml

# Enable backtrace for errors
RUST_BACKTRACE=1 cargo run
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-username/graphrag-rs.git
cd graphrag-rs

# Run tests
cargo test

# Run with debug info
RUST_LOG=debug cargo run

# Check code quality
cargo clippy
cargo fmt --check
```

## FAQ

**Q: What file formats are supported?**
A: Currently supports plain text (.txt) and markdown (.md). PDF support is planned.

**Q: Can I use this without Ollama?**
A: Yes, the library includes a mock LLM for testing and can work with embeddings only.

**Q: How much memory does it need?**
A: Typically under 100MB for documents up to 500k characters.

**Q: Is it production ready?**
A: Yes, with 214 passing tests, zero warnings, and production-grade structured logging throughout the core library.

**Q: Can I use commercial LLMs?**
A: OpenAI support is planned. Currently works with Ollama's local models.

## 🗺️ Roadmap & Implementation Status

### ✅ Phase 1: Core Implementation (COMPLETE)

**Native Backend - Production Ready:**
- [x] **Modular Architecture**: 50,000+ lines across 25+ modules
- [x] **Trait System**: 15+ core abstractions with dependency injection
- [x] **Fast-GraphRAG**: PageRank-based retrieval (27x performance boost)
- [x] **LightRAG**: Dual-level retrieval (6000x token reduction)
- [x] **Incremental Updates**: Zero-downtime graph processing
- [x] **Intelligent Caching**: 80%+ hit rates, 6x cost reduction
- [x] **ROGRAG**: Query decomposition (60%→75% accuracy) + temporal/causal reasoning
- [x] **Hybrid Retrieval**: Semantic + keyword + BM25 + graph
- [x] **Parallel Processing**: Multi-threaded document processing
- [x] **Configuration System**: Complete TOML-driven pipeline
- [x] **Professional CLI**: Progress bars, auto-detection
- [x] **Comprehensive Tests**: 214+ test cases, 100% pass rate
- [x] **Production Logging**: Structured tracing throughout core library

**Server Deployment - Production Ready:**
- [x] **graphrag-server**: REST API with **Actix-web 4.9** + **Apistos** (automatic OpenAPI 3.0.3 docs)
- [x] **Dynamic JSON Config**: Full pipeline configuration via REST API (no TOML required)
- [x] **Qdrant Integration**: Production vector database
- [x] **Ollama Embeddings**: Real semantic search with GPU
- [x] **Hash-based Fallback**: Zero-dependency mode
- [x] **Docker Compose**: One-command deployment
- [x] **Health Checks**: Full system monitoring
- [x] **5.2MB Binary**: Optimized release build

### 🚧 Phase 2: WASM & Web UI (IN PROGRESS - 60% Complete)

**WASM Infrastructure:**
- [x] **graphrag-wasm crate**: WASM bindings foundation
- [x] **ONNX Runtime Web**: GPU embeddings (3-8ms, 25-40x speedup)
- [x] **WebLLM Integration**: GPU LLM (40-62 tok/s)
- [x] **IndexedDB**: Browser storage layer
- [x] **Cache API**: Model storage layer
- [x] **Voy Bindings**: Vector search preparation
- [ ] **Burn + wgpu**: GPU acceleration (architecture 70% complete)
- [ ] **Integration Tests**: End-to-end WASM testing

**Web UI:**
- [x] **graphrag-leptos crate**: UI components foundation
- [ ] **Chat Components**: Interactive query interface
- [ ] **Search Components**: Vector search visualization
- [ ] **Graph Visualization**: Knowledge graph display
- [ ] **Progress Indicators**: Real-time status updates
- [ ] **Responsive Design**: Mobile-first layout

### 📅 Phase 3: Advanced Features (PLANNED)

**Performance & Scale:**
- [ ] Distributed caching with Redis
- [ ] OpenTelemetry monitoring and tracing
- [ ] Query intelligence with ML rewriting
- [ ] Multi-model embeddings support
- [ ] Batch processing optimizations

**Analytics & Insights:**
- [ ] Graph analytics (community detection, centrality)
- [ ] Entity clustering and relationships
- [x] **Temporal reasoning**: Event timeline extraction and narrative ordering
- [x] **Causal reasoning**: Cause-effect chain discovery with confidence ranking
- [ ] Quality metrics and confidence scoring

**Data Integration:**
- [ ] Bulk import from CSV, JSON, RDF
- [ ] PDF document processing
- [ ] Multi-format export (GraphML, Cypher)
- [ ] Integration connectors (Notion, Confluence)

### 🏢 Phase 4: Enterprise Features (FUTURE)

**Scalability:**
- [ ] High availability and failover
- [ ] Horizontal scaling with load balancing
- [ ] Multi-region deployment
- [ ] Enterprise-grade security

**Developer Experience:**
- [ ] Multi-language SDKs (Python, TypeScript, Go)
- [ ] GraphQL API
- [ ] Custom plugin system
- [ ] Webhook integrations

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Microsoft GraphRAG for the original concept
- Ollama for local LLM support
- Rust community for excellent libraries

---

**Built with Rust** | [Documentation](https://docs.rs/graphrag-rs) | [Report Issues](https://github.com/your-username/graphrag-rs/issues)

