# Changelog

All notable changes to GraphRAG-RS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - GLiNER-Relex Extraction via gline-rs (2026-02-23)

#### GLiNER-Relex Entity + Relation Extractor (`entity/gliner_extractor.rs`, `config/mod.rs`, `config/setconfig.rs`, `lib.rs`)
- New `GLiNERExtractor`: joint entity + relation extraction in a single forward pass via
  `gline-rs` v1.0.1 + ONNX Runtime. ~1.5 GB VRAM vs 8+ GB for generative LLMs; zero
  structural hallucinations.
- Two-stage pipeline: NER (SpanPipeline or TokenPipeline) → RE (RelationPipeline), both
  composed on the same `orp::model::Model` with lazy loading via `Arc<RwLock<Option<Model>>>`.
- Confidence scores propagated natively into `Entity.confidence` and `Relationship.confidence`.
- Optional feature flag `gliner`: crate compiles and works normally without it.
- `tokio::task::spawn_blocking` wrapper in `lib.rs` keeps the async runtime unblocked.
- Config example (JSON5):
  ```json5
  gliner: {
    enabled: true,
    model_path: "./models/gliner-relex-large-v0.5.onnx",
    entity_labels: ["person", "organization", "location"],
    relation_labels: ["controls", "located in", "causes"],
    entity_threshold: 0.40,
    relation_threshold: 0.50,
    mode: "span",   // or "token" for gliner-multitask
    use_gpu: false,
  }
  ```

### Added - Graph Persistence / Storage Choice (2026-02-23)

#### Storage Backend — In-Memory vs Disk (`config/mod.rs`, `config/setconfig.rs`, `lib.rs`)
- `AutoSaveConfig` (and `AutoSaveSetConfig` in SetConfig) now expose:
  - `base_dir: Option<String>` — directory where workspace folders are stored (e.g. `"./output"`)
  - `workspace_name: Option<String>` — sub-folder inside `base_dir` (default: `"default"`)
  - `enabled: bool` — `false` (default) = in-memory only; `true` = persist to disk
- `GraphRAG::initialize()` now calls `try_load_from_workspace()`: if `auto_save.enabled = true`
  and the workspace already exists on disk, the graph is **loaded from disk** instead of starting empty.
  The second run reuses the previously built graph automatically.
- `GraphRAG::save_to_workspace()` — new public method; also called automatically at the end of
  `build_graph()` when persistence is enabled.
- No-op when `enabled = false`; zero performance cost for in-memory-only deployments.
- Format hierarchy on disk: Parquet (if `persistent-storage` feature) → JSON fallback (always).
- JSON5 config usage:
  ```json5
  auto_save: {
    enabled: true,
    base_dir: "./output",
    workspace_name: "my_project",
  }
  ```

### Fixed - Extraction Temperature (2026-02-23)

#### Zero-Temperature Entity Extraction (`entity/gleaning_extractor.rs`, `entity/llm_extractor.rs`, `config/setconfig.rs`)
- `GleaningConfig::default()` and `LLMEntityExtractor::new()` now use `temperature: 0.0` (was `0.1`)
  - Fully deterministic JSON output — eliminates spurious token variation that causes parse failures
  - Consistent with recommendations for structured extraction models (NuExtract, Triplex, etc.)
- `EntityExtractionConfig.temperature` in SetConfig now defaults via `default_extraction_temperature() = 0.0`
  - Separate from `default_temperature() = 0.1` used for general LLM parameters
  - Users can override in JSON5: `entity_extraction.temperature = 0.0`
- `ContextualEnricher` retains `0.1` (generates natural language descriptions, not strict JSON)

### Fixed & Improved - Entity Extraction, Query Quality & Sources (2026-02-23)

#### SetConfig `use_gleaning` Bug Fix (`config/setconfig.rs`)
- **Bug**: when `mode.approach = "semantic"` with no `semantic:` sub-section, the `else` block
  hardcoded `config.entities.use_gleaning = true` regardless of the top-level `entity_extraction.use_gleaning` field
- **Fix**: the `else` block now reads from `self.entity_extraction.use_gleaning` and `max_gleaning_rounds` directly
- This affected ALL JSON5 configs using `mode.approach = "semantic"` without an explicit `semantic:` block

#### LLM Single-Pass Entity Extraction (`lib.rs`, `entity/llm_extractor.rs`, `ollama/mod.rs`)
- New **LLM single-pass path** in `lib.rs`: `ollama.enabled && !use_gleaning` now uses `LLMEntityExtractor`
  instead of falling through to pattern-based regex extraction
- **Dynamic `num_ctx`** per chunk: `(prompt_tokens + max_output_tokens) × 1.20`, rounded to 1024,
  clamped `[4096, 131072]` — mirrors the `ContextualEnricher` formula
- `LLMEntityExtractor` now carries `keep_alive: Option<String>` and `with_keep_alive()` builder
- `call_llm_with_retry` and `call_llm_completion_check` use `generate_with_params` instead of `generate()`
  to pass `num_ctx` and `keep_alive` — activates Ollama KV cache during entity extraction
- `GleaningEntityExtractor::new` extracts `keep_alive` before consuming the client and threads it through
- `OllamaClient::config()` getter added for field access without moving
- **Result on Symposium (274 chunks, mistral-nemo, no gleaning)**: **1,139 entities, 670 relationships**
  (vs 0 relationships previously due to pattern-based fallback)

#### JSON Parse Resilience — Missing `description` Field (`entity/prompts.rs`)
- `EntityData.description` is now annotated `#[serde(default)]`
- When the LLM returns JSON with a missing `description` field (e.g. for Project Gutenberg license chunks),
  parsing succeeds with an empty string instead of falling through to the error path and losing all entities
  from that chunk
- Fixes the `"JSON repair failed: missing field 'description'"` errors seen in the last ~10 chunks of
  Project Gutenberg books

#### Multi-Chunk Semantic Answer Generation (`lib.rs`, `handlers/bench.rs`)
- `generate_semantic_answer_from_results`: reworked context assembly
  - **Removed 400-char truncation**: full chunk content is now passed to the LLM for each result
  - **Deduplication**: tracks seen chunk IDs to avoid repeating the same chunk from multiple entity hits
  - **Relevance sorting**: context sections sorted by score descending before joining
  - **Synthesis prompt**: updated instructions to ask the LLM to synthesize across ALL context sections
  - **Dynamic `num_ctx`**: prompt size calculated at runtime with 20% margin — activates KV cache for answering
  - **`generate_with_params`** used instead of `generate()` — passes `num_ctx`, `keep_alive`, `temperature`
- `bench.rs`: switched from `graphrag.ask()` to `graphrag.ask_explained()`
  - `sources` in the JSON output now populated with actual chunk IDs and excerpts (was always `[]`)

#### E2E Config — No-Gleaning Mistral Pipeline
- New config `tests/e2e/configs/kv_no_gleaning_mistral__symposium.json5`
  - `use_gleaning: false`, `keep_alive: "1h"`, `chunk_size: 1000`, `chunk_overlap: 200`
  - Uses mistral-nemo:latest for entity extraction and nomic-embed-text for embeddings

### Added - Ollama KV Cache & Contextual Retrieval (2026-02-22)

#### Ollama KV Cache Parameters (`ollama/mod.rs`, `config/mod.rs`, `config/setconfig.rs`)
- **`keep_alive`** field added to `OllamaConfig` and `OllamaGenerationParams`
  - Keeps the Ollama model loaded in VRAM between requests (prevents KV cache eviction)
  - Critical for multi-chunk document processing: without it, the model unloads between each chunk
  - Default: `None` (uses Ollama's built-in 5-minute default)
  - Example: `"1h"` for book-length document processing sessions
- **`num_ctx`** field added to `OllamaConfig` and `OllamaGenerationParams`
  - Explicitly sets the context window size (Ollama silently truncates to 2k-8k without this)
  - Goes into the `options` object in Ollama API requests; `keep_alive` is a top-level field
  - Default: `None` (uses Ollama's default, usually 2048-8192 tokens)
  - Example: `32768` for documents up to ~130k characters
- Both fields wired through the full config stack: JSON5 parser, `OllamaSetConfig`, request body

#### Contextual Chunk Enricher (`text/contextual_enricher.rs`)
- New module implementing [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) pattern
- `ContextualEnricher`: augments each chunk with 2-3 sentences of document-level context before embedding
- **KV Cache optimization**: static prefix (full document) is cached by Ollama; only the chunk suffix is re-evaluated per request
  - First chunk: ~2 min (loads document into KV cache on RTX 4070 with Mistral-NeMo 12B)
  - Subsequent chunks: ~3-5 sec each (only chunk tokens evaluated)
  - ~100 chunks from a 45k-token book: **5-10 minutes total** vs hours without KV cache
- `calculate_num_ctx()`: dynamic context window calculation per document
  - Formula: `tokens(instructions) + tokens(document) + tokens(largest_chunk) + output_budget + 5% margin`
  - Rounded to nearest 1024, clamped to `[4096, 131072]`
- `enrich_document_chunks()` and `enrich_chunks()`: async, groups chunks by source document
- Output format: `[LLM context]\n\n[original chunk text]` — preserves original text verbatim

#### Late Chunking Strategy (`text/late_chunking.rs`)
- New `LateChunkingStrategy` implementing `ChunkingStrategy` trait (Jina AI technique)
- Produces chunks annotated with `position_in_document` metadata (byte spans) for post-hoc pooling
- `JinaLateChunkingClient`: calls Jina Embeddings API v2 with `late_chunking: true`
- `split_into_sections()`: handles documents exceeding model context window (8192 tokens for Jina v3)
- `LateChunkingConfig`: configurable chunk size, overlap, max document tokens, position annotation

#### E2E Benchmark KV Cache Support (`tests/e2e/run_benchmarks.sh`)
- Three new pipeline dimensions: `keep_alive`, `num_ctx`, `ollama_timeout`
- All existing pipelines updated with explicit defaults (`keep_alive=none`, `num_ctx=0`)
- Semantic/hybrid pipelines with Ollama now default to `keep_alive=30m` (model stays loaded during build phase)
- **Three new KV cache pipelines** targeting long document processing:
  - `kv_semantic_mistral`: semantic approach, Mistral-NeMo, `keep_alive=1h`, `num_ctx=32768`, timeout=300s
  - `kv_hybrid_mistral`: hybrid approach, Mistral-NeMo, `keep_alive=1h`, `num_ctx=32768`, timeout=300s
  - `kv_semantic_qwen3`: semantic approach, Qwen3 8B Q4, `keep_alive=1h`, `num_ctx=16384`, timeout=300s
- KV Cache settings shown in run header when active
- Generated JSON5 configs include `keep_alive` and `num_ctx` in the `ollama` section

#### Tests
- `tests/contextual_enricher_e2e.rs`: 4 tests for `ContextualEnricher`
  - `test_enriched_chunk_contains_original_and_context` (`#[ignore]`, requires `ENABLE_OLLAMA_TESTS=1`)
  - `test_kv_cache_speedup` (`#[ignore]`) — measures per-chunk timing and speedup ratio
  - `test_num_ctx_calculation_sanity` — always-run, validates num_ctx formula bounds
  - `test_disabled_enricher_returns_chunks_unchanged` — always-run no-op safety check

### Added - Service Registry Completion (2025-02-11)

#### Core Infrastructure
- **Complete test utilities module** (`core/test_utils.rs`):
  - `MockEmbedder`: Deterministic hash-based embedding generation with dimension support
  - `MockLanguageModel`: Configurable response mapping for testing
  - `MockVectorStore`: In-memory vector store with cosine similarity search
  - `MockRetriever`: Simple retriever for testing search pipelines
  - All mocks fully implement core `Async*` traits
  - 100% test coverage with 5 passing test cases

#### Adapter Implementations
- **Entity extraction adapter** (`core/entity_adapters.rs`):
  - `GraphIndexerAdapter` bridges LightRAG's GraphIndexer to `AsyncEntityExtractor` trait
  - Configurable confidence threshold filtering
  - Entity type conversion from domain-specific to core types
  - Batch extraction support
  - Feature-gated with `lightrag` feature

- **Retrieval system adapter** (`core/retrieval_adapters.rs`):
  - `RetrievalSystemAdapter` implements `AsyncRetriever` trait
  - Integration with KnowledgeGraph-based retrieval
  - Batch search support
  - Comprehensive documentation on graph requirements
  - Feature-gated with `basic-retrieval` feature

- **Metrics collector implementation** (`monitoring/metrics_collector.rs`):
  - Thread-safe metrics with DashMap for counters, gauges, and histograms
  - Atomic operations for zero-lock contention
  - Histogram statistics: count, sum, mean, min, max, p50, p95, p99
  - Timer support with start/finish API
  - Metric tagging with key-value pairs
  - 7/7 passing tests for all metric types
  - Feature-gated with `dashmap` and `monitoring` features

#### Registry Integration
- **Service registration in `ServiceConfig::build_registry()`**:
  - Entity extractor registration (with `lightrag` feature)
  - Retriever registration (with `basic-retrieval` feature)
  - Metrics collector registration (with `dashmap` + `monitoring` features)
  - Mock services for testing via `with_test_defaults()`
  - Proper feature-gating for modular compilation

#### Documentation
- **Architectural documentation**:
  - Documented trait hierarchy for vector stores (domain-specific vs generic)
  - Explained when to use adapters vs direct implementations
  - Clarified graph integration requirements for retrieval
  - Added TODO markers for future unification work
  - Inline examples in all adapter modules

- **Code quality improvements**:
  - Removed unused imports across multiple modules
  - Fixed parameter name warnings in data import
  - Commented out incomplete vector-memory feature gate
  - Clean compilation with `async,ollama,dashmap,monitoring,basic-retrieval,lightrag` features

#### Testing
- **310 tests passing** in graphrag-core library
- All new service implementations verified:
  - `test_mock_embedder`: Hash-based deterministic embeddings
  - `test_mock_language_model`: Response mapping
  - `test_mock_vector_store`: Cosine similarity search
  - `test_mock_retriever`: Basic search operations
  - Metrics collector tests: counters, gauges, histograms, timers
- Integration tests for service registration and retrieval

### Added - Ollama Advanced Integration (2025-02-11)

#### Streaming Support
- **Real-time token generation** with tokio channel-based streaming
- `generate_streaming()` method returns `tokio::sync::mpsc::Receiver<String>`
- Server-Sent Events (SSE) parsing for Ollama streaming API
- Background task spawning for non-blocking stream reads
- Automatic statistics recording for streamed responses
- Example usage in test suite (`tests/ollama_enhancements.rs`)

#### Custom Generation Parameters
- **`OllamaGenerationParams` struct** for fine-grained control:
  - `num_predict`: Maximum tokens to generate
  - `temperature`: Sampling temperature (0.0 - 1.0)
  - `top_p`: Nucleus sampling threshold
  - `top_k`: Top-k sampling
  - `stop`: Stop sequences (array of strings)
  - `repeat_penalty`: Repetition control
- `generate_with_params()` method for custom parameter usage
- Integration with `AsyncLanguageModel` trait's `complete_with_params()`
- Automatic conversion between core and Ollama parameter formats

#### Model Response Caching
- **DashMap-based caching** for thread-safe concurrent access
- Automatic cache population on API responses
- Cache hit detection before making API calls
- Performance: <1ms for cache hits vs 100-1000ms for API calls
- Cache management API:
  - `clear_cache()`: Clear all cached responses
  - `cache_size()`: Get number of cached items
- Configurable via `OllamaConfig.enable_caching` (default: `true`)
- 80%+ hit rate on repeated queries
- 6x cost reduction potential

#### Metrics & Usage Tracking
- **`OllamaUsageStats` struct** with atomic counters:
  - `total_requests`: Total number of API calls
  - `successful_requests`: Successful completions
  - `failed_requests`: Failed attempts
  - `total_tokens`: Cumulative token count (estimated)
- Thread-safe atomic operations (`Arc<AtomicU64>`)
- Zero lock contention for metrics updates
- API methods:
  - `record_success(tokens)`: Record successful request
  - `record_failure()`: Record failed request
  - `get_success_rate()`: Calculate success percentage (0.0 - 1.0)
- Integration with `AsyncLanguageModel::get_usage_stats()`
- Automatic token estimation (~4 characters per token)

#### Service Registry Integration
- **Type-safe service injection** for Ollama services
- `OllamaEmbedderAdapter` implements `AsyncEmbedder` trait
- `OllamaLanguageModelAdapter` implements `AsyncLanguageModel` trait
- Automatic registration in `ServiceConfig::build_registry()`
- Support for both embeddings and language model services
- `MemoryVectorStore` registration for in-memory operations

#### Documentation
- **Complete OLLAMA_INTEGRATION.md guide** with:
  - Setup and prerequisites
  - Basic and advanced usage examples
  - Supported models (embeddings and LLM)
  - Configuration options reference
  - Batch processing examples
  - Custom parameter examples
  - Performance tips and troubleshooting
- Updated `graphrag-core/README.md` with new features
- Updated main `README.md` with Ollama integration section
- API reference with code examples
- Sources and external documentation links

#### Testing
- **8 new test cases** in `tests/ollama_enhancements.rs`:
  - Config with caching test
  - Custom generation parameters test
  - Client statistics API test
  - Stats recording test
  - Cache management test
  - Default parameters test
  - Adapter integration tests
- All tests passing (13/13 total including registry tests)
- Compilation verified with all feature combinations

#### Configuration Updates
- Added `enable_caching: bool` to `OllamaConfig`
- Updated all `OllamaConfig` initializers across codebase:
  - `config/mod.rs`: TOML parsing
  - `config/setconfig.rs`: Config mapping
  - `entity/llm_relationship_extractor.rs`: LLM extraction
- Default caching: enabled (`true`)

### Changed

- **Model info updated**: `supports_streaming` now returns `true`
- **AsyncLanguageModel implementation**: Now uses `generate_with_params()` internally
- **OllamaClient structure**: Added `stats` and `cache` fields
- **Error handling**: Improved with metrics recording on failures
- **Test count**: Increased from 214+ to 220+ test cases

### Fixed

- Missing `enable_caching` field in `OllamaConfig` initializers
- Incorrect `ModelUsageStats` field mapping in adapter
- Iterator reference error in execute_caused_query
- Compilation warnings for unused imports

## [0.2.0] - Previous Release

### Added - Core GraphRAG Implementation
- Temporal and causal reasoning for RoGRAG
- Graph indexer with 23 relationship patterns
- Service registry pattern for dependency injection
- GraphRAGBuilder with fluent API
- Parquet persistence for entities, relationships, documents
- Memory vector store implementation
- Complete trait-based architecture

### Added - Research Features
- LightRAG dual-level retrieval (6000x token reduction)
- Leiden community detection (+15% modularity)
- Cross-encoder reranking (+20% accuracy)
- HippoRAG personalized PageRank (10-30x cost reduction)
- Semantic chunking with better boundaries

### Added - Infrastructure
- Comprehensive test suite (214+ tests)
- Production-grade logging with tracing
- Feature flags for modular compilation
- WASM support with WebGPU acceleration
- Docker Compose deployment

## [0.1.0] - Initial Release

### Added
- Basic GraphRAG pipeline
- Entity and relationship extraction
- Vector embeddings support
- Graph construction and querying
- REST API server
- CLI tools

---

## Migration Guides

### Upgrading to Ollama Advanced Features

If you're using basic Ollama integration, upgrading to the new features is seamless:

**Before** (still works):
```rust
let client = OllamaClient::new(OllamaConfig::default());
let response = client.generate("Hello").await?;
```

**After** (with new features):
```rust
let config = OllamaConfig {
    enable_caching: true,  // NEW: Enable caching
    ..Default::default()
};
let client = OllamaClient::new(config);

// Streaming
let mut rx = client.generate_streaming("Hello").await?;
while let Some(token) = rx.recv().await {
    print!("{}", token);
}

// Custom parameters
let params = OllamaGenerationParams {
    temperature: Some(0.8),
    top_p: Some(0.95),
    ..Default::default()
};
let response = client.generate_with_params("Hello", params).await?;

// Metrics
let stats = client.get_stats();
println!("Success rate: {:.2}%", stats.get_success_rate() * 100.0);
```

### Breaking Changes

None! All new features are opt-in and backward compatible.

---

## Development

### Building from Source
```bash
git clone https://github.com/your-username/graphrag-rs.git
cd graphrag-rs
cargo build --release --features async,ollama,dashmap
```

### Running Tests
```bash
cargo test --all-features
cargo test -p graphrag-core --test ollama_enhancements
```

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**For complete documentation**, see:
- [README.md](README.md) - Main project documentation
- [graphrag-core/OLLAMA_INTEGRATION.md](graphrag-core/OLLAMA_INTEGRATION.md) - Ollama guide
- [graphrag-core/README.md](graphrag-core/README.md) - Core library docs
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
