# Report: Project Changes (Present vs. Pre-Dec 30, 2025)

This report compares the current state of the project (including uncommitted changs) with the state recorded in commit `99df398` (December 5, 2025).

## 1. Organizational Changes

The project structure has evolved significantly to support better testing and incremental updates.

### New Directories & Modules
- **`tests/e2e/`**: A new End-to-End testing and benchmarking framework has been added.
  - Contains configuration files (`configs/`), reports (`reports/`), and results (`results/`).
  - Includes scripts like `run_benchmarks.sh`.
  - This indicates a major focus on performance validation and system-level testing.
- **`graphrag-core/src/incremental/`**: A new module dedicated to incremental graph updates.
  - Files added: `async_batch.rs`, `delta_computation.rs`, `lazy_propagation.rs`.
  - This suggests a shift from batch-only processing to supporting continuous data ingestion and graph updates.
- **`graphrag_py/`**: Complete Python bindings using `uv` and `maturin`, exposing the core Rust functionality to Python.

### Modified Components
- **`graphrag-wasm`**: Significant updates to `Cargo.toml` and potentially source code to improve WASM compatibility (e.g., `getrandom` 0.3 support, `leiden` community detection).
- **`graphrag-server`**: active development on the REST API server, including integration with `actix-web` and `apistos` for OpenAPI.

## 2. Technological Additions & Changes

### Core Technologies
- **Incremental Processing**: Implementation of delta computation and lazy propagation in `graphrag-core`.
- **Leiden Algorithm**: Explicitly added as a feature (`leiden`) in `graphrag-wasm`'s dependency on `graphrag-core`.

### Dependencies & Integrations (from Cargo.toml)
- **Tokenizers**: Added `tokenizers` crate with `unstable_wasm` feature in `graphrag-wasm`, likely for client-side text processing.
- **WASM Ecosystem**:
  - `getrandom` updated to `0.3` with `wasm_js` feature to fix browser compatibility.
  - Usage of `gloo-net` for HTTP fetching.
  - `wasm-logger` for logging in the browser.
- **Server Stack**:
  - **Actix-web**: The server is using `actix-web` (v4.9) as the web framework.
  - **Apistos**: Added for generating OpenAPI documentation.
  - **Ollama**: Integration via `ollama-rs` (v0.2) for LLM support.
  - **Vector DBs**: Support for `qdrant-client` and `lancedb` (via `arrow` dependencies).

## 3. Pipeline Evolution (2025-2026 Techniques)

Recent updates to the documentation (`README.md`, `tests/e2e/README.md`) reveal a major overhaul of the 7-stage pipeline, incorporating state-of-the-art research from late 2025 and 2026.

### The 7-Stage Pipeline Enhancements

#### Phase 1: Chunking
- **cAST (Context-Aware Splitting)**: Tree-sitter based chunking that preserves syntactic boundaries (functions, classes) for code.
- **Symbolic Anchoring (Phase 2 Prep)**: Grounding abstract concepts to concrete entities.

#### Phase 2: Entity Extraction
- **Dynamic Edge Weighting**: Context-aware ranking using semantic, temporal, and causal signals.
- **Causal Chain Analysis**: Discovery of multi-step causal chains with temporal consistency logic (CatRAG-style).
- **Gleaning Improvements**: Multi-round refinement with iterative LLM calls.

#### Phase 3: Relationship Extraction
- **Hierarchical Relationship Clustering**: Organizing relationships into multi-level hierarchies (Phase 3).
- **Graph Weight Optimization (DW-GRPO)**: Adaptive learning for relationship weights.

#### Phase 4: Graph Construction
- **Leiden Community Detection**: Replaced/augmented Louvain for +15% modularity.
- **Fast-GraphRAG**: PageRank-based retrieval offering 27x performance boost.

#### Phase 5: Embedding Generation
- **New Providers**: Support expanded to 8 backends including **Voyage AI**, **Jina AI**, **Together AI**, and **Mistral**.
- **WASM Support**: ONNX Runtime Web support for client-side embeddings.

#### Phase 6: Retrieval
- **LightRAG Integration**: Dual-level retrieval (high/low level) cited as achieveing 6000x token reduction (EMNLP 2025).
- **HippoRAG**: Personalized PageRank implementation (NeurIPS 2024).
- **Cross-Encoder Reranking**: +20% accuracy improvement.

#### Phase 7: Answer Generation
- **New Models Supported**: Configs validated for **Qwen 2.5/3** (72B/8B), **Llama 3.1**, and **Mistral-Nemo**.
- **RoGRAG**: Logic form reasoning decomposition.

## 4. Uncommitted / Recent Activity
- **Benchmarks**: The `tests/e2e/results` directory contains recent benchmark runs (e.g., `algo_hash_small__symposium.json`), indicating active performance testing is currently underway.
- **Workflows**: `tests/e2e/generate_report.sh` and `benchmark_report.md` suggest an automated pipeline for reporting results is being developed.
- **Python Bindings**: `TODO.md` confirms completion of Phase D (Python bindings) with `uv` and `maturin`, enabling the `graphrag-py` crate.

## Summary
The project has moved from a core library implementation towards a production-ready system with:
1.  **State-of-the-art RAG techniques** (LightRAG, HippoRAG, Causal Analysis).
2.  **Robust Testing** (E2E Benchmarks for Qwen3, Llama 3.1).
3.  **Cross-Platform Expansion** (WASM improvements, Python bindings, Actix server).
