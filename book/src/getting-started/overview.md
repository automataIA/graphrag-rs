# Overview

GraphRAG-RS is a 5-crate Cargo workspace. You pick the entry point that fits your deployment.

| Crate | Role |
|-------|------|
| [`graphrag-core`](../crates/core.md) | Core library — all GraphRAG logic. Native + WASM (`rlib` + `cdylib`). |
| [`graphrag-cli`](../crates/cli.md) | Turnkey TUI + CLI binary. In-process use of the core (no HTTP). |
| [`graphrag-server`](../crates/server.md) | Actix-web REST API with OpenAPI + optional Qdrant. |
| [`graphrag-wasm`](../crates/wasm.md) | Browser bindings (Voy vector store, WebLLM, ONNX). |
| `graphrag` | Wrapper meta-crate that re-exports `graphrag-core` for the hello-world experience. |

## The config-driven pipeline

The same code runs three ways, selected at runtime from `Config` — not at compile time:

- **Pattern-only** — no LLM, regex extractor, < 10 ms per chunk. `Config::default()` works offline
  via hash-fallback embeddings.
- **LLM-enriched** — Ollama with KV-cache reuse (`keep_alive` + dynamic `num_ctx`) for higher-quality
  entity and relationship extraction.
- **Hybrid** — selective LLM stages over a fast base pipeline.

See [How It Works](../concepts/how-it-works.md) for the full 7-stage pipeline.

## Deployment options

- **Server** — multi-tenant, GPU workloads, large corpora. Qdrant + Ollama. See
  [graphrag-server](../crates/server.md).
- **WASM (client-side)** — privacy-first, offline, zero infrastructure. Full pipeline in the browser
  with ONNX embeddings and WebLLM synthesis. See [graphrag-wasm](../crates/wasm.md).
- **Embedded library** — call `graphrag-core` directly from your Rust app.

## Prerequisites

- **Rust 1.85+** (add the `wasm32-unknown-unknown` target for WASM builds).
- **Ollama** (optional) for LLM-quality extraction / real embeddings: `ollama pull nomic-embed-text`.
- **Docker** (optional) for the Qdrant vector database.

Continue to [Installation & Quick Start](quickstart.md).
