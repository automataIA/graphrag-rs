# Overview

GraphRAG-RS is a five-crate Cargo workspace. Choose the entry point that matches the
application you are building.

| Crate | Role |
|---|---|
| [`graphrag-core`](../crates/core.md) | Core chunking, graph, retrieval, and persistence library. |
| [`graphrag`](../crates/facade.md) | Small facade with the default local feature set. |
| [`graphrag-cli`](../crates/cli.md) | Turnkey `index` and `ask` commands plus the interactive TUI. |
| [`graphrag-server`](../crates/server.md) | Optional Actix based HTTP service. |
| [`graphrag-wasm`](../crates/wasm.md) | Experimental browser bindings. |

The default native path is local: text is chunked, embeddings are computed with the
hash fallback, patterns extract graph items, and lexical retrieval selects relevant
chunks. Synchronous retrieval is deterministic for equal inputs. Provider-backed
extraction, embeddings, generation, and remote storage are opt-in features with
additional runtime requirements.

The supported toolchain is Rust 1.88. Add `wasm32-unknown-unknown` separately for WASM
builds. Continue to [Installation and Quick Start](quickstart.md).
