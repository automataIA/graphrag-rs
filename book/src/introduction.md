# GraphRAG-RS

GraphRAG-RS 0.3 is a Rust library and CLI that indexes UTF-8 text into a persistent
knowledge graph and retrieves relevant chunks and graph context for a question. The core
and facade run locally with pattern extraction and hash embeddings; optional providers can
be enabled when an application needs them.

## Where to start

| If you want to… | Start here |
|---|---|
| Install and run a first query | [Installation and Quick Start](getting-started/quickstart.md) |
| Understand the pipeline | [How It Works](concepts/how-it-works.md) |
| Configure chunking and retrieval | [Configuration Guide](configuration/configuration.md) |
| Browse the API | [docs.rs/graphrag-core](https://docs.rs/graphrag-core) |

The workspace contains five crates: `graphrag-core` for the library, `graphrag` as the
small facade, `graphrag-cli` for the turnkey binary, `graphrag-server` for the optional
HTTP service, and `graphrag-wasm` for the experimental browser target.

The supported native toolchain is Rust 1.88. WASM builds need the
`wasm32-unknown-unknown` target. Ollama and Qdrant are optional services for provider-backed
workflows; the base quickstart does not need either one.
