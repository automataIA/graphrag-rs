# GraphRAG-RS

<p align="center">
  <img src="assets/readme.jpg" alt="GraphRAG Network Visualization" width="600"/>
</p>

**GraphRAG-RS** is a modular, portable [GraphRAG](https://microsoft.github.io/graphrag/) implementation
written in Rust. It builds a knowledge graph from your documents — chunking, embeddings, entity and
relationship extraction, community detection — and answers questions over that graph with citations.

The same core library runs **natively** and in the **browser via WebAssembly**, with a config-driven
pipeline that scales from a zero-dependency pattern matcher to a full LLM-enriched extraction stack.

## Why GraphRAG-RS

- **One library, three personalities.** Pattern-only (no LLM, < 10 ms/chunk), LLM + KV-cache
  enrichment (Ollama), or a hybrid — selected at runtime from `Config`, not at compile time.
- **Native + WASM.** `graphrag-core` is `crate-type = ["rlib", "cdylib"]`; the browser build uses a
  Voy vector store.
- **Turnkey.** `cargo run -p graphrag-cli -- index ./docs.txt` then `ask "..."` — zero config to start.
- **Modular crates.** Use the core library, the TUI/CLI, the REST server, or the WASM bindings.

## Where to go next

| If you want to… | Start here |
|-----------------|------------|
| Install and run your first query | [Installation & Quick Start](getting-started/quickstart.md) |
| Understand the pipeline | [How It Works](concepts/how-it-works.md) |
| Configure extraction & models | [Configuration Guide](configuration/configuration.md) |
| Browse the API | [docs.rs/graphrag-core](https://docs.rs/graphrag-core) |

> **Source:** [github.com/automataIA/graphrag-rs](https://github.com/automataIA/graphrag-rs)
