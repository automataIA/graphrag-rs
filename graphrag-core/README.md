# `graphrag-core`

`graphrag-core` contains the portable GraphRAG library. It accepts UTF-8 text,
extracts chunks and entities, builds an in-memory knowledge graph, and retrieves
relevant chunks and graph context. The base path is local and does not call an LLM;
synchronous retrieval is deterministic for equal inputs.

## Add the crate

The default features are `async`, `memory-storage`, and `basic-retrieval`:

```toml
[dependencies]
graphrag-core = "0.3"
```

For a synchronous, minimal application:

```toml
[dependencies]
graphrag-core = { version = "0.3", default-features = false, features = ["memory-storage", "basic-retrieval"] }
```

The checkout examples use the facade, whose path dependency is suitable for
local development and can be changed to `graphrag-core = "0.3"` for a release.

## Quick start

Synchronous code uses the `_sync` methods and needs no Tokio:

```rust,no_run
use graphrag_core::GraphRAG;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rag = GraphRAG::quick_start_sync(
        "The Prisma project is maintained by Maria Rossi.",
    )?;
    println!("{}", rag.ask_sync("Who maintains the Prisma project?")?);
    Ok(())
}
```

With the `async` feature, the corresponding methods are async:

```rust,no_run
use graphrag_core::GraphRAG;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rag = GraphRAG::quick_start(
        "The Prisma project is maintained by Maria Rossi.",
    ).await?;
    println!("{}", rag.ask("Who maintains the Prisma project?").await?);
    Ok(())
}
```

An async application must add Tokio (or another runtime) to its own manifest.
The public sync and async examples in the repository include complete manifests.

## Configuration

Runtime configuration uses one canonical schema:

```toml
[text]
chunk_size = 1000
chunk_overlap = 200

[retrieval]
top_k = 5

[auto_save]
enabled = true
base_dir = "/tmp/graphrag-workspaces"
workspace_name = "prisma"
```

`Config::load()` merges built-in defaults, `~/.graphrag/config.toml`,
`./graphrag.toml`, and `GRAPHRAG_*` environment variables in that order. Nested
environment names can use `GRAPHRAG_TEXT_CHUNK_SIZE` or the explicit
`GRAPHRAG_TEXT__CHUNK_SIZE`; documented legacy names such as
`GRAPHRAG_CHUNK_SIZE` are still accepted. Later layers override earlier layers,
and conflicting aliases in a single layer are errors. Parsing and semantic
validation are applied to the final merged configuration.

`Config::from_toml_file`, `Config::from_json_str`, and the JSON5 loader validate
with the same rules. Invalid types, non-finite numeric values, `top_k = 0`, and
`chunk_overlap >= chunk_size` are rejected. TOML and JSON5/JSON adapters preserve
the complete configuration on roundtrip.

## Workspaces

When `auto_save` has a base directory, call `save_to_workspace()` after building.
A later process created with the same configuration calls `initialize()` and
reopens the saved graph. The workspace stores the graph; the original source
text remains the application's responsibility. The CLI supplies this setting
through its `--workspace` option.

## Features

The core feature bundles are available for explicit selection:

- `memory-storage` and `basic-retrieval` provide the local quickstart.
- `async` adds async pipeline methods; disable it for a synchronous build.
- `persistent-storage`, `qdrant`, and `lancedb` add storage backends.
- `llm` adds the optional OpenAI-compatible LLM client (llama.cpp `llama-server`, Groq, Cerebras).
- `graph-retrieval`, `hybrid-retrieval`, `pagerank`, and `parallel-processing`
  add retrieval or processing strategies.
- `wasm-bundle` selects the portable browser-oriented subset.

Provider-backed extraction, embeddings, or generation need the matching feature,
service, and model configuration. Their quality and availability depend on that
external setup; the local fallback remains the behavior of the base quickstart.

## Verification

From the repository root:

```bash
cargo test -p graphrag-core --lib --locked
cargo check -p graphrag-core --no-default-features --locked
cargo fmt --all --check
```

The package requires Rust 1.88, matching the workspace and its current WASM/UI
dependencies.
