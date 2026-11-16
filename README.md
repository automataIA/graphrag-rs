# GraphRAG-rs

GraphRAG-rs is a Rust library and command line tool for turning UTF-8 text into a
searchable knowledge graph. The 0.3 user path is local: it uses hash embeddings,
lexical retrieval, and rule based entity extraction. Synchronous retrieval is
deterministic for equal inputs. It does not
require an LLM or a network service.

## Maturity

The local core, the facade crate, memory storage, and the `index`/`ask` CLI path
are the maintained 0.3 path. LLM (llama.cpp / OpenAI-compatible) and Qdrant integrations are opt-in and need
their external services. WASM builds are checked, while browser integration and
some research backends remain experimental. The fallback path is useful for local
experiments, tests, and offline applications; evaluate retrieval quality on your
own documents before choosing a provider for a production workload.

## Quick start

Requirements: Rust 1.89 or newer and a UTF-8 text file (`.txt`, Markdown, or
another plain text format). PDF parsing is not part of the base path; extract PDF
text before indexing it.

```bash
git clone https://github.com/automataIA/graphrag-rs.git
cd graphrag-rs
cargo install --path graphrag --features cli --locked

mkdir -p /tmp/graphrag-demo
cat > /tmp/graphrag-demo/prisma.txt <<'TEXT'
The Prisma project is maintained by Maria Rossi. Luca Bianchi reviews releases.
TEXT

graphrag index /tmp/graphrag-demo/prisma.txt \
  --workspace /tmp/graphrag-demo/workspace
graphrag ask "Who maintains the Prisma project?" \
  --workspace /tmp/graphrag-demo/workspace
```

`index` writes the graph to the workspace. The second command is a new process:
it opens that saved graph and retrieves relevant chunks and graph context. Keep the
workspace path stable to reopen it later. For a clean independent run, choose a
new empty workspace; indexing an existing workspace is an update path and may
contain documents already loaded there.

The repository contains the same document used by the public examples at
[`examples/quickstart/document.txt`](examples/quickstart/document.txt). The
repeatable CLI test in `graphrag/tests/quickstart_cli.rs` runs these two commands
in separate processes.

## Library examples

The facade re-exports the core API. The synchronous API is available without an
async runtime:

```toml
[dependencies]
graphrag = { path = "graphrag", default-features = false, features = ["memory-storage", "basic-retrieval"] }
```

```rust,no_run
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mut rag = graphrag::GraphRAG::quick_start_sync(
        "The Prisma project is maintained by Maria Rossi.",
    )?;
    println!("{}", rag.ask_sync("Who maintains Prisma?")?);
    Ok(())
}
```

The async API uses the `async` feature and a runtime supplied by the application:

```toml
[dependencies]
graphrag = { path = "graphrag", features = ["async", "memory-storage", "basic-retrieval"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust,no_run
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rag = graphrag::GraphRAG::quick_start(
        "The Prisma project is maintained by Maria Rossi.",
    ).await?;
    println!("{}", rag.ask("Who maintains Prisma?").await?);
    Ok(())
}
```

Complete, checked manifests and sources are in
[`examples/quickstart-sync`](examples/quickstart-sync) and
[`examples/quickstart-async`](examples/quickstart-async). Run them from the
checkout with:

```bash
cargo run --manifest-path examples/quickstart-sync/Cargo.toml
cargo run --manifest-path examples/quickstart-async/Cargo.toml
```

For a published release, replace the path dependency with `graphrag = "0.3"`.

## Configuration and persistence

`Config::load()` merges sources in this order:

1. built-in defaults;
2. `~/.graphrag/config.toml`;
3. `./graphrag.toml`;
4. `GRAPHRAG_*` environment variables.

Later sources override earlier sources. Nested runtime settings use canonical
sections such as:

```toml
[text]
chunk_size = 1000
chunk_overlap = 200

[retrieval]
top_k = 5
```

Environment names follow the serialized path (`GRAPHRAG_TEXT_CHUNK_SIZE`); the
explicit separator form (`GRAPHRAG_TEXT__CHUNK_SIZE`) and documented legacy names
such as `GRAPHRAG_CHUNK_SIZE` are accepted. A conflicting pair in one layer is an
error. `Config::from_toml_file`, `from_json_str`, and the JSON5 loader use the same
validation rules. Invalid types, non-finite numbers, zero `top_k`, and an overlap
that is not smaller than the chunk size are rejected.

Core persistence is a saved graph workspace. Configure `auto_save` with a base
directory and workspace name when using the library, call `save_to_workspace()`
after building, and create the next process with the same configuration so
`initialize()` reopens the graph. The CLI's `--workspace` option supplies this
configuration for the local path.

## Features

The facade and core default to `async`, `memory-storage`, and `basic-retrieval`.
Disable defaults when a small synchronous dependency is preferred. Provider and
advanced algorithm features are explicit opt-ins, including `llm`, `qdrant`,
`pagerank`, and `persistent-storage`; the facade's `cli` feature adds the binary
and its runtime dependencies.

No-provider answers are local retrieval results formatted with their relevant
chunks. Enabling the `llm` section (llama.cpp `llama-server` or another OpenAI-compatible API) changes extraction and/or answer generation and requires
a running compatible service and configured models. Provider setup is outside the
base quickstart.

## Development

```bash
cargo fmt --all --check
cargo test -p graphrag-core --lib --locked
cargo test -p graphrag --features cli --test quickstart_cli --locked
cargo check --workspace --locked
```

See [`graphrag/README.md`](graphrag/README.md) for the facade install path and
[`graphrag-core/README.md`](graphrag-core/README.md) for the lower-level API and
feature details. For upgrade guidance and the recorded verification matrix, see
[`the 0.2 to 0.3 migration guide`](book/src/reference/migration-0.2-to-0.3.md)
and [`the 0.3 release report`](book/src/reference/release-0.3.md). The license
is MIT; see [`LICENSE`](LICENSE).
