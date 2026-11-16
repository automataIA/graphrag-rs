# `graphrag`

`graphrag` is the 0.3 facade crate. It re-exports `graphrag-core` and, with the
`cli` feature, provides the `graphrag` binary.

## Install the CLI

From a checkout:

```bash
cargo install --path graphrag --features cli --locked
graphrag --help
```

After the release is published:

```bash
cargo install graphrag --features cli --locked
```

The CLI's base path needs no LLM or network service. Index a UTF-8 text file into
a new workspace, then ask from a separate process:

```bash
graphrag index examples/quickstart/document.txt --workspace ./graphrag-data
graphrag ask "Who maintains the Prisma project?" --workspace ./graphrag-data
```

`ask` reopens the graph saved by `index`. Use a new workspace for an independent
run; an existing workspace can already contain documents from earlier indexing.
The base commands accept plain UTF-8 text (including Markdown); convert PDFs to
text before indexing. Pass `--config path/to/config.toml` before `index` or `ask`
when canonical settings should control chunking, retrieval, or a provider. The
workspace option still selects where the graph is persisted.

## Use the library

The facade re-exports every public core type, so applications can depend on one
crate:

```toml
[dependencies]
graphrag = "0.3"
```

For a checkout, use `graphrag = { path = "../graphrag" }` (or the relative path
from your application). A synchronous application can avoid the async runtime:

```toml
[dependencies]
graphrag = { path = "../graphrag", default-features = false, features = ["memory-storage", "basic-retrieval"] }
```

```rust,no_run
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rag = graphrag::GraphRAG::quick_start_sync(
        "The Prisma project is maintained by Maria Rossi.",
    )?;
    println!("{}", rag.ask_sync("Who maintains the Prisma project?")?);
    Ok(())
}
```

The default facade enables async support. Async applications use `.await` and
provide their own runtime:

```rust,no_run
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rag = graphrag::GraphRAG::quick_start(
        "The Prisma project is maintained by Maria Rossi.",
    ).await?;
    println!("{}", rag.ask("Who maintains the Prisma project?").await?);
    Ok(())
}
```

Add `tokio = { version = "1", features = ["macros", "rt-multi-thread"] }` to
an async application's manifest. Complete runnable manifests are in the root
[`examples/quickstart-sync`](../examples/quickstart-sync) and
[`examples/quickstart-async`](../examples/quickstart-async) directories.

## Features and configuration

The library defaults to `async`, `memory-storage`, and `basic-retrieval`. Use
`default-features = false` for a minimal synchronous dependency. Providers and
advanced algorithms are opt-in; the `cli` feature adds the CLI, Tokio, and
`color-eyre`.

Configuration is merged as defaults, `~/.graphrag/config.toml`,
`./graphrag.toml`, then `GRAPHRAG_*` environment variables. The canonical runtime
fields are `[text].chunk_size`, `[text].chunk_overlap`, and `[retrieval].top_k`.
Legacy names remain readable as adapters, while conflicting values in one layer
are rejected. See the core README for the full configuration example.

The no-provider path returns local lexical and graph retrieval results. llama.cpp or
other OpenAI-compatible providers require the `llm` feature, service configuration, and their own
operational setup.

For the complete 0.2 to 0.3 migration notes and release verification matrix,
see the [repository migration guide](../book/src/reference/migration-0.2-to-0.3.md)
and [release report](../book/src/reference/release-0.3.md).
