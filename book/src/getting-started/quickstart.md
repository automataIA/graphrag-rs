# Installation and quick start

GraphRAG-rs 0.3 supports Rust 1.88. The base quickstart uses the local pipeline: it does
not require an LLM, an API key, or a separate database. Input files must be UTF-8 text;
convert PDFs to text before indexing.

## Install the CLI

From a checkout:

```bash
git clone https://github.com/automataIA/graphrag-rs.git
cd graphrag-rs
cargo install --path graphrag --features cli --locked
```

After a 0.3 package is published, install the registry package with:

```bash
cargo install graphrag --features cli --locked
```

## Index and ask

Create a small document and use a named workspace. Indexing and asking are separate
processes; the second command reopens the files written by the first.

```bash
mkdir -p /tmp/graphrag-demo
cat > /tmp/graphrag-demo/document.txt <<'EOF'
The Prisma project is maintained by Maria Rossi. Luca Bianchi reviews releases.
EOF

graphrag index /tmp/graphrag-demo/document.txt \
  --workspace /tmp/graphrag-demo/workspace
graphrag ask "Who maintains the Prisma project?" \
  --workspace /tmp/graphrag-demo/workspace
```

The default extractor uses patterns and local hash embeddings. Synchronous retrieval is
deterministic for equal inputs, but this local path is a retrieval demonstration rather
than a claim of LLM-generated reasoning. Ollama can be enabled for supported extraction
and answering workflows, but it is outside this zero-configuration example.

When a workspace already contains data, `index` adds the document while reopening the
workspace state as part of initialization. Use a new workspace when you need an isolated
quickstart run. The durable CLI behavior is covered by
[`graphrag/tests/quickstart_cli.rs`](https://github.com/automataIA/graphrag-rs/blob/main/graphrag/tests/quickstart_cli.rs).

## Use the library

The checkout contains complete standalone examples in
[`examples/quickstart-sync`](https://github.com/automataIA/graphrag-rs/tree/main/examples/quickstart-sync)
and
[`examples/quickstart-async`](https://github.com/automataIA/graphrag-rs/tree/main/examples/quickstart-async).
The synchronous API is always available and uses the `_sync` suffix. The asynchronous
API is available with the `async` feature and uses `.await`.

```rust
use graphrag::GraphRAG;

fn main() -> graphrag::Result<()> {
    let mut graph = GraphRAG::quick_start_sync("Maria Rossi maintains the Prisma project.")?;
    println!("{}", graph.ask_sync("Who maintains Prisma?")?);
    Ok(())
}
```

For a fully asynchronous program, enable the facade's `async` feature and use a Tokio
runtime:

```rust
use graphrag::GraphRAG;

#[tokio::main]
async fn main() -> graphrag::Result<()> {
    let mut graph = GraphRAG::quick_start("Maria Rossi maintains the Prisma project.").await?;
    println!("{}", graph.ask("Who maintains Prisma?").await?);
    Ok(())
}
```

See the example manifests for the complete path dependency and feature declarations.

## Configuration and persistence

The effective configuration is merged in this order: built-in defaults, the home
configuration (`~/.graphrag/config.toml`), the project configuration (`./graphrag.toml`),
then `GRAPHRAG_*` environment variables. Canonical runtime settings live in sections such
as `[text]` and `[retrieval]`; legacy top-level names remain readable with explicit
conflict errors. Use `graphrag validate` to check a configuration file before using it.

The graph and its metadata are saved in the selected workspace. A later library or CLI
process can reopen that workspace and query it without reindexing. Provider-backed
embeddings, generation, and vector stores are opt-in features; the base facade keeps the
local memory and lexical path small and portable.

Continue to [How It Works](../concepts/how-it-works.md) for the pipeline or to the
[configuration guide](../configuration/configuration.md) for settings.
