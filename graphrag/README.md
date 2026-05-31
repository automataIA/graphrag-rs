# graphrag

Meta-crate for **GraphRAG** — Knowledge Graph Retrieval-Augmented Generation in Rust.

It bundles the workspace into a single hello-world dependency: it **re-exports
[`graphrag-core`](../graphrag-core/README.md)** for library users and ships the
`graphrag` binary (the full TUI + CLI, backed by `graphrag-cli`).

## Install the binary

```bash
cargo install graphrag
```

This installs `graphrag`, the zero-config TUI and CLI:

```bash
graphrag index ./docs/symposium.txt --workspace ./graphrag-data
graphrag ask "Who is Diotima?" --workspace ./graphrag-data
graphrag                       # interactive TUI
```

## Use as a library

```toml
[dependencies]
graphrag = "0.1"
```

Every public type from `graphrag-core` is re-exported, so `use graphrag::*` is all
you need:

```rust,no_run
use graphrag::GraphRAG;

let mut rag = GraphRAG::quick_start("Diotima taught Socrates about love.")?;
let answer = rag.ask("What did Diotima teach?")?;
println!("{answer}");
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Documentation

- [graphrag-core README](../graphrag-core/README.md) — full library API, config, feature flags.
- [Workspace README](../README.md) — project overview, all crates, deployment options.
- [HOW_IT_WORKS.md](../HOW_IT_WORKS.md) — the 7-stage pipeline explained.

## License

Same as the workspace (see the root `LICENSE`).
