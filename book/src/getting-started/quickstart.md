# Installation & Quick Start

## CLI (turnkey, zero config)

```bash
cargo install --path graphrag-cli           # one-time install
graphrag index ./mydoc.txt                  # builds ./graphrag-data
graphrag ask "What is the main topic?"      # answers from the graph
```

Add `--ollama` to either command for LLM-quality entity extraction (requires `ollama serve` running
locally). With no flags, the CLI uses sensible defaults — hash-fallback embeddings, pattern-based
extraction, and a persistent workspace.

Run `graphrag` with no arguments for the interactive **TUI**, or `graphrag setup` for the config
wizard. See [CLI & TUI Usage](cli-tui.md).

## Library (Rust)

```rust
use graphrag::GraphRAG;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut g = GraphRAG::quick_start("Plato's Symposium full text here...").await?;
    println!("{}", g.ask("Who is Diotima?").await?);
    Ok(())
}
```

For more control, use `GraphRAG::builder()` or `Config::quick(workspace)` with `.with_ollama()` /
`.with_chunk_size()`. The typical flow is:

1. `Config::quick(workspace)` or `GraphRAG::builder()`.
2. `add_document(doc)` → `build_graph()` (chunking → embeddings → entities → relationships → persist).
3. `ask(q)` / `ask_explained(q)` / `ask_with_reasoning(q)`.

## System dependencies

| Platform | Install |
|----------|---------|
| Linux (Debian/Ubuntu) | `sudo apt install -y build-essential pkg-config` |
| macOS | `xcode-select --install` |
| Windows | Visual Studio Build Tools with C++ support |

For WASM builds: `rustup target add wasm32-unknown-unknown` and `cargo install trunk wasm-bindgen-cli`.

## Optional services

```bash
ollama pull nomic-embed-text     # local embeddings / LLM extraction
docker-compose up -d             # Qdrant vector database (server mode)
```

Next: understand the [pipeline](../concepts/how-it-works.md) or tune the
[configuration](../configuration/configuration.md).
