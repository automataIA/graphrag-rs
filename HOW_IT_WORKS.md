# How GraphRAG-RS works

GraphRAG-RS turns UTF-8 text into a persistent graph and uses that graph to select
evidence for a question. The default native path is local: it uses pattern extraction,
hash embeddings, and lexical retrieval. Ollama, remote embeddings, generation, and remote
storage are optional integrations and require their corresponding features and services.

## The local pipeline

The main flow has these stages:

1. **Read text.** The library accepts text supplied by the caller or a UTF-8 text file.
   PDF and other binary formats should be converted before they reach the library.
2. **Chunk.** `TextConfig` splits the input into overlapping chunks. `chunk_size` must be
   positive and `chunk_overlap` must be smaller than it.
3. **Embed.** The local fallback maps chunk text to fixed-size hash vectors. This keeps a
   zero-configuration workspace portable and does not call a model service.
4. **Extract.** Pattern-based extraction identifies entities and relationships that can
   be represented in the graph. An optional provider can perform richer extraction.
5. **Build.** Chunks, entities, and relationships are added to the graph. The configured
   storage backend writes the resulting workspace when saving is enabled.
6. **Retrieve.** A query is matched lexically against chunks and graph context. Results
   are ranked using the configured `retrieval.top_k` bound. The synchronous path
   produces deterministic ordering for equal inputs.
7. **Answer.** The local path returns a response assembled from relevant text and source
   metadata. Provider-backed generation can be selected separately when an application
   needs prose generation.

The graph adds useful links between text passages, but it does not guarantee an answer for
every question. Applications should inspect returned sources and handle an empty result.

## Minimal library flow

The facade exposes synchronous methods for programs that do not use Tokio:

```rust
use graphrag::GraphRAG;

fn main() -> graphrag::Result<()> {
    let mut graph = GraphRAG::quick_start_sync(
        "The Prisma project is maintained by Maria Rossi.",
    )?;
    println!("{}", graph.ask_sync("Who maintains Prisma?")?);
    Ok(())
}
```

The asynchronous API has the same operation with `.await` and the `async` feature:

```rust
use graphrag::GraphRAG;

#[tokio::main]
async fn main() -> graphrag::Result<()> {
    let mut graph = GraphRAG::quick_start(
        "The Prisma project is maintained by Maria Rossi.",
    ).await?;
    println!("{}", graph.ask("Who maintains Prisma?").await?);
    Ok(())
}
```

Complete manifests are available in `examples/quickstart-sync` and
`examples/quickstart-async`. They use a path dependency on the `graphrag` facade so the
examples can be checked from this checkout; replace that path with the published package
when consuming a release.

## Configuration

Runtime settings are represented by one `Config`. The canonical sections for the local
path are:

```toml
[text]
chunk_size = 1000
chunk_overlap = 200

[retrieval]
top_k = 5
```

Values are merged in this order: built-in defaults, `~/.graphrag/config.toml`,
`./graphrag.toml`, then `GRAPHRAG_*` environment variables. Legacy top-level names remain
readable for existing files, but conflicting canonical and legacy values are reported as
configuration errors. `graphrag validate path/to/config.toml` performs parsing and semantic
validation without requiring a document.

## Persistence and reopening

`save_to_workspace` writes the graph and configuration to the selected workspace.
`initialize` reopens that state in a later process. The CLI quickstart deliberately runs
`index` and `ask` as separate commands to exercise this boundary. Indexing into an existing
workspace can retain its earlier documents; choose a new workspace when an isolated run is
required.

## Provider boundaries

The local feature set is useful for development, tests, and private text that should stay
in process. It does not represent a hosted LLM or claim model-level generation quality.
When a provider is enabled, configure its endpoint and credentials through the supported
configuration fields and verify the service separately. Synchronous methods return an
explicit error for providers that require an asynchronous or unsupported runtime.

The server and browser crates expose additional deployment surfaces. They are optional
workspace members from the perspective of a local facade application and have their own
runtime and feature requirements.
