# Migrating from 0.2 to 0.3

GraphRAG-rs 0.3 keeps the workspace at five crates and raises the supported
native toolchain to Rust 1.88. The base path remains local and works without an
LLM service. Review the changes below before updating an application.

## Configuration

Runtime values now have one canonical location:

```toml
[text]
chunk_size = 1000
chunk_overlap = 200

[retrieval]
top_k = 5
```

The old top-level `chunk_size`, `chunk_overlap`, and `top_k_results` names are
accepted when reading 0.2 files and are adapted to the canonical fields. A
canonical and legacy spelling that supplies different values in the same layer
is an error. The loader merges built-in defaults, the home config, the project
config, and then `GRAPHRAG_*` environment variables; each later layer replaces
values from earlier layers. Both `GRAPHRAG_TEXT_CHUNK_SIZE` and the explicit
`GRAPHRAG_TEXT__CHUNK_SIZE` form are supported, along with documented legacy
aliases.

Parsing and semantic validation are applied by the core loaders, CLI, factory,
and server. A configuration can be validated without a document. Zero or
non-finite numeric values, an overlap greater than or equal to the chunk size,
and a non-positive `top_k` are rejected. Update code that constructed a 0.2
configuration by assigning duplicate runtime fields to use `Config` and its
nested sections.

## API and features

The async method names remain available when the `async` feature is enabled.
Every corresponding synchronous entry point has the `_sync` suffix, for
example `quick_start_sync`, `build_graph_sync`, `query_internal_sync`, and
`ask_sync`:

```toml
[dependencies]
graphrag = { version = "0.3", default-features = false, features = ["memory-storage", "basic-retrieval"] }
```

This is the small synchronous configuration. The default facade enables
`async`, `memory-storage`, and `basic-retrieval`; applications that need async
add their own Tokio runtime. The `cli` feature is separate and adds the
turnkey binary plus its runtime and terminal dependencies:

```bash
cargo install --path graphrag --features cli --locked
# after publication:
cargo install graphrag --features cli --locked
```

Provider and advanced algorithm features are opt-in. `llama-cpp` requires the
configured local HTTP services and model artifacts; it does not silently fall
back to hash embeddings or mock generation when a configured provider fails.

## Server and network boundaries

The server now binds to `127.0.0.1:8080` by default. Set
`GRAPHRAG_BIND_ADDR` explicitly to use another address. Cross-origin requests
are disabled by default; `GRAPHRAG_CORS_ORIGINS` accepts an explicit comma
separated allowlist of valid `http` or `https` origins. CORS is a browser
policy and does not provide authentication. The service is intended for a
trusted local process; multi-user authentication is not an operational 0.3
feature.

## Persistence

`build_graph()` and `build_graph_sync()` save the configured workspace after a
successful build, so both async and synchronous build paths can be reopened in
the next process with the same configuration. The CLI `--workspace` option
selects this location. Call `save_to_workspace()` explicitly after later
mutations that are outside those build paths. Indexing an existing workspace is
an update path and can retain documents already stored there. A configured
pipeline refuses document deletion until the graph and storage can be updated
atomically; storage-only deletion remains available where it is consistent.

Plain UTF-8 text and Markdown are supported by the base path. Convert PDF files
to text before indexing them. See the [configuration guide](../configuration/configuration.md),
[facade crate guide](../crates/facade.md), and [server guide](../crates/server.md)
for API-specific examples.
