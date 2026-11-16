# Lazy and E2 graph construction

GraphRAG-rs contains optional lazy and E2-style graph construction modules. They
start from indexed text chunks and defer or simplify some graph work so an
application can choose a lower-cost indexing path. Their behavior and feature
requirements differ from the default local quickstart.

Use these modules only when their trade-offs fit the application. They are
advanced APIs and are not required for the standard `index`/`ask` workflow.
Enable the corresponding core features in `Cargo.toml`, inspect the public Rust
types, and validate the resulting retrieval on representative documents.
