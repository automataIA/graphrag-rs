# CLI and TUI usage

Install the user-facing binary from the facade:

```bash
cargo install --path graphrag --features cli --locked
graphrag --help
```

The base `index` and `ask` commands accept UTF-8 text files and save a graph to
the selected workspace. Use `--config path/to/graphrag.toml` before either
subcommand to load and validate canonical settings. Use `graphrag validate`
beforehand when checking a file without indexing.

With no subcommand, or with `graphrag tui`, the interactive interface provides
`/config <file>`, `/load <file>`, `/mode`, `/stats`, `/entities`, and workspace
inspection commands. Provider-backed extraction and generation require their
own feature, service, and model configuration; the local fallback remains
available for the basic path.
