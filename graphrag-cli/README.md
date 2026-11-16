# `graphrag-cli`

`graphrag-cli` contains the command line implementation used by the `graphrag`
facade binary. The maintained 0.3 path is the local UTF-8 text workflow: hash
embeddings, pattern extraction, lexical retrieval, and a graph saved to a
workspace. It does not require an LLM or a network service.

## Install and run

From a repository checkout, install the user facing binary through the facade:

```bash
cargo install --path graphrag --features cli --locked
graphrag --help
```

After a package is published, use `cargo install graphrag --features cli --locked`.
For development without installing, run the same binary with:

```bash
cargo run --manifest-path graphrag/Cargo.toml --features cli -- --help
```

The base commands accept UTF-8 text files such as `.txt` and Markdown. Convert
PDF or other binary documents to text before indexing them.

## Index and ask

```bash
mkdir -p /tmp/graphrag-demo
cat > /tmp/graphrag-demo/document.txt <<'TEXT'
The Prisma project is maintained by Maria Rossi. Luca Bianchi reviews releases.
TEXT

graphrag index /tmp/graphrag-demo/document.txt \
  --workspace /tmp/graphrag-demo/workspace
graphrag ask "Who maintains the Prisma project?" \
  --workspace /tmp/graphrag-demo/workspace
```

`ask` is a separate process and reopens the graph written by `index`. Use a new
workspace for an isolated run; indexing an existing workspace can retain data
already stored there. A configuration file can be supplied before either command:

```bash
graphrag --config ./graphrag.toml index ./document.txt --workspace ./workspace
graphrag --config ./graphrag.toml ask "Who maintains Prisma?" --workspace ./workspace
```

The file is parsed and semantically validated before the command runs. The CLI
workspace option controls persistence, while the file controls runtime settings.
`Config::load()` also supports built-in defaults, the home file, the project file,
and `GRAPHRAG_*` environment variables in that precedence order.

## Configuration commands

```bash
graphrag validate ./graphrag.toml
graphrag explain-config ./graphrag.toml
graphrag doctor --config ./graphrag.toml
```

The canonical local settings use `[text]` and `[retrieval]` sections:

```toml
[text]
chunk_size = 1000
chunk_overlap = 200

[retrieval]
top_k = 5
```

TOML, JSON, and JSON5 files are accepted where the selected build enables the
format. Legacy top-level runtime names remain readable; conflicting canonical
and legacy values are rejected.

## TUI

Run the interactive interface with `graphrag tui` (or with no subcommand):

```bash
graphrag --config ./graphrag.toml tui
```

Within the TUI, `/config <file>` loads a configuration, `/load <file>` indexes a
UTF-8 text file, and `/mode ask|explain|reason` selects a query mode. `/stats`,
`/entities`, `/workspace list`, and `/export <file.md>` provide inspection and
session utilities. Keyboard help is available with `?`.

The TUI can use the same local fallback as `index` and `ask`. LLM backed
extraction or generation (`--llm`) is optional and requires a separately running
llama-server or another OpenAI-compatible endpoint;
the local quickstart does not claim provider generated prose quality.

## Workspace commands

```bash
graphrag workspace list
graphrag workspace create demo
graphrag workspace info demo
graphrag workspace delete demo
```

The workspace manager uses its own user data directory. The `--workspace` path
on `index` and `ask` directly selects the graph directory used by those commands.

## Development

```bash
cargo check -p graphrag-cli --locked
cargo test -p graphrag --features cli --test quickstart_cli --locked
```

The CLI depends on Tokio and external terminal libraries. Applications that only
need the library can use the `graphrag` facade without the `cli` feature and use
the always available `_sync` methods, or add the facade's `async` feature and a
runtime of their choice.
