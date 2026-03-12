# graphrag-cli

A modern Terminal User Interface (TUI) for GraphRAG operations, built with [Ratatui](https://ratatui.rs/).

## Features

- **Multi-pane TUI** — Results viewer, Raw results, tabbed Info panel (Stats / Sources / History)
- **Markdown rendering** — LLM answers rendered with bold, italic, headers, bullet points, code blocks
- **Three query modes** — ASK (fast), EXPLAIN (confidence + sources), REASON (query decomposition)
- **Zero-LLM support** — Algorithmic pipeline with hash embeddings, no model required
- **Vim-style navigation** — j/k scrolling, Ctrl+1/2/3/4 focus switching
- **Slash command system** — `/config`, `/load`, `/mode`, `/reason`, `/export`, `/workspace`, and more
- **Query history** — Tracked per session, exportable to Markdown
- **Workspace persistence** — Save/load knowledge graphs to disk
- **Direct integration** — Uses `graphrag-core` as a library (no HTTP server needed)

---

## Installation

```bash
cd graphrag-rs

# Debug build (fast compile)
cargo build -p graphrag-cli

# Release build (optimized)
cargo build -p graphrag-cli --release
```

---

## Quick Start — Zero LLM (Symposium example)

Build a knowledge graph from Plato's Symposium with **no LLM required** — pure algorithmic extraction using regex patterns, TF-IDF, BM25, and PageRank.

### Option A — Interactive TUI

```bash
cd /home/dio/graphrag-rs

cargo run -p graphrag-cli -- tui
```

Then inside the TUI:

```
/config tests/e2e/configs/algo_hash_medium__symposium.json5
/load docs-example/Symposium.txt
Who is Socrates and what is his role in the Symposium?
```

Graph builds in ~3-5 seconds. No Ollama needed.

### Option B — TUI with config pre-loaded

```bash
cargo run -p graphrag-cli -- tui \
  --config tests/e2e/configs/algo_hash_medium__symposium.json5
```

Then just:
```
/load docs-example/Symposium.txt
What is Eros according to Aristophanes?
```

### Option C — Benchmark (non-interactive, JSON output)

```bash
cargo run -p graphrag-cli -- bench \
  --config tests/e2e/configs/algo_hash_medium__symposium.json5 \
  --book docs-example/Symposium.txt \
  --questions "Who is Socrates?|What is love according to Aristophanes?|What is the Ladder of Beauty?"
```

Outputs structured JSON with timings, entity counts, answers, confidence scores, and source references.

### Available configs

| Config | Graph building | Embeddings | LLM synthesis | Speed |
|--------|---------------|------------|---------------|-------|
| `algo_hash_small__symposium.json5` | NLP/regex | Hash (256d) | ❌ none | ~1-2s |
| `algo_hash_medium__symposium.json5` | NLP/regex | Hash (384d) | ❌ none | ~3-5s |
| `algo_nlp_mistral__symposium.json5` | **NLP/regex** | nomic-embed-text | ✅ mistral-nemo | ~5-15s* |
| `kv_no_gleaning_mistral__symposium.json5` | LLM single-pass | nomic-embed-text | ✅ mistral-nemo | ~30-60s |

\* build ~5s, sintesi ~5-10s per domanda (con KV cache dopo la prima)

**`algo_nlp_mistral__symposium.json5`** è il config raccomandato per chi vuole:
- grafo costruito velocemente con metodi NLP classici (nessun LLM a build time)
- ricerca semantica reale con `nomic-embed-text`
- risposte sintetizzate da Mistral a query time con KV cache abilitata

---

## Quick Start — With Ollama (full semantic pipeline)

Requires Ollama running with `nomic-embed-text` and an LLM (e.g. `mistral-nemo:latest`).

```bash
cargo run -p graphrag-cli -- tui \
  --config tests/e2e/configs/kv_no_gleaning_mistral__symposium.json5
```

Inside TUI:
```
/load docs-example/Symposium.txt
/mode explain
How does Diotima describe the ascent to absolute beauty?
```

The EXPLAIN mode shows confidence score and source references in the Sources tab (Ctrl+4 → Ctrl+N).

---

## CLI Commands

```
graphrag-cli [OPTIONS] [COMMAND]

Options:
  -c, --config <FILE>      Configuration file to pre-load
  -w, --workspace <NAME>   Workspace name
  -d, --debug              Enable debug logging
      --format <text|json> Output format (default: text)

Commands:
  tui        Start interactive TUI (default)
  setup      Interactive wizard to create a config file
  validate   Validate a configuration file
  bench      Run full E2E benchmark (Init → Load → Query)
  workspace  Manage workspaces (list, create, info, delete)
```

### bench example

```bash
cargo run -p graphrag-cli -- bench \
  -c my_config.json5 \
  -b my_document.txt \
  -q "Question 1?|Question 2?|Question 3?"
```

Output JSON includes: `init_ms`, `build_ms`, `total_query_ms`, `entities`, `relationships`, `chunks`, per-query `answer`, `confidence`, `sources`.

---

## TUI Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Query Input (Ctrl+1)  (type queries or /commands here)     │
├────────────────────────────────────┬────────────────────────┤
│  Results Viewer (Ctrl+2)           │  Info Panel (Ctrl+4)   │
│  Markdown-rendered LLM answer      │  ┌─Stats─┬─Sources─┬  │
│  with confidence header in EXPLAIN │  │       │History  │  │
│  mode: [EXPLAIN | 85% ████████░░]  │  └───────┴─────────┘  │
├────────────────────────────────────┤  Ctrl+N cycles tabs    │
│  Raw Results (Ctrl+3)              │  (when Info focused)   │
│  Sources list / search results     │                        │
│  before LLM processing             │                        │
└────────────────────────────────────┴────────────────────────┘
│  Status Bar  [mode badge]  ℹ status message                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Keyboard Shortcuts

### Global (IDE-Safe)

| Key | Action |
|-----|--------|
| `?` / `Ctrl+H` | Toggle help overlay |
| `Ctrl+C` | Quit |
| `Ctrl+N` | Cycle focus forward (Input → Results → Raw → Info) |
| `Ctrl+P` | Cycle focus backward |
| `Ctrl+1` | Focus Query Input |
| `Ctrl+2` | Focus Results Viewer |
| `Ctrl+3` | Focus Raw Results |
| `Ctrl+4` | Focus Info Panel |
| `Ctrl+N` (Info Panel focused) | Cycle tabs: Stats → Sources → History |
| `Esc` | Return focus to input |

### Input Box

| Key | Action |
|-----|--------|
| `Enter` | Submit query or `/command` |
| `Ctrl+D` | Clear input |

### Scrolling (when viewer focused)

| Key | Action |
|-----|--------|
| `j` / `↓` | Scroll down one line |
| `k` / `↑` | Scroll up one line |
| `Alt+↓` / `Alt+↑` | Scroll down/up (works even from input) |
| `PageDown` / `Ctrl+D` | Scroll down one page |
| `PageUp` / `Ctrl+U` | Scroll up one page |
| `Home` / `End` | Jump to top / bottom |

---

## Slash Commands

| Command | Description |
|---------|-------------|
| `/config <file>` | Load a config file (JSON5, JSON, TOML) |
| `/config show` | Display the currently loaded config |
| `/load <file>` | Load and process a document |
| `/load <file> --rebuild` | Force full rebuild before loading |
| `/clear` | Clear graph (keep documents) |
| `/rebuild` | Re-extract from loaded documents |
| `/stats` | Show entity/relationship/chunk counts |
| `/entities [filter]` | List entities, optionally filtered |
| `/mode ask\|explain\|reason` | Switch query mode (sticky) |
| `/reason <query>` | One-shot reasoning query (decomposition) |
| `/export <file.md>` | Export query history to Markdown |
| `/workspace list` | List saved workspaces |
| `/workspace save <name>` | Save current graph to disk |
| `/workspace <name>` | Load a saved workspace |
| `/workspace delete <name>` | Delete a workspace |
| `/help` | Show full command help |

---

## Query Modes

Switch with `/mode <mode>` or the badge in the status bar shows the active mode.

| Mode | Command | What it does |
|------|---------|--------------|
| `ASK` (default) | `/mode ask` | Plain answer, fastest |
| `EXPLAIN` | `/mode explain` | Answer + confidence score + source references; Sources tab auto-opens |
| `REASON` | `/mode reason` | Query decomposition — splits complex questions into sub-queries |

One-shot override (doesn't change sticky mode):
```
/reason Compare the main arguments of each speaker about love
```

---

## Architecture

```
graphrag-cli/src/
├── main.rs                    # CLI entry point (clap)
├── app.rs                     # Main event loop, action routing
├── action.rs                  # Action enum, QueryMode, QueryExplainedPayload
├── commands/mod.rs            # Slash command parser
├── config.rs                  # Config file loading (JSON5/JSON/TOML)
├── theme.rs                   # Dark/light color themes
├── tui.rs                     # Terminal setup/teardown
├── query_history.rs           # Per-session query history
├── workspace.rs               # Workspace metadata management
├── mode.rs                    # Input mode detection
├── handlers/
│   ├── graphrag.rs            # Thread-safe GraphRAG wrapper (Arc<Mutex<>>)
│   ├── bench.rs               # Benchmark runner (JSON output)
│   └── file_ops.rs            # File utilities
└── ui/
    ├── markdown.rs            # Markdown → ratatui Line<'static> parser
    ├── spinner.rs             # Braille spinner animation
    └── components/
        ├── query_input.rs     # Text input widget
        ├── results_viewer.rs  # Markdown-rendered answer + scrollbar
        ├── raw_results_viewer.rs  # Raw search results
        ├── info_panel.rs      # 3-tab panel (Stats/Sources/History)
        ├── status_bar.rs      # Status + query mode badge
        └── help_overlay.rs    # Modal help popup
```

---

## Technology Stack

- **[Ratatui](https://ratatui.rs/) 0.29** — TUI framework (immediate mode rendering)
- **[Crossterm](https://github.com/crossterm-rs/crossterm) 0.28** — Cross-platform terminal events
- **[tui-textarea](https://github.com/rhysd/tui-textarea) 0.7** — Multi-line input widget
- **[Tokio](https://tokio.rs/) 1.32** — Async runtime
- **[Clap](https://github.com/clap-rs/clap) 4.5** — CLI argument parsing
- **[Dialoguer](https://github.com/console-rs/dialoguer) 0.11** — Interactive setup wizard
- **[color-eyre](https://github.com/eyre-rs/eyre) 0.6** — Error reporting
- **[graphrag-core](../graphrag-core/)** — Knowledge graph engine (direct library call)

---

## License

Same license as the parent `graphrag-rs` project.
