# graphrag-cli 🖥️
        
A modern Terminal User Interface (TUI) for GraphRAG operations, built with [Ratatui](https://ratatui.rs/).

## ✨ Features

- 🎨 **Beautiful TUI** - Modern terminal interface with rich components
- ⚡ **Fast & Responsive** - Immediate mode rendering with Ratatui
- 🎹 **Vim-like Keybindings** - Intuitive keyboard shortcuts
- 🎨 **Theme Support** - Dark and light themes
- 📊 **Multi-pane Layout** - Query input, results viewer, entity explorer, and status bar
- 🔍 **Interactive Query Execution** - Execute GraphRAG queries directly from terminal
- 🌲 **Entity Explorer** - Browse and navigate graph entities from real knowledge graph
- 📜 **Scrollable Results** - Vim-style scrolling through query results
- ⌨️ **Help System** - Built-in help overlay (press `?`)
- 📈 **Status Bar** - Real-time progress indicators and error display
- 🔗 **Direct Integration** - Uses graphrag-core library directly (no server needed)
- 📝 **TOML Configuration** - Load and manage configs from TOML files
- 📚 **Document Processing** - Process documents through complete 7-stage pipeline with TRUE LLM-based gleaning extraction
- ⏱️ **Real LLM Processing** - Genuine multi-round entity extraction (15-30 seconds per chunk, not instant pattern matching)

## 🚀 Installation

```bash
cd graphrag-rs
cargo build --package graphrag_cli --release
```

Or run directly:

```bash
cargo run --package graphrag_cli
```

## 📖 Usage

### Quick Start Example

```bash
# 1. Create a config file (or use an example from config/templates/)
cat > my_config.toml << EOF
[general]
output_dir = "./output"
log_level = "info"

[pipeline]
workflows = ["extract_text", "extract_entities", "build_graph"]

[pipeline.text_extraction]
chunk_size = 500
chunk_overlap = 100

[ollama]
enabled = true
host = "http://localhost"
port = 11434
chat_model = "llama3.1:8b"
embedding_model = "nomic-embed-text"
EOF

# 2. Load and process your document (NOTE: Real LLM processing takes time!)
# Small docs (5-10 pages): 5-15 minutes
# Medium docs (50-100 pages): 30-60 minutes
# Large docs (500-1000 pages): 2-4 hours
./target/release/graphrag_cli load your_document.txt --config my_config.toml

# 3. Start the interactive TUI to query
./target/release/graphrag_cli --config my_config.toml tui

# Or query directly from command line
./target/release/graphrag_cli --config my_config.toml query "What are the main themes?"
```

### Interactive TUI Mode (Default)

```bash
# Start the TUI
./target/release/graphrag_cli

# Or with custom server URL
./target/release/graphrag_cli --server http://localhost:8080
```

### Command Line Mode

```bash
# Initialize GraphRAG with a configuration file
./target/release/graphrag_cli init config.toml

# Load and process a document
./target/release/graphrag_cli load document.txt --config config.toml

# Execute a single query
./target/release/graphrag_cli query "find all entities related to AI"

# List entities
./target/release/graphrag_cli entities

# Show graph statistics
./target/release/graphrag_cli stats
```

## ⌨️ Keyboard Shortcuts

### Global Shortcuts
- `q` or `Ctrl+C` - Quit application
- `?` - Toggle help overlay
- `Tab` - Switch to next pane
- `Shift+Tab` - Switch to previous pane

### Query Input (when active)
- `Enter` - Execute query
- `Ctrl+D` - Clear input
- Normal text editing keys

### Results Viewer (when active)
- `j` or `↓` - Scroll down one line
- `k` or `↑` - Scroll up one line
- `Ctrl+D` or `Page Down` - Scroll down one page
- `Ctrl+U` or `Page Up` - Scroll up one page
- `Home` - Jump to top
- `End` - Jump to bottom

### Entity Explorer (when active)
- `j` or `↓` - Next entity
- `k` or `↑` - Previous entity
- `Enter` or `Space` - Expand/collapse entity

## 🎨 Architecture

```
graphrag-cli/
├── src/
│   ├── main.rs              # Entry point & CLI argument parsing
│   ├── app.rs               # Application state & event loop
│   ├── ui/
│   │   ├── mod.rs
│   │   ├── theme.rs         # Color themes
│   │   └── components/
│   │       ├── query_input.rs      # Query input widget
│   │       ├── results_viewer.rs   # Results display widget
│   │       └── entity_explorer.rs  # Entity tree widget
│   ├── handlers/
│   │   └── events.rs        # Event handling
│   └── integrations/
│       └── graphrag.rs      # GraphRAG API client
└── Cargo.toml
```

## 🛠️ Technology Stack

- **TUI Framework**: [Ratatui](https://ratatui.rs/) - Modern Rust TUI library
- **Terminal Backend**: [Crossterm](https://github.com/crossterm-rs/crossterm) - Cross-platform terminal manipulation
- **Widgets**:
  - [tui-textarea](https://github.com/rhysd/tui-textarea) - Multi-line text editor
  - [tui-tree-widget](https://github.com/EdJoPaTo/tui-rs-tree-widget) - Tree view widget
  - [tui-popup](https://github.com/joshka/tui-popup) - Popup dialogs
- **CLI**: [clap](https://github.com/clap-rs/clap) - Command-line argument parser
- **Error Handling**: [color-eyre](https://github.com/eyre-rs/eyre) - Beautiful error reports
- **Logging**: [tracing](https://github.com/tokio-rs/tracing) - Structured logging

## 🎯 Inspiration

This TUI is inspired by modern terminal tools:
- [Gollama](https://github.com/sammcj/gollama) - TUI for Ollama model management
- [GitUI](https://github.com/extrawurst/gitui) - Fast terminal UI for Git
- [ATAC](https://github.com/Julien-cpsn/ATAC) - API testing TUI
- [Claude Code](https://claude.ai/code) - Modern developer experience

## 🚧 Development Status

### ✅ Implemented
- [x] Project structure with proper modularization
- [x] CLI argument parsing with subcommands (`init`, `load`, `query`, `entities`, `stats`)
- [x] Main application loop with event handling
- [x] Four-pane UI layout (Query Input, Results, Entity Explorer, Status Bar)
- [x] All core UI components with rendering logic
- [x] Keyboard shortcuts and navigation
- [x] Theme support (dark/light)
- [x] Help overlay system with status bar documentation
- [x] Vim-style scrolling in results viewer
- [x] GraphRAG Core integration (direct library integration)
- [x] Real entity data in explorer from knowledge graph
- [x] Document loading and processing through 7-stage pipeline
- [x] TOML configuration loading and validation
- [x] Real-time progress indicators in status bar
- [x] Error display with color-coded status icons
- [x] Query execution with graphrag-core

### 🔄 In Progress
- [ ] GraphRAG API integration (HTTP server mode - future)
- [ ] Async query execution with cancellation support

### 📝 Planned Features
- [ ] Query history and session management
- [ ] Save/load queries
- [ ] Export results to various formats
- [ ] Live reload configuration
- [ ] Logging panel (toggle-able)
- [ ] Performance monitoring dashboard
- [ ] Custom theme configuration
- [ ] Graph visualization (ASCII art)
- [ ] Search and filter in results
- [ ] Syntax highlighting in query input

## 🤝 Contributing

Contributions are welcome! This is part of the larger [graphrag-rs](https://github.com/automataIA/graphrag-rs) project.

## 📄 License

Same license as the parent graphrag-rs project.

## 🙏 Acknowledgments

- [Ratatui](https://ratatui.rs/) team for the amazing TUI framework
- [Gollama](https://github.com/sammcj/gollama) for architectural inspiration
- The Rust TUI community for excellent examples and widgets
