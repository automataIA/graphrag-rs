# 🚀 GraphRAG-CLI Complete User Guide

**GraphRAG-CLI** is a modern Terminal User Interface (TUI) for GraphRAG operations, built with Ratatui for interactive knowledge graph exploration.

## 📋 Table of Contents

- [Installation & Build](#installation--build)
- [Quick Start](#quick-start)
- [Usage Modes](#usage-modes)
- [Slash Commands](#slash-commands)
- [Workspace Management](#workspace-management)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Practical Examples](#practical-examples)
- [Troubleshooting](#troubleshooting)

---

## 🔧 Installation & Build

### Prerequisites
- Rust 1.70 or higher
- Ollama installed and running (for LLM models)

### Building

```bash
# From the project root directory
cargo build --release -p graphrag-cli

# The binary will be available at:
# ./target/release/graphrag-cli
```

### Add to PATH (optional)

```bash
# Linux/macOS
export PATH="$PATH:$(pwd)/target/release"

# Or create a symbolic link
sudo ln -s $(pwd)/target/release/graphrag-cli /usr/local/bin/graphrag-cli
```

---

## ⚡ Quick Start

### 1. Launch the Interactive TUI

```bash
# Basic launch
./target/release/graphrag-cli

# With configuration file
./target/release/graphrag-cli --config docs-example/symposium_config.toml

# With specific workspace
./target/release/graphrag-cli --workspace my_project

# With debug logging
./target/release/graphrag-cli --debug
```

### 2. First Time Setup

1. **Start the TUI:**
   ```bash
   ./target/release/graphrag-cli
   ```

2. **Load a configuration:**
   In the TUI, press `Shift+Tab` to enter command mode, then type:
   ```
   /config docs-example/symposium_config.toml
   ```

3. **Load a document:**
   ```
   /load docs-example/platos_symposium.txt
   ```

4. **Execute queries:**
   Return to Query Mode (press `Shift+Tab`) and type:
   ```
   What does Socrates say about love?
   ```

---

## 🎯 Usage Modes

GraphRAG-CLI has **two primary modes**:

### 📝 Query Mode
- **Purpose:** Execute queries on the knowledge graph
- **How to use:** Type your question directly
- **Example:** `What are the main themes in the Symposium?`
- **Border color:** Green

### ⚙️ Command Mode
- **Purpose:** Execute system commands (slash commands)
- **How to use:** Press `Shift+Tab` to switch from Query Mode
- **Example:** `/config myfile.toml`
- **Border color:** Yellow

**Switch between modes:** `Shift+Tab`

---

## 🔀 Slash Commands

Slash commands are only available in **Command Mode**.

### `/config <file>`
Load a GraphRAG configuration file (TOML or JSON5)

```bash
# Example with TOML
/config docs-example/symposium_config.toml

# Example with absolute path
/config /home/user/graphrag-rs/my_config.toml

# Example with relative path
/config ../config/templates/academic_research.toml
```

**What it does:**
- Loads LLM configuration (Ollama, OpenAI, etc.)
- Configures embeddings and chunking strategy
- Initializes the knowledge graph

---

### `/load <file>`
Load and process a document into the knowledge graph

```bash
# Load a text file
/load docs-example/platos_symposium.txt

# Load a markdown file
/load ~/documents/research_paper.md

# Load multiple documents (execute multiple times)
/load document1.txt
/load document2.txt
/load document3.txt
```

**What it does:**
1. Reads the document
2. Splits into chunks (based on config)
3. Extracts entities and relationships using **REAL LLM calls** (not pattern matching!)
4. Builds the knowledge graph
5. Generates embeddings

**Processing time (with TRUE LLM gleaning extraction):**
- **Small docs** (5-10 pages): 5-15 minutes
- **Medium docs** (50-100 pages): 30-60 minutes
- **Large docs** (500-1000 pages): 2-4 hours

This is REAL semantic extraction with multi-round LLM refinement, not instant pattern matching!

---

### `/stats`
Show knowledge graph statistics

```bash
/stats
```

**Output:**
```
Knowledge Graph Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 Documents:      3
📄 Chunks:         127
👤 Entities:       45
🔗 Relationships:  89
📊 Graph Density:  0.087
```

---

### `/entities [filter]`
List entities in the knowledge graph with optional filter

```bash
# List all entities
/entities

# Filter by name
/entities socrates

# Filter by type
/entities PERSON

# Filter by concept
/entities love
```

**Output:**
```
Entities (filtered: "socrates"):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👤 Socrates [PERSON]
   Mentions: 34
   Confidence: 0.95
   Description: Greek philosopher, student of Plato...

💡 Socratic Method [CONCEPT]
   Mentions: 12
   Confidence: 0.89
   Description: Method of inquiry through questioning...
```

---

### `/workspace <name>`
Switch to a different workspace

```bash
# Switch workspace
/workspace my_project

# Switch to different workspace
/workspace philosophy_research
```

**What are workspaces?**
- Separate directories for different projects
- Each workspace has its own knowledge graph
- Each workspace has its own query history
- Allows working on multiple projects simultaneously

---

### `/help`
Show list of all available commands

```bash
/help
```

---

## 🗂️ Workspace Management

Workspaces allow you to organize multiple projects separately.

### Terminal Commands

```bash
# List all workspaces
./target/release/graphrag-cli workspace list

# Create new workspace
./target/release/graphrag-cli workspace create philosophy_research

# Show workspace information
./target/release/graphrag-cli workspace info <workspace-id>

# Delete workspace
./target/release/graphrag-cli workspace delete <workspace-id>
```

### Example: Create and Use a Workspace

```bash
# 1. Create workspace
./target/release/graphrag-cli workspace create philosophy_research

# Output:
# ✅ Workspace created successfully!
#    Name: philosophy_research
#    ID:   a1b2c3d4-e5f6-7890-abcd-ef1234567890
#
# Use it with: graphrag-cli tui --workspace a1b2c3d4-e5f6-7890-abcd-ef1234567890

# 2. Start TUI with the workspace
./target/release/graphrag-cli --workspace a1b2c3d4-e5f6-7890-abcd-ef1234567890

# 3. In the TUI, load configuration and documents
# (command mode)
/config docs-example/symposium_config.toml
/load docs-example/platos_symposium.txt
```

### Workspace Paths

Workspaces are saved in:
```
~/.local/share/graphrag-cli/workspaces/
└── <workspace-id>/
    ├── metadata.json          # Workspace information
    ├── query_history.json     # Query history
    ├── graph.db              # Knowledge graph (if persistent)
    └── embeddings/           # Vector embeddings
```

---

## ⌨️ Keyboard Shortcuts

### General Navigation

| Key | Action |
|-----|--------|
| `Shift+Tab` | Switch mode (Query ↔ Command) |
| `Ctrl+C` | Exit application |
| `?` | Show help overlay (in Query Mode) |
| `Esc` | Close help overlay |

### Input Mode

| Key | Action |
|-----|--------|
| `Enter` | Execute query/command |
| `↑` / `↓` | Navigate query/command history |
| `Ctrl+U` | Clear input line |
| `Ctrl+W` | Delete previous word |

### Results Navigation

| Key | Action |
|-----|--------|
| `j` / `↓` | Scroll down |
| `k` / `↑` | Scroll up |
| `Ctrl+D` / `PgDn` | Page down |
| `Ctrl+U` / `PgUp` | Page up |
| `Home` | Beginning of results |
| `End` | End of results |

### Help Mode (pressing `?`)

| Key | Action |
|-----|--------|
| `Esc` | Close help |
| `q` | Close help |
| `↑` / `↓` | Scroll help |

---

## 💡 Practical Examples

### Example 1: Philosophy Setup (Plato's Symposium)

```bash
# 1. Start TUI
./target/release/graphrag-cli

# 2. In Command Mode (Shift+Tab)
/config docs-example/symposium_config.toml
/load docs-example/platos_symposium.txt

# 3. Wait for processing (30-60 seconds)

# 4. In Query Mode (Shift+Tab to return)
What does Socrates say about love?
Who are the main speakers in the Symposium?
Explain the myth of the androgyne
```

### Example 2: Multi-Document Analysis

```bash
# 1. Start with configuration
./target/release/graphrag-cli --config config/templates/academic_research.toml

# 2. Load multiple documents (Command Mode)
/load papers/paper1.txt
/load papers/paper2.txt
/load papers/paper3.txt

# 3. Check statistics
/stats

# 4. Analytical queries (Query Mode)
Compare the methodologies used in the three papers
What are the common themes across all documents?
List all authors mentioned
```

### Example 3: Technical Research

```bash
# 1. Create dedicated workspace
./target/release/graphrag-cli workspace create rust_docs

# 2. Start with workspace
./target/release/graphrag-cli --workspace <workspace-id> \
  --config config/templates/technical_documentation.toml

# 3. Load documentation (WARNING: This will take time with real LLM processing!)
/load rust-docs/ownership.md    # ~10-15 minutes per document
/load rust-docs/concurrency.md
/load rust-docs/async.md

# 4. Technical queries
Explain Rust's ownership system
How does async/await work in Rust?
What are the main concurrency primitives?
```

### Example 4: Entity Exploration

```bash
# 1. After loading documents
/entities

# 2. Filter by type
/entities PERSON
/entities CONCEPT
/entities LOCATION

# 3. Search specific entities
/entities socrates
/entities love
/entities athens
```

---

## 🔍 Troubleshooting

### Problem: "Configuration not loaded"

**Symptoms:** Cannot load documents or execute queries

**Solution:**
```bash
# 1. Make sure to load configuration first
/config path/to/config.toml

# 2. Verify the file exists
ls -la path/to/config.toml

# 3. Check TOML syntax
cat path/to/config.toml
```

---

### Problem: "Failed to connect to Ollama"

**Symptoms:** Errors during document loading or queries

**Solution:**
```bash
# 1. Verify Ollama is running
ollama list

# 2. Start Ollama if necessary
ollama serve

# 3. Test connection
curl http://localhost:11434/api/tags

# 4. Verify configured model is available
ollama list | grep qwen3
```

---

### Problem: "Document processing is slow"

**Symptoms:** Document loading takes a very long time

**NOTE:** This is **EXPECTED BEHAVIOR** with real LLM-based gleaning extraction!
- Small docs (5-10 pages): 5-15 minutes
- Medium docs (50-100 pages): 30-60 minutes
- Large docs (500-1000 pages): 2-4 hours

**This is NOT a bug** - it's genuine semantic extraction with real LLM API calls.

**If you need faster processing (lower quality):**
```toml
# Speed optimizations in config.toml

[entities]
max_gleaning_rounds = 2      # Reduce from 4 to 2
use_llm_completion_check = false  # Skip LLM completion check

[text.chunking]
chunk_size = 1000            # Larger chunks = fewer LLM calls
chunk_overlap = 50           # Reduce overlap

[ollama]
chat_model = "llama3.1:3b"   # Use smaller/faster model
```

---

### Problem: "TUI is corrupted/garbled"

**Symptoms:** Strange characters, corrupted layout

**Solution:**
```bash
# 1. Reset terminal
reset

# 2. Check TERM variable
echo $TERM
# Should be: xterm-256color or similar

# 3. Set if necessary
export TERM=xterm-256color

# 4. Restart TUI
./target/release/graphrag-cli
```

---

### Problem: "Workspace not found"

**Symptoms:** Error loading workspace

**Solution:**
```bash
# 1. List available workspaces
./target/release/graphrag-cli workspace list

# 2. Create new workspace if necessary
./target/release/graphrag-cli workspace create my_project

# 3. Use correct ID
./target/release/graphrag-cli --workspace <correct-id>
```

---

### Problem: "Out of memory"

**Symptoms:** Crashes during large document processing

**Solution:**
```toml
# Reduce memory usage in config.toml

[text.chunking]
max_tokens = 300  # Reduce chunk size
overlap = 20      # Reduce overlap

[embeddings]
batch_size = 16   # Reduce batch for embeddings

[caching]
enabled = true    # Enable caching to save RAM
max_size_mb = 100 # Limit cache size
```

---

## 📊 Recommended Configurations

### For Small Documents (<100KB)

```toml
[text.chunking]
method = "semantic"
max_tokens = 500
overlap = 50

[entity.extraction]
use_gleaning = true
max_iterations = 3
```

### For Large Documents (>1MB)

```toml
[text.chunking]
method = "simple"
max_tokens = 300
overlap = 20

[entity.extraction]
use_gleaning = false
max_iterations = 1

[caching]
enabled = true
max_size_mb = 500
```

### For Maximum Performance

```toml
[text.chunking]
method = "simple"
max_tokens = 400

[entity.extraction]
use_gleaning = false

[async_processing]
enabled = true
max_concurrent_operations = 8

[caching]
enabled = true
cache_embeddings = true
```

---

## 📝 Configuration Examples

### Basic Template (TOML)

```toml
# my_config.toml

[llm]
provider = "ollama"
model = "qwen3:8b"
temperature = 0.7
max_tokens = 2000

[embeddings]
provider = "ollama"
model = "nomic-embed-text"
dimensions = 768

[text.chunking]
method = "semantic"
max_tokens = 500
overlap = 50

[entity.extraction]
use_gleaning = true
max_iterations = 2

[graph]
enable_pagerank = true
pagerank_damping = 0.85

[storage]
type = "memory"
```

### Advanced Template (JSON5)

```json5
// my_config.json5
{
  llm: {
    provider: "ollama",
    model: "qwen3:8b",
    temperature: 0.7,
    max_tokens: 2000,
  },

  embeddings: {
    provider: "ollama",
    model: "nomic-embed-text",
    dimensions: 768,
  },

  text: {
    chunking: {
      method: "hierarchical",
      max_tokens: 500,
      overlap: 50,
    },
  },

  entity: {
    extraction: {
      use_gleaning: true,
      max_iterations: 3,
      confidence_threshold: 0.7,
    },
  },

  graph: {
    enable_pagerank: true,
    enable_leiden: true,
  },

  caching: {
    enabled: true,
    max_size_mb: 200,
    cache_embeddings: true,
  },
}
```

---

## 🎓 Complete Tutorial: First Project

Follow this step-by-step tutorial for your first GraphRAG project:

### Step 1: Preparation

```bash
# 1. Create project directory
mkdir -p ~/graphrag-projects/philosophy
cd ~/graphrag-projects/philosophy

# 2. Create configuration file
cat > config.toml << 'EOF'
[llm]
provider = "ollama"
model = "qwen3:8b"
temperature = 0.7

[embeddings]
provider = "ollama"
model = "nomic-embed-text"

[text.chunking]
method = "semantic"
max_tokens = 500

[entity.extraction]
use_gleaning = true
max_iterations = 2
EOF

# 3. Prepare documents
mkdir documents
# Copy your documents to ./documents/
```

### Step 2: Launch and Configuration

```bash
# 1. Start GraphRAG-CLI
/path/to/graphrag-rs/target/release/graphrag-cli

# 2. In the TUI, switch to Command Mode
# Press: Shift+Tab

# 3. Load configuration
/config ~/graphrag-projects/philosophy/config.toml

# 4. Verify success (should see confirmation message)
```

### Step 3: Loading Documents

```bash
# In Command Mode, load documents one at a time
/load ~/graphrag-projects/philosophy/documents/doc1.txt

# Wait for completion (30-60 seconds)

/load ~/graphrag-projects/philosophy/documents/doc2.txt

# Repeat for all documents
```

### Step 4: Exploration

```bash
# 1. Check statistics
/stats

# 2. Explore entities
/entities

# 3. Filter entities by type
/entities PERSON
/entities CONCEPT
```

### Step 5: Querying

```bash
# Switch to Query Mode
# Press: Shift+Tab

# Execute analytical queries
What are the main philosophical concepts discussed?

# Queries about specific entities
Tell me about Socrates and his ideas

# Comparative queries
Compare Plato's and Aristotle's views on reality

# Thematic queries
What are the main themes in these documents?
```

---

## 🚀 Tips & Tricks

### 1. Use Query History

- Press `↑` / `↓` to navigate previous queries
- Edit and re-execute queries quickly
- History is persistent per workspace

### 2. Organize by Projects

```bash
# Create workspace for each project
graphrag-cli workspace create project_A
graphrag-cli workspace create project_B
graphrag-cli workspace create project_C

# Switch quickly between projects
graphrag-cli --workspace <project_A_id>
```

### 3. Optimize Performance

```toml
# Enable all performance features
[async_processing]
enabled = true
max_concurrent_operations = 8

[caching]
enabled = true
cache_embeddings = true

[parallel]
max_workers = 4
```

### 4. Debugging

```bash
# Start with debug logging
graphrag-cli --debug

# Logs are saved in:
# Linux: ~/.local/share/graphrag-cli/logs/graphrag-cli.log
# View logs in real-time:
tail -f ~/.local/share/graphrag-cli/logs/graphrag-cli.log
```

### 5. Batch Processing

```bash
# Use script to load many documents
for file in documents/*.txt; do
  echo "/load $file"
done > commands.txt

# Then execute manually in TUI
# (or integrate with future automation)
```

---

## 📚 Additional Resources

### Documentation

- [GraphRAG-Core README](../graphrag-core/README.md)
- [Configuration Guide](../CONFIGURATION_GUIDE.md)
- [Architecture Documentation](../ARCHITECTURE.md)

### Configuration Templates

- `config/templates/academic_research.toml`
- `config/templates/technical_documentation.toml`
- `config/templates/narrative_fiction.toml`

### Examples

- `examples/multi_document_pipeline.rs`
- `examples/symposium_real_search.rs`
- `docs-example/symposium_config.toml`

---

## 🤝 Contributing

Found a bug or want to contribute? Open an issue or pull request at:
https://github.com/yourusername/graphrag-rs

---

## 📄 License

MIT License - See [LICENSE](../LICENSE) for details

---

## 🎯 Quick Reference Card

```
╔════════════════════════════════════════════════════════════╗
║              GraphRAG-CLI Quick Reference                  ║
╠════════════════════════════════════════════════════════════╣
║ STARTUP                                                    ║
║   graphrag-cli                      Start TUI              ║
║   graphrag-cli --config FILE        With configuration     ║
║   graphrag-cli --workspace ID       With workspace         ║
║                                                            ║
║ MODES                                                      ║
║   Shift+Tab    Switch Query ↔ Command Mode                ║
║   ?            Show help (Query Mode)                      ║
║   Ctrl+C       Exit                                        ║
║                                                            ║
║ COMMANDS (Command Mode)                                    ║
║   /config FILE      Load configuration                     ║
║   /load FILE        Load document                          ║
║   /stats            Show statistics                        ║
║   /entities [FILT]  List entities (with optional filter)   ║
║   /workspace NAME   Switch workspace                       ║
║   /help             Show all commands                      ║
║                                                            ║
║ WORKSPACE (from terminal)                                  ║
║   workspace list         List workspaces                   ║
║   workspace create NAME  Create workspace                  ║
║   workspace info ID      Workspace info                    ║
║   workspace delete ID    Delete workspace                  ║
║                                                            ║
║ NAVIGATION                                                 ║
║   ↑/↓         Navigate history / scroll results            ║
║   j/k         Vim-style scroll (down/up)                   ║
║   Ctrl+D/U    Page down/up                                 ║
║   Home/End    Beginning/end of results                     ║
║   Ctrl+U      Clear line (in input)                        ║
║   Ctrl+W      Delete word (in input)                       ║
╚════════════════════════════════════════════════════════════╝
```

---

**Happy GraphRAG-ing! 🚀✨**
