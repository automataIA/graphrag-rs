# GraphRAG-rs

A high-performance Rust implementation of GraphRAG (Graph-based Retrieval Augmented Generation) for building knowledge graphs from documents and querying them with natural language.

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/your-username/graphrag-rs)

## Quick Start (30 seconds)

```bash
# Clone and build
git clone https://github.com/your-username/graphrag-rs.git
cd graphrag-rs
cargo build --release

# Use a pre-configured template (multiple available!)
cp config.toml my_config.toml
# Or choose a specific template:
# cp config_tom_sawyer.toml my_config.toml
# cp config_complete.toml my_config.toml

# Edit the config to point to YOUR document:
# nano my_config.toml
# Change this line: input_document_path = "path/to/your/document.txt"
# Change this line: output_dir = "./output/your_project"

# Process your document and ask questions
cargo run --bin simple_cli my_config.toml "What is this document about?"
```

**Config files explained:**
- `input_document_path` - Path to your text file to analyze
- `output_dir` - Where GraphRAG saves the knowledge graph
- Templates in root: `config.toml`, `config_complete.toml`, `config_tom_sawyer.toml`
- Pick one, copy it, change the document path, and you're ready!

## Installation

### Prerequisites

- Rust 1.70 or later
- (Optional) Ollama for local LLM support - [Install Ollama](https://ollama.ai)

### From Source

```bash
git clone https://github.com/your-username/graphrag-rs.git
cd graphrag-rs
cargo build --release

# Optional: Install globally
cargo install --path .
```

## Basic Usage

### 1. Simple API (One Line)

```rust
use graphrag_rs::simple;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let answer = simple::answer("Your document text", "Your question")?;
    println!("Answer: {}", answer);
    Ok(())
}
```

### 2. Stateful API (Multiple Queries)

```rust
use graphrag_rs::easy::SimpleGraphRAG;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = SimpleGraphRAG::from_text("Your document text")?;

    let answer1 = graph.ask("What is this about?")?;
    let answer2 = graph.ask("Who are the main characters?")?;

    println!("Answer 1: {}", answer1);
    println!("Answer 2: {}", answer2);
    Ok(())
}
```

### 3. Builder API (Configurable)

```rust
use graphrag_rs::{GraphRAG, ConfigPreset};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut graphrag = GraphRAG::builder()
        .with_preset(ConfigPreset::Balanced)
        .auto_detect_llm()
        .build()?;

    graphrag.add_document("Your document")?;
    let answer = graphrag.ask("Your question")?;

    println!("Answer: {}", answer);
    Ok(())
}
```

### 4. CLI Usage

GraphRAG-rs provides two CLI tools:

#### Smart CLI (Recommended) - `simple_cli`
Automatically detects if the knowledge graph needs building and handles everything for you:

```bash
# Build the Smart CLI
cargo build --release --bin simple_cli

# Process document and answer question in one command
cargo run --bin simple_cli config.toml "What are the main themes?"

# Interactive mode - builds graph if needed, then waits for questions
cargo run --bin simple_cli config.toml

# How it works:
# 1. Loads your TOML configuration
# 2. Checks if knowledge graph exists
# 3. Builds graph if needed (shows progress)
# 4. Answers your question using Ollama
# 5. Saves results to output directory
```

#### Manual CLI - `graphrag-rs`
For advanced users who want full control:

```bash
# Build the manual CLI
cargo build --release

# Step 1: Build knowledge graph
./target/release/graphrag-rs config.toml build

# Step 2: Query the graph
./target/release/graphrag-rs config.toml query "Your question"
```

## Configuration

### Basic Configuration (config.toml)

The project includes several ready-to-use configuration templates:

**Available Templates:**
- `config.toml` - Basic configuration for general use
- `config_complete.toml` - Full configuration with all options
- `config_tom_sawyer.toml` - Pre-configured for book processing
- `config_example.toml` - Annotated template with explanations

**Essential Configuration Fields:**

```toml
[general]
# IMPORTANT: Change these two paths for your project!
input_document_path = "path/to/your/document.txt"  # Your document to process
output_dir = "./output/your_project"                # Where to save results

[pipeline]
chunk_size = 800        # Size of text chunks (adjust based on document type)
chunk_overlap = 200     # Overlap to preserve context between chunks

[ollama]
enabled = true
host = "http://localhost"
port = 11434
chat_model = "llama3.1:8b"           # LLM for text generation
embedding_model = "nomic-embed-text"  # Model for embeddings
```

**Quick Setup:**
1. Copy a template: `cp config_complete.toml my_project.toml`
2. Edit `input_document_path` to point to your document
3. Edit `output_dir` to set where results are saved
4. Run: `cargo run --bin simple_cli my_project.toml`

See [config_example.toml](config_example.toml) for detailed explanations of all options.

## Core Features

### Knowledge Graph Construction
- **Entity Extraction**: Automatically identifies people, places, organizations, and concepts
- **Relationship Discovery**: Finds connections between entities
- **Hierarchical Organization**: Creates multi-level document summaries

### Intelligent Retrieval
- **Semantic Search**: Find information using meaning, not just keywords
- **Hybrid Retrieval**: Combines keyword and semantic search for best results
- **Context-Aware Answers**: Generates responses based on document context

### Performance
- **Parallel Processing**: Utilizes all CPU cores for fast processing
- **Efficient Storage**: Minimal memory footprint (<100MB for typical documents)
- **Fast Queries**: Sub-second response times for most queries

### Flexibility
- **Local LLM Support**: Works with Ollama for private, offline processing
- **Configurable Pipeline**: Adjust chunking, extraction, and retrieval parameters
- **Multiple APIs**: Choose complexity level based on your needs

## Examples

### Quick Example: Using Config Templates

```bash
# Example 1: Process a book using existing template
cp config_tom_sawyer.toml my_book_config.toml
# Edit my_book_config.toml:
#   input_document_path = "books/my_book.txt"
#   output_dir = "./output/my_book"
cargo run --bin simple_cli my_book_config.toml "Who are the main characters?"

# Example 2: Process a research paper
cp config.toml research_config.toml
# Edit research_config.toml:
#   input_document_path = "papers/research.txt"
#   output_dir = "./output/research"
#   chunk_size = 500  # Smaller chunks for technical content
cargo run --bin simple_cli research_config.toml "What is the main hypothesis?"

# Example 3: Process with full configuration
cp config_complete.toml advanced_config.toml
# Edit all the parameters you need in advanced_config.toml
cargo run --bin simple_cli advanced_config.toml
```

### Process a Book

```rust
use graphrag_rs::{GraphRAG, Document};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read document
    let content = fs::read_to_string("book.txt")?;

    // Create and configure GraphRAG
    let mut graphrag = GraphRAG::builder()
        .with_chunk_size(1000)
        .with_chunk_overlap(200)
        .build()?;

    // Process document
    let doc = Document::new("book", content);
    graphrag.add_document(doc)?;

    // Query
    let answer = graphrag.ask("What are the main themes?")?;
    println!("Answer: {}", answer);

    Ok(())
}
```

### Use with Ollama

```rust
use graphrag_rs::{GraphRAG, OllamaConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure Ollama
    let ollama = OllamaConfig::new()
        .with_model("llama3.1:8b")
        .with_embedding_model("nomic-embed-text");

    // Create GraphRAG with Ollama
    let mut graphrag = GraphRAG::builder()
        .with_llm(ollama)
        .build()?;

    // Use as normal
    graphrag.add_text("Your document")?;
    let answer = graphrag.ask("Your question")?;

    Ok(())
}
```

### Batch Processing

```rust
use graphrag_rs::GraphRAG;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut graphrag = GraphRAG::new_default()?;

    // Process multiple documents
    for file in ["doc1.txt", "doc2.txt", "doc3.txt"] {
        let content = fs::read_to_string(file)?;
        graphrag.add_text(&content)?;
    }

    // Query across all documents
    let answer = graphrag.ask("What connects these documents?")?;
    println!("Answer: {}", answer);

    Ok(())
}
```

## Architecture Overview

GraphRAG-rs processes documents through a multi-stage pipeline:

```
Document → Chunking → Entity Extraction → Graph Building → Vector Index → Query Engine → Answer
```

For detailed architecture information, see [ARCHITECTURE.md](ARCHITECTURE.md).

## API Reference

### Core Types

```rust
// Main GraphRAG interface
pub struct GraphRAG { /* ... */ }

// Document representation
pub struct Document {
    pub id: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
}

// Query results
pub struct QueryResult {
    pub answer: String,
    pub confidence: f32,
    pub sources: Vec<String>,
}
```

### Main Methods

```rust
impl GraphRAG {
    // Create new instance
    pub fn new(config: Config) -> Result<Self>;

    // Add content
    pub fn add_document(&mut self, doc: Document) -> Result<()>;
    pub fn add_text(&mut self, text: &str) -> Result<()>;

    // Query
    pub fn ask(&self, question: &str) -> Result<String>;
    pub fn query(&self, question: &str) -> Result<QueryResult>;

    // Management
    pub fn clear(&mut self);
    pub fn save(&self, path: &str) -> Result<()>;
    pub fn load(&mut self, path: &str) -> Result<()>;
}
```

## Performance Tuning

### Memory Optimization

```toml
[performance]
chunk_size = 500  # Smaller chunks use less memory
max_entities_per_chunk = 10
enable_caching = false
```

### Speed Optimization

```toml
[performance]
enable_parallel = true
num_threads = 8  # Adjust based on CPU cores
batch_size = 50
```

### Accuracy Optimization

```toml
[pipeline]
chunk_overlap = 400  # Higher overlap preserves more context
min_confidence = 0.7
enable_reranking = true
```

## Troubleshooting

### Common Issues

**Build fails with "rust version" error**
```bash
# Update Rust
rustup update
```

**Out of memory error**
```toml
# Reduce chunk size in config.toml
chunk_size = 300
enable_parallel = false
```

**Slow processing**
```toml
# Enable parallel processing
enable_parallel = true
num_threads = 8
```

**Ollama connection error**
```bash
# Ensure Ollama is running
ollama serve

# Check if model is available
ollama list
```

### Debug Mode

```bash
# Enable debug logging
RUST_LOG=debug cargo run --bin simple_cli config.toml

# Enable backtrace for errors
RUST_BACKTRACE=1 cargo run
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-username/graphrag-rs.git
cd graphrag-rs

# Run tests
cargo test

# Run with debug info
RUST_LOG=debug cargo run

# Check code quality
cargo clippy
cargo fmt --check
```

## FAQ

**Q: What file formats are supported?**
A: Currently supports plain text (.txt) and markdown (.md). PDF support is planned.

**Q: Can I use this without Ollama?**
A: Yes, the library includes a mock LLM for testing and can work with embeddings only.

**Q: How much memory does it need?**
A: Typically under 100MB for documents up to 500k characters.

**Q: Is it production ready?**
A: Yes, the core functionality is stable and well-tested.

**Q: Can I use commercial LLMs?**
A: OpenAI support is planned. Currently works with Ollama's local models.

## Roadmap

- [ ] OpenAI API support
- [ ] PDF document support
- [ ] Web UI interface
- [ ] Incremental index updates
- [ ] Distributed processing
- [ ] GPU acceleration for embeddings

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Microsoft GraphRAG for the original concept
- Ollama for local LLM support
- Rust community for excellent libraries

---

**Built with Rust** | [Documentation](https://docs.rs/graphrag-rs) | [Report Issues](https://github.com/your-username/graphrag-rs/issues)

