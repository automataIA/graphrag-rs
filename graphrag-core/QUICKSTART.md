# GraphRAG Quick Start Guide

Get up and running with GraphRAG in 5 minutes!

## Prerequisites

- Rust 1.70+ installed
- (Optional) [Ollama](https://ollama.ai) for LLM-powered features

## Step 1: Add Dependency

Add to your `Cargo.toml`:

```toml
[dependencies]
graphrag-core = { version = "0.1", features = ["starter"] }
tokio = { version = "1", features = ["full"] }
```

## Step 2: Hello World (5 Lines)

```rust
use graphrag_core::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Create GraphRAG instance with your document
    let mut graphrag = GraphRAG::quick_start(
        "Albert Einstein was a theoretical physicist who developed the theory of relativity. \
         He was born in Germany in 1879 and later moved to the United States. \
         Einstein received the Nobel Prize in Physics in 1921."
    ).await?;

    // Ask questions
    let answer = graphrag.ask("Where was Einstein born?").await?;
    println!("Answer: {}", answer);

    Ok(())
}
```

**Run it:**
```bash
cargo run
```

## Step 3: Choose Your Configuration Style

### Option A: TypedBuilder (Recommended for Beginners)

Compile-time safety - the code won't compile if you forget required settings!

```rust
use graphrag_core::prelude::*;

let graphrag = TypedBuilder::new()
    .with_output_dir("./my-output")  // Required
    .with_ollama()                    // Required: enables LLM features
    .with_chunk_size(500)             // Optional: customize chunking
    .build_and_init()?;               // Build + initialize in one step
```

### Option B: Configuration File

Create `graphrag.toml`:

```toml
output_dir = "./output"
chunk_size = 1000

[ollama]
enabled = true
host = "localhost"
port = 11434
chat_model = "llama3.2:3b"
```

Load it:

```rust
let config = Config::from_toml_file("graphrag.toml")?;
let mut graphrag = GraphRAG::new(config)?;
graphrag.initialize()?;
```

### Option C: CLI Setup Wizard

```bash
# Install CLI
cargo install --path graphrag-cli

# Run interactive wizard
graphrag-cli setup

# Follow prompts to configure
```

## Step 4: Add Documents

```rust
// From text
graphrag.add_document_from_text("Your document content here")?;

// From file
graphrag.add_document_from_file("path/to/document.txt")?;

// Build the knowledge graph
graphrag.build_graph().await?;
```

## Step 5: Query with Explanations

```rust
// Simple query
let answer = graphrag.ask("What are the main topics?").await?;
println!("{}", answer);

// Query with full explanation
let explained = graphrag.ask_explained("What are the main topics?").await?;
println!("Answer: {}", explained.answer);
println!("Confidence: {:.0}%", explained.confidence * 100.0);

// Show reasoning steps
for step in &explained.reasoning_steps {
    println!("{}. {}", step.step_number, step.description);
}

// Show sources
for source in &explained.sources {
    println!("Source: {} (relevance: {:.0}%)",
        source.id,
        source.relevance_score * 100.0
    );
}
```

## Complete Example

```rust
use graphrag_core::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Create and configure
    let mut graphrag = TypedBuilder::new()
        .with_output_dir("./knowledge-base")
        .with_ollama()
        .with_chunk_size(500)
        .with_top_k(10)
        .build_and_init()?;

    // 2. Add documents
    graphrag.add_document_from_text(
        "GraphRAG is a powerful knowledge graph system that combines \
         graph-based retrieval with RAG (Retrieval-Augmented Generation). \
         It was developed to improve question answering over documents."
    )?;

    // 3. Build knowledge graph
    graphrag.build_graph().await?;

    // 4. Query
    let explained = graphrag.ask_explained("What is GraphRAG?").await?;

    // 5. Display results
    println!("{}", explained.format_display());

    Ok(())
}
```

## Without Ollama (Offline Mode)

If you don't have Ollama installed, use hash embeddings:

```rust
let graphrag = TypedBuilder::new()
    .with_output_dir("./output")
    .with_hash_embeddings()  // No LLM needed!
    .build_and_init()?;
```

Note: Hash embeddings are faster but provide less semantic understanding.

## Environment Variables

Override configuration with environment variables:

```bash
export GRAPHRAG_OLLAMA_HOST=my-server
export GRAPHRAG_OLLAMA_PORT=8080
export GRAPHRAG_CHUNK_SIZE=1000

# Then in code:
let config = Config::load()?;  // Automatically uses env vars
```

## Error Handling

GraphRAG provides helpful error messages:

```rust
match graphrag.ask("question").await {
    Ok(answer) => println!("{}", answer),
    Err(e) => {
        // Get actionable suggestion
        let suggestion = e.suggestion();
        println!("Error: {}", e);
        println!("Suggestion: {}", suggestion.action);

        // Code example (if available)
        if let Some(code) = suggestion.code_example {
            println!("Try:\n{}", code);
        }
    }
}
```

## Feature Bundles

Choose the right feature bundle for your use case:

| Bundle | Use Case | Command |
|--------|----------|---------|
| `starter` | Getting started, prototyping | `features = ["starter"]` |
| `full` | Production applications | `features = ["full"]` |
| `wasm-bundle` | Browser/WASM deployment | `features = ["wasm-bundle"]` |
| `research` | Experimental features | `features = ["research"]` |

## Sectoral Templates

Use pre-configured templates for specific domains:

```bash
# Via CLI
graphrag-cli setup --template legal

# Or load directly
let config = Config::from_toml_file("templates/medical.toml")?;
```

Available templates:
- `general.toml` - News, articles, mixed content
- `legal.toml` - Contracts, regulations
- `medical.toml` - Clinical notes, patient records
- `financial.toml` - Reports, SEC filings
- `technical.toml` - API docs, code documentation

## Understanding the Pipeline

When you call `build_graph()`, GraphRAG executes these phases:

```
Document → Chunking → Entity Extraction → Relationship Extraction → Graph
                            │                       │
                            ▼                       ▼
                    ┌───────────────┐      ┌───────────────┐
                    │ Algorithmic   │      │ Co-occurrence │  (fast, offline)
                    │ Semantic      │      │ LLM-based     │  (accurate, needs Ollama)
                    │ Hybrid        │      │ Gleaning      │  (iterative refinement)
                    └───────────────┘      └───────────────┘
```

When you call `ask()`, these phases execute:

```
Query → Embedding → Search (Vector/BM25/PageRank) → LLM Answer Generation
```

**Key insight:** Embeddings are generated **on-demand during queries**, not during graph construction.

### Choosing Your Approach

| Need | Approach | Config |
|------|----------|--------|
| Fast, offline | `algorithmic` | `approach = "algorithmic"` |
| Best accuracy | `semantic` | `approach = "semantic"` + Ollama |
| Balanced | `hybrid` | `approach = "hybrid"` |

## Troubleshooting

### "Ollama not running"

```bash
# Start Ollama
ollama serve

# Pull required models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### "Configuration not found"

```rust
// Use defaults instead
let config = Config::default();

// Or use TypedBuilder
let graphrag = TypedBuilder::new()
    .with_output_dir("./output")
    .with_hash_embeddings()  // Works without config file
    .build()?;
```

### "Not initialized"

```rust
// Make sure to initialize before querying
graphrag.initialize()?;

// Or use build_and_init() which does both
let graphrag = TypedBuilder::new()
    .with_output_dir("./output")
    .with_ollama()
    .build_and_init()?;  // <-- Initializes automatically
```

## Next Steps

- Read the [full README](README.md) for detailed documentation
- Understand [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) for pipeline phases
- Check out [templates/README.md](templates/README.md) for domain-specific configs
- Explore [EMBEDDINGS_CONFIG.md](EMBEDDINGS_CONFIG.md) for embedding options
- See [OLLAMA_INTEGRATION.md](OLLAMA_INTEGRATION.md) for LLM setup

## Getting Help

- GitHub Issues: https://github.com/anthropics/graphrag-rs/issues
- Documentation: https://docs.rs/graphrag-core
