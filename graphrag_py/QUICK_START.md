# 🚀 GraphRAG-py Quick Start Guide

Get up and running with GraphRAG-py in 5 minutes!

## Installation

### 1. Install Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the required models
ollama pull llama3
ollama pull nomic-embed-text
```

### 2. Build GraphRAG-py

```bash
cd graphrag_py

# Option A: Using uv (faster)
uv sync
uv run maturin develop

# Option B: Using pip
pip install maturin
maturin develop
```

### 3. Verify Installation

```bash
uv run python verify_installation.py
```

You should see: ✅ All checks passed!

## Your First Program

Create a file called `my_first_rag.py`:

```python
import asyncio
from graphrag_py import PyGraphRAG

async def main():
    # 1. Create GraphRAG instance
    rag = PyGraphRAG.default_local()

    # 2. Add a document
    await rag.add_document_from_text("""
        Rust is a systems programming language that runs blazingly fast,
        prevents segfaults, and guarantees thread safety. It was created
        by Graydon Hoare and first released in 2010.
    """)

    # 3. Build the knowledge graph
    print("Building graph...")
    await rag.build_graph()

    # 4. Ask questions
    answer = await rag.ask("Who created Rust?")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Run it!

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Run your program
uv run python my_first_rag.py
```

## Common Patterns

### Loading Multiple Documents

```python
documents = [
    "Document 1 text...",
    "Document 2 text...",
    "Document 3 text..."
]

for doc in documents:
    await rag.add_document_from_text(doc)

await rag.build_graph()
```

### Loading from Files

```python
from pathlib import Path

for file_path in Path("./docs").glob("*.txt"):
    with open(file_path) as f:
        await rag.add_document_from_text(f.read())

await rag.build_graph()
```

### Complex Queries

```python
# Simple query
answer = await rag.ask("What is X?")

# Complex query with reasoning
answer = await rag.ask_with_reasoning(
    "Compare X and Y. What are the key differences?"
)
```

### Using Custom Configuration

Create `config.toml`:

```toml
[llm]
provider = "ollama"
model = "llama3"
temperature = 0.7

[retrieval]
strategy = "adaptive"
top_k = 5
```

Then use it:

```python
rag = PyGraphRAG.from_config("config.toml")
```

## Troubleshooting

### "Failed to connect to Ollama"
```bash
# Make sure Ollama is running
ollama serve
```

### "Model not found"
```bash
# Pull the models
ollama pull llama3
ollama pull nomic-embed-text
```

### "Import Error"
```bash
# Rebuild the extension
uv run maturin develop
```

### Slow Performance
- Use smaller models: `ollama pull llama3:8b`
- Reduce document size
- Use `--release` mode: `maturin develop --release`

## Next Steps

- 📖 Read the full [README.md](README.md) for detailed documentation
- 💡 Check out [examples/](examples/) for more complex use cases
- 🔧 Customize with [config.example.toml](config.example.toml)
- 🐛 Report issues on [GitHub](https://github.com/automataIA/graphrag-rs/issues)

## API Cheat Sheet

```python
from graphrag_py import PyGraphRAG

# Create instance
rag = PyGraphRAG.default_local()
rag = PyGraphRAG.from_config("config.toml")

# Add documents
await rag.add_document_from_text(text)

# Build graph
await rag.build_graph()

# Query
answer = await rag.ask(query)
answer = await rag.ask_with_reasoning(query)

# Clear graph
rag.clear_graph()

# Check status
rag.has_documents()  # bool
rag.has_graph()      # bool
```

## Complete Example

```python
import asyncio
from pathlib import Path
from graphrag_py import PyGraphRAG

async def knowledge_base():
    # Initialize
    rag = PyGraphRAG.default_local()

    # Load documents
    print("Loading documents...")
    docs_dir = Path("./documents")
    for file in docs_dir.glob("*.txt"):
        print(f"  - {file.name}")
        with open(file) as f:
            await rag.add_document_from_text(f.read())

    # Build graph
    print("Building knowledge graph...")
    await rag.build_graph()
    print(f"✓ Ready! {rag}")

    # Interactive Q&A
    while True:
        question = input("\nAsk a question (or 'quit'): ")
        if question.lower() in ['quit', 'exit', 'q']:
            break

        answer = await rag.ask(question)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    asyncio.run(knowledge_base())
```

---

**Happy RAGing! 🎉**

For more help, see the full [README.md](README.md) or visit our [GitHub repo](https://github.com/automataIA/graphrag-rs).
