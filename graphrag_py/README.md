# 🚀 GraphRAG-py: Python Bindings for GraphRAG

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Powered by Rust](https://img.shields.io/badge/powered%20by-Rust-orange.svg)](https://www.rust-lang.org/)

**GraphRAG-py** brings the power of Rust-based graph retrieval augmented generation to Python. Build intelligent question-answering systems that leverage knowledge graphs for more accurate and contextual responses.

## 📋 Table of Contents

- [What is GraphRAG?](#what-is-graphrag)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## 🤔 What is GraphRAG?

**GraphRAG** (Graph-based Retrieval Augmented Generation) combines the power of:
- **Knowledge Graphs**: Automatically extracts entities and relationships from your documents
- **Vector Embeddings**: Semantic search for relevant information
- **LLM Integration**: Generates natural language answers using retrieved context

Think of it as **RAG on steroids** - instead of just retrieving similar text chunks, GraphRAG understands the relationships between concepts in your documents.

### Why Use GraphRAG?

✅ **Better Context Understanding**: Captures relationships between entities
✅ **More Accurate Answers**: Retrieves interconnected information, not just similar text
✅ **Explainable Results**: See which entities and relationships led to the answer
✅ **Complex Queries**: Handles multi-hop reasoning across documents

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🚀 **Blazing Fast** | Rust-powered performance with Python convenience |
| 🔗 **Knowledge Graph** | Automatic entity extraction and relationship mapping |
| 🤖 **LLM Integration** | Built-in support for Ollama, OpenAI, and more |
| 🎯 **Smart Retrieval** | Multiple strategies (semantic, keyword, hybrid, adaptive) |
| ⚡ **Async-First** | Full async/await support for non-blocking operations |
| 📦 **Zero Config** | Works out of the box with sensible defaults |
| 🔄 **Flexible** | Use local models or cloud APIs |
| 🧠 **Reasoning** | Advanced query decomposition for complex questions |

## 📦 Installation

### Prerequisites

1. **Python 3.9 or higher**
   ```bash
   python --version  # Should be 3.9+
   ```

2. **Rust toolchain** (for building from source)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

3. **Ollama** (for local LLM support - recommended for getting started)
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.ai/install.sh | sh

   # Then pull a model
   ollama pull llama3
   ```

### Install GraphRAG-py

#### Option 1: Using uv (Recommended - Fast!)

```bash
cd graphrag_py
uv sync
uv run maturin develop
```

#### Option 2: Using pip

```bash
cd graphrag_py
pip install maturin
maturin develop
```

#### Option 3: Install in release mode (faster runtime)

```bash
maturin develop --release
```

### Verify Installation

```bash
uv run python verify_installation.py
```

You should see all checks pass:
```
✅ PASS: Import
✅ PASS: Class Availability
✅ PASS: Instantiation
✅ PASS: Methods
✅ PASS: Async Support
```

## 🚀 Quick Start

### Your First GraphRAG Application

```python
import asyncio
from graphrag_py import PyGraphRAG

async def main():
    # Step 1: Create a GraphRAG instance
    print("Initializing GraphRAG...")
    rag = PyGraphRAG.default_local()

    # Step 2: Add your documents
    print("Adding documents...")
    await rag.add_document_from_text("""
        Python is a high-level programming language created by Guido van Rossum.
        It was first released in 1991 and is known for its simple, readable syntax.
        Python is widely used in web development, data science, artificial intelligence,
        and automation. Popular frameworks include Django, Flask, and FastAPI.
    """)

    # Step 3: Build the knowledge graph
    print("Building knowledge graph (this may take a moment)...")
    await rag.build_graph()

    # Step 4: Ask questions!
    print("\n" + "="*60)
    questions = [
        "Who created Python?",
        "When was Python first released?",
        "What is Python used for?",
    ]

    for question in questions:
        answer = await rag.ask(question)
        print(f"Q: {question}")
        print(f"A: {answer}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

**Run it:**
```bash
# Make sure Ollama is running
ollama serve

# In another terminal
uv run python my_first_rag.py
```

## 🎓 Core Concepts

### 1. Documents and Chunks

When you add a document, GraphRAG:
1. **Chunks** the text into manageable pieces
2. **Embeds** each chunk as a vector
3. **Stores** chunks for retrieval

```python
# Add a single document
await rag.add_document_from_text("Your text here...")

# Add multiple documents
documents = [
    "Document 1 content...",
    "Document 2 content...",
    "Document 3 content...",
]
for doc in documents:
    await rag.add_document_from_text(doc)
```

### 2. Knowledge Graph Building

The knowledge graph extracts:
- **Entities**: People, places, organizations, concepts
- **Relationships**: How entities are connected
- **Properties**: Attributes of entities

```python
# Build graph from all added documents
await rag.build_graph()

# Check status
print(f"Has documents: {rag.has_documents()}")
print(f"Has graph: {rag.has_graph()}")
```

### 3. Querying

Two query modes available:

**Basic Query** - Fast, single-pass retrieval:
```python
answer = await rag.ask("What is Python?")
```

**Reasoning Query** - Multi-step decomposition for complex questions:
```python
answer = await rag.ask_with_reasoning(
    "Compare Python and Rust in terms of performance and use cases"
)
```

### 4. Graph Management

```python
# Clear graph but keep documents (useful for rebuilding)
rag.clear_graph()

# Rebuild from scratch
await rag.build_graph()
```

## 📚 API Reference

### Class: `PyGraphRAG`

#### Initialization

##### `PyGraphRAG.default_local()` → `PyGraphRAG`
Creates a GraphRAG instance with default local configuration.

**Uses:**
- Ollama for LLM (must be running on `localhost:11434`)
- In-memory vector storage
- Default chunking (500 tokens per chunk)

```python
rag = PyGraphRAG.default_local()
```

**Raises:** `RuntimeError` if initialization fails

---

##### `PyGraphRAG.from_config(config_path: str)` → `PyGraphRAG`
Creates a GraphRAG instance from a TOML configuration file.

**Parameters:**
- `config_path` (str): Path to TOML configuration file

```python
rag = PyGraphRAG.from_config("config.toml")
```

**Raises:** `RuntimeError` if config loading or initialization fails

---

#### Document Management

##### `async add_document_from_text(text: str)` → `None`
Adds a document to the system. The document is chunked and embedded automatically.

**Parameters:**
- `text` (str): The document content

```python
await rag.add_document_from_text("Your document content here...")
```

**Raises:** `RuntimeError` if document addition fails

---

#### Graph Operations

##### `async build_graph()` → `None`
Builds the knowledge graph by extracting entities and relationships from all documents.

**Note:** This can take time depending on document size and LLM speed.

```python
await rag.build_graph()
```

**Raises:** `RuntimeError` if graph building fails

---

##### `clear_graph()` → `None`
Clears the knowledge graph while preserving documents and chunks.

```python
rag.clear_graph()
```

**Raises:** `RuntimeError` if clearing fails

---

#### Querying

##### `async ask(query: str)` → `str`
Asks a question and returns an answer based on retrieved context.

**Parameters:**
- `query` (str): The question to ask

**Returns:** `str` - The generated answer

```python
answer = await rag.ask("What is machine learning?")
print(answer)
```

**Raises:** `RuntimeError` if query fails

---

##### `async ask_with_reasoning(query: str)` → `str`
Asks a complex question using reasoning decomposition.

**How it works:**
1. Decomposes query into sub-questions
2. Retrieves context for each sub-question
3. Synthesizes a comprehensive answer

**Parameters:**
- `query` (str): The complex question

**Returns:** `str` - The generated answer with reasoning

```python
answer = await rag.ask_with_reasoning(
    "How do machine learning and deep learning differ, and what are their applications?"
)
print(answer)
```

**Raises:** `RuntimeError` if query fails

---

#### State Checking

##### `has_documents()` → `bool`
Checks if any documents have been added.

```python
if rag.has_documents():
    print("Documents are loaded!")
```

---

##### `has_graph()` → `bool`
Checks if the knowledge graph has been built.

```python
if not rag.has_graph():
    await rag.build_graph()
```

---

## 💡 Examples

### Example 1: Building a Technical Documentation Q&A

```python
import asyncio
from graphrag_py import PyGraphRAG

async def main():
    rag = PyGraphRAG.default_local()

    # Add technical documentation
    docs = [
        """
        FastAPI is a modern web framework for building APIs with Python.
        It's based on standard Python type hints and is very fast.
        FastAPI uses Pydantic for data validation and Starlette for web routing.
        """,
        """
        Pydantic is a data validation library using Python type annotations.
        It provides runtime type checking and automatic data parsing.
        Pydantic models can be used for configuration management and API schemas.
        """,
        """
        Starlette is a lightweight ASGI framework/toolkit for building async web services.
        It includes support for WebSocket, GraphQL, and in-process background tasks.
        FastAPI is built on top of Starlette.
        """
    ]

    for doc in docs:
        await rag.add_document_from_text(doc)

    await rag.build_graph()

    # Interactive Q&A
    questions = [
        "What is FastAPI?",
        "What technologies does FastAPI use?",
        "What is Pydantic used for?",
        "How are FastAPI and Starlette related?",
    ]

    for q in questions:
        answer = await rag.ask(q)
        print(f"\nQ: {q}")
        print(f"A: {answer}")

asyncio.run(main())
```

### Example 2: Research Paper Analysis

```python
import asyncio
from graphrag_py import PyGraphRAG

async def analyze_papers():
    rag = PyGraphRAG.default_local()

    # Add research paper abstracts
    papers = {
        "Attention Is All You Need": """
        The Transformer architecture introduced the self-attention mechanism,
        eliminating recurrence and convolutions entirely. The model achieves
        state-of-the-art results on machine translation tasks while being
        more parallelizable and requiring less time to train.
        """,
        "BERT": """
        BERT (Bidirectional Encoder Representations from Transformers) is
        designed to pre-train deep bidirectional representations by jointly
        conditioning on both left and right context in all layers. BERT can be
        fine-tuned with just one additional output layer for various NLP tasks.
        """,
        "GPT-3": """
        GPT-3 is an autoregressive language model with 175 billion parameters.
        It demonstrates strong performance on many NLP tasks without fine-tuning,
        using few-shot learning with task demonstrations provided as context.
        """
    }

    for title, abstract in papers.items():
        await rag.add_document_from_text(f"{title}: {abstract}")

    await rag.build_graph()

    # Complex comparative analysis
    analysis = await rag.ask_with_reasoning(
        "Compare the approaches of Transformers, BERT, and GPT-3. "
        "What are their key innovations and differences?"
    )

    print("Research Paper Analysis:")
    print("="*70)
    print(analysis)

asyncio.run(analyze_papers())
```

### Example 3: Multi-Document Corporate Knowledge Base

```python
import asyncio
from graphrag_py import PyGraphRAG

async def build_knowledge_base():
    rag = PyGraphRAG.default_local()

    # Simulate loading from multiple sources
    company_docs = [
        # HR policies
        """
        Employee onboarding process:
        1. Complete I-9 and tax forms
        2. Attend orientation on first Monday
        3. Meet with HR for benefits enrollment
        4. Receive laptop and security badge
        5. Complete mandatory training modules
        """,
        # Engineering guidelines
        """
        Code review process:
        All code must be reviewed by at least two senior engineers.
        PRs must pass all CI/CD checks before merging.
        Documentation must be updated for any API changes.
        """,
        # Product information
        """
        Our flagship product, CloudSync Pro, is an enterprise file synchronization
        platform. It supports real-time collaboration, end-to-end encryption,
        and integrates with major cloud providers (AWS, Azure, GCP).
        """
    ]

    print("Loading company knowledge base...")
    for doc in company_docs:
        await rag.add_document_from_text(doc)

    print("Building knowledge graph...")
    await rag.build_graph()

    print("\nKnowledge base ready! Ask me anything:")
    print("-" * 60)

    # Simulate employee questions
    questions = [
        "What is the onboarding process?",
        "What are the code review requirements?",
        "Tell me about CloudSync Pro",
        "What cloud providers does our product support?",
    ]

    for q in questions:
        answer = await rag.ask(q)
        print(f"\nQ: {q}")
        print(f"A: {answer}")

asyncio.run(build_knowledge_base())
```

### Example 4: Loading Documents from Files

```python
import asyncio
from pathlib import Path
from graphrag_py import PyGraphRAG

async def load_from_files():
    rag = PyGraphRAG.default_local()

    # Load all text files from a directory
    docs_dir = Path("./documents")

    for file_path in docs_dir.glob("*.txt"):
        print(f"Loading {file_path.name}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            await rag.add_document_from_text(content)

    print("\nBuilding knowledge graph...")
    await rag.build_graph()

    # Now query across all documents
    answer = await rag.ask("Summarize the main topics covered in these documents")
    print(f"\nSummary: {answer}")

asyncio.run(load_from_files())
```

## ⚙️ Configuration

### Custom Configuration File

Create a `config.toml` file:

```toml
[llm]
provider = "ollama"
model = "llama3"
base_url = "http://localhost:11434"
temperature = 0.7
max_tokens = 2048

[embeddings]
provider = "ollama"
model = "nomic-embed-text"
base_url = "http://localhost:11434"

[chunking]
strategy = "semantic"
chunk_size = 512
chunk_overlap = 50

[retrieval]
strategy = "adaptive"
top_k = 5
similarity_threshold = 0.7

[vector_store]
provider = "memory"  # Options: memory, qdrant, lancedb
```

Then use it:

```python
rag = PyGraphRAG.from_config("config.toml")
```

### Environment Variables

Set environment variables for API keys:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## 🔧 Troubleshooting

### Common Issues

#### 1. "Failed to connect to Ollama"

**Problem:** Ollama is not running or not accessible.

**Solution:**
```bash
# Start Ollama
ollama serve

# In another terminal, verify it's running
curl http://localhost:11434/api/tags
```

#### 2. "Model not found"

**Problem:** The LLM model hasn't been downloaded.

**Solution:**
```bash
# Pull the default model
ollama pull llama3

# Or pull the embedding model
ollama pull nomic-embed-text
```

#### 3. "Import Error: No module named 'graphrag_py'"

**Problem:** The extension module wasn't built.

**Solution:**
```bash
cd graphrag_py
uv run maturin develop
```

#### 4. Graph building is very slow

**Problem:** LLM inference is the bottleneck.

**Solutions:**
- Use a smaller model (e.g., `llama3:8b` instead of `llama3:70b`)
- Use a GPU-accelerated setup
- Process fewer documents at a time
- Consider using cloud APIs (OpenAI, Anthropic) for faster responses

#### 5. Out of memory errors

**Problem:** Too many documents or large documents.

**Solutions:**
- Reduce chunk size in configuration
- Process documents in batches
- Use a vector store with disk persistence (Qdrant, LanceDB)
- Increase system memory or use a machine with more RAM

### Debug Mode

Enable detailed logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Now run your GraphRAG code
rag = PyGraphRAG.default_local()
# ... rest of your code
```

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/automataIA/graphrag-rs/issues)
- **Discussions**: [GitHub Discussions](https://github.com/automataIA/graphrag-rs/discussions)
- **Documentation**: See `IMPLEMENTATION_SUMMARY.md` for technical details

## 🚄 Performance

### Benchmarks

Typical performance on a modern laptop (M1 Mac, 16GB RAM):

| Operation | Time | Notes |
|-----------|------|-------|
| Initialize instance | < 100ms | Creating empty GraphRAG |
| Add document (1KB) | < 50ms | Per document |
| Build graph (10 docs) | 10-30s | Depends on LLM speed |
| Query (simple) | 200-500ms | With pre-built graph |
| Query (reasoning) | 1-3s | Multiple LLM calls |

### Optimization Tips

1. **Use Release Mode** for production:
   ```bash
   maturin develop --release
   ```

2. **Batch Document Addition**:
   ```python
   # Add multiple documents before building graph
   for doc in documents:
       await rag.add_document_from_text(doc)

   # Build graph once
   await rag.build_graph()
   ```

3. **Reuse Instances**: Creating a new instance is expensive; reuse when possible.

4. **Adjust Chunk Size**: Smaller chunks = faster but less context:
   ```toml
   [chunking]
   chunk_size = 256  # Default is 512
   ```

5. **Use Faster Models**: Smaller models are faster but less accurate.

## 🛠️ Development

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=graphrag_py --cov-report=html

# Run specific test
uv run pytest tests/test_binding.py::TestQuerying::test_ask_basic -v
```

### Running Examples

```bash
# Make sure Ollama is running first
ollama serve

# Run basic example
uv run python examples/basic_usage.py

# Run document Q&A example
uv run python examples/document_qa.py
```

### Building for Distribution

```bash
# Build wheel
maturin build --release

# Build and publish to PyPI
maturin publish
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [PyO3](https://pyo3.rs/) for Rust-Python interoperability
- Powered by [Maturin](https://www.maturin.rs/) for building
- Uses [Ollama](https://ollama.ai/) for local LLM support
- Inspired by Microsoft's GraphRAG research

## 📞 Support

- **Documentation**: You're reading it! 📖
- **Examples**: Check the `examples/` directory
- **Issues**: [Report bugs](https://github.com/automataIA/graphrag-rs/issues)
- **Discussions**: [Ask questions](https://github.com/automataIA/graphrag-rs/discussions)

---

**Made with ❤️ using Rust and Python**

*GraphRAG-py brings enterprise-grade graph RAG capabilities to Python developers.*
