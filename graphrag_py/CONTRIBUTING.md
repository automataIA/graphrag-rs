# Contributing to GraphRAG-py

Thank you for your interest in contributing to GraphRAG-py! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Style Guidelines](#style-guidelines)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/graphrag-rs.git
   cd graphrag-rs/graphrag_py
   ```
3. **Set up the development environment** (see below)

## Development Setup

### Prerequisites

- **Rust** (latest stable): Install from [rustup.rs](https://rustup.rs/)
- **Python 3.9+**: Install from [python.org](https://www.python.org/)
- **uv** (recommended): Install with `pip install uv`
- **Ollama** (for testing): Install from [ollama.ai](https://ollama.ai/)

### Setup Steps

1. **Install dependencies:**
   ```bash
   cd graphrag_py
   uv sync
   ```

2. **Build the extension module:**
   ```bash
   uv run maturin develop
   ```

3. **Verify installation:**
   ```bash
   uv run python verify_installation.py
   ```

4. **Pull required models:**
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

### IDE Setup

#### VS Code
Install recommended extensions:
- rust-analyzer
- Python
- Pylance

#### PyCharm
Enable Rust plugin for syntax highlighting.

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-retrieval-strategy`
- `fix/memory-leak-in-graph-builder`
- `docs/improve-installation-guide`
- `refactor/simplify-error-handling`

### Commit Messages

Follow conventional commits format:
```
type(scope): brief description

Longer explanation if needed

Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(bindings): add support for custom embedding models

fix(query): resolve memory leak in async query handling

docs(readme): add troubleshooting section for Ollama setup
```

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_binding.py -v

# Run specific test class
uv run pytest tests/test_binding.py::TestQuerying -v

# Run with coverage
uv run pytest tests/ --cov=graphrag_py --cov-report=html
```

### Writing Tests

Add tests for new features in `tests/test_binding.py`:

```python
import pytest
from graphrag_py import PyGraphRAG

class TestNewFeature:
    """Test the new feature."""

    @pytest.mark.asyncio
    async def test_feature_works(self):
        """Test that the feature works as expected."""
        rag = PyGraphRAG.default_local()
        # Your test code here
        assert result == expected
```

### Test Requirements

- All new features must have tests
- Tests should be isolated and not depend on external state
- Use `pytest.skip()` for tests requiring Ollama if not available
- Aim for >80% code coverage

## Submitting Changes

### Pull Request Process

1. **Update documentation** if you've changed APIs or added features
2. **Add tests** for any new functionality
3. **Ensure all tests pass:**
   ```bash
   uv run pytest tests/ -v
   ```
4. **Build the project** to check for compilation errors:
   ```bash
   maturin develop --release
   ```
5. **Update the CHANGELOG** (if applicable)
6. **Push to your fork** and submit a pull request

### Pull Request Template

When opening a PR, include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for new features
- [ ] Updated documentation

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed my code
- [ ] Commented complex code sections
- [ ] Updated documentation
- [ ] No new warnings
```

## Style Guidelines

### Rust Code

Follow Rust standard style:
```bash
# Format code
cargo fmt

# Check for issues
cargo clippy
```

**Guidelines:**
- Use descriptive variable names
- Add doc comments for public APIs
- Handle errors properly (no unwrap in production code)
- Keep functions focused and small

**Example:**
```rust
/// Adds a document to the GraphRAG system.
///
/// # Arguments
///
/// * `text` - The document text to add
///
/// # Errors
///
/// Returns an error if document processing fails.
fn add_document_from_text<'p>(&self, py: Python<'p>, text: String) -> PyResult<Bound<'p, PyAny>> {
    // Implementation
}
```

### Python Code

Follow PEP 8 and use type hints:

```python
from typing import List, Optional

async def process_documents(
    documents: List[str],
    max_workers: Optional[int] = None
) -> int:
    """
    Process multiple documents concurrently.

    Args:
        documents: List of document texts to process
        max_workers: Maximum number of concurrent workers

    Returns:
        Number of successfully processed documents

    Raises:
        RuntimeError: If processing fails
    """
    # Implementation
```

**Format code:**
```bash
# Using black
black tests/ examples/

# Using ruff
ruff check tests/ examples/
```

### Documentation

- Use clear, concise language
- Include code examples
- Keep examples up-to-date with API changes
- Use proper markdown formatting

## Reporting Bugs

### Before Submitting

1. **Search existing issues** to avoid duplicates
2. **Check the troubleshooting guide** in README.md
3. **Verify you're using the latest version**

### Bug Report Template

```markdown
## Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., macOS 13.0]
- Python version: [e.g., 3.11]
- GraphRAG-py version: [e.g., 0.1.0]
- Ollama version: [e.g., 0.1.20]

## Additional Context
Any other relevant information
```

## Suggesting Enhancements

### Enhancement Template

```markdown
## Feature Description
Clear description of the proposed feature

## Motivation
Why is this feature needed?

## Proposed Solution
How should it work?

## Alternatives Considered
Other approaches you've considered

## Additional Context
Any other relevant information
```

## Development Tips

### Quick Development Cycle

```bash
# Make changes to Rust code
# Then rebuild and test
maturin develop && uv run pytest tests/ -v
```

### Testing with Examples

```bash
# Start Ollama
ollama serve

# In another terminal
uv run python examples/basic_usage.py
```

### Debugging Rust Code

Add debug prints:
```rust
eprintln!("Debug: value = {:?}", value);
```

Or use a debugger:
```bash
rust-lldb target/debug/deps/graphrag_py-*
```

### Debugging Python Bindings

Enable Rust backtraces:
```bash
RUST_BACKTRACE=1 uv run python your_script.py
```

## Documentation

### Building Documentation

Documentation is primarily in:
- `README.md` - User guide
- `IMPLEMENTATION_SUMMARY.md` - Technical details
- Doc comments in Rust code
- Docstrings in Python examples

### Updating Documentation

When making changes:
1. Update relevant README sections
2. Update API reference if APIs changed
3. Add examples for new features
4. Update IMPLEMENTATION_SUMMARY if architecture changed

## Release Process

(For maintainers)

1. Update version in `Cargo.toml` and `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create and push a version tag
4. Build and publish to PyPI:
   ```bash
   maturin build --release
   maturin publish
   ```

## Questions?

- **GitHub Discussions**: For general questions
- **GitHub Issues**: For bugs and feature requests
- **Code Review**: Tag maintainers in PRs for review

## Recognition

Contributors will be recognized in:
- README.md acknowledgments
- Release notes
- GitHub contributors page

Thank you for contributing to GraphRAG-py! 🎉
