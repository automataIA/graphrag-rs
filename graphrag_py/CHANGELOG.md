# Changelog

All notable changes to GraphRAG Python bindings will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive user-friendly README with examples
- Configuration example file (config.example.toml)
- Contributing guidelines (CONTRIBUTING.md)
- Enhanced __init__.py with proper module documentation

### Changed
- Improved README structure with table of contents
- Enhanced documentation with troubleshooting section
- Added performance benchmarks and optimization tips

### Removed
- Placeholder hello() function from __init__.py

## [0.1.0] - 2024-12-30

### Added
- Initial Python bindings for GraphRAG-rs
- PyGraphRAG class with full async/await support
- Core methods:
  - `default_local()` - Create instance with local config
  - `from_config()` - Create from TOML config file
  - `add_document_from_text()` - Add documents
  - `build_graph()` - Build knowledge graph
  - `clear_graph()` - Clear graph
  - `ask()` - Query with basic retrieval
  - `ask_with_reasoning()` - Query with reasoning decomposition
  - `has_documents()` - Check document status
  - `has_graph()` - Check graph status
- Comprehensive test suite with 15+ tests
- Example scripts:
  - basic_usage.py - Quick start example
  - document_qa.py - Advanced Q&A example
- Installation verification script
- Complete documentation:
  - README.md with installation and usage guide
  - IMPLEMENTATION_SUMMARY.md with technical details
- PyO3 0.21 integration with abi3 support (Python 3.9+)
- pyo3-async-runtimes for async support
- Maturin-based build system
- Thread-safe Arc<Mutex> wrapper for GraphRAG

### Dependencies
- pyo3 = "0.21" with extension-module and abi3-py39 features
- pyo3-async-runtimes = "0.21" with tokio-runtime feature
- graphrag-core (workspace dependency)
- tokio = "1" with full features
- tracing = "0.1"
- pytest >= 7.0 (dev)
- pytest-asyncio >= 0.21.0 (dev)
- pytest-cov >= 4.0 (dev)

### Technical Details
- Built with PyO3 for Rust-Python interoperability
- Uses Bound<PyAny> types for modern PyO3 API
- Proper error propagation from Rust to Python
- Cross-platform support (Linux, macOS, Windows)
- Python 3.9+ compatibility via abi3

### Known Limitations
- Requires Ollama running locally for default configuration
- Graph clearing requires full rebuild (incremental updates planned)
- Config loading is synchronous

## Version History

- **0.1.0** (2024-12-30) - Initial release with core functionality

[Unreleased]: https://github.com/automataIA/graphrag-rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/automataIA/graphrag-rs/releases/tag/v0.1.0
