//! Python bindings for graphrag-core
//!
//! This module provides Python bindings for the GraphRAG system using PyO3.
//! It exposes the main GraphRAG functionality through a `PyGraphRAG` class.

use graphrag_core::{config::Config, GraphRAG};
use pyo3::prelude::*;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Python wrapper for the GraphRAG system
///
/// This class provides a Python interface to the Rust GraphRAG implementation.
/// All async methods are exposed as Python coroutines using pyo3-async-runtimes.
///
/// # Example
///
/// ```python
/// import asyncio
/// from graphrag_py import PyGraphRAG
///
/// async def main():
///     # Create a local instance
///     rag = PyGraphRAG.default_local()
///
///     # Add documents
///     await rag.add_document_from_text("Your document text here")
///
///     # Build the knowledge graph
///     await rag.build_graph()
///
///     # Query the system
///     answer = await rag.ask("What is this document about?")
///     print(f"Answer: {answer}")
///
/// asyncio.run(main())
/// ```
#[pyclass]
struct PyGraphRAG {
    inner: Arc<Mutex<GraphRAG>>,
}

#[pymethods]
impl PyGraphRAG {
    /// Create a new GraphRAG instance with default local configuration
    ///
    /// This configures GraphRAG to use:
    /// - Ollama for LLM interactions (must be running locally)
    /// - In-memory vector storage
    /// - Default chunking and embedding settings
    ///
    /// Returns:
    ///     PyGraphRAG: A new GraphRAG instance
    ///
    /// Raises:
    ///     RuntimeError: If initialization fails
    #[staticmethod]
    fn default_local() -> PyResult<Self> {
        let rag = GraphRAG::default_local()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create GraphRAG: {}", e)))?;

        let inner = Arc::new(Mutex::new(rag));

        // Initialize the GraphRAG system
        {
            let mut guard = inner.blocking_lock();
            guard
                .initialize()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to initialize GraphRAG: {}", e)))?;
        }

        Ok(PyGraphRAG { inner })
    }

    /// Create a new GraphRAG instance with custom configuration
    ///
    /// Args:
    ///     config_path (str): Path to a TOML configuration file
    ///
    /// Returns:
    ///     PyGraphRAG: A new GraphRAG instance
    ///
    /// Raises:
    ///     RuntimeError: If initialization fails
    #[staticmethod]
    fn from_config(config_path: String) -> PyResult<Self> {
        let config = Config::from_file(&config_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to load config: {}", e)))?;

        let mut rag = GraphRAG::new(config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create GraphRAG: {}", e)))?;

        rag.initialize()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to initialize GraphRAG: {}", e)))?;

        Ok(PyGraphRAG {
            inner: Arc::new(Mutex::new(rag)),
        })
    }

    /// Add a document from text content
    ///
    /// This method chunks the text and adds it to the knowledge graph.
    ///
    /// Args:
    ///     text (str): The text content to add
    ///
    /// Returns:
    ///     None (coroutine)
    ///
    /// Raises:
    ///     RuntimeError: If adding the document fails
    fn add_document_from_text<'p>(&self, py: Python<'p>, text: String) -> PyResult<Bound<'p, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = inner.lock().await;
            guard
                .add_document_from_text(&text)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to add document: {}", e)))?;
            Ok(Python::with_gil(|py| py.None()))
        })
    }

    /// Build the knowledge graph from added documents
    ///
    /// This method extracts entities and relationships from all added documents
    /// and builds the knowledge graph. This can take some time for large documents.
    ///
    /// Returns:
    ///     None (coroutine)
    ///
    /// Raises:
    ///     RuntimeError: If graph building fails
    fn build_graph<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = inner.lock().await;
            guard
                .build_graph()
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to build graph: {}", e)))?;
            Ok(Python::with_gil(|py| py.None()))
        })
    }

    /// Clear the knowledge graph while preserving documents
    ///
    /// This removes all extracted entities and relationships but keeps
    /// the documents and chunks. Useful for rebuilding the graph from scratch.
    ///
    /// Returns:
    ///     None
    ///
    /// Raises:
    ///     RuntimeError: If clearing fails
    fn clear_graph(&self) -> PyResult<()> {
        let mut guard = self.inner.blocking_lock();
        guard
            .clear_graph()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to clear graph: {}", e)))?;
        Ok(())
    }

    /// Query the system for relevant information
    ///
    /// This method retrieves relevant context from the knowledge graph and
    /// generates an answer using the configured LLM.
    ///
    /// Args:
    ///     query (str): The question to ask
    ///
    /// Returns:
    ///     str: The generated answer (coroutine)
    ///
    /// Raises:
    ///     RuntimeError: If the query fails
    fn ask<'p>(&self, py: Python<'p>, query: String) -> PyResult<Bound<'p, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = inner.lock().await;
            let answer = guard
                .ask(&query)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Query failed: {}", e)))?;
            Ok(answer)
        })
    }

    /// Query the system with reasoning decomposition
    ///
    /// This method breaks down complex queries into sub-queries, retrieves
    /// context for each, and synthesizes a comprehensive answer.
    ///
    /// Args:
    ///     query (str): The complex question to ask
    ///
    /// Returns:
    ///     str: The generated answer with reasoning (coroutine)
    ///
    /// Raises:
    ///     RuntimeError: If the query fails
    fn ask_with_reasoning<'p>(&self, py: Python<'p>, query: String) -> PyResult<Bound<'p, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = inner.lock().await;
            let answer = guard
                .ask_with_reasoning(&query)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Query with reasoning failed: {}", e)))?;
            Ok(answer)
        })
    }

    /// Check if the system has any documents
    ///
    /// Returns:
    ///     bool: True if documents have been added
    fn has_documents(&self) -> PyResult<bool> {
        let guard = self.inner.blocking_lock();
        Ok(guard.has_documents())
    }

    /// Check if the knowledge graph has been built
    ///
    /// Returns:
    ///     bool: True if the graph has been built
    fn has_graph(&self) -> PyResult<bool> {
        let guard = self.inner.blocking_lock();
        Ok(guard.has_graph())
    }

    /// String representation of the PyGraphRAG instance
    fn __repr__(&self) -> PyResult<String> {
        let guard = self.inner.blocking_lock();
        let has_docs = guard.has_documents();
        let has_graph = guard.has_graph();

        Ok(format!(
            "PyGraphRAG(documents={}, graph_built={})",
            has_docs, has_graph
        ))
    }
}

/// Python module for graphrag_py
///
/// This module exports the PyGraphRAG class for use in Python.
#[pymodule]
fn graphrag_py(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add the PyGraphRAG class
    m.add_class::<PyGraphRAG>()?;

    // Add module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__doc__", "Python bindings for GraphRAG - Graph-based Retrieval Augmented Generation")?;

    Ok(())
}
