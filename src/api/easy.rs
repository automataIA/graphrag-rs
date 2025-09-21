//! Level 2: Easy stateful API for basic usage
//!
//! This module provides a simplified stateful interface that hides
//! most of the complexity while allowing multiple queries on the same document.

use crate::{GraphRAG, Result};
use std::path::Path;

/// Simple wrapper around GraphRAG for easy usage
///
/// This provides a stateful interface where you can load content once
/// and ask multiple questions about it.
///
/// # Examples
///
/// ```rust
/// use graphrag_rs::easy::SimpleGraphRAG;
///
/// let mut graph = SimpleGraphRAG::from_text("The quick brown fox...")?;
/// let answer1 = graph.ask("What animal is mentioned?")?;
/// let answer2 = graph.ask("What color is the fox?")?;
/// # Ok::<(), graphrag_rs::GraphRAGError>(())
/// ```
pub struct SimpleGraphRAG {
    inner: GraphRAG,
}

impl SimpleGraphRAG {
    /// Create a new SimpleGraphRAG instance from text content
    pub fn from_text(text: &str) -> Result<Self> {
        let inner = GraphRAG::from_text(text)?;
        Ok(Self { inner })
    }

    /// Create a new SimpleGraphRAG instance from a file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = GraphRAG::from_file(path)?;
        Ok(Self { inner })
    }

    /// Ask a question and get an answer
    pub fn ask(&mut self, question: &str) -> Result<String> {
        self.inner.ask(question)
    }

    /// Add more text content to analyze
    pub fn add_text(&mut self, text: &str) -> Result<()> {
        self.inner.add_document_from_text(text)
    }

    /// Add a file to analyze
    pub fn add_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let text = std::fs::read_to_string(path)?;
        self.add_text(&text)
    }

    /// Check if the system is ready to answer questions
    pub fn is_ready(&self) -> bool {
        self.inner.is_initialized() && self.inner.has_documents()
    }
}