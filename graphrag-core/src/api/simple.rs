//! Level 1: Simple one-function API for absolute beginners
//!
//! This module provides the simplest possible interface to GraphRAG,
//! allowing users to get started with a single function call.

use crate::{GraphRAG, Result};

/// Answer a question about a document with one function call
///
/// This is the simplest way to use GraphRAG. Just provide text content
/// and a question, and get back an answer.
///
/// # Examples
///
/// ```rust
/// use graphrag_rs::simple::answer;
///
/// let text = "The quick brown fox jumps over the lazy dog.";
/// let response = answer(text, "What animal jumps?").unwrap();
/// println!("{}", response);
/// ```
pub fn answer(document: &str, question: &str) -> Result<String> {
    GraphRAG::quick_answer(document, question)
}

/// Answer a question about a file with one function call
///
/// # Examples
///
/// ```rust,no_run
/// use graphrag_rs::simple::answer_file;
///
/// let response = answer_file("document.txt", "What is this about?").unwrap();
/// println!("{}", response);
/// ```
pub fn answer_file(file_path: &str, question: &str) -> Result<String> {
    let text = std::fs::read_to_string(file_path)?;
    answer(&text, question)
}