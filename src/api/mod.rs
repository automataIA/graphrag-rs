//! Simplified APIs for different user experience levels
//!
//! This module provides progressive disclosure of GraphRAG functionality,
//! allowing users to start simple and add complexity as needed.

pub mod simple;
pub mod easy;

#[cfg(test)]
mod tests;

// Re-export for convenience
pub use simple::*;
pub use easy::SimpleGraphRAG;