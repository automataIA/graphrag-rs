//! Simplified APIs for different user experience levels
//!
//! This module provides progressive disclosure of GraphRAG functionality,
//! allowing users to start simple and add complexity as needed.

pub mod easy;
pub mod handlers;
pub mod rest;
pub mod simple;

#[cfg(test)]
mod tests;

// Re-export for convenience
pub use easy::SimpleGraphRAG;
pub use handlers::AppState;
pub use simple::*;
