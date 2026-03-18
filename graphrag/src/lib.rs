//! # graphrag
//!
//! Meta-crate that re-exports [`graphrag_core`] for library users and provides
//! the `graphrag` binary (backed by [`graphrag_cli`]).
//!
//! ## Installation
//!
//! ```bash
//! cargo install graphrag
//! ```
//!
//! This installs the `graphrag` binary which is the full-featured TUI and CLI
//! for GraphRAG operations.
//!
//! ## Library usage
//!
//! ```toml
//! [dependencies]
//! graphrag = "0.1"
//! ```
//!
//! All types from `graphrag-core` are re-exported:
//!
//! ```rust,no_run
//! use graphrag::GraphRAG;
//! ```

pub use graphrag_core::*;
