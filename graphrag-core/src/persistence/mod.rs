//! Persistence layer for GraphRAG knowledge graphs
//!
//! This module provides storage backends for persisting knowledge graphs to disk
//! and loading them back into memory for fast querying.
//!
//! ## Supported Formats
//!
//! - **Parquet**: Columnar format for entities, relationships, chunks (Apache Arrow ecosystem)
//! - **LanceDB**: Vector storage for embeddings (Lance columnar format)
//! - **JSON**: Human-readable backup format (already implemented in core)
//! - **GraphML**: Export format for visualization tools (already implemented in core)
//!
//! ## Architecture
//!
//! ```text
//! workspace/
//! ├── default/
//! │   ├── entities.parquet
//! │   ├── relationships.parquet
//! │   ├── chunks.parquet
//! │   ├── documents.parquet
//! │   ├── vectors.lance/
//! │   ├── graph.json
//! │   └── metadata.toml
//! └── project_a/
//!     └── ...
//! ```
//!
//! ## Example
//!
//! ```no_run
//! use graphrag_core::{KnowledgeGraph, persistence::WorkspaceManager};
//!
//! # fn example() -> graphrag_core::Result<()> {
//! // Create workspace manager
//! let workspace = WorkspaceManager::new("./workspace")?;
//!
//! // Save graph to default workspace
//! let graph = KnowledgeGraph::new();
//! workspace.save_graph(&graph, "default")?;
//!
//! // Load graph from workspace
//! let loaded_graph = workspace.load_graph("default")?;
//! # Ok(())
//! # }
//! ```

use crate::core::Result;

// Submodules
pub mod workspace;

#[cfg(feature = "persistent-storage")]
pub mod parquet;

// Lance storage temporarily disabled due to version conflicts
// TODO: Re-enable when lancedb dependencies are resolved
// #[cfg(feature = "lance-storage")]
// pub mod lance;

// Re-exports (workspace always available)
pub use workspace::{WorkspaceManager, WorkspaceMetadata, WorkspaceInfo};

#[cfg(feature = "persistent-storage")]
pub use parquet::{ParquetPersistence, ParquetConfig};

// Lance storage temporarily disabled
// #[cfg(feature = "lance-storage")]
// pub use lance::{LanceVectorStore, LanceConfig};

/// Persistence trait for knowledge graphs
pub trait Persistence {
    /// Save knowledge graph to storage
    fn save(&self, path: &str) -> Result<()>;

    /// Load knowledge graph from storage
    fn load(path: &str) -> Result<Self>
    where
        Self: Sized;

    /// Check if storage exists
    fn exists(path: &str) -> bool;

    /// Get storage size in bytes
    fn size(path: &str) -> Result<u64>;
}
