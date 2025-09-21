//! Storage backend implementations
//!
//! This module provides different storage backends for the GraphRAG system,
//! allowing data to be persisted in memory, files, or databases.

pub mod memory;
pub mod workspace;

// Re-export implementations
pub use memory::MemoryStorage;
pub use workspace::{Checkpoint, Workspace, WorkspaceInfo, WorkspaceManager, WorkspaceStatistics};

// Persistent storage will be added when needed

#[cfg(test)]
mod tests;
