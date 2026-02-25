//! GraphRAG Server Library
//!
//! Core types and modules for the GraphRAG REST API server.

pub mod distributed_cache;
pub mod embeddings;
pub mod lancedb_store;
pub mod multi_model_embeddings;
pub mod observability;
pub mod qdrant_store;

#[cfg(feature = "auth")]
pub mod auth;

// Re-export common types
pub use qdrant_store::{QdrantError, QdrantStore};

pub use lancedb_store::{LanceDBError, LanceDBStore};

// Re-export shared types (they're identical between stores)
pub use qdrant_store::{DocumentMetadata, Entity, Relationship, SearchResult};

pub use embeddings::{EmbeddingConfig, EmbeddingError, EmbeddingService, EmbeddingStats};

pub use distributed_cache::{CacheConfig, CacheStats, DistributedCache};

pub use observability::{Metrics, Observability, Span, TracingMiddleware};

pub use multi_model_embeddings::{
    CohereProvider, EmbeddingProvider, EmbeddingResult, EmbeddingRouter, ModelConfig,
    ModelRegistry, OpenAIProvider,
};

#[cfg(feature = "auth")]
pub use auth::{AuthState, Claims};
