//! GraphRAG Server Library
//!
//! Core types and modules for the GraphRAG REST API server.

pub mod qdrant_store;
pub mod lancedb_store;
pub mod embeddings;
pub mod distributed_cache;
pub mod observability;
pub mod multi_model_embeddings;

#[cfg(feature = "auth")]
pub mod auth;

// Re-export common types
pub use qdrant_store::{
    QdrantStore, QdrantError,
};

pub use lancedb_store::{
    LanceDBStore, LanceDBError,
};

// Re-export shared types (they're identical between stores)
pub use qdrant_store::{Entity, Relationship, DocumentMetadata, SearchResult};

pub use embeddings::{EmbeddingService, EmbeddingConfig, EmbeddingError, EmbeddingStats};

pub use distributed_cache::{DistributedCache, CacheConfig, CacheStats};

pub use observability::{Observability, Metrics, Span, TracingMiddleware};

pub use multi_model_embeddings::{
    EmbeddingProvider, ModelConfig, EmbeddingResult, EmbeddingRouter,
    ModelRegistry, OpenAIProvider, CohereProvider,
};

#[cfg(feature = "auth")]
pub use auth::{AuthState, Claims};
