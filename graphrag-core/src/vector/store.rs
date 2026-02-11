use crate::core::Result;
use async_trait::async_trait;
use std::collections::HashMap;

/// Result of a vector search
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// ID of the vector/document
    pub id: String,
    /// Similarity score (higher is better)
    pub score: f32,
    /// Metadata associated with the vector
    pub metadata: HashMap<String, String>,
}

/// Abstract interface for vector storage backends
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Initialize the collection/table
    async fn initialize(&self) -> Result<()>;

    /// Add a single vector with ID and metadata
    async fn add_vector(
        &self,
        id: &str,
        embedding: Vec<f32>,
        metadata: HashMap<String, String>,
    ) -> Result<()>;

    /// Add multiple vectors in batch
    async fn add_vectors_batch(
        &self,
        vectors: Vec<(&str, Vec<f32>, HashMap<String, String>)>,
    ) -> Result<()>;

    /// Search for nearest neighbors
    async fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<SearchResult>>;

    /// Delete a vector by ID
    async fn delete(&self, id: &str) -> Result<()>;
}
