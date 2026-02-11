use crate::core::Result;
use crate::vector::store::{SearchResult, VectorStore};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::RwLock;

/// Type alias for vector storage with metadata
type VectorWithMetadata = (Vec<f32>, HashMap<String, String>);

/// Simple in-memory vector store for testing and default usage
#[derive(Default)]
pub struct MemoryVectorStore {
    vectors: RwLock<HashMap<String, VectorWithMetadata>>,
}

impl MemoryVectorStore {
    /// Create a new in-memory vector store
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl VectorStore for MemoryVectorStore {
    async fn initialize(&self) -> Result<()> {
        Ok(())
    }

    async fn add_vector(
        &self,
        id: &str,
        embedding: Vec<f32>,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        self.vectors
            .write()
            .unwrap()
            .insert(id.to_string(), (embedding, metadata));
        Ok(())
    }

    async fn add_vectors_batch(
        &self,
        vectors: Vec<(&str, Vec<f32>, HashMap<String, String>)>,
    ) -> Result<()> {
        let mut guard = self.vectors.write().unwrap();
        for (id, emb, meta) in vectors {
            guard.insert(id.to_string(), (emb, meta));
        }
        Ok(())
    }

    async fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<SearchResult>> {
        let guard = self.vectors.read().unwrap();
        let mut scored: Vec<SearchResult> = guard
            .iter()
            .map(|(id, (emb, meta))| {
                let score = crate::vector::VectorUtils::cosine_similarity(query_embedding, emb);
                SearchResult {
                    id: id.clone(),
                    score,
                    metadata: meta.clone(),
                }
            })
            .collect();

        // Sort desc
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(scored.into_iter().take(top_k).collect())
    }

    async fn delete(&self, id: &str) -> Result<()> {
        self.vectors.write().unwrap().remove(id);
        Ok(())
    }
}
