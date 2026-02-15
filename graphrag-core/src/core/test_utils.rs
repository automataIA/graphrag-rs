//! Test utilities and mock implementations for testing
//!
//! This module provides mock implementations of core traits for unit testing
//! without requiring real services or external dependencies.

use crate::core::error::{GraphRAGError, Result};
use crate::core::traits::*;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Mock embedder for testing
#[derive(Clone)]
pub struct MockEmbedder {
    dimension: usize,
    embeddings: Arc<Mutex<HashMap<String, Vec<f32>>>>,
}

impl MockEmbedder {
    /// Create a new mock embedder with the given dimension
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            embeddings: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Pre-populate with known embeddings for testing
    pub fn with_embedding(self, text: impl Into<String>, embedding: Vec<f32>) -> Self {
        self.embeddings
            .lock()
            .unwrap()
            .insert(text.into(), embedding);
        self
    }

    /// Generate a deterministic embedding based on text hash
    fn generate_embedding(&self, text: &str) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();

        // Generate deterministic but different values for each dimension
        (0..self.dimension)
            .map(|i| {
                let seed = hash.wrapping_add(i as u64);
                (seed % 1000) as f32 / 1000.0
            })
            .collect()
    }
}

#[async_trait]
impl AsyncEmbedder for MockEmbedder {
    type Error = GraphRAGError;

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Check if we have a pre-populated embedding
        if let Some(embedding) = self.embeddings.lock().unwrap().get(text) {
            return Ok(embedding.clone());
        }

        // Otherwise generate one
        Ok(self.generate_embedding(text))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn is_ready(&self) -> bool {
        true
    }
}

/// Mock language model for testing
#[derive(Clone)]
pub struct MockLanguageModel {
    responses: Arc<Mutex<HashMap<String, String>>>,
    default_response: String,
}

impl MockLanguageModel {
    /// Create a new mock language model
    pub fn new() -> Self {
        Self {
            responses: Arc::new(Mutex::new(HashMap::new())),
            default_response: "Mock response".to_string(),
        }
    }

    /// Set a specific response for a prompt
    pub fn with_response(self, prompt: impl Into<String>, response: impl Into<String>) -> Self {
        self.responses
            .lock()
            .unwrap()
            .insert(prompt.into(), response.into());
        self
    }

    /// Set the default response for unmatched prompts
    pub fn with_default_response(mut self, response: impl Into<String>) -> Self {
        self.default_response = response.into();
        self
    }
}

impl Default for MockLanguageModel {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AsyncLanguageModel for MockLanguageModel {
    type Error = GraphRAGError;

    async fn complete(&self, prompt: &str) -> Result<String> {
        if let Some(response) = self.responses.lock().unwrap().get(prompt) {
            Ok(response.clone())
        } else {
            Ok(self.default_response.clone())
        }
    }

    async fn complete_with_params(&self, prompt: &str, _params: GenerationParams) -> Result<String> {
        self.complete(prompt).await
    }

    async fn is_available(&self) -> bool {
        true
    }

    async fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "mock-model".to_string(),
            version: Some("1.0.0".to_string()),
            max_context_length: Some(4096),
            supports_streaming: false,
        }
    }

    async fn get_usage_stats(&self) -> Result<ModelUsageStats> {
        Ok(ModelUsageStats {
            total_requests: 0,
            total_tokens_processed: 0,
            average_response_time_ms: 0.0,
            error_rate: 0.0,
        })
    }
}

/// Mock vector store for testing
pub struct MockVectorStore {
    vectors: Arc<Mutex<HashMap<String, Vec<f32>>>>,
    dimension: usize,
}

impl MockVectorStore {
    /// Create a new mock vector store
    pub fn new(dimension: usize) -> Self {
        Self {
            vectors: Arc::new(Mutex::new(HashMap::new())),
            dimension,
        }
    }

    /// Pre-populate with vectors for testing
    pub fn with_vector(self, id: impl Into<String>, vector: Vec<f32>) -> Self {
        self.vectors.lock().unwrap().insert(id.into(), vector);
        self
    }

    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            0.0
        } else {
            dot / (mag_a * mag_b)
        }
    }
}

#[async_trait]
impl AsyncVectorStore for MockVectorStore {
    type Error = GraphRAGError;

    async fn add_vector(&mut self, id: String, vector: Vec<f32>, _metadata: VectorMetadata) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(GraphRAGError::Embedding {
                message: format!(
                    "Vector dimension mismatch: expected {}, got {}",
                    self.dimension,
                    vector.len()
                ),
            });
        }
        self.vectors.lock().unwrap().insert(id, vector);
        Ok(())
    }

    async fn add_vectors_batch(&mut self, vectors: VectorBatch) -> Result<()> {
        for (id, vector, metadata) in vectors {
            self.add_vector(id, vector, metadata).await?;
        }
        Ok(())
    }

    async fn search(&self, query_vector: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query_vector.len() != self.dimension {
            return Err(GraphRAGError::Embedding {
                message: format!(
                    "Query vector dimension mismatch: expected {}, got {}",
                    self.dimension,
                    query_vector.len()
                ),
            });
        }

        let vectors = self.vectors.lock().unwrap();
        let mut results: Vec<_> = vectors
            .iter()
            .map(|(id, vector)| {
                let similarity = Self::cosine_similarity(query_vector, vector);
                SearchResult {
                    id: id.clone(),
                    distance: 1.0 - similarity, // Convert similarity to distance
                    metadata: None,
                }
            })
            .collect();

        // Sort by distance (ascending)
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        // Take top k
        Ok(results.into_iter().take(k).collect())
    }

    async fn search_with_threshold(
        &self,
        query_vector: &[f32],
        k: usize,
        threshold: f32,
    ) -> Result<Vec<SearchResult>> {
        let results = self.search(query_vector, k).await?;
        Ok(results
            .into_iter()
            .filter(|r| r.distance <= threshold)
            .collect())
    }

    async fn remove_vector(&mut self, id: &str) -> Result<bool> {
        Ok(self.vectors.lock().unwrap().remove(id).is_some())
    }

    async fn len(&self) -> usize {
        self.vectors.lock().unwrap().len()
    }
}

/// Mock retriever for testing
pub struct MockRetriever {
    results: Arc<Mutex<Vec<String>>>,
}

impl MockRetriever {
    /// Create a new mock retriever
    pub fn new() -> Self {
        Self {
            results: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Pre-populate with results for testing
    pub fn with_results(self, results: Vec<String>) -> Self {
        *self.results.lock().unwrap() = results;
        self
    }
}

impl Default for MockRetriever {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AsyncRetriever for MockRetriever {
    type Query = String;
    type Result = String;
    type Error = GraphRAGError;

    async fn search(&self, _query: Self::Query, k: usize) -> Result<Vec<Self::Result>> {
        let results = self.results.lock().unwrap();
        Ok(results.iter().take(k).cloned().collect())
    }

    async fn search_with_context(
        &self,
        query: Self::Query,
        _context: &str,
        k: usize,
    ) -> Result<Vec<Self::Result>> {
        self.search(query, k).await
    }

    async fn update(&mut self, content: Vec<String>) -> Result<()> {
        *self.results.lock().unwrap() = content;
        Ok(())
    }

    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_embedder() {
        let embedder = MockEmbedder::new(128)
            .with_embedding("test", vec![0.5; 128]);

        let result = embedder.embed("test").await.unwrap();
        assert_eq!(result.len(), 128);
        assert_eq!(result[0], 0.5);

        // Test unknown text gets generated embedding
        let result2 = embedder.embed("unknown").await.unwrap();
        assert_eq!(result2.len(), 128);
    }

    #[tokio::test]
    async fn test_mock_language_model() {
        let llm = MockLanguageModel::new()
            .with_response("Hello", "Hi there!")
            .with_default_response("Default response");

        assert_eq!(llm.complete("Hello").await.unwrap(), "Hi there!");
        assert_eq!(llm.complete("Unknown").await.unwrap(), "Default response");
    }

    #[tokio::test]
    async fn test_mock_vector_store() {
        let mut store = MockVectorStore::new(3)
            .with_vector("vec1", vec![1.0, 0.0, 0.0])
            .with_vector("vec2", vec![0.0, 1.0, 0.0]);

        assert_eq!(store.len().await, 2);

        let results = store.search(&[1.0, 0.0, 0.0], 2).await.unwrap();
        assert_eq!(results[0].id, "vec1");

        assert!(store.remove_vector("vec1").await.unwrap());
        assert_eq!(store.len().await, 1);
    }

    #[tokio::test]
    async fn test_mock_retriever() {
        let retriever = MockRetriever::new()
            .with_results(vec!["result1".to_string(), "result2".to_string(), "result3".to_string()]);

        let results = retriever.search("query".to_string(), 2).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], "result1");
    }
}
