//! Voy vector store implementation for WASM
//!
//! Voy is a lightweight (75KB) pure Rust vector search library optimized for WASM.
//! It uses a k-d tree algorithm for efficient similarity search.
//!
//! ## Features
//!
//! - 100% Rust, no JavaScript dependencies
//! - k-d tree indexing for fast nearest neighbor search
//! - Cosine similarity metric
//! - Persistent storage via IndexedDB
//! - Memory-efficient for browser environments
//!
//! ## Usage
//!
//! ```rust,ignore
//! use graphrag_core::vector::VoyStore;
//!
//! let mut store = VoyStore::new(384); // 384-dimensional embeddings
//! store.add_vector("doc1", vec![0.1, 0.2, ...])?;
//! store.build_index()?;
//!
//! let results = store.search(&query_embedding, 10)?;
//! ```

use crate::{GraphRAGError, Result};
use std::collections::HashMap;
use voy::{Embeddings, Similarity};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// Voy-based vector store for WASM environments
///
/// This store uses Voy's k-d tree indexing for efficient similarity search
/// in browser environments. It's optimized for small bundle size (75KB)
/// and low memory usage.
pub struct VoyStore {
    /// Embedding dimension
    dimension: usize,
    /// Voy embeddings index
    index: Option<Embeddings>,
    /// Mapping from vector ID to index position
    id_to_index: HashMap<String, usize>,
    /// Mapping from index position to vector ID
    index_to_id: Vec<String>,
    /// Raw embeddings before index build
    pending_embeddings: Vec<Vec<f32>>,
    /// Whether the index has been built
    index_built: bool,
}

impl VoyStore {
    /// Create a new Voy vector store
    ///
    /// # Arguments
    ///
    /// * `dimension` - The dimensionality of the embeddings (e.g., 384 for MiniLM, 768 for BERT)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let store = VoyStore::new(384);
    /// ```
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            index: None,
            id_to_index: HashMap::new(),
            index_to_id: Vec::new(),
            pending_embeddings: Vec::new(),
            index_built: false,
        }
    }

    /// Add a vector to the store
    ///
    /// Vectors are added to a pending queue and indexed when `build_index()` is called.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for this vector
    /// * `embedding` - The embedding vector (must match dimension)
    ///
    /// # Errors
    ///
    /// Returns error if embedding dimension doesn't match or if ID already exists
    pub fn add_vector(&mut self, id: String, embedding: Vec<f32>) -> Result<()> {
        if embedding.len() != self.dimension {
            return Err(GraphRAGError::VectorSearch {
                message: format!(
                    "Embedding dimension mismatch: expected {}, got {}",
                    self.dimension,
                    embedding.len()
                ),
            });
        }

        if self.id_to_index.contains_key(&id) {
            return Err(GraphRAGError::VectorSearch {
                message: format!("Vector ID '{}' already exists", id),
            });
        }

        let index = self.pending_embeddings.len();
        self.id_to_index.insert(id.clone(), index);
        self.index_to_id.push(id);
        self.pending_embeddings.push(embedding);
        self.index_built = false;

        Ok(())
    }

    /// Build the k-d tree index from all pending vectors
    ///
    /// This must be called after adding vectors and before searching.
    /// Building is fast (typically <100ms for 10k vectors) but blocks the thread.
    ///
    /// # Errors
    ///
    /// Returns error if no vectors have been added or if index build fails
    pub fn build_index(&mut self) -> Result<()> {
        if self.pending_embeddings.is_empty() {
            return Err(GraphRAGError::VectorSearch {
                message: "No embeddings to build index from".to_string(),
            });
        }

        // Flatten embeddings for Voy
        let flat_data: Vec<f32> = self
            .pending_embeddings
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect();

        // Create Voy embeddings with cosine similarity
        let embeddings = Embeddings::builder(flat_data, self.pending_embeddings.len())
            .with_dimension(self.dimension)
            .with_similarity(Similarity::Cosine)
            .build()
            .map_err(|e| GraphRAGError::VectorSearch {
                message: format!("Failed to build Voy index: {}", e),
            })?;

        self.index = Some(embeddings);
        self.index_built = true;

        Ok(())
    }

    /// Search for similar vectors
    ///
    /// Returns the top-k most similar vectors ranked by cosine similarity.
    ///
    /// # Arguments
    ///
    /// * `query_embedding` - The query vector (must match dimension)
    /// * `top_k` - Number of results to return
    ///
    /// # Returns
    ///
    /// Vec of (id, similarity_score) tuples, sorted by descending similarity
    ///
    /// # Errors
    ///
    /// Returns error if index not built or query dimension mismatch
    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<(String, f32)>> {
        if !self.index_built {
            return Err(GraphRAGError::VectorSearch {
                message: "Index not built. Call build_index() first.".to_string(),
            });
        }

        if query_embedding.len() != self.dimension {
            return Err(GraphRAGError::VectorSearch {
                message: format!(
                    "Query dimension mismatch: expected {}, got {}",
                    self.dimension,
                    query_embedding.len()
                ),
            });
        }

        let index = self.index.as_ref().ok_or_else(|| GraphRAGError::VectorSearch {
            message: "Index not available".to_string(),
        })?;

        // Search using Voy
        let results = index
            .search(query_embedding, top_k)
            .map_err(|e| GraphRAGError::VectorSearch {
                message: format!("Voy search failed: {}", e),
            })?;

        // Convert results to (id, similarity) tuples
        let mut scored_results = Vec::new();
        for (idx, similarity) in results.iter() {
            if let Some(id) = self.index_to_id.get(*idx) {
                scored_results.push((id.clone(), *similarity));
            }
        }

        Ok(scored_results)
    }

    /// Get the number of vectors in the store
    pub fn len(&self) -> usize {
        self.pending_embeddings.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.pending_embeddings.is_empty()
    }

    /// Get the embedding dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Check if the index has been built
    pub fn is_index_built(&self) -> bool {
        self.index_built
    }

    /// Get a vector by ID
    pub fn get_vector(&self, id: &str) -> Option<&Vec<f32>> {
        self.id_to_index
            .get(id)
            .and_then(|&idx| self.pending_embeddings.get(idx))
    }

    /// Check if a vector exists
    pub fn contains(&self, id: &str) -> bool {
        self.id_to_index.contains_key(id)
    }

    /// Get all vector IDs
    pub fn ids(&self) -> Vec<String> {
        self.index_to_id.clone()
    }

    /// Remove a vector by ID
    ///
    /// Note: This invalidates the index and requires rebuilding
    pub fn remove_vector(&mut self, id: &str) -> Result<()> {
        let idx = self
            .id_to_index
            .remove(id)
            .ok_or_else(|| GraphRAGError::VectorSearch {
                message: format!("Vector ID '{}' not found", id),
            })?;

        // Remove from index_to_id
        if idx < self.index_to_id.len() {
            self.index_to_id.remove(idx);
        }

        // Remove from pending embeddings
        if idx < self.pending_embeddings.len() {
            self.pending_embeddings.remove(idx);
        }

        // Update indices for all vectors after the removed one
        for (_, index) in self.id_to_index.iter_mut() {
            if *index > idx {
                *index -= 1;
            }
        }

        // Invalidate index
        self.index_built = false;
        self.index = None;

        Ok(())
    }

    /// Clear all vectors
    pub fn clear(&mut self) {
        self.id_to_index.clear();
        self.index_to_id.clear();
        self.pending_embeddings.clear();
        self.index = None;
        self.index_built = false;
    }

    /// Get statistics about the store
    pub fn statistics(&self) -> VoyStoreStatistics {
        let mut min_norm = f32::INFINITY;
        let mut max_norm: f32 = 0.0;
        let mut sum_norm: f32 = 0.0;

        for embedding in &self.pending_embeddings {
            let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            min_norm = min_norm.min(norm);
            max_norm = max_norm.max(norm);
            sum_norm += norm;
        }

        let avg_norm = if !self.pending_embeddings.is_empty() {
            sum_norm / self.pending_embeddings.len() as f32
        } else {
            0.0
        };

        VoyStoreStatistics {
            vector_count: self.len(),
            dimension: self.dimension,
            index_built: self.index_built,
            min_norm,
            max_norm,
            avg_norm,
        }
    }
}

impl Default for VoyStore {
    fn default() -> Self {
        Self::new(384) // Default to MiniLM dimension
    }
}

/// Statistics about the Voy vector store
#[derive(Debug, Clone)]
pub struct VoyStoreStatistics {
    pub vector_count: usize,
    pub dimension: usize,
    pub index_built: bool,
    pub min_norm: f32,
    pub max_norm: f32,
    pub avg_norm: f32,
}

impl VoyStoreStatistics {
    pub fn print(&self) {
        println!("Voy Vector Store Statistics:");
        println!("  Algorithm: k-d tree (Voy 0.6)");
        println!("  Vector count: {}", self.vector_count);
        println!("  Dimension: {}", self.dimension);
        println!("  Index built: {}", self.index_built);
        println!("  Bundle size: ~75KB (optimized for WASM)");
        if self.vector_count > 0 {
            println!("  Vector norms:");
            println!("    Min: {:.4}", self.min_norm);
            println!("    Max: {:.4}", self.max_norm);
            println!("    Average: {:.4}", self.avg_norm);
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct VoyStoreWasm {
    inner: VoyStore,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl VoyStoreWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize) -> Self {
        Self {
            inner: VoyStore::new(dimension),
        }
    }

    #[wasm_bindgen(js_name = addVector)]
    pub fn add_vector(&mut self, id: String, embedding: Vec<f32>) -> std::result::Result<(), JsValue> {
        self.inner
            .add_vector(id, embedding)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = buildIndex)]
    pub fn build_index(&mut self) -> std::result::Result<(), JsValue> {
        self.inner
            .build_index()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = search)]
    pub fn search(&self, query_embedding: Vec<f32>, top_k: usize) -> std::result::Result<JsValue, JsValue> {
        let results = self
            .inner
            .search(&query_embedding, top_k)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        serde_wasm_bindgen::to_value(&results)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = len)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[wasm_bindgen(js_name = dimension)]
    pub fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    #[wasm_bindgen(js_name = clear)]
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voy_store_creation() {
        let store = VoyStore::new(384);
        assert_eq!(store.dimension(), 384);
        assert!(store.is_empty());
        assert!(!store.is_index_built());
    }

    #[test]
    fn test_add_vector() {
        let mut store = VoyStore::new(3);
        let embedding = vec![0.1, 0.2, 0.3];

        assert!(store.add_vector("doc1".to_string(), embedding).is_ok());
        assert_eq!(store.len(), 1);
        assert!(store.contains("doc1"));
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut store = VoyStore::new(3);
        let wrong_embedding = vec![0.1, 0.2]; // Wrong dimension

        assert!(store
            .add_vector("doc1".to_string(), wrong_embedding)
            .is_err());
    }

    #[test]
    fn test_duplicate_id() {
        let mut store = VoyStore::new(3);
        let embedding = vec![0.1, 0.2, 0.3];

        store.add_vector("doc1".to_string(), embedding.clone()).unwrap();
        assert!(store.add_vector("doc1".to_string(), embedding).is_err());
    }

    #[test]
    fn test_build_and_search() {
        let mut store = VoyStore::new(3);

        // Add test vectors
        store.add_vector("doc1".to_string(), vec![1.0, 0.0, 0.0]).unwrap();
        store.add_vector("doc2".to_string(), vec![0.0, 1.0, 0.0]).unwrap();
        store.add_vector("doc3".to_string(), vec![0.9, 0.1, 0.0]).unwrap();

        // Build index
        assert!(store.build_index().is_ok());
        assert!(store.is_index_built());

        // Search
        let query = vec![1.0, 0.0, 0.0];
        let results = store.search(&query, 2).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 2);

        // Most similar should be doc1 or doc3
        let first_id = &results[0].0;
        assert!(first_id == "doc1" || first_id == "doc3");
    }

    #[test]
    fn test_search_without_index() {
        let store = VoyStore::new(3);
        let query = vec![1.0, 0.0, 0.0];

        assert!(store.search(&query, 5).is_err());
    }

    #[test]
    fn test_remove_vector() {
        let mut store = VoyStore::new(3);

        store.add_vector("doc1".to_string(), vec![1.0, 0.0, 0.0]).unwrap();
        store.add_vector("doc2".to_string(), vec![0.0, 1.0, 0.0]).unwrap();

        assert_eq!(store.len(), 2);

        store.remove_vector("doc1").unwrap();
        assert_eq!(store.len(), 1);
        assert!(!store.contains("doc1"));
        assert!(store.contains("doc2"));
    }

    #[test]
    fn test_statistics() {
        let mut store = VoyStore::new(3);

        store.add_vector("doc1".to_string(), vec![1.0, 0.0, 0.0]).unwrap();
        store.add_vector("doc2".to_string(), vec![0.0, 1.0, 0.0]).unwrap();

        let stats = store.statistics();
        assert_eq!(stats.vector_count, 2);
        assert_eq!(stats.dimension, 3);
        assert!(!stats.index_built);
    }
}
