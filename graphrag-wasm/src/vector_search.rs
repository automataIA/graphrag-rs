///! Pure Rust vector search implementation using cosine similarity
///! This replaces the JavaScript-based Voy bindings with native WASM code

use std::cmp::Ordering;

/// Simple in-memory vector index for semantic search
#[derive(Clone)]
pub struct VectorIndex {
    embeddings: Vec<Vec<f32>>,
    metadata: Vec<DocumentMetadata>,
}

#[derive(Clone)]
pub struct DocumentMetadata {
    pub id: String,
    pub title: String,
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub id: String,
    pub title: String,
    pub similarity: f64,
    pub distance: f64,
}

impl VectorIndex {
    /// Create a new empty vector index
    pub fn new() -> Self {
        Self {
            embeddings: Vec::new(),
            metadata: Vec::new(),
        }
    }

    /// Create index from embeddings
    pub fn from_embeddings(embeddings: Vec<Vec<f32>>) -> Self {
        let metadata = embeddings
            .iter()
            .enumerate()
            .map(|(i, _)| DocumentMetadata {
                id: i.to_string(),
                title: format!("doc_{}", i),
            })
            .collect();

        Self {
            embeddings,
            metadata,
        }
    }

    /// Add an embedding to the index
    pub fn add(&mut self, embedding: Vec<f32>, id: String, title: String) {
        self.embeddings.push(embedding);
        self.metadata.push(DocumentMetadata { id, title });
    }

    /// Search for k nearest neighbors using cosine similarity
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        if self.embeddings.is_empty() {
            return Vec::new();
        }

        // Calculate similarities for all embeddings
        let mut results: Vec<SearchResult> = self
            .embeddings
            .iter()
            .zip(self.metadata.iter())
            .map(|(embedding, meta)| {
                let similarity = cosine_similarity(query, embedding);
                SearchResult {
                    id: meta.id.clone(),
                    title: meta.title.clone(),
                    similarity,
                    distance: 1.0 - similarity,
                }
            })
            .collect();

        // Sort by similarity (descending)
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(Ordering::Equal)
        });

        // Take top k results
        results.truncate(k);
        results
    }

    /// Get the number of embeddings in the index
    pub fn size(&self) -> usize {
        self.embeddings.len()
    }

    /// Clear all embeddings
    pub fn clear(&mut self) {
        self.embeddings.clear();
        self.metadata.clear();
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..a.len() {
        dot_product += (a[i] * b[i]) as f64;
        norm_a += (a[i] * a[i]) as f64;
        norm_b += (b[i] * b[i]) as f64;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a.sqrt() * norm_b.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![1.0, 0.0, 0.0];
        let d = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&c, &d) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_vector_index() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let index = VectorIndex::from_embeddings(embeddings);
        assert_eq!(index.size(), 3);

        let query = vec![0.9, 0.1, 0.0];
        let results = index.search(&query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "0");
        assert!(results[0].similarity > results[1].similarity);
    }
}
