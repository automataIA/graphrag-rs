#[cfg(feature = "parallel-processing")]
use crate::parallel::ParallelProcessor;
use crate::{GraphRAGError, Result};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

#[cfg(feature = "vector-hnsw")]
use instant_distance::{Builder, Point, Search};

// Vector store abstractions

/// In-memory vector store implementation for testing and development
pub mod memory_store;

/// Vector store trait and common types
pub mod store;

#[cfg(feature = "lancedb")]
/// LanceDB vector store implementation
pub mod lancedb;

#[cfg(feature = "qdrant")]
/// Qdrant vector store implementation
pub mod qdrant;

// Note: Voy vector store (WASM-optimized, 75KB) has been moved to graphrag-wasm crate
// where it belongs alongside other WASM-specific features.
// See: graphrag-wasm/src/vector/voy_store.rs

/// Wrapper for Vec<f32> to implement Point trait for vector operations
#[derive(Debug, Clone, PartialEq)]
pub struct Vector(Vec<f32>);

impl Vector {
    /// Create a new vector from raw data
    pub fn new(vector_data: Vec<f32>) -> Self {
        Self(vector_data)
    }

    /// Get the vector data as a slice
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }
}

#[cfg(feature = "vector-hnsw")]
impl Point for Vector {
    fn distance(&self, other: &Self) -> f32 {
        // Euclidean distance
        if self.0.len() != other.0.len() {
            return f32::INFINITY;
        }

        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

/// Vector index for semantic search
pub struct VectorIndex {
    #[cfg(feature = "vector-hnsw")]
    index: Option<instant_distance::HnswMap<Vector, String>>,
    #[cfg(not(feature = "vector-hnsw"))]
    index: Option<()>, // Placeholder when HNSW is not available
    embeddings: HashMap<String, Vec<f32>>,
    #[cfg(feature = "parallel-processing")]
    parallel_processor: Option<ParallelProcessor>,
}

impl VectorIndex {
    /// Create a new vector index
    pub fn new() -> Self {
        Self {
            index: None,
            embeddings: HashMap::new(),
            #[cfg(feature = "parallel-processing")]
            parallel_processor: None,
        }
    }

    /// Create a new vector index with parallel processing support
    #[cfg(feature = "parallel-processing")]
    pub fn with_parallel_processing(parallel_processor: ParallelProcessor) -> Self {
        Self {
            index: None,
            embeddings: HashMap::new(),
            parallel_processor: Some(parallel_processor),
        }
    }

    /// Add a vector to the index
    pub fn add_vector(&mut self, id: String, embedding: Vec<f32>) -> Result<()> {
        if embedding.is_empty() {
            return Err(GraphRAGError::VectorSearch {
                message: "Empty embedding vector".to_string(),
            });
        }

        self.embeddings.insert(id, embedding);
        Ok(())
    }

    /// Build the index from all added vectors
    pub fn build_index(&mut self) -> Result<()> {
        if self.embeddings.is_empty() {
            return Err(GraphRAGError::VectorSearch {
                message: "No embeddings to build index from".to_string(),
            });
        }

        #[cfg(feature = "vector-hnsw")]
        {
            let points: Vec<Vector> = self
                .embeddings
                .values()
                .map(|v| Vector::new(v.clone()))
                .collect();

            let values: Vec<String> = self.embeddings.keys().cloned().collect();

            let builder = Builder::default();
            let index = builder.build(points, values);

            self.index = Some(index);
        }

        #[cfg(not(feature = "vector-hnsw"))]
        {
            println!(
                "Warning: HNSW vector indexing not available. Install with --features vector-hnsw"
            );
            self.index = Some(());
        }

        Ok(())
    }

    /// Search for similar vectors
    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<(String, f32)>> {
        let _index = self
            .index
            .as_ref()
            .ok_or_else(|| GraphRAGError::VectorSearch {
                message: "Index not built. Call build_index() first.".to_string(),
            })?;

        #[cfg(feature = "vector-hnsw")]
        {
            let query_point = Vector::new(query_embedding.to_vec());
            let mut search = Search::default();

            let results = _index.search(&query_point, &mut search);

            let mut scored_results = Vec::new();
            for item in results.into_iter().take(top_k) {
                let distance = item.distance;
                // Convert distance to similarity using exponential decay for better score distribution
                let similarity = (-distance).exp().clamp(0.0, 1.0);
                scored_results.push((item.value.clone(), similarity));
            }

            Ok(scored_results)
        }

        #[cfg(not(feature = "vector-hnsw"))]
        {
            // Fallback to brute force similarity search
            let query_vec = query_embedding;
            let mut scored_results = Vec::new();

            for (id, embedding) in &self.embeddings {
                let similarity = self.cosine_similarity(query_vec, embedding);
                scored_results.push((id.clone(), similarity));
            }

            // Sort by similarity (highest first) and take top_k
            scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scored_results.truncate(top_k);

            Ok(scored_results)
        }
    }

    /// Calculate cosine similarity between two vectors (fallback when HNSW is not available)
    #[cfg(not(feature = "vector-hnsw"))]
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Get the number of vectors in the index
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Get embedding dimension (assuming all embeddings have the same dimension)
    pub fn dimension(&self) -> Option<usize> {
        self.embeddings.values().next().map(|v| v.len())
    }

    /// Remove a vector from the index
    pub fn remove_vector(&mut self, id: &str) -> Result<()> {
        self.embeddings.remove(id);
        // Note: instant-distance doesn't support removal, so we need to rebuild
        if !self.embeddings.is_empty() {
            self.build_index()?;
        } else {
            self.index = None;
        }
        Ok(())
    }

    /// Get all vector IDs
    pub fn get_ids(&self) -> Vec<String> {
        self.embeddings.keys().cloned().collect()
    }

    /// Check if a vector exists
    pub fn contains(&self, id: &str) -> bool {
        self.embeddings.contains_key(id)
    }

    /// Get embedding by ID
    pub fn get_embedding(&self, id: &str) -> Option<&Vec<f32>> {
        self.embeddings.get(id)
    }

    /// Batch add multiple vectors in parallel with proper synchronization
    pub fn batch_add_vectors(&mut self, vectors: Vec<(String, Vec<f32>)>) -> Result<()> {
        #[cfg(feature = "parallel-processing")]
        if let Some(processor) = self.parallel_processor.clone() {
            return self.batch_add_vectors_parallel(vectors, &processor);
        }

        // Sequential fallback
        for (id, embedding) in vectors {
            self.add_vector(id, embedding)?;
        }
        Ok(())
    }

    /// Parallel batch vector addition with conflict detection and chunked processing
    #[cfg(feature = "parallel-processing")]
    fn batch_add_vectors_parallel(
        &mut self,
        vectors: Vec<(String, Vec<f32>)>,
        processor: &ParallelProcessor,
    ) -> Result<()> {
        if !processor.should_use_parallel(vectors.len()) {
            // Use sequential processing for small batches
            for (id, embedding) in vectors {
                self.add_vector(id, embedding)?;
            }
            return Ok(());
        }

        #[cfg(feature = "parallel-processing")]
        {
            use rayon::prelude::*;
            use std::collections::HashMap;

            // Pre-validate all vectors in parallel
            let validation_results: std::result::Result<Vec<_>, crate::GraphRAGError> = vectors
                .par_iter()
                .map(|(id, embedding)| {
                    if embedding.is_empty() {
                        Err(crate::GraphRAGError::VectorSearch {
                            message: format!("Empty embedding vector for ID: {id}"),
                        })
                    } else {
                        Ok((id.clone(), embedding.clone()))
                    }
                })
                .collect();

            let validated_vectors = match validation_results {
                Ok(vectors) => vectors,
                Err(e) => {
                    eprintln!("Vector validation failed: {e}");
                    // Fall back to sequential processing with validation
                    for (id, embedding) in vectors {
                        self.add_vector(id, embedding)?;
                    }
                    return Ok(());
                },
            };

            // Check for duplicate IDs and resolve conflicts
            let mut unique_vectors = HashMap::new();
            for (id, embedding) in validated_vectors {
                if unique_vectors.contains_key(&id) {
                    eprintln!("Warning: Duplicate vector ID '{id}' - using latest");
                }
                unique_vectors.insert(id, embedding);
            }

            // Convert to vector pairs for sequential insertion
            let vector_pairs: Vec<_> = unique_vectors.into_iter().collect();

            // Vector pairs are already validated and deduplicated

            // Apply the validated vectors to the embeddings map sequentially
            for (id, embedding) in vector_pairs {
                self.embeddings.insert(id, embedding);
            }

            println!("Added {} vectors in parallel batch", vectors.len());
        }

        #[cfg(not(feature = "parallel-processing"))]
        {
            // Sequential fallback when parallel processing is not available
            for (id, embedding) in vectors {
                self.add_vector(id, embedding)?;
            }
        }

        Ok(())
    }

    /// Batch search for multiple queries in parallel
    pub fn batch_search(
        &self,
        queries: &[Vec<f32>],
        top_k: usize,
    ) -> Result<Vec<Vec<(String, f32)>>> {
        #[cfg(feature = "parallel-processing")]
        {
            if let Some(processor) = &self.parallel_processor {
                if processor.should_use_parallel(queries.len()) {
                    use rayon::prelude::*;
                    return queries
                        .par_iter()
                        .map(|query| self.search(query, top_k))
                        .collect();
                }
            }
        }

        // Sequential fallback
        queries
            .iter()
            .map(|query| self.search(query, top_k))
            .collect()
    }

    /// Parallel similarity computation between all vectors with optimized chunking
    pub fn compute_all_similarities(&self) -> HashMap<(String, String), f32> {
        #[cfg(feature = "parallel-processing")]
        if let Some(processor) = &self.parallel_processor {
            return self.compute_similarities_parallel(processor);
        }

        // Sequential fallback
        self.compute_similarities_sequential()
    }

    /// Parallel similarity computation with work-stealing and memory optimization
    #[cfg(feature = "parallel-processing")]
    fn compute_similarities_parallel(
        &self,
        processor: &ParallelProcessor,
    ) -> HashMap<(String, String), f32> {
        let ids: Vec<String> = self.embeddings.keys().cloned().collect();
        let total_pairs = (ids.len() * (ids.len() - 1)) / 2;

        if !processor.should_use_parallel(total_pairs) {
            return self.compute_similarities_sequential();
        }

        #[cfg(feature = "parallel-processing")]
        {
            use rayon::prelude::*;

            // Pre-collect embeddings for efficient parallel access
            let embedding_vec: Vec<(String, Vec<f32>)> = ids
                .iter()
                .filter_map(|id| self.embeddings.get(id).map(|emb| (id.clone(), emb.clone())))
                .collect();

            if embedding_vec.len() < 2 {
                return HashMap::new();
            }

            // Generate pairs for parallel processing
            let mut pairs = Vec::new();
            for i in 0..embedding_vec.len() {
                for j in (i + 1)..embedding_vec.len() {
                    pairs.push((i, j));
                }
            }

            // Parallel similarity computation with chunked processing
            let chunk_size = processor.config().chunk_batch_size.min(pairs.len());
            let similarities: HashMap<(String, String), f32> = pairs
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut local_similarities = HashMap::new();

                    for &(i, j) in chunk {
                        let (first_id, first_emb) = &embedding_vec[i];
                        let (second_id, second_emb) = &embedding_vec[j];

                        let similarity = VectorUtils::cosine_similarity(first_emb, second_emb);

                        // Only store similarities above a threshold to save memory
                        if similarity > 0.1 {
                            local_similarities
                                .insert((first_id.clone(), second_id.clone()), similarity);
                        }
                    }

                    local_similarities
                })
                .reduce(HashMap::new, |mut acc, chunk_similarities| {
                    acc.extend(chunk_similarities);
                    acc
                });

            println!(
                "Computed {} similarities from {} vectors in parallel",
                similarities.len(),
                embedding_vec.len()
            );

            similarities
        }

        #[cfg(not(feature = "parallel-processing"))]
        {
            self.compute_similarities_sequential()
        }
    }

    /// Sequential similarity computation (fallback)
    fn compute_similarities_sequential(&self) -> HashMap<(String, String), f32> {
        let ids: Vec<String> = self.embeddings.keys().cloned().collect();
        let mut similarities = HashMap::new();

        for (i, id1) in ids.iter().enumerate() {
            if let Some(emb1) = self.embeddings.get(id1) {
                for id2 in ids.iter().skip(i + 1) {
                    if let Some(emb2) = self.embeddings.get(id2) {
                        let sim = VectorUtils::cosine_similarity(emb1, emb2);
                        // Only store similarities above a threshold to save memory
                        if sim > 0.1 {
                            similarities.insert((id1.clone(), id2.clone()), sim);
                        }
                    }
                }
            }
        }

        similarities
    }

    /// Find vectors within a similarity threshold
    pub fn find_similar(
        &self,
        query_embedding: &[f32],
        threshold: f32,
    ) -> Result<Vec<(String, f32)>> {
        let results = self.search(query_embedding, self.len())?;
        Ok(results
            .into_iter()
            .filter(|(_, similarity)| *similarity >= threshold)
            .collect())
    }

    /// Calculate statistics about the index
    pub fn statistics(&self) -> VectorIndexStatistics {
        let dimension = self.dimension().unwrap_or(0);
        let vector_count = self.len();

        // Calculate basic statistics
        let mut min_norm = f32::INFINITY;
        let mut max_norm: f32 = 0.0;
        let mut sum_norm = 0.0;

        for embedding in self.embeddings.values() {
            let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            min_norm = min_norm.min(norm);
            max_norm = max_norm.max(norm);
            sum_norm += norm;
        }

        let avg_norm = if vector_count > 0 {
            sum_norm / vector_count as f32
        } else {
            0.0
        };

        VectorIndexStatistics {
            vector_count,
            dimension,
            min_norm,
            max_norm,
            avg_norm,
            index_built: self.index.is_some(),
        }
    }
}

impl Default for VectorIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the vector index
#[derive(Debug)]
pub struct VectorIndexStatistics {
    /// Total number of vectors in the index
    pub vector_count: usize,
    /// Dimensionality of vectors
    pub dimension: usize,
    /// Minimum vector norm
    pub min_norm: f32,
    /// Maximum vector norm
    pub max_norm: f32,
    /// Average vector norm
    pub avg_norm: f32,
    /// Whether the index has been built
    pub index_built: bool,
}

impl VectorIndexStatistics {
    /// Print statistics
    pub fn print(&self) {
        println!("Vector Index Statistics:");
        println!("  Vector count: {}", self.vector_count);
        println!("  Dimension: {}", self.dimension);
        println!("  Index built: {}", self.index_built);
        if self.vector_count > 0 {
            println!("  Vector norms:");
            println!("    Min: {:.4}", self.min_norm);
            println!("    Max: {:.4}", self.max_norm);
            println!("    Average: {:.4}", self.avg_norm);
        }
    }
}

/// Utility functions for vector operations
pub struct VectorUtils;

/// Simple embedding generator using hash-based approach for consistent vectors
pub struct EmbeddingGenerator {
    dimension: usize,
    word_vectors: HashMap<String, Vec<f32>>,
}

impl EmbeddingGenerator {
    /// Create a new embedding generator with specified dimension
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            word_vectors: HashMap::new(),
        }
    }

    /// Create a new embedding generator with parallel processing support
    #[cfg(feature = "parallel-processing")]
    pub fn with_parallel_processing(
        dimension: usize,
        _parallel_processor: ParallelProcessor,
    ) -> Self {
        Self {
            dimension,
            word_vectors: HashMap::new(),
        }
    }

    /// Generate embedding for a text string
    pub fn generate_embedding(&mut self, text: &str) -> Vec<f32> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return vec![0.0; self.dimension];
        }

        // Get or create word vectors
        let mut word_embeddings = Vec::new();
        for word in &words {
            let normalized_word = word.to_lowercase();
            if !self.word_vectors.contains_key(&normalized_word) {
                self.word_vectors.insert(
                    normalized_word.clone(),
                    self.generate_word_vector(&normalized_word),
                );
            }
            word_embeddings.push(self.word_vectors[&normalized_word].clone());
        }

        // Average the word vectors
        let mut result = vec![0.0; self.dimension];
        for word_vec in word_embeddings {
            for (i, value) in word_vec.iter().enumerate() {
                result[i] += value;
            }
        }

        // Normalize by number of words
        let word_count = words.len() as f32;
        for value in &mut result {
            *value /= word_count;
        }

        // Normalize to unit vector
        VectorUtils::normalize(&mut result);
        result
    }

    /// Generate a consistent vector for a word using hash-based approach
    fn generate_word_vector(&self, word: &str) -> Vec<f32> {
        let mut vector = Vec::with_capacity(self.dimension);

        // Use multiple hash seeds for better distribution
        for i in 0..self.dimension {
            let mut hasher = DefaultHasher::new();
            word.hash(&mut hasher);
            i.hash(&mut hasher);

            let hash = hasher.finish();
            // Convert hash to float in range [-1, 1]
            let value = ((hash % 2000) as f32 - 1000.0) / 1000.0;
            vector.push(value);
        }

        // Normalize to unit vector for better similarity properties
        VectorUtils::normalize(&mut vector);
        vector
    }

    /// Generate embeddings for multiple texts in batch with parallel processing
    pub fn batch_generate(&mut self, texts: &[&str]) -> Vec<Vec<f32>> {
        // Use sequential approach to avoid borrowing issues
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.generate_embedding(text));
        }
        results
    }

    /// Parallel batch generation with chunking for very large datasets
    pub fn batch_generate_chunked(&mut self, texts: &[&str], chunk_size: usize) -> Vec<Vec<f32>> {
        if texts.len() <= chunk_size {
            return self.batch_generate(texts);
        }

        #[cfg(feature = "parallel-processing")]
        {
            use rayon::prelude::*;

            // Process in chunks to manage memory usage
            let results: Vec<Vec<f32>> = texts
                .par_chunks(chunk_size)
                .map(|chunk| {
                    // Each chunk is processed with its own generator state
                    let mut local_generator = EmbeddingGenerator::new(self.dimension);
                    local_generator.word_vectors = self.word_vectors.clone(); // Share cached words

                    chunk
                        .iter()
                        .map(|&text| local_generator.generate_embedding(text))
                        .collect::<Vec<_>>()
                })
                .flatten()
                .collect();

            // Update the main generator's word cache with new words from parallel processing
            // Note: This is a simplified approach - in a more sophisticated implementation,
            // we would merge the word caches from all parallel workers

            println!(
                "Generated {} embeddings in parallel chunks of size {}",
                texts.len(),
                chunk_size
            );

            results
        }

        #[cfg(not(feature = "parallel-processing"))]
        {
            // Sequential chunked processing when parallel is not available
            let mut results = Vec::with_capacity(texts.len());

            for chunk in texts.chunks(chunk_size) {
                for &text in chunk {
                    results.push(self.generate_embedding(text));
                }
            }

            results
        }
    }

    /// Get the embedding dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the number of cached word vectors
    pub fn cached_words(&self) -> usize {
        self.word_vectors.len()
    }

    /// Clear the word vector cache
    pub fn clear_cache(&mut self) {
        self.word_vectors.clear();
    }
}

impl Default for EmbeddingGenerator {
    fn default() -> Self {
        Self::new(128) // Default to 128-dimensional embeddings
    }
}

impl VectorUtils {
    /// Calculate cosine similarity between two vectors
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Calculate Euclidean distance between two vectors
    pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::INFINITY;
        }

        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Normalize a vector to unit length
    pub fn normalize(vector: &mut [f32]) {
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in vector {
                *x /= norm;
            }
        }
    }

    /// Generate a random vector (for testing)
    pub fn random_vector(dimension: usize) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut vector = Vec::with_capacity(dimension);
        let mut hasher = DefaultHasher::new();

        for i in 0..dimension {
            i.hash(&mut hasher);
            let hash = hasher.finish();
            let value = ((hash % 1000) as f32 - 500.0) / 1000.0; // Range [-0.5, 0.5]
            vector.push(value);
        }

        vector
    }

    /// Calculate the centroid of multiple vectors
    pub fn centroid(vectors: &[Vec<f32>]) -> Option<Vec<f32>> {
        if vectors.is_empty() {
            return None;
        }

        let dimension = vectors[0].len();
        if !vectors.iter().all(|v| v.len() == dimension) {
            return None; // All vectors must have the same dimension
        }

        let mut centroid = vec![0.0; dimension];
        for vector in vectors {
            for (i, &value) in vector.iter().enumerate() {
                centroid[i] += value;
            }
        }

        let count = vectors.len() as f32;
        for value in &mut centroid {
            *value /= count;
        }

        Some(centroid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_index_creation() {
        let mut index = VectorIndex::new();
        assert!(index.is_empty());

        let embedding = vec![0.1, 0.2, 0.3];
        index.add_vector("test".to_string(), embedding).unwrap();

        assert!(!index.is_empty());
        assert_eq!(index.len(), 1);
        assert_eq!(index.dimension(), Some(3));
    }

    #[test]
    fn test_vector_search() {
        let mut index = VectorIndex::new();

        // Add some test vectors
        index
            .add_vector("doc1".to_string(), vec![1.0, 0.0, 0.0])
            .unwrap();
        index
            .add_vector("doc2".to_string(), vec![0.0, 1.0, 0.0])
            .unwrap();
        index
            .add_vector("doc3".to_string(), vec![0.8, 0.2, 0.0])
            .unwrap();

        index.build_index().unwrap();

        // Search for similar vectors
        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 2).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 2);

        // First result should be most similar
        assert_eq!(results[0].0, "doc1");
    }

    #[test]
    fn test_cosine_similarity() {
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![1.0, 0.0, 0.0];
        let vec3 = vec![0.0, 1.0, 0.0];

        assert!((VectorUtils::cosine_similarity(&vec1, &vec2) - 1.0).abs() < 0.001);
        assert!((VectorUtils::cosine_similarity(&vec1, &vec3) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_vector_normalization() {
        let mut vector = vec![3.0, 4.0];
        VectorUtils::normalize(&mut vector);

        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_centroid_calculation() {
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];

        let centroid = VectorUtils::centroid(&vectors).unwrap();
        assert!((centroid[0] - 2.0 / 3.0).abs() < 0.001);
        assert!((centroid[1] - 2.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_embedding_generator() {
        let mut generator = EmbeddingGenerator::new(64);

        let text1 = "hello world";
        let text2 = "hello world";
        let text3 = "goodbye world";

        let embedding1 = generator.generate_embedding(text1);
        let embedding2 = generator.generate_embedding(text2);
        let embedding3 = generator.generate_embedding(text3);

        // Same text should produce identical embeddings
        assert_eq!(embedding1, embedding2);

        // Different text should produce different embeddings
        assert_ne!(embedding1, embedding3);

        // Check dimension
        assert_eq!(embedding1.len(), 64);

        // Check that embeddings are normalized
        let norm1 = embedding1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_batch_embedding_generation() {
        let mut generator = EmbeddingGenerator::new(32);

        let texts = vec!["first text", "second text", "third text"];
        let embeddings = generator.batch_generate(&texts);

        assert_eq!(embeddings.len(), 3);
        assert!(embeddings.iter().all(|e| e.len() == 32));

        // Each embedding should be different
        assert_ne!(embeddings[0], embeddings[1]);
        assert_ne!(embeddings[1], embeddings[2]);
    }

    #[test]
    fn test_embedding_similarity() {
        let mut generator = EmbeddingGenerator::new(64);

        let similar1 = generator.generate_embedding("machine learning artificial intelligence");
        let similar2 = generator.generate_embedding("artificial intelligence machine learning");
        let different = generator.generate_embedding("cooking recipes kitchen");

        let sim1 = VectorUtils::cosine_similarity(&similar1, &similar2);
        let sim2 = VectorUtils::cosine_similarity(&similar1, &different);

        // Similar content should have higher similarity
        assert!(sim1 > sim2);
    }
}
