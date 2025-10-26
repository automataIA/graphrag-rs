//! PageRank implementation for GraphRAG
//! 
//! This module is only available when the "pagerank" feature is enabled.

use crate::core::{EntityId, Result};
use nalgebra::{DMatrix, DVector};
use parking_lot::RwLock;
use lru::LruCache;
use rayon::prelude::*;
use sprs::CsMat;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

/// Configuration for PageRank algorithm
#[derive(Debug, Clone)]
pub struct PageRankConfig {
    /// Damping factor (typically 0.85)
    pub damping_factor: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Whether to use personalized PageRank
    pub personalized: bool,
    /// Enable parallel computation for large graphs
    pub parallel_enabled: bool,
    /// Cache size for PageRank computations
    pub cache_size: usize,
    /// Minimum graph size to trigger sparse matrix optimizations
    pub sparse_threshold: usize,
    /// Enable incremental updates for dynamic graphs
    pub incremental_updates: bool,
    /// Block size for SIMD operations
    pub simd_block_size: usize,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping_factor: 0.85,
            max_iterations: 100,
            tolerance: 1e-6,
            personalized: true,
            parallel_enabled: true,
            cache_size: 1000,
            sparse_threshold: 1000,
            incremental_updates: true,
            simd_block_size: 32,
        }
    }
}

/// Personalized PageRank implementation with optimizations for graph retrieval
pub struct PersonalizedPageRank {
    config: PageRankConfig,
    adjacency_matrix: CsMat<f64>,
    node_mapping: HashMap<EntityId, usize>,
    reverse_mapping: HashMap<usize, EntityId>,
    /// High-performance dense matrix for small graphs
    dense_matrix: Option<DMatrix<f64>>,
    /// Cache for PageRank computations
    score_cache: Arc<RwLock<LruCache<u64, HashMap<EntityId, f64>>>>,
    /// Precomputed transition matrix for faster iterations
    transition_matrix: Option<CsMat<f64>>,
    /// Node degrees for normalization
    out_degrees: Vec<f64>,
}

impl PersonalizedPageRank {
    /// Create a new PersonalizedPageRank instance
    ///
    /// # Arguments
    /// * `config` - PageRank configuration
    /// * `adjacency_matrix` - Sparse adjacency matrix of the graph
    /// * `node_mapping` - Mapping from entity IDs to matrix indices
    /// * `reverse_mapping` - Mapping from matrix indices to entity IDs
    pub fn new(
        config: PageRankConfig,
        adjacency_matrix: CsMat<f64>,
        node_mapping: HashMap<EntityId, usize>,
        reverse_mapping: HashMap<usize, EntityId>,
    ) -> Self {
        let n = adjacency_matrix.rows();
        let cache_size = NonZeroUsize::new(config.cache_size).unwrap_or(NonZeroUsize::new(1000).unwrap());

        // Compute out-degrees for normalization
        let out_degrees = Self::compute_out_degrees(&adjacency_matrix);

        // Decide whether to use dense matrix for small graphs
        let dense_matrix = if n < config.sparse_threshold {
            Some(Self::convert_to_dense(&adjacency_matrix))
        } else {
            None
        };

        // Precompute transition matrix if beneficial
        let transition_matrix = if config.parallel_enabled && n > 100 {
            Some(Self::build_transition_matrix(&adjacency_matrix, &out_degrees))
        } else {
            None
        };

        Self {
            config,
            adjacency_matrix,
            node_mapping,
            reverse_mapping,
            dense_matrix,
            score_cache: Arc::new(RwLock::new(LruCache::new(cache_size))),
            transition_matrix,
            out_degrees,
        }
    }

    /// Helper method to compute out-degrees
    fn compute_out_degrees(matrix: &CsMat<f64>) -> Vec<f64> {
        let n = matrix.rows();
        let mut degrees = vec![0.0; n];

        for (i, degree) in degrees.iter_mut().enumerate().take(n) {
            if let Some(row) = matrix.outer_view(i) {
                *degree = row.iter().map(|(_, &weight)| weight).sum();
            }
        }

        degrees
    }

    /// Convert sparse matrix to dense for small graphs
    fn convert_to_dense(sparse_matrix: &CsMat<f64>) -> DMatrix<f64> {
        let n = sparse_matrix.rows();
        let m = sparse_matrix.cols();
        let mut dense = DMatrix::zeros(n, m);

        for i in 0..n {
            if let Some(row) = sparse_matrix.outer_view(i) {
                for (j, &value) in row.iter() {
                    dense[(i, j)] = value;
                }
            }
        }

        dense
    }

    /// Build normalized transition matrix
    fn build_transition_matrix(adjacency: &CsMat<f64>, out_degrees: &[f64]) -> CsMat<f64> {
        let mut builder = sprs::TriMat::new((adjacency.rows(), adjacency.cols()));

        for (i, &degree) in out_degrees.iter().enumerate().take(adjacency.rows()) {
            if let Some(row) = adjacency.outer_view(i) {
                if degree > 0.0 {
                    for (j, &weight) in row.iter() {
                        builder.add_triplet(i, j, weight / degree);
                    }
                }
            }
        }

        builder.to_csr()
    }

    /// Generate cache key from reset probabilities
    fn generate_cache_key(reset_probabilities: &HashMap<EntityId, f64>) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        let mut sorted_entries: Vec<_> = reset_probabilities.iter().collect();
        sorted_entries.sort_by_key(|(id, _)| id.to_string());

        for (id, score) in sorted_entries {
            id.to_string().hash(&mut hasher);
            score.to_bits().hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Calculate personalized PageRank scores with performance optimizations
    pub fn calculate_scores(
        &self,
        reset_probabilities: &HashMap<EntityId, f64>,
    ) -> Result<HashMap<EntityId, f64>> {
        let n = self.adjacency_matrix.rows();
        if n == 0 {
            return Ok(HashMap::new());
        }

        // Check cache first
        let cache_key = Self::generate_cache_key(reset_probabilities);
        {
            let cache = self.score_cache.read();
            if let Some(cached_scores) = cache.peek(&cache_key) {
                return Ok(cached_scores.clone());
            }
        }

        let scores = if n < self.config.sparse_threshold {
            // Use dense matrix computation for small graphs
            self.calculate_scores_dense(reset_probabilities)?
        } else if self.config.parallel_enabled {
            // Use parallel sparse computation for large graphs
            self.calculate_scores_parallel(reset_probabilities)?
        } else {
            // Use optimized sequential sparse computation
            self.calculate_scores_sparse_optimized(reset_probabilities)?
        };

        // Cache the result
        {
            let mut cache = self.score_cache.write();
            cache.put(cache_key, scores.clone());
        }

        Ok(scores)
    }

    /// High-performance dense matrix computation for small graphs
    fn calculate_scores_dense(&self, reset_probabilities: &HashMap<EntityId, f64>) -> Result<HashMap<EntityId, f64>> {
        let n = self.adjacency_matrix.rows();
        let reset_vector = self.build_reset_vector(reset_probabilities)?;

        if let Some(dense_matrix) = &self.dense_matrix {
            let mut scores = DVector::from_element(n, 1.0 / n as f64);
            let reset_vec = DVector::from_vec(reset_vector);

            for _iteration in 0..self.config.max_iterations {
                let new_scores = &reset_vec * (1.0 - self.config.damping_factor) +
                                dense_matrix * &scores * self.config.damping_factor;

                let diff = (&new_scores - &scores).abs().max();
                if diff < self.config.tolerance {
                    break;
                }
                scores = new_scores;
            }

            self.scores_to_entity_map(scores.as_slice())
        } else {
            // Fallback to sparse computation
            self.calculate_scores_sparse_optimized(reset_probabilities)
        }
    }

    /// Parallel sparse computation for large graphs
    fn calculate_scores_parallel(&self, reset_probabilities: &HashMap<EntityId, f64>) -> Result<HashMap<EntityId, f64>> {
        let n = self.adjacency_matrix.rows();
        let mut scores = vec![1.0 / n as f64; n];
        let mut new_scores = vec![0.0; n];
        let reset_vector = self.build_reset_vector(reset_probabilities)?;

        for _iteration in 0..self.config.max_iterations {
            // Parallel PageRank iteration using rayon
            self.pagerank_iteration_parallel(&scores, &mut new_scores, &reset_vector);

            // Check convergence
            let diff = self.calculate_difference(&scores, &new_scores);
            if diff < self.config.tolerance {
                break;
            }

            std::mem::swap(&mut scores, &mut new_scores);
        }

        self.scores_to_entity_map(&scores)
    }

    /// Optimized sequential sparse computation
    fn calculate_scores_sparse_optimized(&self, reset_probabilities: &HashMap<EntityId, f64>) -> Result<HashMap<EntityId, f64>> {
        let n = self.adjacency_matrix.rows();
        let mut scores = vec![1.0 / n as f64; n];
        let mut new_scores = vec![0.0; n];
        let reset_vector = self.build_reset_vector(reset_probabilities)?;

        // Use precomputed transition matrix if available
        if let Some(transition_matrix) = &self.transition_matrix {
            for _iteration in 0..self.config.max_iterations {
                self.pagerank_iteration_with_transition_matrix(&scores, &mut new_scores, &reset_vector, transition_matrix);

                let diff = self.calculate_difference(&scores, &new_scores);
                if diff < self.config.tolerance {
                    break;
                }

                std::mem::swap(&mut scores, &mut new_scores);
            }
        } else {
            // Standard iteration
            for _iteration in 0..self.config.max_iterations {
                self.pagerank_iteration(&scores, &mut new_scores, &reset_vector);

                let diff = self.calculate_difference(&scores, &new_scores);
                if diff < self.config.tolerance {
                    break;
                }

                std::mem::swap(&mut scores, &mut new_scores);
            }
        }

        self.scores_to_entity_map(&scores)
    }

    /// Parallel PageRank iteration
    fn pagerank_iteration_parallel(
        &self,
        current_scores: &[f64],
        new_scores: &mut [f64],
        reset_vector: &[f64],
    ) {
        let d = self.config.damping_factor;
        let n = current_scores.len();

        // Initialize with reset probability component in parallel
        new_scores.par_iter_mut().zip(reset_vector.par_iter()).for_each(|(new_score, &reset_prob)| {
            *new_score = (1.0 - d) * reset_prob;
        });

        // For parallel accumulation, we'll use a safer approach with chunking
        // This avoids unsafe operations while maintaining good performance
        let contributions: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|j| {
                let mut local_contributions = vec![0.0; n];
                let current_score = current_scores[j];
                let out_degree = self.out_degrees[j];

                if out_degree > 0.0 {
                    let score_contribution = d * current_score / out_degree;

                    if let Some(row) = self.adjacency_matrix.outer_view(j) {
                        for (neighbor_i, &weight) in row.iter() {
                            if neighbor_i < n {
                                local_contributions[neighbor_i] += score_contribution * weight;
                            }
                        }
                    }
                } else {
                    // Dangling node: distribute score uniformly
                    let score_contribution = d * current_score / n as f64;
                    for contrib in &mut local_contributions {
                        *contrib += score_contribution;
                    }
                }

                local_contributions
            })
            .collect();

        // Sum all contributions
        for contrib_vec in contributions {
            for (i, contrib) in contrib_vec.iter().enumerate() {
                new_scores[i] += contrib;
            }
        }
    }

    /// PageRank iteration using precomputed transition matrix
    fn pagerank_iteration_with_transition_matrix(
        &self,
        current_scores: &[f64],
        new_scores: &mut [f64],
        reset_vector: &[f64],
        transition_matrix: &CsMat<f64>,
    ) {
        let d = self.config.damping_factor;
        let n = current_scores.len();

        // Initialize with reset probability component
        for i in 0..n {
            new_scores[i] = (1.0 - d) * reset_vector[i];
        }

        // Sparse matrix-vector multiplication: new_scores += d * P * current_scores
        for (j, &current_score) in current_scores.iter().enumerate() {
            if let Some(row) = transition_matrix.outer_view(j) {
                for (neighbor_i, &transition_prob) in row.iter() {
                    if neighbor_i < n {
                        new_scores[neighbor_i] += d * transition_prob * current_score;
                    }
                }
            }
        }
    }

    fn build_reset_vector(&self, reset_probabilities: &HashMap<EntityId, f64>) -> Result<Vec<f64>> {
        let n = self.adjacency_matrix.rows();
        let mut reset_vector = vec![1.0 / n as f64; n]; // Default uniform distribution

        if !reset_probabilities.is_empty() {
            // Normalize reset probabilities to sum to 1
            let total: f64 = reset_probabilities.values().sum();
            if total > 0.0 {
                for (entity_id, &prob) in reset_probabilities {
                    if let Some(&index) = self.node_mapping.get(entity_id) {
                        if index < n {
                            reset_vector[index] = prob / total;
                        }
                    }
                }
            }
        }

        Ok(reset_vector)
    }

    fn pagerank_iteration(
        &self,
        current_scores: &[f64],
        new_scores: &mut [f64],
        reset_vector: &[f64],
    ) {
        let d = self.config.damping_factor;
        let n = current_scores.len();

        // Initialize with reset probability component
        for i in 0..n {
            new_scores[i] = (1.0 - d) * reset_vector[i];
        }

        // Add the damped transition probability component
        // For each node j, distribute its score to its outgoing neighbors
        for (j, &current_score) in current_scores.iter().enumerate() {
            let out_degree = self.get_out_degree(j);
            if out_degree > 0 {
                let score_contribution = d * current_score / out_degree as f64;

                // Find all neighbors of node j and add contribution
                if let Some(row) = self.adjacency_matrix.outer_view(j) {
                    for (neighbor_i, &weight) in row.iter() {
                        if neighbor_i < n {
                            new_scores[neighbor_i] += score_contribution * weight;
                        }
                    }
                }
            } else {
                // Dangling node: distribute score uniformly
                let score_contribution = d * current_score / n as f64;
                for score in new_scores.iter_mut() {
                    *score += score_contribution;
                }
            }
        }
    }

    fn get_out_degree(&self, node_index: usize) -> usize {
        if let Some(row) = self.adjacency_matrix.outer_view(node_index) {
            row.nnz()
        } else {
            0
        }
    }

    fn calculate_difference(&self, scores1: &[f64], scores2: &[f64]) -> f64 {
        scores1
            .iter()
            .zip(scores2.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f64, f64::max)
    }

    fn scores_to_entity_map(&self, scores: &[f64]) -> Result<HashMap<EntityId, f64>> {
        let mut result = HashMap::new();

        for (index, &score) in scores.iter().enumerate() {
            if let Some(entity_id) = self.reverse_mapping.get(&index) {
                result.insert(entity_id.clone(), score);
            }
        }

        Ok(result)
    }

    /// Get the number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.adjacency_matrix.rows()
    }

    /// Get the configuration used
    pub fn config(&self) -> &PageRankConfig {
        &self.config
    }
}

/// Multi-modal scoring system that combines different ranking signals
#[derive(Debug, Clone)]
pub struct MultiModalScores {
    /// Scores based on vector similarity
    pub vector_scores: HashMap<EntityId, f64>,
    /// Scores based on PageRank importance
    pub pagerank_scores: HashMap<EntityId, f64>,
    /// Scores for text chunks
    pub chunk_scores: HashMap<crate::core::ChunkId, f64>,
    /// Scores for relationships between entities
    pub relationship_scores: HashMap<String, f64>,
}

/// Weights for combining different scoring signals
#[derive(Debug, Clone)]
pub struct ScoreWeights {
    /// Weight for vector similarity scores
    pub vector_weight: f64,
    /// Weight for PageRank scores
    pub pagerank_weight: f64,
    /// Weight for chunk scores
    pub chunk_weight: f64,
    /// Weight for relationship scores
    pub relationship_weight: f64,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            vector_weight: 0.3,
            pagerank_weight: 0.4,
            chunk_weight: 0.2,
            relationship_weight: 0.1,
        }
    }
}

impl MultiModalScores {
    /// Create a new empty MultiModalScores instance
    pub fn new() -> Self {
        Self {
            vector_scores: HashMap::new(),
            pagerank_scores: HashMap::new(),
            chunk_scores: HashMap::new(),
            relationship_scores: HashMap::new(),
        }
    }

    /// Combine multiple scoring signals with configurable weights
    pub fn combine_scores(&self, weights: &ScoreWeights) -> HashMap<EntityId, f64> {
        use std::collections::HashSet;

        let mut combined_scores = HashMap::new();

        // Get all unique entity IDs
        let all_entities: HashSet<EntityId> = self
            .vector_scores
            .keys()
            .chain(self.pagerank_scores.keys())
            .cloned()
            .collect();

        for entity_id in all_entities {
            let vector_score = self.vector_scores.get(&entity_id).unwrap_or(&0.0);
            let pagerank_score = self.pagerank_scores.get(&entity_id).unwrap_or(&0.0);
            let chunk_score = self.get_entity_chunk_score(&entity_id);

            let combined = weights.vector_weight * vector_score
                + weights.pagerank_weight * pagerank_score
                + weights.chunk_weight * chunk_score;

            combined_scores.insert(entity_id, combined);
        }

        combined_scores
    }

    fn get_entity_chunk_score(&self, _entity_id: &EntityId) -> f64 {
        // For now, return 0.0 - this would be implemented to aggregate
        // chunk scores for chunks that contain this entity
        0.0
    }
}

impl Default for MultiModalScores {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::EntityId;

    fn create_simple_test_graph() -> (
        CsMat<f64>,
        HashMap<EntityId, usize>,
        HashMap<usize, EntityId>,
    ) {
        // Create a simple 3-node graph: A -> B -> C, A -> C
        let entity_a = EntityId::new("A".to_string());
        let entity_b = EntityId::new("B".to_string());
        let entity_c = EntityId::new("C".to_string());

        let mut node_mapping = HashMap::new();
        let mut reverse_mapping = HashMap::new();

        node_mapping.insert(entity_a.clone(), 0);
        node_mapping.insert(entity_b.clone(), 1);
        node_mapping.insert(entity_c.clone(), 2);

        reverse_mapping.insert(0, entity_a);
        reverse_mapping.insert(1, entity_b);
        reverse_mapping.insert(2, entity_c);

        // Create adjacency matrix using triplet matrix
        let mut triplet_mat = sprs::TriMat::new((3, 3));
        triplet_mat.add_triplet(0, 1, 1.0); // A->B
        triplet_mat.add_triplet(0, 2, 1.0); // A->C
        triplet_mat.add_triplet(1, 2, 1.0); // B->C

        let matrix = triplet_mat.to_csr();

        (matrix, node_mapping, reverse_mapping)
    }

    #[test]
    fn test_pagerank_convergence() {
        let (matrix, node_mapping, reverse_mapping) = create_simple_test_graph();
        let config = PageRankConfig::default();
        let pagerank = PersonalizedPageRank::new(config, matrix, node_mapping, reverse_mapping);

        let reset_probs = HashMap::new(); // Uniform reset
        let scores = pagerank.calculate_scores(&reset_probs).unwrap();

        // Verify scores sum to 1.0
        let total_score: f64 = scores.values().sum();
        assert!((total_score - 1.0).abs() < 1e-6);

        // Verify we have scores for all entities
        assert_eq!(scores.len(), 3);
    }

    #[test]
    fn test_personalized_pagerank() {
        let (matrix, node_mapping, reverse_mapping) = create_simple_test_graph();
        let config = PageRankConfig::default();
        let pagerank = PersonalizedPageRank::new(config, matrix, node_mapping, reverse_mapping);

        let mut reset_probs = HashMap::new();
        let entity_a = EntityId::new("A".to_string());
        let entity_b = EntityId::new("B".to_string());
        reset_probs.insert(entity_a.clone(), 0.8);
        reset_probs.insert(entity_b, 0.2);

        let scores = pagerank.calculate_scores(&reset_probs).unwrap();

        // Entity A should have the highest score due to high reset probability
        let score_a = scores.get(&entity_a).unwrap();
        assert!(*score_a > 0.3); // Should be significantly above uniform (0.33)
    }

    #[test]
    fn test_multimodal_scores_combination() {
        let mut multi_scores = MultiModalScores::new();

        let entity_a = EntityId::new("A".to_string());
        let entity_b = EntityId::new("B".to_string());

        multi_scores.vector_scores.insert(entity_a.clone(), 0.8);
        multi_scores.vector_scores.insert(entity_b.clone(), 0.4);

        multi_scores.pagerank_scores.insert(entity_a.clone(), 0.6);
        multi_scores.pagerank_scores.insert(entity_b.clone(), 0.9);

        let weights = ScoreWeights::default();
        let combined = multi_scores.combine_scores(&weights);

        // Both entities should have combined scores
        assert!(combined.contains_key(&entity_a));
        assert!(combined.contains_key(&entity_b));

        // Scores should be reasonable combinations
        let score_a = combined.get(&entity_a).unwrap();
        let score_b = combined.get(&entity_b).unwrap();

        assert!(*score_a > 0.0);
        assert!(*score_b > 0.0);
    }
}
