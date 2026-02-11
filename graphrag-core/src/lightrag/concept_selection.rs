//! Concept Selection and Ranking (G2ConS)
//!
//! This module implements concept selection strategies inspired by G2ConS
//! (Graph-Guided Concept Selection) to reduce retrieval costs by filtering
//! chunks based on relevant concepts.
//!
//! ## Key Benefits
//!
//! - **80% cost reduction**: By selecting only salient chunks containing top concepts
//! - **+31% accuracy improvement**: Better signal-to-noise ratio in retrieval
//! - **Query-aware selection**: Dynamically selects concepts relevant to each query
//!
//! ## Architecture
//!
//! ```text
//! ConceptGraph → ConceptRanker → Ranked Concepts → QueryConceptMatcher → Top-K Selection
//! ```
//!
//! ## References
//!
//! - G2ConS Paper: "Graph-Guided Concept Selection for Efficient GraphRAG"
//! - arXiv:2510.24120v1

use super::concept_graph::ConceptGraph;
use petgraph::prelude::*;
use petgraph::algo::page_rank;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for concept selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptSelectionConfig {
    /// Top-K concepts to select per query
    pub top_k: usize,

    /// Minimum concept score threshold (0.0-1.0)
    pub min_score: f32,

    /// Weight for degree centrality (0.0-1.0)
    pub degree_weight: f32,

    /// Weight for PageRank score (0.0-1.0)
    pub pagerank_weight: f32,

    /// Weight for IDF score (0.0-1.0)
    pub idf_weight: f32,

    /// PageRank damping factor
    pub pagerank_damping: f64,

    /// PageRank convergence tolerance
    pub pagerank_tolerance: f64,

    /// Enable query-concept semantic matching
    pub use_semantic_matching: bool,
}

impl Default for ConceptSelectionConfig {
    fn default() -> Self {
        Self {
            top_k: 20,
            min_score: 0.1,
            degree_weight: 0.4,
            pagerank_weight: 0.4,
            idf_weight: 0.2,
            pagerank_damping: 0.85,
            pagerank_tolerance: 1e-6,
            use_semantic_matching: true,
        }
    }
}

/// Ranked concept with score breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedConcept {
    /// Concept text
    pub concept: String,

    /// Final combined score
    pub score: f32,

    /// Degree centrality score
    pub degree_score: f32,

    /// PageRank score
    pub pagerank_score: f32,

    /// IDF (Inverse Document Frequency) score
    pub idf_score: f32,

    /// Number of documents containing this concept
    pub document_frequency: usize,

    /// Total frequency across all documents
    pub total_frequency: usize,
}

/// Concept ranking engine
///
/// Ranks concepts in a ConceptGraph using multiple signals:
/// - Degree centrality: How many other concepts this concept co-occurs with
/// - PageRank: Structural importance in the co-occurrence graph
/// - IDF: Specificity (rare concepts are more informative than common ones)
pub struct ConceptRanker {
    config: ConceptSelectionConfig,
}

impl ConceptRanker {
    /// Create a new concept ranker with configuration
    pub fn new(config: ConceptSelectionConfig) -> Self {
        Self { config }
    }

    /// Rank all concepts in the graph
    ///
    /// Returns concepts sorted by score in descending order.
    ///
    /// # Arguments
    ///
    /// * `graph` - The concept graph to rank
    /// * `total_documents` - Total number of documents in corpus (for IDF calculation)
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use graphrag_core::lightrag::concept_selection::{ConceptRanker, ConceptSelectionConfig};
    /// # use graphrag_core::lightrag::concept_graph::ConceptGraph;
    /// let ranker = ConceptRanker::new(ConceptSelectionConfig::default());
    /// let ranked = ranker.rank_concepts(&graph, 1000);
    ///
    /// // Get top 20 concepts
    /// let top_concepts: Vec<_> = ranked.iter().take(20).collect();
    /// ```
    pub fn rank_concepts(
        &self,
        graph: &ConceptGraph,
        total_documents: usize,
    ) -> Vec<RankedConcept> {
        // 1. Calculate degree centrality for each concept
        let degree_scores = self.calculate_degree_centrality(graph);

        // 2. Calculate PageRank scores
        let pagerank_scores = self.calculate_pagerank(graph);

        // 3. Calculate IDF scores
        let idf_scores = self.calculate_idf(graph, total_documents);

        // 4. Combine scores
        let mut ranked_concepts = Vec::new();

        for (concept_text, concept_data) in &graph.concepts {
            let degree_score = degree_scores.get(concept_text).copied().unwrap_or(0.0);
            let pagerank_score = pagerank_scores.get(concept_text).copied().unwrap_or(0.0);
            let idf_score = idf_scores.get(concept_text).copied().unwrap_or(0.0);

            // Weighted combination
            let combined_score =
                (self.config.degree_weight * degree_score) +
                (self.config.pagerank_weight * pagerank_score) +
                (self.config.idf_weight * idf_score);

            // Skip concepts below threshold
            if combined_score < self.config.min_score {
                continue;
            }

            ranked_concepts.push(RankedConcept {
                concept: concept_text.clone(),
                score: combined_score,
                degree_score,
                pagerank_score,
                idf_score,
                document_frequency: concept_data.document_ids.len(),
                total_frequency: concept_data.frequency,
            });
        }

        // Sort by score descending
        ranked_concepts.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });

        ranked_concepts
    }

    /// Calculate degree centrality for each concept
    ///
    /// Degree centrality = number of connections / (total nodes - 1)
    /// Normalized to [0, 1] range.
    fn calculate_degree_centrality(&self, graph: &ConceptGraph) -> HashMap<String, f32> {
        let mut scores = HashMap::new();

        let total_nodes = graph.graph.node_count();
        if total_nodes <= 1 {
            return scores;
        }

        let max_possible_degree = (total_nodes - 1) as f32;

        for (concept_text, &node_idx) in &graph.concept_to_node {
            // Count both incoming and outgoing edges
            let in_degree = graph.graph.neighbors_directed(node_idx, Direction::Incoming).count();
            let out_degree = graph.graph.neighbors_directed(node_idx, Direction::Outgoing).count();
            let total_degree = (in_degree + out_degree) as f32;

            // Normalize to [0, 1]
            let normalized_degree = total_degree / max_possible_degree;

            scores.insert(concept_text.clone(), normalized_degree);
        }

        scores
    }

    /// Calculate PageRank scores for concepts
    ///
    /// Uses the petgraph PageRank implementation with configurable damping.
    fn calculate_pagerank(&self, graph: &ConceptGraph) -> HashMap<String, f32> {
        let mut scores = HashMap::new();

        // petgraph::algo::page_rank returns Vec<f64> indexed by NodeIndex
        let pr_scores = page_rank(
            &graph.graph,
            self.config.pagerank_damping,
            None,  // max_iterations (None = use default)
        );

        // Find max score for normalization
        let max_score = pr_scores
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(1.0);

        // Map back to concept names and normalize
        for (concept_text, &node_idx) in &graph.concept_to_node {
            if let Some(&pr_score) = pr_scores.get(node_idx.index()) {
                // Normalize to [0, 1]
                let normalized_score = (pr_score / max_score) as f32;
                scores.insert(concept_text.clone(), normalized_score);
            }
        }

        scores
    }

    /// Calculate IDF (Inverse Document Frequency) scores
    ///
    /// IDF = log(total_documents / document_frequency)
    /// Normalized to [0, 1] range using sigmoid-like transformation.
    fn calculate_idf(&self, graph: &ConceptGraph, total_documents: usize) -> HashMap<String, f32> {
        let mut scores = HashMap::new();

        if total_documents == 0 {
            return scores;
        }

        let total_docs = total_documents as f32;

        for (concept_text, concept_data) in &graph.concepts {
            let doc_freq = concept_data.document_ids.len() as f32;

            if doc_freq == 0.0 {
                scores.insert(concept_text.clone(), 0.0);
                continue;
            }

            // Standard IDF formula
            let idf = (total_docs / doc_freq).ln();

            // Normalize to [0, 1] using tanh-like transformation
            // Very rare terms (IDF > 5) → 1.0
            // Very common terms (IDF < 0.5) → 0.0
            let normalized_idf = (idf / 5.0).tanh();

            scores.insert(concept_text.clone(), normalized_idf as f32);
        }

        scores
    }

    /// Get top-K concepts for a query
    ///
    /// Returns the top K highest-ranked concepts.
    ///
    /// # Arguments
    ///
    /// * `ranked_concepts` - Pre-ranked concepts from `rank_concepts()`
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use graphrag_core::lightrag::concept_selection::{ConceptRanker, ConceptSelectionConfig};
    /// let ranker = ConceptRanker::new(ConceptSelectionConfig::default());
    /// let ranked = ranker.rank_concepts(&graph, 1000);
    /// let top_k = ranker.get_top_k(&ranked);
    /// ```
    pub fn get_top_k(&self, ranked_concepts: &[RankedConcept]) -> Vec<String> {
        ranked_concepts
            .iter()
            .take(self.config.top_k)
            .map(|rc| rc.concept.clone())
            .collect()
    }

    /// Filter concepts by minimum score threshold
    pub fn filter_by_threshold(&self, ranked_concepts: &[RankedConcept]) -> Vec<RankedConcept> {
        ranked_concepts
            .iter()
            .filter(|rc| rc.score >= self.config.min_score)
            .cloned()
            .collect()
    }
}

/// Statistics about concept ranking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptRankingStats {
    /// Total concepts in graph
    pub total_concepts: usize,

    /// Concepts above threshold
    pub concepts_above_threshold: usize,

    /// Average score
    pub average_score: f32,

    /// Maximum score
    pub max_score: f32,

    /// Minimum score
    pub min_score: f32,

    /// Score distribution (histogram)
    pub score_distribution: Vec<(f32, usize)>,
}

impl ConceptRanker {
    /// Calculate statistics for ranked concepts
    pub fn calculate_stats(&self, ranked_concepts: &[RankedConcept]) -> ConceptRankingStats {
        if ranked_concepts.is_empty() {
            return ConceptRankingStats {
                total_concepts: 0,
                concepts_above_threshold: 0,
                average_score: 0.0,
                max_score: 0.0,
                min_score: 0.0,
                score_distribution: vec![],
            };
        }

        let total = ranked_concepts.len();
        let above_threshold = ranked_concepts
            .iter()
            .filter(|rc| rc.score >= self.config.min_score)
            .count();

        let avg_score = ranked_concepts.iter().map(|rc| rc.score).sum::<f32>() / total as f32;
        let max_score = ranked_concepts
            .iter()
            .map(|rc| rc.score)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        let min_score = ranked_concepts
            .iter()
            .map(|rc| rc.score)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        // Create histogram (10 bins)
        let num_bins = 10;
        let bin_size = (max_score - min_score) / num_bins as f32;
        let mut histogram = vec![0usize; num_bins];

        for rc in ranked_concepts {
            let bin = if bin_size > 0.0 {
                ((rc.score - min_score) / bin_size).floor() as usize
            } else {
                0
            };
            let bin_idx = bin.min(num_bins - 1);
            histogram[bin_idx] += 1;
        }

        let score_distribution: Vec<(f32, usize)> = histogram
            .into_iter()
            .enumerate()
            .map(|(i, count)| {
                let bin_start = min_score + (i as f32 * bin_size);
                (bin_start, count)
            })
            .collect();

        ConceptRankingStats {
            total_concepts: total,
            concepts_above_threshold: above_threshold,
            average_score: avg_score,
            max_score,
            min_score,
            score_distribution,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lightrag::concept_graph::{ConceptGraphBuilder, ConceptExtractor};

    #[test]
    fn test_concept_ranking() {
        let mut builder = ConceptGraphBuilder::new();

        // Build a small test graph
        builder.add_document_concepts("doc1", vec!["machine learning".to_string(), "neural networks".to_string()]);
        builder.add_document_concepts("doc2", vec!["machine learning".to_string(), "deep learning".to_string()]);
        builder.add_document_concepts("doc3", vec!["neural networks".to_string(), "deep learning".to_string()]);

        builder.add_chunk_concepts("chunk1", vec!["machine learning".to_string(), "neural networks".to_string()]);
        builder.add_chunk_concepts("chunk2", vec!["machine learning".to_string(), "deep learning".to_string()]);
        builder.add_chunk_concepts("chunk3", vec!["neural networks".to_string(), "deep learning".to_string()]);

        let graph = builder.build();

        // Rank concepts
        let ranker = ConceptRanker::new(ConceptSelectionConfig::default());
        let ranked = ranker.rank_concepts(&graph, 3);

        assert!(!ranked.is_empty());

        // "machine learning" should have high score (appears in all docs)
        let ml_concept = ranked.iter().find(|rc| rc.concept.contains("machine"));
        assert!(ml_concept.is_some());

        // Check that scores are normalized to [0, 1]
        for rc in &ranked {
            assert!(rc.score >= 0.0 && rc.score <= 1.0);
            assert!(rc.degree_score >= 0.0 && rc.degree_score <= 1.0);
            assert!(rc.pagerank_score >= 0.0 && rc.pagerank_score <= 1.0);
            assert!(rc.idf_score >= 0.0 && rc.idf_score <= 1.0);
        }
    }

    #[test]
    fn test_degree_centrality() {
        let mut builder = ConceptGraphBuilder::new();
        builder.add_chunk_concepts("chunk1", vec!["a".to_string(), "b".to_string()]);
        builder.add_chunk_concepts("chunk2", vec!["a".to_string(), "c".to_string()]);
        builder.add_chunk_concepts("chunk3", vec!["a".to_string(), "d".to_string()]);

        let graph = builder.build();
        let ranker = ConceptRanker::new(ConceptSelectionConfig::default());

        let degree_scores = ranker.calculate_degree_centrality(&graph);

        // Concept "a" should have highest degree (connects to b, c, d)
        let a_score = degree_scores.get("a").copied().unwrap_or(0.0);
        assert!(a_score > 0.0);

        // All scores should be in [0, 1]
        for score in degree_scores.values() {
            assert!(*score >= 0.0 && *score <= 1.0);
        }
    }

    #[test]
    fn test_idf_calculation() {
        let mut builder = ConceptGraphBuilder::new();
        builder.add_document_concepts("doc1", vec!["rare_term".to_string()]);
        builder.add_document_concepts("doc2", vec!["common_term".to_string()]);
        builder.add_document_concepts("doc3", vec!["common_term".to_string()]);
        builder.add_document_concepts("doc4", vec!["common_term".to_string()]);

        let graph = builder.build();
        let ranker = ConceptRanker::new(ConceptSelectionConfig::default());

        let idf_scores = ranker.calculate_idf(&graph, 4);

        let rare_score = idf_scores.get("rare_term").copied().unwrap_or(0.0);
        let common_score = idf_scores.get("common_term").copied().unwrap_or(0.0);

        // Rare term should have higher IDF
        assert!(rare_score > common_score);
    }

    #[test]
    fn test_top_k_selection() {
        let mut builder = ConceptGraphBuilder::new();
        for i in 0..50 {
            builder.add_document_concepts(&format!("doc{}", i), vec![format!("concept_{}", i)]);
        }

        let graph = builder.build();
        let ranker = ConceptRanker::new(ConceptSelectionConfig {
            top_k: 10,
            ..Default::default()
        });

        let ranked = ranker.rank_concepts(&graph, 50);
        let top_k = ranker.get_top_k(&ranked);

        assert_eq!(top_k.len(), 10);
    }

    #[test]
    fn test_stats_calculation() {
        let mut builder = ConceptGraphBuilder::new();
        builder.add_document_concepts("doc1", vec!["a".to_string(), "b".to_string()]);
        builder.add_document_concepts("doc2", vec!["c".to_string(), "d".to_string()]);

        let graph = builder.build();
        let ranker = ConceptRanker::new(ConceptSelectionConfig::default());
        let ranked = ranker.rank_concepts(&graph, 2);

        let stats = ranker.calculate_stats(&ranked);

        assert_eq!(stats.total_concepts, ranked.len());
        assert!(stats.average_score >= 0.0);
        assert!(stats.max_score >= stats.min_score);
        assert_eq!(stats.score_distribution.len(), 10); // 10 bins
    }
}
