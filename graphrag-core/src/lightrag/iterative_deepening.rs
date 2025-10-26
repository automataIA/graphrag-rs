//! Iterative Deepening Search for LazyGraphRAG
//!
//! This module implements iterative deepening search that progressively explores
//! the concept graph to find relevant information without requiring full graph traversal.
//!
//! ## Key Features
//!
//! - **Progressive Exploration**: Start with high-confidence results, deepen if needed
//! - **Early Termination**: Stop when sufficient relevant chunks are found
//! - **Depth-Limited**: Control exploration depth to balance speed vs completeness
//! - **Relevance-Guided**: Use query refinement to guide exploration
//!
//! ## Algorithm
//!
//! 1. Start with initial query concepts (depth 0)
//! 2. Retrieve chunks for current concepts
//! 3. If sufficient results, return
//! 4. Otherwise, expand to related concepts (depth + 1)
//! 5. Repeat until max depth or sufficient results
//!
//! ## Example
//!
//! ```rust
//! use graphrag_core::lightrag::iterative_deepening::{IterativeDeepeningSearch, SearchConfig};
//!
//! let config = SearchConfig {
//!     max_depth: 3,
//!     min_chunks: 5,
//!     max_chunks: 20,
//!     ..Default::default()
//! };
//!
//! let search = IterativeDeepeningSearch::new(config);
//! let results = search.search("machine learning", &concept_graph, &bidirectional_index);
//!
//! println!("Found {} chunks at depth {}", results.chunk_count(), results.depth_reached);
//! ```

use crate::core::ChunkId;
use crate::entity::BidirectionalIndex;
use crate::lightrag::concept_graph::ConceptGraph;
use crate::lightrag::query_refinement::{QueryRefiner, QueryRefinementConfig, RefinedQuery};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Configuration for iterative deepening search
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Maximum search depth (number of expansion iterations)
    pub max_depth: usize,

    /// Minimum number of chunks to retrieve before stopping
    pub min_chunks: usize,

    /// Maximum number of chunks to retrieve (stop if exceeded)
    pub max_chunks: usize,

    /// Number of concepts to expand per depth level
    pub concepts_per_depth: usize,

    /// Use adaptive depth (stop early if results are good)
    pub use_adaptive_depth: bool,

    /// Minimum quality threshold for adaptive stopping (0.0-1.0)
    pub adaptive_quality_threshold: f64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            min_chunks: 5,
            max_chunks: 50,
            concepts_per_depth: 5,
            use_adaptive_depth: true,
            adaptive_quality_threshold: 0.7,
        }
    }
}

/// Iterative deepening search implementation
pub struct IterativeDeepeningSearch {
    config: SearchConfig,
    query_refiner: QueryRefiner,
}

impl IterativeDeepeningSearch {
    /// Create a new iterative deepening search with configuration
    pub fn new(config: SearchConfig) -> Self {
        let refinement_config = QueryRefinementConfig {
            max_iterations: config.max_depth,
            concepts_per_iteration: config.concepts_per_depth,
            min_cooccurrence: 1,
            max_total_concepts: config.concepts_per_depth * config.max_depth,
            use_relevance_feedback: true,
        };

        Self {
            config,
            query_refiner: QueryRefiner::new(refinement_config),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(SearchConfig::default())
    }

    /// Perform iterative deepening search
    ///
    /// Returns search results with relevant chunks and search statistics
    pub fn search(
        &self,
        query: &str,
        concept_graph: &ConceptGraph,
        bidirectional_index: &BidirectionalIndex,
    ) -> SearchResults {
        let mut results = SearchResults::new(query.to_string());
        let mut current_concepts: HashSet<String> = HashSet::new();
        let mut visited_chunks: HashSet<ChunkId> = HashSet::new();

        // Perform query refinement to get initial concepts
        let refined_query = self.query_refiner.refine_query(
            query,
            concept_graph,
            bidirectional_index,
        );

        if refined_query.initial_concepts.is_empty() {
            return results;
        }

        // Add initial concepts
        current_concepts.extend(refined_query.initial_concepts.iter().cloned());

        // Iterative deepening
        for depth in 0..self.config.max_depth {
            let depth_results = self.search_at_depth(
                depth,
                &current_concepts,
                concept_graph,
                bidirectional_index,
                &mut visited_chunks,
            );

            // Add results for this depth
            results.add_depth_results(depth, depth_results.clone());

            // Check stopping conditions
            if visited_chunks.len() >= self.config.max_chunks {
                results.depth_reached = depth;
                results.stop_reason = StopReason::MaxChunksReached;
                break;
            }

            if visited_chunks.len() >= self.config.min_chunks
                && self.config.use_adaptive_depth
                && self.should_stop_early(&results, depth)
            {
                results.depth_reached = depth;
                results.stop_reason = StopReason::QualityThresholdMet;
                break;
            }

            // Expand to related concepts for next depth
            let expanded_concepts = self.expand_concepts(
                &current_concepts,
                concept_graph,
                self.config.concepts_per_depth,
            );

            if expanded_concepts.is_empty() {
                results.depth_reached = depth;
                results.stop_reason = StopReason::NoMoreConcepts;
                break;
            }

            current_concepts.extend(expanded_concepts);
            results.depth_reached = depth + 1;
        }

        // Final statistics
        results.total_chunks = visited_chunks.len();
        results.total_concepts_explored = current_concepts.len();
        results.chunk_ids = visited_chunks.into_iter().collect();

        results
    }

    /// Search at a specific depth level
    fn search_at_depth(
        &self,
        depth: usize,
        concepts: &HashSet<String>,
        concept_graph: &ConceptGraph,
        bidirectional_index: &BidirectionalIndex,
        visited_chunks: &mut HashSet<ChunkId>,
    ) -> DepthResults {
        let mut depth_results = DepthResults {
            depth,
            concepts_explored: concepts.len(),
            new_chunks_found: 0,
            chunk_ids: Vec::new(),
        };

        // Get chunks for all current concepts
        for concept in concepts {
            // Convert concept to entity ID
            let entity_id = crate::core::EntityId::new(self.normalize_concept(concept));

            // Get chunks for this concept
            let chunks = bidirectional_index.get_chunks_for_entity(&entity_id);

            for chunk_id in chunks {
                if visited_chunks.insert(chunk_id.clone()) {
                    depth_results.new_chunks_found += 1;
                    depth_results.chunk_ids.push(chunk_id);
                }
            }
        }

        depth_results
    }

    /// Expand current concepts to related concepts
    fn expand_concepts(
        &self,
        current_concepts: &HashSet<String>,
        concept_graph: &ConceptGraph,
        max_expand: usize,
    ) -> Vec<String> {
        let mut related_concepts: HashMap<String, f64> = HashMap::new();

        for concept in current_concepts {
            // Get related concepts from graph
            let related = concept_graph.get_related_concepts(concept, max_expand);

            for related_concept in related {
                if !current_concepts.contains(&related_concept) {
                    // Score based on connectivity
                    let score = self.score_concept(&related_concept, current_concepts, concept_graph);
                    *related_concepts.entry(related_concept).or_insert(0.0) += score;
                }
            }
        }

        // Sort by score and take top-k
        let mut sorted_concepts: Vec<_> = related_concepts.into_iter().collect();
        sorted_concepts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        sorted_concepts
            .into_iter()
            .take(max_expand)
            .map(|(c, _)| c)
            .collect()
    }

    /// Score a concept based on its relevance to current concepts
    fn score_concept(
        &self,
        concept: &str,
        current_concepts: &HashSet<String>,
        concept_graph: &ConceptGraph,
    ) -> f64 {
        if let Some(concept_data) = concept_graph.concepts.get(concept) {
            let mut score = (concept_data.frequency as f64).ln() + 1.0;

            // Count connections to current concepts
            let mut connections = 0;
            for current in current_concepts {
                let has_relation = concept_graph.relations.iter().any(|rel| {
                    (rel.source == *concept && rel.target == *current)
                        || (rel.source == *current && rel.target == *concept)
                });

                if has_relation {
                    connections += 1;
                }
            }

            score *= 1.0 + (connections as f64 * 0.5);
            score
        } else {
            0.0
        }
    }

    /// Check if we should stop early based on quality metrics
    fn should_stop_early(&self, results: &SearchResults, current_depth: usize) -> bool {
        if current_depth == 0 {
            return false; // Always explore at least 2 depths
        }

        // Calculate quality metric: chunks per concept explored
        let quality = if results.total_concepts_explored > 0 {
            results.total_chunks as f64 / results.total_concepts_explored as f64
        } else {
            0.0
        };

        quality >= self.config.adaptive_quality_threshold
    }

    /// Normalize concept for entity ID
    fn normalize_concept(&self, concept: &str) -> String {
        concept
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .collect::<String>()
            .replace(' ', "_")
    }
}

/// Search results from iterative deepening
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    /// Original query
    pub query: String,

    /// Depth level reached
    pub depth_reached: usize,

    /// Total chunks found
    pub total_chunks: usize,

    /// Total concepts explored
    pub total_concepts_explored: usize,

    /// Results per depth level
    pub depth_results: Vec<DepthResults>,

    /// All chunk IDs found
    pub chunk_ids: Vec<ChunkId>,

    /// Reason for stopping search
    pub stop_reason: StopReason,
}

impl SearchResults {
    fn new(query: String) -> Self {
        Self {
            query,
            depth_reached: 0,
            total_chunks: 0,
            total_concepts_explored: 0,
            depth_results: Vec::new(),
            chunk_ids: Vec::new(),
            stop_reason: StopReason::MaxDepthReached,
        }
    }

    fn add_depth_results(&mut self, depth: usize, results: DepthResults) {
        self.depth_results.push(results);
    }

    /// Get chunk count
    pub fn chunk_count(&self) -> usize {
        self.total_chunks
    }

    /// Get concept count
    pub fn concept_count(&self) -> usize {
        self.total_concepts_explored
    }

    /// Get results for a specific depth
    pub fn get_depth_results(&self, depth: usize) -> Option<&DepthResults> {
        self.depth_results.iter().find(|r| r.depth == depth)
    }
}

/// Results for a specific depth level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthResults {
    /// Depth level
    pub depth: usize,

    /// Number of concepts explored at this depth
    pub concepts_explored: usize,

    /// New chunks found at this depth (not seen in previous depths)
    pub new_chunks_found: usize,

    /// Chunk IDs found at this depth
    pub chunk_ids: Vec<ChunkId>,
}

/// Reason for stopping the search
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    /// Reached maximum depth
    MaxDepthReached,

    /// Reached maximum number of chunks
    MaxChunksReached,

    /// Met quality threshold (adaptive stopping)
    QualityThresholdMet,

    /// No more related concepts to explore
    NoMoreConcepts,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lightrag::concept_graph::{ConceptGraphBuilder, ConceptExtractor};

    #[test]
    fn test_iterative_deepening_basic() {
        let config = SearchConfig {
            max_depth: 2,
            min_chunks: 1,
            max_chunks: 10,
            concepts_per_depth: 3,
            use_adaptive_depth: false,
            adaptive_quality_threshold: 0.7,
        };

        let search = IterativeDeepeningSearch::new(config);

        // Create test data
        let mut builder = ConceptGraphBuilder::new();
        builder.add_document_concepts("doc1", vec![
            "machine".to_string(),
            "learning".to_string(),
        ]);
        builder.add_chunk_concepts("chunk1", vec!["machine".to_string()]);

        let concept_graph = builder.build();
        let bidirectional_index = BidirectionalIndex::new();

        let results = search.search("machine", &concept_graph, &bidirectional_index);

        assert_eq!(results.query, "machine");
        assert!(results.depth_reached <= 2);
    }

    #[test]
    fn test_search_config_default() {
        let config = SearchConfig::default();

        assert_eq!(config.max_depth, 3);
        assert_eq!(config.min_chunks, 5);
        assert!(config.use_adaptive_depth);
    }

    #[test]
    fn test_stop_reasons() {
        assert_ne!(StopReason::MaxDepthReached, StopReason::QualityThresholdMet);
        assert_eq!(StopReason::NoMoreConcepts, StopReason::NoMoreConcepts);
    }
}
