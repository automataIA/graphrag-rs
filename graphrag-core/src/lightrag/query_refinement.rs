//! Query Refinement for LazyGraphRAG
//!
//! This module implements query expansion and refinement without requiring LLM calls,
//! using the concept graph and bidirectional entity-chunk index for iterative deepening.
//!
//! ## Key Features
//!
//! - **Zero LLM Cost**: Query refinement using graph traversal only
//! - **Iterative Deepening**: Progressively expand query with related concepts
//! - **Fast Lookups**: Uses bidirectional index for instant entity-chunk mapping
//! - **Relevance Scoring**: Ranks refined queries by concept co-occurrence
//!
//! ## Algorithm
//!
//! 1. Extract initial concepts from query
//! 2. Find co-occurring concepts in the concept graph
//! 3. Expand query with top-k related concepts
//! 4. Retrieve relevant chunks using bidirectional index
//! 5. Repeat for N iterations with relevance feedback
//!
//! ## Example
//!
//! ```rust
//! use graphrag_core::lightrag::query_refinement::{QueryRefiner, QueryRefinementConfig};
//! use graphrag_core::lightrag::concept_graph::ConceptGraph;
//! use graphrag_core::entity::BidirectionalIndex;
//!
//! let config = QueryRefinementConfig::default();
//! let refiner = QueryRefiner::new(config);
//!
//! // Refine query using concept graph and index
//! let refined = refiner.refine_query(
//!     "machine learning applications",
//!     &concept_graph,
//!     &bidirectional_index,
//! );
//!
//! println!("Original: {}", refined.original_query);
//! println!("Expanded: {:?}", refined.expanded_concepts);
//! println!("Relevant chunks: {}", refined.relevant_chunk_ids.len());
//! ```

use crate::core::{ChunkId, EntityId};
use crate::entity::BidirectionalIndex;
use crate::lightrag::concept_graph::{ConceptExtractor, ConceptGraph};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Configuration for query refinement
#[derive(Debug, Clone)]
pub struct QueryRefinementConfig {
    /// Maximum number of refinement iterations
    pub max_iterations: usize,

    /// Number of related concepts to expand per iteration
    pub concepts_per_iteration: usize,

    /// Minimum co-occurrence count for concept expansion
    pub min_cooccurrence: usize,

    /// Maximum total concepts in expanded query
    pub max_total_concepts: usize,

    /// Use relevance feedback from previous iterations
    pub use_relevance_feedback: bool,
}

impl Default for QueryRefinementConfig {
    fn default() -> Self {
        Self {
            max_iterations: 3,
            concepts_per_iteration: 5,
            min_cooccurrence: 2,
            max_total_concepts: 20,
            use_relevance_feedback: true,
        }
    }
}

/// Query refiner for LazyGraphRAG
pub struct QueryRefiner {
    config: QueryRefinementConfig,
    concept_extractor: ConceptExtractor,
}

impl QueryRefiner {
    /// Create a new query refiner with configuration
    pub fn new(config: QueryRefinementConfig) -> Self {
        Self {
            config,
            concept_extractor: ConceptExtractor::new(),
        }
    }

    /// Create a query refiner with default configuration
    pub fn default() -> Self {
        Self::new(QueryRefinementConfig::default())
    }

    /// Refine a query using the concept graph and bidirectional index
    ///
    /// This performs iterative deepening to expand the query with related concepts
    /// and identify relevant chunks without requiring LLM calls.
    pub fn refine_query(
        &self,
        query: &str,
        concept_graph: &ConceptGraph,
        bidirectional_index: &BidirectionalIndex,
    ) -> RefinedQuery {
        // Step 1: Extract initial concepts from query
        let initial_concepts = self.concept_extractor.extract_concepts(query);

        if initial_concepts.is_empty() {
            return RefinedQuery {
                original_query: query.to_string(),
                initial_concepts: Vec::new(),
                expanded_concepts: Vec::new(),
                relevant_chunk_ids: Vec::new(),
                iterations: 0,
                relevance_scores: HashMap::new(),
            };
        }

        // Step 2: Iterative refinement
        let mut current_concepts: HashSet<String> = initial_concepts.iter().cloned().collect();
        let mut all_expanded_concepts = Vec::new();
        let mut relevant_chunks: HashSet<ChunkId> = HashSet::new();
        let mut concept_scores: HashMap<String, f64> = HashMap::new();

        // Initialize scores for initial concepts
        for concept in &initial_concepts {
            concept_scores.insert(concept.clone(), 1.0);
        }

        for iteration in 0..self.config.max_iterations {
            // Get related concepts from current concepts
            let mut new_concepts = Vec::new();

            for concept in &current_concepts {
                // Get co-occurring concepts from concept graph
                let related = concept_graph.get_related_concepts(
                    concept,
                    self.config.concepts_per_iteration,
                );

                for related_concept in related {
                    if !current_concepts.contains(&related_concept)
                        && current_concepts.len() < self.config.max_total_concepts
                    {
                        // Score based on graph connectivity
                        let score = self.calculate_concept_relevance(
                            &related_concept,
                            &current_concepts,
                            concept_graph,
                        );

                        if score > 0.0 {
                            concept_scores.insert(related_concept.clone(), score);
                            new_concepts.push(related_concept);
                        }
                    }
                }
            }

            // Add new concepts to current set
            for concept in &new_concepts {
                current_concepts.insert(concept.clone());
                all_expanded_concepts.push(concept.clone());
            }

            // Stop if no new concepts found
            if new_concepts.is_empty() {
                break;
            }

            // Retrieve chunks for current concepts using bidirectional index
            let iteration_chunks = self.get_chunks_for_concepts(
                &new_concepts,
                bidirectional_index,
            );

            // Add to relevant chunks
            relevant_chunks.extend(iteration_chunks);

            // Relevance feedback: boost scores of concepts in retrieved chunks
            if self.config.use_relevance_feedback && iteration < self.config.max_iterations - 1 {
                self.apply_relevance_feedback(
                    &mut concept_scores,
                    &relevant_chunks,
                    bidirectional_index,
                );
            }
        }

        // Sort expanded concepts by relevance score
        let mut expanded_with_scores: Vec<_> = all_expanded_concepts
            .into_iter()
            .map(|c| {
                let score = concept_scores.get(&c).copied().unwrap_or(0.0);
                (c, score)
            })
            .collect();
        expanded_with_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let expanded_concepts: Vec<String> = expanded_with_scores
            .iter()
            .map(|(c, _)| c.clone())
            .collect();

        // Convert relevance scores to HashMap<String, f64>
        let relevance_scores: HashMap<String, f64> = expanded_with_scores.into_iter().collect();

        RefinedQuery {
            original_query: query.to_string(),
            initial_concepts: initial_concepts.clone(),
            expanded_concepts,
            relevant_chunk_ids: relevant_chunks.into_iter().collect(),
            iterations: self.config.max_iterations,
            relevance_scores,
        }
    }

    /// Calculate relevance score for a concept based on graph connectivity
    fn calculate_concept_relevance(
        &self,
        concept: &str,
        current_concepts: &HashSet<String>,
        concept_graph: &ConceptGraph,
    ) -> f64 {
        // Get concept data from graph
        if let Some(concept_data) = concept_graph.concepts.get(concept) {
            // Base score from frequency
            let mut score = (concept_data.frequency as f64).ln() + 1.0;

            // Boost score based on connections to current concepts
            let mut connection_count = 0;
            for current_concept in current_concepts {
                // Check if there's a relationship
                let has_relation = concept_graph.relations.iter().any(|rel| {
                    (rel.source == *concept && rel.target == *current_concept)
                        || (rel.source == *current_concept && rel.target == *concept)
                });

                if has_relation {
                    connection_count += 1;
                }
            }

            // Boost score based on connectivity
            score *= 1.0 + (connection_count as f64 * 0.5);

            score
        } else {
            0.0
        }
    }

    /// Get chunks for a set of concepts using bidirectional index
    fn get_chunks_for_concepts(
        &self,
        concepts: &[String],
        bidirectional_index: &BidirectionalIndex,
    ) -> HashSet<ChunkId> {
        let mut chunks = HashSet::new();

        for concept in concepts {
            // Convert concept to EntityId (using normalized form)
            let entity_id = EntityId::new(self.normalize_concept(concept));

            // Get chunks for this entity
            let entity_chunks = bidirectional_index.get_chunks_for_entity(&entity_id);
            chunks.extend(entity_chunks);
        }

        chunks
    }

    /// Apply relevance feedback to boost concept scores
    fn apply_relevance_feedback(
        &self,
        concept_scores: &mut HashMap<String, f64>,
        relevant_chunks: &HashSet<ChunkId>,
        bidirectional_index: &BidirectionalIndex,
    ) {
        // Get all entities in relevant chunks
        let mut entity_frequencies: HashMap<String, usize> = HashMap::new();

        for chunk_id in relevant_chunks {
            let entities = bidirectional_index.get_entities_for_chunk(chunk_id);

            for entity_id in entities {
                let concept = self.denormalize_entity_id(&entity_id);
                *entity_frequencies.entry(concept).or_insert(0) += 1;
            }
        }

        // Boost scores based on frequency in retrieved chunks
        for (concept, frequency) in entity_frequencies {
            if let Some(score) = concept_scores.get_mut(&concept) {
                // Boost by log frequency
                *score *= 1.0 + (frequency as f64).ln();
            }
        }
    }

    /// Normalize concept for EntityId
    fn normalize_concept(&self, concept: &str) -> String {
        concept
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .collect::<String>()
            .replace(' ', "_")
    }

    /// Denormalize EntityId back to concept
    fn denormalize_entity_id(&self, entity_id: &EntityId) -> String {
        // Extract concept from entity ID (remove prefix if present)
        let id_str = entity_id.as_str();
        id_str
            .split('_')
            .skip_while(|part| part.to_uppercase() == *part) // Skip entity type prefix
            .collect::<Vec<_>>()
            .join("_")
    }
}

/// Result of query refinement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinedQuery {
    /// Original query text
    pub original_query: String,

    /// Initial concepts extracted from query
    pub initial_concepts: Vec<String>,

    /// Expanded concepts from iterative refinement
    pub expanded_concepts: Vec<String>,

    /// Relevant chunk IDs identified during refinement
    pub relevant_chunk_ids: Vec<ChunkId>,

    /// Number of refinement iterations performed
    pub iterations: usize,

    /// Relevance scores for each concept
    pub relevance_scores: HashMap<String, f64>,
}

impl RefinedQuery {
    /// Get all concepts (initial + expanded)
    pub fn all_concepts(&self) -> Vec<String> {
        let mut concepts = self.initial_concepts.clone();
        concepts.extend(self.expanded_concepts.clone());
        concepts
    }

    /// Get top-k concepts by relevance score
    pub fn top_concepts(&self, k: usize) -> Vec<String> {
        let mut concepts_with_scores: Vec<_> = self
            .all_concepts()
            .into_iter()
            .map(|c| {
                let score = self.relevance_scores.get(&c).copied().unwrap_or(0.0);
                (c, score)
            })
            .collect();

        concepts_with_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        concepts_with_scores
            .into_iter()
            .take(k)
            .map(|(c, _)| c)
            .collect()
    }

    /// Get the number of concepts found
    pub fn concept_count(&self) -> usize {
        self.initial_concepts.len() + self.expanded_concepts.len()
    }

    /// Get the number of relevant chunks found
    pub fn chunk_count(&self) -> usize {
        self.relevant_chunk_ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lightrag::concept_graph::{ConceptGraphBuilder, ConceptExtractor};

    #[test]
    fn test_query_refinement_basic() {
        let config = QueryRefinementConfig {
            max_iterations: 2,
            concepts_per_iteration: 3,
            min_cooccurrence: 1,
            max_total_concepts: 10,
            use_relevance_feedback: false,
        };

        let refiner = QueryRefiner::new(config);

        // Create a simple concept graph
        let mut builder = ConceptGraphBuilder::new();
        builder.add_document_concepts("doc1", vec![
            "machine learning".to_string(),
            "neural networks".to_string(),
            "deep learning".to_string(),
        ]);
        builder.add_chunk_concepts("chunk1", vec![
            "machine learning".to_string(),
            "neural networks".to_string(),
        ]);
        builder.add_chunk_concepts("chunk2", vec![
            "neural networks".to_string(),
            "deep learning".to_string(),
        ]);

        let concept_graph = builder.build();
        let bidirectional_index = BidirectionalIndex::new();

        let refined = refiner.refine_query(
            "machine learning",
            &concept_graph,
            &bidirectional_index,
        );

        assert!(!refined.initial_concepts.is_empty());
        assert_eq!(refined.original_query, "machine learning");
    }

    #[test]
    fn test_empty_query() {
        let refiner = QueryRefiner::default();
        let concept_graph = ConceptGraphBuilder::new().build();
        let bidirectional_index = BidirectionalIndex::new();

        let refined = refiner.refine_query("", &concept_graph, &bidirectional_index);

        assert!(refined.initial_concepts.is_empty());
        assert!(refined.expanded_concepts.is_empty());
        assert_eq!(refined.iterations, 0);
    }

    #[test]
    fn test_refined_query_methods() {
        let refined = RefinedQuery {
            original_query: "test query".to_string(),
            initial_concepts: vec!["concept1".to_string(), "concept2".to_string()],
            expanded_concepts: vec!["concept3".to_string()],
            relevant_chunk_ids: vec![
                ChunkId::new("chunk1".to_string()),
                ChunkId::new("chunk2".to_string()),
            ],
            iterations: 2,
            relevance_scores: vec![
                ("concept1".to_string(), 1.0),
                ("concept2".to_string(), 0.8),
                ("concept3".to_string(), 0.6),
            ]
            .into_iter()
            .collect(),
        };

        assert_eq!(refined.concept_count(), 3);
        assert_eq!(refined.chunk_count(), 2);

        let top_2 = refined.top_concepts(2);
        assert_eq!(top_2.len(), 2);
        assert_eq!(top_2[0], "concept1");
    }
}
