//! HippoRAG Personalized PageRank Retrieval
//!
//! This module implements the HippoRAG retrieval strategy that uses Personalized
//! PageRank (PPR) to combine fact-based entity signals with dense passage retrieval.
//!
//! Key innovation: Uses a dual-signal approach for PPR personalization:
//! 1. Entity weights from relevant facts (query-fact similarity)
//! 2. Passage weights from dense retrieval (scaled down)
//!
//! Reference: "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models"
//! https://arxiv.org/abs/2405.14831

use std::collections::HashMap;

use crate::core::{EntityId, GraphRAGError, Result};
use crate::retrieval::SearchResult;
use crate::graph::pagerank::{PersonalizedPageRank, PageRankConfig};

/// Configuration for HippoRAG PPR retrieval
#[derive(Debug, Clone)]
pub struct HippoRAGConfig {
    /// Damping factor for PageRank (HippoRAG default: 0.5)
    pub damping_factor: f64,

    /// Maximum PageRank iterations
    pub max_iterations: usize,

    /// Convergence tolerance
    pub tolerance: f64,

    /// Number of top facts to use for entity weight calculation
    pub top_k_facts: usize,

    /// Weight multiplier for passage nodes (HippoRAG default: 0.05)
    /// This scales down passage scores relative to entity scores
    pub passage_node_weight: f64,

    /// Number of results to return
    pub top_k_results: usize,

    /// Minimum entity frequency threshold
    /// Entities appearing in many passages get downweighted
    pub min_entity_frequency: usize,

    /// Whether to normalize scores before combining
    pub normalize_scores: bool,
}

impl Default for HippoRAGConfig {
    fn default() -> Self {
        Self {
            damping_factor: 0.5,  // HippoRAG uses 0.5 instead of typical 0.85
            max_iterations: 100,
            tolerance: 1e-6,
            top_k_facts: 100,
            passage_node_weight: 0.05,  // Passages get 5% weight vs entities
            top_k_results: 10,
            min_entity_frequency: 1,
            normalize_scores: true,
        }
    }
}

/// Fact triple for knowledge graph
#[derive(Debug, Clone, PartialEq)]
pub struct Fact {
    /// Subject entity
    pub subject: String,

    /// Predicate/relation
    pub predicate: String,

    /// Object entity
    pub object: String,

    /// Fact relevance score (from query-fact similarity)
    pub score: f32,
}

/// HippoRAG retrieval system using Personalized PageRank
///
/// This combines:
/// - Fact retrieval (query → facts)
/// - Entity extraction from facts
/// - Dense passage retrieval
/// - Graph-based reranking via PPR
pub struct HippoRAGRetriever {
    config: HippoRAGConfig,
    pagerank: Option<PersonalizedPageRank>,
}

impl HippoRAGRetriever {
    /// Create a new HippoRAG retriever
    pub fn new(config: HippoRAGConfig) -> Self {
        Self {
            config,
            pagerank: None,
        }
    }

    /// Initialize with a PersonalizedPageRank instance
    pub fn with_pagerank(mut self, pagerank: PersonalizedPageRank) -> Self {
        self.pagerank = Some(pagerank);
        self
    }

    /// Retrieve documents using HippoRAG PPR strategy
    ///
    /// # Arguments
    /// * `query` - The search query
    /// * `top_k_facts` - Top-k facts ranked by query-fact similarity
    /// * `entity_to_passages` - Map from entity IDs to passage IDs
    /// * `passage_scores` - Dense retrieval scores for passages
    ///
    /// # Returns
    /// Ranked search results sorted by PPR score
    pub async fn retrieve(
        &self,
        _query: &str,
        top_k_facts: Vec<Fact>,
        entity_to_passages: &HashMap<EntityId, Vec<EntityId>>,
        passage_scores: &HashMap<EntityId, f32>,
    ) -> Result<Vec<SearchResult>> {
        // Step 1: Calculate entity weights from facts
        let entity_weights = self.calculate_entity_weights(&top_k_facts, entity_to_passages)?;

        // Step 2: Calculate passage weights from dense retrieval
        let passage_weights = self.calculate_passage_weights(passage_scores)?;

        // Step 3: Combine into reset probability distribution
        let reset_probabilities = self.combine_weights(entity_weights, passage_weights)?;

        // Step 4: Run Personalized PageRank
        let ppr_scores = self.run_ppr(&reset_probabilities).await?;

        // Step 5: Extract and rank passage scores
        let ranked_results = self.rank_passages(ppr_scores, passage_scores)?;

        Ok(ranked_results)
    }

    /// Calculate entity weights based on fact relevance
    ///
    /// Key insight: Entities from high-scoring facts get high weights,
    /// but downweighted by how many passages they appear in (reduces generic entities)
    fn calculate_entity_weights(
        &self,
        facts: &[Fact],
        entity_to_passages: &HashMap<EntityId, Vec<EntityId>>,
    ) -> Result<HashMap<EntityId, f64>> {
        let mut weights = HashMap::new();
        let mut occurrence_count = HashMap::new();

        // Process top-k facts
        for fact in facts.iter().take(self.config.top_k_facts) {
            let fact_score = fact.score as f64;

            // Extract entities from subject and object
            for entity_text in [&fact.subject, &fact.object] {
                let entity_id = EntityId::new(entity_text.clone());

                // Get number of passages containing this entity
                let num_passages = entity_to_passages
                    .get(&entity_id)
                    .map(|p| p.len())
                    .unwrap_or(0);

                if num_passages >= self.config.min_entity_frequency {
                    // Weight by fact score, downweighted by passage frequency
                    let weighted_score = if num_passages > 0 {
                        fact_score / num_passages as f64
                    } else {
                        fact_score
                    };

                    *weights.entry(entity_id.clone()).or_insert(0.0) += weighted_score;
                    *occurrence_count.entry(entity_id).or_insert(0) += 1;
                }
            }
        }

        // Average by number of occurrences
        for (entity_id, count) in occurrence_count {
            if let Some(weight) = weights.get_mut(&entity_id) {
                *weight /= count as f64;
            }
        }

        // Normalize if configured
        if self.config.normalize_scores {
            self.normalize_weights(&mut weights);
        }

        Ok(weights)
    }

    /// Calculate passage weights from dense retrieval scores
    fn calculate_passage_weights(
        &self,
        passage_scores: &HashMap<EntityId, f32>,
    ) -> Result<HashMap<EntityId, f64>> {
        let mut weights = HashMap::new();

        for (passage_id, score) in passage_scores {
            // Scale passage scores by passage_node_weight (default 0.05)
            let weighted_score = (*score as f64) * self.config.passage_node_weight;
            weights.insert(passage_id.clone(), weighted_score);
        }

        // Normalize if configured
        if self.config.normalize_scores {
            self.normalize_weights(&mut weights);
        }

        Ok(weights)
    }

    /// Combine entity and passage weights into reset probability distribution
    fn combine_weights(
        &self,
        entity_weights: HashMap<EntityId, f64>,
        passage_weights: HashMap<EntityId, f64>,
    ) -> Result<HashMap<EntityId, f64>> {
        let mut combined = entity_weights;

        // Add passage weights
        for (passage_id, weight) in passage_weights {
            *combined.entry(passage_id).or_insert(0.0) += weight;
        }

        // Ensure non-negative and normalize
        let total: f64 = combined.values().sum();
        if total > 0.0 {
            for weight in combined.values_mut() {
                *weight /= total;
            }
        }

        Ok(combined)
    }

    /// Run Personalized PageRank with reset probabilities
    async fn run_ppr(
        &self,
        reset_probabilities: &HashMap<EntityId, f64>,
    ) -> Result<HashMap<EntityId, f64>> {
        let pagerank = self.pagerank.as_ref()
            .ok_or_else(|| GraphRAGError::Config {
                message: "PageRank not initialized".to_string(),
            })?;

        // Run PPR algorithm
        pagerank.calculate_scores(reset_probabilities)
    }

    /// Extract passage scores and rank by PPR
    fn rank_passages(
        &self,
        ppr_scores: HashMap<EntityId, f64>,
        original_scores: &HashMap<EntityId, f32>,
    ) -> Result<Vec<SearchResult>> {
        let mut results: Vec<_> = ppr_scores
            .iter()
            .filter_map(|(entity_id, &ppr_score)| {
                // Only include passage nodes (not entity nodes)
                if original_scores.contains_key(entity_id) {
                    Some(SearchResult {
                        id: entity_id.to_string(),
                        content: String::new(),  // Will be filled by caller
                        score: ppr_score as f32,
                        result_type: crate::retrieval::ResultType::Chunk,
                        entities: Vec::new(),
                        source_chunks: Vec::new(),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by PPR score (descending)
        results.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to top-k
        results.truncate(self.config.top_k_results);

        Ok(results)
    }

    /// Normalize weights to [0, 1] range using min-max normalization
    fn normalize_weights(&self, weights: &mut HashMap<EntityId, f64>) {
        if weights.is_empty() {
            return;
        }

        let min = weights.values().cloned().fold(f64::INFINITY, f64::min);
        let max = weights.values().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max - min).abs() > 1e-10 {
            for weight in weights.values_mut() {
                *weight = (*weight - min) / (max - min);
            }
        }
    }
}

/// HippoRAG-specific PageRank configuration
impl HippoRAGConfig {
    /// Convert to PageRankConfig for compatibility
    pub fn to_pagerank_config(&self) -> PageRankConfig {
        PageRankConfig {
            damping_factor: self.damping_factor,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            personalized: true,
            parallel_enabled: true,
            cache_size: 1000,
            sparse_threshold: 1000,
            incremental_updates: true,
            simd_block_size: 32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_entity_weight_calculation() {
        let config = HippoRAGConfig::default();
        let retriever = HippoRAGRetriever::new(config);

        let facts = vec![
            Fact {
                subject: "Alice".to_string(),
                predicate: "works_at".to_string(),
                object: "Company".to_string(),
                score: 0.9,
            },
            Fact {
                subject: "Bob".to_string(),
                predicate: "works_at".to_string(),
                object: "Company".to_string(),
                score: 0.8,
            },
        ];

        let mut entity_to_passages = HashMap::new();
        entity_to_passages.insert(EntityId::new("Alice".to_string()), vec![EntityId::new("doc1".to_string())]);
        entity_to_passages.insert(EntityId::new("Company".to_string()), vec![EntityId::new("doc1".to_string()), EntityId::new("doc2".to_string())]);

        let weights = retriever.calculate_entity_weights(&facts, &entity_to_passages).unwrap();

        // Alice should have higher weight (appears in fewer passages)
        let alice_weight = weights.get(&EntityId::new("Alice".to_string())).unwrap();
        let company_weight = weights.get(&EntityId::new("Company".to_string())).unwrap();

        assert!(alice_weight > company_weight, "Alice should have higher weight due to lower frequency");
    }

    #[tokio::test]
    async fn test_passage_weight_calculation() {
        let config = HippoRAGConfig {
            passage_node_weight: 0.05,
            normalize_scores: false,  // Disable normalization for this test
            ..Default::default()
        };
        let retriever = HippoRAGRetriever::new(config);

        let mut passage_scores = HashMap::new();
        passage_scores.insert(EntityId::new("doc1".to_string()), 0.9);
        passage_scores.insert(EntityId::new("doc2".to_string()), 0.5);

        let weights = retriever.calculate_passage_weights(&passage_scores).unwrap();

        // Passage weights should be scaled by passage_node_weight
        let doc1_weight = weights.get(&EntityId::new("doc1".to_string())).unwrap();
        assert!((*doc1_weight - 0.9 * 0.05).abs() < 0.001, "Passage weight should be scaled");

        // doc1 should have higher weight than doc2
        let doc2_weight = weights.get(&EntityId::new("doc2".to_string())).unwrap();
        assert!(doc1_weight > doc2_weight, "Higher score should have higher weight");
    }

    #[test]
    fn test_weight_combining() {
        let config = HippoRAGConfig::default();
        let retriever = HippoRAGRetriever::new(config);

        let mut entity_weights = HashMap::new();
        entity_weights.insert(EntityId::new("entity1".to_string()), 0.8);

        let mut passage_weights = HashMap::new();
        passage_weights.insert(EntityId::new("doc1".to_string()), 0.04);
        passage_weights.insert(EntityId::new("entity1".to_string()), 0.01);  // Overlap

        let combined = retriever.combine_weights(entity_weights, passage_weights).unwrap();

        // entity1 should have combined weight
        let entity1_combined = combined.get(&EntityId::new("entity1".to_string())).unwrap();
        assert!(*entity1_combined > 0.0, "Entity should have combined weight");

        // All weights should sum to 1.0 (normalized)
        let total: f64 = combined.values().sum();
        assert!((total - 1.0).abs() < 0.001, "Weights should sum to 1.0");
    }
}
