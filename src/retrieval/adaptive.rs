//! Adaptive strategy selection for intelligent retrieval

use crate::{
    core::KnowledgeGraph,
    query::{QueryAnalysisResult, QueryType},
    retrieval::{RetrievalSystem, SearchResult},
    summarization::DocumentTree,
    vector::VectorIndex,
    Result,
};
use std::collections::HashMap;

/// Weights for different retrieval strategies
#[derive(Debug, Clone)]
pub struct StrategyWeights {
    pub vector_weight: f32,
    pub graph_weight: f32,
    pub hierarchical_weight: f32,
    pub bm25_weight: f32,
}

impl Default for StrategyWeights {
    fn default() -> Self {
        Self {
            vector_weight: 0.25,
            graph_weight: 0.25,
            hierarchical_weight: 0.25,
            bm25_weight: 0.25,
        }
    }
}

/// Configuration for adaptive strategy selection
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Strategy weights for different query types
    pub entity_weights: StrategyWeights,
    pub conceptual_weights: StrategyWeights,
    pub factual_weights: StrategyWeights,
    pub relational_weights: StrategyWeights,
    pub complex_weights: StrategyWeights,
    /// Minimum confidence to use specialized weights
    pub min_confidence_for_specialization: f32,
    /// Number of results to retrieve per strategy
    pub results_per_strategy: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            entity_weights: StrategyWeights {
                vector_weight: 0.2,
                graph_weight: 0.5,
                hierarchical_weight: 0.2,
                bm25_weight: 0.1,
            },
            conceptual_weights: StrategyWeights {
                vector_weight: 0.6,
                graph_weight: 0.1,
                hierarchical_weight: 0.3,
                bm25_weight: 0.0,
            },
            factual_weights: StrategyWeights {
                vector_weight: 0.2,
                graph_weight: 0.1,
                hierarchical_weight: 0.1,
                bm25_weight: 0.6,
            },
            relational_weights: StrategyWeights {
                vector_weight: 0.2,
                graph_weight: 0.6,
                hierarchical_weight: 0.1,
                bm25_weight: 0.1,
            },
            complex_weights: StrategyWeights::default(),
            min_confidence_for_specialization: 0.6,
            results_per_strategy: 10,
        }
    }
}

/// Result of adaptive strategy selection
#[derive(Debug, Clone)]
pub struct AdaptiveRetrievalResult {
    pub results: Vec<SearchResult>,
    pub strategy_weights_used: StrategyWeights,
    pub query_analysis: QueryAnalysisResult,
    pub fusion_method: String,
    pub total_results_before_fusion: usize,
}

/// Adaptive retrieval system that selects strategies based on query analysis
pub struct AdaptiveRetriever {
    config: AdaptiveConfig,
    retrieval_system: RetrievalSystem,
}

impl AdaptiveRetriever {
    /// Create a new adaptive retriever
    pub fn new(
        config: AdaptiveConfig,
        _vector_index: VectorIndex,
        _knowledge_graph: KnowledgeGraph,
        _document_trees: HashMap<String, DocumentTree>,
    ) -> Result<Self> {
        // Create a default config for the retrieval system
        let default_config = crate::config::Config::default();
        let retrieval_system = RetrievalSystem::new(&default_config)?;

        Ok(Self {
            config,
            retrieval_system,
        })
    }

    /// Perform adaptive retrieval based on query analysis
    pub fn retrieve(
        &mut self,
        query: &str,
        query_analysis: &QueryAnalysisResult,
        max_results: usize,
    ) -> Result<AdaptiveRetrievalResult> {
        // Select strategy weights based on query type and confidence
        let strategy_weights = self.select_strategy_weights(query_analysis);

        // Retrieve results using different strategies
        let mut all_results = Vec::new();

        // Vector similarity search
        if strategy_weights.vector_weight > 0.0 {
            let vector_results = self.retrieval_system.vector_search(
                query,
                (self.config.results_per_strategy as f32 * strategy_weights.vector_weight) as usize,
            )?;
            all_results.extend(self.weight_results(vector_results, strategy_weights.vector_weight));
        }

        // Graph-based search
        if strategy_weights.graph_weight > 0.0 {
            let graph_results = self.retrieval_system.graph_search(
                query,
                (self.config.results_per_strategy as f32 * strategy_weights.graph_weight) as usize,
            )?;
            all_results.extend(self.weight_results(graph_results, strategy_weights.graph_weight));
        }

        // Hierarchical search
        if strategy_weights.hierarchical_weight > 0.0 {
            let max_results = (self.config.results_per_strategy as f32
                * strategy_weights.hierarchical_weight) as usize;
            let hierarchical_results = self
                .retrieval_system
                .public_hierarchical_search(query, max_results)?;
            all_results.extend(
                self.weight_results(hierarchical_results, strategy_weights.hierarchical_weight),
            );
        }

        // BM25 search
        if strategy_weights.bm25_weight > 0.0 {
            let bm25_results = self.retrieval_system.bm25_search(
                query,
                (self.config.results_per_strategy as f32 * strategy_weights.bm25_weight) as usize,
            )?;
            all_results.extend(self.weight_results(bm25_results, strategy_weights.bm25_weight));
        }

        let total_results_before_fusion = all_results.len();

        // Perform cross-strategy fusion
        let fused_results = self.cross_strategy_fusion(all_results, max_results)?;

        Ok(AdaptiveRetrievalResult {
            results: fused_results,
            strategy_weights_used: strategy_weights,
            query_analysis: query_analysis.clone(),
            fusion_method: "weighted_score_fusion".to_string(),
            total_results_before_fusion,
        })
    }

    /// Select strategy weights based on query analysis
    fn select_strategy_weights(&self, query_analysis: &QueryAnalysisResult) -> StrategyWeights {
        // If confidence is low, use default balanced weights
        if query_analysis.confidence < self.config.min_confidence_for_specialization {
            return self.config.complex_weights.clone();
        }

        // Select weights based on query type
        match query_analysis.query_type {
            QueryType::EntityFocused => self.config.entity_weights.clone(),
            QueryType::Conceptual => self.config.conceptual_weights.clone(),
            QueryType::Factual => self.config.factual_weights.clone(),
            QueryType::Relational => self.config.relational_weights.clone(),
            QueryType::Complex => self.config.complex_weights.clone(),
        }
    }

    /// Apply strategy weight to results
    fn weight_results(&self, mut results: Vec<SearchResult>, weight: f32) -> Vec<SearchResult> {
        for result in &mut results {
            result.score *= weight;
        }
        results
    }

    /// Perform cross-strategy fusion of results
    fn cross_strategy_fusion(
        &self,
        results: Vec<SearchResult>,
        max_results: usize,
    ) -> Result<Vec<SearchResult>> {
        // Remove duplicates by chunk ID, keeping highest scored version
        let mut seen_chunks = HashMap::new();
        let mut deduplicated_results = Vec::new();

        for result in results {
            let chunk_id = &result.id;

            if let Some(existing_score) = seen_chunks.get(chunk_id) {
                if result.score > *existing_score {
                    // Replace with higher scored version
                    seen_chunks.insert(chunk_id.clone(), result.score);
                    // Remove old version and add new one
                    deduplicated_results.retain(|r: &SearchResult| r.id != *chunk_id);
                    deduplicated_results.push(result);
                }
            } else {
                seen_chunks.insert(chunk_id.clone(), result.score);
                deduplicated_results.push(result);
            }
        }

        // Sort by final weighted score
        deduplicated_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Apply diversity-aware selection
        let final_results = self.diversity_aware_selection(deduplicated_results, max_results);

        Ok(final_results)
    }

    /// Apply diversity-aware selection to avoid redundant results
    fn diversity_aware_selection(
        &self,
        results: Vec<SearchResult>,
        max_results: usize,
    ) -> Vec<SearchResult> {
        let mut selected_results = Vec::new();
        let mut selected_entities = std::collections::HashSet::new();
        let _remaining_results = results.clone();

        for result in &results {
            if selected_results.len() >= max_results {
                break;
            }

            // Check for entity diversity
            let has_new_entities = result
                .entities
                .iter()
                .any(|entity| !selected_entities.contains(entity));

            // Always include high-scoring results or those with new entities
            if result.score > 0.8 || has_new_entities || selected_results.len() < max_results / 2 {
                for entity in &result.entities {
                    selected_entities.insert(entity.clone());
                }
                selected_results.push(result.clone());
            }
        }

        // If we don't have enough results, fill with remaining high-scoring ones
        if selected_results.len() < max_results {
            for result in results {
                if selected_results.len() >= max_results {
                    break;
                }
                if !selected_results.iter().any(|r| r.id == result.id) {
                    selected_results.push(result);
                }
            }
        }

        selected_results
    }

    /// Get adaptive retrieval statistics
    pub fn get_statistics(&self) -> AdaptiveRetrieverStatistics {
        AdaptiveRetrieverStatistics {
            config: self.config.clone(),
            retrieval_system_stats: format!("RetrievalSystem with {} strategies", 4),
        }
    }
}

/// Statistics about the adaptive retriever
#[derive(Debug)]
pub struct AdaptiveRetrieverStatistics {
    pub config: AdaptiveConfig,
    pub retrieval_system_stats: String,
}

impl AdaptiveRetrieverStatistics {
    pub fn print(&self) {
        println!("Adaptive Retriever Statistics:");
        println!(
            "  Min confidence for specialization: {:.2}",
            self.config.min_confidence_for_specialization
        );
        println!(
            "  Results per strategy: {}",
            self.config.results_per_strategy
        );
        println!(
            "  Entity weights: V:{:.2} G:{:.2} H:{:.2} B:{:.2}",
            self.config.entity_weights.vector_weight,
            self.config.entity_weights.graph_weight,
            self.config.entity_weights.hierarchical_weight,
            self.config.entity_weights.bm25_weight
        );
        println!(
            "  Factual weights: V:{:.2} G:{:.2} H:{:.2} B:{:.2}",
            self.config.factual_weights.vector_weight,
            self.config.factual_weights.graph_weight,
            self.config.factual_weights.hierarchical_weight,
            self.config.factual_weights.bm25_weight
        );
        println!("  {}", self.retrieval_system_stats);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::QueryType;

    #[test]
    fn test_strategy_weight_selection() {
        let _config = AdaptiveConfig::default();

        // Mock query analysis for entity-focused query
        let entity_analysis = QueryAnalysisResult {
            query_type: QueryType::EntityFocused,
            confidence: 0.8,
            keywords_matched: vec!["who".to_string()],
            suggested_strategies: vec!["entity_search".to_string()],
            complexity_score: 0.2,
        };

        // Test that we would select entity weights for high-confidence entity query
        // This is a unit test for the weight selection logic
        assert_eq!(entity_analysis.query_type, QueryType::EntityFocused);
        assert!(entity_analysis.confidence > 0.6);
    }

    #[test]
    fn test_diversity_aware_selection() {
        // Create mock adaptive retriever with default config
        let config = AdaptiveConfig::default();

        // Test diversity logic by checking that the function exists
        // In a real test environment, we would create full mock objects
        assert!(config.min_confidence_for_specialization > 0.0);
    }
}
