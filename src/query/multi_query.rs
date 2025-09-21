use crate::{
    query::expansion::QueryExpander,
    retrieval::hybrid::{HybridRetriever, HybridSearchResult},
    Result,
};
use std::collections::{HashMap, HashSet};

/// Configuration for multi-query retrieval
#[derive(Debug, Clone)]
pub struct MultiQueryConfig {
    /// Maximum number of results per individual query
    pub max_results_per_query: usize,
    /// Final number of results to return after fusion
    pub final_result_limit: usize,
    /// Score aggregation method
    pub aggregation_method: ScoreAggregation,
    /// Diversity threshold to avoid too similar results
    pub diversity_threshold: f32,
    /// Weight decay for lower-ranked expansions
    pub expansion_weight_decay: f32,
}

impl Default for MultiQueryConfig {
    fn default() -> Self {
        Self {
            max_results_per_query: 20,
            final_result_limit: 10,
            aggregation_method: ScoreAggregation::WeightedAverage,
            diversity_threshold: 0.8,
            expansion_weight_decay: 0.9,
        }
    }
}

/// Methods for aggregating scores across multiple queries
#[derive(Debug, Clone, PartialEq)]
pub enum ScoreAggregation {
    /// Take maximum score across all queries
    Maximum,
    /// Take average score across all queries
    Average,
    /// Weighted average using query expansion weights
    WeightedAverage,
    /// Sum of scores
    Sum,
    /// Weighted sum using query expansion weights
    WeightedSum,
}

/// Multi-query search result with aggregated information
#[derive(Debug, Clone)]
pub struct MultiQueryResult {
    /// Unique identifier for the result
    pub id: String,
    /// Content of the result
    pub content: String,
    /// Final aggregated score
    pub score: f32,
    /// Individual scores from each query that found this result
    pub query_scores: HashMap<String, f32>,
    /// Number of queries that returned this result
    pub query_frequency: usize,
    /// Result type
    pub result_type: crate::retrieval::ResultType,
    /// Associated entities
    pub entities: Vec<String>,
    /// Source chunks
    pub source_chunks: Vec<String>,
    /// Queries that found this result
    pub found_by_queries: Vec<String>,
}

/// Multi-query retrieval system that expands queries and aggregates results
pub struct MultiQueryRetriever {
    /// Query expander for generating query variations
    expander: QueryExpander,
    /// Hybrid retriever for individual query execution
    hybrid_retriever: HybridRetriever,
    /// Configuration for multi-query retrieval
    config: MultiQueryConfig,
}

impl MultiQueryRetriever {
    /// Create a new multi-query retriever
    pub fn new(expander: QueryExpander, hybrid_retriever: HybridRetriever) -> Self {
        Self::with_config(expander, hybrid_retriever, MultiQueryConfig::default())
    }

    /// Create a new multi-query retriever with custom configuration
    pub fn with_config(
        expander: QueryExpander,
        hybrid_retriever: HybridRetriever,
        config: MultiQueryConfig,
    ) -> Self {
        Self {
            expander,
            hybrid_retriever,
            config,
        }
    }

    /// Perform multi-query search
    pub fn search(&mut self, query: &str, limit: usize) -> Result<Vec<MultiQueryResult>> {
        // 1. Expand the original query
        let expanded_queries = self.expander.expand_query(query);

        // 2. Execute all expanded queries
        let mut all_results: HashMap<String, Vec<(HybridSearchResult, String, f32)>> =
            HashMap::new();

        for (query_idx, expanded_query) in expanded_queries.iter().enumerate() {
            // Apply weight decay for lower-ranked expansions
            let query_weight =
                expanded_query.weight * self.config.expansion_weight_decay.powi(query_idx as i32);

            match self
                .hybrid_retriever
                .search(&expanded_query.text, self.config.max_results_per_query)
            {
                Ok(results) => {
                    for result in results {
                        all_results.entry(result.id.clone()).or_default().push((
                            result,
                            expanded_query.text.clone(),
                            query_weight,
                        ));
                    }
                }
                Err(e) => {
                    eprintln!("Warning: Query '{}' failed: {}", expanded_query.text, e);
                    continue;
                }
            }
        }

        // 3. Aggregate and rank results
        let aggregated_results = self.aggregate_results(all_results)?;

        // 4. Apply diversity filtering and limit results
        let final_results = self.apply_diversity_filtering(aggregated_results, limit);

        Ok(final_results)
    }

    /// Aggregate results from multiple queries
    fn aggregate_results(
        &self,
        all_results: HashMap<String, Vec<(HybridSearchResult, String, f32)>>,
    ) -> Result<Vec<MultiQueryResult>> {
        let mut aggregated = Vec::new();

        for (result_id, query_results) in all_results {
            if query_results.is_empty() {
                continue;
            }

            // Use the first result as the base (they should all have the same content)
            let base_result = &query_results[0].0;

            // Calculate aggregated score
            let scores: Vec<f32> = query_results.iter().map(|(r, _, w)| r.score * w).collect();
            let weights: Vec<f32> = query_results.iter().map(|(_, _, w)| *w).collect();
            let queries: Vec<String> = query_results.iter().map(|(_, q, _)| q.clone()).collect();

            let aggregated_score = self.calculate_aggregated_score(&scores, &weights);

            // Create query scores map
            let mut query_scores = HashMap::new();
            for (result, query, weight) in &query_results {
                query_scores.insert(query.clone(), result.score * weight);
            }

            // Collect unique entities and source chunks
            let mut all_entities: HashSet<String> = HashSet::new();
            let mut all_source_chunks: HashSet<String> = HashSet::new();

            for (result, _, _) in &query_results {
                all_entities.extend(result.entities.iter().cloned());
                all_source_chunks.extend(result.source_chunks.iter().cloned());
            }

            aggregated.push(MultiQueryResult {
                id: result_id,
                content: base_result.content.clone(),
                score: aggregated_score,
                query_scores,
                query_frequency: query_results.len(),
                result_type: base_result.result_type.clone(),
                entities: all_entities.into_iter().collect(),
                source_chunks: all_source_chunks.into_iter().collect(),
                found_by_queries: queries,
            });
        }

        // Sort by aggregated score
        aggregated.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        Ok(aggregated)
    }

    /// Calculate aggregated score based on configuration
    fn calculate_aggregated_score(&self, scores: &[f32], weights: &[f32]) -> f32 {
        if scores.is_empty() {
            return 0.0;
        }

        match self.config.aggregation_method {
            ScoreAggregation::Maximum => scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            ScoreAggregation::Average => scores.iter().sum::<f32>() / scores.len() as f32,
            ScoreAggregation::WeightedAverage => {
                let weighted_sum: f32 = scores.iter().zip(weights.iter()).map(|(s, w)| s * w).sum();
                let weight_sum: f32 = weights.iter().sum();
                if weight_sum > 0.0 {
                    weighted_sum / weight_sum
                } else {
                    0.0
                }
            }
            ScoreAggregation::Sum => scores.iter().sum(),
            ScoreAggregation::WeightedSum => {
                scores.iter().zip(weights.iter()).map(|(s, w)| s * w).sum()
            }
        }
    }

    /// Apply diversity filtering to avoid too similar results
    fn apply_diversity_filtering(
        &self,
        mut results: Vec<MultiQueryResult>,
        limit: usize,
    ) -> Vec<MultiQueryResult> {
        let mut filtered_results: Vec<MultiQueryResult> = Vec::new();
        let actual_limit = limit.min(self.config.final_result_limit);

        for result in results.drain(..) {
            let mut is_diverse = true;

            // Check diversity against already selected results
            for existing in &filtered_results {
                let similarity =
                    self.calculate_content_similarity(&result.content, &existing.content);
                if similarity > self.config.diversity_threshold {
                    is_diverse = false;
                    break;
                }
            }

            if is_diverse {
                filtered_results.push(result);
                if filtered_results.len() >= actual_limit {
                    break;
                }
            }
        }

        filtered_results
    }

    /// Calculate content similarity (simplified implementation)
    fn calculate_content_similarity(&self, content1: &str, content2: &str) -> f32 {
        // Simple Jaccard similarity based on words
        let words1: HashSet<&str> = content1.split_whitespace().collect();
        let words2: HashSet<&str> = content2.split_whitespace().collect();

        let intersection_size = words1.intersection(&words2).count();
        let union_size = words1.union(&words2).count();

        if union_size == 0 {
            0.0
        } else {
            intersection_size as f32 / union_size as f32
        }
    }

    /// Get configuration
    pub fn get_config(&self) -> &MultiQueryConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: MultiQueryConfig) {
        self.config = config;
    }

    /// Get query expander
    pub fn get_expander(&self) -> &QueryExpander {
        &self.expander
    }

    /// Get hybrid retriever
    pub fn get_hybrid_retriever(&self) -> &HybridRetriever {
        &self.hybrid_retriever
    }

    /// Get statistics about multi-query retrieval
    pub fn get_statistics(&self) -> MultiQueryStatistics {
        MultiQueryStatistics {
            expander_stats: self.expander.get_statistics(),
            hybrid_stats: self.hybrid_retriever.get_statistics(),
            config: self.config.clone(),
        }
    }

    /// Batch search for multiple queries
    pub fn batch_search(
        &mut self,
        queries: &[&str],
        limit: usize,
    ) -> Result<Vec<Vec<MultiQueryResult>>> {
        queries
            .iter()
            .map(|&query| self.search(query, limit))
            .collect()
    }

    /// Search with detailed timing information
    pub fn search_with_timing(
        &mut self,
        query: &str,
        limit: usize,
    ) -> Result<(Vec<MultiQueryResult>, MultiQueryTiming)> {
        use std::time::Instant;

        let start_time = Instant::now();

        // Time query expansion
        let expansion_start = Instant::now();
        let expanded_queries = self.expander.expand_query(query);
        let expansion_time = expansion_start.elapsed();

        // Time individual query execution
        let retrieval_start = Instant::now();
        let mut all_results: HashMap<String, Vec<(HybridSearchResult, String, f32)>> =
            HashMap::new();

        for (query_idx, expanded_query) in expanded_queries.iter().enumerate() {
            let query_weight =
                expanded_query.weight * self.config.expansion_weight_decay.powi(query_idx as i32);

            match self
                .hybrid_retriever
                .search(&expanded_query.text, self.config.max_results_per_query)
            {
                Ok(results) => {
                    for result in results {
                        all_results.entry(result.id.clone()).or_default().push((
                            result,
                            expanded_query.text.clone(),
                            query_weight,
                        ));
                    }
                }
                Err(_) => continue,
            }
        }
        let retrieval_time = retrieval_start.elapsed();

        // Time aggregation
        let aggregation_start = Instant::now();
        let aggregated_results = self.aggregate_results(all_results)?;
        let final_results = self.apply_diversity_filtering(aggregated_results, limit);
        let aggregation_time = aggregation_start.elapsed();

        let total_time = start_time.elapsed();

        let timing = MultiQueryTiming {
            total_time,
            expansion_time,
            retrieval_time,
            aggregation_time,
            query_count: expanded_queries.len(),
        };

        Ok((final_results, timing))
    }
}

/// Statistics about multi-query retrieval
#[derive(Debug, Clone)]
pub struct MultiQueryStatistics {
    pub expander_stats: crate::query::expansion::ExpansionStatistics,
    pub hybrid_stats: crate::retrieval::hybrid::HybridStatistics,
    pub config: MultiQueryConfig,
}

impl MultiQueryStatistics {
    /// Print statistics
    pub fn print(&self) {
        println!("Multi-Query Retrieval Statistics:");
        println!(
            "  Max results per query: {}",
            self.config.max_results_per_query
        );
        println!("  Final result limit: {}", self.config.final_result_limit);
        println!("  Aggregation method: {:?}", self.config.aggregation_method);
        println!(
            "  Diversity threshold: {:.2}",
            self.config.diversity_threshold
        );
        println!(
            "  Expansion weight decay: {:.2}",
            self.config.expansion_weight_decay
        );
        println!();
        self.expander_stats.print();
        println!();
        self.hybrid_stats.print();
    }
}

/// Timing information for multi-query retrieval
#[derive(Debug, Clone)]
pub struct MultiQueryTiming {
    pub total_time: std::time::Duration,
    pub expansion_time: std::time::Duration,
    pub retrieval_time: std::time::Duration,
    pub aggregation_time: std::time::Duration,
    pub query_count: usize,
}

impl MultiQueryTiming {
    /// Print timing information
    pub fn print(&self) {
        println!("Multi-Query Retrieval Timing:");
        println!("  Total time: {:?}", self.total_time);
        println!("  Query expansion: {:?}", self.expansion_time);
        println!("  Retrieval: {:?}", self.retrieval_time);
        println!("  Aggregation: {:?}", self.aggregation_time);
        println!("  Number of queries: {}", self.query_count);
        println!(
            "  Average time per query: {:?}",
            self.retrieval_time / self.query_count.max(1) as u32
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{query::expansion::QueryExpander, retrieval::hybrid::HybridRetriever};

    #[test]
    fn test_multi_query_config_default() {
        let config = MultiQueryConfig::default();
        assert_eq!(config.max_results_per_query, 20);
        assert_eq!(config.final_result_limit, 10);
        assert_eq!(config.aggregation_method, ScoreAggregation::WeightedAverage);
    }

    #[test]
    fn test_score_aggregation_variants() {
        assert_eq!(ScoreAggregation::Maximum, ScoreAggregation::Maximum);
        assert_ne!(ScoreAggregation::Maximum, ScoreAggregation::Average);
    }

    #[test]
    fn test_multi_query_retriever_creation() {
        let expander = QueryExpander::new();
        let hybrid_retriever = HybridRetriever::new();
        let retriever = MultiQueryRetriever::new(expander, hybrid_retriever);

        let stats = retriever.get_statistics();
        assert_eq!(
            stats.config.aggregation_method,
            ScoreAggregation::WeightedAverage
        );
    }

    #[test]
    fn test_content_similarity() {
        let expander = QueryExpander::new();
        let hybrid_retriever = HybridRetriever::new();
        let retriever = MultiQueryRetriever::new(expander, hybrid_retriever);

        let similarity =
            retriever.calculate_content_similarity("the quick brown fox", "the fast brown fox");

        assert!(similarity > 0.0);
        assert!(similarity < 1.0);
    }

    #[test]
    fn test_score_calculation() {
        let expander = QueryExpander::new();
        let hybrid_retriever = HybridRetriever::new();
        let config = MultiQueryConfig {
            aggregation_method: ScoreAggregation::Average,
            ..Default::default()
        };
        let retriever = MultiQueryRetriever::with_config(expander, hybrid_retriever, config);

        let scores = vec![0.8, 0.6, 0.7];
        let weights = vec![1.0, 0.8, 0.9];

        let avg_score = retriever.calculate_aggregated_score(&scores, &weights);
        assert!((avg_score - 0.7).abs() < 0.01); // Should be approximately 0.7
    }
}
