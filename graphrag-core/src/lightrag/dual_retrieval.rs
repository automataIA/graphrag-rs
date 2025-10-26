//! Dual-level retrieval implementation
//!
//! Retrieves at two levels simultaneously:
//! 1. High-level: Topic/community-level retrieval
//! 2. Low-level: Entity-level retrieval
//!
//! Then merges and deduplicates results for optimal context.

use std::collections::HashSet;
use std::sync::Arc;

use crate::core::error::GraphRAGError;
use crate::lightrag::keyword_extraction::{KeywordExtractor, DualLevelKeywords};
use crate::retrieval::SearchResult;

/// Dual-level retrieval results
#[derive(Debug, Clone)]
pub struct DualRetrievalResults {
    /// Results from high-level (topic) retrieval
    pub high_level_chunks: Vec<SearchResult>,

    /// Results from low-level (entity) retrieval
    pub low_level_chunks: Vec<SearchResult>,

    /// Merged and deduplicated results
    pub merged_chunks: Vec<SearchResult>,

    /// Keywords used for retrieval
    pub keywords: DualLevelKeywords,
}

/// Configuration for dual-level retrieval
#[derive(Debug, Clone)]
pub struct DualRetrievalConfig {
    /// Weight for high-level results (0.0-1.0)
    pub high_level_weight: f32,

    /// Weight for low-level results (0.0-1.0)
    pub low_level_weight: f32,

    /// How to merge results
    pub merge_strategy: MergeStrategy,
}

impl Default for DualRetrievalConfig {
    fn default() -> Self {
        Self {
            high_level_weight: 0.6,   // Favor topics slightly
            low_level_weight: 0.4,    // But still consider entities
            merge_strategy: MergeStrategy::Interleave,
        }
    }
}

/// Strategy for merging high and low level results
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MergeStrategy {
    /// Interleave results alternating between levels
    Interleave,

    /// Concatenate high-level first, then low-level
    HighFirst,

    /// Concatenate low-level first, then high-level
    LowFirst,

    /// Score-based weighted merge
    Weighted,
}

/// Semantic search interface that combines embedder and vector store
#[async_trait::async_trait]
pub trait SemanticSearcher: Send + Sync {
    /// Search using a text query
    async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>, GraphRAGError>;
}

/// LightRAG-style dual-level retriever
pub struct DualLevelRetriever {
    keyword_extractor: Arc<KeywordExtractor>,
    high_level_store: Arc<dyn SemanticSearcher>,  // Community/topic search
    low_level_store: Arc<dyn SemanticSearcher>,   // Entity/chunk search
    config: DualRetrievalConfig,
}

impl DualLevelRetriever {
    /// Create a new dual-level retriever
    pub fn new(
        keyword_extractor: Arc<KeywordExtractor>,
        high_level_store: Arc<dyn SemanticSearcher>,
        low_level_store: Arc<dyn SemanticSearcher>,
        config: DualRetrievalConfig,
    ) -> Self {
        Self {
            keyword_extractor,
            high_level_store,
            low_level_store,
            config,
        }
    }

    /// Main dual-level retrieval function
    pub async fn retrieve(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<DualRetrievalResults, GraphRAGError> {
        // 1. Extract dual-level keywords
        let keywords = self.keyword_extractor.extract_with_fallback(query).await?;

        log::debug!(
            "Dual-level keywords - High: {:?}, Low: {:?}",
            keywords.high_level,
            keywords.low_level
        );

        // 2. Parallel retrieval at both levels
        let (high_results, low_results) = tokio::join!(
            self.retrieve_high_level(&keywords.high_level, top_k),
            self.retrieve_low_level(&keywords.low_level, top_k)
        );

        let high_level_chunks = high_results?;
        let low_level_chunks = low_results?;

        // 3. Merge and deduplicate
        let merged_chunks = self.merge_results(
            &high_level_chunks,
            &low_level_chunks,
            top_k,
        )?;

        log::info!(
            "Dual retrieval: {} high-level, {} low-level → {} merged",
            high_level_chunks.len(),
            low_level_chunks.len(),
            merged_chunks.len()
        );

        Ok(DualRetrievalResults {
            high_level_chunks,
            low_level_chunks,
            merged_chunks,
            keywords,
        })
    }

    /// Retrieve using high-level keywords (topics, themes, communities)
    async fn retrieve_high_level(
        &self,
        keywords: &[String],
        top_k: usize,
    ) -> Result<Vec<SearchResult>, GraphRAGError> {
        if keywords.is_empty() {
            log::debug!("No high-level keywords, skipping high-level retrieval");
            return Ok(Vec::new());
        }

        // Combine keywords into query
        let combined_query = keywords.join(" ");

        log::debug!("High-level query: '{}'", combined_query);

        // Search in topic-level index (community summaries, abstracts)
        let results = self.high_level_store
            .search(&combined_query, top_k)
            .await?;

        Ok(results)
    }

    /// Retrieve using low-level keywords (entities, specifics)
    async fn retrieve_low_level(
        &self,
        keywords: &[String],
        top_k: usize,
    ) -> Result<Vec<SearchResult>, GraphRAGError> {
        if keywords.is_empty() {
            log::debug!("No low-level keywords, skipping low-level retrieval");
            return Ok(Vec::new());
        }

        // Combine keywords
        let combined_query = keywords.join(" ");

        log::debug!("Low-level query: '{}'", combined_query);

        // Search in entity-level index (chunks, entities)
        let results = self.low_level_store
            .search(&combined_query, top_k)
            .await?;

        Ok(results)
    }

    /// Merge results from both levels
    fn merge_results(
        &self,
        high: &[SearchResult],
        low: &[SearchResult],
        top_k: usize,
    ) -> Result<Vec<SearchResult>, GraphRAGError> {
        match self.config.merge_strategy {
            MergeStrategy::Interleave => self.merge_interleave(high, low, top_k),
            MergeStrategy::HighFirst => self.merge_concat(high, low, top_k),
            MergeStrategy::LowFirst => self.merge_concat(low, high, top_k),
            MergeStrategy::Weighted => self.merge_weighted(high, low, top_k),
        }
    }

    /// Interleave results alternating between high and low
    fn merge_interleave(
        &self,
        high: &[SearchResult],
        low: &[SearchResult],
        top_k: usize,
    ) -> Result<Vec<SearchResult>, GraphRAGError> {
        let mut seen_ids = HashSet::new();
        let mut merged = Vec::new();

        let mut high_iter = high.iter();
        let mut low_iter = low.iter();
        let mut use_high = true;

        while merged.len() < top_k {
            let chunk = if use_high {
                high_iter.next()
            } else {
                low_iter.next()
            };

            match chunk {
                Some(c) => {
                    if seen_ids.insert(c.id.clone()) {
                        merged.push(c.clone());
                    }
                }
                None => {
                    // This source exhausted, continue with the other
                    if high_iter.len() == 0 && low_iter.len() == 0 {
                        break;
                    }
                }
            }

            use_high = !use_high;
        }

        Ok(merged)
    }

    /// Concatenate results (first, then second)
    fn merge_concat(
        &self,
        first: &[SearchResult],
        second: &[SearchResult],
        top_k: usize,
    ) -> Result<Vec<SearchResult>, GraphRAGError> {
        let mut seen_ids = HashSet::new();
        let mut merged = Vec::new();

        // Add from first source
        for chunk in first {
            if merged.len() >= top_k {
                break;
            }
            if seen_ids.insert(chunk.id.clone()) {
                merged.push(chunk.clone());
            }
        }

        // Add from second source
        for chunk in second {
            if merged.len() >= top_k {
                break;
            }
            if seen_ids.insert(chunk.id.clone()) {
                merged.push(chunk.clone());
            }
        }

        Ok(merged)
    }

    /// Weighted merge based on scores and configured weights
    fn merge_weighted(
        &self,
        high: &[SearchResult],
        low: &[SearchResult],
        top_k: usize,
    ) -> Result<Vec<SearchResult>, GraphRAGError> {
        let mut all_chunks: Vec<(SearchResult, f32)> = Vec::new();

        // Add high-level with weight
        for chunk in high {
            let weighted_score = chunk.score * self.config.high_level_weight;
            all_chunks.push((chunk.clone(), weighted_score));
        }

        // Add low-level with weight
        for chunk in low {
            let weighted_score = chunk.score * self.config.low_level_weight;
            all_chunks.push((chunk.clone(), weighted_score));
        }

        // Sort by weighted score
        all_chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Deduplicate and take top_k
        let mut seen_ids = HashSet::new();
        let merged: Vec<SearchResult> = all_chunks
            .into_iter()
            .filter_map(|(chunk, _score)| {
                if seen_ids.insert(chunk.id.clone()) {
                    Some(chunk)
                } else {
                    None
                }
            })
            .take(top_k)
            .collect();

        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::ResultType;

    fn create_test_result(id: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            content: format!("Content of {}", id),
            score,
            result_type: ResultType::Chunk,
            entities: Vec::new(),
            source_chunks: Vec::new(),
        }
    }

    // Note: Full integration tests would require mock implementations
    // of SemanticSearcher and AsyncLanguageModel, which are better
    // suited for the main graphrag-rs crate's test suite.

    #[test]
    fn test_merge_strategies_basic() {
        let config = DualRetrievalConfig::default();

        // Test that merge strategies are properly defined
        assert_eq!(config.merge_strategy, MergeStrategy::Interleave);
        assert!(config.high_level_weight > 0.0);
        assert!(config.low_level_weight > 0.0);
    }

    #[test]
    fn test_search_result_creation() {
        let result = create_test_result("test_1", 0.95);
        assert_eq!(result.id, "test_1");
        assert_eq!(result.score, 0.95);
        assert_eq!(result.result_type, ResultType::Chunk);
    }
}
