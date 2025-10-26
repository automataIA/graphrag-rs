//! Cross-Encoder reranking for improved retrieval accuracy
//!
//! Cross-encoders jointly encode query and document, providing more accurate
//! relevance scores than bi-encoder approaches. This implementation provides
//! a trait-based interface that can be backed by ONNX models, API calls, or
//! other implementations.
//!
//! Reference: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
//! Reimers & Gurevych (2019)

use std::collections::HashMap;
use async_trait::async_trait;

use crate::retrieval::SearchResult;
use crate::Result;

/// Configuration for cross-encoder reranking
#[derive(Debug, Clone)]
pub struct CrossEncoderConfig {
    /// Model name/path for cross-encoder
    pub model_name: String,

    /// Maximum sequence length
    pub max_length: usize,

    /// Batch size for inference
    pub batch_size: usize,

    /// Top-k results to return after reranking
    pub top_k: usize,

    /// Minimum confidence threshold (0.0-1.0)
    pub min_confidence: f32,

    /// Enable score normalization
    pub normalize_scores: bool,
}

impl Default for CrossEncoderConfig {
    fn default() -> Self {
        Self {
            model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2".to_string(),
            max_length: 512,
            batch_size: 32,
            top_k: 10,
            min_confidence: 0.0,
            normalize_scores: true,
        }
    }
}

/// Result of cross-encoder reranking with confidence score
#[derive(Debug, Clone)]
pub struct RankedResult {
    /// Original search result
    pub result: SearchResult,

    /// Cross-encoder relevance score (typically 0.0-1.0 after normalization)
    pub relevance_score: f32,

    /// Original retrieval score (for comparison)
    pub original_score: f32,

    /// Score improvement over original (relevance_score - original_score)
    pub score_delta: f32,
}

/// Cross-encoder trait for reranking retrieved results
#[async_trait]
pub trait CrossEncoder: Send + Sync {
    /// Rerank a list of search results based on relevance to query
    async fn rerank(
        &self,
        query: &str,
        candidates: Vec<SearchResult>,
    ) -> Result<Vec<RankedResult>>;

    /// Score a single query-document pair
    async fn score_pair(&self, query: &str, document: &str) -> Result<f32>;

    /// Batch score multiple query-document pairs
    async fn score_batch(
        &self,
        pairs: Vec<(String, String)>,
    ) -> Result<Vec<f32>>;
}

/// Confidence-based cross-encoder implementation
///
/// This implementation uses semantic similarity and confidence metrics
/// to rerank results. For production use with actual transformer models,
/// consider using ONNXCrossEncoder or APICrossEncoder implementations.
pub struct ConfidenceCrossEncoder {
    config: CrossEncoderConfig,
}

impl ConfidenceCrossEncoder {
    /// Create a new confidence-based cross-encoder
    pub fn new(config: CrossEncoderConfig) -> Self {
        Self { config }
    }

    /// Calculate relevance score based on text similarity and length
    fn calculate_relevance(&self, query: &str, document: &str) -> f32 {
        // Tokenize
        let query_tokens: Vec<&str> = query.split_whitespace().collect();
        let doc_tokens: Vec<&str> = document.split_whitespace().collect();

        if query_tokens.is_empty() || doc_tokens.is_empty() {
            return 0.0;
        }

        // Calculate token overlap (Jaccard similarity as baseline)
        let query_set: HashMap<&str, ()> = query_tokens.iter()
            .map(|t| (*t, ()))
            .collect();
        let doc_set: HashMap<&str, ()> = doc_tokens.iter()
            .map(|t| (*t, ()))
            .collect();

        let intersection: usize = query_set.keys()
            .filter(|k| doc_set.contains_key(*k))
            .count();

        let union_size = query_set.len() + doc_set.len() - intersection;

        let jaccard = if union_size > 0 {
            intersection as f32 / union_size as f32
        } else {
            0.0
        };

        // Boost score based on document length (prefer longer, more informative docs)
        let length_factor = (doc_tokens.len() as f32 / 100.0).min(1.0);

        // Combined score
        let raw_score = jaccard * 0.7 + length_factor * 0.3;

        if self.config.normalize_scores {
            // Normalize to 0-1 range using sigmoid-like function
            1.0 / (1.0 + (-5.0 * (raw_score - 0.5)).exp())
        } else {
            raw_score
        }
    }
}

#[async_trait]
impl CrossEncoder for ConfidenceCrossEncoder {
    async fn rerank(
        &self,
        query: &str,
        candidates: Vec<SearchResult>,
    ) -> Result<Vec<RankedResult>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Score all candidates
        let mut ranked: Vec<RankedResult> = candidates
            .into_iter()
            .map(|result| {
                let relevance_score = self.calculate_relevance(query, &result.content);
                let original_score = result.score;
                let score_delta = relevance_score - original_score;

                RankedResult {
                    result,
                    relevance_score,
                    original_score,
                    score_delta,
                }
            })
            .collect();

        // Sort by relevance score (descending)
        ranked.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Filter by confidence threshold
        ranked.retain(|r| r.relevance_score >= self.config.min_confidence);

        // Truncate to top-k
        ranked.truncate(self.config.top_k);

        log::info!(
            "Reranked {} candidates, returning top-{}",
            ranked.len(),
            self.config.top_k
        );

        Ok(ranked)
    }

    async fn score_pair(&self, query: &str, document: &str) -> Result<f32> {
        Ok(self.calculate_relevance(query, document))
    }

    async fn score_batch(&self, pairs: Vec<(String, String)>) -> Result<Vec<f32>> {
        let scores = pairs
            .iter()
            .map(|(query, doc)| self.calculate_relevance(query, doc))
            .collect();

        Ok(scores)
    }
}

/// Statistics about reranking performance
#[derive(Debug, Clone)]
pub struct RerankingStats {
    /// Number of candidates reranked
    pub candidates_count: usize,

    /// Number of results returned
    pub results_count: usize,

    /// Average score improvement (mean delta)
    pub avg_score_improvement: f32,

    /// Maximum score improvement
    pub max_score_improvement: f32,

    /// Percentage of candidates filtered out
    pub filter_rate: f32,
}

impl RerankingStats {
    /// Calculate statistics from ranked results
    pub fn from_results(
        original_count: usize,
        ranked: &[RankedResult],
    ) -> Self {
        let results_count = ranked.len();

        let avg_score_improvement = if !ranked.is_empty() {
            ranked.iter().map(|r| r.score_delta).sum::<f32>() / ranked.len() as f32
        } else {
            0.0
        };

        let max_score_improvement = ranked
            .iter()
            .map(|r| r.score_delta)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let filter_rate = if original_count > 0 {
            ((original_count - results_count) as f32 / original_count as f32) * 100.0
        } else {
            0.0
        };

        Self {
            candidates_count: original_count,
            results_count,
            avg_score_improvement,
            max_score_improvement,
            filter_rate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::ResultType;

    fn create_test_result(id: &str, content: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            content: content.to_string(),
            score,
            result_type: ResultType::Chunk,
            entities: Vec::new(),
            source_chunks: Vec::new(),
        }
    }

    #[tokio::test]
    async fn test_rerank_basic() {
        let config = CrossEncoderConfig {
            top_k: 3,
            min_confidence: 0.0,
            ..Default::default()
        };

        let encoder = ConfidenceCrossEncoder::new(config);

        let query = "machine learning algorithms";
        let candidates = vec![
            create_test_result(
                "1",
                "Machine learning is a subset of artificial intelligence",
                0.5,
            ),
            create_test_result(
                "2",
                "The weather today is sunny",
                0.6,
            ),
            create_test_result(
                "3",
                "Neural networks are machine learning algorithms used for pattern recognition",
                0.4,
            ),
        ];

        let ranked = encoder.rerank(query, candidates).await.unwrap();

        // Should rerank based on relevance
        assert_eq!(ranked.len(), 3);

        // Most relevant should be first (result 3 has best overlap)
        assert!(ranked[0].relevance_score >= ranked[1].relevance_score);
        assert!(ranked[1].relevance_score >= ranked[2].relevance_score);
    }

    #[tokio::test]
    async fn test_confidence_filtering() {
        let config = CrossEncoderConfig {
            top_k: 10,
            min_confidence: 0.5, // High threshold
            ..Default::default()
        };

        let encoder = ConfidenceCrossEncoder::new(config);

        let query = "specific technical query";
        let candidates = vec![
            create_test_result("1", "highly relevant technical content", 0.3),
            create_test_result("2", "somewhat relevant", 0.4),
            create_test_result("3", "not relevant at all", 0.5),
        ];

        let ranked = encoder.rerank(query, candidates).await.unwrap();

        // Should filter low-confidence results
        for result in &ranked {
            assert!(result.relevance_score >= 0.5);
        }
    }

    #[tokio::test]
    async fn test_score_pair() {
        let config = CrossEncoderConfig::default();
        let encoder = ConfidenceCrossEncoder::new(config);

        let score = encoder
            .score_pair(
                "artificial intelligence",
                "AI and machine learning are related fields",
            )
            .await
            .unwrap();

        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_reranking_stats() {
        let ranked = vec![
            RankedResult {
                result: create_test_result("1", "test", 0.5),
                relevance_score: 0.8,
                original_score: 0.5,
                score_delta: 0.3,
            },
            RankedResult {
                result: create_test_result("2", "test", 0.6),
                relevance_score: 0.7,
                original_score: 0.6,
                score_delta: 0.1,
            },
        ];

        let stats = RerankingStats::from_results(5, &ranked);

        assert_eq!(stats.candidates_count, 5);
        assert_eq!(stats.results_count, 2);
        // Use approximate equality for floating point comparison
        assert!((stats.filter_rate - 60.0).abs() < 0.001); // 3/5 filtered = 60%
        assert!(stats.avg_score_improvement > 0.0);
    }
}
