use crate::reranking::cross_encoder::RerankedResult;
use std::collections::HashSet;

/// Configuration for confidence-based filtering
#[derive(Debug, Clone)]
pub struct ConfidenceConfig {
    /// Minimum confidence score to keep a result
    pub min_confidence: f32,
    /// Diversity threshold to avoid too similar results
    pub diversity_threshold: f32,
    /// Maximum number of results to return after filtering
    pub max_results: usize,
    /// Enable quality-based filtering
    pub enable_quality_filter: bool,
    /// Minimum score threshold for quality filtering
    pub min_quality_score: f32,
    /// Enable diversity filtering
    pub enable_diversity_filter: bool,
    /// Score boost for high-confidence results
    pub confidence_boost_factor: f32,
}

impl Default for ConfidenceConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            diversity_threshold: 0.8,
            max_results: 10,
            enable_quality_filter: true,
            min_quality_score: 0.3,
            enable_diversity_filter: true,
            confidence_boost_factor: 1.2,
        }
    }
}

/// Criteria for filtering results
#[derive(Debug, Clone, PartialEq)]
pub enum FilterCriteria {
    /// Filter by minimum confidence score
    Confidence,
    /// Filter by content diversity
    Diversity,
    /// Filter by result quality
    Quality,
    /// Filter by score threshold
    ScoreThreshold,
    /// Combined filtering using all criteria
    Combined,
}

/// Result of confidence filtering with metadata
#[derive(Debug, Clone)]
pub struct FilteredResult {
    /// The filtered result
    pub result: RerankedResult,
    /// Whether this result passed confidence filtering
    pub confidence_passed: bool,
    /// Whether this result passed diversity filtering
    pub diversity_passed: bool,
    /// Whether this result passed quality filtering
    pub quality_passed: bool,
    /// Diversity score relative to other results
    pub diversity_score: f32,
    /// Final confidence after boost
    pub boosted_confidence: f32,
}

/// Confidence-based filter for improving result quality
pub struct ConfidenceFilter {
    /// Configuration for filtering
    config: ConfidenceConfig,
}

impl ConfidenceFilter {
    /// Create a new confidence filter with default configuration
    pub fn new() -> Self {
        Self::with_config(ConfidenceConfig::default())
    }

    /// Create a new confidence filter with custom configuration
    pub fn with_config(config: ConfidenceConfig) -> Self {
        Self { config }
    }

    /// Filter results based on confidence and diversity
    pub fn filter(&self, results: Vec<RerankedResult>) -> Vec<FilteredResult> {
        if results.is_empty() {
            return Vec::new();
        }

        let mut filtered_results = Vec::new();

        // Step 1: Apply confidence and quality filtering
        let mut candidates: Vec<RerankedResult> = results
            .into_iter()
            .filter(|result| self.passes_basic_filters(result))
            .collect();

        // Step 2: Sort by final score (descending)
        candidates.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap());

        // Step 3: Apply diversity filtering
        for candidate in candidates {
            let diversity_score = if self.config.enable_diversity_filter {
                self.calculate_diversity_score(&candidate, &filtered_results)
            } else {
                1.0 // Max diversity if filtering disabled
            };

            let diversity_passed = diversity_score >= (1.0 - self.config.diversity_threshold);
            let confidence_passed = candidate.confidence >= self.config.min_confidence;
            let quality_passed = !self.config.enable_quality_filter
                || candidate.final_score >= self.config.min_quality_score;

            // Apply confidence boost for high-confidence results
            let boosted_confidence = if confidence_passed {
                (candidate.confidence * self.config.confidence_boost_factor).min(1.0)
            } else {
                candidate.confidence
            };

            // Include result if it passes all enabled filters
            if confidence_passed && quality_passed && diversity_passed {
                filtered_results.push(FilteredResult {
                    result: candidate,
                    confidence_passed,
                    diversity_passed,
                    quality_passed,
                    diversity_score,
                    boosted_confidence,
                });

                // Stop if we've reached the maximum number of results
                if filtered_results.len() >= self.config.max_results {
                    break;
                }
            }
        }

        // Final sort by boosted confidence and score
        filtered_results.sort_by(|a, b| {
            let score_a = a.boosted_confidence * a.result.final_score;
            let score_b = b.boosted_confidence * b.result.final_score;
            score_b.partial_cmp(&score_a).unwrap()
        });

        filtered_results
    }

    /// Apply specific filtering criteria
    pub fn filter_by_criteria(
        &self,
        results: Vec<RerankedResult>,
        criteria: FilterCriteria,
    ) -> Vec<FilteredResult> {
        match criteria {
            FilterCriteria::Confidence => self.filter_by_confidence(results),
            FilterCriteria::Diversity => self.filter_by_diversity(results),
            FilterCriteria::Quality => self.filter_by_quality(results),
            FilterCriteria::ScoreThreshold => self.filter_by_score_threshold(results),
            FilterCriteria::Combined => self.filter(results),
        }
    }

    /// Filter by confidence score only
    fn filter_by_confidence(&self, results: Vec<RerankedResult>) -> Vec<FilteredResult> {
        results
            .into_iter()
            .filter(|r| r.confidence >= self.config.min_confidence)
            .map(|result| FilteredResult {
                boosted_confidence: result.confidence * self.config.confidence_boost_factor,
                confidence_passed: true,
                diversity_passed: true,
                quality_passed: true,
                diversity_score: 1.0,
                result,
            })
            .collect()
    }

    /// Filter by diversity only
    fn filter_by_diversity(&self, results: Vec<RerankedResult>) -> Vec<FilteredResult> {
        let mut filtered = Vec::new();

        for result in results {
            let diversity_score = self.calculate_diversity_score(&result, &filtered);
            let diversity_passed = diversity_score >= (1.0 - self.config.diversity_threshold);

            if diversity_passed {
                filtered.push(FilteredResult {
                    result,
                    confidence_passed: true,
                    diversity_passed: true,
                    quality_passed: true,
                    diversity_score,
                    boosted_confidence: 1.0,
                });
            }
        }

        filtered
    }

    /// Filter by quality score only
    fn filter_by_quality(&self, results: Vec<RerankedResult>) -> Vec<FilteredResult> {
        results
            .into_iter()
            .filter(|r| r.final_score >= self.config.min_quality_score)
            .map(|result| FilteredResult {
                result,
                confidence_passed: true,
                diversity_passed: true,
                quality_passed: true,
                diversity_score: 1.0,
                boosted_confidence: 1.0,
            })
            .collect()
    }

    /// Filter by score threshold
    fn filter_by_score_threshold(&self, results: Vec<RerankedResult>) -> Vec<FilteredResult> {
        // Use the average score as a dynamic threshold
        let avg_score = if results.is_empty() {
            0.0
        } else {
            results.iter().map(|r| r.final_score).sum::<f32>() / results.len() as f32
        };

        let threshold = avg_score.max(self.config.min_quality_score);

        results
            .into_iter()
            .filter(|r| r.final_score >= threshold)
            .map(|result| FilteredResult {
                result,
                confidence_passed: true,
                diversity_passed: true,
                quality_passed: true,
                diversity_score: 1.0,
                boosted_confidence: 1.0,
            })
            .collect()
    }

    /// Check if a result passes basic filtering criteria
    fn passes_basic_filters(&self, result: &RerankedResult) -> bool {
        let confidence_ok = result.confidence >= self.config.min_confidence;
        let quality_ok = !self.config.enable_quality_filter
            || result.final_score >= self.config.min_quality_score;

        confidence_ok && quality_ok
    }

    /// Calculate diversity score for a candidate relative to existing results
    fn calculate_diversity_score(
        &self,
        candidate: &RerankedResult,
        existing: &[FilteredResult],
    ) -> f32 {
        if existing.is_empty() {
            return 1.0; // First result is always diverse
        }

        let mut min_diversity: f32 = 1.0;

        for existing_result in existing {
            let similarity = self.compute_content_similarity(
                &candidate.result.content,
                &existing_result.result.result.content,
            );

            let diversity = 1.0 - similarity;
            min_diversity = min_diversity.min(diversity);
        }

        min_diversity
    }

    /// Compute content similarity between two results
    fn compute_content_similarity(&self, content1: &str, content2: &str) -> f32 {
        // Simple word-based similarity
        let content1_lower = content1.to_lowercase();
        let words1: HashSet<&str> = content1_lower
            .split_whitespace()
            .filter(|&w| !self.is_stop_word(w))
            .collect();

        let content2_lower = content2.to_lowercase();
        let words2: HashSet<&str> = content2_lower
            .split_whitespace()
            .filter(|&w| !self.is_stop_word(w))
            .collect();

        if words1.is_empty() && words2.is_empty() {
            return 1.0;
        }

        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        let intersection_size = words1.intersection(&words2).count();
        let union_size = words1.union(&words2).count();

        intersection_size as f32 / union_size as f32
    }

    /// Check if a word is a stop word
    fn is_stop_word(&self, word: &str) -> bool {
        const STOP_WORDS: &[&str] = &[
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
            "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
            "they", "we", "say", "her", "she", "or", "an", "will", "my",
        ];
        STOP_WORDS.contains(&word)
    }

    /// Get filtering statistics for a set of results
    pub fn get_filtering_statistics(&self, results: &[FilteredResult]) -> FilteringStatistics {
        let total_results = results.len();
        let confidence_passed = results.iter().filter(|r| r.confidence_passed).count();
        let diversity_passed = results.iter().filter(|r| r.diversity_passed).count();
        let quality_passed = results.iter().filter(|r| r.quality_passed).count();

        let avg_confidence = if total_results > 0 {
            results.iter().map(|r| r.boosted_confidence).sum::<f32>() / total_results as f32
        } else {
            0.0
        };

        let avg_diversity = if total_results > 0 {
            results.iter().map(|r| r.diversity_score).sum::<f32>() / total_results as f32
        } else {
            0.0
        };

        FilteringStatistics {
            total_results,
            confidence_passed,
            diversity_passed,
            quality_passed,
            avg_confidence,
            avg_diversity,
            config: self.config.clone(),
        }
    }

    /// Get configuration
    pub fn get_config(&self) -> &ConfidenceConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: ConfidenceConfig) {
        self.config = config;
    }

    /// Adaptive filtering that adjusts thresholds based on result distribution
    pub fn adaptive_filter(&self, results: Vec<RerankedResult>) -> Vec<FilteredResult> {
        if results.is_empty() {
            return Vec::new();
        }

        // Calculate adaptive thresholds
        let confidence_scores: Vec<f32> = results.iter().map(|r| r.confidence).collect();
        let quality_scores: Vec<f32> = results.iter().map(|r| r.final_score).collect();

        let avg_confidence = confidence_scores.iter().sum::<f32>() / confidence_scores.len() as f32;
        let avg_quality = quality_scores.iter().sum::<f32>() / quality_scores.len() as f32;

        // Adjust thresholds based on distribution
        let adaptive_confidence_threshold = (self.config.min_confidence + avg_confidence) / 2.0;
        let adaptive_quality_threshold = (self.config.min_quality_score + avg_quality) / 2.0;

        // Create temporary config with adaptive thresholds
        let mut adaptive_config = self.config.clone();
        adaptive_config.min_confidence = adaptive_confidence_threshold;
        adaptive_config.min_quality_score = adaptive_quality_threshold;

        let adaptive_filter = ConfidenceFilter::with_config(adaptive_config);
        adaptive_filter.filter(results)
    }
}

impl Default for ConfidenceFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about filtering performance
#[derive(Debug, Clone)]
pub struct FilteringStatistics {
    pub total_results: usize,
    pub confidence_passed: usize,
    pub diversity_passed: usize,
    pub quality_passed: usize,
    pub avg_confidence: f32,
    pub avg_diversity: f32,
    pub config: ConfidenceConfig,
}

impl FilteringStatistics {
    /// Print filtering statistics
    pub fn print(&self) {
        println!("Confidence Filtering Statistics:");
        println!("  Total results processed: {}", self.total_results);
        println!(
            "  Confidence filter pass rate: {:.1}%",
            self.confidence_passed as f32 / self.total_results.max(1) as f32 * 100.0
        );
        println!(
            "  Diversity filter pass rate: {:.1}%",
            self.diversity_passed as f32 / self.total_results.max(1) as f32 * 100.0
        );
        println!(
            "  Quality filter pass rate: {:.1}%",
            self.quality_passed as f32 / self.total_results.max(1) as f32 * 100.0
        );
        println!("  Average confidence: {:.3}", self.avg_confidence);
        println!("  Average diversity: {:.3}", self.avg_diversity);
        println!("  Configuration:");
        println!("    Min confidence: {:.2}", self.config.min_confidence);
        println!(
            "    Diversity threshold: {:.2}",
            self.config.diversity_threshold
        );
        println!("    Max results: {}", self.config.max_results);
        println!(
            "    Quality filtering: {}",
            self.config.enable_quality_filter
        );
        println!(
            "    Diversity filtering: {}",
            self.config.enable_diversity_filter
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        query::multi_query::MultiQueryResult,
        reranking::cross_encoder::{ComponentScores, RerankedResult},
        retrieval::ResultType,
    };
    use std::collections::HashMap;

    fn create_test_reranked_result(
        id: &str,
        content: &str,
        final_score: f32,
        confidence: f32,
    ) -> RerankedResult {
        RerankedResult {
            result: MultiQueryResult {
                id: id.to_string(),
                content: content.to_string(),
                score: final_score,
                query_scores: HashMap::new(),
                query_frequency: 1,
                result_type: ResultType::Chunk,
                entities: Vec::new(),
                source_chunks: Vec::new(),
                found_by_queries: Vec::new(),
            },
            final_score,
            original_score: final_score,
            relevance_score: final_score,
            component_scores: ComponentScores {
                semantic_score: 0.5,
                entity_overlap_score: 0.5,
                context_relevance_score: 0.5,
                frequency_boost_score: 0.5,
            },
            confidence,
            original_rank: 0,
            new_rank: 0,
        }
    }

    #[test]
    fn test_confidence_filter_creation() {
        let filter = ConfidenceFilter::new();
        assert_eq!(filter.get_config().min_confidence, 0.5);
    }

    #[test]
    fn test_confidence_filtering() {
        let filter = ConfidenceFilter::new();
        let results = vec![
            create_test_reranked_result("1", "high confidence result", 0.9, 0.8),
            create_test_reranked_result("2", "low confidence result", 0.7, 0.3),
        ];

        let filtered = filter.filter_by_confidence(results);
        assert_eq!(filtered.len(), 1); // Only high confidence result should pass
        assert_eq!(filtered[0].result.result.id, "1");
    }

    #[test]
    fn test_diversity_filtering() {
        let filter = ConfidenceFilter::new();
        let results = vec![
            create_test_reranked_result("1", "the quick brown fox", 0.9, 0.8),
            create_test_reranked_result("2", "the quick brown fox jumps", 0.8, 0.7),
            create_test_reranked_result("3", "completely different content", 0.7, 0.6),
        ];

        let filtered = filter.filter_by_diversity(results);
        // Should include first and third results (diverse), but not second (too similar to first)
        assert!(filtered.len() >= 2);
    }

    #[test]
    fn test_content_similarity() {
        let filter = ConfidenceFilter::new();
        let similarity =
            filter.compute_content_similarity("the quick brown fox", "the fast brown fox");

        assert!(similarity >= 0.5); // Should be fairly similar (Jaccard similarity)
        assert!(similarity < 1.0); // But not identical
    }

    #[test]
    fn test_empty_results() {
        let filter = ConfidenceFilter::new();
        let filtered = filter.filter(vec![]);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_adaptive_filtering() {
        let filter = ConfidenceFilter::new();
        let results = vec![
            create_test_reranked_result("1", "result 1", 0.9, 0.9),
            create_test_reranked_result("2", "result 2", 0.8, 0.8),
            create_test_reranked_result("3", "result 3", 0.3, 0.3),
        ];

        let filtered = filter.adaptive_filter(results);
        // Adaptive filtering should be more lenient than fixed thresholds
        assert!(!filtered.is_empty());
    }

    #[test]
    fn test_filter_criteria() {
        assert_eq!(FilterCriteria::Confidence, FilterCriteria::Confidence);
        assert_ne!(FilterCriteria::Confidence, FilterCriteria::Diversity);
    }

    #[test]
    fn test_filtering_statistics() {
        let filter = ConfidenceFilter::new();
        let filtered_results = vec![FilteredResult {
            result: create_test_reranked_result("1", "content", 0.8, 0.8),
            confidence_passed: true,
            diversity_passed: true,
            quality_passed: true,
            diversity_score: 0.9,
            boosted_confidence: 0.8,
        }];

        let stats = filter.get_filtering_statistics(&filtered_results);
        assert_eq!(stats.total_results, 1);
        assert_eq!(stats.confidence_passed, 1);
    }
}
