use crate::{
    query::multi_query::MultiQueryResult,
    vector::{EmbeddingGenerator, VectorUtils},
    Result,
};
use std::collections::HashMap;

/// Configuration for cross-encoder reranking
#[derive(Debug, Clone)]
pub struct RerankingConfig {
    /// Confidence threshold for relevance scoring
    pub confidence_threshold: f32,
    /// Weight for semantic similarity in final score
    pub semantic_weight: f32,
    /// Weight for entity overlap in final score
    pub entity_weight: f32,
    /// Weight for context relevance in final score
    pub context_weight: f32,
    /// Weight for query frequency boost
    pub frequency_weight: f32,
    /// Strategy for combining retrieval and relevance scores
    pub score_combination: ScoreCombination,
    /// Maximum number of results to rerank (for performance)
    pub max_rerank_candidates: usize,
}

impl Default for RerankingConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            semantic_weight: 0.4,
            entity_weight: 0.3,
            context_weight: 0.2,
            frequency_weight: 0.1,
            score_combination: ScoreCombination::HarmonicMean,
            max_rerank_candidates: 50,
        }
    }
}

/// Methods for combining retrieval and relevance scores
#[derive(Debug, Clone, PartialEq)]
pub enum ScoreCombination {
    /// Arithmetic mean of scores
    ArithmeticMean,
    /// Harmonic mean for balanced combination
    HarmonicMean,
    /// Weighted average
    WeightedAverage,
    /// Maximum score
    Maximum,
    /// Product of normalized scores
    Product,
}

/// Strategy for reranking
#[derive(Debug, Clone, PartialEq)]
pub enum RerankingStrategy {
    /// Semantic similarity-based reranking
    Semantic,
    /// Entity overlap-based reranking
    EntityOverlap,
    /// Context relevance-based reranking
    ContextRelevance,
    /// Multi-factor reranking combining all strategies
    MultiFactor,
}

/// Reranked result with detailed scoring information
#[derive(Debug, Clone)]
pub struct RerankedResult {
    /// Original result
    pub result: MultiQueryResult,
    /// Final reranked score
    pub final_score: f32,
    /// Original retrieval score
    pub original_score: f32,
    /// Relevance score computed by reranker
    pub relevance_score: f32,
    /// Individual component scores
    pub component_scores: ComponentScores,
    /// Confidence level of the reranking
    pub confidence: f32,
    /// Ranking position before reranking
    pub original_rank: usize,
    /// Ranking position after reranking
    pub new_rank: usize,
}

/// Individual component scores used in reranking
#[derive(Debug, Clone)]
pub struct ComponentScores {
    /// Semantic similarity score
    pub semantic_score: f32,
    /// Entity overlap score
    pub entity_overlap_score: f32,
    /// Context relevance score
    pub context_relevance_score: f32,
    /// Query frequency boost score
    pub frequency_boost_score: f32,
}

/// Cross-encoder reranker for improving result relevance
pub struct CrossEncoderReranker {
    /// Embedding generator for semantic analysis
    embedding_generator: EmbeddingGenerator,
    /// Configuration for reranking
    config: RerankingConfig,
    /// Entity dictionary for overlap calculation
    entity_dictionary: HashMap<String, Vec<String>>,
}

impl CrossEncoderReranker {
    /// Create a new cross-encoder reranker with default configuration
    pub fn new() -> Self {
        Self::with_config(RerankingConfig::default())
    }

    /// Create a new cross-encoder reranker with custom configuration
    pub fn with_config(config: RerankingConfig) -> Self {
        Self {
            embedding_generator: EmbeddingGenerator::new(128),
            config,
            entity_dictionary: HashMap::new(),
        }
    }

    /// Rerank search results to improve relevance
    pub fn rerank(
        &mut self,
        query: &str,
        results: Vec<MultiQueryResult>,
    ) -> Result<Vec<RerankedResult>> {
        if results.is_empty() {
            return Ok(Vec::new());
        }

        // Limit candidates for performance
        let candidates = if results.len() > self.config.max_rerank_candidates {
            results
                .into_iter()
                .take(self.config.max_rerank_candidates)
                .collect()
        } else {
            results
        };

        let mut reranked_results = Vec::new();

        // Generate query embedding for semantic comparison
        let query_embedding = self.embedding_generator.generate_embedding(query);

        for (original_rank, result) in candidates.into_iter().enumerate() {
            // Calculate component scores
            let component_scores =
                self.calculate_component_scores(query, &query_embedding, &result);

            // Calculate overall relevance score
            let relevance_score = self.calculate_relevance_score(&component_scores);

            // Combine with original score
            let final_score = self.combine_scores(result.score, relevance_score);

            // Calculate confidence
            let confidence = self.calculate_confidence(&component_scores, result.query_frequency);

            reranked_results.push(RerankedResult {
                original_score: result.score,
                relevance_score,
                final_score,
                component_scores,
                confidence,
                original_rank,
                new_rank: 0, // Will be set after sorting
                result,
            });
        }

        // Sort by final score and update new ranks
        reranked_results.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap());
        for (new_rank, result) in reranked_results.iter_mut().enumerate() {
            result.new_rank = new_rank;
        }

        // Filter by confidence threshold
        let filtered_results: Vec<RerankedResult> = reranked_results
            .into_iter()
            .filter(|r| r.confidence >= self.config.confidence_threshold)
            .collect();

        Ok(filtered_results)
    }

    /// Calculate component scores for a result
    fn calculate_component_scores(
        &mut self,
        query: &str,
        query_embedding: &[f32],
        result: &MultiQueryResult,
    ) -> ComponentScores {
        // 1. Semantic similarity score
        let semantic_score = self.compute_semantic_similarity(query_embedding, &result.content);

        // 2. Entity overlap score
        let entity_overlap_score = self.compute_entity_overlap(query, result);

        // 3. Context relevance score
        let context_relevance_score = self.compute_context_relevance(query, &result.content);

        // 4. Query frequency boost score
        let frequency_boost_score = self.compute_frequency_boost(result);

        ComponentScores {
            semantic_score,
            entity_overlap_score,
            context_relevance_score,
            frequency_boost_score,
        }
    }

    /// Compute semantic similarity between query and result content
    fn compute_semantic_similarity(&mut self, query_embedding: &[f32], content: &str) -> f32 {
        let content_embedding = self.embedding_generator.generate_embedding(content);
        VectorUtils::cosine_similarity(query_embedding, &content_embedding).max(0.0)
    }

    /// Compute entity overlap between query and result
    fn compute_entity_overlap(&self, query: &str, result: &MultiQueryResult) -> f32 {
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();
        let mut overlap_count = 0;
        let total_entities = result.entities.len().max(1);

        for entity in &result.entities {
            let entity_lower = entity.to_lowercase();
            let entity_words: Vec<&str> = entity_lower.split_whitespace().collect();

            // Check for exact matches or partial matches
            for entity_word in &entity_words {
                if query_words
                    .iter()
                    .any(|&qw| qw.contains(entity_word) || entity_word.contains(qw))
                {
                    overlap_count += 1;
                    break; // Count each entity only once
                }
            }
        }

        overlap_count as f32 / total_entities as f32
    }

    /// Compute context relevance score
    fn compute_context_relevance(&self, query: &str, content: &str) -> f32 {
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower
            .split_whitespace()
            .filter(|&w| !self.is_stop_word(w))
            .collect();

        let content_lower = content.to_lowercase();
        let content_words: Vec<&str> = content_lower
            .split_whitespace()
            .filter(|&w| !self.is_stop_word(w))
            .collect();

        if query_words.is_empty() || content_words.is_empty() {
            return 0.0;
        }

        // Calculate term frequency overlap
        let mut term_matches = 0;
        for query_word in &query_words {
            if content_words
                .iter()
                .any(|&cw| cw.contains(query_word) || query_word.contains(cw))
            {
                term_matches += 1;
            }
        }

        // Normalize by query length
        let base_relevance = term_matches as f32 / query_words.len() as f32;

        // Boost for exact phrase matches
        let phrase_boost = if content.to_lowercase().contains(&query.to_lowercase()) {
            0.3
        } else {
            0.0
        };

        // Boost for term proximity
        let proximity_boost = self.calculate_term_proximity(&query_words, &content_words);

        (base_relevance + phrase_boost + proximity_boost).min(1.0)
    }

    /// Compute frequency boost based on how many queries found this result
    fn compute_frequency_boost(&self, result: &MultiQueryResult) -> f32 {
        // Normalize frequency score (more queries finding the same result = higher confidence)
        let max_expected_frequency = 5.0; // Reasonable upper bound
        (result.query_frequency as f32 / max_expected_frequency).min(1.0)
    }

    /// Calculate term proximity score
    fn calculate_term_proximity(&self, query_words: &[&str], content_words: &[&str]) -> f32 {
        if query_words.len() < 2 {
            return 0.0;
        }

        let mut proximity_score = 0.0;
        let mut matches_found = 0;

        for i in 0..query_words.len() - 1 {
            let word1 = query_words[i];
            let word2 = query_words[i + 1];

            // Find positions of both words in content
            let positions1: Vec<usize> = content_words
                .iter()
                .enumerate()
                .filter_map(|(pos, &w)| if w.contains(word1) { Some(pos) } else { None })
                .collect();

            let positions2: Vec<usize> = content_words
                .iter()
                .enumerate()
                .filter_map(|(pos, &w)| if w.contains(word2) { Some(pos) } else { None })
                .collect();

            // Calculate minimum distance between any pair
            let min_distance = positions1
                .iter()
                .flat_map(|&p1| {
                    positions2
                        .iter()
                        .map(move |&p2| (p1 as i32 - p2 as i32).abs())
                })
                .min()
                .unwrap_or(i32::MAX);

            if min_distance < i32::MAX {
                // Closer words get higher scores
                let distance_score = 1.0 / (1.0 + min_distance as f32);
                proximity_score += distance_score;
                matches_found += 1;
            }
        }

        if matches_found > 0 {
            proximity_score / matches_found as f32
        } else {
            0.0
        }
    }

    /// Calculate overall relevance score from components
    fn calculate_relevance_score(&self, scores: &ComponentScores) -> f32 {
        self.config.semantic_weight * scores.semantic_score
            + self.config.entity_weight * scores.entity_overlap_score
            + self.config.context_weight * scores.context_relevance_score
            + self.config.frequency_weight * scores.frequency_boost_score
    }

    /// Combine retrieval score with relevance score
    fn combine_scores(&self, retrieval_score: f32, relevance_score: f32) -> f32 {
        match self.config.score_combination {
            ScoreCombination::ArithmeticMean => (retrieval_score + relevance_score) / 2.0,
            ScoreCombination::HarmonicMean => {
                if retrieval_score + relevance_score > 0.0 {
                    2.0 * retrieval_score * relevance_score / (retrieval_score + relevance_score)
                } else {
                    0.0
                }
            }
            ScoreCombination::WeightedAverage => 0.6 * retrieval_score + 0.4 * relevance_score,
            ScoreCombination::Maximum => retrieval_score.max(relevance_score),
            ScoreCombination::Product => {
                // Normalize scores to 0-1 range first
                let norm_retrieval = retrieval_score.clamp(0.0, 1.0);
                let norm_relevance = relevance_score.clamp(0.0, 1.0);
                norm_retrieval * norm_relevance
            }
        }
    }

    /// Calculate confidence in the reranking decision
    fn calculate_confidence(&self, scores: &ComponentScores, query_frequency: usize) -> f32 {
        // Base confidence from component scores
        let score_confidence =
            (scores.semantic_score + scores.entity_overlap_score + scores.context_relevance_score)
                / 3.0;

        // Frequency boost for confidence
        let frequency_confidence = (query_frequency as f32 / 5.0).min(1.0);

        // Combine with slight weight toward score-based confidence
        0.7 * score_confidence + 0.3 * frequency_confidence
    }

    /// Check if a word is a stop word
    fn is_stop_word(&self, word: &str) -> bool {
        const STOP_WORDS: &[&str] = &[
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
            "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
            "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
            "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
            "go", "me",
        ];
        STOP_WORDS.contains(&word)
    }

    /// Update entity dictionary for better entity overlap calculation
    pub fn update_entity_dictionary(&mut self, entities: HashMap<String, Vec<String>>) {
        self.entity_dictionary = entities;
    }

    /// Get configuration
    pub fn get_config(&self) -> &RerankingConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: RerankingConfig) {
        self.config = config;
    }

    /// Get reranking statistics
    pub fn get_statistics(&self) -> RerankingStatistics {
        RerankingStatistics {
            config: self.config.clone(),
            entity_dictionary_size: self.entity_dictionary.len(),
        }
    }
}

impl Default for CrossEncoderReranker {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the reranker
#[derive(Debug, Clone)]
pub struct RerankingStatistics {
    pub config: RerankingConfig,
    pub entity_dictionary_size: usize,
}

impl RerankingStatistics {
    /// Print statistics
    pub fn print(&self) {
        println!("Cross-Encoder Reranker Statistics:");
        println!(
            "  Confidence threshold: {:.2}",
            self.config.confidence_threshold
        );
        println!("  Score combination: {:?}", self.config.score_combination);
        println!(
            "  Max rerank candidates: {}",
            self.config.max_rerank_candidates
        );
        println!("  Component weights:");
        println!("    Semantic: {:.2}", self.config.semantic_weight);
        println!("    Entity: {:.2}", self.config.entity_weight);
        println!("    Context: {:.2}", self.config.context_weight);
        println!("    Frequency: {:.2}", self.config.frequency_weight);
        println!("  Entity dictionary size: {}", self.entity_dictionary_size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{query::multi_query::MultiQueryResult, retrieval::ResultType};

    fn create_test_result(id: &str, content: &str, score: f32) -> MultiQueryResult {
        MultiQueryResult {
            id: id.to_string(),
            content: content.to_string(),
            score,
            query_scores: HashMap::new(),
            query_frequency: 1,
            result_type: ResultType::Chunk,
            entities: vec!["test_entity".to_string()],
            source_chunks: vec![id.to_string()],
            found_by_queries: vec!["test query".to_string()],
        }
    }

    #[test]
    fn test_reranker_creation() {
        let reranker = CrossEncoderReranker::new();
        let stats = reranker.get_statistics();
        assert_eq!(stats.config.confidence_threshold, 0.5);
    }

    #[test]
    fn test_score_combination_variants() {
        assert_eq!(
            ScoreCombination::HarmonicMean,
            ScoreCombination::HarmonicMean
        );
        assert_ne!(
            ScoreCombination::HarmonicMean,
            ScoreCombination::ArithmeticMean
        );
    }

    #[test]
    fn test_component_scores() {
        let scores = ComponentScores {
            semantic_score: 0.8,
            entity_overlap_score: 0.6,
            context_relevance_score: 0.7,
            frequency_boost_score: 0.5,
        };

        assert_eq!(scores.semantic_score, 0.8);
        assert_eq!(scores.entity_overlap_score, 0.6);
    }

    #[test]
    fn test_harmonic_mean_calculation() {
        let reranker = CrossEncoderReranker::new();
        let combined = reranker.combine_scores(0.8, 0.6);

        // Harmonic mean of 0.8 and 0.6 should be approximately 0.686
        assert!((combined - 0.686).abs() < 0.01);
    }

    #[test]
    fn test_empty_results_reranking() {
        let mut reranker = CrossEncoderReranker::new();
        let results = reranker.rerank("test query", vec![]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_entity_overlap_calculation() {
        let reranker = CrossEncoderReranker::new();
        let mut result = create_test_result("test1", "content about test entity", 0.8);
        result.entities = vec!["test".to_string(), "entity".to_string()];

        let overlap = reranker.compute_entity_overlap("test query", &result);
        assert!(overlap > 0.0);
    }

    #[test]
    fn test_context_relevance() {
        let reranker = CrossEncoderReranker::new();
        let relevance = reranker
            .compute_context_relevance("test query", "this content contains test and query words");

        assert!(relevance > 0.0);
        assert!(relevance <= 1.0);
    }
}
