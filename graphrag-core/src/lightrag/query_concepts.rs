//! Query-Concept Semantic Matching
//!
//! This module provides semantic matching between user queries and concept graphs
//! to dynamically select the most relevant concepts for each query.
//!
//! ## Matching Strategies
//!
//! 1. **Exact Match**: Direct substring/token overlap
//! 2. **Fuzzy Match**: Edit distance for typos and variations
//! 3. **Semantic Match**: Embedding-based cosine similarity
//!
//! ## Example
//!
//! ```rust,no_run
//! use graphrag_core::lightrag::query_concepts::{QueryConceptMatcher, QueryMatchConfig};
//!
//! let matcher = QueryConceptMatcher::new(QueryMatchConfig::default());
//! let relevant = matcher.match_query_to_concepts(
//!     "What is machine learning?",
//!     &ranked_concepts
//! );
//! ```

use super::concept_selection::RankedConcept;
use serde::{Deserialize, Serialize};

/// Configuration for query-concept matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMatchConfig {
    /// Enable exact token matching
    pub use_exact_match: bool,

    /// Enable fuzzy matching (edit distance)
    pub use_fuzzy_match: bool,

    /// Enable semantic embedding matching
    pub use_semantic_match: bool,

    /// Weight for exact match score
    pub exact_match_weight: f32,

    /// Weight for fuzzy match score
    pub fuzzy_match_weight: f32,

    /// Weight for semantic match score
    pub semantic_match_weight: f32,

    /// Minimum edit distance threshold for fuzzy matching
    pub fuzzy_threshold: usize,

    /// Minimum semantic similarity for embedding match
    pub semantic_threshold: f32,

    /// Boost factor for concepts already ranked high
    pub ranking_boost: f32,

    /// Maximum number of concepts to return
    pub max_results: usize,
}

impl Default for QueryMatchConfig {
    fn default() -> Self {
        Self {
            use_exact_match: true,
            use_fuzzy_match: true,
            use_semantic_match: false, // Disabled by default (requires embeddings)
            exact_match_weight: 0.5,
            fuzzy_match_weight: 0.3,
            semantic_match_weight: 0.2,
            fuzzy_threshold: 2,
            semantic_threshold: 0.6,
            ranking_boost: 0.2,
            max_results: 20,
        }
    }
}

/// Matched concept with relevance scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchedConcept {
    /// Concept text
    pub concept: String,

    /// Final match score
    pub match_score: f32,

    /// Original ranking score
    pub ranking_score: f32,

    /// Exact match score
    pub exact_score: f32,

    /// Fuzzy match score
    pub fuzzy_score: f32,

    /// Semantic match score (if embeddings enabled)
    pub semantic_score: f32,

    /// Matched query tokens
    pub matched_tokens: Vec<String>,
}

/// Query-to-concept semantic matcher
pub struct QueryConceptMatcher {
    config: QueryMatchConfig,
}

impl QueryConceptMatcher {
    /// Create a new query concept matcher
    pub fn new(config: QueryMatchConfig) -> Self {
        Self { config }
    }

    /// Match a query to ranked concepts
    ///
    /// Returns concepts sorted by relevance to the query, combining:
    /// - Original ranking score
    /// - Query-concept match score
    ///
    /// # Arguments
    ///
    /// * `query` - The user's query text
    /// * `ranked_concepts` - Pre-ranked concepts from ConceptRanker
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use graphrag_core::lightrag::query_concepts::{QueryConceptMatcher, QueryMatchConfig};
    /// let matcher = QueryConceptMatcher::new(QueryMatchConfig::default());
    /// let matched = matcher.match_query_to_concepts(
    ///     "How does neural network backpropagation work?",
    ///     &ranked_concepts
    /// );
    /// ```
    pub fn match_query_to_concepts(
        &self,
        query: &str,
        ranked_concepts: &[RankedConcept],
    ) -> Vec<MatchedConcept> {
        // Tokenize query
        let query_tokens = self.tokenize(query);

        let mut matched_concepts = Vec::new();

        for ranked_concept in ranked_concepts {
            let concept_text = &ranked_concept.concept;

            // Calculate match scores
            let exact_score = if self.config.use_exact_match {
                self.calculate_exact_match(concept_text, &query_tokens)
            } else {
                0.0
            };

            let fuzzy_score = if self.config.use_fuzzy_match {
                self.calculate_fuzzy_match(concept_text, &query_tokens)
            } else {
                0.0
            };

            // Semantic matching using word overlap as proxy for semantic similarity
            // Note: True embedding-based matching requires embeddings to be precomputed and passed in
            let semantic_score = if self.config.use_semantic_match {
                self.calculate_semantic_similarity(query, concept_text)
            } else {
                0.0
            };

            // Combine match scores
            let match_score =
                (self.config.exact_match_weight * exact_score) +
                (self.config.fuzzy_match_weight * fuzzy_score) +
                (self.config.semantic_match_weight * semantic_score);

            // Boost with original ranking score
            let final_score = match_score + (self.config.ranking_boost * ranked_concept.score);

            // Skip if no match at all
            if match_score == 0.0 && exact_score == 0.0 {
                continue;
            }

            // Find which query tokens matched
            let matched_tokens = self.find_matched_tokens(concept_text, &query_tokens);

            matched_concepts.push(MatchedConcept {
                concept: concept_text.clone(),
                match_score: final_score,
                ranking_score: ranked_concept.score,
                exact_score,
                fuzzy_score,
                semantic_score,
                matched_tokens,
            });
        }

        // Sort by final match score descending
        matched_concepts.sort_by(|a, b| {
            b.match_score.partial_cmp(&a.match_score).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return top-K
        matched_concepts.into_iter().take(self.config.max_results).collect()
    }

    /// Tokenize text into normalized tokens
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .filter(|t| t.len() >= 2) // Filter very short tokens
            .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|t| !t.is_empty())
            .map(String::from)
            .collect()
    }

    /// Calculate exact match score
    ///
    /// Score = (matched_tokens / total_query_tokens)
    fn calculate_exact_match(&self, concept: &str, query_tokens: &[String]) -> f32 {
        if query_tokens.is_empty() {
            return 0.0;
        }

        let concept_lower = concept.to_lowercase();
        let mut matched_count = 0;

        for token in query_tokens {
            if concept_lower.contains(token) {
                matched_count += 1;
            }
        }

        matched_count as f32 / query_tokens.len() as f32
    }

    /// Calculate fuzzy match score using edit distance
    ///
    /// Uses Levenshtein distance for typo tolerance
    fn calculate_fuzzy_match(&self, concept: &str, query_tokens: &[String]) -> f32 {
        if query_tokens.is_empty() {
            return 0.0;
        }

        let concept_tokens = self.tokenize(concept);
        let mut total_similarity = 0.0;

        for query_token in query_tokens {
            let mut best_match = 0.0;

            for concept_token in &concept_tokens {
                let distance = self.levenshtein_distance(query_token, concept_token);
                let max_len = query_token.len().max(concept_token.len());

                if max_len == 0 {
                    continue;
                }

                // Similarity = 1 - (distance / max_length)
                let similarity = 1.0 - (distance as f32 / max_len as f32);

                // Only consider good matches (within threshold)
                if distance <= self.config.fuzzy_threshold {
                    best_match = best_match.max(similarity);
                }
            }

            total_similarity += best_match;
        }

        total_similarity / query_tokens.len() as f32
    }

    /// Calculate Levenshtein distance between two strings
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let len1 = s1.len();
        let len2 = s2.len();

        if len1 == 0 {
            return len2;
        }
        if len2 == 0 {
            return len1;
        }

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        // Initialize first row and column
        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        // Fill matrix
        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] { 0 } else { 1 };

                matrix[i][j] = *[
                    matrix[i - 1][j] + 1,      // deletion
                    matrix[i][j - 1] + 1,      // insertion
                    matrix[i - 1][j - 1] + cost, // substitution
                ].iter().min().unwrap();
            }
        }

        matrix[len1][len2]
    }

    /// Calculate semantic similarity using word overlap and semantic relatedness
    ///
    /// This is a lightweight proxy for true embedding-based semantic similarity.
    /// For production use with embeddings, concepts and query should be pre-embedded
    /// and cosine similarity calculated.
    fn calculate_semantic_similarity(&self, query: &str, concept: &str) -> f32 {
        let query_tokens: std::collections::HashSet<String> =
            self.tokenize(query).into_iter().collect();
        let concept_tokens: std::collections::HashSet<String> =
            self.tokenize(concept).into_iter().collect();

        if query_tokens.is_empty() || concept_tokens.is_empty() {
            return 0.0;
        }

        // Calculate Jaccard similarity (intersection over union)
        let intersection: std::collections::HashSet<_> =
            query_tokens.intersection(&concept_tokens).collect();
        let union: std::collections::HashSet<_> = query_tokens.union(&concept_tokens).collect();

        let jaccard_score = intersection.len() as f32 / union.len() as f32;

        // Calculate token containment (concept contains query tokens)
        let containment_score = intersection.len() as f32 / query_tokens.len() as f32;

        // Weighted combination
        let semantic_score = (0.6 * jaccard_score) + (0.4 * containment_score);

        // Apply threshold
        if semantic_score < self.config.semantic_threshold {
            0.0
        } else {
            semantic_score
        }
    }

    /// Find which query tokens matched the concept
    fn find_matched_tokens(&self, concept: &str, query_tokens: &[String]) -> Vec<String> {
        let concept_lower = concept.to_lowercase();
        let mut matched = Vec::new();

        for token in query_tokens {
            if concept_lower.contains(token) {
                matched.push(token.clone());
            }
        }

        matched
    }

    /// Get concepts by exact phrase match
    ///
    /// More strict than token matching - requires the entire concept to appear in query
    pub fn get_exact_phrase_matches(
        &self,
        query: &str,
        ranked_concepts: &[RankedConcept],
    ) -> Vec<String> {
        let query_lower = query.to_lowercase();
        let mut matches = Vec::new();

        for concept in ranked_concepts {
            let concept_lower = concept.concept.to_lowercase();
            if query_lower.contains(&concept_lower) {
                matches.push(concept.concept.clone());
            }
        }

        matches
    }

    /// Filter concepts by minimum match score
    pub fn filter_by_match_score(
        &self,
        matched_concepts: &[MatchedConcept],
        min_score: f32,
    ) -> Vec<MatchedConcept> {
        matched_concepts
            .iter()
            .filter(|mc| mc.match_score >= min_score)
            .cloned()
            .collect()
    }
}

/// Query concept matching statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchingStats {
    /// Total concepts evaluated
    pub total_concepts: usize,

    /// Concepts that matched
    pub matched_concepts: usize,

    /// Concepts with exact match
    pub exact_matches: usize,

    /// Concepts with fuzzy match
    pub fuzzy_matches: usize,

    /// Average match score
    pub average_match_score: f32,

    /// Top match score
    pub top_match_score: f32,
}

impl QueryConceptMatcher {
    /// Calculate matching statistics
    pub fn calculate_stats(&self, matched_concepts: &[MatchedConcept]) -> MatchingStats {
        if matched_concepts.is_empty() {
            return MatchingStats {
                total_concepts: 0,
                matched_concepts: 0,
                exact_matches: 0,
                fuzzy_matches: 0,
                average_match_score: 0.0,
                top_match_score: 0.0,
            };
        }

        let exact_matches = matched_concepts.iter().filter(|mc| mc.exact_score > 0.0).count();
        let fuzzy_matches = matched_concepts.iter().filter(|mc| mc.fuzzy_score > 0.0).count();

        let avg_score = matched_concepts.iter().map(|mc| mc.match_score).sum::<f32>()
            / matched_concepts.len() as f32;

        let top_score = matched_concepts
            .iter()
            .map(|mc| mc.match_score)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        MatchingStats {
            total_concepts: matched_concepts.len(),
            matched_concepts: matched_concepts.len(),
            exact_matches,
            fuzzy_matches,
            average_match_score: avg_score,
            top_match_score: top_score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lightrag::concept_selection::RankedConcept;

    fn create_test_concept(text: &str, score: f32) -> RankedConcept {
        RankedConcept {
            concept: text.to_string(),
            score,
            degree_score: 0.0,
            pagerank_score: 0.0,
            idf_score: 0.0,
            document_frequency: 1,
            total_frequency: 1,
        }
    }

    #[test]
    fn test_exact_matching() {
        let matcher = QueryConceptMatcher::new(QueryMatchConfig::default());

        let concepts = vec![
            create_test_concept("machine learning", 0.8),
            create_test_concept("deep learning", 0.7),
            create_test_concept("neural networks", 0.6),
        ];

        let matched = matcher.match_query_to_concepts("machine learning basics", &concepts);

        assert!(!matched.is_empty());

        // "machine learning" should match
        let ml_match = matched.iter().find(|m| m.concept == "machine learning");
        assert!(ml_match.is_some());
        assert!(ml_match.unwrap().exact_score > 0.0);
    }

    #[test]
    fn test_fuzzy_matching() {
        let matcher = QueryConceptMatcher::new(QueryMatchConfig::default());

        let concepts = vec![
            create_test_concept("neural network", 0.8),
        ];

        // Query has typo: "neurla" instead of "neural"
        let matched = matcher.match_query_to_concepts("neurla network training", &concepts);

        assert!(!matched.is_empty());

        let match_result = &matched[0];
        assert!(match_result.fuzzy_score > 0.0);
    }

    #[test]
    fn test_levenshtein_distance() {
        let matcher = QueryConceptMatcher::new(QueryMatchConfig::default());

        assert_eq!(matcher.levenshtein_distance("cat", "cat"), 0);
        assert_eq!(matcher.levenshtein_distance("cat", "hat"), 1);
        assert_eq!(matcher.levenshtein_distance("cat", "cats"), 1);
        assert_eq!(matcher.levenshtein_distance("machine", "machin"), 1);
    }

    #[test]
    fn test_exact_phrase_match() {
        let matcher = QueryConceptMatcher::new(QueryMatchConfig::default());

        let concepts = vec![
            create_test_concept("machine learning", 0.8),
            create_test_concept("deep learning", 0.7),
        ];

        let matches = matcher.get_exact_phrase_matches(
            "I want to learn about machine learning",
            &concepts
        );

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0], "machine learning");
    }

    #[test]
    fn test_tokenization() {
        let matcher = QueryConceptMatcher::new(QueryMatchConfig::default());

        let tokens = matcher.tokenize("What is Machine Learning?");

        assert!(tokens.contains(&"what".to_string()));
        assert!(tokens.contains(&"machine".to_string()));
        assert!(tokens.contains(&"learning".to_string()));

        // Should filter out punctuation
        assert!(!tokens.iter().any(|t| t.contains('?')));
    }

    #[test]
    fn test_ranking_boost() {
        let matcher = QueryConceptMatcher::new(QueryMatchConfig {
            ranking_boost: 0.5,
            ..Default::default()
        });

        let concepts = vec![
            create_test_concept("machine learning", 0.9), // High ranking score
            create_test_concept("learning", 0.1),         // Low ranking score
        ];

        let matched = matcher.match_query_to_concepts("learning systems", &concepts);

        // "machine learning" should rank higher due to ranking boost,
        // even though "learning" has exact match
        let top_concept = &matched[0];
        assert!(top_concept.ranking_score > 0.5);
    }

    #[test]
    fn test_stats_calculation() {
        let matched_concepts = vec![
            MatchedConcept {
                concept: "a".to_string(),
                match_score: 0.8,
                ranking_score: 0.5,
                exact_score: 0.7,
                fuzzy_score: 0.1,
                semantic_score: 0.0,
                matched_tokens: vec!["a".to_string()],
            },
            MatchedConcept {
                concept: "b".to_string(),
                match_score: 0.6,
                ranking_score: 0.4,
                exact_score: 0.0,
                fuzzy_score: 0.6,
                semantic_score: 0.0,
                matched_tokens: vec![],
            },
        ];

        let matcher = QueryConceptMatcher::new(QueryMatchConfig::default());
        let stats = matcher.calculate_stats(&matched_concepts);

        assert_eq!(stats.matched_concepts, 2);
        assert_eq!(stats.exact_matches, 1); // Only first concept has exact match
        assert_eq!(stats.fuzzy_matches, 2); // Both have fuzzy matches
        assert!(stats.average_match_score > 0.0);
        assert_eq!(stats.top_match_score, 0.8);
    }

    #[test]
    fn test_max_results_limit() {
        let matcher = QueryConceptMatcher::new(QueryMatchConfig {
            max_results: 2,
            ..Default::default()
        });

        let concepts = vec![
            create_test_concept("machine learning", 0.9),
            create_test_concept("deep learning", 0.8),
            create_test_concept("learning algorithms", 0.7),
            create_test_concept("supervised learning", 0.6),
        ];

        let matched = matcher.match_query_to_concepts("learning", &concepts);

        // Should return at most 2 results
        assert!(matched.len() <= 2);
    }
}
