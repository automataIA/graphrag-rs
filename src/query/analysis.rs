//! Query analysis for automatic query type detection and strategy selection

use crate::Result;
use std::collections::HashMap;

/// Types of queries for strategy selection
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryType {
    /// Entity-focused queries (Who is X? What does Y do?)
    EntityFocused,
    /// Conceptual queries (What is the concept of X?)
    Conceptual,
    /// Factual queries (When did X happen?)
    Factual,
    /// Relational queries (How are X and Y related?)
    Relational,
    /// Complex queries requiring multiple strategies
    Complex,
}

/// Configuration for query analysis
#[derive(Debug, Clone)]
pub struct QueryAnalysisConfig {
    /// Minimum confidence threshold for classification
    pub min_confidence: f32,
    /// Keywords for different query types
    pub entity_keywords: Vec<String>,
    pub conceptual_keywords: Vec<String>,
    pub factual_keywords: Vec<String>,
    pub relational_keywords: Vec<String>,
}

impl Default for QueryAnalysisConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            entity_keywords: vec![
                "who".to_string(),
                "what person".to_string(),
                "which person".to_string(),
                "character".to_string(),
                "individual".to_string(),
            ],
            conceptual_keywords: vec![
                "what is".to_string(),
                "define".to_string(),
                "concept of".to_string(),
                "meaning of".to_string(),
                "explain".to_string(),
            ],
            factual_keywords: vec![
                "when".to_string(),
                "where".to_string(),
                "how many".to_string(),
                "what happened".to_string(),
                "what year".to_string(),
            ],
            relational_keywords: vec![
                "relationship".to_string(),
                "connected".to_string(),
                "related".to_string(),
                "between".to_string(),
                "and".to_string(),
                "with".to_string(),
            ],
        }
    }
}

/// Result of query analysis
#[derive(Debug, Clone)]
pub struct QueryAnalysisResult {
    pub query_type: QueryType,
    pub confidence: f32,
    pub keywords_matched: Vec<String>,
    pub suggested_strategies: Vec<String>,
    pub complexity_score: f32,
}

/// Query analyzer for automatic type detection
pub struct QueryAnalyzer {
    config: QueryAnalysisConfig,
    keyword_weights: HashMap<String, f32>,
}

impl QueryAnalyzer {
    /// Create a new query analyzer
    pub fn new(config: QueryAnalysisConfig) -> Self {
        let mut keyword_weights = HashMap::new();

        // Assign weights to different keywords
        for keyword in &config.entity_keywords {
            keyword_weights.insert(keyword.clone(), 1.0);
        }
        for keyword in &config.conceptual_keywords {
            keyword_weights.insert(keyword.clone(), 0.8);
        }
        for keyword in &config.factual_keywords {
            keyword_weights.insert(keyword.clone(), 0.9);
        }
        for keyword in &config.relational_keywords {
            keyword_weights.insert(keyword.clone(), 0.7);
        }

        Self {
            config,
            keyword_weights,
        }
    }

    /// Analyze a query and determine its type
    pub fn analyze_query(&self, query: &str) -> Result<QueryAnalysisResult> {
        let query_lower = query.to_lowercase();

        let mut entity_score = 0.0;
        let mut conceptual_score = 0.0;
        let mut factual_score = 0.0;
        let mut relational_score = 0.0;
        let mut matched_keywords = Vec::new();

        // Score different query types based on keyword presence
        for keyword in &self.config.entity_keywords {
            if query_lower.contains(keyword) {
                entity_score += self.keyword_weights.get(keyword).unwrap_or(&1.0);
                matched_keywords.push(keyword.clone());
            }
        }

        for keyword in &self.config.conceptual_keywords {
            if query_lower.contains(keyword) {
                conceptual_score += self.keyword_weights.get(keyword).unwrap_or(&1.0);
                matched_keywords.push(keyword.clone());
            }
        }

        for keyword in &self.config.factual_keywords {
            if query_lower.contains(keyword) {
                factual_score += self.keyword_weights.get(keyword).unwrap_or(&1.0);
                matched_keywords.push(keyword.clone());
            }
        }

        for keyword in &self.config.relational_keywords {
            if query_lower.contains(keyword) {
                relational_score += self.keyword_weights.get(keyword).unwrap_or(&1.0);
                matched_keywords.push(keyword.clone());
            }
        }

        // Determine query type based on highest score
        let scores = vec![
            (QueryType::EntityFocused, entity_score),
            (QueryType::Conceptual, conceptual_score),
            (QueryType::Factual, factual_score),
            (QueryType::Relational, relational_score),
        ];

        let (query_type, max_score) = scores
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or((QueryType::Complex, 0.0));

        // Calculate complexity based on query length and multiple high scores
        let complexity_score = self.calculate_complexity(
            &query_lower,
            entity_score,
            conceptual_score,
            factual_score,
            relational_score,
        );

        // If complexity is high or no clear winner, mark as complex
        let final_query_type = if complexity_score > 0.7 || max_score < self.config.min_confidence {
            QueryType::Complex
        } else {
            query_type
        };

        // Suggest strategies based on query type
        let suggested_strategies = self.suggest_strategies(&final_query_type);

        let confidence = if final_query_type == QueryType::Complex {
            0.5 // Lower confidence for complex queries
        } else {
            (max_score / 3.0).min(1.0_f32) // Normalize confidence
        };

        Ok(QueryAnalysisResult {
            query_type: final_query_type,
            confidence,
            keywords_matched: matched_keywords,
            suggested_strategies,
            complexity_score,
        })
    }

    /// Calculate query complexity score
    fn calculate_complexity(
        &self,
        query: &str,
        entity_score: f32,
        conceptual_score: f32,
        factual_score: f32,
        relational_score: f32,
    ) -> f32 {
        let word_count = query.split_whitespace().count();
        let length_complexity = if word_count > 10 { 0.3_f32 } else { 0.0_f32 };

        // Check for multiple high scores (indicates mixed query type)
        let high_scores = [
            entity_score,
            conceptual_score,
            factual_score,
            relational_score,
        ]
        .iter()
        .filter(|&&score| score > 1.0)
        .count();

        let type_complexity = if high_scores > 1 { 0.4_f32 } else { 0.0_f32 };

        // Check for complex conjunctions
        let conjunction_complexity =
            if query.contains(" and ") || query.contains(" or ") || query.contains(" but ") {
                0.3_f32
            } else {
                0.0_f32
            };

        (length_complexity + type_complexity + conjunction_complexity).min(1.0_f32)
    }

    /// Suggest retrieval strategies based on query type
    fn suggest_strategies(&self, query_type: &QueryType) -> Vec<String> {
        match query_type {
            QueryType::EntityFocused => {
                vec!["entity_search".to_string(), "graph_traversal".to_string()]
            }
            QueryType::Conceptual => vec![
                "vector_similarity".to_string(),
                "hierarchical_summary".to_string(),
            ],
            QueryType::Factual => vec!["bm25_search".to_string(), "exact_match".to_string()],
            QueryType::Relational => vec![
                "graph_traversal".to_string(),
                "relationship_analysis".to_string(),
            ],
            QueryType::Complex => vec![
                "hybrid_retrieval".to_string(),
                "multi_strategy_fusion".to_string(),
                "adaptive_weighting".to_string(),
            ],
        }
    }

    /// Get analysis statistics
    pub fn get_statistics(&self) -> QueryAnalyzerStatistics {
        QueryAnalyzerStatistics {
            total_keywords: self.keyword_weights.len(),
            entity_keywords: self.config.entity_keywords.len(),
            conceptual_keywords: self.config.conceptual_keywords.len(),
            factual_keywords: self.config.factual_keywords.len(),
            relational_keywords: self.config.relational_keywords.len(),
            min_confidence_threshold: self.config.min_confidence,
        }
    }
}

impl Default for QueryAnalyzer {
    fn default() -> Self {
        Self::new(QueryAnalysisConfig::default())
    }
}

/// Statistics about the query analyzer
#[derive(Debug)]
pub struct QueryAnalyzerStatistics {
    pub total_keywords: usize,
    pub entity_keywords: usize,
    pub conceptual_keywords: usize,
    pub factual_keywords: usize,
    pub relational_keywords: usize,
    pub min_confidence_threshold: f32,
}

impl QueryAnalyzerStatistics {
    pub fn print(&self) {
        println!("Query Analyzer Statistics:");
        println!("  Total keywords: {}", self.total_keywords);
        println!("  Entity keywords: {}", self.entity_keywords);
        println!("  Conceptual keywords: {}", self.conceptual_keywords);
        println!("  Factual keywords: {}", self.factual_keywords);
        println!("  Relational keywords: {}", self.relational_keywords);
        println!("  Min confidence: {:.2}", self.min_confidence_threshold);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_query_analysis() {
        let analyzer = QueryAnalyzer::default();
        let result = analyzer.analyze_query("Who is Entity Name?").unwrap();

        assert_eq!(result.query_type, QueryType::EntityFocused);
        assert!(result.confidence > 0.0);
        assert!(result
            .suggested_strategies
            .contains(&"entity_search".to_string()));
    }

    #[test]
    fn test_conceptual_query_analysis() {
        let analyzer = QueryAnalyzer::default();
        let result = analyzer
            .analyze_query("What is the concept of friendship?")
            .unwrap();

        assert_eq!(result.query_type, QueryType::Conceptual);
        assert!(result
            .suggested_strategies
            .contains(&"vector_similarity".to_string()));
    }

    #[test]
    fn test_complex_query_analysis() {
        let analyzer = QueryAnalyzer::default();
        let result = analyzer.analyze_query("Who is Entity Name and what is his relationship with Second Entity when they go on adventures?").unwrap();

        // This should be detected as complex due to length and multiple query types
        assert!(result.complexity_score > 0.5);
    }

    #[test]
    fn test_factual_query_analysis() {
        let analyzer = QueryAnalyzer::default();
        let result = analyzer
            .analyze_query("When did Entity Name main activity?")
            .unwrap();

        assert_eq!(result.query_type, QueryType::Factual);
        assert!(result
            .suggested_strategies
            .contains(&"bm25_search".to_string()));
    }
}
