//! Query Intelligence and Rewriting
//!
//! This module provides intelligent query processing including:
//! - Query rewriting and expansion
//! - Synonym expansion
//! - Relevance feedback learning
//! - Query templates
//! - Natural language to structured query conversion
//! - Query performance analysis

use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

/// Query intelligence engine
pub struct QueryIntelligence {
    synonyms: HashMap<String, Vec<String>>,
    templates: Vec<QueryTemplate>,
    stop_words: HashSet<String>,
    relevance_scores: HashMap<String, f32>,
}

/// Query template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryTemplate {
    /// Regex pattern to match queries
    pub pattern: String,
    /// Type of query this template matches
    pub query_type: QueryType,
    /// Rewrite template for query optimization
    pub rewrite: String,
}

/// Query type classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QueryType {
    /// Entity lookup queries
    EntityLookup,
    /// Relationship queries
    Relationship,
    /// Aggregation queries
    Aggregation,
    /// Comparison queries
    Comparison,
    /// Temporal queries
    Temporal,
    /// Causal queries
    Causal,
    /// General queries
    General,
}

/// Rewritten query
#[derive(Debug, Clone)]
pub struct RewrittenQuery {
    /// Original query text
    pub original: String,
    /// Rewritten optimized query
    pub rewritten: String,
    /// Detected query type
    pub query_type: QueryType,
    /// Expanded search terms
    pub expanded_terms: Vec<String>,
    /// Confidence score of rewrite
    pub confidence: f32,
}

impl QueryIntelligence {
    /// Create a new query intelligence engine with default settings
    pub fn new() -> Self {
        let mut engine = Self {
            synonyms: HashMap::new(),
            templates: Vec::new(),
            stop_words: HashSet::new(),
            relevance_scores: HashMap::new(),
        };

        // Initialize with default templates and synonyms
        engine.load_default_synonyms();
        engine.load_default_templates();
        engine.load_default_stop_words();

        engine
    }

    /// Rewrite and expand a query
    ///
    /// # Arguments
    /// * `query` - The original query string
    ///
    /// # Returns
    /// RewrittenQuery with expanded terms and detected query type
    pub fn rewrite_query(&self, query: &str) -> RewrittenQuery {
        // Normalize query
        let normalized = self.normalize_query(query);

        // Detect query type
        let query_type = self.detect_query_type(&normalized);

        // Apply template matching
        let template_rewritten = self.apply_templates(&normalized, &query_type);

        // Expand synonyms
        let expanded = self.expand_synonyms(&template_rewritten);

        // Extract key terms
        let expanded_terms = self.extract_key_terms(&expanded);

        // Calculate confidence
        let confidence = self.calculate_confidence(&normalized, &expanded_terms);

        RewrittenQuery {
            original: query.to_string(),
            rewritten: expanded,
            query_type,
            expanded_terms,
            confidence,
        }
    }

    /// Add a custom synonym mapping
    ///
    /// # Arguments
    /// * `term` - The original term
    /// * `synonyms` - List of synonyms
    pub fn add_synonym(&mut self, term: impl Into<String>, synonyms: Vec<String>) {
        // Normalize the term to lowercase for consistent lookup
        self.synonyms.insert(term.into().to_lowercase(), synonyms);
    }

    /// Add a query template
    ///
    /// # Arguments
    /// * `template` - Query template
    pub fn add_template(&mut self, template: QueryTemplate) {
        self.templates.push(template);
    }

    /// Record relevance feedback
    ///
    /// # Arguments
    /// * `term` - The search term
    /// * `score` - Relevance score (0.0 to 1.0)
    pub fn record_feedback(&mut self, term: impl Into<String>, score: f32) {
        let term = term.into();
        let current_score = self.relevance_scores.get(&term).unwrap_or(&0.5);
        // Weighted average with new feedback (equal weight for faster learning)
        let new_score = current_score * 0.5 + score * 0.5;
        self.relevance_scores.insert(term, new_score);
    }

    /// Get relevance score for a term
    ///
    /// # Arguments
    /// * `term` - The term to check
    ///
    /// # Returns
    /// Relevance score between 0.0 and 1.0
    pub fn get_relevance(&self, term: &str) -> f32 {
        *self.relevance_scores.get(term).unwrap_or(&0.5)
    }

    // --- Private methods ---

    /// Normalize query (lowercase, trim, etc.)
    fn normalize_query(&self, query: &str) -> String {
        query.trim().to_lowercase()
    }

    /// Detect query type based on patterns
    fn detect_query_type(&self, query: &str) -> QueryType {
        let query_lower = query.to_lowercase();

        // Relationship patterns (check before entity lookup to handle "what is the relationship...")
        if query_lower.contains("relationship between")
            || query_lower.contains("how does")
            || query_lower.contains("related to")
            || query_lower.contains("connection between")
        {
            return QueryType::Relationship;
        }

        // Entity lookup patterns
        if query_lower.starts_with("who is")
            || query_lower.starts_with("what is")
            || query_lower.starts_with("define")
        {
            return QueryType::EntityLookup;
        }

        // Aggregation patterns
        if query_lower.starts_with("how many")
            || query_lower.starts_with("count")
            || query_lower.contains("total")
            || query_lower.contains("sum")
            || query_lower.contains("average")
        {
            return QueryType::Aggregation;
        }

        // Comparison patterns
        if query_lower.contains("compare")
            || query_lower.contains("difference between")
            || query_lower.contains("versus")
            || query_lower.contains("vs")
        {
            return QueryType::Comparison;
        }

        // Temporal patterns
        if query_lower.contains("when")
            || query_lower.contains("before")
            || query_lower.contains("after")
            || query_lower.contains("during")
            || query_lower.contains("timeline")
        {
            return QueryType::Temporal;
        }

        // Causal patterns
        if query_lower.contains("why")
            || query_lower.contains("because")
            || query_lower.contains("cause")
            || query_lower.contains("reason")
            || query_lower.contains("led to")
        {
            return QueryType::Causal;
        }

        QueryType::General
    }

    /// Apply query templates
    fn apply_templates(&self, query: &str, query_type: &QueryType) -> String {
        for template in &self.templates {
            if &template.query_type == query_type && query.contains(&template.pattern) {
                return query.replace(&template.pattern, &template.rewrite);
            }
        }
        query.to_string()
    }

    /// Expand query with synonyms
    fn expand_synonyms(&self, query: &str) -> String {
        let words: Vec<&str> = query.split_whitespace().collect();
        let mut expanded_words = Vec::new();

        for word in words {
            expanded_words.push(word.to_string());

            // Add synonyms if available
            if let Some(synonyms) = self.synonyms.get(word) {
                for synonym in synonyms {
                    if !expanded_words.contains(synonym) {
                        expanded_words.push(synonym.clone());
                    }
                }
            }
        }

        expanded_words.join(" ")
    }

    /// Extract key terms (remove stop words)
    fn extract_key_terms(&self, query: &str) -> Vec<String> {
        query
            .split_whitespace()
            .filter(|word| !self.stop_words.contains(*word))
            .map(|s| s.to_string())
            .collect()
    }

    /// Calculate confidence score
    fn calculate_confidence(&self, query: &str, expanded_terms: &[String]) -> f32 {
        if expanded_terms.is_empty() {
            return 0.5;
        }

        // Base confidence on query length and term count
        let word_count = query.split_whitespace().count() as f32;
        let term_count = expanded_terms.len() as f32;

        // Higher confidence for more specific queries
        let specificity_score = (term_count / (word_count + 1.0)).min(1.0);

        // Factor in relevance feedback
        let relevance_score: f32 = expanded_terms
            .iter()
            .map(|t| self.get_relevance(t))
            .sum::<f32>()
            / term_count;

        // Weighted average
        specificity_score * 0.6 + relevance_score * 0.4
    }

    /// Load default synonyms
    fn load_default_synonyms(&mut self) {
        // Common synonyms
        self.add_synonym("find", vec!["search".to_string(), "locate".to_string()]);
        self.add_synonym("person", vec!["individual".to_string(), "people".to_string()]);
        self.add_synonym("company", vec!["organization".to_string(), "business".to_string(), "firm".to_string()]);
        self.add_synonym("show", vec!["display".to_string(), "present".to_string()]);
        self.add_synonym("get", vec!["retrieve".to_string(), "fetch".to_string()]);
        self.add_synonym("large", vec!["big".to_string(), "huge".to_string(), "significant".to_string()]);
        self.add_synonym("small", vec!["tiny".to_string(), "minor".to_string()]);
        self.add_synonym("important", vec!["significant".to_string(), "critical".to_string(), "key".to_string()]);
    }

    /// Load default query templates
    fn load_default_templates(&mut self) {
        self.add_template(QueryTemplate {
            pattern: "who is".to_string(),
            query_type: QueryType::EntityLookup,
            rewrite: "entity:".to_string(),
        });

        self.add_template(QueryTemplate {
            pattern: "what is".to_string(),
            query_type: QueryType::EntityLookup,
            rewrite: "define:".to_string(),
        });

        self.add_template(QueryTemplate {
            pattern: "how many".to_string(),
            query_type: QueryType::Aggregation,
            rewrite: "count:".to_string(),
        });

        self.add_template(QueryTemplate {
            pattern: "compare".to_string(),
            query_type: QueryType::Comparison,
            rewrite: "compare:".to_string(),
        });
    }

    /// Load default stop words
    fn load_default_stop_words(&mut self) {
        let stop_words = vec![
            "a", "an", "and", "are", "as", "at", "be", "by", "for",
            "from", "has", "he", "in", "is", "it", "its", "of", "on",
            "that", "the", "to", "was", "will", "with",
        ];

        for word in stop_words {
            self.stop_words.insert(word.to_string());
        }
    }
}

impl Default for QueryIntelligence {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_type_detection() {
        let engine = QueryIntelligence::new();

        let query = "who is the CEO of OpenAI?";
        let result = engine.rewrite_query(query);
        assert_eq!(result.query_type, QueryType::EntityLookup);

        let query = "how many employees work at Google?";
        let result = engine.rewrite_query(query);
        assert_eq!(result.query_type, QueryType::Aggregation);

        let query = "what is the relationship between Apple and Microsoft?";
        let result = engine.rewrite_query(query);
        assert_eq!(result.query_type, QueryType::Relationship);
    }

    #[test]
    fn test_synonym_expansion() {
        let engine = QueryIntelligence::new();

        let query = "find large companies";
        let result = engine.rewrite_query(query);

        // Should expand "find" and "large"
        assert!(result.expanded_terms.contains(&"search".to_string()) ||
                result.expanded_terms.contains(&"big".to_string()));
    }

    #[test]
    fn test_stop_word_removal() {
        let engine = QueryIntelligence::new();

        let query = "what is the best approach";
        let result = engine.rewrite_query(query);

        // "the" and "is" should be filtered out
        assert!(!result.expanded_terms.contains(&"the".to_string()));
        assert!(!result.expanded_terms.contains(&"is".to_string()));
    }

    #[test]
    fn test_relevance_feedback() {
        let mut engine = QueryIntelligence::new();

        engine.record_feedback("artificial_intelligence", 0.9);
        engine.record_feedback("artificial_intelligence", 0.8);

        let score = engine.get_relevance("artificial_intelligence");
        assert!(score > 0.7);
    }

    #[test]
    fn test_custom_synonyms() {
        let mut engine = QueryIntelligence::new();
        engine.add_synonym("AI", vec!["artificial intelligence".to_string(), "machine learning".to_string()]);

        let query = "AI applications";
        let result = engine.rewrite_query(query);

        assert!(result.rewritten.contains("artificial") || result.rewritten.contains("machine"));
    }
}
